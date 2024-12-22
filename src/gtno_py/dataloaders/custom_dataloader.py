from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils import draw_graph, pretty_print_graph_data


def parse_vector_field(value: str) -> List[float]:
    value = value.strip(" []")
    parts = [x.strip() for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Field does not have 3 components: {value}")
    return [float(x) for x in parts]


def build_knn_edges(coords: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    dist_matrix = torch.cdist(coords, coords)
    dist_matrix[torch.eye(dist_matrix.size(0)).bool()] = float("inf")
    knn_indices = torch.argsort(dist_matrix, dim=1)[:, :k]
    return knn_indices, dist_matrix


def build_n_hop_edges(
    coords: torch.Tensor, knn_indices: torch.Tensor, dist_matrix: torch.Tensor, n_hops: int = 2
) -> List[Tuple[int, int, float]]:
    N = coords.size(0)
    knn_sets = [set(knn_indices[i].tolist()) for i in range(N)]
    edges = []

    if n_hops == 1:
        for i in range(N):
            for nbr in knn_sets[i]:
                d = dist_matrix[i, nbr].item()
                edges.append((i, nbr.item(), d))
    elif n_hops == 2:
        for i in range(N):
            nbr_2hop = set()
            for nbr in knn_sets[i]:
                nbr_2hop.update(knn_sets[nbr])
            nbr_2hop.discard(i)
            nbr_2hop = nbr_2hop - knn_sets[i]
            for j in nbr_2hop:
                d = torch.norm(coords[i] - coords[j], p=2).item()
                edges.append((i, j, d))
    else:
        raise ValueError("n_hops must be 1 or 2")
    return edges


def build_graph(data_item):
    coords = data_item["coords"]
    node_features = data_item["node_features"]
    edges = data_item["edges"]

    G = nx.Graph()
    for i in range(coords.size(0)):
        G.add_node(i, features=node_features[i].tolist())
    G.add_weighted_edges_from([(u, v, d) for (u, v, d) in edges], weight="distance")
    return G


class MD17Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        save_dir: str = "data/rmd17_binaries",
        force_download: bool = False,
        n_hops: int = 2,
        k: int = 3,
        build_nx_graph_encodings: bool = True,
        num_workers: int = 8,
    ):
        self.save_dir = Path(save_dir)
        self.csv_path = Path(csv_path)
        self.pt_path = self.save_dir / (self.csv_path.stem + ".pt")
        self.n_hops = n_hops
        self.k = k
        self.num_workers = num_workers
        self.build_nx_graph_encodings = build_nx_graph_encodings
        self.data: List[Dict[str, Any]] = []

        if not self.pt_path.exists() or force_download:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._load_and_process_data()
            self._save_data()
        else:
            self._load_from_pt()

    def _load_and_process_data(self):
        df = pd.read_csv(self.csv_path)
        charge_cols = [c for c in df.columns if c.endswith("_charge")]
        coord_cols = [c for c in df.columns if c.endswith("_coord")]
        force_cols = [c for c in df.columns if c.endswith("_force")]

        timesteps = torch.tensor(df["timestep"].values, dtype=torch.int32)
        energies = torch.tensor(df["energy"].values, dtype=torch.float32)

        all_entries = []
        for i in tqdm(range(len(df)), total=len(df), desc="Parsing Data into Pytorch Tensors"):
            charges = torch.tensor([int(df.iloc[i][c]) for c in charge_cols], dtype=torch.int32)
            coords = torch.tensor([parse_vector_field(df.iloc[i][c]) for c in coord_cols], dtype=torch.float32)
            forces = torch.tensor([parse_vector_field(df.iloc[i][c]) for c in force_cols], dtype=torch.float32)

            node_features = self._prepare_node_features(charges, coords, forces)
            knn_indices, dist_matrix = build_knn_edges(coords, self.k)
            edges = build_n_hop_edges(coords, knn_indices, dist_matrix, n_hops=self.n_hops)

            all_entries.append(
                {
                    "timestep": timesteps[i],
                    "energy": energies[i],
                    "nuclear_charges": charges,
                    "coords": coords,
                    "forces": forces,
                    "node_features": node_features,
                    "edges": edges,
                }
            )

        if self.build_nx_graph_encodings:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                graphs = list(
                    tqdm(executor.map(build_graph, all_entries), total=len(all_entries), desc="Generating NX Graphs")
                )

        # Convert edges to tensors, add edge features
        for i, entry in enumerate(all_entries):
            if self.build_nx_graph_encodings:
                entry["graph"] = graphs[i]

            # Convert edges from list of (u, v, d) to tensors
            edge_list = entry["edges"]
            if len(edge_list) > 0:
                # Extract source nodes (u), target nodes (v), and distances (d) from edge list
                u = torch.tensor([e[0] for e in edge_list], dtype=torch.long)  # source node indices
                v = torch.tensor([e[1] for e in edge_list], dtype=torch.long)  # target node indices
                # distances between nodes
                d = torch.tensor([e[2] for e in edge_list], dtype=torch.float32)

                # Validate edge indices are within bounds
                num_nodes = len(entry["nuclear_charges"])
                assert torch.all(u >= 0) and torch.all(u < num_nodes), f"Source node indices out of bounds: {u}"
                assert torch.all(v >= 0) and torch.all(v < num_nodes), f"Target node indices out of bounds: {v}"
                # distances should be positive
                assert torch.all(d > 0), f"Found non-positive distances: {d}"
                assert len(u) == len(v) == len(d), f"Unequal lengths of u, v, and d: {len(u)}, {len(v)}, {len(d)}"

                # Pad edges tensor to fixed size (50, 2) with -1
                edges_tensor = torch.full((50, 2), -1, dtype=torch.long)
                edges_tensor[: len(u)] = torch.stack([u, v], dim=-1)

                # Create edge features: [charge_u, charge_v, distance]
                charges = entry["nuclear_charges"]

                # Create and pad edge features tensor to fixed size (50, 3) with zeros
                edge_features = torch.zeros((50, 3), dtype=torch.float32)
                features = torch.stack([charges[u].float(), charges[v].float(), d], dim=-1)
                edge_features[: len(u)] = features

                entry["edges"] = edges_tensor
                entry["edge_features"] = edge_features
            else:
                print("WARNING: No edges found for entry", i)
                entry["edges"] = torch.empty((0, 2), dtype=torch.long)
                entry["edge_features"] = torch.empty((0, 3), dtype=torch.float32)

            self.data.append(entry)

        if self.build_nx_graph_encodings:
            if len(self.data) > 0:
                draw_graph(self.data[0]["graph"])

    def _prepare_node_features(self, charges, coords, forces):
        centroid = torch.mean(coords, dim=0, keepdim=True)
        relative_pos = coords - centroid
        # node_features: [charge, forces(x,y,z), rel_pos(x,y,z)]
        node_features = torch.cat(
            [
                charges.unsqueeze(-1).float(),
                forces.view(forces.size(0), -1),
                # relative_pos.view(relative_pos.size(0), -1),
            ],
            dim=-1,
        )
        return node_features

    def _save_data(self):
        torch.save(self.data, self.pt_path)
        print(f"Processed data saved to {self.pt_path}")

    def _load_from_pt(self):
        self.data = torch.load(self.pt_path, weights_only=False)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = MD17Dataset(
        "data/rmd17_cleaned/rmd17_aspirin.csv",
        force_download=True,
        n_hops=2,
        k=3,
        num_workers=8,
        build_nx_graph_encodings=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch in dataloader:
        pretty_print_graph_data(batch, print_node_features=True)
        print("Edges:", batch["edges"])
        print("Edge Features:", batch["edge_features"])
        break
