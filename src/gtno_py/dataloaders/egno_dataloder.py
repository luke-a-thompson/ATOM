import numpy as np
import torch
import pickle as pkl
from torch.utils.data import Dataset
import os
from enum import Enum
from typing import final, override
import numpy.typing as npt
from tensordict import TensorDict


@final
class DataPartition(str, Enum):
    train = "train"
    val = "val"
    test = "test"


@final
class MoleculeType(str, Enum):
    aspirin = "aspirin"
    benzene = "benzene"
    ethanol = "ethanol"
    malonaldehyde = "malonaldehyde"
    naphthalene = "naphthalene"
    salicylic = "salicylic"
    toluene = "toluene"
    uracil = "uracil"


class MD17Dataset(Dataset[dict[str, torch.Tensor]]):
    """
    MD17 Dataset

    """

    def __init__(
        self,
        partition: DataPartition,
        max_samples: int,
        delta_frame: int,
        data_dir: str,
        split_dir: str,
        molecule_type: MoleculeType,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        seed: int = 100,
        force_regenerate: bool = False,
        num_timesteps: int = 1,  # Added for dynamics dataset compatibility, default to 1 for standard dataset
    ):
        """
        Args:
            partition (str): The partition to load ('train', 'val', 'test').
            max_samples (int): The maximum number of samples to load into the initial frame.
            delta_frame (int): The number of frames to skip between the initial and target frames.
            data_dir (str): The directory to load the data from.
            split_dir (str): The directory to load or store splits.
            molecule_type (str): The type of molecule to load ('aspirin', 'benzene_old', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil').
            train_par (float): The percentage of the data to use for training.
            val_par (float): The percentage of the data to use for validation.
            test_par (float): The percentage of the data to use for testing.
            num_timesteps (int): Number of timesteps for dynamics dataset, default 1 for static.
        """

        self.partition: DataPartition = partition
        self.molecule_type: MoleculeType = molecule_type
        self.delta_frame: int = delta_frame
        self.num_timesteps: int = num_timesteps

        # setup a split, tentative setting
        train_par, val_par, test_par = train_par, val_par, test_par
        full_dir = os.path.join(data_dir + "md17_" + molecule_type + ".npz")
        split_dir = os.path.join(split_dir + molecule_type + "_split.pkl")
        data: np.lib.npyio.NpzFile = np.load(full_dir)

        x: npt.NDArray[np.float64] = data["R"]
        v: npt.NDArray[np.float64] = x[1:] - x[:-1]
        x = x[:-1]

        # Load or generate split
        split = self._get_or_generate_split(
            split_dir=split_dir,
            x=x,
            train_par=train_par,
            val_par=val_par,
            test_par=test_par,
            force_regenerate=force_regenerate,
            seed=seed,
        )

        match partition:
            case DataPartition.train:
                split_times = split[0]
            case DataPartition.val:
                split_times = split[1]
            case DataPartition.test:
                split_times = split[2]
            case _:
                raise ValueError(f"Invalid partition: {partition}, select from one of {DataPartition.__members__.keys()}")

        #  st is the index of the first frame to load, this gives the max number of samples from the split
        split_times = split_times[:max_samples]
        self.split_times = split_times  # Store split_times for potential reuse in subclasses

        z = data["z"]
        # Select all atoms with atomic number 'z' greater than 1
        heavy_atom_mask: npt.NDArray[np.bool_] = z > 1
        x = x[:, heavy_atom_mask, ...]
        v = v[:, heavy_atom_mask, ...]
        z = z[heavy_atom_mask]

        self.x_all = x  # Store for potential reuse in subclasses
        self.v_all = v  # Store for potential reuse in subclasses
        self.z_all = z  # Store for potential reuse in subclasses

        self.process_data(split_times, x, v, z)

    def process_data(self, split_times, x, v, z):
        """Processes the loaded data, common to both MD17Dataset and MD17DynamicsDataset"""
        x_0, v_0 = self.get_initial_frames(split_times, x, v)
        x_t, v_t = self.get_target_frames(split_times, x, v)

        mole_idx = z
        n_node = mole_idx.shape[0]
        self.n_node: int = n_node

        # Build edges
        one_hop_adjacency, two_hop_adjacency = self._compute_adjacency_matrix(x, n_node)
        edge_attr, edges = self._build_edge_attributes(one_hop_adjacency, two_hop_adjacency, mole_idx, x_0, v_0)
        self.edge_attr = edge_attr
        self.edges = edges

        # Build conf_edges
        all_edges = self._compute_all_edges(x=x, z=z)
        conf_edges = self._compute_conf_edges(all_edges=all_edges)
        self.conf_edges: list[list[int]] = conf_edges

        # Convert to tensors
        self.x_0 = torch.cat([torch.Tensor(x_0), torch.norm(torch.Tensor(x_0), dim=-1, keepdim=True)], dim=-1)
        self.v_0 = torch.cat([torch.Tensor(v_0), torch.norm(torch.Tensor(v_0), dim=-1, keepdim=True)], dim=-1)
        self.mole_idx = torch.Tensor(mole_idx)
        self.Z = torch.Tensor(z)
        self.cfg = self.sample_cfg()

        # Conditionally convert x_t and v_t to tensors only if they are not None
        if x_t is not None:
            self.x_t: torch.Tensor = torch.Tensor(x_t)
        if v_t is not None:
            self.v_t: torch.Tensor = torch.Tensor(v_t)

        self.concatenated_features: torch.Tensor = self._compute_concatenated_features()

    def _compute_concatenated_features(self) -> torch.Tensor:
        """Pre-compute concatenated features for all samples."""
        x_0_xyz = self.x_0[..., :3]  # First 3 elements of x_0 (x,y,z)
        v_0_xyz = self.v_0[..., :3]  # First 3 elements of v_0 (vx,vy,vz)
        x_0_norm = self.x_0[..., 3:]  # Last element of x_0 (norm(x))
        v_0_norm = self.v_0[..., 3:]  # Last element of v_0 (norm(v))
        Z_unsqueeze = self.Z.unsqueeze(-1)  # Z values

        concatenated_features = torch.cat(
            [
                x_0_xyz,
                v_0_xyz,
                x_0_norm,
                v_0_norm,
                Z_unsqueeze.expand(self.x_0.shape[0], -1, -1),  # expand Z to match batch size
            ],
            dim=-1,
        )

        return concatenated_features

    def get_initial_frames(
        self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get initial frames x_0, v_0. Can be overridden for different dataset types."""
        x_0 = x[split_times]
        v_0 = v[split_times]
        return x_0, v_0

    def get_target_frames(
        self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get target frames x_t, v_t. Can be overridden for different dataset types."""
        x_t = x[split_times + self.delta_frame]
        v_t = v[split_times + self.delta_frame]
        return x_t, v_t

    def _get_or_generate_split(
        self, split_dir: str, x: npt.NDArray[np.float64], train_par: float, val_par: float, test_par: float, force_regenerate: bool = False, seed: int = 100
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        Load the train/val/test split from split_dir if it exists; otherwise generate it.
        Returns:
            A tuple (train_idx, val_idx, test_idx) of arrays containing split indices.
        """
        try:
            if force_regenerate:
                raise FileNotFoundError("Force regeneration of dataset")

            with open(split_dir, "rb") as f:
                split: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]] = pkl.load(f)

        except FileNotFoundError as e:
            print(f"Error loading split file: {e}, regenerating split")
            np.random.seed(seed)

            _x = x[10000:-10000]

            train_idx = np.random.choice(np.arange(_x.shape[0]), size=int(train_par * _x.shape[0]), replace=False)
            flag = np.zeros(_x.shape[0])
            for _ in train_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            val_idx = np.random.choice(rest, size=int(val_par * _x.shape[0]), replace=False)
            for _ in val_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            test_idx = np.random.choice(rest, size=int(test_par * _x.shape[0]), replace=False)

            train_idx += 10000
            val_idx += 10000
            test_idx += 10000

            split = (train_idx, val_idx, test_idx)

            with open(split_dir, "wb") as f:
                pkl.dump(split, f)
            print("Generate and save split!")

        return split

    def _compute_adjacency_matrix(self, x: npt.NDArray[np.float64], num_atoms: int, threshold: float = 1.6) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes node x node adjacency matrix (one_hop_edges) and squared adjacency (two_hop_edges) matrices based on inter-atomic distances at the first frame.

        One-hop edges is all edges that are within the threshold distance.
        Two-hop edges is all edges that are within 2 hops of each other.
        """

        def d(_i: int, _j: int, _t: int) -> float:
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        one_hop_edges = torch.zeros(num_atoms, num_atoms).int()
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < threshold:
                        one_hop_edges[i][j] = 1

        two_hop_edges = (one_hop_edges @ one_hop_edges).clamp(max=1)  # Matrix-square for 2-hop edges
        assert (
            one_hop_edges.shape == two_hop_edges.shape == (num_atoms, num_atoms)
        ), f"one_hop_edges.shape: {one_hop_edges.shape}, two_hop_edges.shape: {two_hop_edges.shape}, num_atoms: {num_atoms}"
        return one_hop_edges, two_hop_edges

    def _build_edge_attributes(
        self, one_hop_adjacency: torch.Tensor, two_hop_adjacency: torch.Tensor, mole_idx: npt.NDArray[np.int_], x_0: npt.NDArray[np.float64], v_0: npt.NDArray[np.float64]
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Build edge_attr (torch.Tensor) and edges (list) based on atom_edges and atom_edges2. The edge_attr array stores [atom_type1, atom_type2, path_distance].

        The edges list has two sublists [rows, cols] for edges.
        """
        n_node = mole_idx.shape[0]
        edge_attr = []
        rows: list[int] = []
        cols: list[int] = []

        assert len(x_0.shape) == len(v_0.shape) == 3, f"Expected the full shape of x_0 and v_0 to be [n_frames, n_nodes, 3], but got x_0.shape: {x_0.shape}, v_0.shape: {v_0.shape}"
        x_0_frame_one: npt.NDArray[np.float64] = x_0[0]

        # Loop through all node pairs (i, j), without self-loops
        for i in range(n_node):
            for j in range(n_node):
                if i == j:
                    continue
                else:
                    first_frame_distance = np.linalg.norm(x_0_frame_one[i] - x_0_frame_one[j])

                    # One-hop edges
                    if one_hop_adjacency[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 1, first_frame_distance])
                        assert not two_hop_adjacency[i][j]
                    # Two-hop edges
                    if two_hop_adjacency[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 2, first_frame_distance])
                        assert not one_hop_adjacency[i][j]

        edges = [rows, cols]
        edge_attr_tensor = torch.Tensor(np.array(edge_attr))

        return edge_attr_tensor, edges

    def _compute_all_edges(self, x: npt.NDArray[np.float64], z: npt.NDArray[np.int_], threshold: float = 1.6) -> dict[tuple[int, int], list[list[int]]]:
        """
        Efficiently builds a dictionary of edges keyed by sorted atom type pairs (high, low),
        where each value is a list of atom index pairs within the distance threshold.
        """
        from collections import defaultdict

        n = z.shape[0]
        positions = x[0]  # Assuming x[0] is the relevant coordinate set

        # Calculate upper triangle indices (i < j)
        i_indices, j_indices = np.triu_indices(n, k=1)

        # Vectorized distance calculation for all pairs (i < j)
        deltas = positions[i_indices] - positions[j_indices]
        distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))  # Faster than sum

        # Apply distance threshold
        mask = distances < threshold
        valid_i = i_indices[mask]
        valid_j = j_indices[mask]

        # Determine atom type pairs and sort them
        type_pairs = np.sort(np.column_stack([z[valid_i], z[valid_j]]), axis=1)

        # Group edges by type pairs using defaultdict
        all_edges: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
        for (a, b), i, j in zip(type_pairs, valid_i, valid_j):
            all_edges[(a, b)].append([int(i), int(j)])  # Ensure Python int types

        return dict(all_edges)

    def _compute_conf_edges(self, all_edges: dict[tuple[int, int], list[list[int]]]) -> list[list[int]]:
        conf_edges = []
        for key in all_edges:
            conf_edges.extend(all_edges[key])
        return conf_edges

    def sample_cfg(self) -> dict[str, list[tuple[int, int]] | list[list[int]]]:
        """
        Kinematics Decomposition
        """
        cfg = {}
        if self.molecule_type == MoleculeType.benzene:
            cfg["Stick"] = [(0, 1), (2, 3), (4, 5)]
        elif self.molecule_type == MoleculeType.aspirin:
            cfg["Stick"] = [(0, 2), (1, 3), (5, 6), (7, 10), (11, 12)]
        elif self.molecule_type == MoleculeType.ethanol:
            cfg["Stick"] = [(0, 1)]
        elif self.molecule_type == MoleculeType.malonaldehyde:
            cfg["Stick"] = [(1, 2)]
        elif self.molecule_type == MoleculeType.naphthalene:
            cfg["Stick"] = [(0, 1), (2, 3), (4, 9), (5, 6), (7, 8)]
        elif self.molecule_type == MoleculeType.salicylic:
            cfg["Stick"] = [(0, 9), (1, 2), (4, 5), (6, 7)]
        elif self.molecule_type == MoleculeType.toluene:
            cfg["Stick"] = [(2, 3), (5, 6), (0, 1)]
        elif self.molecule_type == MoleculeType.uracil:
            cfg["Stick"] = [(0, 1), (3, 4)]
        else:
            raise NotImplementedError()
        cur_selected = []
        for _ in cfg["Stick"]:
            cur_selected.append(_[0])
            cur_selected.append(_[1])
        cfg["Isolated"] = [[_] for _ in range(self.n_node) if _ not in cur_selected]
        if len(cfg["Isolated"]) == 0:
            cfg.pop("Isolated")

        return cfg

    @override
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x_0": self.x_0[i],
            "v_0": self.v_0[i],
            "x_t": self.x_t[i],
            "v_t": self.v_t[i],
            "concatenated_features": self.concatenated_features[i],  # Use pre-computed features
        }

    def __len__(self):
        return len(self.x_0)

    def get_edges(self, batch_size: int, n_nodes: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        edges: tuple[torch.LongTensor, torch.LongTensor] = (torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1]))
        if batch_size == 1:
            return edges

        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = (torch.cat(rows).long(), torch.cat(cols).long())
        return edges

    @staticmethod
    def get_cfg(batch_size: int, n_nodes: int, cfg: TensorDict) -> TensorDict:
        """
        Expands a molecule's configuration dictionary (cfg) to a batch.
        """

        # Make sure offset is on the same device as cfg
        offset = torch.arange(batch_size, device=cfg.device) * n_nodes

        # 'cfg.keys()' returns all fields in this TensorDict, e.g. ["Isolated", "Stick", ...]
        for bond_type in cfg.keys():
            # index is now a proper torch.Tensor of shape [B, n_type, node_per_type]
            index = cfg.get(bond_type)

            # Perform the exact same operations as EGNO
            index = index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)

            if bond_type == "Isolated":
                index = index.squeeze(-1)

            # Store it back in the tensordict
            cfg.set(bond_type, index)

        return cfg


@final
class MD17DynamicsDataset(MD17Dataset):
    """
    MD17 Dynamics Dataset

    """

    def __init__(
        self,
        partition: DataPartition,
        max_samples: int,
        delta_frame: int,
        data_dir: str,
        split_dir: str,
        molecule_type: MoleculeType,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        num_timesteps: int = 8,
        seed: int = 100,
        force_regenerate: bool = False,
    ):
        # First call the parent constructor to load and process common data
        super().__init__(
            partition=partition,
            max_samples=max_samples,
            delta_frame=delta_frame,
            data_dir=data_dir,
            split_dir=split_dir,
            molecule_type=molecule_type,
            train_par=train_par,
            val_par=val_par,
            test_par=test_par,
            seed=seed,
            force_regenerate=force_regenerate,
            num_timesteps=num_timesteps,  # Pass num_timesteps to base class
        )
        self.x_t, self.v_t = self.get_dynamic_target_frames()

        # Overwrite x_t and v_t with the multi-step versions
        self.x_t = torch.Tensor(self.x_t)
        self.v_t = torch.Tensor(self.v_t)

    def get_dynamic_target_frames(self):
        """
        Generates multi-step target frames for dynamics dataset.
        Uses attributes initialized in the parent class.
        """
        x = self.x_all
        v = self.v_all
        split_times = self.split_times
        delta_frame = self.delta_frame
        num_timesteps = self.num_timesteps

        x_t_list: list[npt.NDArray[np.float64]] = [x[split_times + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        x_t = np.stack(x_t_list, axis=2)
        v_t_list: list[npt.NDArray[np.float64]] = [v[split_times + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        v_t = np.stack(v_t_list, axis=2)
        return x_t, v_t

    @override
    def get_target_frames(self, split_times, x, v):
        """Override to prevent single frame target loading in dynamics dataset"""
        # In MD17DynamicsDataset, target frames are handled in `get_dynamic_target_frames`
        return None, None  # Return None to indicate no single frame target loading, will be overwritten by multi-frame targets


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Test MD17Dataset
    dataset_static = MD17Dataset(
        partition=DataPartition.train,
        max_samples=5000,
        delta_frame=5000,
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=MoleculeType.aspirin,
        force_regenerate=True,
    )
    dataloader_static: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset_static, batch_size=1, shuffle=True)
    print("MD17Dataset Output Shapes:")
    for data in dataloader_static:
        for key in data:
            if key not in ["cfg", "edge_attr"]:
                print(f"  {key}:", data[key].shape)
        print("  cfg shapes:")
        for key in data["cfg"]:
            print(f"    {key}:", data["cfg"][key].shape)
        break

    # Test MD17DynamicsDataset
    dataset_dynamic = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=5000,
        delta_frame=5000,
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=MoleculeType.aspirin,
        force_regenerate=True,
    )

    dataloader_dynamic: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset_dynamic, batch_size=1, shuffle=True)
    print("\nMD17DynamicsDataset Output Shapes:")
    for data in dataloader_dynamic:
        for key in data:
            if key not in ["cfg", "edge_attr"]:
                print(f"  {key}:", data[key].shape)

        print("  cfg shapes:")
        for key in data["cfg"]:
            print(f"    {key}:", data["cfg"][key].shape)
        break
