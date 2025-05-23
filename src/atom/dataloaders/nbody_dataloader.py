import numpy as np
import torch
from torch.utils.data import Dataset
import os
from atom.training.config_options import DataPartition


class NBodyDynamicsDataset(Dataset[dict[str, torch.Tensor]]):
    """
    NBody Dynamics Dataset with pre-replication along num_timesteps axis.
    Each sample's initial state (first frame of the trajectory) is replicated.
    Output of __getitem__ will be [num_timesteps, nodes, features].
    """

    def __init__(
        self,
        data_dir: str,
        partition: DataPartition = DataPartition.train,
        max_samples: int = int(1e8),
        dataset_name: str = "nbody_small",
        num_timesteps: int = 8,
        return_edge_data=False,
    ) -> None:
        self.data_dir: str = data_dir
        self.partition: str = partition
        self.max_samples: int = int(max_samples)
        self.dataset_name: str = dataset_name
        self.num_timesteps: int = num_timesteps
        self.return_edge_data: bool = return_edge_data

        match self.partition:
            case DataPartition.train:
                suffix = "train"
            case DataPartition.val:
                suffix = "valid"
            case DataPartition.test:
                suffix = "test"

        if dataset_name == "nbody":
            suffix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            suffix += "_charged5_initvel1small"
        else:
            raise Exception(f"Wrong dataset name {self.dataset_name}")
        self.suffix: str = suffix

        # Load full trajectory data
        self.loc_full, self.vel_full, self.edges, self.edge_attr, self.charges = self._load_data()

        self.n_samples: int = min(self.loc_full.shape[0], self.max_samples)
        # Ensure we only process up to n_samples if max_samples is smaller than file
        self.loc_full = self.loc_full[: self.n_samples]
        self.vel_full = self.vel_full[: self.n_samples]
        self.charges = self.charges[: self.n_samples]
        self.edge_attr = self.edge_attr[: self.n_samples]

        self.n_nodes: int = self.loc_full.shape[2]  # Shape is [S, T, N, D]

        # Determine frame indices based on dataset
        if self.dataset_name == "nbody":
            self.frame_0, self.frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            self.frame_0, self.frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            self.frame_0, self.frame_T = 20, 30
        else:
            raise Exception(f"Wrong dataset partition {self.dataset_name}")

        # Extract initial state
        self.loc_0 = self.loc_full[:, self.frame_0]  # [S, N, D]
        self.vel_0 = self.vel_full[:, self.frame_0]  # [S, N, D]

        # Pre-replicate initial states
        self.replicated_loc: torch.Tensor = self._replicate_tensor(self.loc_0)
        self.replicated_vel: torch.Tensor = self._replicate_tensor(self.vel_0)
        self.replicated_charges: torch.Tensor = self._replicate_tensor(self.charges)

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor, torch.Tensor]:
        # Loads the full trajectories for loc and vel and charges
        # loc and vel will be [S, T, N, D]
        # charges will be [S, N, 1]
        # edge_attr will be [S, E, 1]

        loc_full = np.load(os.path.join(self.data_dir, f"loc_{self.suffix}.npy"))
        vel_full = np.load(os.path.join(self.data_dir, f"vel_{self.suffix}.npy"))
        edges_raw = np.load(os.path.join(self.data_dir, f"edges_{self.suffix}.npy"))
        charges_raw = np.load(os.path.join(self.data_dir, f"charges_{self.suffix}.npy"))

        # loc_full is [S_file, T_orig, D_raw_order, N_raw_order]
        # Transpose to [S_file, T_orig, N, D]
        loc_full_torch = torch.tensor(loc_full, dtype=torch.float32).transpose(2, 3)
        vel_full_torch = torch.tensor(vel_full, dtype=torch.float32).transpose(2, 3)

        charges_tensor = torch.tensor(charges_raw, dtype=torch.float32)
        if charges_tensor.ndim == 2:  # Expected [S_file, N]
            charges = charges_tensor.unsqueeze(-1)  # [S_file, N, 1]
        elif charges_tensor.ndim == 3 and charges_tensor.shape[-1] == 1:  # Already [S_file, N, 1]
            charges = charges_tensor
        else:
            raise ValueError(f"Unexpected shape for charges: {charges_tensor.shape}. Expected [S, N] or [S, N, 1].")

        n_nodes = loc_full_torch.size(2)
        rows, cols = [], []
        edge_attr_list = []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr_list.append(edges_raw[:, i, j])
                    rows.append(i)
                    cols.append(j)

        edges = [rows, cols]

        # Convert edge_attr to tensor with shape [S, E, 1]
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float32).transpose(0, 1).unsqueeze(2)

        return loc_full_torch, vel_full_torch, edges, edge_attr, charges

    def _replicate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # Input tensor shape: [S, N, D] (e.g. loc_0) or [S, N, 1] (e.g. charges)
        # Output tensor shape: [S, num_timesteps, N, D] or [S, num_timesteps, N, 1]

        # Add new time dimension for replication
        tensor_unsqueeze = tensor.unsqueeze(1)  # [S, 1, N, D]

        # Expand along the new time dimension to num_timesteps
        # tensor_unsqueeze.shape[2:] will be (N,D) or (N,1)
        replicated_tensor = tensor_unsqueeze.expand(-1, self.num_timesteps, *tensor_unsqueeze.shape[2:]).contiguous()
        return replicated_tensor

    def _sample_trajectory(self, sample_idx: int) -> torch.Tensor:
        """Sample future frames between frame_0 and frame_T"""
        delta_frame = self.frame_T - self.frame_0

        # Sample num_timesteps points between frame_0 and frame_T
        future_locs = []
        for i in range(1, self.num_timesteps + 1):
            # Calculate frame index proportionally between frame_0 and frame_T
            frame_idx = self.frame_0 + delta_frame * i // self.num_timesteps
            future_locs.append(self.loc_full[sample_idx, frame_idx])

        # Stack along new time dimension
        return torch.stack(future_locs, dim=0)  # [num_timesteps, N, D]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # Get ground truth trajectory
        trajectory_gt = self._sample_trajectory(i)  # [num_timesteps, N, D]

        nodes = torch.sqrt(torch.sum(self.replicated_vel[i] ** 2, dim=-1, keepdim=True))
        concatenated_features = torch.cat((self.replicated_loc[i], self.replicated_vel[i]), dim=-1)

        # Returns:
        # - x_0: replicated initial positions [num_timesteps, N, D]
        # - v_0: replicated initial velocities [num_timesteps, N, D]
        # - edge_attr: edge attributes [E, 1]
        # - charges: replicated charges [num_timesteps, N, 1]
        # - source_node_indices: source node indices [E]
        # - target_node_indices: target node indices [E]
        # - x_t: ground truth future positions [num_timesteps, N, D]
        sample = {
            "x_0": self.replicated_loc[i],
            "v_0": self.replicated_vel[i],
            "concatenated_features": concatenated_features,
            "nodes": nodes,
            "edge_attr": self.edge_attr[i],
            "charges": self.replicated_charges[i],
            "x_t": trajectory_gt,
        }
        if self.return_edge_data:
            sample["source_node_indices"] = torch.tensor(self.edges[0], dtype=torch.long).contiguous()
            sample["target_node_indices"] = torch.tensor(self.edges[1], dtype=torch.long).contiguous()
        return sample

    def __len__(self) -> int:
        return self.n_samples


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = NBodyDynamicsDataset(
        data_dir="data/n_body_simple",
        partition=DataPartition.train,
        max_samples=10,  # Using a small number for quick testing
        dataset_name="nbody_small",
        num_timesteps=8,
    )

    # Print total number of timesteps available in the dataset
    print(f"Total timesteps available in dataset: {dataset.loc_full.shape}")

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    assert False, np.load("data/n_body_simple/loc_train_charged5_initvel1small.npy").shape

    batch = next(iter(dataloader))
    for key, tensor in batch.items():
        if not isinstance(tensor, list):
            print(f"{key}: {tensor.shape}")
        else:
            print(f"{key}: {len(tensor)}")
