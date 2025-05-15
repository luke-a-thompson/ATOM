import numpy as np
import torch
from torch.utils.data import Dataset
import os


class NBodyDynamicsDataset(Dataset[dict[str, torch.Tensor]]):
    """
    NBody Dynamics Dataset with pre-replication along num_timesteps axis.
    Each sample's initial state (first frame of the trajectory) is replicated.
    Output of __getitem__ will be [num_timesteps, nodes, features].
    """

    def __init__(
        self,
        data_dir: str,
        partition: str = "train",
        max_samples: int = int(1e8),
        dataset_name: str = "nbody_small",
        num_timesteps: int = 8,
    ) -> None:
        self.data_dir: str = data_dir
        self.partition: str = partition
        self.max_samples: int = int(max_samples)
        self.dataset_name: str = dataset_name
        self.num_timesteps: int = num_timesteps

        if self.partition == "val":
            suffix = "valid"
        else:
            suffix = self.partition
        if dataset_name == "nbody":
            suffix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            suffix += "_charged5_initvel1small"
        else:
            raise Exception(f"Wrong dataset name {self.dataset_name}")
        self.suffix: str = suffix

        # Load initial states (loc_0, vel_0, charges)
        self.loc_0, self.vel_0, self.edges, self.charges_base = self._load_initial_states()

        self.n_samples: int = min(self.loc_0.shape[0], self.max_samples)
        # Ensure we only process up to n_samples if max_samples is smaller than file
        self.loc_0 = self.loc_0[: self.n_samples]
        self.vel_0 = self.vel_0[: self.n_samples]
        self.charges_base = self.charges_base[: self.n_samples]

        self.n_nodes: int = self.loc_0.shape[1]  # Shape is [S, N, D]

        # Pre-replicate initial states
        self.replicated_loc: torch.Tensor = self._replicate_tensor(self.loc_0)
        self.replicated_vel: torch.Tensor = self._replicate_tensor(self.vel_0)
        self.replicated_charges: torch.Tensor = self._replicate_tensor(self.charges_base)

    def _load_initial_states(self) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor]:
        # Loads the initial frame (t=0) for loc and vel, and base charges.
        # loc and vel will be [S, N, D]
        # charges will be [S, N, 1]

        loc_full = np.load(os.path.join(self.data_dir, f"loc_{self.suffix}.npy"))
        vel_full = np.load(os.path.join(self.data_dir, f"vel_{self.suffix}.npy"))
        edges_raw = np.load(os.path.join(self.data_dir, f"edges_{self.suffix}.npy"))
        charges_raw = np.load(os.path.join(self.data_dir, f"charges_{self.suffix}.npy"))

        # loc_full is [S_file, T_orig, D_raw_order, N_raw_order]
        # Transpose to [S_file, T_orig, N, D]
        loc_full_torch = torch.tensor(loc_full, dtype=torch.float32).transpose(2, 3)
        vel_full_torch = torch.tensor(vel_full, dtype=torch.float32).transpose(2, 3)

        # Select initial frame (t=0)
        loc_0 = loc_full_torch[:, 0, :, :]  # [S_file, N, D]
        vel_0 = vel_full_torch[:, 0, :, :]  # [S_file, N, D]

        charges_tensor = torch.tensor(charges_raw, dtype=torch.float32)
        if charges_tensor.ndim == 2:  # Expected [S_file, N]
            charges = charges_tensor.unsqueeze(-1)  # [S_file, N, 1]
        elif charges_tensor.ndim == 3 and charges_tensor.shape[-1] == 1:  # Already [S_file, N, 1]
            charges = charges_tensor
        else:
            raise ValueError(f"Unexpected shape for charges: {charges_tensor.shape}. Expected [S, N] or [S, N, 1].")

        n_nodes = loc_0.size(1)
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Assuming edges_raw might be [T, N, N] or just [N, N]
                    # For simplicity, building a fully connected graph if edges_raw is not used here.
                    # User's original code built edges based on n_nodes, not edges.npy directly for attributes
                    rows.append(i)
                    cols.append(j)
        edge_index = [rows, cols]

        return loc_0, vel_0, edge_index, charges

    def _replicate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # Input tensor shape: [S, N, D] (e.g. loc_0) or [S, N, 1] (e.g. charges_base)
        # Output tensor shape: [S, num_timesteps, N, D] or [S, num_timesteps, N, 1]

        # Add new time dimension for replication
        tensor_unsqueeze = tensor.unsqueeze(1)  # [S, 1, N, D]

        # Expand along the new time dimension to num_timesteps
        # tensor_unsqueeze.shape[2:] will be (N,D) or (N,1)
        replicated_tensor = tensor_unsqueeze.expand(-1, self.num_timesteps, *tensor_unsqueeze.shape[2:]).contiguous()
        return replicated_tensor

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # Returns [num_timesteps, N, D] for loc/vel, [num_timesteps, N, 1] for charges
        return {
            "loc": self.replicated_loc[i],
            "vel": self.replicated_vel[i],
            "charges": self.replicated_charges[i],
            "edge_index": torch.tensor(self.edges, dtype=torch.long),
        }

    def __len__(self) -> int:
        return self.n_samples


if __name__ == "__main__":
    dataset = NBodyDynamicsDataset(
        data_dir="data/n_body_simple",  # Corrected path from previous error
        partition="train",
        max_samples=10,  # Using a small number for quick testing
        dataset_name="nbody_small",
        num_timesteps=8,
    )
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample shapes (time, nodes, features):")
        for key, value in sample.items():
            print(f"  {key}: {value.shape}")
    else:
        print("Dataset is empty.")
