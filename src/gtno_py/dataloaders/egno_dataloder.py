import numpy as np
import numpy.typing as npt
import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import os
from typing import final, override
from pathlib import Path
import torch.nn.functional as F
from gtno_py.training.config_options import DataPartition, MD17Version, MD17MoleculeType, RMD17MoleculeType


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
        md17_version: MD17Version,
        molecule_type: MD17MoleculeType | RMD17MoleculeType,
        max_nodes: int,
        explicit_hydrogen: bool = False,
        rrwp_length: int = 8,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        seed: int = 100,
        force_regenerate: bool = False,
        num_timesteps: int = 1,  # Number of timesteps to replicate
        verbose: bool = False,
    ):
        """
        Args:
            partition (str): The partition to load ('train', 'val', 'test').
            max_samples (int): The maximum number of samples to load into the initial frame.
            delta_frame (int): The number of frames to skip between the initial and target frames.
            data_dir (str): The directory which stores the MD17 and RMD17 data.
            split_dir (str): The directory to load or store splits.
            molecule_type (str): The type of molecule to load ('aspirin', 'benzene_old', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil').
            train_par (float): The percentage of the data to use for training.
            val_par (float): The percentage of the data to use for validation.
            test_par (float): The percentage of the data to use for testing.
            num_timesteps (int): Number of timesteps for replication.
        """
        self.partition: DataPartition = partition
        self.md17_version: MD17Version = md17_version
        self.molecule_type: MD17MoleculeType | RMD17MoleculeType = molecule_type
        self.max_nodes: int = max_nodes
        self.delta_frame: int = delta_frame
        self.num_timesteps: int = num_timesteps
        self.max_samples: int = max_samples
        self.verbose: bool = verbose
        self.rrwp_length: int = rrwp_length
        self.dft_imprecision_margin: int = 10_000

        match md17_version:
            case MD17Version.md17:
                if not isinstance(molecule_type, MD17MoleculeType):
                    raise ValueError(f"For MD17Version.md17, molecule_type must be MD17MoleculeType, got {molecule_type}")
                full_dir = os.path.join(data_dir + "md17_npz/" + "md17_" + molecule_type + ".npz")
                split_dir = os.path.join(split_dir + "md17_splits/" + "md17_" + molecule_type + "_split.pkl")
                positions_col = "R"
                charges_col = "z"
            case MD17Version.rmd17:
                if not isinstance(molecule_type, RMD17MoleculeType):
                    raise ValueError(f"For MD17Version.rmd17, molecule_type must be RMD17MoleculeType, got {molecule_type}")
                full_dir = os.path.join(data_dir + "rmd17_npz/" + "rmd17_" + molecule_type + ".npz")
                split_dir = os.path.join(split_dir + "rmd17_splits/" + "rmd17_" + molecule_type + "_split.pkl")
                positions_col = "coords"
                charges_col = "nuclear_charges"
            case _:
                raise ValueError(f"Invalid MD17 version: {md17_version}")

        data_file: np.lib.npyio.NpzFile = np.load(full_dir)
        self.x: npt.NDArray[np.float64] = data_file[positions_col]
        self.z: npt.NDArray[np.uint8] = data_file[charges_col]
        self.v: npt.NDArray[np.float64] = self.x[1:] - self.x[:-1]  # Construct velocities from successive coords
        self.x = self.x[:-1]  # Remove last coord to ensure len(x) == len(v)
        assert self.x.shape == self.v.shape

        split = self._get_or_generate_split(
            split_dir=Path(split_dir),
            x=self.x,
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
                raise ValueError(f"Invalid partition: {partition}")

        self.split_times = split_times[:max_samples]

        # Remove hydrogens if specified
        if not explicit_hydrogen:
            heavy_atom_mask = self.z > 1
            self.x = self.x[:, heavy_atom_mask, ...]
            self.v = self.v[:, heavy_atom_mask, ...]
            self.z = self.z[heavy_atom_mask]

        self.process_data(self.split_times, self.x, self.v, self.z)

        # --- Precompute Replication ---
        self.replicated_x_0: torch.Tensor = self._replicate_tensor(self.x_0)
        self.replicated_v_0: torch.Tensor = self._replicate_tensor(self.v_0)
        self.replicated_concatenated_features: torch.Tensor = self._replicate_tensor(self.concatenated_features)
        self.replicated_z_0: torch.Tensor = self._replicate_tensor(self.z_0)

    def process_data(self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64], z: npt.NDArray[np.uint8]):
        """Processes loaded data, common to both MD17Dataset and MD17DynamicsDataset"""
        x_0, v_0 = self.get_initial_frames(split_times, x, v)
        x_t, v_t = self.get_target_frames(split_times, x, v)

        self.num_nodes: int = z.shape[0]

        one_hop_adjacency, two_hop_adjacency = self._compute_adjacency_matrix(x, self.num_nodes, 1.6)
        self.edge_attr, self.edges = self._build_edge_attributes(one_hop_adjacency, two_hop_adjacency, z, x_0, v_0)
        if self.rrwp_length > 0:
            self.rrwp: torch.Tensor = self.calculate_rrwp(one_hop_adjacency, self.rrwp_length)

        self.x_0: torch.Tensor = torch.cat([x_0, torch.norm(x_0, dim=-1, keepdim=True)], dim=-1)
        self.v_0: torch.Tensor = torch.cat([v_0, torch.norm(v_0, dim=-1, keepdim=True)], dim=-1)
        # Expand atomic numbers to match batch dimension of x_0 and v_0
        self.z_0: torch.Tensor = torch.Tensor(z).unsqueeze(-1).unsqueeze(0).expand(self.x_0.shape[0], -1, -1)
        self.concatenated_features: torch.Tensor = self._compute_concatenated_features()
        self.mole_idx: torch.Tensor = torch.Tensor(torch.arange(z.shape[0])).unsqueeze(-1).expand(self.x_0.shape[0], -1, -1)

        self.x_0 = self._pad_tensor(self.x_0)
        self.v_0 = self._pad_tensor(self.v_0)
        self.z_0 = self._pad_tensor(self.z_0)
        self.concatenated_features = self._pad_tensor(self.concatenated_features)
        self.mole_idx = self._pad_tensor(self.mole_idx)

        if x_t is not None:
            self.x_t: torch.Tensor = torch.Tensor(x_t)
            self.x_t = self._pad_tensor(self.x_t)
        if v_t is not None:
            self.v_t: torch.Tensor = torch.Tensor(v_t)
            self.v_t = self._pad_tensor(self.v_t)

        assert (
            self.x_t.shape[:2]
            == self.v_t.shape[:2]
            == self.x_0.shape[:2]
            == self.v_0.shape[:2]
            == self.z_0.shape[:2]
            == self.concatenated_features.shape[:2]
            == self.mole_idx.shape[:2]
        ), (
            f"Shape mismatch:\n"
            f"x_t.shape: {self.x_t.shape}\n"
            f"v_t.shape: {self.v_t.shape}\n"
            f"x_0.shape: {self.x_0.shape}\n"
            f"v_0.shape: {self.v_0.shape}\n"
            f"z_0.shape: {self.z_0.shape}\n"
            f"concatenated_features.shape: {self.concatenated_features.shape}\n"
            f"mole_idx.shape: {self.mole_idx.shape}"
        )

    def _compute_concatenated_features(self) -> torch.Tensor:
        """Pre-compute concatenated features for all samples.

        Returns:
            torch.Tensor: Concatenated features with shape (max_samples * num_timesteps, N, d)
            Contents = [x_0_xyz, x_0_norm, v_0_xyz, v_0_norm, Z_unsqueeze]
        """
        x_0_xyz = self.x_0[..., :3]
        v_0_xyz = self.v_0[..., :3]
        x_0_norm = self.x_0[..., 3:]
        v_0_norm = self.v_0[..., 3:]

        features_to_concat = [
            x_0_xyz,
            x_0_norm,
            v_0_xyz,
            v_0_norm,
            self.z_0,
        ]

        if self.rrwp_length > 0:
            rrwp = self.rrwp.unsqueeze(0).expand(self.max_samples, -1, -1)  # Expand from [1, 6, 8] to [max_samples, 6, 8]
            features_to_concat.append(rrwp)

        concatenated_features = torch.cat(features_to_concat, dim=-1)
        return concatenated_features

    def _replicate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Replicates a single tensor along the batch dimension.

        Input tensor shape: [max_samples, d]
        Output tensor shape: [max_samples * num_timesteps, d]

        Returns:
            torch.Tensor: The replicated tensor.
        """
        # Add new time dimension
        assert tensor.shape[0] == self.max_samples, f"Tensor shape: {tensor.shape}, max_samples: {self.max_samples}"
        tensor_with_time = tensor.unsqueeze(1)

        # Expand along time dimension to num_timesteps
        tensor_expanded = tensor_with_time.expand(-1, self.num_timesteps, *tensor.shape[1:])

        return tensor_expanded

    def get_initial_frames(self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> tuple[torch.Tensor, torch.Tensor]:
        x_0 = torch.Tensor(x[split_times])
        v_0 = torch.Tensor(v[split_times])
        return x_0, v_0

    def get_target_frames(self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.Tensor(x[split_times + self.delta_frame])
        v_t = torch.Tensor(v[split_times + self.delta_frame])
        return x_t, v_t

    def _get_or_generate_split(
        self,
        split_dir: Path,
        x: npt.NDArray[np.float64],
        train_par: float,
        val_par: float,
        test_par: float,
        seed: int,
        force_regenerate: bool = False,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        Get or generate train/val/test split indices.

        Args:
            split_dir: Path to save/load the split file
            x: Input data array
            train_par: Proportion of data for training
            val_par: Proportion of data for validation
            test_par: Proportion of data for testing
            seed: Random seed for reproducibility
            force_regenerate: Whether to force regeneration of the split

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Calculate valid frame range considering margins
        start = self.dft_imprecision_margin
        end = x.shape[0] - self.dft_imprecision_margin - self.delta_frame + 1

        # Try to load existing split file
        if not force_regenerate:
            try:
                with open(split_dir, "rb") as f:
                    return pkl.load(f)
            except FileNotFoundError:
                print("Split file not found, generating new split") if self.verbose else None
        else:
            print("Forcing regeneration of dataset split") if self.verbose else None

        # Generate new split
        return self._generate_new_split(start=start, end=end, x=x, train_par=train_par, val_par=val_par, test_par=test_par, seed=seed, split_dir=split_dir)

    def _generate_new_split(
        self,
        start: int,
        end: int,
        x: npt.NDArray[np.float64],
        train_par: float,
        val_par: float,
        test_par: float,
        seed: int,
        split_dir: Path,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        Generate a new train/val/test split.

        Args:
            start: Start index for valid frames
            end: End index for valid frames
            x: Input data array
            train_par: Proportion of data for training
            val_par: Proportion of data for validation
            test_par: Proportion of data for testing
            seed: Random seed for reproducibility
            split_dir: Path to save the split file

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        np.random.seed(seed)

        # Extract valid frame range
        x_middle = x[start:end]
        num_timesteps = x_middle.shape[0]

        # Create mask to track assigned indices
        assigned_mask = np.zeros(num_timesteps, dtype=bool)

        # Select training indices
        train_size = int(train_par * num_timesteps)
        train_idx = np.random.choice(np.arange(num_timesteps), size=train_size, replace=False)
        assigned_mask[train_idx] = True

        # Select validation indices from remaining frames
        unassigned_indices = np.where(~assigned_mask)[0]
        val_size = int(val_par * num_timesteps)
        val_idx = np.random.choice(unassigned_indices, size=val_size, replace=False)
        assigned_mask[val_idx] = True

        # Select test indices from remaining frames
        unassigned_indices = np.where(~assigned_mask)[0]
        test_size = int(test_par * num_timesteps)
        test_idx = np.random.choice(unassigned_indices, size=test_size, replace=False)

        # Adjust indices to original frame range
        train_idx = train_idx + start
        val_idx = val_idx + start
        test_idx = test_idx + start

        # Create and save split
        split = (train_idx, val_idx, test_idx)
        with open(split_dir, "wb") as f:
            pkl.dump(split, f)

        # Print information about the generated split
        if self.verbose:
            print(f"Generated and saved split with {len(train_idx)} train, {len(val_idx)} val, and {len(test_idx)} test samples")
            print(f"Note: Max samples will be limited to {self.max_samples if hasattr(self, 'max_samples') else 'unlimited'} during dataset usage")

        return split

    def _compute_adjacency_matrix(self, x: npt.NDArray[np.float64], num_atoms: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one-hop and two-hop adjacency matrices based on distance threshold.

        Args:
            x: Atom positions of shape (time_steps, num_atoms, 3)
            num_atoms: Number of atoms
            threshold: Distance threshold for considering atoms as connected
                Recommended 1.6 due to atomic distances

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (one_hop_adjacency_matrix, two_hop_adjacency_matrix)
        """
        # Extract positions at time 0
        positions: torch.Tensor = torch.tensor(x[0], dtype=torch.float32)  # Shape: (num_atoms, 3)

        # Compute pairwise distances using vectorized operations
        # Expand dimensions for broadcasting
        pos_i: torch.Tensor = positions.unsqueeze(1)  # Shape: (num_atoms, 1, 3)
        pos_j: torch.Tensor = positions.unsqueeze(0)  # Shape: (1, num_atoms, 3)

        # Compute distances between all pairs of atoms
        distances: torch.Tensor = torch.norm(pos_i - pos_j, dim=2)  # Shape: (num_atoms, num_atoms)

        # Create adjacency matrix based on threshold
        one_hop_edges: torch.Tensor = (distances < threshold).int()

        # Set diagonal to zero (no self-loops)
        one_hop_edges.fill_diagonal_(0)

        # Compute two-hop connections
        two_hop_edges: torch.Tensor = (one_hop_edges @ one_hop_edges).clamp(max=1)

        assert one_hop_edges.shape == two_hop_edges.shape == (num_atoms, num_atoms)
        return one_hop_edges, two_hop_edges

    def _build_edge_attributes(self, one_hop_adjacency, two_hop_adjacency, z, x_0, v_0):
        """
        Legacy EGNO function for computing edge attributes.

        Args:
            one_hop_adjacency: One-hop adjacency matrix
            two_hop_adjacency: Two-hop adjacency matrix
            mole_idx: Molecule index
            x_0: Initial positions
            v_0: Initial velocities

        Returns:
            edge_attr: Edge attributes containing source, target, type, and distance
            edges: Edges containing source and target indices
        """

        n_node = z.shape[0]
        edge_attr = []
        rows = []
        cols = []

        x_0_frame_one = x_0[0]  # use frame 0 to compute edges
        for i in range(n_node):
            for j in range(n_node):
                if i == j:
                    continue
                else:
                    first_frame_distance = np.linalg.norm(x_0_frame_one[i] - x_0_frame_one[j])
                    if one_hop_adjacency[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([z[i], z[j], 1, first_frame_distance])
                        assert not two_hop_adjacency[i][j]
                    if two_hop_adjacency[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([z[i], z[j], 2, first_frame_distance])
                        assert not one_hop_adjacency[i][j]

        edges = [rows, cols]
        edge_attr_tensor = torch.Tensor(np.array(edge_attr))
        return edge_attr_tensor, edges

    def calculate_rrwp(self, adj: torch.Tensor, walk_length: int = 8) -> torch.Tensor:
        """
        Calculate random walk return probabilities (RRWP) for each node given an adjacency matrix.

        Parameters:
            adj (torch.Tensor): An (n x n) adjacency matrix.
            walk_length (int): K, the total number of walk steps.

        Returns:
            torch.Tensor: A tensor of shape (n, k) where each row holds the self-return probability at each walk length.
        """
        # Ensure adjacency matrix is in float format
        adj = adj.float()

        # Row-normalise the adjacency matrix: D^{-1}A
        deg = adj.sum(dim=1)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        A_norm = torch.diag(deg_inv) @ adj

        rrwp_list = []
        # The first nontrivial step is the row-normalised A itself.
        current = A_norm.clone()
        rrwp_list.append(current)

        # Compute subsequent steps by repeated multiplication with A_norm
        for _ in range(len(rrwp_list), walk_length):
            current = current @ A_norm
            rrwp_list.append(current)

        # Extract the diagonal from each matrix to get the self-return probabilities
        rrwp = torch.stack([mat.diag() for mat in rrwp_list], dim=1)  # Shape: (n, k)
        assert rrwp.shape == (self.num_nodes, walk_length), f"RRWP shape: {rrwp.shape}, num_nodes: {self.num_nodes}, walk_length: {walk_length}"
        return rrwp

    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape assumed to be (num_samples, N, d)
        assert tensor.shape[-2] <= self.max_nodes, f"Tensor shape: {tensor.shape}, max_nodes: {self.max_nodes}"
        pad_amt = self.max_nodes - tensor.shape[-2]
        if pad_amt > 0:
            # pad (last_dim_left, last_dim_right, node_dim_left, node_dim_right)
            return F.pad(tensor, (0, 0, 0, pad_amt))
        return tensor

    @override
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """
        Retrieve the i-th sample, including its replicated time steps, from the dataset.

        The base tensors (e.g. x_0, v_0, concatenated_features) are precomputed and stored in a
        flattened form with shape (max_samples * num_timesteps, N, d). This flattening is performed
        by first expanding each tensor to have an explicit time dimension (resulting in shape
        (max_samples, num_timesteps, N, d)) and then reshaping it to merge the sample and time dimensions.

        For a given sample index i, slicing from index (i * num_timesteps) to ((i + 1) * num_timesteps)
        retrieves the contiguous block corresponding to that sample's time steps. This operation recovers
        the time dimension, yielding a tensor of shape (num_timesteps, N, d).

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - "x_0": Tensor of initial positions with shape (num_timesteps, N, d)
                - "v_0": Tensor of initial velocities with shape (num_timesteps, N, d)
                - "x_t": Tensor of target positions with shape (num_timesteps, N, d)
                - "v_t": Tensor of target velocities with shape (num_timesteps, N, d)
                - "concatenated_features": Tensor of concatenated features with shape (num_timesteps, N, d)
        """
        # For sample index i, slice out the contiguous block of timesteps (of size num_timesteps)
        # from the pre-replicated tensors. This recovers the T timesteps associated with the i-th sample.
        # i * self.num_timesteps : (i + 1) * self.num_timesteps - We want to be this many i * timesteps *frames* from the start, and capture the whole frame
        sample = {
            "x_0": self.replicated_x_0[i],
            "v_0": self.replicated_v_0[i],
            "concatenated_features": self.replicated_concatenated_features[i],
            "Z": self.replicated_z_0[i],
            "x_t": self.x_t[i],
            "v_t": self.v_t[i],
        }

        if self.num_nodes < self.max_nodes:
            mask = torch.cat(
                [
                    torch.ones(self.num_nodes, dtype=torch.bool),
                    torch.zeros(self.max_nodes - self.num_nodes, dtype=torch.bool),
                ]
            )

            mask = mask.unsqueeze(0).expand(self.num_timesteps, -1).unsqueeze(-1).bool()
            sample["padded_nodes_mask"] = mask

        return sample

    def __len__(self):
        return len(self.split_times)


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
        md17_version: MD17Version,
        molecule_type: MD17MoleculeType | RMD17MoleculeType,
        max_nodes: int,
        explicit_hydrogen: bool = False,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        num_timesteps: int = 8,  # Number of timesteps for dynamics
        rrwp_length: int = 8,
        seed: int = 100,
        force_regenerate: bool = False,
    ):
        super().__init__(
            partition=partition,
            max_samples=max_samples,
            delta_frame=delta_frame,
            data_dir=data_dir,
            split_dir=split_dir,
            md17_version=md17_version,
            molecule_type=molecule_type,
            max_nodes=max_nodes,
            explicit_hydrogen=explicit_hydrogen,
            train_par=train_par,
            val_par=val_par,
            test_par=test_par,
            seed=seed,
            force_regenerate=force_regenerate,
            num_timesteps=num_timesteps,  # Pass num_timesteps to base class for replication
            rrwp_length=rrwp_length,
        )
        self.x_t, self.v_t = self.get_dynamic_target_frames()

        # Re-replicate dataset after defining x_t, v_t for dynamics dataset
        self.replicated_x_0: torch.Tensor = self._replicate_tensor(self.x_0)
        self.replicated_v_0: torch.Tensor = self._replicate_tensor(self.v_0)
        self.replicated_concatenated_features: torch.Tensor = self._replicate_tensor(self.concatenated_features)
        self.replicated_z_0: torch.Tensor = self._replicate_tensor(self.z_0)
        self.replicated_mole_idx: torch.Tensor = self._replicate_tensor(self.mole_idx)

        if self.x_t is not None:
            self.x_t = torch.Tensor(self.x_t).transpose(1, 2)
            self.x_t = self._pad_tensor(self.x_t)
        if self.v_t is not None:
            self.v_t = torch.Tensor(self.v_t).transpose(1, 2)
            self.v_t = self._pad_tensor(self.v_t)

        assert (
            self.x_t.shape[:3]
            == self.v_t.shape[:3]
            == self.replicated_x_0.shape[:3]
            == self.replicated_v_0.shape[:3]
            == self.replicated_z_0.shape[:3]
            == self.replicated_concatenated_features.shape[:3]
            == self.replicated_mole_idx.shape[:3]
        ), (
            f"Shape mismatch in first 3 dims:\n"
            f"x_t.shape: {self.x_t.shape}\n"
            f"v_t.shape: {self.v_t.shape}\n"
            f"replicated_x_0.shape: {self.replicated_x_0.shape}\n"
            f"replicated_v_0.shape: {self.replicated_v_0.shape}\n"
            f"replicated_concatenated_features.shape: {self.replicated_concatenated_features.shape}\n"
            f"replicated_z_0.shape: {self.replicated_z_0.shape}\n"
            f"replicated_mole_idx.shape: {self.replicated_mole_idx.shape}"
        )

    def get_dynamic_target_frames(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        split_times = self.split_times
        delta_frame = self.delta_frame
        num_timesteps = self.num_timesteps

        x_t_list = [self.x[split_times + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        x_t = np.stack(x_t_list, axis=2)
        v_t_list = [self.v[split_times + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        v_t = np.stack(v_t_list, axis=2)
        return x_t, v_t


if __name__ == "__main__":
    # Test MD17Dataset
    dataset_static = MD17Dataset(
        partition=DataPartition.train,
        max_samples=5000,
        delta_frame=30000,
        data_dir="data/",
        split_dir="data/",
        md17_version=MD17Version.rmd17,
        molecule_type=MD17MoleculeType.benzene,
        max_nodes=20,
        force_regenerate=False,
        num_timesteps=1,  # Set num_timesteps for replication
    )
    dataloader_static = DataLoader(dataset_static, batch_size=100, shuffle=True)
    print("MD17Dataset Output Shapes:")
    for data in dataloader_static:
        for key in data:
            if key not in ["cfg", "edge_attr"]:
                print(f"  {key}:", data[key].shape)
        if "cfg" in data:
            print("  cfg shapes:")
            for key in data["cfg"]:
                print(f"    {key}:", data["cfg"][key].shape)
        break

    # Test MD17DynamicsDataset
    dataset_dynamic = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=3000,
        data_dir="data/",
        split_dir="data/",
        md17_version=MD17Version.md17,
        molecule_type=MD17MoleculeType.aspirin,
        max_nodes=13,
        force_regenerate=True,
        num_timesteps=8,  # Set num_timesteps for replication
    )

    dataloader_dynamic = DataLoader(dataset_dynamic, batch_size=100, shuffle=True)
    print("MD17DynamicsDataset Output Shapes:")
    for data in dataloader_dynamic:
        for key in data:
            if key not in ["cfg", "edge_attr"]:
                print(f"  {key}:", data[key].shape)
        if "cfg" in data:
            print("  cfg shapes:")
            for key in data["cfg"]:
                print(f"    {key}:", data["cfg"][key].shape)
        break

    # datasets = []
    # molecule_list = [MD17MoleculeType.benzene, MD17MoleculeType.ethanol]
    # for molecule in molecule_list:
    #     dataset_to_concat = MD17DynamicsDataset(
    #         partition=DataPartition.train,
    #         max_samples=500,
    #         delta_frame=3000,
    #         data_dir="data/",
    #         split_dir="data/",
    #         md17_version=MD17Version.md17,
    #         molecule_type=molecule,
    #         max_nodes=20,
    #         force_regenerate=True,
    #         num_timesteps=8,  # Set num_timesteps for replication
    #     )
    #     datasets.append(dataset_to_concat)

    # dataset_dynamic = MD17DynamicsDataset(
    #     partition=DataPartition.train,
    #     max_samples=500,
    #     delta_frame=3000,
    #     data_dir="data/",
    #     split_dir="data/",
    #     md17_version=MD17Version.md17,
    #     molecule_type=MD17MoleculeType.benzene,
    #     max_nodes=6,
    #     force_regenerate=True,
    #     num_timesteps=8,  # Set num_timesteps for replication
    # )

    # # dataloader_dynamic = DataLoader(dataset_dynamic, batch_size=100, shuffle=True)
    # combined_dataset = torch.utils.data.ConcatDataset(datasets)
    # dataloader_concat = DataLoader(combined_dataset, batch_size=100, shuffle=True)

    # print("\nMD17DynamicsMultitaskDataset Output Shapes:")
    # print("Shape: Batch size, Time steps, Nodes, Features")
    # data = next(iter(dataloader_concat))
    # for key in data:
    #     if key != "Molecule_type":
    #         print(f"  {key}:", data[key].shape)
    #     else:
    #         print(f"  Molecule_type: {data['Molecule_type'][0]}")
