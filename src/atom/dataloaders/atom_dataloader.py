import numpy as np
import numpy.typing as npt
import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import os
from typing import final, override
from pathlib import Path
import torch.nn.functional as F
from atom.training.config_options import DataPartition, Datasets, MD17MoleculeType, RMD17MoleculeType, TG80MoleculeType, MD22MoleculeType, TimeLagMode


# Centralized stick definitions per molecule (indices assume no hydrogens)
MOLECULE_STICKS: dict[str, list[tuple[int, int]]] = {
    "benzene": [(0, 1), (2, 3), (4, 5)],
    "aspirin": [(0, 2), (1, 3), (5, 6), (7, 10), (11, 12)],
    "ethanol": [(0, 1)],
    "malonaldehyde": [(1, 2)],
    "naphthalene": [(0, 1), (2, 3), (4, 9), (5, 6), (7, 8)],
    "salicylic": [(0, 9), (1, 2), (4, 5), (6, 7)],
    "toluene": [(2, 3), (5, 6), (0, 1)],
    "uracil": [(0, 1), (3, 4)],
}


class MD17Dataset(Dataset[dict[str, torch.Tensor]]):
    """
    MD17 Dataset
    """

    def __init__(
        self,
        partition: DataPartition,
        max_samples: int,
        delta_frame: int | tuple[int, int],
        data_dir: str,
        split_dir: str,
        md17_version: Datasets,
        molecule_type: MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType,
        max_nodes: int | None,
        return_edge_data: bool,
        center_data: bool = False,
        num_timesteps: int = 1,  # Number of timesteps to replicate
        explicit_hydrogen: bool = False,
        radius_graph_threshold: float = 1.6,
        rrwp_length: int = 8,
        normalize_z: bool = False,
        egno_mode: bool = False,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        seed: int = 42,
        force_regenerate: bool = False,
        verbose: bool = False,
        max_edges: int | None = None,  # Maximum number of edges to pad to
        time_lag_mode: TimeLagMode = TimeLagMode.LAST,
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
        self.data_partition: DataPartition = partition
        self.md17_version: Datasets = md17_version
        self.molecule_type: MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType = molecule_type
        self.max_nodes: int | None = max_nodes
        # Support fixed or ranged delta frame. Use max for indexing safety and eval.
        if isinstance(delta_frame, tuple):
            self.delta_frame_min: int = int(delta_frame[0])
            self.delta_frame_max: int = int(delta_frame[1])
            if self.delta_frame_min <= 0 or self.delta_frame_max <= 0:
                raise ValueError(f"delta_frame bounds must be positive, got {delta_frame}")
            if self.delta_frame_min > self.delta_frame_max:
                raise ValueError(f"delta_frame min must be <= max, got {delta_frame}")
        else:
            self.delta_frame_min = int(delta_frame)
            self.delta_frame_max = int(delta_frame)
        self.num_timesteps: int = num_timesteps
        self.center_data: bool = center_data
        self.return_edge_data: bool = return_edge_data
        self.max_samples: int = max_samples
        self.verbose: bool = verbose
        self.radius_graph_threshold: float = radius_graph_threshold
        self.rrwp_length: int = rrwp_length
        self.normalize_z: bool = normalize_z
        self.egno_mode: bool = egno_mode
        self.max_edges: int | None = max_edges
        self.time_lag_mode: TimeLagMode = time_lag_mode
        # Predeclare attributes for type checkers
        self.split_times: npt.NDArray[np.int_] = np.array([], dtype=np.int_)
        self.cfg: dict[str, list[tuple[int, int]] | list[list[int]]] = {}
        self.replicated_x_0_mean: torch.Tensor = torch.empty(0)
        self.edge_attr: torch.Tensor = torch.empty(0, 4, dtype=torch.float32)
        self.edge_index: tuple[torch.Tensor, torch.Tensor] = (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        # Removed absolute time indices; model uses local time only
        # Resolve molecule name from enum/value robustly
        try:
            molecule_name: str = str(molecule_type.value)
        except Exception:
            molecule_name = str(molecule_type)

        match md17_version:
            case Datasets.md17:
                full_dir = os.path.join(data_dir, "md17_npz", f"md17_{molecule_name}.npz")
                split_dir = os.path.join(split_dir, "md17_splits", f"md17_{molecule_name}_split.pkl")
                positions_col = "R"
                charges_col = "z"
                self.dft_imprecision_margin: int = 10_000
            case Datasets.rmd17:
                full_dir = os.path.join(data_dir, "rmd17_npz", f"rmd17_{molecule_name}.npz")
                split_dir = os.path.join(split_dir, "rmd17_splits", f"rmd17_{molecule_name}_split.pkl")
                positions_col = "coords"
                charges_col = "nuclear_charges"
                self.dft_imprecision_margin: int = 10_000
            case Datasets.tg80:
                full_dir = os.path.join(data_dir, "tg80_npz", f"tg80_{molecule_name}.npz")
                split_dir = os.path.join(split_dir, "tg80_splits", f"tg80_{molecule_name}_split.pkl")
                positions_col = "coords"
                charges_col = "nuclear_charges"
                self.dft_imprecision_margin: int = 500
                train_par = 0.2
                val_par = 0.1
                test_par = 0.1
            case Datasets.md22:
                full_dir = os.path.join(data_dir, "md22_npz", f"md22_{molecule_name}.npz")
                split_dir = os.path.join(split_dir, "md22_splits", f"md22_{molecule_name}_split.pkl")
                self.dft_imprecision_margin: int = 500
                if molecule_type == MD22MoleculeType.STACHYOSE:
                    train_par = 0.2
                    val_par = 0.1
                    test_par = 0.1
                positions_col = "R"
                charges_col = "z"
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

        if egno_mode is True:
            self.cfg = self._sample_cfg()

        self.process_data(self.split_times, self.x, self.v, self.z)

        # --- Precompute Replication ---
        # Shape: [max_samples, num_timesteps, nodes, d]
        self.replicated_x_0: torch.Tensor = self._replicate_tensor(self.x_0)
        self.replicated_v_0: torch.Tensor = self._replicate_tensor(self.v_0)
        self.replicated_concatenated_features: torch.Tensor = self._replicate_tensor(self.concatenated_features)
        self.replicated_z_0: torch.Tensor = self._replicate_tensor(self.z_0)
        if self.center_data:
            self.replicated_x_0_mean = self.replicated_x_0[..., :3].mean(dim=(2), keepdim=True)
            self.replicated_x_0[..., :3] = self.replicated_x_0[..., :3] - self.replicated_x_0_mean

        # Assert that self.replicated_x_0 contains identical data across all timesteps
        # This means for each sample, all timesteps should have the same initial positions
        if self.num_timesteps > 1:
            # Get the first timestep data
            first_timestep_data = self.replicated_x_0[0]

            # Check that all other timesteps have identical data
            for t in range(1, self.num_timesteps):
                assert torch.allclose(self.replicated_x_0[0][t], first_timestep_data), (
                    f"Initial positions (x_0) at timestep {t} differ from timestep 0. " f"Shape: {self.replicated_x_0.shape}"
                )

    def process_data(self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64], z: npt.NDArray[np.uint8]):
        """Processes loaded data, common to both MD17Dataset and MD17DynamicsDataset"""
        x_0, v_0 = self.get_initial_frames(split_times, x, v)
        x_t, v_t = self.get_target_frames(split_times, x, v)

        self.x_t: torch.Tensor = self._pad_tensor(x_t)
        self.v_t: torch.Tensor = self._pad_tensor(v_t)

        self.num_nodes: int = z.shape[0]

        one_hop_adjacency, two_hop_adjacency = self._compute_adjacency_matrix(x, self.num_nodes, self.radius_graph_threshold)

        if self.return_edge_data:
            stick_set: set[tuple[int, int]] | None = None
            if self.egno_mode:
                # Build stick set from cfg with explicit typing
                if "Stick" in self.cfg:
                    stick_pairs: list[tuple[int, int]] = []
                    for pair in self.cfg["Stick"]:
                        i = int(pair[0])
                        j = int(pair[1])
                        if i > j:
                            i, j = j, i
                        stick_pairs.append((i, j))
                    stick_set = set(stick_pairs)
            self.edge_attr, self.edge_index = self._build_edge_attributes(one_hop_adjacency, two_hop_adjacency, torch.tensor(z), x_0, stick_set)
        if self.max_edges is not None and self.egno_mode:
            current_edges = self.edge_attr.shape[0]
            if current_edges < self.max_edges:
                pad_len = self.max_edges - current_edges
                # Pad edge attributes with zeros
                edge_attr_pad = torch.zeros(pad_len, self.edge_attr.shape[1], dtype=self.edge_attr.dtype)
                self.edge_attr = torch.cat([self.edge_attr, edge_attr_pad], dim=0)
                # Pad edge indices with zeros (valid self-loop indices)
                source_pad = torch.zeros(pad_len, dtype=self.edge_index[0].dtype)
                target_pad = torch.zeros(pad_len, dtype=self.edge_index[1].dtype)
                self.edge_index = (torch.cat([self.edge_index[0], source_pad], dim=0), torch.cat([self.edge_index[1], target_pad], dim=0))

        if self.rrwp_length > 0:
            self.rrwp: torch.Tensor = self.calculate_rrwp(one_hop_adjacency, self.rrwp_length)

        self.x_0: torch.Tensor = torch.cat([x_0, torch.norm(x_0, dim=-1, keepdim=True)], dim=-1)
        self.v_0: torch.Tensor = torch.cat([v_0, torch.norm(v_0, dim=-1, keepdim=True)], dim=-1)
        # Expand atomic numbers to match batch dimension of x_0 and v_0
        self.z_0: torch.Tensor = torch.Tensor(z).unsqueeze(-1).unsqueeze(0).expand(self.x_0.shape[0], -1, -1)
        if self.normalize_z:
            self.z_0 = self.z_0 / self.z_0.max()
        self.concatenated_features: torch.Tensor = self._compute_concatenated_features()
        self.mole_idx: torch.Tensor = torch.Tensor(torch.arange(z.shape[0])).unsqueeze(-1).expand(self.x_0.shape[0], -1, -1)

        self.x_0 = self._pad_tensor(self.x_0)
        self.v_0 = self._pad_tensor(self.v_0)
        self.z_0 = self._pad_tensor(self.z_0)
        self.concatenated_features = self._pad_tensor(self.concatenated_features)
        self.mole_idx = self._pad_tensor(self.mole_idx)

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
            rrwp = self.rrwp.unsqueeze(0).expand(self.x_0.shape[0], -1, -1)  # Expand to match actual sample count
            features_to_concat.append(rrwp)

        concatenated_features = torch.cat(features_to_concat, dim=-1)
        return concatenated_features

    def _replicate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Replicates a single tensor along the batch dimension.

        Input tensor shape: [max_samples, d]
        Output tensor shape: [max_samples, num_timesteps, nodes, d]

        Returns:
            torch.Tensor: The replicated tensor.
        """
        # Add new time dimension
        assert tensor.shape[0] == len(
            self.split_times
        ), f"Tensor shape: {tensor.shape}, expected samples: {len(self.split_times)}. Molecule type: {self.molecule_type} for split: {self.data_partition}"
        tensor_with_time = tensor.unsqueeze(1)

        # Expand along time dimension to num_timesteps
        tensor_expanded = tensor_with_time.expand(-1, self.num_timesteps, *tensor.shape[1:]).contiguous()

        return tensor_expanded

    def get_initial_frames(self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> tuple[torch.Tensor, torch.Tensor]:
        x_0 = torch.Tensor(x[split_times])
        v_0 = torch.Tensor(v[split_times])
        return x_0, v_0

    def get_target_frames(self, split_times: npt.NDArray[np.int_], x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.Tensor(x[split_times + self.delta_frame_max])
        v_t = torch.Tensor(v[split_times + self.delta_frame_max])
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
        end = x.shape[0] - self.dft_imprecision_margin - self.delta_frame_max + 1

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
        # Ensure the directory exists
        os.makedirs(split_dir.parent, exist_ok=True)
        with open(split_dir, "wb") as f:
            pkl.dump(split, f)

        if self.verbose:
            print(f"Generated and saved split with {len(train_idx)} train, {len(val_idx)} val, and {len(test_idx)} test samples")
            print(f"Note: Max samples will be limited to {self.max_samples if hasattr(self, 'max_samples') else 'unlimited'} during dataset usage")

        return split

    def _compute_adjacency_matrix(self, x: npt.NDArray[np.float64], num_atoms: int, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one-hop and two-hop adjacency matrices based on distance threshold.

        Args:
            x: Atom positions of shape (time_steps, num_atoms, 3)
            num_atoms: Number of atoms
            threshold: Initial distance threshold for considering atoms as connected
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

        one_hop_edges: torch.Tensor = (distances < threshold).int()

        # Set diagonal to zero (no self-loops)
        one_hop_edges.fill_diagonal_(0)

        # If no edges are found, gradually increase threshold until we get edges
        current_threshold = threshold
        if one_hop_edges.sum() == 0 and self.verbose:
            print(f"[{self.molecule_type}] No edges found with initial threshold {threshold}. Increasing threshold to find edges...")

        while one_hop_edges.sum() == 0 and current_threshold < 10.0:  # Cap at 10.0 to prevent infinite loop
            current_threshold *= 1.5
            one_hop_edges = (distances < current_threshold).int()
            one_hop_edges.fill_diagonal_(0)

        if one_hop_edges.sum() > 0 and current_threshold != threshold and self.verbose:
            print(f"[{self.molecule_type}] Found edges with new threshold {current_threshold:.2f}.")

        if one_hop_edges.sum() == 0:
            raise ValueError(
                f"Could not find any edges even with threshold {current_threshold:.2f}. This suggests the molecule data may be corrupted.  Molecule type: {self.molecule_type}"
            )

        # Compute two-hop connections
        two_hop_all: torch.Tensor = (one_hop_edges @ one_hop_edges).clamp(max=1)
        two_hop_all.fill_diagonal_(0)
        # Exclude direct neighbors from two-hop set
        two_hop_only: torch.Tensor = (two_hop_all - one_hop_edges).clamp(min=0)

        assert one_hop_edges.shape == two_hop_only.shape == (num_atoms, num_atoms)

        if self.verbose and self.data_partition == DataPartition.train:
            one_hop_count = int(one_hop_edges.sum().item() // 2)
            two_hop_count = int(two_hop_only.sum().item() // 2)
            total_edges = one_hop_count + two_hop_count
            print(f"[{self.molecule_type}] Total edges: {total_edges}. One-hop: {one_hop_count}, Two-hop: {two_hop_count}.")

        return one_hop_edges, two_hop_only

    def _build_edge_attributes(
        self,
        one_hop_adjacency: torch.Tensor,
        two_hop_adjacency: torch.Tensor,
        z: torch.Tensor,
        x_0: torch.Tensor,
        stick_set: set[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Vectorized edge attribute builder supporting optional stick indicators.

        Returns edge_attr with columns: [z_i, z_j, edge_type (1 or 2), stick_ind or distance]
        If stick_set is provided, the 4th column is stick_ind (0/1). Otherwise it is distance.
        """
        # Indices for one- and two-hop edges
        one_idx = one_hop_adjacency.nonzero(as_tuple=False)
        two_idx = two_hop_adjacency.nonzero(as_tuple=False)

        if one_idx.numel() == 0 and two_idx.numel() == 0:
            return torch.empty(0, 4, dtype=torch.float32), (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        sources = torch.cat([one_idx[:, 0], two_idx[:, 0]], dim=0).long()
        targets = torch.cat([one_idx[:, 1], two_idx[:, 1]], dim=0).long()

        # Edge type: 1 for one-hop, 2 for two-hop
        edge_type = torch.cat([torch.ones(one_idx.size(0)), 2 * torch.ones(two_idx.size(0))], dim=0).float().unsqueeze(1)

        z_i = z[sources].float().unsqueeze(1)
        z_j = z[targets].float().unsqueeze(1)

        if stick_set is None:
            # Use distance as 4th feature
            x0 = x_0[0]
            d = (x0[sources] - x0[targets]).norm(dim=1, keepdim=True)
            edge_attr = torch.cat([z_i, z_j, edge_type, d], dim=1)
        else:
            # Use stick indicator as 4th feature
            pairs = torch.stack([sources, targets], dim=1)
            is_stick = torch.tensor([tuple(sorted(p.tolist())) in stick_set for p in pairs], dtype=torch.float32).unsqueeze(1)
            edge_attr = torch.cat([z_i, z_j, edge_type, is_stick], dim=1)

        return edge_attr, (sources, targets)

    # Removed: merged into unified _build_edge_attributes

    def _sample_cfg(self) -> dict[str, list[tuple[int, int]] | list[list[int]]]:
        """
        Define rigid 'Stick' bonds based on molecule type, mimicking original implementation.
        Indices here refer to the atoms *after* potential hydrogen removal.
        """
        cfg = {}
        n_node = self.z.shape[0]  # Get number of nodes *after* potential H removal
        try:
            mol_type_str = str(self.molecule_type.value).lower()
        except Exception:
            mol_type_str = str(self.molecule_type).lower()
        cfg["Stick"] = MOLECULE_STICKS.get(mol_type_str, [])
        if not cfg["Stick"] and self.verbose:
            print(f"Warning: No specific 'Stick' configuration defined for molecule type: {self.molecule_type}. No stick indices will be used.")

        # Calculate 'Isolated' nodes (nodes not part of any stick)
        cur_selected = []
        if "Stick" in cfg:
            for stick_pair in cfg["Stick"]:
                cur_selected.extend(stick_pair)
        cfg["Isolated"] = [[node_idx] for node_idx in range(n_node) if node_idx not in cur_selected]
        if not cfg["Isolated"]:  # Remove if empty
            cfg.pop("Isolated")

        # Convert tuples/lists in cfg to numpy arrays like the original
        # (Although we only use 'Stick' directly in _build_edge_attributes)
        # cfg_np = {_key: np.array(value) for _key, value in cfg.items()}
        # return cfg_np
        # Let's keep it as list of tuples/lists for easier processing in _build_edge_attributes
        return cfg

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
        if self.max_nodes is not None:
            assert tensor.shape[-2] <= self.max_nodes, f"Node dim of tensor indicates {tensor.shape[-2]} nodes, max_nodes: {self.max_nodes}. Overall shape is {tensor.shape}"
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
            "x_0": self.replicated_x_0[i].contiguous(),
            "v_0": self.replicated_v_0[i].contiguous(),
            "concatenated_features": self.replicated_concatenated_features[i].contiguous(),
            "Z": self.replicated_z_0[i].contiguous(),
            "x_t": self.x_t[i].contiguous(),
            "v_t": self.v_t[i].contiguous(),
        }

        # Removed delta_t and absolute time indices; model handles local time encoding

        # No time PE concatenation here; time PE will be added inside the model if enabled

        if self.max_nodes is not None:
            mask = torch.cat(
                [
                    torch.ones(self.num_nodes, dtype=torch.bool),
                    torch.zeros(self.max_nodes - self.num_nodes, dtype=torch.bool),
                ]
            )

            mask = mask.unsqueeze(0).expand(self.num_timesteps, -1).unsqueeze(-1).bool()
            sample["padded_nodes_mask"] = mask.contiguous()

        if self.return_edge_data:
            sample["edge_attr"] = self.edge_attr.contiguous()
            sample["source_node_indices"] = self.edge_index[0].contiguous()
            sample["target_node_indices"] = self.edge_index[1].contiguous()

        if self.center_data:
            sample["x_0_mean"] = self.replicated_x_0_mean[i].contiguous()

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
        delta_frame: int | tuple[int, int],
        data_dir: str,
        split_dir: str,
        md17_version: Datasets,
        molecule_type: MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType,
        max_nodes: int | None,
        return_edge_data: bool,
        num_timesteps: int = 8,  # Number of timesteps for dynamics
        explicit_hydrogen: bool = False,
        train_par: float = 0.1,
        val_par: float = 0.05,
        test_par: float = 0.05,
        radius_graph_threshold: float = 1.6,
        rrwp_length: int = 0,
        normalize_z: bool = False,
        seed: int = 42,
        force_regenerate: bool = False,
        egno_mode: bool = False,
        max_edges: int | None = None,
        time_lag_mode: TimeLagMode = TimeLagMode.LAST,
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
            return_edge_data=return_edge_data,
            num_timesteps=num_timesteps,  # Pass num_timesteps to base class for replication
            explicit_hydrogen=explicit_hydrogen,
            train_par=train_par,
            val_par=val_par,
            test_par=test_par,
            seed=seed,
            force_regenerate=force_regenerate,
            rrwp_length=rrwp_length,
            radius_graph_threshold=radius_graph_threshold,
            normalize_z=normalize_z,
            egno_mode=egno_mode,
            max_edges=max_edges,
            time_lag_mode=time_lag_mode,
        )
        x_t, v_t = self.get_dynamic_target_frames()
        self.x_t = self._pad_tensor(x_t)
        self.v_t = self._pad_tensor(v_t)

        # Re-replicate dataset after defining x_t, v_t for dynamics dataset
        self.replicated_x_0: torch.Tensor = self._replicate_tensor(self.x_0)
        self.replicated_v_0: torch.Tensor = self._replicate_tensor(self.v_0)
        self.replicated_concatenated_features: torch.Tensor = self._replicate_tensor(self.concatenated_features)
        self.replicated_z_0: torch.Tensor = self._replicate_tensor(self.z_0)
        self.replicated_mole_idx: torch.Tensor = self._replicate_tensor(self.mole_idx)

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

        # Removed absolute time indices; not needed for local sinusoidal PE

    def get_dynamic_target_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        split_times = self.split_times
        delta_frame = self.delta_frame_max
        num_timesteps = self.num_timesteps

        if self.time_lag_mode == TimeLagMode.UNIFORM:
            x_t_list = [self.x[split_times + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
            v_t_list = [self.v[split_times + delta_frame * i // num_timesteps] for i in range(1, num_timesteps + 1)]
        else:
            x_t_list = [self.x[split_times + delta_frame] for _ in range(1, num_timesteps + 1)]
            v_t_list = [self.v[split_times + delta_frame] for _ in range(1, num_timesteps + 1)]
        x_t = np.stack(x_t_list, axis=1)
        v_t = np.stack(v_t_list, axis=1)

        x_t = torch.Tensor(x_t)
        v_t = torch.Tensor(v_t)
        return x_t, v_t

    @override
    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(i)

        # Randomize during training if a range was supplied
        if self.data_partition == DataPartition.train and (self.delta_frame_min != self.delta_frame_max):
            # Log-uniform sampling over [delta_min, delta_max]
            log_min = float(np.log(max(1, self.delta_frame_min)))
            log_max = float(np.log(self.delta_frame_max))
            u = float(np.random.uniform(low=log_min, high=log_max))
            delta_i = int(np.floor(np.exp(u)))
            delta_i = max(1, min(delta_i, self.delta_frame_max))

            if self.time_lag_mode == TimeLagMode.UNIFORM:
                inc = (delta_i * np.arange(1, self.num_timesteps + 1) // self.num_timesteps).astype(np.int64)
            else:
                inc = delta_i * np.ones(self.num_timesteps, dtype=np.int64)
            frame_idx = (int(self.split_times[i]) + inc).astype(np.int64)

            x_t_np = self.x[frame_idx]
            v_t_np = self.v[frame_idx]
            # Use Tensor() to keep float32 consistency with precomputed tensors
            x_t_t = torch.Tensor(x_t_np)
            v_t_t = torch.Tensor(v_t_np)

            if self.max_nodes is not None:
                pad_amt = self.max_nodes - x_t_t.shape[-2]
                if pad_amt > 0:
                    x_t_t = F.pad(x_t_t, (0, 0, 0, pad_amt))
                    v_t_t = F.pad(v_t_t, (0, 0, 0, pad_amt))

            sample["x_t"] = x_t_t.contiguous()
            sample["v_t"] = v_t_t.contiguous()
            # Removed absolute time indices and delta_t; only targets are updated

        # Provide per-step time increments for T-RoPE if using dynamics
        # Shape: [T], values equal to (delta / T) so cumulative gives [0, Δ, 2Δ, ...]
        if hasattr(self, "num_timesteps") and self.num_timesteps > 0:
            # Use configured max horizon for uniform per-step increments; keeps T-RoPE relative
            dt_val = float(self.delta_frame_max)
            per_step = dt_val / float(self.num_timesteps)
            sample["time_increments"] = torch.full((self.num_timesteps,), per_step, dtype=torch.float32)

        return sample


if __name__ == "__main__":
    # Test MD17Dataset
    # dataset_static = MD17Dataset(
    #     partition=DataPartition.train,
    #     max_samples=5000,
    #     delta_frame=3000,
    #     data_dir="data/",
    #     split_dir="data/",
    #     md17_version=Datasets.tg80,
    #     molecule_type=TG80MoleculeType.uracil,
    #     max_nodes=None,
    #     force_regenerate=False,
    #     num_timesteps=1,  # Set num_timesteps for replication
    #     return_edge_data=False,
    # )
    # dataloader_static = DataLoader(dataset_static, batch_size=100, shuffle=True)
    # print("MD17Dataset Output Shapes:")
    # for data in dataloader_static:
    #     for key in data:
    #         if key not in ["cfg", "edge_attr", "edge_index"]:
    #             print(f"  {key}:", data[key].shape)
    #     if "cfg" in data:
    #         print("  cfg shapes:")
    #         for key in data["cfg"]:
    #             print(f"    {key}:", data["cfg"][key].shape)
    #     break

    # Test MD17DynamicsDataset
    dataset_dynamic = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=3000,
        data_dir="data/",
        split_dir="data/",
        md17_version=Datasets.tg80,
        molecule_type=TG80MoleculeType.uracil,
        max_nodes=None,
        force_regenerate=False,
        num_timesteps=8,  # Set num_timesteps for replication
        return_edge_data=False,
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
