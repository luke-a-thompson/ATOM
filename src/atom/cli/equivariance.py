import argparse
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from torch.utils.data import DataLoader

from atom.inference.inference_utils import clean_state_dict_prefixes
from atom.training import (
    Config,
    create_dataloaders_multitask,
    create_dataloaders_single,
    initialize_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model equivariance to 3D rotations")
    _ = parser.add_argument("--config", type=str, help="Path to the config file")
    _ = parser.add_argument("--model", type=str, help="Path to the model checkpoint")
    _ = parser.add_argument(
        "--test_model",
        action="store_true",
        help="Run the supervised rotation robustness test (requires --config and --model)",
    )
    _ = parser.add_argument(
        "--test_model_equiv_defect",
        action="store_true",
        help="Run the Monte Carlo equivariance defect test (requires --config and --model)",
    )
    _ = parser.add_argument(
        "--rotation_seed",
        type=int,
        default=42,
        help="Random seed for generating rotations (default: 42)",
    )
    _ = parser.add_argument(
        "--num_rotations",
        type=int,
        default=20,
        help="Number of random rotations to test per batch (default: 20)",
    )
    _ = parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of test batches to average over (default: 1)",
    )
    return parser.parse_args()


def load_model_and_loader(config_path: str, model_path: str) -> tuple[Config, torch.nn.Module, DataLoader]:
    """Loads config, model, and the test dataloader."""
    try:
        config = Config.from_toml(Path(config_path))
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)

    try:
        model_state_dict = torch.load(model_path, map_location=config.training.device, weights_only=True)
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)

    if config.dataloader.multitask:
        _, _, test_loader = create_dataloaders_multitask(config)
    else:
        _, _, test_loader = create_dataloaders_single(config)

    model = initialize_model(config).to(config.training.device)
    _ = model.load_state_dict(clean_state_dict_prefixes(model_state_dict))
    _ = model.eval()

    return config, model, test_loader


def fixed_rotation_matrix() -> npt.NDArray[np.float64]:
    """Deterministic SO(3) used across runs for rotation tests."""
    return Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix()


def generate_random_rotations(seed: int, num_rotations: int) -> list[npt.NDArray[np.float64]]:
    """Generate multiple random rotation matrices using a seed.

    Args:
        seed: Random seed for reproducibility
        num_rotations: Number of rotation matrices to generate

    Returns:
        List of 3x3 rotation matrices
    """
    rng = np.random.default_rng(seed)
    rotations: list[npt.NDArray[np.float64]] = []
    for _ in range(num_rotations):
        # Generate random rotation by sampling random Euler angles
        # This ensures reproducibility with the seed
        euler_angles = rng.uniform(0, 2 * np.pi, size=3)
        rot = Rotation.from_euler("xyz", euler_angles)
        rotations.append(rot.as_matrix())
    return rotations


def apply_rotation(data: TensorDict, rotation_matrix: npt.NDArray[np.float64]) -> TensorDict:
    """Applies a rotation matrix to the spatial components of the data dictionary."""
    rotated_data = data.clone()

    feature_config: dict[str, dict[str, slice]] = {
        "x_0": {"xyz": slice(0, 3), "invariant": slice(3, None)},
        "v_0": {"xyz": slice(0, 3), "invariant": slice(3, None)},
        "concatenated_features": {
            "x_0_xyz": slice(0, 3),
            "v_0_xyz": slice(4, 7),
        },
        "output": {"xyz": slice(0, 3), "invariant": slice(3, None)},
    }

    def _rotate_slice(tensor: torch.Tensor, rot_mat: npt.NDArray[np.float64], xyz_slice: slice) -> torch.Tensor:
        xyz = tensor[..., xyz_slice].cpu().numpy()
        original_shape = xyz.shape
        xyz_reshaped = xyz.reshape(-1, 3)
        rotated_xyz = xyz_reshaped @ rot_mat.T
        rotated_xyz_tensor = torch.tensor(
            rotated_xyz.reshape(original_shape),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        new_tensor = tensor.clone()
        new_tensor[..., xyz_slice] = rotated_xyz_tensor
        return new_tensor

    for key in ["x_0", "v_0"]:
        if key in rotated_data:
            xyz_slice = feature_config[key]["xyz"]
            rotated_data[key] = _rotate_slice(rotated_data[key], rotation_matrix, xyz_slice)

    if "concatenated_features" in rotated_data:
        tensor = rotated_data["concatenated_features"]
        feature_spec = feature_config["concatenated_features"]
        new_tensor = tensor.clone()
        for _, xyz_slice in feature_spec.items():
            xyz = tensor[..., xyz_slice].cpu().numpy()
            original_shape = xyz.shape
            xyz_reshaped = xyz.reshape(-1, 3)
            rotated_xyz = xyz_reshaped @ rotation_matrix.T
            rotated_xyz_tensor = torch.tensor(
                rotated_xyz.reshape(original_shape),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            new_tensor[..., xyz_slice] = rotated_xyz_tensor
        rotated_data["concatenated_features"] = new_tensor

    return rotated_data


def test_model_equivariance(
    config_path: str,
    model_path: str,
    rotation_seed: int = 42,
    num_rotations: int = 20,
    num_batches: int = 10,
) -> None:
    """Tests supervised performance under rotated inputs and prints results.

    Averages metrics over multiple batches and multiple random rotations.

    Args:
        config_path: Path to config file.
        model_path: Path to model checkpoint.
        rotation_seed: Random seed for generating rotations.
        num_rotations: Number of random rotations to test per batch.
        num_batches: Number of test batches to average over.
    """
    print("Loading model and data...")
    config, model, test_loader = load_model_and_loader(config_path, model_path)
    device = config.training.device

    # Pre-generate rotations once and reuse across batches
    print(f"Generating {num_rotations} random rotations with seed {rotation_seed}...")
    rotation_matrices = generate_random_rotations(rotation_seed, num_rotations)

    output_spec = {"xyz": slice(0, 3), "invariant": slice(3, None)}
    xyz_slice = output_spec["xyz"]

    mse_unrot_vs_gt_list: list[float] = []
    mse_rot_vs_gt_list: list[float] = []

    print(f"Using up to {num_batches} batch(es) from the test loader...")
    for batch_index, batch in enumerate(test_loader):
        if batch_index >= num_batches:
            break

        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        data_sample = TensorDict(batch_on_device, batch_size=batch_on_device["x_0"].shape[0])

        print(f"\nBatch {batch_index + 1}: computing original output...")
        with torch.no_grad():
            original_output = model(data_sample)

        # Model may return a dict (e.g., {"pos": tensor, ...}) or a raw tensor
        original_pos = original_output["pos"] if isinstance(original_output, dict) else original_output
        original_xyz = original_pos[..., xyz_slice].cpu().numpy()
        batch_size, timesteps, nodes, _ = original_xyz.shape

        gt_xyz = data_sample["x_t"][..., :3].cpu().numpy()
        mse_unrot_vs_gt = float(np.mean((original_xyz - gt_xyz) ** 2))
        mse_unrot_vs_gt_list.append(mse_unrot_vs_gt)

        print(f"Batch {batch_index + 1}: testing {num_rotations} rotations...")
        for i, rotation_matrix in enumerate(rotation_matrices, 1):
            print(f"  Rotation {i}/{num_rotations}...", end="\r")

            rotated_data_sample = apply_rotation(data_sample, rotation_matrix)

            with torch.no_grad():
                rotated_output = model(rotated_data_sample)

            rotated_pos = rotated_output["pos"] if isinstance(rotated_output, dict) else rotated_output
            rotated_xyz = rotated_pos[..., xyz_slice].cpu().numpy()

            # Compute MSE vs rotated ground truth
            gt_xyz_rot = gt_xyz.reshape(-1, 3) @ rotation_matrix.T
            gt_xyz_rot = gt_xyz_rot.reshape(batch_size, timesteps, nodes, 3)
            mse_rot_vs_gt = float(np.mean((rotated_xyz - gt_xyz_rot) ** 2))
            mse_rot_vs_gt_list.append(mse_rot_vs_gt)

    print()  # New line after progress updates

    if len(mse_unrot_vs_gt_list) == 0:
        print("Error: No batches were processed from the test loader.")
        return

    # Aggregate results
    mse_unrot_vs_gt_mean = float(np.mean(mse_unrot_vs_gt_list))
    mse_unrot_vs_gt_std = float(np.std(mse_unrot_vs_gt_list))

    mse_rot_vs_gt_mean = float(np.mean(mse_rot_vs_gt_list))
    mse_rot_vs_gt_std = float(np.std(mse_rot_vs_gt_list))
    mse_rot_vs_gt_min = float(np.min(mse_rot_vs_gt_list))
    mse_rot_vs_gt_max = float(np.max(mse_rot_vs_gt_list))

    print("\n=== MODEL SUPERVISED METRICS ===")
    scale_label: str = "x10^-2"
    scale_factor: float = 1e2
    mse_unrot_vs_gt_mean_scaled: float = float(mse_unrot_vs_gt_mean * scale_factor)
    mse_unrot_vs_gt_2std_scaled: float = float(2.0 * mse_unrot_vs_gt_std * scale_factor)

    mse_rot_vs_gt_mean_scaled: float = float(mse_rot_vs_gt_mean * scale_factor)
    mse_rot_vs_gt_2std_scaled: float = float(2.0 * mse_rot_vs_gt_std * scale_factor)
    mse_rot_vs_gt_min_scaled: float = float(mse_rot_vs_gt_min * scale_factor)
    mse_rot_vs_gt_max_scaled: float = float(mse_rot_vs_gt_max * scale_factor)

    print(
        f"MSE vs GT (unrot input) over {len(mse_unrot_vs_gt_list)} batch(es): "
        f"{mse_unrot_vs_gt_mean_scaled:.2f} ± {mse_unrot_vs_gt_2std_scaled:.2f} {scale_label}"
    )
    print(f"\nMSE vs GT (rot input) over {len(mse_rot_vs_gt_list)} (batch, rotation) pairs:")
    print(f"  Mean: {mse_rot_vs_gt_mean_scaled:.2f} {scale_label}")
    print(f"  2Std: {mse_rot_vs_gt_2std_scaled:.2f} {scale_label}")
    print(f"  Min:  {mse_rot_vs_gt_min_scaled:.2f} {scale_label}")
    print(f"  Max:  {mse_rot_vs_gt_max_scaled:.2f} {scale_label}")


def test_model_equivariance_defect_mc(
    config_path: str,
    model_path: str,
    rotation_seed: int = 42,
    num_rotations: int = 20,
    num_batches: int = 1,
) -> None:
    """Monte Carlo estimate of the equivariance defect for the model.

    This approximates the group-averaged equivariance defect
    E_x [ || (1/|G|) sum_g f(phi(g)(x)) - (1/|G|) sum_g rho(g)(f(x)) || ]
    using a finite set of sampled rotations.

    Args:
        config_path: Path to config file
        model_path: Path to model checkpoint
        rotation_seed: Random seed for generating rotations.
        num_rotations: Number of random rotations to sample from SO(3) per batch.
        num_batches: Number of test batches to average over.
    """
    print("Loading model and data...")
    config, model, test_loader = load_model_and_loader(config_path, model_path)
    device = config.training.device

    output_spec = {"xyz": slice(0, 3), "invariant": slice(3, None)}
    xyz_slice = output_spec["xyz"]

    # Generate random rotations once and move to device
    print(f"Generating {num_rotations} random rotations with seed {rotation_seed} for equivariance defect...")
    rotation_matrices_np = generate_random_rotations(rotation_seed, num_rotations)
    rotation_matrices_torch: list[torch.Tensor] = [torch.tensor(R_np, device=device, dtype=torch.float32) for R_np in rotation_matrices_np]

    all_norms: list[np.ndarray] = []

    print(f"Estimating equivariance defect over {num_rotations} rotations and up to {num_batches} batch(es)...")
    for batch_index, batch in enumerate(test_loader):
        if batch_index >= num_batches:
            break

        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        data_sample = TensorDict(batch_on_device, batch_size=batch_on_device["x_0"].shape[0])

        print(f"\nBatch {batch_index + 1}: computing original output...")
        with torch.no_grad():
            original_output = model(data_sample)

        # Model may return a dict (e.g., {"pos": tensor, ...}) or a raw tensor
        original_pos = original_output["pos"] if isinstance(original_output, dict) else original_output
        original_xyz: torch.Tensor = original_pos[..., xyz_slice]
        batch_size, timesteps, nodes, _ = original_xyz.shape

        dtype: torch.dtype = original_xyz.dtype
        rotation_matrices_batch: list[torch.Tensor] = [R.to(device=device, dtype=dtype) for R in rotation_matrices_torch]

        # Accumulate Monte Carlo estimates of the two group-averaged terms for this batch
        sum_f_phi: torch.Tensor = torch.zeros_like(original_xyz)
        sum_rho_fx: torch.Tensor = torch.zeros_like(original_xyz)

        for i, R in enumerate(rotation_matrices_batch, 1):
            print(f"  Rotation {i}/{num_rotations}...", end="\r")

            # 1) f(phi(R)(x)): rotate inputs, run model
            R_np: npt.NDArray[np.float64] = R.detach().cpu().numpy()
            rotated_data_sample = apply_rotation(data_sample, R_np)
            with torch.no_grad():
                rotated_output = model(rotated_data_sample)
            rotated_pos = rotated_output["pos"] if isinstance(rotated_output, dict) else rotated_output
            f_phi_R_x: torch.Tensor = rotated_pos[..., xyz_slice]
            sum_f_phi = sum_f_phi + f_phi_R_x

            # 2) rho(R)(f(x)): rotate the original prediction
            original_flat: torch.Tensor = original_xyz.reshape(-1, 3)
            rho_R_fx_flat: torch.Tensor = original_flat @ R.T
            rho_R_fx: torch.Tensor = rho_R_fx_flat.reshape(batch_size, timesteps, nodes, 3)
            sum_rho_fx = sum_rho_fx + rho_R_fx

        # Compute Monte Carlo averages over sampled rotations for this batch
        avg_f_phi: torch.Tensor = sum_f_phi / float(num_rotations)
        avg_rho_fx: torch.Tensor = sum_rho_fx / float(num_rotations)

        # Equivariance defect tensor: shape [B, T, N, 3]
        defect: torch.Tensor = avg_f_phi - avg_rho_fx

        # Pointwise L2 norm over xyz
        per_point_norm: torch.Tensor = torch.sqrt(torch.sum(defect * defect, dim=-1))
        all_norms.append(per_point_norm.detach().cpu().numpy().ravel())

    print()  # New line after progress updates

    if len(all_norms) == 0:
        print("Error: No batches were processed from the test loader.")
        return

    all_norms_concat = np.concatenate(all_norms, axis=0)
    eps_mean: float = float(all_norms_concat.mean())
    eps_std: float = float(all_norms_concat.std())
    eps_max: float = float(all_norms_concat.max())

    print("\n=== MODEL EQUIVARIANCE DEFECT (MONTE CARLO) ===")
    scale_label: str = "x10^-2"
    scale_factor: float = 1e2
    eps_mean_scaled: float = float(eps_mean * scale_factor)
    eps_2std_scaled: float = float(2.0 * eps_std * scale_factor)
    eps_max_scaled: float = float(eps_max * scale_factor)

    print(f"Average ||defect||: {eps_mean_scaled:.2f} ± {eps_2std_scaled:.2f} {scale_label}")
    print(f"Max     ||defect||: {eps_max_scaled:.2f} {scale_label}")


def main_rotation_loss_robustness() -> None:
    """Entry point for the rotation loss robustness CLI."""
    args = parse_args()
    if not args.config or not args.model:
        print("Error: rotation_loss_robustness requires both --config and --model.")
        sys.exit(2)
    test_model_equivariance(
        args.config,
        args.model,
        rotation_seed=args.rotation_seed,
        num_rotations=args.num_rotations,
        num_batches=args.num_batches,
    )


def main_equivariance_defect() -> None:
    """Entry point for the Monte Carlo equivariance defect CLI."""
    args = parse_args()
    if not args.config or not args.model:
        print("Error: equivariance_defect requires both --config and --model.")
        sys.exit(2)
    test_model_equivariance_defect_mc(
        args.config,
        args.model,
        rotation_seed=args.rotation_seed,
        num_rotations=args.num_rotations,
        num_batches=args.num_batches,
    )


def main() -> None:
    args = parse_args()
    if args.test_model_equiv_defect:
        if not args.config or not args.model:
            print("Error: --test_model_equiv_defect requires both --config and --model.")
            sys.exit(2)
        test_model_equivariance_defect_mc(
            args.config,
            args.model,
            rotation_seed=args.rotation_seed,
            num_rotations=args.num_rotations,
            num_batches=args.num_batches,
        )
        return

    # Default to supervised rotation robustness when --test_model_equiv_defect is not set
    if not args.config or not args.model:
        print("Error: --test_model requires both --config and --model.")
        sys.exit(2)
    test_model_equivariance(
        args.config,
        args.model,
        rotation_seed=args.rotation_seed,
        num_rotations=args.num_rotations,
        num_batches=args.num_batches,
    )


if __name__ == "__main__":
    main()
