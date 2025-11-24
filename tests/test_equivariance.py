import torch
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from pathlib import Path
import argparse
import sys
from tensordict import TensorDict
from e3nn import o3

from atom.training import (
    Config,
    initialize_model,
    create_dataloaders_single,
    create_dataloaders_multitask,
)
from atom.inference.inference_utils import clean_state_dict_prefixes
from atom.atom.lifting_layers import CanonicalizationLift


# Define the structure of features for rotation. This centralizes the logic and
# makes the tests more robust to changes in feature representation.
# It specifies which parts of the tensors are 3D vectors ("xyz") and which
# are invariant scalars ("invariant").
FEATURE_CONFIG: dict[str, dict[str, slice]] = {
    "x_0": {"xyz": slice(0, 3), "invariant": slice(3, None)},
    "v_0": {"xyz": slice(0, 3), "invariant": slice(3, None)},
    "concatenated_features": {
        "x_0_xyz": slice(0, 3),
        "v_0_xyz": slice(4, 7),
    },
    "output": {"xyz": slice(0, 3), "invariant": slice(3, None)},
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for simple directory-based evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate equivariance MSEs over an experiments directory"
    )
    _ = parser.add_argument(
        "root_dir",
        type=str,
        help="Path to root directory containing experiment subdirectories",
    )
    return parser.parse_args()


def load_model_and_data(
    config_path: str, model_path: str
) -> tuple[Config, torch.nn.Module, TensorDict]:
    """Loads config, model, and a single data sample."""
    try:
        config = Config.from_toml(Path(config_path))
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)

    try:
        model_state_dict = torch.load(
            model_path, map_location=config.training.device, weights_only=True
        )
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

    # Get a single sample and convert to TensorDict
    data_sample = next(iter(test_loader))
    # Ensure all tensors are on the same device
    data_sample = {k: v.to(config.training.device) for k, v in data_sample.items()}
    data_sample = TensorDict(data_sample, batch_size=data_sample["x_0"].shape[0])

    return config, model, data_sample


def apply_rotation(
    data: TensorDict, rotation_matrix: npt.NDArray[np.float64]
) -> TensorDict:
    """Applies a rotation matrix to the spatial components of the data dictionary."""
    rotated_data = data.clone()

    def _rotate_slice(
        tensor: torch.Tensor, rot_mat: npt.NDArray[np.float64], xyz_slice: slice
    ) -> torch.Tensor:
        """Helper to rotate a specific slice of a tensor."""
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

    # Rotate simple features like x_0 and v_0
    for key in ["x_0", "v_0"]:
        if key in rotated_data:
            xyz_slice = FEATURE_CONFIG[key]["xyz"]
            rotated_data[key] = _rotate_slice(
                rotated_data[key], rotation_matrix, xyz_slice
            )

    # Handle concatenated_features which may have multiple vectors
    if "concatenated_features" in rotated_data:
        tensor = rotated_data["concatenated_features"]
        feature_spec = FEATURE_CONFIG["concatenated_features"]

        # Create a new tensor to hold all rotated components
        new_tensor = tensor.clone()

        # Rotate each vector component specified in the config
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


def _fixed_rotation_matrix() -> npt.NDArray[np.float64]:
    return Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix()


def evaluate_single_run(config_path: str, model_path: str) -> dict[str, float]:
    """Return supervised MSEs for one model on one deterministic batch.

    Returns a dict with keys: "mse_unrot" and "mse_rot".
    """
    config, model, data_sample = load_model_and_data(config_path, model_path)

    base_sample = data_sample.clone()
    with torch.no_grad():
        original_output = model(base_sample.clone())

    R = _fixed_rotation_matrix()
    rotated_data_sample = apply_rotation(base_sample.clone(), R)
    with torch.no_grad():
        rotated_output = model(rotated_data_sample)

    xyz_slice = FEATURE_CONFIG["output"]["xyz"]
    original_xyz = original_output[..., xyz_slice].cpu().numpy()
    rotated_xyz = rotated_output[..., xyz_slice].cpu().numpy()

    gt_xyz = data_sample["x_t"][..., :3].cpu().numpy()
    mse_unrot_vs_gt = float(np.mean((original_xyz - gt_xyz) ** 2))

    batch_size, timesteps, nodes, _ = original_xyz.shape
    gt_xyz_rot = gt_xyz.reshape(-1, 3) @ R.T
    gt_xyz_rot = gt_xyz_rot.reshape(batch_size, timesteps, nodes, 3)
    mse_rot_vs_gt = float(np.mean((rotated_xyz - gt_xyz_rot) ** 2))

    return {"mse_unrot": mse_unrot_vs_gt, "mse_rot": mse_rot_vs_gt}


def run_equivariance_ablations(root_dir: str) -> None:
    """Scan experiments under root_dir and print mean ± 2SD for MSEs.

    Expected layout per experiment:
      <exp_dir>/config.toml (or any *.toml)
      <exp_dir>/run_*/best_val_model.pth
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        print(f"Error: '{root_dir}' is not a directory")
        return

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(exp_dirs) == 0:
        print(f"No experiment directories found under '{root_dir}'")
        return

    for exp in exp_dirs:
        # Find a config TOML at the experiment root
        tomls = sorted(exp.glob("*.toml"))
        if len(tomls) == 0:
            print(f"[skip] No TOML found in {exp}")
            continue
        config_path = str(tomls[0])

        # Find model checkpoints under run_*/
        run_dirs = sorted([p for p in exp.glob("run_*") if p.is_dir()])
        model_paths: list[str] = []
        for rd in run_dirs:
            model_file = rd / "best_val_model.pth"
            if model_file.exists():
                model_paths.append(str(model_file))

        if len(model_paths) == 0:
            print(f"[skip] No runs with checkpoints in {exp}")
            continue

        per_run_unrot: list[float] = []
        per_run_rot: list[float] = []
        for mp in model_paths:
            res = evaluate_single_run(config_path, mp)
            per_run_unrot.append(res["mse_unrot"])
            per_run_rot.append(res["mse_rot"])

        mean_unrot = float(np.mean(per_run_unrot))
        sd_unrot = (
            float(np.std(per_run_unrot, ddof=1)) if len(per_run_unrot) > 1 else 0.0
        )
        mean_rot = float(np.mean(per_run_rot))
        sd_rot = float(np.std(per_run_rot, ddof=1)) if len(per_run_rot) > 1 else 0.0

        # Per-run differences (rotated - unrotated) for mean ± 2SD reporting
        per_run_diff = [r - u for r, u in zip(per_run_rot, per_run_unrot)]
        mean_diff = float(np.mean(per_run_diff))
        sd_diff = float(np.std(per_run_diff, ddof=1)) if len(per_run_diff) > 1 else 0.0

        # Scale to ×10^-2 units for display
        scale = 100.0
        mean_unrot_scaled = mean_unrot * scale
        two_sd_unrot_scaled = (2 * sd_unrot) * scale
        mean_rot_scaled = mean_rot * scale
        two_sd_rot_scaled = (2 * sd_rot) * scale
        mean_diff_scaled = mean_diff * scale
        two_sd_diff_scaled = (2 * sd_diff) * scale

        print(f"{exp.name} ({len(model_paths)} runs)")
        print(
            f"  S2T MSE vs ground truth (unrotated input): \\({mean_unrot_scaled:.3f}{{\\scriptstyle \\pm{two_sd_unrot_scaled:.3f}}}\\) \\times 10^{{-2}}"
        )
        print(
            f"  S2T MSE vs ground truth (rotated input):   \\({mean_rot_scaled:.3f}{{\\scriptstyle \\pm{two_sd_rot_scaled:.3f}}}\\) \\times 10^{{-2}}"
        )
        print(
            f"  Difference (rotated − unrotated):      \\({mean_diff_scaled:.3f}{{\\scriptstyle \\pm{two_sd_diff_scaled:.3f}}}\\) \\times 10^{{-2}}"
        )


def test_e3nn_linear_equivariance() -> None:
    """Tests that a simple E3NN linear layer is equivariant to rotations."""
    print("Testing E3NN linear layer equivariance...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a simple input tensor with shape [batch, timesteps, nodes, xyz_dim]
    batch_size, timesteps, nodes, xyz_dim = 2, 1, 5, 3
    input_tensor = torch.randn(batch_size, timesteps, nodes, xyz_dim, device=device)

    # Create a simple E3NN linear layer
    # This layer takes 3D vectors as input and outputs 3D vectors
    irreps_in = o3.Irreps("1o")  # 3D vectors
    irreps_out = o3.Irreps("1o")  # 3D vectors
    e3nn_linear = o3.Linear(irreps_in, irreps_out).to(device)

    # 1. Get output for the original input
    with torch.no_grad():
        original_output = e3nn_linear(input_tensor)

    # 2. Use a fixed deterministic rotation matrix (Euler xyz: 0.3, -0.7, 1.1)
    random_rotation: npt.NDArray[np.float64] = Rotation.from_euler(
        "xyz", [0.3, -0.7, 1.1]
    ).as_matrix()
    print(f"Generated rotation matrix:\n{random_rotation}")

    # 3. Apply rotation to the input
    # Reshape for easier processing
    input_reshaped = input_tensor.reshape(-1, 3).cpu().numpy()
    rotated_input_np = input_reshaped @ random_rotation.T
    rotated_input = torch.tensor(
        rotated_input_np.reshape(batch_size, timesteps, nodes, 3),
        device=device,
        dtype=input_tensor.dtype,
    )

    # 4. Get output for the rotated input
    with torch.no_grad():
        rotated_output = e3nn_linear(rotated_input)

    # 5. Compare outputs
    # Extract the xyz coordinates from the output tensors
    original_xyz = original_output.cpu().numpy()
    rotated_xyz = rotated_output.cpu().numpy()

    # Reshape for easier processing
    original_xyz_reshaped = original_xyz.reshape(-1, 3)

    # Apply the same rotation to the original output
    expected_rotated_xyz = original_xyz_reshaped @ random_rotation.T
    expected_rotated_xyz = expected_rotated_xyz.reshape(batch_size, timesteps, nodes, 3)

    # Calculate the error
    error = np.abs(rotated_xyz - expected_rotated_xyz)
    max_error = np.max(error)
    mean_error = np.mean(error)
    mse = np.mean((rotated_xyz - expected_rotated_xyz) ** 2)

    print("\n=== E3NN LINEAR LAYER TEST RESULTS ===")
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")
    print(f"MSE: {mse:.2e}")

    # Check if the layer is equivariant
    tolerance = 1e-6
    is_equivariant = max_error < tolerance

    if is_equivariant:
        print("✅ The E3NN linear layer is EQUIVARIANT to 3D rotations!")
    else:
        print("❌ The E3NN linear layer is NOT EQUIVARIANT to 3D rotations!")
        print(f"   Error exceeds tolerance of {tolerance:.2e}")


def test_canonicalizer_equivariance(config_path: str, model_path: str) -> None:
    """Tests the canonicalizer module's equivariance to 3D rotations."""
    print("Testing canonicalizer equivariance (module)...")

    # Create a small synthetic batch [B, T, N, D]
    B, T, N = 2, 1, 8
    x = torch.randn(B, T, N, 3)
    v = torch.randn(B, T, N, 3)
    Z = torch.randn(B, T, N, 1)

    # Instantiate the canonicalizer with simple irreps consistent with our tensors
    # x, v are 3D vectors (1x1o). We output vectors again (1x1o).
    # test_linear input is cat(x_can (1x1o), vz_can (1x1o + 1x0e)) -> 1x1o + 1x1o + 1x0e
    canonicalizer = CanonicalizationLift(
        x_0_in_irreps="1x1o",
        v_0_in_irreps="1x1o",
        concat_feats_in_irreps="1x1o + 1x1o + 1x0e",
        lifting_dim_irreps="1x1o",
    )

    # Forward on original inputs
    _, _, _, Q = canonicalizer(x, v, Z)

    # Use a fixed deterministic rotation (Euler xyz: 0.3, -0.7, 1.1)
    R_np: npt.NDArray[np.float64] = Rotation.from_euler(
        "xyz", [0.3, -0.7, 1.1]
    ).as_matrix()
    R: torch.Tensor = torch.tensor(R_np, device=x.device, dtype=x.dtype)
    x_rot: torch.Tensor = x @ R.T
    v_rot: torch.Tensor = v @ R.T

    # Forward on rotated inputs
    _, _, _, Q_rot = canonicalizer(x_rot, v_rot, Z)

    # The frame should transform as Q_rot = R @ Q
    assert torch.allclose(Q_rot, R @ Q, atol=1e-5), (
        f"Frame does not transform correctly, ||Q_rot - RQ||={torch.norm(Q_rot - R @ Q)}"
    )

    # Canonicalized coordinates should be invariant
    x_can: torch.Tensor = x @ Q
    v_can: torch.Tensor = v @ Q
    x_can_rot: torch.Tensor = x_rot @ Q_rot
    v_can_rot: torch.Tensor = v_rot @ Q_rot

    assert torch.allclose(x_can, x_can_rot, atol=1e-5), (
        f"x canonicalization is not equivariant, error: {torch.norm(x_can - x_can_rot)}"
    )
    assert torch.allclose(v_can, v_can_rot, atol=1e-5), (
        f"v canonicalization is not equivariant, error: {torch.norm(v_can - v_can_rot)}"
    )

    print("Canonicalizer module equivariance test passed.")


def main() -> None:
    args = parse_args()
    run_equivariance_ablations(args.root_dir)


if __name__ == "__main__":
    main()
