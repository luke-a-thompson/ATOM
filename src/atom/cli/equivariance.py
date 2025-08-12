import argparse
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import torch
from e3nn import o3
from tensordict import TensorDict

from atom.inference.inference_utils import clean_state_dict_prefixes
from atom.training import Config, create_dataloaders_multitask, create_dataloaders_single, initialize_model
from atom.atom.lifting_layers import CanonicalizationLift


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model equivariance to 3D rotations")
    _ = parser.add_argument("--config", type=str, help="Path to the config file")
    _ = parser.add_argument("--model", type=str, help="Path to the model checkpoint")
    _ = parser.add_argument("--test_e3nn", action="store_true", help="Test with a simple E3NN linear layer")
    _ = parser.add_argument("--test_canonicalizer", action="store_true", help="Test with a canonicalizer")
    return parser.parse_args()


def load_model_and_data(config_path: str, model_path: str) -> tuple[Config, torch.nn.Module, TensorDict]:
    """Loads config, model, and a single data sample."""
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

    data_sample = next(iter(test_loader))
    data_sample = {k: v.to(config.training.device) for k, v in data_sample.items()}
    data_sample = TensorDict(data_sample, batch_size=data_sample["x_0"].shape[0])

    return config, model, data_sample


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
        rotated_xyz_tensor = torch.tensor(rotated_xyz.reshape(original_shape), device=tensor.device, dtype=tensor.dtype)
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
            rotated_xyz_tensor = torch.tensor(rotated_xyz.reshape(original_shape), device=tensor.device, dtype=tensor.dtype)
            new_tensor[..., xyz_slice] = rotated_xyz_tensor
        rotated_data["concatenated_features"] = new_tensor

    return rotated_data


def test_model_equivariance(config_path: str, model_path: str) -> None:
    """Tests the model's equivariance to 3D rotations and prints results."""
    print("Loading model and data...")
    _, model, data_sample = load_model_and_data(config_path, model_path)

    print("Computing original output...")
    with torch.no_grad():
        original_output = model(data_sample)

    random_rotation: npt.NDArray[np.float64] = o3.rand_matrix().cpu().numpy()
    print(f"Generated rotation matrix:\n{random_rotation}")

    print("Applying rotation to input data...")
    rotated_data_sample = apply_rotation(data_sample, random_rotation)

    print("Computing rotated output...")
    with torch.no_grad():
        rotated_output = model(rotated_data_sample)

    output_spec = {"xyz": slice(0, 3), "invariant": slice(3, None)}
    xyz_slice = output_spec["xyz"]

    original_xyz = original_output[..., xyz_slice].cpu().numpy()
    rotated_xyz = rotated_output[..., xyz_slice].cpu().numpy()

    batch_size, timesteps, nodes, _ = original_xyz.shape
    original_xyz_reshaped = original_xyz.reshape(-1, 3)
    expected_rotated_xyz = original_xyz_reshaped @ random_rotation.T
    expected_rotated_xyz = expected_rotated_xyz.reshape(batch_size, timesteps, nodes, 3)

    error = np.abs(rotated_xyz - expected_rotated_xyz)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print("\n=== EQUIVARIANCE TEST RESULTS ===")
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")

    tolerance = 1e-6
    is_equivariant = max_error < tolerance

    if is_equivariant:
        print("✅ The model is EQUIVARIANT to 3D rotations!")
    else:
        print("❌ The model is NOT EQUIVARIANT to 3D rotations!")
        print(f"   Error exceeds tolerance of {tolerance:.2e}")


def test_e3nn_linear_equivariance() -> None:
    """Tests that a simple E3NN linear layer is equivariant to rotations."""
    print("Testing E3NN linear layer equivariance...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size, timesteps, nodes, xyz_dim = 2, 1, 5, 3
    input_tensor = torch.randn(batch_size, timesteps, nodes, xyz_dim, device=device)

    irreps_in = o3.Irreps("1o")
    irreps_out = o3.Irreps("1o")
    e3nn_linear = o3.Linear(irreps_in, irreps_out).to(device)

    with torch.no_grad():
        original_output = e3nn_linear(input_tensor)

    random_rotation: npt.NDArray[np.float64] = o3.rand_matrix().cpu().numpy()
    print(f"Generated rotation matrix:\n{random_rotation}")

    input_reshaped = input_tensor.reshape(-1, 3).cpu().numpy()
    rotated_input_np = input_reshaped @ random_rotation.T
    rotated_input = torch.tensor(rotated_input_np.reshape(batch_size, timesteps, nodes, 3), device=device, dtype=input_tensor.dtype)

    with torch.no_grad():
        rotated_output = e3nn_linear(rotated_input)

    original_xyz = original_output.cpu().numpy()
    rotated_xyz = rotated_output.cpu().numpy()

    original_xyz_reshaped = original_xyz.reshape(-1, 3)
    expected_rotated_xyz = original_xyz_reshaped @ random_rotation.T
    expected_rotated_xyz = expected_rotated_xyz.reshape(batch_size, timesteps, nodes, 3)

    error = np.abs(rotated_xyz - expected_rotated_xyz)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print("\n=== E3NN LINEAR LAYER TEST RESULTS ===")
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")

    tolerance = 1e-6
    is_equivariant = max_error < tolerance

    if is_equivariant:
        print("✅ The E3NN linear layer is EQUIVARIANT to 3D rotations!")
    else:
        print("❌ The E3NN linear layer is NOT EQUIVARIANT to 3D rotations!")
        print(f"   Error exceeds tolerance of {tolerance:.2e}")


def test_canonicalizer_equivariance() -> None:
    """Tests the canonicalizer module's equivariance to 3D rotations."""
    print("Testing canonicalizer equivariance (module)...")

    B, T, N = 2, 1, 8
    x = torch.randn(B, T, N, 3)
    v = torch.randn(B, T, N, 3)
    Z = torch.randn(B, T, N, 1)

    canonicalizer = CanonicalizationLift(
        x_0_in_irreps="1x1o",
        v_0_in_irreps="1x1o",
        concat_feats_in_irreps="1x1o + 1x1o + 1x0e",
        lifting_dim_irreps="1x1o",
    )

    _, _, _, Q = canonicalizer(x, v, Z)

    R: torch.Tensor = o3.rand_matrix().to(x.device, x.dtype)
    x_rot: torch.Tensor = x @ R.T
    v_rot: torch.Tensor = v @ R.T

    _, _, _, Q_rot = canonicalizer(x_rot, v_rot, Z)

    assert torch.allclose(Q_rot, R @ Q, atol=1e-5), f"Frame does not transform correctly, ||Q_rot - RQ||={torch.norm(Q_rot - R @ Q)}"

    x_can: torch.Tensor = x @ Q
    v_can: torch.Tensor = v @ Q
    x_can_rot: torch.Tensor = x_rot @ Q_rot
    v_can_rot: torch.Tensor = v_rot @ Q_rot

    assert torch.allclose(x_can, x_can_rot, atol=1e-5), f"x canonicalization is not equivariant, error: {torch.norm(x_can - x_can_rot)}"
    assert torch.allclose(v_can, v_can_rot, atol=1e-5), f"v canonicalization is not equivariant, error: {torch.norm(v_can - v_can_rot)}"

    print("Canonicalizer module equivariance test passed.")


def main() -> None:
    args = parse_args()
    if args.test_e3nn:
        test_e3nn_linear_equivariance()
        return

    if args.test_canonicalizer:
        test_canonicalizer_equivariance()
        return

    if args.config and args.model:
        test_model_equivariance(args.config, args.model)
        return

    print("Please provide one of: --test_e3nn, --test_canonicalizer, or both --config and --model for model test.")
    sys.exit(2)


if __name__ == "__main__":
    main()
