import torch
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
import pytest
from pathlib import Path
import argparse
import sys
from tensordict import TensorDict
from e3nn import o3

from atom.training import Config, initialize_model, create_dataloaders_single, create_dataloaders_multitask
from atom.inference.inference_utils import clean_state_dict_prefixes


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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model equivariance to 3D rotations")
    _ = parser.add_argument("--config", type=str, help="Path to the config file", default="configs/ablations_atom/default.toml")
    _ = parser.add_argument(
        "--model",
        type=str,
        help="Path to the model checkpoint",
        default="benchmark_runs/atom_default_singletask_09-Aug-2025_22-26-38/run_1/best_val_model.pth",
    )
    _ = parser.add_argument("--test_e3nn", action="store_true", help="Test with a simple E3NN linear layer")
    return parser.parse_args()


@pytest.fixture(scope="module")
def setup_data() -> tuple[Config, torch.nn.Module, TensorDict]:
    """Loads config, model, and a single data sample."""
    args = parse_args()

    # Skip this fixture if we're only testing E3NN
    if args.test_e3nn:
        pytest.skip("Skipping model test when running E3NN test")

    # Check if config and model paths are provided
    if not args.config or not args.model:
        pytest.skip("Config and model paths are required for the model test")

    try:
        config = Config.from_toml(Path(args.config))
    except FileNotFoundError:
        pytest.skip(f"Config file {args.config} not found")

    try:
        model_state_dict = torch.load(args.model, map_location=config.training.device, weights_only=True)
    except FileNotFoundError:
        pytest.skip(f"Model file {args.model} not found")

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


def apply_rotation(data: TensorDict, rotation_matrix: npt.NDArray[np.float64]) -> TensorDict:
    """Applies a rotation matrix to the spatial components of the data dictionary."""
    rotated_data = data.clone()

    def _rotate_slice(tensor: torch.Tensor, rot_mat: npt.NDArray[np.float64], xyz_slice: slice) -> torch.Tensor:
        """Helper to rotate a specific slice of a tensor."""
        xyz = tensor[..., xyz_slice].cpu().numpy()
        original_shape = xyz.shape
        xyz_reshaped = xyz.reshape(-1, 3)
        rotated_xyz = xyz_reshaped @ rot_mat.T
        rotated_xyz_tensor = torch.tensor(rotated_xyz.reshape(original_shape), device=tensor.device, dtype=tensor.dtype)

        new_tensor = tensor.clone()
        new_tensor[..., xyz_slice] = rotated_xyz_tensor
        return new_tensor

    # Rotate simple features like x_0 and v_0
    for key in ["x_0", "v_0"]:
        if key in rotated_data:
            xyz_slice = FEATURE_CONFIG[key]["xyz"]
            rotated_data[key] = _rotate_slice(rotated_data[key], rotation_matrix, xyz_slice)

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
            rotated_xyz_tensor = torch.tensor(rotated_xyz.reshape(original_shape), device=tensor.device, dtype=tensor.dtype)
            new_tensor[..., xyz_slice] = rotated_xyz_tensor

        rotated_data["concatenated_features"] = new_tensor

    return rotated_data


def test_rotation_equivariance(setup_data: tuple[Config, torch.nn.Module, TensorDict]) -> None:
    """Tests the model's equivariance to 3D rotations."""
    _, model, data_sample = setup_data

    # 1. Get model output for the original input
    with torch.no_grad():
        original_output = model(data_sample)

    # 2. Generate a random rotation matrix
    random_rotation: npt.NDArray[np.float64] = Rotation.random().as_matrix()

    # 3. Apply rotation to the input data
    rotated_data_sample = apply_rotation(data_sample, random_rotation)

    # 4. Get model output for the rotated input
    with torch.no_grad():
        rotated_output = model(rotated_data_sample)

    # 5. Compare outputs
    output_spec = FEATURE_CONFIG["output"]
    xyz_slice = output_spec["xyz"]

    original_xyz = original_output[..., xyz_slice].cpu().numpy()
    rotated_xyz = rotated_output[..., xyz_slice].cpu().numpy()

    # Reshape for easier processing
    batch_size, timesteps, nodes, _ = original_xyz.shape
    original_xyz_reshaped = original_xyz.reshape(-1, 3)

    # Apply the same rotation to the original output
    expected_rotated_xyz = original_xyz_reshaped @ random_rotation.T
    expected_rotated_xyz = expected_rotated_xyz.reshape(batch_size, timesteps, nodes, 3)

    # Compare the expected rotated output with the actual rotated output
    assert np.allclose(rotated_xyz, expected_rotated_xyz, atol=1e-6), "Output tensor did not rotate as expected. The model is not equivariant to rotations."

    # Check if the invariant part (if any) remains unchanged
    invariant_slice = output_spec.get("invariant")
    if invariant_slice and original_output.shape[-1] > xyz_slice.stop:
        original_invariant = original_output[..., invariant_slice].cpu().numpy()
        rotated_invariant = rotated_output[..., invariant_slice].cpu().numpy()
        assert np.allclose(original_invariant, rotated_invariant, atol=1e-6), "Invariant part of the output changed after rotation."

    print("Equivariance test passed. The model is equivariant to 3D rotations.")


def test_e3nn_linear_equivariance() -> None:
    """Tests that a simple E3NN linear layer is equivariant to rotations."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 2. Generate a random rotation matrix
    random_rotation: npt.NDArray[np.float64] = Rotation.random().as_matrix()

    # 3. Apply rotation to the input
    # Reshape for easier processing
    input_reshaped = input_tensor.reshape(-1, 3).cpu().numpy()
    rotated_input_np = input_reshaped @ random_rotation.T
    rotated_input = torch.tensor(rotated_input_np.reshape(batch_size, timesteps, nodes, 3), device=device, dtype=input_tensor.dtype)

    # 4. Get output for the rotated input
    with torch.no_grad():
        rotated_output = e3nn_linear(rotated_input)

    # 5. Compare outputs
    # For an E3NN linear layer, the output should rotate with the input

    # Extract the xyz coordinates from the output tensors
    original_xyz = original_output.cpu().numpy()
    rotated_xyz = rotated_output.cpu().numpy()

    # Reshape for easier processing
    original_xyz_reshaped = original_xyz.reshape(-1, 3)

    # Apply the same rotation to the original output
    expected_rotated_xyz = original_xyz_reshaped @ random_rotation.T
    expected_rotated_xyz = expected_rotated_xyz.reshape(batch_size, timesteps, nodes, 3)

    # Compare the expected rotated output with the actual rotated output
    assert np.allclose(rotated_xyz, expected_rotated_xyz, atol=1e-6), "E3NN linear layer output did not rotate as expected. The layer is not equivariant to rotations."

    print("E3NN linear layer test passed. The layer is equivariant to 3D rotations.")


if __name__ == "__main__":
    # When running this file directly, use pytest to run the test
    args = parse_args()
    if args.test_e3nn:
        # Run only the E3NN test
        sys.exit(pytest.main([__file__, "-v", "-k", "test_e3nn_linear_equivariance"]))
    else:
        # Run the model test
        sys.exit(pytest.main([__file__, "-v"]))
