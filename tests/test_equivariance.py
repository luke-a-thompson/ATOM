import torch
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
import pytest
from pathlib import Path
import argparse
import sys
from tensordict import TensorDict
from e3nn import o3

from gtno_py.training import Config, initialize_model, create_dataloaders_single, create_dataloaders_multitask


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model equivariance to 3D rotations")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--model", type=str, help="Path to the model checkpoint")
    parser.add_argument("--test_e3nn", action="store_true", help="Test with a simple E3NN linear layer")
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
    _ = model.load_state_dict(model_state_dict)
    _ = model.eval()

    # Get a single sample and convert to TensorDict
    data_sample = next(iter(test_loader))
    # Ensure all tensors are on the same device
    data_sample = {k: v.to(config.training.device) for k, v in data_sample.items()}
    data_sample = TensorDict(data_sample, batch_size=data_sample["x_0"].shape[0])

    return config, model, data_sample


def apply_rotation(data: TensorDict, rotation_matrix: npt.NDArray[np.float64]) -> TensorDict:
    """Applies a rotation matrix to the spatial components of the data dictionary."""
    # Create a new TensorDict with the same structure
    rotated_data = data.clone()

    # Rotate x_0 (positions) - shape [batch, timesteps, nodes, xyz_dim]
    if "x_0" in rotated_data:
        x_0 = rotated_data["x_0"]
        # Extract the xyz coordinates (first 3 dimensions)
        xyz = x_0[..., :3].cpu().numpy()
        # Apply rotation to each node's coordinates
        # Reshape to [batch*timesteps*nodes, 3] for easier rotation
        batch_size, timesteps, nodes, _ = xyz.shape
        xyz_reshaped = xyz.reshape(-1, 3)
        rotated_xyz = xyz_reshaped @ rotation_matrix.T
        # Reshape back to original shape
        rotated_xyz = rotated_xyz.reshape(batch_size, timesteps, nodes, 3)
        # Reconstruct x_0 with rotated coordinates and original norm
        rotated_data["x_0"] = torch.cat([torch.tensor(rotated_xyz, device=x_0.device, dtype=x_0.dtype), x_0[..., 3:]], dim=-1)  # Keep the norm part unchanged

    # Rotate v_0 (velocities) if present - shape [batch, timesteps, nodes, xyz_dim]
    if "v_0" in rotated_data:
        v_0 = rotated_data["v_0"]
        # Extract the xyz coordinates (first 3 dimensions)
        xyz = v_0[..., :3].cpu().numpy()
        # Apply rotation to each node's coordinates
        batch_size, timesteps, nodes, _ = xyz.shape
        xyz_reshaped = xyz.reshape(-1, 3)
        rotated_xyz = xyz_reshaped @ rotation_matrix.T
        # Reshape back to original shape
        rotated_xyz = rotated_xyz.reshape(batch_size, timesteps, nodes, 3)
        # Reconstruct v_0 with rotated coordinates and original norm
        rotated_data["v_0"] = torch.cat([torch.tensor(rotated_xyz, device=v_0.device, dtype=v_0.dtype), v_0[..., 3:]], dim=-1)  # Keep the norm part unchanged

    # Reconstruct concatenated_features if present
    if "concatenated_features" in rotated_data:
        # Extract components from concatenated_features
        # Assuming order: [x_0_xyz, x_0_norm, v_0_xyz, v_0_norm, Z]
        features = rotated_data["concatenated_features"]
        x_0_xyz = features[..., :3]
        x_0_norm = features[..., 3:4]
        v_0_xyz = features[..., 4:7]
        v_0_norm = features[..., 7:8]
        Z = features[..., 8:]

        # Rotate x_0_xyz and v_0_xyz
        x_0_xyz_np = x_0_xyz.cpu().numpy()
        v_0_xyz_np = v_0_xyz.cpu().numpy()

        # Reshape for rotation
        batch_size, timesteps, nodes, _ = x_0_xyz_np.shape
        x_0_xyz_reshaped = x_0_xyz_np.reshape(-1, 3)
        v_0_xyz_reshaped = v_0_xyz_np.reshape(-1, 3)

        rotated_x_0_xyz = x_0_xyz_reshaped @ rotation_matrix.T
        rotated_v_0_xyz = v_0_xyz_reshaped @ rotation_matrix.T

        # Reshape back
        rotated_x_0_xyz = rotated_x_0_xyz.reshape(batch_size, timesteps, nodes, 3)
        rotated_v_0_xyz = rotated_v_0_xyz.reshape(batch_size, timesteps, nodes, 3)

        # Reconstruct concatenated_features
        rotated_data["concatenated_features"] = torch.cat(
            [
                torch.tensor(rotated_x_0_xyz, device=features.device, dtype=features.dtype),
                x_0_norm,
                torch.tensor(rotated_v_0_xyz, device=features.device, dtype=features.dtype),
                v_0_norm,
                Z,
            ],
            dim=-1,
        )

    return rotated_data


def test_rotation_equivariance(setup_data: tuple[Config, torch.nn.Module, TensorDict]) -> None:
    """Tests the model's equivariance to 3D rotations."""
    config, model, data_sample = setup_data
    device = config.training.device

    # 1. Get model output for the original input
    with torch.no_grad():
        original_output = model(data_sample)  # Shape: [batch, timesteps, nodes, xyz_dim]

    # 2. Generate a random rotation matrix
    random_rotation = R.random().as_matrix()

    # 3. Apply rotation to the input data
    rotated_data_sample = apply_rotation(data_sample, random_rotation)

    # 4. Get model output for the rotated input
    with torch.no_grad():
        rotated_output = model(rotated_data_sample)  # Shape: [batch, timesteps, nodes, xyz_dim]

    # 5. Compare outputs
    # For a model that is equivariant to rotations, the output should rotate with the input
    # So if we rotate the input by R, the output should also rotate by R

    # Extract the xyz coordinates from the output tensors
    original_xyz = original_output[..., :3].cpu().numpy()  # Shape: [batch, timesteps, nodes, 3]
    rotated_xyz = rotated_output[..., :3].cpu().numpy()  # Shape: [batch, timesteps, nodes, 3]

    # Reshape for easier processing
    batch_size, timesteps, nodes, _ = original_xyz.shape
    original_xyz_reshaped = original_xyz.reshape(-1, 3)  # Shape: [batch*timesteps*nodes, 3]

    # Apply the same rotation to the original output
    expected_rotated_xyz = original_xyz_reshaped @ random_rotation.T  # Shape: [batch*timesteps*nodes, 3]
    expected_rotated_xyz = expected_rotated_xyz.reshape(batch_size, timesteps, nodes, 3)

    # Compare the expected rotated output with the actual rotated output
    assert np.allclose(rotated_xyz, expected_rotated_xyz, atol=1e-6), "Output tensor did not rotate as expected. The model is not equivariant to rotations."

    # Check if the norm part (if any) remains unchanged
    if original_output.shape[-1] > 3:
        original_norm = original_output[..., 3:].cpu().numpy()
        rotated_norm = rotated_output[..., 3:].cpu().numpy()
        assert np.allclose(original_norm, rotated_norm, atol=1e-6), "Norm part of the output changed after rotation, but should remain invariant."

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
    random_rotation = R.random().as_matrix()

    # 3. Apply rotation to the input
    # Reshape for easier processing
    input_reshaped = input_tensor.reshape(-1, 3).cpu().numpy()
    rotated_input = input_reshaped @ random_rotation.T
    rotated_input = torch.tensor(rotated_input.reshape(batch_size, timesteps, nodes, 3), device=device, dtype=input_tensor.dtype)

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
