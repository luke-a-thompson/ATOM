import inspect
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

def get_context(instance: object) -> str:
    """
    Returns the class name and method name of the caller context.

    Parameters:
        instance: The instance of the class (typically `self`).

    Returns:
        str: The class name and method name in the format `ClassName.MethodName`.
    """
    frame = inspect.currentframe().f_back  # Access the caller's frame
    if frame is None:
        raise RuntimeError("Unable to determine the calling context.")

    class_name = instance.__class__.__name__
    method_name = frame.f_code.co_name
    return f"{class_name}.{method_name}"


def log_weights(named_parameters: list[tuple[str, torch.Tensor]], epoch: int, save_dir: Path):
    """Log feature weights to wandb and/or save as numpy arrays.

    Args:
        named_parameters: Model's named parameters
        epoch: Current training epoch
        save_dir: Directory to save feature weights as npz files
    """
    feat_weights_per_layer = []
    attention_denom_per_layer = []
    lambda_v_residual_per_layer = []

    for name, param in named_parameters:
        if "feature_weights" in name:
            feat_weights_per_layer.append(param.detach().cpu())
        elif "attention_denom" in name:
            attention_denom_per_layer.append(param.detach().cpu())
        elif "lambda_v_residual" in name:
            lambda_v_residual_per_layer.append(param.detach().cpu())

    # Save to npz files if directory is provided, accumulating across epochs
    if save_dir and os.path.exists(save_dir):
        # Load existing data if available
        feat_weights_history = []
        attention_denom_history = []
        lambda_v_residual_history = []

        # Feature weights
        if feat_weights_per_layer:
            feat_weights_path = f"{save_dir}/feature_weights.npz"
            if os.path.exists(feat_weights_path):
                # Load existing data
                loaded_data = np.load(feat_weights_path, allow_pickle=True)
                feat_weights_history = loaded_data["feature_weights"].tolist()

            # Add current epoch data
            current_feat_weights = torch.stack(feat_weights_per_layer, dim=0).numpy()
            feat_weights_history.append(current_feat_weights)

            # Save updated history
            np.savez(feat_weights_path, feature_weights=np.array(feat_weights_history))

        # Attention denominator
        if attention_denom_per_layer:
            attention_denom_path = f"{save_dir}/attention_denom.npz"
            if os.path.exists(attention_denom_path):
                # Load existing data
                loaded_data = np.load(attention_denom_path, allow_pickle=True)
                attention_denom_history = loaded_data["attention_denom"].tolist()

            # Add current epoch data - this will be a 3D array: [train_step, layer, values]
            current_attention_denom = torch.stack(attention_denom_per_layer, dim=0).numpy()
            attention_denom_history.append(current_attention_denom)

            # Save updated history
            np.savez(attention_denom_path, attention_denom=np.array(attention_denom_history))

        # Lambda v residual
        if lambda_v_residual_per_layer:
            lambda_v_path = f"{save_dir}/lambda_v_residual.npz"
            if os.path.exists(lambda_v_path):
                # Load existing data
                loaded_data = np.load(lambda_v_path, allow_pickle=True)
                lambda_v_residual_history = loaded_data["lambda_v_residual"].tolist()

            # Add current epoch data
            current_lambda_v = torch.stack(lambda_v_residual_per_layer, dim=0).numpy()
            lambda_v_residual_history.append(current_lambda_v)

            # Save updated history
            np.savez(lambda_v_path, lambda_v_residual=np.array(lambda_v_residual_history))

    # Original wandb logging code
    if feat_weights_per_layer:
        averaged_param = torch.stack(feat_weights_per_layer, dim=0).mean(dim=0)
        softmaxed_param = F.softmax(averaged_param, dim=0)
        bin_labels = ["x_0", "v_0", "concat"]
        values = softmaxed_param.tolist()

        fig, ax = plt.subplots()
        ax.bar(bin_labels, values)
        ax.set_title("Averaged Feature Weights")
        wandb.log({"feature_weights/averaged": wandb.Image(fig)}, step=epoch)
        plt.close(fig)
    if attention_denom_per_layer:
        averaged_param = torch.stack(attention_denom_per_layer, dim=0).mean(dim=0)
        wandb.log({"attention_denom/averaged": wandb.Histogram(averaged_param.tolist())}, step=epoch)

        # Create a table of attention denominator values
        attention_table = wandb.Table(columns=["Layer", "Value"])
        for i, denom in enumerate(attention_denom_per_layer):
            attention_table.add_data(i, denom[0].item())
        wandb.log({"attention_denom/table": attention_table}, step=epoch)
    if lambda_v_residual_per_layer:
        averaged_param = torch.stack(lambda_v_residual_per_layer, dim=0).mean(dim=0)
        wandb.log({"lambda_v_residual/averaged": wandb.Histogram(averaged_param.tolist())}, step=epoch)


def add_brownian_noise(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    concat: torch.Tensor,
    noise_std: float = 0.20,
):
    """
    Add Langevin-type (Brownian) noise to velocities (and optionally positions)
    for molecular dynamics data using PyTorch.

    Args:
        positions (torch.Tensor): Tensor of shape [Batch, Timesteps, Atoms, Dim].
        velocities (torch.Tensor): Tensor of shape [Batch, Timesteps, Atoms, Dim].
        noise_std (float): Std dev of Gaussian noise for velocities and concatenated features.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Noised positions and velocities.
    """
    # Add noise to velocities
    noise_vel = torch.randn_like(velocities) * noise_std
    noisy_velocities = velocities + noise_vel

    noise_concat_feats = torch.randn_like(concat) * noise_std

    # Zero out the last entry along the last dimension
    noise_concat_feats[..., -1] = 0

    # Apply noise
    noisy_concat_feats = concat + noise_concat_feats

    # Optionally add noise to positions
    noise_pos = torch.randn_like(positions) * noise_std
    noisy_positions = positions + noise_pos

    return noisy_positions, noisy_velocities, noisy_concat_feats
