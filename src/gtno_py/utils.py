import inspect
from collections.abc import Iterable
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt


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


def log_feature_weights(named_parameters: Iterable[tuple[str, torch.Tensor]], epoch: int):
    """
    Computes and logs the per-feature average of parameters whose names contain 'feature_weights'.

    This function collects all parameters with 'feature_weights' in their name (and that require gradients),
    stacks them, and computes the element-wise mean across layers. The averaged weights are logged as a wandb Histogram.

    Args:
        named_parameters (Iterable[Tuple[str, torch.Tensor]]): Iterable of (name, parameter) pairs.
        epoch (int): Current epoch for logging.
    """
    weights_per_layer = []
    attention_denom_per_layer = []
    for name, param in named_parameters:
        if "feature_weights" in name and param.requires_grad:
            weights_per_layer.append(param.detach().cpu())
        if "attention_denom" in name and param.requires_grad:
            attention_denom_per_layer.append(param.detach().cpu())

    if weights_per_layer:
        averaged_param = torch.stack(weights_per_layer, dim=0).mean(dim=0)
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


def add_brownian_noise(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    concat: torch.Tensor,
    noise_std_vel: float = 0.20,
    noise_std_pos: float = 0.20,
    noise_std_concat: float = 0.20,
):
    """
    Add Langevin-type (Brownian) noise to velocities (and optionally positions)
    for molecular dynamics data using PyTorch.

    Args:
        positions (torch.Tensor): Tensor of shape [Batch, Timesteps, Atoms, Dim].
        velocities (torch.Tensor): Tensor of shape [Batch, Timesteps, Atoms, Dim].
        noise_std_vel (float): Std dev of Gaussian noise for velocities.
        noise_std_pos (float): Std dev of Gaussian noise for positions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Noised positions and velocities.
    """
    # Add noise to velocities
    noise_vel = torch.randn_like(velocities) * noise_std_vel
    noisy_velocities = velocities + noise_vel

    noise_concat = torch.randn_like(concat) * noise_std_concat

    # Zero out the last entry along the last dimension
    noise_concat[..., -1] = 0

    # Apply noise
    noisy_concat = concat + noise_concat

    # Optionally add noise to positions
    if noise_std_pos > 0:
        noise_pos = torch.randn_like(positions) * noise_std_pos
        noisy_positions = positions + noise_pos
    else:
        noisy_positions = positions.clone()

    return noisy_positions, noisy_velocities, noisy_concat
