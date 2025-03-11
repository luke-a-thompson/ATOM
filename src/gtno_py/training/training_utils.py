import torch.nn as nn
import torch


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def reset_weights(model: nn.Module) -> None:
    """Reinitialize model weights without recompiling."""
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):  # Applies to layers like Linear, Conv, etc.
            layer.reset_parameters()
