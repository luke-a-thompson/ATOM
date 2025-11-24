import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import override, final
from atom.training.config_options import FFNActivation


# ReLU² Activation Function
class ReLU2(nn.Module):
    def __init__(self):
        super(ReLU2, self).__init__()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)


# SwiGLU Activation Function
@final
class SwiGLU(nn.Module):
    def __init__(self, input_dim: int):  # 'dim' is the input dimension
        super().__init__()
        self.linear_gate = nn.Linear(input_dim, input_dim)  # Linear layer for the gate
        self.linear_value = nn.Linear(input_dim, input_dim)  # Linear layer for the value

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate: torch.Tensor = self.linear_gate(x)
        value: torch.Tensor = self.linear_value(x)
        return F.silu(gate) * value  # Element-wise product


def get_activation(activation: FFNActivation, input_dim: int | None = None) -> nn.Module:
    if activation == FFNActivation.SWIGLU and input_dim is None:
        raise ValueError("input_dim must be provided for SwiGLU activation")

    activation_fn: nn.Module
    match activation:
        case FFNActivation.RELU:
            activation_fn = nn.ReLU()
        case FFNActivation.LEAKY_RELU:
            activation_fn = nn.LeakyReLU()
        case FFNActivation.RELU2:
            activation_fn = ReLU2()
        case FFNActivation.GELU:
            activation_fn = nn.GELU()
        case FFNActivation.SILU:
            activation_fn = nn.SiLU()
        case FFNActivation.SWIGLU:
            assert input_dim is not None
            activation_fn = SwiGLU(input_dim=input_dim)
        case _:
            raise ValueError(f"Invalid activation function: {activation}, select from one of {FFNActivation.__members__.keys()}")

    return activation_fn
