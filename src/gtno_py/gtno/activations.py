import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import override, final


class FFNActivation(str, Enum):
    RELU = "relu"
    RELU2 = "relu2"
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"


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
