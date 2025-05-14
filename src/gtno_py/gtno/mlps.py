import torch
import torch.nn as nn
from e3nn import o3
from typing import final, override


@final
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        hidden_layers: int,
        activation: nn.Module,
        dropout_p: float,
    ) -> None:
        """
        A simple MLP with a specified number of layers and hidden features.

        Parameters:
            in_features: The number of input features.
            hidden_features: The number of features in the hidden layers.
            out_features: The number of output features.
            hidden_layers: The number of hidden layers.
            activation: The activation function to use in the hidden layers.
            dropout_p: The dropout probability.
        num_layers = 1 produces an MLP of the form:
            Linear(in_features, hidden_features) -> activation -> Linear(hidden_features, out_features)
        """
        super().__init__()
        layers = []
        for i in range(hidden_layers):
            in_size = in_dim if i == 0 else hidden_dim
            layers.extend([nn.Linear(in_size, hidden_dim), activation])
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)

        # Zero initialize the final layer - Following NanoGPT
        final_layer = self.layers[-1]
        if isinstance(final_layer, nn.Linear):
            _ = nn.init.zeros_(final_layer.weight)
            _ = nn.init.zeros_(final_layer.bias)

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        if mask is not None:
            x = x * mask  # Ensures fake nodes output zeros
        return x


@final
class EquivariantMLP(nn.Module):
    def __init__(
        self,
        in_irreps: str,
        hidden_irreps: str,
        out_irreps: str,
        hidden_layers: int,
        activation: nn.Module,
        dropout_p: float,
    ) -> None:
        """
        A simple MLP with a specified number of layers and hidden features.

        Parameters:
            in_features: The number of input features.
            hidden_features: The number of features in the hidden layers.
            out_features: The number of output features.
            hidden_layers: The number of hidden layers.
            activation: The activation function to use in the hidden layers.
            dropout_p: The dropout probability.
        num_layers = 1 produces an MLP of the form:
            Linear(in_features, hidden_features) -> activation -> Linear(hidden_features, out_features)
        """
        super().__init__()
        layers = []
        for i in range(hidden_layers):
            in_size = in_irreps if i == 0 else hidden_irreps
            layers.extend([o3.Linear(in_size, hidden_irreps), activation])
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(o3.Linear(hidden_irreps, out_irreps))
        self.layers = nn.ModuleList(layers)

        # Zero initialize the final layer - Following NanoGPT
        final_layer = self.layers[-1]
        if isinstance(final_layer, nn.Linear):
            _ = nn.init.zeros_(final_layer.weight)
            _ = nn.init.zeros_(final_layer.bias)

    @override
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        if mask is not None:
            x = x * mask  # Ensures fake nodes output zeros
        return x
