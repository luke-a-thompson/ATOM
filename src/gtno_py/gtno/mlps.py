import torch
import torch.nn as nn
from typing import final, override


@final
class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, hidden_layers: int, activation: nn.Module, dropout_p: float = 0.0) -> None:
        """
        A simple MLP with a specified number of layers and hidden features.

        Parameters:
            in_features: The number of input features.
            out_features: The number of output features.
            hidden_features: The number of features in the hidden layers.
            hidden_layers: The number of hidden layers.
            activation: The activation function to use in the hidden layers.
            dropout_p: The dropout probability.
        num_layers = 1 produces an MLP of the form:
            Linear(in_features, hidden_features) -> activation -> Linear(hidden_features, out_features)
        """
        super().__init__()
        layers = []
        for i in range(hidden_layers):
            in_size = in_features if i == 0 else hidden_features
            layers.extend([nn.Linear(in_size, hidden_features), activation, nn.Dropout(dropout_p)])
        layers.append(nn.Linear(hidden_features, out_features))
        self.layers = nn.ModuleList(layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
