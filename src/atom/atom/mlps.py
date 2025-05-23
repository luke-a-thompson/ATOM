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
        """A simple Multi-Layer Perceptron (MLP).

        Parameters
        ----------
        in_dim : int
            Number of input features.
        hidden_dim : int
            Number of features in the hidden layers.
        out_dim : int
            Number of output features.
        hidden_layers : int
            Number of hidden layers.
        activation : nn.Module
            Activation function to use in the hidden layers.
        dropout_p : float
            Dropout probability.

        Notes
        -----
        If `hidden_layers` is 1, the MLP structure is:
        Linear(in_dim, hidden_dim) -> activation -> Dropout (if >0) -> Linear(hidden_dim, out_dim)
        The final linear layer's weights and biases are zero-initialized.
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
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mask : torch.Tensor | None, optional
            Optional mask to apply to the output, by default None.
            If provided, the output is element-wise multiplied by the mask.

        Returns
        -------
        torch.Tensor
            Output tensor of the MLP.
        """
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
        """An equivariant Multi-Layer Perceptron (MLP) using e3nn.

        Parameters
        ----------
        in_irreps : str
            Irreducible representations (irreps) of the input features.
        hidden_irreps : str
            Irreps of the hidden layers.
        out_irreps : str
            Irreps of the output features.
        hidden_layers : int
            Number of hidden layers.
        activation : nn.Module
            Activation function to use in the hidden layers (e.g., o3.NormActivation).
        dropout_p : float
            Dropout probability.

        Notes
        -----
        If `hidden_layers` is 1, the MLP structure is:
        o3.Linear(in_irreps, hidden_irreps) -> activation -> Dropout (if >0) -> o3.Linear(hidden_irreps, out_irreps)
        The final linear layer's weights and biases are zero-initialized if it's an `nn.Linear` (for scalar outputs).
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
        """Forward pass of the Equivariant MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with features corresponding to `in_irreps`.
        mask : torch.Tensor | None, optional
            Optional mask to apply to the output, by default None.
            If provided, the output is element-wise multiplied by the mask.

        Returns
        -------
        torch.Tensor
            Output tensor of the Equivariant MLP.
        """
        for layer in self.layers:
            x = layer(x)

        if mask is not None:
            x = x * mask  # Ensures fake nodes output zeros
        return x
