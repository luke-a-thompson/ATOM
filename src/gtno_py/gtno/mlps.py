import torch
import torch.nn as nn
from typing import final, override
from e3nn import o3


class E3NNLifting(nn.Module):
    def __init__(self):
        super().__init__()
        # Map 4 input dims => 128 output dims (with mostly l=1 and a couple of l=0)
        self.lift = o3.Linear(o3.Irreps("1x1o + 1x0e"), o3.Irreps("42x1o + 2x0e"))  # in: (x,y,z) + scalar  # out: 126 + 2 = 128

    def forward(self, x):
        # x shape is [..., 4], returning [..., 128]
        return self.lift(x)


class E3NNLiftingTensorProduct(nn.Module):
    """
    A fully-connected tensor product block that takes in irreps_in,
    multiplies by a dummy scalar (0e) so it behaves like a learnable linear map,
    and produces irreps_out.

    Example usage:
      # (x,y,z, ||x||) => (1x1o + 1x0e) in -> (42x1o + 2x0e) out => 128 dims
      equivariant_lifter = E3NNLiftingTensorProduct(
          in_irreps="1x1o + 1x0e",
          out_irreps="42x1o + 2x0e"
      )
    """

    def __init__(self, in_irreps: str, out_irreps: str, shared_weights: bool = True, internal_weights: bool = True):
        super().__init__()

        # Convert string to Irreps objects
        self.irreps_in = o3.Irreps(in_irreps)
        self.irreps_out = o3.Irreps(out_irreps)

        # We'll multiply x by a dummy scalar (0e)
        scalar_irreps = o3.Irreps("0e")

        # The FullyConnectedTensorProduct takes two inputs:
        #   1) 'in_irreps'
        #   2) 'scalar_irreps'
        # and produces 'out_irreps'
        self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, scalar_irreps, self.irreps_out, shared_weights=shared_weights, internal_weights=internal_weights)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: a tensor whose trailing dimension matches irreps_in.dim
           (e.g., 4D or 9D).
        Returns a tensor with trailing dimension = irreps_out.dim
        """
        # Create dummy scalar (1.0) for each example in x
        # shape [..., 1]
        s = torch.ones_like(x[..., :1])

        # out shape => [..., irreps_out.dim]
        out = self.tp(x, s)
        return out


@final
class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, hidden_layers: int, activation: nn.Module) -> None:
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
            layers.extend([nn.Linear(in_size, hidden_features), activation])
        layers.append(nn.Linear(hidden_features, out_features))
        self.layers = nn.ModuleList(layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
