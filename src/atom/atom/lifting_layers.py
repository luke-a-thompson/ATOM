from typing import final, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from atom.atom.activations import ReLU2, SwiGLU
from atom.training.config_options import FFNActivation, NormType, ValueResidualType, AttentionType, EquivariantLiftingType, PositionalEncodingType
from tensordict import TensorDict
from atom.atom.attentions import QuadraticHeterogenousCrossAttention, QuadraticSelfAttention
from atom.atom.mlps import MLP
from e3nn import o3


class StandardLift(nn.Module):
    def __init__(
        self,
        x_0_in_features: int,
        v_0_in_features: int,
        concat_feats_in_features: int,
        lifting_dim: int,
    ) -> None:
        super().__init__()
        self.x_0_linear = nn.Linear(x_0_in_features, lifting_dim)
        self.v_0_linear = nn.Linear(v_0_in_features, lifting_dim)
        self.concat_feats_linear = nn.Linear(concat_feats_in_features, lifting_dim)

    @override
    def forward(self, x_0: torch.Tensor, v_0: torch.Tensor, concatenated_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_0 = self.x_0_linear(x_0)
        v_0 = self.v_0_linear(v_0)
        concatenated_features = self.concat_feats_linear(concatenated_features)
        return x_0, v_0, concatenated_features


class EquivariantLift(nn.Module):
    def __init__(self, x_0_in_irreps: str, v_0_in_irreps: str, concat_feats_in_irreps: str, lifting_dim_irreps: str) -> None:
        super().__init__()
        self.x_0_linear = o3.Linear(x_0_in_irreps, lifting_dim_irreps)
        self.v_0_linear = o3.Linear(v_0_in_irreps, lifting_dim_irreps)
        self.concat_feats_linear = o3.Linear(concat_feats_in_irreps, lifting_dim_irreps)

    @override
    def forward(self, x_0: torch.Tensor, v_0: torch.Tensor, concatenated_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_0 = self.x_0_linear(x_0)
        v_0 = self.v_0_linear(v_0)
        concatenated_features = self.concat_feats_linear(concatenated_features)
        return x_0, v_0, concatenated_features


class EquivariantLiftTensorProduct(nn.Module):
    def __init__(self, x_0_in_irreps: str, v_0_in_irreps: str, concat_feats_in_irreps: str, lifting_dim_irreps: str) -> None:
        super().__init__()
        self.x_0_linear = o3.Linear(x_0_in_irreps, lifting_dim_irreps)
        self.v_0_linear = o3.Linear(v_0_in_irreps, lifting_dim_irreps)

        vz_0_in_irreps: str = v_0_in_irreps + " + 1x0e"  # (vx,vy,vz, ||v||, Z)
        self.vz_0_linear = o3.Linear(vz_0_in_irreps, lifting_dim_irreps)
        self.concat_feats_linear = o3.FullyConnectedTensorProduct(lifting_dim_irreps, lifting_dim_irreps, lifting_dim_irreps)

    @override
    def forward(self, x_0: torch.Tensor, v_0: torch.Tensor, concatenated_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_0 = self.x_0_linear(x_0)
        v_0 = self.v_0_linear(v_0)
        vz_0 = self.vz_0_linear(concatenated_features[..., 4:])
        # assert False, (x_0.shape, v_0.shape, vz_0.shape, concatenated_features.shape)
        concatenated_features = self.concat_feats_linear(x_0, vz_0)
        return x_0, v_0, concatenated_features
