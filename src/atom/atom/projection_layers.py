from typing import final, override

import torch
import torch.nn as nn
from e3nn import o3


@final
class EquivariantProjectPosOnly(nn.Module):
    def __init__(self, lifting_dim_irreps: str, out_irreps: str) -> None:
        super().__init__()
        self.linear_pos = o3.Linear(lifting_dim_irreps, out_irreps)

    @override
    def forward(self, lifted_x_0: torch.Tensor, lifted_concat_features: torch.Tensor) -> torch.Tensor:
        _ = lifted_concat_features
        pos = self.linear_pos(lifted_x_0)
        return pos


@final
class EquivariantProjectFull(nn.Module):
    def __init__(self, lifting_dim_irreps: str, out_irreps: str) -> None:
        super().__init__()
        self.linear_pos = o3.Linear(lifting_dim_irreps, out_irreps)
        self.linear_vel = o3.Linear(lifting_dim_irreps, out_irreps)
        self.linear_energy = o3.Linear(lifting_dim_irreps, "1x0e")

    @override
    def forward(self, lifted_x_0: torch.Tensor, lifted_concat_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = lifted_concat_features
        pos = self.linear_pos(lifted_x_0)
        vel = self.linear_vel(lifted_x_0)
        energy_per_node = self.linear_energy(lifted_x_0)
        return pos, vel, energy_per_node


# MoE projection removed per request


@final
class DecanonicalizationProject(nn.Module):
    def __init__(self, lifting_dim_irreps: str, out_irreps: str) -> None:
        super().__init__()
        self.linear_pos = o3.Linear(lifting_dim_irreps, out_irreps)
        self.linear_vel = o3.Linear(lifting_dim_irreps, out_irreps)
        self.linear_energy = o3.Linear(lifting_dim_irreps, "1x0e")

    @override
    def forward(
        self,
        lifted_x_0: torch.Tensor,
        lifted_concat_features: torch.Tensor,
        so3_matrix: torch.Tensor,
        x_0_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = lifted_concat_features
        pos_canonical = self.linear_pos(lifted_x_0)
        vel_canonical = self.linear_vel(lifted_x_0)
        energy_per_node = self.linear_energy(lifted_x_0)
        pos_world = pos_canonical @ so3_matrix.transpose(-2, -1) + x_0_mean
        vel_world = vel_canonical @ so3_matrix.transpose(-2, -1)
        return pos_world, vel_world, energy_per_node


@final
class DecanonicalizationProjectPosOnly(nn.Module):
    def __init__(self, lifting_dim_irreps: str, out_irreps: str) -> None:
        super().__init__()
        self.linear_pos = o3.Linear(lifting_dim_irreps, out_irreps)

    @override
    def forward(
        self,
        lifted_x_0: torch.Tensor,
        lifted_concat_features: torch.Tensor,
        so3_matrix: torch.Tensor,
        x_0_mean: torch.Tensor,
    ) -> torch.Tensor:
        _ = lifted_concat_features
        pos_canonical = self.linear_pos(lifted_x_0)
        pos_world = pos_canonical @ so3_matrix.transpose(-2, -1) + x_0_mean
        return pos_world
