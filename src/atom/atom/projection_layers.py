from typing import final, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from atom.atom.activations import SwiGLU
from atom.atom.mlps import MLP
from e3nn import o3


@final
class EquivariantProject(nn.Module):
    def __init__(self, lifting_dim_irreps: str, out_irreps: str) -> None:
        super().__init__()
        self.linear = o3.Linear(lifting_dim_irreps, out_irreps)

    @override
    def forward(self, lifted_x_0: torch.Tensor, lifted_concat_features: torch.Tensor) -> torch.Tensor:
        _ = lifted_concat_features
        return self.linear(lifted_x_0)


@final
class EquivariantMoEProject(nn.Module):
    def __init__(self, lifting_dim: int, out_irreps: str, num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([o3.Linear(lifting_dim, out_irreps) for _ in range(num_experts)])
        self.gate_net = MLP(
            in_dim=lifting_dim,
            hidden_dim=lifting_dim // 8,
            out_dim=num_experts,
            hidden_layers=2,
            activation=SwiGLU(lifting_dim // 4),
            dropout_p=0.1,
        )

    @override
    def forward(self, x: torch.Tensor, lifted_concat_features: torch.Tensor) -> torch.Tensor:
        gate_logits: torch.Tensor = self.gate_net(lifted_concat_features.mean(dim=(1, 2)))
        if self.training:
            routing_mask = F.gumbel_softmax(gate_logits, tau=1.0, hard=True, dim=-1)
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
            routing_mask = routing_mask.view(*routing_mask.shape, 1, 1, 1)
            final_pred_pos = (expert_outputs * routing_mask).sum(dim=1)
        else:
            top_expert_idx = torch.argmax(gate_logits, dim=-1)
            final_pred_pos = torch.zeros((*x.shape[:-1], 3), device=x.device, dtype=x.dtype)

            for i, expert in enumerate(self.experts):
                mask = top_expert_idx == i
                if not torch.any(mask):
                    continue
                expert_out = expert(x[mask])
                final_pred_pos[mask] = expert_out

        return final_pred_pos
