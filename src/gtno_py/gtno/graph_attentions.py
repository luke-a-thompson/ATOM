from typing import final, override
import torch
import torch.nn as nn
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal


@final
class UnifiedInputMHA(nn.Module):
    """
    Implicit message passing via attention using concatenated node features of the graph: x_0 (position), v_0 (velocity), Z (atomic number).

    The attention matrix is of shape: [B, N*T, N*T]

    The idea is to construct a learned unified graph representation accounting for dependencies between these primary node features.
    This may learn some position-velocity dependence across time (i.e., Newton's second law).
    """

    def __init__(self, lifting_dim: int, num_heads: int, num_timesteps: int, batch_first: bool = True) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps
        self.graph_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        concat_over_time = flatten_spatiotemporal(batch["concatenated_features"], self.num_timesteps)

        attn_output: torch.Tensor = self.graph_attention(concat_over_time, concat_over_time, concat_over_time)[0]

        batch["concatenated_features"] = unflatten_spatiotemporal(attn_output, self.num_timesteps)

        return batch


@final
class SplitInputMHA(nn.Module):
    """
    Implict message passing via attention using separate MHA for x_0 (position), v_0 (velocity). We do not include Z (atomic number) as it is a scalar.

    The attention matrices are of shape: [B, N*T, N*T]

    The idea here is to construct learned graph representations for each of the primary node features.
    """

    def __init__(self, lifting_dim: int, num_heads: int, num_timesteps: int, batch_first: bool = True) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps

        self.position_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)
        self.velocity_attention = nn.MultiheadAttention(embed_dim=lifting_dim, num_heads=num_heads, batch_first=batch_first)

    @override
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_over_time = flatten_spatiotemporal(batch["x_0"], self.num_timesteps)
        v_over_time = flatten_spatiotemporal(batch["v_0"], self.num_timesteps)
        assert x_over_time.shape[-1] == v_over_time.shape[-1], f"Positions and velocities must have the same last dimension. Got {x_over_time.shape} and {v_over_time.shape}"

        position_attn_output: torch.Tensor = self.position_attention(x_over_time, x_over_time, x_over_time)[0]
        velocity_attn_output: torch.Tensor = self.velocity_attention(v_over_time, v_over_time, v_over_time)[0]

        batch["x_0"] = unflatten_spatiotemporal(position_attn_output, self.num_timesteps)
        batch["v_0"] = unflatten_spatiotemporal(velocity_attn_output, self.num_timesteps)

        return batch
