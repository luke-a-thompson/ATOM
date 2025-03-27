import torch
from torch import nn
import math
from gtno_py.training.config_options import FFNActivation
from gtno_py.egno.layers import TimeConvMode
from gtno_py.egno.layers import EGNN
from gtno_py.egno.layers import TimeConv
from typing import assert_type, override, final
from gtno_py.gtno.activations import get_activation
from tensordict import TensorDict


class EGNO(nn.Module):
    def __init__(
        self,
        num_layers: int,
        lifting_dim: int,
        activation: FFNActivation,
        use_time_conv: bool,
        num_fourier_modes: int,
        time_embed_dim: int,
        num_timesteps: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_fourier_modes = num_fourier_modes
        self.time_embed_dim = time_embed_dim
        self.use_time_conv = use_time_conv
        self.lifting_dim = lifting_dim

        in_dim = 2 + time_embed_dim

        self.egnn = EGNN(
            in_dim=in_dim,
            lifting_dim=lifting_dim,
            num_layers=num_layers,
            activation=get_activation(activation, lifting_dim),
            with_v=True,
            flat=False,
            norm=False,
        )

        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()
            self.time_conv_x_modules = nn.ModuleList()
            for _ in range(num_layers):
                _ = self.time_conv_modules.append(
                    TimeConv(
                        lifting_dim,
                        lifting_dim,
                        num_fourier_modes,
                        TimeConvMode.TIME_CONV,
                        num_timesteps,
                    )
                )
                _ = self.time_conv_x_modules.append(
                    TimeConv(
                        2,
                        2,
                        num_fourier_modes,
                        TimeConvMode.TIME_CONV_X,
                        num_timesteps,
                    )
                )

    @override
    def forward(
        self,
        batch: TensorDict,
    ):
        B, T, N, D = batch["x_0"].shape
        E = batch["source_node_indices"].shape[1]

        time_emb = self._timestep_embedding(T, self.time_embed_dim).to(batch["x_0"].device)
        time_emb = time_emb.unsqueeze(1).repeat(1, B * N, 1).view(B * T * N, -1)  # [B*T*N, H_t]
        # X = position, h = node features (||x||, Z)

        loc_mean: torch.Tensor = batch["x_0"][..., :3].mean(dim=2, keepdim=True).repeat(1, 1, N, 1)  # [B, T, N, 3]
        loc_mean = loc_mean.view(B * T * N, -1)  # [B*T*N, 3]

        # Get x, v to shape [B*T*N, 3]
        x: torch.Tensor = batch["x_0"][..., :3].view(B * T * N, -1)
        v: torch.Tensor = batch["v_0"][..., :3].view(B * T * N, -1)
        h = batch["concatenated_features"][..., -2:]  # ||v||, Z
        h = h.view(B * T * N, -1)
        h = torch.cat((h, time_emb), dim=-1)  # [B * T * N, H]
        h: torch.Tensor = self.egnn.embedding(h)

        # Handle distances
        loc = batch["x_0"][:, 0, :, :3].reshape(B * N, 3)  # Ignoring norm for distances
        # Pre-duplicated edge indices already have shape [B*T*E]
        rows = batch["source_node_indices"].reshape(B * E)
        cols = batch["target_node_indices"].reshape(B * E)

        # Compute squared distances for each edge
        loc_dist = torch.sum((loc[rows.to(torch.long)] - loc[cols.to(torch.long)]) ** 2, dim=-1, keepdim=True)  # [B*E, 1]
        loc_dist = loc_dist.repeat(T, 1)  # [T*B*E, 1]

        time_offsets = (torch.arange(T, device=batch["x_0"].device) * N).repeat_interleave(B * E)
        edge_index = (
            batch["source_node_indices"].reshape(B * E).repeat(T) + time_offsets,
            batch["target_node_indices"].reshape(B * E).repeat(T) + time_offsets,
        )
        edge_attr = batch["edge_attr"].reshape(B * E, -1).repeat(T, 1)  # [T*B*E, feat_dim]
        edge_attr = torch.cat((edge_attr, loc_dist), dim=-1)  # [T*B*E, feat_dim + 1]

        for i in range(self.num_layers):
            if self.use_time_conv:
                # To the shape for FFT and back
                h = h.view(T, B * N, self.lifting_dim)
                h = self.time_conv_modules[i](h)
                h = h.view(B * T * N, self.lifting_dim)

                x = x - loc_mean  # Shape [B*T*N, 3] matches
                x = torch.stack((x, v), dim=-1)  # Shape [B*T*N, 3, 2] matches
                x = x.reshape(T, B * N, 3, 2)  # Shape [T, B*N, 3, 2] matches
                temp = self.time_conv_x_modules[i](x)  # Shape [T, B*N, 3, 2] matches
                x = temp[..., 0].view(B * T * N, 3) + loc_mean  # Shape [B*T*N, 3] matches
                v = temp[..., 1].view(B * T * N, 3)  # Shape [B*T*N, 3] matches

            loc_pred, vel_pred, h = self.egnn.layers[i](x, h, edge_index, edge_attr, v)

        if v is not None:
            return loc_pred.reshape(B, T, N, 3)
        else:
            return loc_pred.reshape(B, T, N, 3), h.reshape(B, T, N, self.lifting_dim)

    def _timestep_embedding(self, num_timesteps: int, lifting_dim: int, max_positions: int = 10_000) -> torch.Tensor:
        half_dim = lifting_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb_tensor = torch.exp(torch.arange(half_dim) * -emb)

        timesteps: torch.Tensor = torch.arange(num_timesteps)
        emb_tensor = timesteps.float()[:, None] * emb_tensor[None, :]
        emb_tensor = torch.cat((emb_tensor.sin(), emb_tensor.cos()), dim=-1)
        assert emb_tensor.shape == (num_timesteps, lifting_dim)
        return emb_tensor
