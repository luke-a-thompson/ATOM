import torch
from torch import nn
import math
from atom.training.config_options import FFNActivation
from atom.egno.layers import TimeConvMode
from atom.egno.layers import EGNN
from atom.egno.layers import TimeConv
from typing import override
from atom.atom.activations import get_activation
from tensordict import TensorDict


class EGNO(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_layers: int,
        lifting_dim: int,
        activation: FFNActivation,
        use_time_conv: bool,
        num_fourier_modes: int,
        time_embed_dim: int,
        num_timesteps: int,
    ) -> None:
        super().__init__()
        self.num_layers: int = num_layers
        self.num_fourier_modes: int = num_fourier_modes
        self.time_embed_dim: int = time_embed_dim
        self.use_time_conv: bool = use_time_conv
        self.lifting_dim: int = lifting_dim

        self.in_dim: int = num_node_features + time_embed_dim

        self.egnn: EGNN = EGNN(
            in_dim=self.in_dim,
            num_edge_features=num_edge_features,
            lifting_dim=lifting_dim,
            num_layers=num_layers,
            activation=get_activation(activation, lifting_dim),
            with_v=True,
            flat=False,
            norm=False,
        )

        if use_time_conv:
            self.time_conv_modules: nn.ModuleList = nn.ModuleList()
            self.time_conv_x_modules: nn.ModuleList = nn.ModuleList()
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
    ) -> torch.Tensor:
        B, T, N, _ = batch["x_0"].shape
        E = batch["source_node_indices"].shape[1]

        time_emb = self._timestep_embedding(T, self.time_embed_dim).to(batch["x_0"].device)
        time_emb = time_emb.unsqueeze(1).repeat(1, B * N, 1).view(B * T * N, -1)  # [B*T*N, H_t]
        # X = position, h = node features (||x||, Z)

        initial_loc = batch["x_0"][:, 0, :, :3]  # Use time step 0 [B, N, 3]
        loc_mean_per_batch = initial_loc.mean(dim=1, keepdim=True)  # [B, 1, 3]
        loc_mean_repeated_nodes = loc_mean_per_batch.repeat(1, N, 1)  # [B, N, 3]
        loc_mean = loc_mean_repeated_nodes.unsqueeze(1).repeat(1, T, 1, 1).view(B * T * N, 3)  # [B*T*N, 3]

        # Get x, v to shape [B*T*N, 3]
        x: torch.Tensor = batch["x_0"][..., :3].reshape(B * T * N, -1)
        v: torch.Tensor = batch["v_0"][..., :3].reshape(B * T * N, -1)

        # Build scalar node features h
        if "nodes" in batch:
            # N-Body style datasets provide precomputed scalar node features
            h = batch["nodes"]  # [B, T, N, K]
            h = h.view(B * T * N, -1)
        elif "concatenated_features" in batch:
            # MD17-style: concatenated features layout = [x_xyz(3), x_norm(1), v_xyz(3), v_norm(1), Z(1), rrwp(L)...]
            cf = batch["concatenated_features"]
            x_norm = cf[..., 3:4]
            Z = cf[..., 8:9]
            h = torch.cat((x_norm, Z), dim=-1).view(B * T * N, -1)
        else:
            # Fallback to zeros if nothing provided (should not happen in normal configs)
            h = torch.zeros(B * T * N, 2, device=batch["x_0"].device, dtype=batch["x_0"].dtype)

        # Append time embedding then lift to hidden dim
        h = torch.cat((h, time_emb), dim=-1)  # [B*T*N, K + H_t]
        h = self.egnn.embedding(h)

        # Handle distances and build batched, time-expanded edge indices/attributes
        device = batch["x_0"].device

        # Base node positions at t=0 used for static distance feature (kept as in original, but properly batched)
        loc0 = batch["x_0"][:, 0, :, :3]  # [B, N, 3]
        loc0_flat = loc0.reshape(B * N, 3)  # [B*N, 3]

        src = batch["source_node_indices"].to(torch.long)  # [B, E]
        tgt = batch["target_node_indices"].to(torch.long)  # [B, E]

        # Compute per-batch offsets for node indexing at t=0
        batch_node_offsets = (torch.arange(B, device=device) * N).unsqueeze(1)  # [B, 1]
        rows0 = src + batch_node_offsets  # [B, E]
        cols0 = tgt + batch_node_offsets  # [B, E]

        # Squared distances per (batch, edge) at t=0, then expand across time
        loc_dist0 = torch.sum((loc0_flat[rows0] - loc0_flat[cols0]) ** 2, dim=-1, keepdim=True)  # [B, E, 1]
        loc_dist = loc_dist0.unsqueeze(1).expand(B, T, E, 1).reshape(B * T * E, 1)  # [B*T*E, 1]

        # Build edge_index across (batch, time); flatten order matches x/h reshaping
        src_bte = src.unsqueeze(1).expand(B, T, E)  # [B, T, E]
        tgt_bte = tgt.unsqueeze(1).expand(B, T, E)  # [B, T, E]
        bt_index = torch.arange(B * T, device=device).view(B, T)  # [B, T]
        base_nodes = (bt_index * N).unsqueeze(-1)  # [B, T, 1]
        row = (src_bte + base_nodes).reshape(B * T * E)
        col = (tgt_bte + base_nodes).reshape(B * T * E)
        edge_index = (row, col)

        # Expand edge attributes across time and concatenate distance feature
        edge_attr_base = batch["edge_attr"]  # [B, E, F]
        edge_attr = edge_attr_base.unsqueeze(1).expand(B, T, E, edge_attr_base.shape[-1]).reshape(B * T * E, -1)
        edge_attr = torch.cat((edge_attr, loc_dist), dim=-1)  # [B*T*E, F+1]

        # Build edge mask if present (true for valid edges; padded zeros are invalid)
        edge_mask_b: torch.Tensor | None = None
        if "edge_mask" in batch:
            edge_mask_b = batch["edge_mask"].unsqueeze(1).expand(B, T, E).reshape(B * T * E)

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

            x, v, h = self.egnn.layers[i](x, h, edge_index, edge_attr, v, edge_mask_b)

        return x.reshape(B, T, N, 3)

    def _timestep_embedding(self, num_timesteps: int, lifting_dim: int, max_positions: int = 10_000) -> torch.Tensor:
        half_dim = lifting_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb_tensor = torch.exp(torch.arange(half_dim) * -emb)

        timesteps: torch.Tensor = torch.arange(num_timesteps)
        emb_tensor = timesteps.float()[:, None] * emb_tensor[None, :]
        emb_tensor = torch.cat((emb_tensor.sin(), emb_tensor.cos()), dim=-1)
        assert emb_tensor.shape == (num_timesteps, lifting_dim)
        return emb_tensor
