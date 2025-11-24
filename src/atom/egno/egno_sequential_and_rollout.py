import math
import torch
from torch import nn
from typing import override
from tensordict import TensorDict

from atom.training.config_options import FFNActivation
from atom.atom.activations import get_activation
from atom.egno.layers import EGNN


class EGNNSequential(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_layers: int,
        lifting_dim: int,
        activation: FFNActivation,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.num_layers: int = num_layers
        self.time_embed_dim: int = time_embed_dim
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

    @override
    def forward(self, batch: TensorDict) -> torch.Tensor:
        B, T, N, _ = batch["x_0"].shape
        device = batch["x_0"].device

        # One frame per layer; require T <= num_layers
        if T > self.num_layers:
            raise ValueError(f"EGNNSequential requires num_layers >= num_timesteps (P). Got num_layers={self.num_layers}, P={T}.")

        # Base scalar node feats from t=0
        if "concatenated_features" in batch:
            cf = batch["concatenated_features"]
            x_norm0 = cf[:, 0, :, 3:4]
            Z = cf[:, 0, :, 8:9]
            base_h0 = torch.cat((x_norm0, Z), dim=-1)  # [B, N, K]
        else:
            x_norm0 = batch["x_0"][:, 0, :, 3:4]
            Z = batch["Z"][:, 0]
            base_h0 = torch.cat((x_norm0, Z), dim=-1)

        x_cur: torch.Tensor = batch["x_0"][:, 0, :, :3]
        v_cur: torch.Tensor = batch["v_0"][:, 0, :, :3]

        src = batch["source_node_indices"].to(torch.long)
        tgt = batch["target_node_indices"].to(torch.long)
        E = src.shape[1]
        edge_attr_base = batch["edge_attr"]
        edge_mask_b: torch.Tensor | None = batch.get("edge_mask", None)

        # Precompute edge index offsets per batch
        batch_node_offsets = (torch.arange(B, device=device) * N).unsqueeze(1)
        row0 = (src + batch_node_offsets).reshape(B * E)
        col0 = (tgt + batch_node_offsets).reshape(B * E)
        edge_index = (row0, col0)

        # Embed initial scalars
        h_emb = self.egnn.embedding(torch.cat((base_h0, torch.zeros(B, N, self.time_embed_dim, device=device)), dim=-1).view(B * N, -1))

        preds: list[torch.Tensor] = []
        for layer_idx in range(T):
            x_flat = x_cur.reshape(B * N, 3)
            v_flat = v_cur.reshape(B * N, 3)

            # Dynamic distance feature from current x
            dist_sq = torch.sum((x_flat[row0] - x_flat[col0]) ** 2, dim=-1, keepdim=True)
            edge_attr_t = torch.cat((edge_attr_base.reshape(B * E, -1), dist_sq), dim=-1)
            edge_mask_t: torch.Tensor | None = None
            if edge_mask_b is not None:
                edge_mask_t = edge_mask_b.reshape(B * E)

            x_flat, v_flat, h_emb = self.egnn.layers[layer_idx](x_flat, h_emb, edge_index, edge_attr_t, v_flat, edge_mask_t)
            x_cur = x_flat.view(B, N, 3)
            v_cur = v_flat.view(B, N, 3)
            preds.append(x_cur)

        return torch.stack(preds, dim=1)

    def _timestep_embedding_vector(self, num_timesteps: int, lifting_dim: int, t: int, max_positions: int = 10_000) -> torch.Tensor:
        half_dim = lifting_dim // 2
        if half_dim == 0:
            return torch.zeros(lifting_dim, dtype=torch.float32)
        emb = math.log(max_positions) / (half_dim - 1) if half_dim > 1 else 0.0
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        val = float(t)
        sin = torch.sin(val * freq)
        cos = torch.cos(val * freq)
        out = torch.cat((sin, cos), dim=-1)
        if out.shape[0] < lifting_dim:
            out = torch.cat(
                (out, torch.zeros(lifting_dim - out.shape[0], dtype=torch.float32)),
                dim=0,
            )
        return out


class EGNNRollout(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_layers: int,
        lifting_dim: int,
        activation: FFNActivation,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.num_layers: int = num_layers
        self.time_embed_dim: int = time_embed_dim
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

    @override
    def forward(self, batch: TensorDict) -> torch.Tensor:
        B, T, N, _ = batch["x_0"].shape
        device = batch["x_0"].device

        # Build initial h from norms and Z; time embedding appended each step
        if "concatenated_features" in batch:
            cf = batch["concatenated_features"]
            x_norm0 = cf[:, 0, :, 3:4]
            Z = cf[:, 0, :, 8:9]
            base_h0 = torch.cat((x_norm0, Z), dim=-1)
        else:
            x_norm0 = batch["x_0"][:, 0, :, 3:4]
            Z = batch["Z"][:, 0]
            base_h0 = torch.cat((x_norm0, Z), dim=-1)

        # Initial positions/velocities
        x_t = batch["x_0"][:, 0, :, :3]  # [B, N, 3]
        v_t = batch["v_0"][:, 0, :, :3]  # [B, N, 3]

        src = batch["source_node_indices"].to(torch.long)  # [B, E]
        tgt = batch["target_node_indices"].to(torch.long)  # [B, E]
        E = src.shape[1]
        edge_attr_base = batch["edge_attr"]  # [B, E, F]
        edge_mask_b: torch.Tensor | None = batch.get("edge_mask", None)

        preds = []
        for t in range(T):
            # time embedding one-hot
            t_emb = self._timestep_embedding(T=self.time_embed_dim, t=t).to(device)
            t_emb = t_emb.unsqueeze(0).expand(B, N, -1)
            h = torch.cat((base_h0, t_emb), dim=-1).reshape(B * N, -1)

            x_flat = x_t.reshape(B * N, 3)
            v_flat = v_t.reshape(B * N, 3)

            # Edge index and attributes using current x_t
            batch_node_offsets = (torch.arange(B, device=device) * N).unsqueeze(1)
            row = (src + batch_node_offsets).reshape(B * E)
            col = (tgt + batch_node_offsets).reshape(B * E)
            edge_index = (row, col)

            loc = x_t.reshape(B * N, 3)
            dist_sq = torch.sum((loc[row] - loc[col]) ** 2, dim=-1, keepdim=True)
            edge_attr_t = torch.cat((edge_attr_base.reshape(B * E, -1), dist_sq), dim=-1)

            edge_mask_t: torch.Tensor | None = None
            if edge_mask_b is not None:
                edge_mask_t = edge_mask_b.reshape(B * E)

            # Run full EGNN stack per step, then update x_t, optionally velocity-like residual
            h_emb = self.egnn.embedding(h)
            x_step = x_flat
            v_step = v_flat
            for layer in self.egnn.layers:
                x_step, v_step, h_emb = layer(x_step, h_emb, edge_index, edge_attr_t, v_step, edge_mask_t)
            x_t = x_step.view(B, N, 3)
            v_t = v_step.view(B, N, 3)
            preds.append(x_t)

        return torch.stack(preds, dim=1)

    def _timestep_embedding(self, T: int, t: int) -> torch.Tensor:
        vec = torch.zeros(T, dtype=torch.float32)
        if 0 <= t < T:
            vec[t] = 1.0
        return vec
