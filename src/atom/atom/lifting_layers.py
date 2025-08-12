from typing import final, override

import torch
import torch.nn as nn
from e3nn import o3


@final
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
        lifted_x_0 = self.x_0_linear(x_0)
        lifted_v_0 = self.v_0_linear(v_0)
        lifted_concat_features = self.concat_feats_linear(concatenated_features)
        return lifted_x_0, lifted_v_0, lifted_concat_features


@final
class EquivariantLift(nn.Module):
    def __init__(self, x_0_in_irreps: str, v_0_in_irreps: str, concat_feats_in_irreps: str, lifting_dim_irreps: str) -> None:
        super().__init__()
        self.x_0_linear = o3.Linear(x_0_in_irreps, lifting_dim_irreps)
        self.v_0_linear = o3.Linear(v_0_in_irreps, lifting_dim_irreps)
        self.concat_feats_linear = o3.Linear(concat_feats_in_irreps, lifting_dim_irreps)

    @override
    def forward(self, x_0: torch.Tensor, v_0: torch.Tensor, concatenated_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lifted_x_0 = self.x_0_linear(x_0)
        lifted_v_0 = self.v_0_linear(v_0)
        lifted_concat_features = self.concat_feats_linear(concatenated_features)
        return lifted_x_0, lifted_v_0, lifted_concat_features


@final
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
        lifted_x_0 = self.x_0_linear(x_0)
        lifted_v_0 = self.v_0_linear(v_0)

        lifted_vz_0 = self.vz_0_linear(concatenated_features[..., 4:])
        # assert False, (x_0.shape, v_0.shape, vz_0.shape, concatenated_features.shape)
        lifted_concat_features = self.concat_feats_linear(lifted_x_0, lifted_vz_0)
        return lifted_x_0, lifted_v_0, lifted_concat_features


@final
class CanonicalizationLift(nn.Module):
    def __init__(self, x_0_in_irreps: str, v_0_in_irreps: str, concat_feats_in_irreps: str, lifting_dim_irreps: str) -> None:
        super().__init__()
        self.canonical_matrix_maker = o3.Linear("1x1o + 1x1o", "3x1o")

        self.x_0_linear = o3.Linear(x_0_in_irreps, lifting_dim_irreps)
        self.v_0_linear = o3.Linear(v_0_in_irreps, lifting_dim_irreps)

        vz_0_in_irreps: str = v_0_in_irreps + " + 1x0e"  # (vx,vy,vz, ||v||, Z)
        self.vz_0_linear = o3.Linear(vz_0_in_irreps, lifting_dim_irreps)
        self.concat_feats_linear = o3.FullyConnectedTensorProduct(lifting_dim_irreps, lifting_dim_irreps, lifting_dim_irreps)

        self.test_linear = o3.Linear(concat_feats_in_irreps, lifting_dim_irreps)

    @override
    def forward(
        self, x_0: torch.Tensor, v_0: torch.Tensor, concatenated_features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use only xyz parts for building the frame
        x_xyz: torch.Tensor = x_0[..., :3]
        v_xyz: torch.Tensor = v_0[..., :3]

        # Validity mask per node (broadcast to xyz). Expect mask shape [..., N, 1] or [..., N, D]
        if mask is not None:
            valid_mask: torch.Tensor = mask[..., :1]
        else:
            # Infer from zeros if mask not provided
            valid_mask = (x_xyz.abs().sum(dim=-1, keepdim=True) > 0).to(x_xyz.dtype)

        # Translation-invariant positions (centered per batch/time), masked
        denom: torch.Tensor = valid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        x_mean: torch.Tensor = (x_xyz * valid_mask).sum(dim=2, keepdim=True) / denom
        x_centered: torch.Tensor = x_xyz - x_mean
        x_cent_masked: torch.Tensor = x_centered * valid_mask
        v_masked: torch.Tensor = v_xyz * valid_mask

        # Build symmetric second-moment tensors (batchable) using masked values
        # Sa = mean_i x_i x_i^T, Sb = mean_i v_i v_i^T (normalize by denom to stabilize scale)
        x0_m: torch.Tensor = x_cent_masked[:, 0]  # [B, N, 3]
        v0_m: torch.Tensor = v_masked[:, 0]  # [B, N, 3]
        denom0: torch.Tensor = denom[:, 0].to(x0_m.dtype)  # [B, 1, 1]
        Sa0: torch.Tensor = (x0_m.transpose(1, 2) @ x0_m) / denom0  # [B, 3, 3]

        # Numerical helpers
        eps: float = 1e-8
        eye3: torch.Tensor = torch.eye(3, device=x_0.device, dtype=x_0.dtype)

        # Principal axes from Sa0 (positions only) for stability
        evals_a, evecs_a = torch.linalg.eigh(Sa0 + eps * eye3)
        e1: torch.Tensor = evecs_a[..., :, -1]
        e2_pc: torch.Tensor = evecs_a[..., :, -2]
        e1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)
        e2_pc = e2_pc / (e2_pc.norm(dim=-1, keepdim=True) + eps)

        # Pseudoscalar for deterministic sign using t=0: c0 = sum_i x0_i × v0_i
        c0: torch.Tensor = torch.sum(torch.linalg.cross(x0_m, v0_m, dim=-1), dim=1)
        sign_e1: torch.Tensor = torch.where((e1 * c0).sum(dim=-1, keepdim=True) >= 0, 1.0, -1.0).to(e1.dtype)
        e1 = e1 * sign_e1

        # Orthonormalize e2 against e1
        e2_ortho: torch.Tensor = e2_pc - (e2_pc * e1).sum(dim=-1, keepdim=True) * e1
        e2: torch.Tensor = e2_ortho / (e2_ortho.norm(dim=-1, keepdim=True) + eps)

        # e3 completes right-handed frame; sign with c0
        e3: torch.Tensor = torch.linalg.cross(e1, e2)
        e3 = e3 / (e3.norm(dim=-1, keepdim=True) + eps)
        sign_e3: torch.Tensor = torch.where((e3 * c0).sum(dim=-1, keepdim=True) >= 0, 1.0, -1.0).to(e3.dtype)
        e2 = e2 * sign_e3
        e3 = e3 * sign_e3

        # Rotation matrix Q0 with columns [e1, e2, e3], then broadcast over time
        Q0: torch.Tensor = torch.stack([e1, e2, e3], dim=-1)  # [B, 3, 3]
        Q: torch.Tensor = Q0.unsqueeze(1).expand(-1, x_0.shape[1], -1, -1)  # [B, T, 3, 3]

        # Apply rotation to xyz (row-vector convention)
        x_can_xyz: torch.Tensor = x_xyz @ Q
        v_can_xyz: torch.Tensor = v_xyz @ Q

        # Concatenate back invariant/scalar channels unchanged
        x_can: torch.Tensor = torch.cat((x_can_xyz, x_0[..., 3:]), dim=-1)
        v_can: torch.Tensor = torch.cat((v_can_xyz, v_0[..., 3:]), dim=-1)

        # Include additional features and downstream lifts
        vz_can: torch.Tensor = torch.cat((v_can, concatenated_features[..., -1:]), dim=-1)  # Include Z channel

        lifted_x_can: torch.Tensor = self.x_0_linear(x_can)
        lifted_v_can: torch.Tensor = self.v_0_linear(v_can)
        lifted_vz_can: torch.Tensor = self.vz_0_linear(vz_can)
        lifted_concat_features: torch.Tensor = self.concat_feats_linear(lifted_x_can, lifted_vz_can)

        so3_matrix: torch.Tensor = Q  # [..., 3, 3]
        return lifted_x_can, lifted_v_can, lifted_concat_features, so3_matrix, x_mean
