import torch
from atom.atom.positional_encodings import TemporalRoPE

device = "cuda"


class TestTemporalRoPEGroup:
    def test_temporal_rope_preserves_vector_norms_and_is_nontrivial(self) -> None:
        batch_size: int = 2
        num_timesteps: int = 2
        num_nodes: int = 3
        d: int = 4
        num_heads: int = 2
        d_head: int = d // num_heads

        # Shape [B, H, N * T, d_head]
        flattened_x: torch.Tensor = torch.ones(
            batch_size,
            num_heads,
            num_timesteps * num_nodes,
            d_head,
            device=device,
        )

        trope = TemporalRoPE(num_timesteps=num_timesteps, d_head=d_head, n_heads=num_heads)
        rope_output: torch.Tensor = trope(flattened_x, mask=None, time_increments=None)

        # Shape should be preserved
        assert rope_output.shape == flattened_x.shape

        # Norms of the last half of the sequence (where rotation is non-trivial) should be preserved
        # Flatten over batch, head and sequence, then compare per‑vector norms
        orig_norms: torch.Tensor = torch.linalg.vector_norm(flattened_x, dim=-1)
        out_norms: torch.Tensor = torch.linalg.vector_norm(rope_output, dim=-1)
        assert torch.allclose(orig_norms, out_norms, atol=1e-5)

        # Check that at least one vector actually changed (rotation is not identity everywhere)
        assert torch.any(torch.abs(rope_output - flattened_x) > 1e-6)

    def test_rope_timestep_interleave(self):
        num_timesteps = 3
        num_nodes = 4

        times = torch.arange(num_timesteps).unsqueeze(1)  # [T,1]
        positions = torch.repeat_interleave(times, num_nodes, dim=1).flatten(0, 1)  # [N*T=seq_len]

        assert torch.equal(positions, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))

    def test_rope_stack_interleave(self):
        original = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.int32)

        t1 = original[..., 0::2]
        t2 = original[..., 1::2]

        new = torch.stack([t1, t2], dim=-1).view_as(original)

        assert torch.equal(new, original)

    def test_rope_cos_sin_rotations(self):
        num_timesteps = 2
        num_nodes = 3
        num_heads = 2
        d_head = 4

        times = torch.arange(num_timesteps).unsqueeze(1)  # [T,1]
        positions = torch.repeat_interleave(times, num_nodes, dim=1).flatten(0, 1).to(device)  # [N*T=seq_len]

        offset = torch.zeros(num_heads, device=device)
        freqs = torch.tensor([[[1.0, 0.0316227766016838]]], device=device)  # Analytically derived from d_head = 4, manually unsqueezed

        positions_broadcast = positions.unsqueeze(0)  # [1, seq_len]
        offset_broadcast = offset.unsqueeze(-1)  # [H, 1]
        shifted_positions = positions_broadcast + offset_broadcast

        angle = shifted_positions.unsqueeze(-1) * freqs

        assert angle.shape == torch.Size([num_heads, num_nodes * num_timesteps, d_head // 2])

        cos_t = angle.cos().unsqueeze(0)
        sin_t = angle.sin().unsqueeze(0)

        # [1, H, seq_len, half_dim]
        # Using cosine of freqs
        # Head = yellow
        expected_cos_t = torch.tensor(
            [
                [
                    [
                        [1, 1],  # T = 0; left = timestep, right = timestep * freq
                        [1, 1],  # T = 0
                        [1, 1],  # T = 0
                        [0.5403, 0.9995],  # T = 1
                        [0.5403, 0.9995],  # T = 1
                        [0.5403, 0.9995],  # T = 1
                    ],
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [0.5403, 0.9995],
                        [0.5403, 0.9995],
                        [0.5403, 0.9995],
                    ],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )

        expected_sin_t = torch.tensor(
            [
                [
                    [
                        [0, 0],  # T = 0; left = timestep, right = timestep * freq
                        [0, 0],  # T = 0
                        [0, 0],  # T = 0
                        [0.8414, 0.0316],  # T = 1
                        [0.8414, 0.0316],  # T = 1
                        [0.8414, 0.0316],  # T = 1
                    ],
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0.8414, 0.0316],
                        [0.8414, 0.0316],
                        [0.8414, 0.0316],
                    ],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )

        assert torch.allclose(cos_t, expected_cos_t, atol=1e-3), f"cos_t: \n{cos_t}, \nexpected_cos_t: \n{expected_cos_t}"
        assert torch.allclose(sin_t, expected_sin_t, atol=1e-3), f"sin_t: \n{sin_t}, \nexpected_sin_t: \n{expected_sin_t}"
