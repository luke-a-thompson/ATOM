import torch
from gtno_py.gtno.attentions import TemporalRoPEWithOffset

device = "cuda"


class TestTemporalRoPEWithOffsetGroup:
    def test_temporal_rope_vector_norm_unchanged(self):
        batch_size = 2
        num_timesteps = 2
        num_nodes = 3
        d = 4
        num_heads = 2
        d_head = d // num_heads

        # Batches = blue; heads = yellow; nodes * timesteps = purple
        # [Batch, heads, nodes * timesteps, d_head]
        flattened_x = torch.tensor(
            [
                [
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                    ],
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                    ],
                ],
                [
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                    ],
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [1, 1],
                    ],
                ],
            ],
            device=device,
        )
        # Freqs =  1.0, 0.0316227766016838
        rope_output = TemporalRoPEWithOffset(num_timesteps=num_timesteps, d_head=d_head, n_heads=num_heads).forward(flattened_x)

        # [Batch, heads, nodes * timesteps, d_head]
        # Purple = [even, odd]
        expected_rope_output = torch.tensor(
            [
                [
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                    ],
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                    ],
                ],
                [
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                    ],
                    [
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                        [-0.3011, 1.3817],
                    ],
                ],
            ],
            dtype=torch.float32,
            device=device,
        )

        assert rope_output.shape == expected_rope_output.shape
        assert torch.allclose(rope_output, expected_rope_output, atol=1e-2), f"rope_output: \n{rope_output}, \nexpected_rope_output: \n{expected_rope_output}"

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
