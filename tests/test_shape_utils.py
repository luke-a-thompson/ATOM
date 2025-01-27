import torch
from gtno_py.gtno.shape_utils import flatten_spatiotemporal, unflatten_spatiotemporal

device = "cuda"


class TestShapeUtilsGrouped:
    def test_flatten_unflatten_spatiotemporal(self):
        batch_size = 2
        num_timesteps = 3
        num_nodes = 4
        d = 5

        unflattened_x = torch.tensor(
            [
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40]],
                [[41, 42, 43, 44, 45], [46, 47, 48, 49, 50], [51, 52, 53, 54, 55], [56, 57, 58, 59, 60]],
                [[61, 62, 63, 64, 65], [66, 67, 68, 69, 70], [71, 72, 73, 74, 75], [76, 77, 78, 79, 80]],
                [[81, 82, 83, 84, 85], [86, 87, 88, 89, 90], [91, 92, 93, 94, 95], [96, 97, 98, 99, 100]],
                [[101, 102, 103, 104, 105], [106, 107, 108, 109, 110], [111, 112, 113, 114, 115], [116, 117, 118, 119, 120]],
            ],
            dtype=torch.int32,
            device=device,
        )

        flattened_x = flatten_spatiotemporal(unflattened_x, num_timesteps)
        unflattened_x_2 = unflatten_spatiotemporal(flattened_x, num_timesteps)

        assert torch.equal(unflattened_x_2, unflattened_x)

    def test_flatten_spatiotemporal(self):
        batch_size = 2
        num_timesteps = 3
        num_nodes = 4
        d = 5

        # Batches = blue; Nodes = yellow
        unflattened_x = torch.tensor(
            [
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40]],
                [[41, 42, 43, 44, 45], [46, 47, 48, 49, 50], [51, 52, 53, 54, 55], [56, 57, 58, 59, 60]],
                [[61, 62, 63, 64, 65], [66, 67, 68, 69, 70], [71, 72, 73, 74, 75], [76, 77, 78, 79, 80]],
                [[81, 82, 83, 84, 85], [86, 87, 88, 89, 90], [91, 92, 93, 94, 95], [96, 97, 98, 99, 100]],
                [[101, 102, 103, 104, 105], [106, 107, 108, 109, 110], [111, 112, 113, 114, 115], [116, 117, 118, 119, 120]],
            ],
            dtype=torch.int32,
            device=device,
        )

        flattened_x = flatten_spatiotemporal(unflattened_x, num_timesteps)

        assert flattened_x.shape == (batch_size, num_nodes * num_timesteps, d)

        # Batches = blue; Nodes * time = yellow
        expected_flattened_x = torch.tensor(
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                    [26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35],
                    [36, 37, 38, 39, 40],
                    [41, 42, 43, 44, 45],
                    [46, 47, 48, 49, 50],
                    [51, 52, 53, 54, 55],
                    [56, 57, 58, 59, 60],
                ],
                [
                    [61, 62, 63, 64, 65],
                    [66, 67, 68, 69, 70],
                    [71, 72, 73, 74, 75],
                    [76, 77, 78, 79, 80],
                    [81, 82, 83, 84, 85],
                    [86, 87, 88, 89, 90],
                    [91, 92, 93, 94, 95],
                    [96, 97, 98, 99, 100],
                    [101, 102, 103, 104, 105],
                    [106, 107, 108, 109, 110],
                    [111, 112, 113, 114, 115],
                    [116, 117, 118, 119, 120],
                ],
            ],
            device=device,
        )

        assert flattened_x.shape == expected_flattened_x.shape
        assert torch.equal(flattened_x, expected_flattened_x), f"Flattened x: {flattened_x}\nExpected flattened x: {expected_flattened_x}"

    def test_unflatten_spatiotemporal(self):
        batch_size = 2
        num_timesteps = 3
        num_nodes = 4
        d = 5

        # Batches = blue; Nodes * time = yellow
        flattened_x = torch.tensor(
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                    [26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35],
                    [36, 37, 38, 39, 40],
                    [41, 42, 43, 44, 45],
                    [46, 47, 48, 49, 50],
                    [51, 52, 53, 54, 55],
                    [56, 57, 58, 59, 60],
                ],
                [
                    [61, 62, 63, 64, 65],
                    [66, 67, 68, 69, 70],
                    [71, 72, 73, 74, 75],
                    [76, 77, 78, 79, 80],
                    [81, 82, 83, 84, 85],
                    [86, 87, 88, 89, 90],
                    [91, 92, 93, 94, 95],
                    [96, 97, 98, 99, 100],
                    [101, 102, 103, 104, 105],
                    [106, 107, 108, 109, 110],
                    [111, 112, 113, 114, 115],
                    [116, 117, 118, 119, 120],
                ],
            ],
            device=device,
        )

        # Batches = blue; Nodes = yellow
        expected_unflattened = torch.tensor(
            [
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40]],
                [[41, 42, 43, 44, 45], [46, 47, 48, 49, 50], [51, 52, 53, 54, 55], [56, 57, 58, 59, 60]],
                [[61, 62, 63, 64, 65], [66, 67, 68, 69, 70], [71, 72, 73, 74, 75], [76, 77, 78, 79, 80]],
                [[81, 82, 83, 84, 85], [86, 87, 88, 89, 90], [91, 92, 93, 94, 95], [96, 97, 98, 99, 100]],
                [[101, 102, 103, 104, 105], [106, 107, 108, 109, 110], [111, 112, 113, 114, 115], [116, 117, 118, 119, 120]],
            ],
            dtype=torch.int32,
            device=device,
        )

        unflattened_x = unflatten_spatiotemporal(flattened_x, num_timesteps)

        assert unflattened_x.shape == expected_unflattened.shape
        assert torch.equal(unflattened_x, expected_unflattened)
