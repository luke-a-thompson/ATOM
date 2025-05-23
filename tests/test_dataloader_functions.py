import torch
from atom.dataloaders.atom_dataloader import MD17DynamicsDataset, DataPartition, MD17MoleculeType
from atom.training.config_options import Datasets
from torch.utils.data import DataLoader

device = "cuda"


class TestMD17DynamicsDataset:
    def test_assert_replicated_dataset_all_identical_md17(self):
        """
        Test that for each sample in the dataset, the replicated tensors are identical along the time dimension.

        MD17DynamicsDataset precomputes replicated versions of x_0, v_0, and concatenated_features by
        expanding them along a time dimension and flattening into a shape of (max_samples * num_timesteps, N, d).
        In __getitem__, a contiguous block of num_timesteps is sliced, resulting in tensors of shape (T, N, d)
        for each sample. For the static features, these T time steps should be identical.

        This function loads one batch from the DataLoader and asserts that, for each key in
        ["x_0", "v_0", "concatenated_features"], all time slices (along the T dimension) are equal.
        """
        config = {
            "model": {"num_timesteps": 8},
            "training": {"batch_size": 2},
            "dataloader": {
                "persistent_workers": False,
                "num_workers": 0,
                "pin_memory": False,
            },
        }

        # Instantiate the dynamics dataset using the actual dataset
        train_dataset = MD17DynamicsDataset(
            partition=DataPartition.train,
            max_samples=500,
            delta_frame=3000,
            md17_version=Datasets.md17,
            num_timesteps=config["model"]["num_timesteps"],
            data_dir="data/",
            split_dir="data/",
            molecule_type=MD17MoleculeType.aspirin,
            max_nodes=13,
            return_edge_data=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            persistent_workers=config["dataloader"]["persistent_workers"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=config["dataloader"]["pin_memory"],
        )

        # Get one batch from the loader
        batch = next(iter(train_loader))

        # Check replicated keys for consistency along the time (T) dimension.
        # Expected shape for these keys: (B, T, N, d)
        for key in ["x_0", "v_0", "concatenated_features"]:
            tensor = batch[key]
            B, T = tensor.shape[0], tensor.shape[1]
            for t in range(1, T):
                assert torch.allclose(tensor[:, 0, ...], tensor[:, t, ...]), f"Replication error for key '{key}': time slice 0 and {t} differ."

        print("Test passed: All replicated tensors are identical along the time dimension.")

    def test_assert_replicated_dataset_all_identical_rmd17(self):
        """
        Test that for each sample in the dataset, the replicated tensors are identical along the time dimension.

        MD17DynamicsDataset precomputes replicated versions of x_0, v_0, and concatenated_features by
        expanding them along a time dimension and flattening into a shape of (max_samples * num_timesteps, N, d).
        In __getitem__, a contiguous block of num_timesteps is sliced, resulting in tensors of shape (T, N, d)
        for each sample. For the static features, these T time steps should be identical.

        This function loads one batch from the DataLoader and asserts that, for each key in
        ["x_0", "v_0", "concatenated_features"], all time slices (along the T dimension) are equal.
        """
        config = {
            "model": {"num_timesteps": 8},
            "training": {"batch_size": 2},
            "dataloader": {
                "persistent_workers": False,
                "num_workers": 0,
                "pin_memory": False,
            },
        }

        # Instantiate the dynamics dataset using the actual dataset
        train_dataset = MD17DynamicsDataset(
            partition=DataPartition.train,
            max_samples=500,
            delta_frame=3000,
            md17_version=Datasets.rmd17,
            num_timesteps=config["model"]["num_timesteps"],
            data_dir="data/",
            split_dir="data/",
            molecule_type=MD17MoleculeType.benzene,
            max_nodes=6,
            return_edge_data=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            persistent_workers=config["dataloader"]["persistent_workers"],
            num_workers=config["dataloader"]["num_workers"],
            pin_memory=config["dataloader"]["pin_memory"],
        )

        # Get one batch from the loader
        batch = next(iter(train_loader))

        # Check replicated keys for consistency along the time (T) dimension.
        # Expected shape for these keys: (B, T, N, d)
        for key in ["x_0", "v_0", "concatenated_features"]:
            tensor = batch[key]
            B, T = tensor.shape[0], tensor.shape[1]
            for t in range(1, T):
                assert torch.allclose(tensor[:, 0, ...], tensor[:, t, ...]), f"Replication error for key '{key}': time slice 0 and {t} differ."

        print("Test passed: All replicated tensors are identical along the time dimension.")

    def test_replicate_tensor(self):
        device = torch.device("cpu")
        num_timesteps = 2
        max_samples = 1  # Set to 1 so that the input tensor must have shape (1, 10, 4)

        # Instantiate a real dataset (it will load data, but we only use it here to access _replicate_tensor)
        dataset = MD17DynamicsDataset(
            partition=DataPartition.train,
            max_samples=max_samples,
            delta_frame=3000,
            num_timesteps=num_timesteps,
            data_dir="data/",
            split_dir="data/",
            md17_version=Datasets.md17,
            molecule_type=MD17MoleculeType.aspirin,
            force_regenerate=True,
            max_nodes=13,
            return_edge_data=False,
        )

        # Create a simple input tensor with shape (max_samples, 10, 4)
        # For example, imagine 10 nodes (N=10) and feature dimension d=4.
        input_tensor = torch.tensor(
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                    [33, 34, 35, 36],
                    [37, 38, 39, 40],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        # Verify the input tensor shape is (1, 10, 4)
        assert input_tensor.shape == (max_samples, 10, 4), f"Input tensor shape {input_tensor.shape} != (1, 10, 4)"

        # Expected output: replicate input_tensor along a new time dimension, resulting in shape (max_samples*num_timesteps, 10, 4)
        # Since num_timesteps=2 and max_samples=1, expected shape is (2, 10, 4) with both time slices identical.
        expected_output_tensor = torch.cat([input_tensor, input_tensor], dim=0).unsqueeze(0)
        assert expected_output_tensor.shape == (1, num_timesteps, 10, 4), f"Expected tensor shape {expected_output_tensor.shape} != (1, 2, 10, 4)"

        # Use the dataset's _replicate_tensor method to perform replication.
        output_tensor = dataset._replicate_tensor(input_tensor)

        # Assert that the output tensor has the expected shape and content.
        assert output_tensor.shape == expected_output_tensor.shape, f"Output shape {output_tensor.shape} != Expected shape {expected_output_tensor.shape}"
        assert torch.equal(output_tensor, expected_output_tensor), f"Output tensor:\n{output_tensor}\nExpected tensor:\n{expected_output_tensor}"

        print("test_replicate_tensor passed!")

    def test_pad_tensor(self):
        device = torch.device("cpu")
        num_timesteps = 2
        max_samples = 1
        max_nodes = 15  # Set max_nodes larger than actual nodes

        # Instantiate a real dataset
        dataset = MD17DynamicsDataset(
            partition=DataPartition.train,
            max_samples=max_samples,
            delta_frame=3000,
            num_timesteps=num_timesteps,
            data_dir="data/",
            split_dir="data/",
            md17_version=MD17Version.md17,
            molecule_type=MD17MoleculeType.aspirin,
            force_regenerate=True,
            max_nodes=max_nodes,
            return_edge_data=False,
        )

        # Create a test tensor with shape (batch_size, num_nodes, feature_dim)
        num_nodes = 10  # Smaller than max_nodes
        feature_dim = 4
        input_tensor = torch.ones((max_samples, num_nodes, feature_dim), dtype=torch.float32)

        # Apply padding
        padded_tensor = dataset._pad_tensor(input_tensor)

        # Verify shape is correct
        expected_shape = (max_samples, max_nodes, feature_dim)
        assert padded_tensor.shape == expected_shape, f"Padded tensor shape {padded_tensor.shape} != Expected shape {expected_shape}"

        # Verify original values are preserved
        assert torch.all(padded_tensor[:, :num_nodes, :] == 1.0), "Original values were modified during padding"

        # Verify padded values are zeros
        assert torch.all(padded_tensor[:, num_nodes:, :] == 0.0), "Padded values are not zeros"

        print("test_pad_tensor passed!")

    def test_node_masking(self):
        """
        Test that the node masking correctly identifies real vs. padded nodes in the dataset.
        """
        device = torch.device("cpu")
        num_timesteps = 3
        max_samples = 2
        max_nodes = 12  # Set max_nodes larger than actual nodes

        # Instantiate a dataset with a molecule that has fewer atoms than max_nodes
        dataset = MD17DynamicsDataset(
            partition=DataPartition.train,
            max_samples=max_samples,
            delta_frame=3000,
            num_timesteps=num_timesteps,
            data_dir="data/",
            split_dir="data/",
            md17_version=MD17Version.md17,
            molecule_type=MD17MoleculeType.benzene,  # Benzene has 6 carbon atoms (or 12 with hydrogens)
            force_regenerate=False,
            max_nodes=max_nodes,
            return_edge_data=False,
        )

        # Get a sample from the dataset
        sample = dataset[0]

        # Check that the mask has the correct shape: (num_timesteps, max_nodes, 1)
        expected_mask_shape = (num_timesteps, max_nodes, 1)
        assert sample["padded_nodes_mask"].shape == expected_mask_shape, f"Mask shape {sample['padded_nodes_mask'].shape} != Expected shape {expected_mask_shape}"

        # Check that the number of True values in the mask equals the actual number of nodes
        actual_nodes = dataset.num_nodes
        assert actual_nodes == 6, f"Actual number of nodes {actual_nodes} != Expected number of nodes 6"
        true_count = sample["padded_nodes_mask"].sum().item()
        expected_true_count = actual_nodes * num_timesteps  # True for each real node across all timesteps
        assert true_count == expected_true_count, f"True count in mask {true_count} != Expected count {expected_true_count}"

        # Verify that the mask is True for real nodes and False for padded nodes
        for t in range(num_timesteps):
            assert torch.all(sample["padded_nodes_mask"][t, :actual_nodes, 0] == True), f"Real nodes not correctly masked as True at timestep {t}"
            assert torch.all(sample["padded_nodes_mask"][t, actual_nodes:, 0] == False), f"Padded nodes not correctly masked as False at timestep {t}"

        print("test_node_masking passed!")
