import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gtno_py.dataloaders.egno_dataloder import MD17DynamicsDataset
from gtno_py.training.config_options import (
    DataPartition,
    MD17MoleculeType,
    ModelType,
    RMD17MoleculeType,
)
from gtno_py.training.load_config import Config


def create_datasets(
    config: Config,
    molecule_type: MD17MoleculeType | RMD17MoleculeType,
    max_nodes: int | None = None,
) -> tuple[MD17DynamicsDataset, MD17DynamicsDataset, MD17DynamicsDataset]:
    """Create train, test and validation Torch datasets.

    Args:
        config (Config): The configuration file.
        molecule_type (MD17MoleculeType | RMD17MoleculeType): The molecule type to use.

    Returns:
        tuple[MD17DynamicsDataset, MD17DynamicsDataset, MD17DynamicsDataset]: The train/val/test Torch datasets.
    """

    # If we are using a message passing model, we need to return the edge data
    if config.benchmark.model_type == ModelType.EGNO:
        return_edge_data = True
    else:
        return_edge_data = False

    train_dataset = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=config.dataloader.delta_T,
        num_timesteps=config.dataloader.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.dataloader.md17_version,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=max_nodes,
        force_regenerate=config.dataloader.force_regenerate,
        radius_graph_threshold=config.dataloader.radius_graph_threshold,
        rrwp_length=config.dataloader.rrwp_length,
        return_edge_data=return_edge_data,
    )

    val_dataset = MD17DynamicsDataset(
        partition=DataPartition.val,
        max_samples=2000,
        delta_frame=config.dataloader.delta_T,
        num_timesteps=config.dataloader.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.dataloader.md17_version,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=max_nodes,
        force_regenerate=config.dataloader.force_regenerate,
        radius_graph_threshold=config.dataloader.radius_graph_threshold,
        rrwp_length=config.dataloader.rrwp_length,
        return_edge_data=return_edge_data,
    )

    test_dataset = MD17DynamicsDataset(
        partition=DataPartition.test,
        max_samples=2000,
        delta_frame=config.dataloader.delta_T,
        num_timesteps=config.dataloader.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.dataloader.md17_version,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=max_nodes,
        force_regenerate=config.dataloader.force_regenerate,
        radius_graph_threshold=config.dataloader.radius_graph_threshold,
        rrwp_length=config.dataloader.rrwp_length,
        return_edge_data=return_edge_data,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders_single(
    config: Config,
    molecule_type: MD17MoleculeType | RMD17MoleculeType,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Create train, test and validation Torch dataloaders.

    Args:
        config (Config): The configuration file.
        molecule_type (MD17MoleculeType | RMD17MoleculeType): The molecule type to use.

    Returns:
        tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]: The train/val/test Torch dataloaders.
    """
    train_dataset, val_dataset, test_dataset = create_datasets(config, molecule_type, max_nodes=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
    )

    return train_loader, val_loader, test_loader


def create_dataloaders_multitask(
    config: Config,
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Create train, test and validation Torch dataloaders for multiple molecule types and concatenate them into a single dataloader.

    Args:
        config (Config): The configuration file.

    Returns:
        tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]: The train/val/test Torch dataloaders.
    """
    max_nodes = 0
    # We return a single dataset, so we can just take the num_nodes from that
    assert config.dataloader.train_molecules is not None
    assert config.dataloader.validation_molecules is not None
    assert config.dataloader.test_molecules is not None
    for all_molecule_types in config.dataloader.train_molecules + config.dataloader.validation_molecules + config.dataloader.test_molecules:
        max_nodes_finder, _, _ = create_datasets(config, all_molecule_types, max_nodes=None)
        max_nodes = max(max_nodes, max_nodes_finder.num_nodes)

    tqdm.write(f"Inferred max_nodes across all molecules as: {max_nodes}")
    train_loaders = []
    val_loaders = []
    test_loaders = []

    if config.dataloader.train_molecules is None or config.dataloader.validation_molecules is None or config.dataloader.test_molecules is None:
        raise ValueError("train_molecules, validation_molecules, and test_molecules must be specified for multitask dataloaders")

    for train_molecule_type in config.dataloader.train_molecules:
        train_dataset, _, _ = create_datasets(config, train_molecule_type, max_nodes=max_nodes)
        train_loaders.append(train_dataset)
    for validation_molecule_type in config.dataloader.validation_molecules:
        _, val_dataset, _ = create_datasets(config, validation_molecule_type, max_nodes=max_nodes)
        val_loaders.append(val_dataset)
    for test_molecule_type in config.dataloader.test_molecules:
        _, _, test_dataset = create_datasets(config, test_molecule_type, max_nodes=max_nodes)
        test_loaders.append(test_dataset)

    multitask_train_dataset: torch.utils.data.ConcatDataset[MD17DynamicsDataset] = torch.utils.data.ConcatDataset(train_loaders)
    multitask_val_dataset: torch.utils.data.ConcatDataset[MD17DynamicsDataset] = torch.utils.data.ConcatDataset(val_loaders)
    multitask_test_dataset: torch.utils.data.ConcatDataset[MD17DynamicsDataset] = torch.utils.data.ConcatDataset(test_loaders)

    train_loader = DataLoader(
        multitask_train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )
    val_loader = DataLoader(
        multitask_val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )
    test_loader = DataLoader(
        multitask_test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )

    return train_loader, val_loader, test_loader
