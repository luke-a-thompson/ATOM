import torch
from torch.utils.data import DataLoader
from gtno_py.dataloaders.egno_dataloder import MD17DynamicsDataset, DataPartition, MD17MoleculeType, RMD17MoleculeType
from gtno_py.training.load_config import Config


def create_datasets(
    config: Config,
    molecule_type: MD17MoleculeType | RMD17MoleculeType,
) -> tuple[MD17DynamicsDataset, MD17DynamicsDataset, MD17DynamicsDataset]:
    """Create train, test and validation Torch datasets.

    Args:
        config (Config): The configuration file.
        molecule_type (MD17MoleculeType | RMD17MoleculeType): The molecule type to use.

    Returns:
        tuple[MD17DynamicsDataset, MD17DynamicsDataset, MD17DynamicsDataset]: The train/val/test Torch datasets.
    """
    train_dataset = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=config.benchmark.delta_T,
        num_timesteps=config.model.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.benchmark.md17_version,
        force_regenerate=config.dataloader.force_regenerate,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=config.benchmark.max_nodes,
    )

    val_dataset = MD17DynamicsDataset(
        partition=DataPartition.val,
        max_samples=2000,
        delta_frame=config.benchmark.delta_T,
        num_timesteps=config.model.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.benchmark.md17_version,
        force_regenerate=config.dataloader.force_regenerate,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=config.benchmark.max_nodes,
    )

    test_dataset = MD17DynamicsDataset(
        partition=DataPartition.test,
        max_samples=2000,
        delta_frame=config.benchmark.delta_T,
        num_timesteps=config.model.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.benchmark.md17_version,
        force_regenerate=config.dataloader.force_regenerate,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=config.benchmark.max_nodes,
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
    train_dataset, val_dataset, test_dataset = create_datasets(config, molecule_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
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
    train_loaders = []
    val_loaders = []
    test_loaders = []
    for molecule_type in config.benchmark.molecule_type:
        train_dataset, val_dataset, test_dataset = create_datasets(config, molecule_type)
        train_loaders.append(train_dataset)
        val_loaders.append(val_dataset)
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
