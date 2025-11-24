import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from typing import cast

from atom.dataloaders.atom_dataloader import MDDynamicsDataset
from atom.training.config_options import (
    DataPartition,
    MD17MoleculeType,
    RMD17MoleculeType,
    MD22MoleculeType,
    TG80MoleculeType,
    ModelType,
    AttentionType,
)
from atom.training.create_config import Config


def create_datasets(
    config: Config,
    molecule_type: MD17MoleculeType | RMD17MoleculeType | TG80MoleculeType | MD22MoleculeType,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> tuple[MDDynamicsDataset, MDDynamicsDataset, MDDynamicsDataset]:
    """Create train, test and validation Torch datasets.

    Args:
        config (Config): The configuration file.
        molecule_type (MD17MoleculeType | RMD17MoleculeType): The molecule type to use.
        max_nodes (int | None): Maximum number of nodes to pad to.
        max_edges (int | None): Maximum number of edges to pad to.

    Returns:
        tuple[MDDynamicsDataset, MDDynamicsDataset, MDDynamicsDataset]: The train/val/test Torch datasets.
    """

    # Decide when to construct edge data.
    if config.benchmark.model_type in (
        ModelType.EGNO,
        ModelType.EGNN_S,
        ModelType.EGNN_R,
    ):
        return_edge_data = True
        egno_mode = True
        use_two_hop_edges = True
    elif (
        config.benchmark.model_type == ModelType.ATOM
        and config.atom_config is not None
        and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
    ):
        # ATOM + GATV2: graph-based attention over one-hop edges only.
        return_edge_data = True
        egno_mode = True
        use_two_hop_edges = False
    else:
        return_edge_data = False
        egno_mode = False
        use_two_hop_edges = True

    train_dataset = MDDynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=config.dataloader.delta_T,
        num_timesteps=config.dataloader.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.dataloader.dataset,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=max_nodes,
        force_regenerate=config.dataloader.force_regenerate,
        radius_graph_threshold=config.dataloader.radius_graph_threshold,
        rrwp_length=config.dataloader.rrwp_length,
        return_edge_data=return_edge_data,
        egno_mode=egno_mode,
        max_edges=max_edges,
        use_two_hop_edges=use_two_hop_edges,
    )

    val_dataset = MDDynamicsDataset(
        partition=DataPartition.val,
        max_samples=2000,
        delta_frame=config.dataloader.delta_T,
        num_timesteps=config.dataloader.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.dataloader.dataset,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=max_nodes,
        force_regenerate=config.dataloader.force_regenerate,
        radius_graph_threshold=config.dataloader.radius_graph_threshold,
        rrwp_length=config.dataloader.rrwp_length,
        return_edge_data=return_edge_data,
        egno_mode=egno_mode,
        max_edges=max_edges,
        use_two_hop_edges=use_two_hop_edges,
    )

    test_dataset = MDDynamicsDataset(
        partition=DataPartition.test,
        max_samples=2000,
        delta_frame=config.dataloader.delta_T,
        num_timesteps=config.dataloader.num_timesteps,
        data_dir="data/",
        split_dir="data/",
        molecule_type=molecule_type,
        md17_version=config.dataloader.dataset,
        explicit_hydrogen=config.dataloader.explicit_hydrogen,
        max_nodes=max_nodes,
        force_regenerate=config.dataloader.force_regenerate,
        radius_graph_threshold=config.dataloader.radius_graph_threshold,
        rrwp_length=config.dataloader.rrwp_length,
        return_edge_data=return_edge_data,
        egno_mode=egno_mode,
        max_edges=max_edges,
        use_two_hop_edges=use_two_hop_edges,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders_single(
    config: Config,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Create train, test and validation Torch dataloaders.

    Args:
        config (Config): The configuration file.
        molecule_type (MD17MoleculeType | RMD17MoleculeType): The molecule type to use.

    Returns:
        tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]: The train/val/test Torch dataloaders.
    """
    molecule_type = config.dataloader.molecule_type
    if molecule_type is None:
        msg = "For single-task training, 'dataloader.molecule_type' must be set."
        raise ValueError(msg)

    assert molecule_type is not None

    train_dataset, val_dataset, test_dataset = create_datasets(config, molecule_type, max_nodes=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        collate_fn=_pad_edges_to_uniform_length
        if (
            config.benchmark.model_type in (ModelType.EGNO, ModelType.EGNN_S, ModelType.EGNN_R)
            or (
                config.benchmark.model_type == ModelType.ATOM
                and config.atom_config is not None
                and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
            )
        )
        else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        collate_fn=_pad_edges_to_uniform_length
        if (
            config.benchmark.model_type in (ModelType.EGNO, ModelType.EGNN_S, ModelType.EGNN_R)
            or (
                config.benchmark.model_type == ModelType.ATOM
                and config.atom_config is not None
                and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
            )
        )
        else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        collate_fn=_pad_edges_to_uniform_length
        if (
            config.benchmark.model_type in (ModelType.EGNO, ModelType.EGNN_S, ModelType.EGNN_R)
            or (
                config.benchmark.model_type == ModelType.ATOM
                and config.atom_config is not None
                and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
            )
        )
        else None,
    )

    return train_loader, val_loader, test_loader


def create_dataloaders_multitask(
    config: Config,
) -> tuple[
    DataLoader[MDDynamicsDataset],
    DataLoader[MDDynamicsDataset],
    DataLoader[MDDynamicsDataset],
]:
    """Create train, test and validation Torch dataloaders for multiple molecule types and concatenate them into a single dataloader.

    Args:
        config (Config): The configuration file.

    Returns:
        tuple[DataLoader[MD17DynamicsDataset], DataLoader[MD17DynamicsDataset], DataLoader[MD17DynamicsDataset]]: The train/val/test Torch dataloaders.
    """
    max_nodes: int = 0
    max_edges: int = 0
    # We return a single dataset, so we can just take the num_nodes from that
    assert config.dataloader.train_molecules is not None
    assert config.dataloader.validation_molecules is not None
    assert config.dataloader.test_molecules is not None
    for molecule_type in config.dataloader.train_molecules + config.dataloader.validation_molecules + config.dataloader.test_molecules:
        try:
            max_nodes_finder, _, _ = create_datasets(config, molecule_type, max_nodes=None)
            max_nodes = max(max_nodes, max_nodes_finder.num_nodes)
            # Compute max edges for this molecule
            one_hop_adjacency, _ = max_nodes_finder._compute_adjacency_matrix(
                max_nodes_finder.x,
                max_nodes_finder.num_nodes,
                max_nodes_finder.radius_graph_threshold,
            )
            num_edges = int(one_hop_adjacency.sum().item())
            max_edges = max(max_edges, num_edges)
        except Exception as e:
            tqdm.write(f"Skipping molecule {molecule_type} due to dataset/graph error: {e}")
            continue

    tqdm.write(f"Inferred max_nodes across all molecules as: {max_nodes}")
    tqdm.write(f"Inferred max_edges across all molecules as: {max_edges}")
    train_loaders: list[MDDynamicsDataset] = []
    val_loaders: list[MDDynamicsDataset] = []
    test_loaders: list[MDDynamicsDataset] = []

    for train_molecule_type in config.dataloader.train_molecules:
        try:
            train_dataset, _, _ = create_datasets(config, train_molecule_type, max_nodes=max_nodes, max_edges=max_edges)
            train_loaders.append(train_dataset)
        except Exception as e:
            tqdm.write(f"Skipping train molecule {train_molecule_type} due to dataset/graph error: {e}")
    for validation_molecule_type in config.dataloader.validation_molecules:
        try:
            _, val_dataset, _ = create_datasets(
                config,
                validation_molecule_type,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
            val_loaders.append(val_dataset)
        except Exception as e:
            tqdm.write(f"Skipping validation molecule {validation_molecule_type} due to dataset/graph error: {e}")
    for test_molecule_type in config.dataloader.test_molecules:
        try:
            _, _, test_dataset = create_datasets(config, test_molecule_type, max_nodes=max_nodes, max_edges=max_edges)
            test_loaders.append(test_dataset)
        except Exception as e:
            tqdm.write(f"Skipping test molecule {test_molecule_type} due to dataset/graph error: {e}")

    if len(train_loaders) == 0 or len(val_loaders) == 0 or len(test_loaders) == 0:
        raise RuntimeError("No valid datasets remained after skipping failing molecules. Check your data/configs.")
    multitask_train_dataset: torch.utils.data.ConcatDataset[MDDynamicsDataset] = torch.utils.data.ConcatDataset(train_loaders)
    multitask_val_dataset: torch.utils.data.ConcatDataset[MDDynamicsDataset] = torch.utils.data.ConcatDataset(val_loaders)
    multitask_test_dataset: torch.utils.data.ConcatDataset[MDDynamicsDataset] = torch.utils.data.ConcatDataset(test_loaders)

    train_loader = DataLoader(
        multitask_train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        collate_fn=_pad_edges_to_uniform_length
        if (
            config.benchmark.model_type in (ModelType.EGNO, ModelType.EGNN_S, ModelType.EGNN_R)
            or (
                config.benchmark.model_type == ModelType.ATOM
                and config.atom_config is not None
                and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
            )
        )
        else None,
    )
    val_loader = DataLoader(
        multitask_val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        collate_fn=_pad_edges_to_uniform_length
        if (
            config.benchmark.model_type in (ModelType.EGNO, ModelType.EGNN_S, ModelType.EGNN_R)
            or (
                config.benchmark.model_type == ModelType.ATOM
                and config.atom_config is not None
                and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
            )
        )
        else None,
    )
    test_loader = DataLoader(
        multitask_test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        persistent_workers=config.dataloader.persistent_workers,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        collate_fn=_pad_edges_to_uniform_length
        if (
            config.benchmark.model_type in (ModelType.EGNO, ModelType.EGNN_S, ModelType.EGNN_R)
            or (
                config.benchmark.model_type == ModelType.ATOM
                and config.atom_config is not None
                and config.atom_config.heterogenous_attention_type == AttentionType.GATV2
            )
        )
        else None,
    )

    return train_loader, val_loader, test_loader


# -----------------------
# Custom collate function
# -----------------------


def _pad_edges_to_uniform_length(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function that zero-pads *edge* tensors so that all samples in a batch have
    the same number of edges.

    The EGNO implementation expects tensors with shape ``[B, E, ...]`` where ``E`` is
    identical for every sample in the batch.  When batching molecules with different
    numbers of edges the default PyTorch ``default_collate`` fails.  This helper first
    determines the maximum number of edges in the batch and then right-pads the
    following keys to that size:
        * ``edge_attr``               - float tensor of shape ``[E, d_e]``
        * ``source_node_indices``     - integer tensor of shape ``[E]``
        * ``target_node_indices``     - integer tensor of shape ``[E]``

    All other keys are collated with the stock ``default_collate``.
    """

    # Keys that require special handling
    edge_keys = {"edge_attr", "source_node_indices", "target_node_indices"}

    # Determine the maximum edge count in the incoming mini-batch
    max_edges = max(sample["edge_attr"].shape[0] for sample in batch)

    padded_batch: list[dict[str, torch.Tensor]] = []
    for sample in batch:
        padded_sample: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if key in edge_keys:
                # Compute the amount of padding required for this sample
                pad_len = max_edges - value.shape[0]
                if pad_len > 0:
                    if key == "edge_attr":
                        # value shape: [E, d_e] – pad rows with zeros
                        pad_tensor = torch.zeros(pad_len, value.shape[1], dtype=value.dtype)
                    else:
                        # index tensors are 1-D – pad with zeros (valid self-loop indices)
                        pad_tensor = torch.zeros(pad_len, dtype=value.dtype)
                    value = torch.cat([value, pad_tensor], dim=0)
                padded_sample[key] = value
            else:
                padded_sample[key] = value
        # Provide an explicit boolean edge mask for padded vs real edges
        edge_count = sample["edge_attr"].shape[0]
        edge_mask = torch.ones(edge_count, dtype=torch.bool)
        if edge_count < max_edges:
            pad_len = max_edges - edge_count
            edge_mask = torch.cat([edge_mask, torch.zeros(pad_len, dtype=torch.bool)], dim=0)
        padded_sample["edge_mask"] = edge_mask
        padded_batch.append(padded_sample)

    # Delegate the heavy lifting of stacking to the default collate implementation
    return cast(dict[str, torch.Tensor], default_collate(padded_batch))
