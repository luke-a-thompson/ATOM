from datetime import datetime
from pathlib import Path

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.std import tqdm
import wandb
from torch import autocast
from torch.amp.grad_scaler import GradScaler

from atom.dataloaders.atom_dataloader import MD17DynamicsDataset
from atom.training import (
    Config,
    SingleRunResults,
    add_brownian_noise,
    create_dataloaders_multitask,
    create_dataloaders_single,
    initialize_optimizer,
    initialize_scheduler,
    log_weights,
)


def train_model(config: Config, model: nn.Module, benchmark_dir: Path, run_number: int) -> SingleRunResults:
    """Full training pipeline."""

    if config.dataloader.multitask:
        train_loader, val_loader, test_loader = create_dataloaders_multitask(config)
    else:
        train_loader, val_loader, test_loader = create_dataloaders_single(config)

    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer)

    # Create a temporary directory for this run
    run_dir = benchmark_dir / f"run_{run_number+1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_model = run_dir / "best_val_model.pth"

    # Training loop
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    scaler = GradScaler(enabled=config.training.use_amp)

    start_training_time = datetime.now()
    progress_bar = tqdm(range(config.training.epochs), desc="Training", leave=False, unit="epoch", position=2)
    for epoch in progress_bar:
        train_s2t_loss = train_epoch(config, model, optimizer, train_loader, scheduler, scaler)
        val_s2t_loss, val_s2s_loss = eval_epoch(config, model, val_loader)

        # Log gate parameters and save to weights_dir if provided
        if config.benchmark.log_weights:
            log_weights(list(model.named_parameters()), epoch, save_dir=run_dir)

        wandb.log({"train_s2t_loss": train_s2t_loss, "val_s2t_loss": val_s2t_loss, "lr": optimizer.param_groups[0]["lr"]})

        # if val_loss < best_val_loss and epoch > 0.5 * num_epochs:
        if val_s2t_loss < best_val_loss:
            best_val_loss = val_s2t_loss
            best_val_loss_epoch = epoch
            torch.save(model.state_dict(), best_val_model)

        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_s2t_loss)

        # Update progress bar with losses
        progress_bar.set_postfix(
            {
                "Train s2t loss": f"{train_s2t_loss*100:.2f}x10^-2",
                "Val s2t loss": f"{val_s2t_loss*100:.2f}x10^-2",
                "Best val s2t loss": f"{best_val_loss*100:.2f}x10^-2",
                f"Current {config.optimizer.type} LR": f"{optimizer.param_groups[0]['lr']:.4f}",
            }
        )
    end_training_time = datetime.now()

    # Final evaluation
    _ = model.load_state_dict(torch.load(best_val_model, weights_only=True))
    s2t_test_loss, s2s_test_loss = eval_epoch(config, model, test_loader)

    results = SingleRunResults(
        s2t_test_loss=s2t_test_loss,
        s2s_test_loss=s2s_test_loss,
        best_val_loss_epoch=best_val_loss_epoch,
        start_time=start_training_time,
        end_time=end_training_time,
        model_path=Path(best_val_model),
    )

    return results


def train_epoch(
    config: Config,
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader[dict[str, torch.Tensor]] | DataLoader[MD17DynamicsDataset],
    scheduler: optim.lr_scheduler._LRScheduler | None,
    scaler: GradScaler,
) -> float:
    """Single training epoch.

    Args:
        config (Config): The configuration file.
        model (nn.Module): The model to train.
        optimizer (optim.Optimizer): The optimizer to use.
        dataloader (DataLoader[dict[str, torch.Tensor]]): The dataloader to use.
        scheduler (optim.lr_scheduler._LRScheduler | None): The scheduler to use.

    Returns:
        float: The loss of the epoch.
    """
    _ = model.train()
    total_s2t_loss = 0.0

    for batch in dataloader:
        batch = TensorDict.from_dict(batch, device=torch.device(config.training.device), auto_batch_size=True)
        if config.dataloader.multitask is False:
            assert "padded_nodes_mask" not in batch, "padded_nodes_mask should not exist in batch when multitask is False"

        assert batch["x_0"].shape[1] == (config.dataloader.num_timesteps), f"{batch['x_0'].shape[1]} != {config.dataloader.num_timesteps}"
        target_coords: torch.Tensor = batch.pop("x_t")
        _ = batch.pop("v_t") if "v_t" in batch else None
        mask: torch.Tensor | None = batch.get("padded_nodes_mask", None)

        optimizer.zero_grad()

        if config.training.label_noise_std > 0.0:
            batch["x_0"], batch["v_0"], batch["concatenated_features"] = add_brownian_noise(
                batch["x_0"],
                batch["v_0"],
                batch["concatenated_features"],
                config.training.label_noise_std,
            )

        with autocast(device_type=str(config.training.device), dtype=config.training.amp_dtype, enabled=config.training.use_amp):
            pred_coords: torch.Tensor = model(batch)

            # Calculate MSE loss
            assert pred_coords.shape == target_coords.shape, f"{pred_coords.shape} != {target_coords.shape}"

            # Do not compute gradients for heavy atoms if explicit_hydrogen is True and explicit_hydrogen_gradients is False
            if config.dataloader.explicit_hydrogen and config.dataloader.explicit_hydrogen_gradients is False:
                heavy_atom_mask: torch.Tensor = batch["Z"][..., 0] > 1  # shape: [Batch, Time, Nodes]

                # Apply mask along the nodes dimension
                pred_heavy: torch.Tensor = pred_coords[heavy_atom_mask]  # shape: [Total_selected_nodes, 3]
                target_heavy: torch.Tensor = target_coords[heavy_atom_mask]  # shape: [Total_selected_nodes, 3]

                loss = F.mse_loss(pred_heavy, target_heavy)
            # Compute gradients for all atoms (heavy and hydrogen)
            else:
                # Compute element‐wise MSE loss without reduction.
                loss_raw = F.mse_loss(pred_coords, target_coords, reduction="none")
                # Mask is [B,T,N,H]
                # Apply mask and compute average loss over valid nodes.
                if mask is not None:
                    mask_expanded = mask.expand_as(loss_raw)
                    loss = (loss_raw * mask_expanded).sum() / mask_expanded.sum()
                else:
                    loss = loss_raw.mean()

        total_s2t_loss += loss.item() * batch.batch_size[0]

        _ = scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    return total_s2t_loss / float(len(dataloader.dataset))


def eval_epoch(
    config: Config,
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]] | DataLoader[MD17DynamicsDataset],
) -> tuple[float, float]:
    """Evaluation loop.

    Args:
        config (Config): The configuration file.
        model (nn.Module): The model to evaluate.
        loader (DataLoader[dict[str, torch.Tensor]]): The dataloader to use.

    Returns:
        tuple[float, float]: The S2T and S2S loss of the epoch.
    """
    model.eval()
    total_s2t_loss = 0.0
    total_s2s_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch: TensorDict = TensorDict.from_dict(batch, device=torch.device(config.training.device), auto_batch_size=True)
            target_coords: torch.Tensor = batch.pop(key="x_t")
            _ = batch.pop("v_t") if "v_t" in batch else None
            mask: torch.Tensor | None = batch.get("padded_nodes_mask", None)

            pred_coords: torch.Tensor = model(batch)

            if config.dataloader.explicit_hydrogen and config.dataloader.explicit_hydrogen_gradients is False:
                # Get atomic numbers Z from batch and create mask for heavy atoms (Z > 1)
                heavy_atom_mask_s2t: torch.Tensor = batch["Z"][..., 0] > 1  # shape: [Batch, Time, Nodes]
                pred_heavy_s2t: torch.Tensor = pred_coords[heavy_atom_mask_s2t]  # shape: [Total_selected_nodes, 3]
                target_heavy_s2t: torch.Tensor = target_coords[heavy_atom_mask_s2t]  # shape: [Total_selected_nodes, 3]
                s2t_loss = F.mse_loss(pred_heavy_s2t, target_heavy_s2t)
                total_s2t_loss += s2t_loss.item() * batch.batch_size[0]

                pred_last_t = pred_coords[:, -1, :, :]  # [B, N, 3]
                target_last_t = target_coords[:, -1, :, :]  # [B, N, 3]
                heavy_atom_mask_s2s: torch.Tensor = batch["Z"][:, -1, :, 0] > 1  # [B, N]
                pred_heavy_s2s: torch.Tensor = pred_last_t[heavy_atom_mask_s2s]  # [Total_selected_nodes, 3]
                target_heavy_s2s: torch.Tensor = target_last_t[heavy_atom_mask_s2s]  # [Total_selected_nodes, 3]
                s2s_loss = F.mse_loss(pred_heavy_s2s, target_heavy_s2s)
                total_s2s_loss += s2s_loss.item() * batch.batch_size[0]
            else:
                # For the full coordinates loss (shape: [batch, 8, 20, 4])
                loss_raw_s2t = F.mse_loss(pred_coords, target_coords, reduction="none")
                if mask is not None:
                    mask_expanded_s2t = mask.expand_as(loss_raw_s2t)
                    s2t_loss = (loss_raw_s2t * mask_expanded_s2t).sum() / mask_expanded_s2t.sum()
                else:
                    s2t_loss = loss_raw_s2t.mean()

                # For the last slice loss (shape: [batch, 20, 4])
                loss_raw_s2s = F.mse_loss(pred_coords[:, -1, :, :], target_coords[:, -1, :, :], reduction="none")
                if mask is not None:
                    mask_last = mask[:, -1, :]  # Shape: [B, N, 1]
                    mask_last = mask_last.expand_as(loss_raw_s2s)  # Now shape: [B, N, 3]
                    s2s_loss = (loss_raw_s2s * mask_last).sum() / mask_last.sum()
                else:
                    s2s_loss = loss_raw_s2s.mean()

                total_s2t_loss += s2t_loss.item() * batch.batch_size[0]
                total_s2s_loss += s2s_loss.item() * batch.batch_size[0]

    return total_s2t_loss / len(loader.dataset), total_s2s_loss / len(loader.dataset)
