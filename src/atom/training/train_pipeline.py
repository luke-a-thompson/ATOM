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

from atom.dataloaders.atom_dataloader import MDDynamicsDataset
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
from atom.training.config_options import TimeLagMode


def train_model(config: Config, model: nn.Module, benchmark_dir: Path, run_number: int) -> SingleRunResults:
    """Full training pipeline."""

    if config.dataloader.multitask:
        train_loader, val_loader, test_loader = create_dataloaders_multitask(config)
    else:
        train_loader, val_loader, test_loader = create_dataloaders_single(config)

    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer)
    torch.set_float32_matmul_precision("high")

    # Create a temporary directory for this run
    run_dir = benchmark_dir / f"run_{run_number+1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_model = run_dir / "best_val_model.pth"
    last_model = run_dir / "last_model.pth"

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

        # Always save latest model checkpoint for this epoch
        torch.save(model.state_dict(), last_model)

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
    dataloader: DataLoader[dict[str, torch.Tensor]] | DataLoader[MDDynamicsDataset],
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
    # Accumulate element-wise MSE across the entire epoch
    s2t_numerator: float = 0.0
    s2t_denominator: float = 0.0

    for batch in dataloader:
        batch = TensorDict.from_dict(batch, device=torch.device(config.training.device), auto_batch_size=True)
        if config.dataloader.multitask is False:
            assert "padded_nodes_mask" not in batch, "padded_nodes_mask should not exist in batch when multitask is False"

        assert batch["x_0"].shape[1] == (config.dataloader.num_timesteps), f"{batch['x_0'].shape[1]} != {config.dataloader.num_timesteps}"
        target_coords: torch.Tensor = batch.pop("x_t")
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
            outputs = model(batch)
            # Expect dict: {"pos": [B,T,N,3], "vel": [B,T,N,3], "energy": [B,T]}
            if isinstance(outputs, dict):
                pred_coords = outputs["pos"]
                pred_vel = outputs.get("vel")
                pred_energy = outputs.get("energy")
            else:
                pred_coords = outputs
                pred_vel = None
                pred_energy = None

            # Calculate MSE loss (positions only for backprop)
            assert pred_coords.shape == target_coords.shape, f"{pred_coords.shape} != {target_coords.shape}"

            # Do not compute gradients for heavy atoms if explicit_hydrogen is True and explicit_hydrogen_gradients is False
            if config.dataloader.explicit_hydrogen and config.dataloader.explicit_hydrogen_gradients is False:
                # Use all timesteps (both UNIFORM and LAST)
                heavy_atom_mask = batch["Z"][..., 0] > 1  # [B, T, N]
                pred_heavy = pred_coords[heavy_atom_mask]
                target_heavy = target_coords[heavy_atom_mask]

                loss = F.mse_loss(pred_heavy, target_heavy)
                if pred_vel is not None and "v_t" in batch:
                    pred_vel_heavy = pred_vel[heavy_atom_mask]
                    target_vel_heavy = batch["v_t"][heavy_atom_mask]
                    loss = loss + F.mse_loss(pred_vel_heavy, target_vel_heavy)
                if pred_energy is not None and "E_t" in batch:
                    loss = loss + F.mse_loss(pred_energy, batch["E_t"])  # [B,T]

                loss_raw_heavy = F.mse_loss(pred_heavy, target_heavy, reduction="none")
                s2t_numerator += loss_raw_heavy.sum().item()
                s2t_denominator += float(loss_raw_heavy.numel())
            else:
                # Element-wise MSE without reduction over all timesteps
                loss_raw = F.mse_loss(pred_coords, target_coords, reduction="none")
                loss_raw_used = loss_raw
                mask_used = batch.get("padded_nodes_mask", None)

                mol_ids = batch.get("molecule_id", None)
                if mol_ids is not None:
                    unique_ids, inv = torch.unique(mol_ids, return_inverse=True)
                    per_mol_losses: list[torch.Tensor] = []
                    for m in range(unique_ids.shape[0]):
                        sel = (inv == m).view(-1, 1, 1, 1)
                        if mask_used is not None:
                            mask_m = mask_used.expand_as(loss_raw_used) * sel
                            denom_m = mask_m.sum()
                            if denom_m > 0:
                                per_m = (loss_raw_used * mask_m).sum() / denom_m
                                per_mol_losses.append(per_m)
                        else:
                            count_m = (sel.sum() * loss_raw_used.shape[1] * loss_raw_used.shape[2] * loss_raw_used.shape[3]).to(loss_raw_used.dtype)
                            if count_m > 0:
                                per_m = (loss_raw_used * sel).sum() / count_m
                                per_mol_losses.append(per_m)
                    loss = torch.stack(per_mol_losses).mean() if per_mol_losses else loss_raw_used.mean()
                    s2t_numerator += float(sum(p.item() for p in per_mol_losses))
                    s2t_denominator += float(len(per_mol_losses))
                elif mask_used is not None:
                    mask_expanded = mask_used.expand_as(loss_raw_used)
                    loss = (loss_raw_used * mask_expanded).sum() / mask_expanded.sum()
                    s2t_numerator += (loss_raw_used * mask_expanded).sum().item()
                    s2t_denominator += mask_expanded.sum().item()
                    if pred_vel is not None and "v_t" in batch:
                        vel_loss_raw = F.mse_loss(pred_vel, batch["v_t"], reduction="none")
                        vel_loss = (vel_loss_raw * mask_used.expand_as(vel_loss_raw)).sum() / mask_used.expand_as(vel_loss_raw).sum()
                        loss = loss + vel_loss
                    if pred_energy is not None and "E_t" in batch:
                        loss = loss + F.mse_loss(pred_energy, batch["E_t"])  # [B,T]
                else:
                    loss = loss_raw_used.mean()
                    s2t_numerator += loss_raw_used.sum().item()
                    s2t_denominator += float(loss_raw_used.numel())
                    if pred_vel is not None and "v_t" in batch:
                        loss = loss + F.mse_loss(pred_vel, batch["v_t"])  # [B,T,N,3]
                    if pred_energy is not None and "E_t" in batch:
                        loss = loss + F.mse_loss(pred_energy, batch["E_t"])  # [B,T]

        _ = scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    # Element-weighted MSE across the entire epoch
    return (s2t_numerator / s2t_denominator) if s2t_denominator > 0 else 0.0


def eval_epoch(
    config: Config,
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]] | DataLoader[MDDynamicsDataset],
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
    # Accumulate per-molecule averaged losses across the entire split
    s2t_numerator: float = 0.0
    s2t_denominator: float = 0.0
    s2s_numerator: float = 0.0
    s2s_denominator: float = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = TensorDict.from_dict(batch, device=torch.device(config.training.device), auto_batch_size=True)
            target_coords: torch.Tensor = batch.pop("x_t")
            # _ = batch.pop("v_t") if "v_t" in batch else None
            mask: torch.Tensor | None = batch.get("padded_nodes_mask", None)

            outputs = model(batch)
            if isinstance(outputs, dict):
                pred_coords = outputs["pos"]
            else:
                pred_coords = outputs

            if config.dataloader.explicit_hydrogen and config.dataloader.explicit_hydrogen_gradients is False:
                # Get atomic numbers Z from batch and create mask for heavy atoms (Z > 1)
                heavy_atom_mask_s2t: torch.Tensor = batch["Z"][..., 0] > 1  # shape: [Batch, Time, Nodes]
                pred_heavy_s2t: torch.Tensor = pred_coords[heavy_atom_mask_s2t]  # shape: [Total_selected_nodes, 3]
                target_heavy_s2t: torch.Tensor = target_coords[heavy_atom_mask_s2t]  # shape: [Total_selected_nodes, 3]
                # Aggregate over all selected elements
                s2t_loss_raw = F.mse_loss(pred_heavy_s2t, target_heavy_s2t, reduction="none")
                s2t_numerator += s2t_loss_raw.sum().item()
                s2t_denominator += float(s2t_loss_raw.numel())

                pred_last_t = pred_coords[:, -1, :, :]  # [B, N, 3]
                target_last_t = target_coords[:, -1, :, :]  # [B, N, 3]
                heavy_atom_mask_s2s: torch.Tensor = batch["Z"][:, -1, :, 0] > 1  # [B, N]
                pred_heavy_s2s: torch.Tensor = pred_last_t[heavy_atom_mask_s2s]  # [Total_selected_nodes, 3]
                target_heavy_s2s: torch.Tensor = target_last_t[heavy_atom_mask_s2s]  # [Total_selected_nodes, 3]
                s2s_loss_raw = F.mse_loss(pred_heavy_s2s, target_heavy_s2s, reduction="none")
                s2s_numerator += s2s_loss_raw.sum().item()
                s2s_denominator += float(s2s_loss_raw.numel())
            else:
                # For the full coordinates loss (shape: [batch, T, N, 3])
                loss_raw_s2t = F.mse_loss(pred_coords, target_coords, reduction="none")
                mol_ids: torch.Tensor | None = batch.get("molecule_id", None)
                if mol_ids is not None:
                    unique_ids, inv = torch.unique(mol_ids, return_inverse=True)
                    per_mol_losses = []
                    for m in range(unique_ids.shape[0]):
                        sel = (inv == m).view(-1, 1, 1, 1)
                        if mask is not None:
                            mask_m = mask.expand_as(loss_raw_s2t) * sel
                            denom_m = mask_m.sum()
                            if denom_m > 0:
                                per_m = (loss_raw_s2t * mask_m).sum() / denom_m
                                per_mol_losses.append(per_m)
                        else:
                            count_m = (sel.sum() * loss_raw_s2t.shape[1] * loss_raw_s2t.shape[2] * loss_raw_s2t.shape[3]).to(loss_raw_s2t.dtype)
                            if count_m > 0:
                                per_m = (loss_raw_s2t * sel).sum() / count_m
                                per_mol_losses.append(per_m)
                    if per_mol_losses:
                        s2t_numerator += float(sum(p.item() for p in per_mol_losses))
                        s2t_denominator += float(len(per_mol_losses))
                    else:
                        s2t_numerator += loss_raw_s2t.mean().item()
                        s2t_denominator += 1.0
                else:
                    if mask is not None:
                        mask_expanded_s2t = mask.expand_as(loss_raw_s2t)
                        s2t_numerator += (loss_raw_s2t * mask_expanded_s2t).sum().item()
                        s2t_denominator += mask_expanded_s2t.sum().item()
                    else:
                        s2t_numerator += loss_raw_s2t.sum().item()
                        s2t_denominator += float(loss_raw_s2t.numel())

                # For the last slice loss (shape: [batch, N, 3])
                loss_raw_s2s = F.mse_loss(pred_coords[:, -1, :, :], target_coords[:, -1, :, :], reduction="none")
                if mol_ids is not None:
                    unique_ids2, inv2 = torch.unique(mol_ids, return_inverse=True)
                    per_mol_losses2 = []
                    for m in range(unique_ids2.shape[0]):
                        sel_b = (inv2 == m).view(-1, 1, 1)
                        if mask is not None:
                            mask_last = mask[:, -1, :].expand_as(loss_raw_s2s)
                            mask_m2 = mask_last * sel_b
                            denom_m2 = mask_m2.sum()
                            if denom_m2 > 0:
                                per_m2 = (loss_raw_s2s * mask_m2).sum() / denom_m2
                                per_mol_losses2.append(per_m2)
                        else:
                            count_m2 = (sel_b.sum() * loss_raw_s2s.shape[1] * loss_raw_s2s.shape[2]).to(loss_raw_s2s.dtype)
                            if count_m2 > 0:
                                per_m2 = (loss_raw_s2s * sel_b).sum() / count_m2
                                per_mol_losses2.append(per_m2)
                    if per_mol_losses2:
                        s2s_numerator += float(sum(p.item() for p in per_mol_losses2))
                        s2s_denominator += float(len(per_mol_losses2))
                    else:
                        s2s_numerator += loss_raw_s2s.mean().item()
                        s2s_denominator += 1.0
                else:
                    if mask is not None:
                        mask_last = mask[:, -1, :]  # Shape: [B, N, 1]
                        mask_last = mask_last.expand_as(loss_raw_s2s)  # Now shape: [B, N, 3]
                        s2s_numerator += (loss_raw_s2s * mask_last).sum().item()
                        s2s_denominator += mask_last.sum().item()
                    else:
                        s2s_numerator += loss_raw_s2s.sum().item()
                        s2s_denominator += float(loss_raw_s2s.numel())

    s2t = (s2t_numerator / s2t_denominator) if s2t_denominator > 0 else 0.0
    s2s = (s2s_numerator / s2s_denominator) if s2s_denominator > 0 else 0.0
    return s2t, s2s
