from tqdm.std import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime
from gtno_py.training.config_options import MD17MoleculeType, RMD17MoleculeType
import wandb
from gtno_py.utils import log_weights
import os
from pathlib import Path

from gtno_py.training import (
    Config,
    SingleRunResults,
    MultiRunResults,
    train_epoch,
    eval_epoch,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    create_dataloaders_single,
    create_dataloaders_multitask,
    set_seeds,
    reset_weights,
)

config = Config.from_toml("config.toml")

project_name = input("Enter project name: ")

_ = wandb.init(project="GTNO", name=project_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")

# Set device and seeds
device = torch.device(config.training.device)
set_seeds(config.training.seed)


noise_scale: float | torch.nn.Parameter
if config.training.learnable_noise_std:
    noise_scale = torch.nn.Parameter(torch.tensor(config.training.brownian_noise_std, device=device))
else:
    noise_scale = config.training.brownian_noise_std


def main(model: nn.Module, molecule_type: MD17MoleculeType | RMD17MoleculeType | None, benchmark_dir: Path, run_number: int) -> SingleRunResults:
    """Full training pipeline."""
    start_time = datetime.now()

    # Initialize components
    if config.dataloader.multitask and molecule_type is None:
        train_loader, val_loader, test_loader = create_dataloaders_multitask(config)
    elif molecule_type is not None:
        train_loader, val_loader, test_loader = create_dataloaders_single(config, molecule_type)
    else:
        raise ValueError("molecule_type must be provided for single-task dataloaders")
    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer)

    # Create a temporary directory for this run
    run_dir = benchmark_dir / f"run_{run_number+1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_model = run_dir / "best_val_model.pth"

    # Training loop
    best_val_loss = float("inf")
    best_val_loss_epoch = 0

    start_training_time = datetime.now()
    progress_bar = tqdm(range(config.training.epochs), desc="Training", leave=False, unit="epoch", position=2)
    for epoch in progress_bar:
        train_s2t_loss = train_epoch(
            config,
            model,
            optimizer,
            train_loader,
            scheduler,
        )
        val_s2t_loss, _ = eval_epoch(config, model, val_loader)

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
                "Current LR": f"{optimizer.param_groups[0]['lr']*100:.4f}x10^-2",
            }
        )
    end_training_time = datetime.now()

    # Final evaluation
    _ = model.load_state_dict(torch.load(best_val_model, weights_only=True))
    s2t_test_loss, s2s_test_loss = eval_epoch(config, model, test_loader)

    total_time = (datetime.now() - start_time).total_seconds()
    reset_weights(model)

    results = SingleRunResults(
        s2t_test_loss=s2t_test_loss,
        s2s_test_loss=s2s_test_loss,
        best_val_loss_epoch=best_val_loss_epoch,
        run_time=total_time,
        seconds_per_epoch=total_time / config.training.epochs,
        start_time=start_training_time,
        end_time=end_training_time,
        model_path=Path(best_val_model),
    )

    return results


def singletask_benchmark(config: Config) -> None:
    """
    Benchmarking function with JSON results logging.

    Args:
        runs: Number of runs to perform
        epochs_per_run: Number of epochs to run per run
        molecule_type: Molecule type to run on

    Returns:
        None
    """

    if isinstance(config.dataloader.molecule_type, list):
        molecules = config.dataloader.molecule_type
    else:
        molecules = [config.dataloader.molecule_type]

    if config.benchmark.compile:
        model = torch.compile(initialize_model(config).to(device), dynamic=True)
    else:
        model = initialize_model(config).to(device)
    tqdm.write(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    tqdm.write(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    molecule_progress_bar: tqdm[MD17MoleculeType | RMD17MoleculeType] = tqdm(molecules, leave=False, unit="molecule", position=0)
    for molecule in molecule_progress_bar:
        molecule_progress_bar.set_description(f"Running {str(molecule)}")

        # Create a directory for this molecule's benchmark
        timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
        benchmark_dir = Path(f"benchmark_runs/{project_name}_{str(molecule)}_{timestamp}")
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        run_results: list[SingleRunResults] = []

        runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
        for run in runs_progress_bar:
            runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")

            # Pass the weights directory to main function
            run_results.append(main(model, molecule, benchmark_dir=benchmark_dir, run_number=run))

        multi_run_results = MultiRunResults(single_run_results=run_results, config=config)

        # Save to JSON
        multi_run_results_json = multi_run_results.model_dump_json(indent=2)
        results_filename = f"{benchmark_dir}/results.json"
        with open(results_filename, "w") as f:
            f.write(multi_run_results_json)

        wandb.log(
            {
                "mean_test_loss": multi_run_results.s2s_test_loss_mean,
                "mean_test_loss_final": multi_run_results.s2s_test_loss_mean,
                "mean_secs_per_run": multi_run_results.mean_secs_per_run,
                "mean_secs_per_epoch": multi_run_results.mean_secs_per_epoch,
            }
        )

        tqdm.write(f"\nSaved benchmark results to {results_filename}")
        tqdm.write(f"Benchmark Results ({config.benchmark.runs} runs, {config.training.epochs} epochs/run):")
        tqdm.write(f"  Average S2S Test Loss Final Timestep: {multi_run_results.s2s_test_loss_mean:.4f} ± {multi_run_results.s2s_test_loss_std:.4f}")
        tqdm.write(f"  Average S2T Test Loss: {multi_run_results.s2t_test_loss_mean:.4f} ± {multi_run_results.s2t_test_loss_std:.4f}")
        tqdm.write(f"  Average Time per Run: {multi_run_results.mean_secs_per_run:.1f}s")
        tqdm.write(f"  Average Time per Epoch: {multi_run_results.mean_secs_per_epoch:.1f}s")
        tqdm.write(f"  Average Best Val Loss Epoch: {multi_run_results.mean_best_val_loss_epoch:.1f}")


def multitask_benchmark(config: Config) -> None:
    if config.benchmark.compile:
        model = torch.compile(initialize_model(config).to(device), dynamic=True)
    else:
        model = initialize_model(config).to(device)
    tqdm.write(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    tqdm.write(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create a directory for this molecule's benchmark
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    benchmark_dir = Path(f"benchmark_runs/{project_name}_multitask_{timestamp}")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    run_results: list[SingleRunResults] = []

    runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
    for run in runs_progress_bar:
        runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")

        # Pass the weights directory to main function
        run_results.append(main(model, None, benchmark_dir, run))

    multi_run_results = MultiRunResults(single_run_results=run_results, config=config)

    # Save to JSON
    multi_run_results_json = multi_run_results.model_dump_json(indent=2)
    results_filename = f"{benchmark_dir}/results.json"
    with open(results_filename, "w") as f:
        f.write(multi_run_results_json)

    wandb.log(
        {
            "mean_test_loss": multi_run_results.s2s_test_loss_mean,
            "mean_test_loss_final": multi_run_results.s2s_test_loss_mean,
            "mean_secs_per_run": multi_run_results.mean_secs_per_run,
            "mean_secs_per_epoch": multi_run_results.mean_secs_per_epoch,
        }
    )

    tqdm.write(f"\nSaved benchmark results to {results_filename}")
    tqdm.write(f"Benchmark Results ({config.benchmark.runs} runs, {config.training.epochs} epochs/run):")
    tqdm.write(f"  Average Test Loss: {multi_run_results.s2s_test_loss_mean:.4f} ± {multi_run_results.s2s_test_loss_std:.4f}")
    tqdm.write(f"  Average Test Loss Final Timestep: {multi_run_results.s2s_test_loss_mean:.4f} ± {multi_run_results.s2s_test_loss_std:.4f}")
    tqdm.write(f"  Average Time per Run: {multi_run_results.mean_secs_per_run:.1f}s")
    tqdm.write(f"  Average Time per Epoch: {multi_run_results.mean_secs_per_epoch:.1f}s")
    tqdm.write(f"  Average Best Val Loss Epoch: {multi_run_results.mean_best_val_loss_epoch:.1f}")


if __name__ == "__main__":
    if config.dataloader.multitask:
        multitask_benchmark(config)
    else:
        singletask_benchmark(config)
