import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime
from gtno_py.dataloaders.egno_dataloder import MD17MoleculeType, MD17Version, RMD17MoleculeType
from typing import Literal
import wandb
from gtno_py.utils import log_weights
import os

from gtno_py.training import (
    Config,
    train_epoch,
    eval_epoch,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    create_dataloaders_single,
    set_seeds,
    reset_weights,
)

config = Config.from_toml("config.toml")

project_name = input("Enter project name: ")

_ = wandb.init(project="GTNO", name=project_name, config=dict(config), mode="disabled" if not config.wandb.use_wandb else "online")

# Set device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seeds(config.training.seed)


noise_scale: float | torch.nn.Parameter
if config.training.learnable_noise_std:
    noise_scale = torch.nn.Parameter(torch.tensor(config.training.brownian_noise_std, device=device))
else:
    noise_scale = config.training.brownian_noise_std


def main(num_epochs: int, model: nn.Module, molecule_type: MD17MoleculeType | RMD17MoleculeType, weights_dir: str | None = None) -> tuple[float, float, float, int, str]:
    """Full training pipeline."""
    start_time = datetime.now()

    # Initialize components
    train_loader, val_loader, test_loader = create_dataloaders_single(config, molecule_type)
    optimizer = initialize_optimizer(config, model)
    scheduler = initialize_scheduler(config, optimizer)

    # Create a temporary directory for this run
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    temp_model_path = f"benchmark_runs/temp_model_{timestamp}.pth"

    # Training loop
    best_val_loss = float("inf")
    best_val_loss_epoch = 0

    progress_bar = tqdm(range(num_epochs), desc="Training", leave=False, unit="epoch", position=2)
    for epoch in progress_bar:
        train_loss = train_epoch(config, model, optimizer, train_loader, scheduler)
        val_loss, _ = eval_epoch(config, model, val_loader)

        # Log gate parameters and save to weights_dir if provided
        if config.benchmark.log_weights:
            log_weights(model.named_parameters(), epoch, save_dir=weights_dir)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        # if val_loss < best_val_loss and epoch > 0.5 * num_epochs:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            torch.save(model.state_dict(), temp_model_path)

        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Update progress bar with losses
        progress_bar.set_postfix(
            {
                "Train loss": f"{train_loss:.4f}",
                "Val loss": f"{val_loss:.4f}",
                "Best val loss": f"{best_val_loss:.4f}",
                "Current LR": f"{optimizer.param_groups[0]['lr']:.6f}",
            }
        )

    # Final evaluation
    _ = model.load_state_dict(torch.load(temp_model_path, weights_only=True))
    s2t_test_loss, s2s_test_loss = eval_epoch(config, model, test_loader)

    total_time = (datetime.now() - start_time).total_seconds()
    reset_weights(model)
    return s2t_test_loss, s2s_test_loss, total_time, best_val_loss_epoch, temp_model_path


def benchmark(
    config: Config,
) -> None:
    """
    Benchmarking function with JSON results logging.

    Args:
        runs: Number of runs to perform
        epochs_per_run: Number of epochs to run per run
        molecule_type: Molecule type to run on

    Returns:
        None
    """

    results = {
        "config_name": project_name,
        "runs": {},
        "summary": {},
        "timestamp": datetime.now().isoformat(),
    }

    if isinstance(config.benchmark.molecule_type, list):
        molecules = config.benchmark.molecule_type
    else:
        molecules = [config.benchmark.molecule_type]

    if config.benchmark.compile:
        model = torch.compile(initialize_model(config).to(device), dynamic=True)
    else:
        model = initialize_model(config).to(device)

    molecule_progress_bar = tqdm(molecules, leave=False, unit="molecule", position=0)
    for molecule in molecule_progress_bar:
        molecule_progress_bar.set_description(f"Running {str(molecule)}")

        # Create a directory for this molecule's benchmark
        timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
        benchmark_dir = f"benchmark_runs/{project_name}_{str(molecule)}_{timestamp}"
        os.makedirs(benchmark_dir, exist_ok=True)

        runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
        for run in runs_progress_bar:
            runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")
            start_time = datetime.now()

            # Create a run-specific weights directory before starting the run
            run_weights_dir = f"{benchmark_dir}/weights_run{run+1}"
            os.makedirs(run_weights_dir, exist_ok=True)

            # Pass the weights directory to main function
            s2t_test_loss, s2s_test_loss, duration, best_val_loss_epoch, temp_model_path = main(config.training.epochs, model, molecule, weights_dir=run_weights_dir)
            end_time = datetime.now()

            # Create a run-specific model file
            run_model_path = f"{benchmark_dir}/model_run{run+1}.pth"
            os.rename(temp_model_path, run_model_path)

            # Store run results
            results["runs"][f"run{run+1}"] = {
                "s2t_test_loss": float(s2t_test_loss),
                "s2s_test_loss_final": float(s2s_test_loss),
                "best_val_loss_epoch": float(best_val_loss_epoch),
                "time_seconds": float(duration),
                "time_per_epoch": float(duration / epochs_per_run),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "model_path": run_model_path,
            }

        # Calculate summary statistics
        test_losses = [res["s2t_test_loss"] for res in results["runs"].values()]
        test_losses_final = [res["s2s_test_loss_final"] for res in results["runs"].values()]
        times = [res["time_seconds"] for res in results["runs"].values()]
        times_per_epoch = [res["time_per_epoch"] for res in results["runs"].values()]
        best_val_loss_epochs = [res["best_val_loss_epoch"] for res in results["runs"].values()]
        results["summary"] = {
            "mean_test_loss": float(sum(test_losses) / len(test_losses)),
            "std_dev_test_loss": float(torch.std(torch.tensor(test_losses)).item()),
            "mean_test_loss_final": float(sum(test_losses_final) / len(test_losses_final)),
            "std_dev_test_loss_final": float(torch.std(torch.tensor(test_losses_final)).item()),
            "mean_secs_per_run": float(sum(times) / len(times)),
            "total_secs": float(sum(times)),
            "min_test_loss": float(min(test_losses)),
            "max_test_loss": float(max(test_losses)),
            "mean_secs_per_epoch": float(sum(times_per_epoch) / len(times_per_epoch)),
            "config": config,
            "best_val_loss_epochs": float(sum(best_val_loss_epochs) / len(best_val_loss_epochs)),
        }
        results["latex_format"] = {
            "S2S": f"\\({results['summary']['mean_test_loss_final']*100:.2f}{{\\scriptstyle \\pm{results['summary']['std_dev_test_loss_final']*100:.2f}}}\\)",
            "S2T": f"\\({results['summary']['mean_test_loss']*100:.2f}{{\\scriptstyle \\pm{results['summary']['std_dev_test_loss']*100:.2f}}}\\)",
        }

        wandb.log(
            {
                "mean_test_loss": results["summary"]["mean_test_loss"],
                "mean_test_loss_final": results["summary"]["mean_test_loss_final"],
                "mean_secs_per_run": results["summary"]["mean_secs_per_run"],
                "mean_secs_per_epoch": results["summary"]["mean_secs_per_epoch"],
            }
        )

        # Save to JSON
        results_filename = f"{benchmark_dir}/results.json"
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=2)

        tqdm.write(f"\nSaved benchmark results to {results_filename}")
        tqdm.write(f"Benchmark Results ({runs} runs, {epochs_per_run} epochs/run):")
        tqdm.write(f"  Average Test Loss: {results['summary']['mean_test_loss']:.4f} ± {results['summary']['std_dev_test_loss']:.4f}")
        tqdm.write(f"  Average Test Loss Final Timestep: {results['summary']['mean_test_loss_final']:.4f} ± {results['summary']['std_dev_test_loss_final']:.4f}")
        tqdm.write(f"  Average Time per Run: {results['summary']['mean_secs_per_run']:.1f}s")
        tqdm.write(f"  Total Benchmark Time: {results['summary']['total_secs']:.1f}s")
        tqdm.write(f"  Average Time per Epoch: {results['summary']['mean_secs_per_epoch']:.1f}s")
        tqdm.write(f"  Average Best Val Loss Epoch: {results['summary']['best_val_loss_epochs']:.1f}")


if __name__ == "__main__":
    benchmark(
        config,
    )
