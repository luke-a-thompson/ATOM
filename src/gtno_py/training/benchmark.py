from datetime import datetime
from pathlib import Path

import torch
from tqdm.std import tqdm
import wandb

from gtno_py.training import (
    Config,
    set_seeds,
    MD17MoleculeType,
    MultiRunResults,
    RMD17MoleculeType,
    SingleRunResults,
    initialize_model,
    train_model,
)


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
    # Create a directory for this molecule's benchmark
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    benchmark_dir = Path(f"benchmark_runs/{config.benchmark.benchmark_name}_singletask_{timestamp}")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    run_results: list[SingleRunResults] = []

    runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
    for run in runs_progress_bar:
        set_seeds(config.training.seed + run)
        runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")
        model = initialize_model(config).to(config.training.device)
        if config.benchmark.compile:
            model = torch.compile(model)

        # Pass the weights directory to main function
        single_run_results = train_model(
            config,
            model,
            benchmark_dir=benchmark_dir,
            run_number=run,
        )
        run_results.append(single_run_results)

    multi_run_results = MultiRunResults(single_run_results=run_results, config=config)

    # Save to JSON
    multi_run_results_json = multi_run_results.model_dump_json(
        indent=2,
        exclude={
            # We don't care about multitask options when our model is single task
            "config": {"training": {"device"}, "dataloader": {"train_molecules", "validation_molecules", "test_molecules"}},
            "single_run_results": {"__all__": {"device"}},
        },
    )
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
    tqdm.write(f"  Average S2S Test Loss Final Timestep: {multi_run_results.s2s_test_loss_mean*100:.2f}x10^-2 ± {multi_run_results.s2s_test_loss_std*100:.2f}x10^-2")  # type: ignore
    tqdm.write(f"  Average S2T Test Loss: {multi_run_results.s2t_test_loss_mean*100:.2f}x10^-2 ± {multi_run_results.s2t_test_loss_std*100:.2f}x10^-2")  # type: ignore
    tqdm.write(f"  Average Time per Run: {multi_run_results.mean_secs_per_run:.1f}x10^-2s")
    tqdm.write(f"  Average Time per Epoch: {multi_run_results.mean_secs_per_epoch:.1f}x10^-2s")
    tqdm.write(f"  Average Best Val Loss Epoch: {multi_run_results.mean_best_val_loss_epoch:.1f}")
    tqdm.write(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    tqdm.write(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def multitask_benchmark(config: Config) -> None:
    # Create a directory for this molecule's benchmark
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    benchmark_dir = Path(f"benchmark_runs/{config.benchmark.benchmark_name}_multitask_{timestamp}")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    run_results: list[SingleRunResults] = []

    runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
    for run in runs_progress_bar:
        set_seeds(config.training.seed + run)
        runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")
        model = initialize_model(config).to(config.training.device)
        if config.benchmark.compile:
            model = torch.compile(model)

        # Pass the weights directory to main function
        single_run_results = train_model(
            config,
            model,
            benchmark_dir,
            run,
        )
        run_results.append(single_run_results)

    multi_run_results = MultiRunResults(single_run_results=run_results, config=config)

    # Save to JSON
    multi_run_results_json = multi_run_results.model_dump_json(
        indent=2,
        exclude={
            "config": {"training": {"device"}, "dataloader": {"molecule_type"}},
            "single_run_results": {"__all__": {"device"}},
        },
    )
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
    tqdm.write(f"  Average Test Loss: {multi_run_results.s2s_test_loss_mean*100:.2f}x10^-2 ± {multi_run_results.s2s_test_loss_std*100:.2f}x10^-2")
    tqdm.write(f"  Average Test Loss Final Timestep: {multi_run_results.s2s_test_loss_mean*100:.2f}x10^-2 ± {multi_run_results.s2s_test_loss_std*100:.2f}x10^-2")
    tqdm.write(f"  Average Time per Run: {multi_run_results.mean_secs_per_run:.1f}s")
    tqdm.write(f"  Average Time per Epoch: {multi_run_results.mean_secs_per_epoch:.1f}s")
    tqdm.write(f"  Average Best Val Loss Epoch: {multi_run_results.mean_best_val_loss_epoch:.1f}")
    tqdm.write(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    tqdm.write(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
