from datetime import datetime
from pathlib import Path
from shutil import copy2
import shutil
from tempfile import TemporaryDirectory

import torch
from tqdm.std import tqdm
import wandb

from atom.training import (
    Config,
    set_seeds,
    MultiRunResults,
    SingleRunResults,
    initialize_model,
    train_model,
)


def singletask_benchmark(
    config: Config,
    config_path: Path | None = None,
    benchmark_dir: Path | None = None,
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
    # Determine final destination directory but do not create it yet
    if benchmark_dir is None:
        timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
        benchmark_dir = Path(f"benchmark_runs/{config.benchmark.benchmark_name}_singletask_{timestamp}")

    created_final_dir: bool = False
    single_run_results: list[SingleRunResults] = []

    runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
    with TemporaryDirectory() as tmp_root_str:
        tmp_root: Path = Path(tmp_root_str)
        for run in runs_progress_bar:
            set_seeds(config.training.seed + run)
            runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")
            model = initialize_model(config).to(config.training.device)
            if config.benchmark.compile:
                model = torch.compile(model)

            # Train into a temporary location first
            single_run_result = train_model(
                config,
                model,
                tmp_root,
                run,
            )
            single_run_results.append(single_run_result)

            # On first successful run completion, create final dir and copy config(s)
            if not created_final_dir:
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                if config_path is not None and config_path.exists():
                    try:
                        # Name the inner config after the experiment directory
                        inner_config_path: Path = benchmark_dir / f"{benchmark_dir.name}.toml"
                        _ = copy2(config_path, inner_config_path)
                    except Exception as e:
                        tqdm.write(f"Warning: failed to copy config TOML: {e}")
                created_final_dir = True

            # Move the completed run directory into the final benchmark dir
            try:
                shutil.move(str(tmp_root / f"run_{run+1}"), str(benchmark_dir / f"run_{run+1}"))
            except Exception as e:
                tqdm.write(f"Warning: failed to move run directory for run {run+1}: {e}")

    # If no runs completed successfully, avoid creating output
    if not created_final_dir:
        return None

    multi_run_results = MultiRunResults(single_run_results=single_run_results, config=config)

    # Save to JSON
    multi_run_results_json = multi_run_results.model_dump_json(
        indent=2,
        exclude={
            # We don't care about multitask options when our model is single task
            "config": {"training": {"device", "use_amp", "amp_dtype"}, "dataloader": {"train_molecules", "validation_molecules", "test_molecules"}},
            "single_run_results": {"__all__": {"device"}},
        },
    )
    results_filename = f"{benchmark_dir}/results.json"
    with open(results_filename, "w") as f:
        _ = f.write(multi_run_results_json)

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


def multitask_benchmark(
    config: Config,
    config_path: Path | None = None,
    benchmark_dir: Path | None = None,
) -> None:
    # Determine final destination directory but do not create it yet
    if benchmark_dir is None:
        timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
        benchmark_dir = Path(f"benchmark_runs/{config.benchmark.benchmark_name}_multitask_{timestamp}")

    created_final_dir: bool = False
    run_results: list[SingleRunResults] = []

    runs_progress_bar = tqdm(range(config.benchmark.runs), leave=False, unit="run", position=1)
    with TemporaryDirectory() as tmp_root_str:
        tmp_root: Path = Path(tmp_root_str)
        for run in runs_progress_bar:
            set_seeds(config.training.seed + run)
            runs_progress_bar.set_description(f"Run {run+1}/{config.benchmark.runs}")
            model = initialize_model(config).to(config.training.device)
        # Calculate and print model parameter counts
        total_params: int = sum(p.numel() for p in model.parameters())
        trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tqdm.write(f"Total params: {total_params:,}")
        tqdm.write(f"Total trainable params: {trainable_params:,}")

        from atom.atom.atom_model import ATOM

        if isinstance(model, ATOM):
            if model.output_heads > 1:
                gating_params = sum(p.numel() for p in model.weight_pred_gate_net.parameters())
                single_expert_params = sum(p.numel() for p in model.projection_layers[0].parameters())
                active_params = gating_params + single_expert_params
                total_expert_params = sum(p.numel() for p in model.projection_layers.parameters())

                tqdm.write("\n--- MoE Expert Layer Analysis ---")
                tqdm.write(f"Gating network params: {gating_params:,}")
                tqdm.write(f"Params per expert: {single_expert_params:,}")
                tqdm.write(f"Total params for all experts: {total_expert_params:,}")
                tqdm.write(f"Active params per inference (gate + 1 expert): {active_params:,}")
                tqdm.write(f"A non-MoE model would have {single_expert_params:,} params in its final projection layer.")
                tqdm.write("------------------------------------")
            else:
                if hasattr(model, "projection_layer"):
                    projection_params = sum(p.numel() for p in model.projection_layer.parameters())
                    tqdm.write("\n--- Non-MoE Model ---")
                    tqdm.write(f"Final projection layer params: {projection_params:,}")
                    tqdm.write("-----------------------")
            # Train into a temporary location first
            single_run_results = train_model(
                config,
                model,
                tmp_root,
                run,
            )
            run_results.append(single_run_results)

            # On first successful run completion, create final dir and copy config(s)
            if not created_final_dir:
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                if config_path is not None and config_path.exists():
                    try:
                        inner_config_path = benchmark_dir / f"{benchmark_dir.name}.toml"
                        _ = copy2(config_path, inner_config_path)
                    except Exception as e:
                        tqdm.write(f"Warning: failed to copy config TOML: {e}")
                created_final_dir = True

            # Move the completed run directory into the final benchmark dir
            try:
                shutil.move(str(tmp_root / f"run_{run+1}"), str(benchmark_dir / f"run_{run+1}"))
            except Exception as e:
                tqdm.write(f"Warning: failed to move run directory for run {run+1}: {e}")

    # If no runs completed successfully, avoid creating output
    if not created_final_dir:
        return None

    multi_run_results = MultiRunResults(single_run_results=run_results, config=config)

    # Save to JSON
    multi_run_results_json = multi_run_results.model_dump_json(
        indent=2,
        exclude={
            "config": {"training": {"device", "use_amp", "amp_dtype"}, "dataloader": {"molecule_type"}},
            "single_run_results": {"__all__": {"device"}},
        },
    )
    results_filename = f"{benchmark_dir}/results.json"
    with open(results_filename, "w") as f:
        _ = f.write(multi_run_results_json)

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
