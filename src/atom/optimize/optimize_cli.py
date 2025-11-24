from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, cast
import gc

import optuna
import torch
from torch.amp.grad_scaler import GradScaler
from tqdm.std import tqdm

from atom.training import (
    Config,
    initialize_model,
    create_dataloaders_single,
    create_dataloaders_multitask,
    initialize_optimizer,
    initialize_scheduler,
)
from atom.training.train_pipeline import eval_epoch, train_epoch
from atom.training.training_utils import set_seeds


def _trial_to_config(base_config: Config, trial: optuna.trial.Trial) -> Config:
    """Clone base config and apply trial-sampled hyperparameters.

    Returns a validated Config with safe-to-tune fields updated.
    """
    cfg: dict[str, object] = base_config.model_dump()

    dataloader_cfg: dict[str, object] = cast(dict[str, object], cfg["dataloader"])  # type: ignore[index]
    training_cfg: dict[str, object] = cast(dict[str, object], cfg["training"])  # type: ignore[index]
    optimizer_cfg: dict[str, object] = cast(dict[str, object], cfg["optimizer"])  # type: ignore[index]
    atom_cfg: dict[str, object] = cast(dict[str, object], cfg["atom_config"])  # type: ignore[index]

    # Training
    training_cfg["label_noise_std"] = trial.suggest_float("label_noise_std", 0.001, 0.2, log=True)

    # Optimizer
    optimizer_cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)

    # Model architecture
    atom_cfg["num_layers"] = trial.suggest_int("num_layers", 5, 8)
    # Allow heads to vary from 4 to 12
    atom_cfg["num_heads"] = trial.suggest_int("num_heads", 4, 12)
    heads: int = int(atom_cfg["num_heads"])  # type: ignore[index]

    # Constrain lifting_dim to be divisible by num_heads and have an even d_head
    # d_head = lifting_dim // heads, require d_head % 2 == 0 → lifting_dim % (2 * heads) == 0
    max_lifting: int = 512
    divisible_by: int = 2 * max(1, heads)
    min_lifting: int = divisible_by
    # Round max down to nearest multiple of divisible_by to satisfy Optuna step grid
    if max_lifting % divisible_by != 0:
        max_lifting = max_lifting - (max_lifting % divisible_by)
    lifting_dim: int = trial.suggest_int("lifting_dim", min_lifting, max_lifting, step=divisible_by)
    atom_cfg["lifting_dim"] = lifting_dim
    atom_cfg["delta_update"] = trial.suggest_categorical("delta_update", [True, False])

    return Config.model_validate(cfg)


def _objective_factory(base_config: Config) -> Callable[[optuna.trial.Trial], float]:
    def objective(trial: optuna.trial.Trial) -> float:
        # Vary seed per trial for robustness
        set_seeds(base_config.training.seed + int(trial.number))

        # Build per-trial config
        config: Config = _trial_to_config(base_config, trial)

        # Cap epochs for faster HPO while respecting user's setting
        n_epochs: int = max(1, min(base_config.training.epochs, 250))

        train_loader = None
        val_loader = None
        model = None
        optimizer = None
        scheduler = None
        scaler = None

        try:
            # Data loaders - always use base config's multitask flag
            if base_config.dataloader.multitask:
                train_loader, val_loader, _ = create_dataloaders_multitask(config)
            else:
                train_loader, val_loader, _ = create_dataloaders_single(config)

            # Model and optimization
            device: torch.device = config.training.device
            model = initialize_model(config).to(device)
            optimizer = initialize_optimizer(config, model)
            scheduler = initialize_scheduler(config, optimizer)
            scaler = GradScaler(enabled=bool(config.training.use_amp))

            best_val: float = float("inf")
            epoch_bar = tqdm(range(n_epochs), desc=f"Trial {trial.number}", leave=False, unit="epoch")
            for epoch_index in epoch_bar:
                _ = train_epoch(config, model, optimizer, train_loader, scheduler, scaler)
                val_s2t, _ = eval_epoch(config, model, val_loader)
                if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_s2t)

                # Report for pruning
                trial.report(val_s2t, epoch_index)
                if trial.should_prune():
                    epoch_bar.close()
                    raise optuna.TrialPruned()

                if val_s2t < best_val:
                    best_val = val_s2t

                epoch_bar.set_postfix({"val_s2t": f"{val_s2t:.5f}", "best": f"{best_val:.5f}"})
            epoch_bar.close()

            return best_val

        except torch.cuda.OutOfMemoryError:
            # Mark trial as failed; cleanup below in finally
            raise
        except RuntimeError as runtime_error:
            # Catch CUDA OOMs that manifest as RuntimeError or other transient training issues
            error_message: str = str(runtime_error).lower()
            if "out of memory" in error_message or "cuda error" in error_message or "cublas" in error_message:
                raise
            # Re-raise other runtime errors to be handled by Optuna catch
            raise
        except ValueError:
            # Bad hyperparameter combos etc.
            raise
        finally:
            # Free GPU/CPU memory between trials to avoid leaks that break subsequent trials
            try:
                del model, optimizer, scheduler, scaler, train_loader, val_loader
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            gc.collect()

    return objective


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ATOM using Optuna")
    _ = parser.add_argument("--config", type=str, required=True, help="Path to base config TOML")
    _ = parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    _ = parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna.db",
        help="Optuna storage URL (e.g. sqlite:///optuna.db)",
    )
    _ = parser.add_argument("--study", type=str, default="atom-optimize", help="Study name")
    _ = parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        default="minimize",
        help="Optimization direction",
    )
    _ = parser.add_argument(
        "--save",
        type=str,
        default="optuna_results.json",
        help="Where to save study summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path_str: str = str(args.config)
    base_config_path: Path = Path(config_path_str).expanduser().resolve()
    base_config: Config = Config.from_toml(base_config_path)

    objective = _objective_factory(base_config)

    storage_url: str = str(args.storage)
    study_name: str = str(args.study)
    direction: str = str(args.direction)

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=10,
        interval_steps=1,
    )

    if storage_url:
        study = optuna.create_study(
            storage=storage_url,
            load_if_exists=True,
            study_name=study_name,
            direction=direction,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(direction=direction, study_name=study_name, pruner=pruner)

    trials_count: int = int(args.trials)
    # Continue after common errors; failing trials are recorded without aborting the study
    study.optimize(
        objective,
        n_trials=trials_count,
        catch=(
            RuntimeError,
            ValueError,
            MemoryError,
            torch.cuda.OutOfMemoryError,
        ),
    )

    best_trial = study.best_trial
    result: dict[str, object] = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "n_trials": len(study.trials),
        "direction": direction,
        "study_name": study_name,
    }
    save_path_str: str = str(args.save)
    with open(save_path_str, "w") as f:
        _ = json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
