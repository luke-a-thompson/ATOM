import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_optimizer as pt_optim
from gtno_py.dataloaders.egno_dataloder import MD17DynamicsDataset, DataPartition
from torch.utils.data import DataLoader
from tqdm import tqdm
import tomllib
from gtno_py.gtno.gtno_model import GTNO
import torch.nn.functional as F
import json
from datetime import datetime
from gtno_py.dataloaders.egno_dataloder import MoleculeType
from typing import Literal
import wandb
from gtno_py.utils import log_feature_weights

# Load configuration
with open("config.toml", "rb") as file:
    config = tomllib.load(file)

project_name = input("Enter project name: ")

wandb.init(project="GTNO", name=project_name, config=config, mode="disabled" if not config["wandb"]["use_wandb"] else "online")

# Set device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["training"]["seed"])
torch.cuda.manual_seed(config["training"]["seed"])


def create_dataloaders(molecule_type: MoleculeType) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Create train/val/test dataloaders."""
    train_dataset = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=3000,
        num_timesteps=config["model"]["num_timesteps"],
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=molecule_type,
    )

    val_dataset = MD17DynamicsDataset(
        partition=DataPartition.val,
        max_samples=2000,
        delta_frame=3000,
        num_timesteps=config["model"]["num_timesteps"],
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=molecule_type,
    )

    test_dataset = MD17DynamicsDataset(
        partition=DataPartition.test,
        max_samples=2000,
        delta_frame=3000,
        num_timesteps=config["model"]["num_timesteps"],
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=molecule_type,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        persistent_workers=config["dataloader"]["persistent_workers"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        persistent_workers=config["dataloader"]["persistent_workers"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        persistent_workers=config["dataloader"]["persistent_workers"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    return train_loader, val_loader, test_loader


def initialize_model() -> nn.Module:
    """Initialize model with config parameters."""
    return GTNO(
        lifting_dim=config["model"]["lifting_dim"],
        norm=config["model"]["norm"],
        activation=config["model"]["activation"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        heterogenous_attention_type=config["model"]["heterogenous_attention_type"],
        num_timesteps=config["model"]["num_timesteps"],
        use_rope=config["model"]["use_rope"],
        use_spherical_harmonics=config["model"]["use_spherical_harmonics"],
        use_equivariant_lifting=config["model"]["use_equivariant_lifting"],
        value_residual_type=config["model"]["value_residual_type"],
    ).to(device)


def initialize_optimizer(model: nn.Module) -> optim.Optimizer:
    """Initialize optimizer based on config."""
    match config["optimizer"]["type"]:
        case "sgd":
            return optim.SGD(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
        case "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=config["optimizer"]["learning_rate"],
                weight_decay=config["optimizer"]["weight_decay"],
                betas=tuple(config["optimizer"]["adam_betas"]),
                eps=config["optimizer"]["adam_eps"],
                amsgrad=True,
                fused=True,
            )
        case "muon":
            return pt_optim.Muon(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
        case "kron":
            return pt_optim.Kron(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
        case _:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['type']}")


def reset_weights(model: torch.nn.Module):
    """Reinitialize model weights without recompiling."""
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):  # Applies to layers like Linear, Conv, etc.
            layer.reset_parameters()


def train_step(model: nn.Module, optimizer: optim.Optimizer, loader: DataLoader[dict[str, torch.Tensor]], scheduler: optim.lr_scheduler._LRScheduler | None) -> float:
    """Single training epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = tensordict.from_dict(batch).to(device)
        assert batch["x_0"].shape[1] == (config["model"]["num_timesteps"]), batch["x_0"].shape
        target_coords = batch.pop("x_t")
        _ = batch.pop("v_t")

        optimizer.zero_grad()
        pred_coords: torch.Tensor = model(batch)

        # Calculate MSE loss
        assert pred_coords.shape == target_coords.shape, f"{pred_coords.shape} != {target_coords.shape}"
        loss = F.mse_loss(pred_coords, target_coords)
        total_loss += loss.item() * batch.batch_size[0]

        _ = loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        optimizer.step()

        if scheduler:
            scheduler.step()

    return total_loss / float(len(loader.dataset))


def evaluate(model: nn.Module, loader: DataLoader[dict[str, torch.Tensor]]) -> float:
    """Evaluation loop."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = tensordict.from_dict(batch).to(device)
            target_coords = batch.pop("x_t").reshape(-1, 3)
            _ = batch.pop("v_t")

            pred_coords: torch.Tensor = model(batch).reshape(-1, 3)
            loss = F.mse_loss(pred_coords, target_coords)
            total_loss += loss.item() * batch.batch_size[0]

    return total_loss / len(loader.dataset)


def main(num_epochs: int, model: nn.Module, molecule_type: MoleculeType) -> tuple[float, float, int]:
    """Full training pipeline."""
    start_time = datetime.now()

    # Initialize components
    train_loader, val_loader, test_loader = create_dataloaders(molecule_type)
    reset_weights(model)
    optimizer = initialize_optimizer(model)
    total_steps: int = len(train_loader.dataset) // config["training"]["batch_size"] * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Training loop
    best_val_loss = float("inf")
    best_val_loss_epoch = 0

    progress_bar = tqdm(range(num_epochs), desc="Training", leave=False, unit="epoch", position=2)
    for epoch in progress_bar:
        train_loss = train_step(model, optimizer, train_loader, scheduler)
        val_loss = evaluate(model, val_loader)

        # Log gate parameters
        log_feature_weights(model.named_parameters(), epoch)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss and epoch > 0.5 * num_epochs:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            torch.save(model.state_dict(), "benchmark_runs/temp_best_model.pth")

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
    _ = model.load_state_dict(torch.load("benchmark_runs/temp_best_model.pth", weights_only=True))
    test_loss = evaluate(model, test_loader)

    total_time = (datetime.now() - start_time).total_seconds()
    return test_loss, total_time, best_val_loss_epoch


def benchmark(runs: int, epochs_per_run: int, compile: bool, molecule_type: MoleculeType | Literal["all_mols"]) -> None:
    """
    Benchmarking function with JSON results logging.

    Args:
        runs: Number of runs to perform
        epochs_per_run: Number of epochs to run per run
        compile: Whether to compile the model
        molecule_type: Molecule type to run on
            "all_mols": Run on all molecules

    Returns:
        None
    """
    results = {"config_name": project_name, "runs": {}, "summary": {}, "timestamp": datetime.now().isoformat()}
    molecules = list(MoleculeType) if molecule_type == "all_mols" else [molecule_type]

    if compile:
        model = torch.compile(initialize_model(), dynamic=True)
    else:
        model = initialize_model()

    molecule_progress_bar = tqdm(molecules, leave=False, unit="molecule", position=0)
    for molecule in molecule_progress_bar:
        molecule_progress_bar.set_description(f"Running {molecule.value}")
        runs_progress_bar = tqdm(range(runs), leave=False, unit="run", position=1)
        for run in runs_progress_bar:
            runs_progress_bar.set_description(f"Run {run+1}/{runs}")
            start_time = datetime.now()
            test_loss, duration, best_val_loss_epoch = main(epochs_per_run, model, molecule)
            end_time = datetime.now()

            # Store run results
            results["runs"][f"run{run+1}"] = {
                "test_loss": float(test_loss),
                "best_val_loss_epoch": float(best_val_loss_epoch),
                "time_seconds": float(duration),
                "time_per_epoch": float(duration / epochs_per_run),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }

        # Calculate summary statistics
        test_losses = [res["test_loss"] for res in results["runs"].values()]
        times = [res["time_seconds"] for res in results["runs"].values()]
        times_per_epoch = [res["time_per_epoch"] for res in results["runs"].values()]
        best_val_loss_epochs = [res["best_val_loss_epoch"] for res in results["runs"].values()]
        results["summary"] = {
            "mean_test_loss": float(sum(test_losses) / len(test_losses)),
            "std_dev_test_loss": float(torch.std(torch.tensor(test_losses)).item()),
            "mean_secs_per_run": float(sum(times) / len(times)),
            "total_secs": float(sum(times)),
            "min_test_loss": float(min(test_losses)),
            "max_test_loss": float(max(test_losses)),
            "mean_secs_per_epoch": float(sum(times_per_epoch) / len(times_per_epoch)),
            "config": config,
            "best_val_loss_epochs": float(sum(best_val_loss_epochs) / len(best_val_loss_epochs)),
        }

        wandb.log(
            {
                "mean_test_loss": results["summary"]["mean_test_loss"],
                "mean_secs_per_run": results["summary"]["mean_secs_per_run"],
                "mean_secs_per_epoch": results["summary"]["mean_secs_per_epoch"],
            }
        )

        # Save to JSON
        filename = f"benchmark_runs/{project_name}_{molecule.value}_{results['summary']['mean_test_loss']:.2e}x10-2_{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        tqdm.write(f"\nSaved benchmark results to {filename}")
        tqdm.write(f"Benchmark Results ({runs} runs, {epochs_per_run} epochs/run):")
        tqdm.write(f"  Average Test Loss: {results['summary']['mean_test_loss']:.4f} ± {results['summary']['std_dev_test_loss']:.4f}")
        tqdm.write(f"  Average Time per Run: {results['summary']['mean_secs_per_run']:.1f}s")
        tqdm.write(f"  Total Benchmark Time: {results['summary']['total_secs']:.1f}s")
        tqdm.write(f"  Average Time per Epoch: {results['summary']['mean_secs_per_epoch']:.1f}s")
        tqdm.write(f"  Average Best Val Loss Epoch: {results['summary']['best_val_loss_epochs']:.1f}")


if __name__ == "__main__":
    benchmark(
        int(config["benchmark"]["runs"]),
        int(config["training"]["epochs"]),
        bool(config["benchmark"]["compile"]),
        MoleculeType(config["benchmark"]["molecule_type"]),
    )
