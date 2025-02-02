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

# Load configuration
with open("config.toml", "rb") as file:
    config = tomllib.load(file)

# Set device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["training"]["seed"])
torch.cuda.manual_seed(config["training"]["seed"])


def create_dataloaders() -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    """Create train/val/test dataloaders."""
    train_dataset = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=3000,
        num_timesteps=config["model"]["num_timesteps"],
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=config["dataset"]["molecule_type"],
    )

    val_dataset = MD17DynamicsDataset(
        partition=DataPartition.val,
        max_samples=2000,
        delta_frame=3000,
        num_timesteps=config["model"]["num_timesteps"],
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=config["dataset"]["molecule_type"],
    )

    test_dataset = MD17DynamicsDataset(
        partition=DataPartition.test,
        max_samples=2000,
        delta_frame=3000,
        num_timesteps=config["model"]["num_timesteps"],
        data_dir="data/md17_npz/",
        split_dir="data/md17_egno_splits/",
        molecule_type=config["dataset"]["molecule_type"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, persistent_workers=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, persistent_workers=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, persistent_workers=True, num_workers=4)

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
                betas=config["optimizer"]["adam_betas"],
                eps=config["optimizer"]["adam_eps"],
                amsgrad=True,
                fused=True,
            )
        case "muon":
            return pt_optim.Muon(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
        case "kron":
            return pt_optim.Kron(model.parameters(), lr=config["optimizer"]["learning_rate"] / 3, weight_decay=config["optimizer"]["weight_decay"])
        case _:
            raise ValueError(f"Invalid optimizer: {config['optimizer']['type']}")


def train_step(model: nn.Module, optimizer: optim.Optimizer, loader: DataLoader[dict[str, torch.Tensor]], scheduler: optim.lr_scheduler._LRScheduler | None) -> float:
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    num_nodes: int = next(iter(loader))["x_0"].shape[1]

    for batch in loader:
        batch = tensordict.from_dict(batch).to(device)
        target_coords = batch.pop("x_t").reshape(-1, 3)

        optimizer.zero_grad()
        pred_coords = model(batch).reshape(-1, 3)

        # Calculate MSE loss
        loss = F.mse_loss(pred_coords, target_coords)
        total_loss += loss.item() * batch.batch_size[0]

        _ = loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        optimizer.step()

        if scheduler:
            scheduler.step()

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader[dict[str, torch.Tensor]]) -> float:
    """Evaluation loop."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = tensordict.from_dict(batch).to(device)
            target_coords = batch.pop("x_t").reshape(-1, 3)

            pred_coords: torch.Tensor = model(batch).reshape(-1, 3)
            loss = F.mse_loss(pred_coords, target_coords)
            total_loss += loss.item() * batch.batch_size[0]

    return total_loss / len(loader.dataset)


def main(num_epochs: int) -> tuple[float, float]:
    """Full training pipeline."""
    start_time = datetime.now()

    # Initialize components
    train_loader, val_loader, test_loader = create_dataloaders()
    model = initialize_model()
    optimizer = initialize_optimizer(model)

    # Training loop
    best_val_loss = float("inf")

    progress_bar = tqdm(range(num_epochs), desc="Training", leave=False, unit="epoch")
    for epoch in progress_bar:
        train_loss = train_step(model, optimizer, train_loader, None)
        val_loss = evaluate(model, val_loader)

        # Save best model
        if val_loss < best_val_loss and epoch > 0.5 * num_epochs:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "benchmark_runs/temp_best_model.pth")

        # Update progress bar with losses
        progress_bar.set_postfix({"Train loss": f"{train_loss:.4f}", "Val loss": f"{val_loss:.4f}", "Best val loss": f"{best_val_loss:.4f}"})

    # Final evaluation
    _ = model.load_state_dict(torch.load("benchmark_runs/temp_best_model.pth", weights_only=True))
    test_loss = evaluate(model, test_loader)

    total_time = (datetime.now() - start_time).total_seconds()
    return test_loss, total_time


def benchmark(runs: int = 5, epochs_per_run: int = 100) -> None:
    """Benchmarking function with JSON results logging."""
    results = {"config_name": config["wandb"]["project_name"], "runs": {}, "summary": {}, "timestamp": datetime.now().isoformat()}

    for run in range(runs):
        print(f"Run {run+1}/{runs}")
        start_time = datetime.now()
        test_loss, duration = main(num_epochs=epochs_per_run)
        end_time = datetime.now()

        # Store run results
        results["runs"][f"run{run+1}"] = {"loss": float(test_loss), "time_seconds": float(duration), "start_time": start_time.isoformat(), "end_time": end_time.isoformat()}

        print(f"Run {run+1} - Test Loss: {test_loss:.4f}, Duration: {duration:.1f}s")

    # Calculate summary statistics
    losses = [res["loss"] for res in results["runs"].values()]
    times = [res["time_seconds"] for res in results["runs"].values()]

    results["summary"] = {
        "mean_loss": float(sum(losses) / len(losses)),
        "std_loss": float(torch.std(torch.tensor(losses)).item()),
        "mean_time": float(sum(times) / len(times)),
        "total_time": float(sum(times)),
        "min_loss": float(min(losses)),
        "max_loss": float(max(losses)),
        "config": config,
    }

    # Save to JSON
    filename = f"benchmark_runs/benchmark_{config['wandb']['project_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {filename}")
    print(f"Benchmark Results ({runs} runs, {epochs_per_run} epochs/run):")
    print(f"  Average Test Loss: {results['summary']['mean_loss']:.4f} ± {results['summary']['std_loss']:.4f}")
    print(f"  Average Time per Run: {results['summary']['mean_time']:.1f}s")
    print(f"  Total Benchmark Time: {results['summary']['total_time']:.1f}s ()")


if __name__ == "__main__":
    # For single run
    # test_loss, duration = main()
    # print(f"Final test loss: {test_loss:.4f} | Duration: {duration:.1f}s")

    # For benchmarking (uncomment)
    benchmark(1, config["training"]["epochs"])
