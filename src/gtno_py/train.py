import datetime
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_optimizer as pt_optim
from gtno_py.dataloaders.egno_dataloder import MD17DynamicsDataset, DataPartition
from gtno_py.gtno.activations import FFNActivation
from torch.utils.data import DataLoader
from tqdm import tqdm
import tomllib
from gtno_py.gtno.gtno_model import IMPGTNO, GraphAttentionType, GraphHeterogenousAttentionType, NormType

with open("config.toml", "rb") as file:
    config = tomllib.load(file)

# wandb.login()
# _ = wandb.init(
#     project=config["wandb"]["project_name"],
#     name=config["wandb"]["run_name"],
#     config={
#         "training": config["training"],
#         "model": config["model"],
#     },
# )

seed = config["training"]["seed"]
_ = torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = MD17DynamicsDataset(
    partition=DataPartition.train,
    max_samples=500,
    delta_frame=5000,
    num_timesteps=config["model"]["num_timesteps"],
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=config["dataset"]["molecule_type"],
)
loader_train = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
num_nodes: int = next(iter(loader_train))["x_0"].shape[1]

dataset_val = MD17DynamicsDataset(
    partition=DataPartition.val,
    max_samples=2000,
    delta_frame=5000,
    num_timesteps=config["model"]["num_timesteps"],
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=config["dataset"]["molecule_type"],
)
loader_val = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

dataset_test = MD17DynamicsDataset(
    partition=DataPartition.test,
    max_samples=2000,
    delta_frame=5000,
    num_timesteps=config["model"]["num_timesteps"],
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=config["dataset"]["molecule_type"],
)
loader_test = DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=False)

match config["model"]["model_type"]:
    case "gtno":
        model = IMPGTNO(
            lifting_dim=config["model"]["lifting_dim"],
            norm=NormType.RMS,
            activation=FFNActivation.SWIGLU,
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            graph_attention_type=GraphAttentionType.SPLIT_MHA,
            heterogenous_attention_type=GraphHeterogenousAttentionType.GHCA,
            num_timesteps=config["model"]["num_timesteps"],
        ).to(device)
    case _:
        raise ValueError(f"Invalid model type: {config['model']['model_type']}")


total_params = sum(p.numel() for p in model.parameters())
model_name: str = config["model"]["model_type"]
print(f"Initialised {model_name.capitalize()} model with {total_params} parameters")

optimizer: optim.Optimizer
match config["optimizer"]["type"]:
    case "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
    case "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
            betas=config["optimizer"]["adam_betas"],
            eps=config["optimizer"]["adam_eps"],
            amsgrad=True,
            fused=True,
        )
    case "muon":
        optimizer = pt_optim.Muon(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
    case _:
        raise ValueError(f"Invalid optimizer: {config['optimizer']['type']}")

scheduler: optim.lr_scheduler.LRScheduler | None
match config["scheduler"]["type"]:
    case "none":
        scheduler = None
    case "linear":
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config["training"]["epochs"])
    case "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])
    case "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader_train) * config["training"]["epochs"])
    case "cosine_annealing_warmup":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    case "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["optimizer"]["learning_rate"], epochs=config["training"]["epochs"], steps_per_epoch=len(loader_train))
    case _:
        raise ValueError(f"Invalid scheduler: {config['scheduler']['type']}")

loss_fn = nn.MSELoss(reduction="none")


def train_step(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    model.train()
    results = {"epoch": epoch, "loss": 0, "counter": 0}

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = tensordict.from_dict(batch).to(device)
        x_t: torch.Tensor = batch.pop("x_t")  # Pop them out to avoid tiling in the model
        v_t: torch.Tensor = batch.pop("v_t")
        optimizer.zero_grad()

        # Get predicted coordinates
        pred_coords: torch.Tensor
        match config["model"]["model_type"]:
            case "gtno":
                pred_coords = model(batch)
                pred_coords: torch.Tensor = pred_coords.reshape(-1, 3)
            case _:
                raise ValueError(f"Invalid model type: {config['model']['model_type']}")

        # Get target coordinates and reshape to align with predictions
        target_coords: torch.Tensor = x_t.reshape(-1, 3)
        assert pred_coords.shape == target_coords.shape, f"Predicted and target coordinates must have the same shape. Got {pred_coords.shape} and {target_coords.shape}"

        # Calculate MSE loss
        losses = loss_fn(pred_coords, target_coords).view(config["model"]["num_timesteps"], batch.batch_size[0] * num_nodes, 3)  # [T, B*13 (nodes), 3]
        losses: torch.Tensor = torch.mean(losses, dim=(1, 2))  # [T, B*13]
        loss = torch.mean(losses)
        results["loss"] += losses[-1].item() * batch.batch_size[0]
        results["counter"] += batch.batch_size[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["max_grad_norm"])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    for name, param in model.named_parameters():
        if "offset" in name:
            print(f"{name}: {param}")

    return results["loss"] / results["counter"]


def evaluate_step(model: nn.Module, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    model.eval()
    total_loss = 0.0
    res = {"epoch": epoch, "loss": 0, "counter": 0}

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = tensordict.from_dict(batch).to(device)
        x_t: torch.Tensor = batch.pop("x_t")  # Pop them out to avoid tiling in the model

        # Get predicted coordinates
        match config["model"]["model_type"]:
            case "gtno":
                pred_coords: torch.Tensor = model(batch).reshape(-1, 3)
            case _:
                raise ValueError(f"Invalid model type: {config['model']['model_type']}")

        # Get target coordinates and reshape to align with predictions
        target_coords: torch.Tensor = x_t.reshape(-1, 3)

        # Calculate MSE loss
        losses = loss_fn(pred_coords, target_coords).view(config["model"]["num_timesteps"], batch.batch_size[0] * num_nodes, 3)
        losses: torch.Tensor = torch.mean(losses, dim=(1, 2))
        loss = torch.mean(losses)
        res["loss"] += losses[-1].item() * batch.batch_size[0]
        res["counter"] += batch.batch_size[0]

        total_loss += loss.item()

    return res["loss"] / res["counter"]


best_eval_loss: float = float("inf")
eval_losses: list[float] = []
num_epochs: int = config["training"]["epochs"]
date: str = datetime.datetime.now().strftime("%Y%m%d")
for epoch in range(num_epochs):
    train_loss = train_step(model, optimizer, loader_train)
    val_loss = evaluate_step(model, loader_val)
    eval_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}. LR: {optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_eval_loss and epoch > 0.8 * num_epochs:
        best_eval_loss = val_loss
        torch.save(model.state_dict(), f"trained_models/best_eval_model_{date}.pth")
        print(f"Saved best model to trained_models/best_eval_model_{date}.pth with val loss {val_loss:.4f}")

# Load the saved model weights into a new model instance
model.load_state_dict(torch.load(f"trained_models/best_eval_model_{date}.pth", weights_only=True))
model.eval()  # Set the model to evaluation mode

# Evaluate on the test set
test_loss = evaluate_step(model, loader_test)
torch.save(model.state_dict(), f"trained_models/best_test_model_with_loss_{(test_loss * 1e2):.4f}e-2.pth")
print(f"Best validation loss: {best_eval_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}, {(test_loss * 1e2):.4f}x10^-2")
