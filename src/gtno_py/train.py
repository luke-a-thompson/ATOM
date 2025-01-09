import datetime
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
from gtno_py.dataloaders.egno_dataloder import MD17DynamicsDataset, MoleculeType, DataPartition
from gtno_py.modules.activations import FFNActivation
from torch.utils.data import DataLoader
from tqdm import tqdm
import tomllib
from gtno_py.model import IMPGTNO, GraphAttentionType, GraphHeterogenousAttentionType, NormType

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

seed = 42
_ = torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = MD17DynamicsDataset(
    partition=DataPartition.train,
    max_samples=5000,
    delta_frame=5000,
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=MoleculeType.aspirin,
)
loader_train = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)

dataset_val = MD17DynamicsDataset(
    partition=DataPartition.val,
    max_samples=2000,
    delta_frame=5000,
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=MoleculeType.aspirin,
)
loader_val = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

dataset_test = MD17DynamicsDataset(
    partition=DataPartition.test,
    max_samples=2000,
    delta_frame=5000,
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=MoleculeType.aspirin,
)
loader_test = DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=False)


model = IMPGTNO(
    lifting_dim=config["model"]["lifting_dim"],
    norm=NormType.RMS,
    activation=FFNActivation.RELU,
    num_layers=config["model"]["num_layers"],
    num_heads=config["model"]["num_heads"],
    graph_attention_type=GraphAttentionType.SPLIT_MHA,
    heterogenous_attention_type=GraphHeterogenousAttentionType.GHCNA,
    num_timesteps=config["model"]["num_timesteps"],
).to(device)

optimizer: optim.Optimizer
match config["optimizer"]["type"]:
    case "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
            betas=config["optimizer"]["adam_betas"],
        )
    case _:
        raise ValueError(f"Invalid optimizer: {config['optimizer']['type']}")

loss_fn = nn.MSELoss()


def train_step(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    _ = model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = tensordict.from_dict(batch).to(device)
        x_t = batch.pop("x_t")  # Pop them out to avoid tiling in the model

        optimizer.zero_grad()

        # Get predicted coordinates
        pred_coords: torch.Tensor = model(batch)

        # Get target coordinates and reshape to align with predictions
        target_coords: torch.Tensor = x_t
        assert pred_coords.shape == target_coords.shape, f"Predicted and target coordinates must have the same shape. Got {pred_coords.shape} and {target_coords.shape}"

        assert pred_coords.shape[-1] == target_coords.shape[-1], f"Predicted and target coordinates must have the same shape. Got {pred_coords.shape} and {target_coords.shape}"
        assert pred_coords.shape[-1] == 3, f"Predicted and target coordinates must have the last dimension of 3 (x, y, z). Got {pred_coords.shape}"
        assert target_coords.shape[-1] == 3, f"Predicted and target coordinates must have the last dimension of 3 (x, y, z). Got {target_coords.shape}"

        # Calculate MSE loss
        loss: torch.Tensor = loss_fn(pred_coords, target_coords)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.inference_mode()
def evaluate_step(model: nn.Module, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    _ = model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = tensordict.from_dict(batch).to(device)

        # Get predicted coordinates
        pred_coords: torch.Tensor = model(batch)

        # Get target coordinates and reshape to align with predictions
        target_coords: torch.Tensor = batch["x_t"]

        # Calculate MSE loss
        loss: torch.Tensor = loss_fn(pred_coords, target_coords)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


best_eval_loss: float = float("inf")
eval_losses: list[float] = []
num_epochs: int = config["training"]["epochs"]
for epoch in range(num_epochs):
    train_loss = train_step(model, optimizer, loader_train)
    val_loss = evaluate_step(model, loader_val)
    eval_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_eval_loss:
        best_eval_loss = val_loss
        date = datetime.datetime.now().strftime("%Y%m%d")
        torch.save(model.state_dict(), f"trained_models/best_model_{date}.pth")

test_loss = evaluate_step(torch.load(f"trained_models/best_model_{date}.pth"), loader_test)
print(f"Best validation loss: {best_eval_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")
