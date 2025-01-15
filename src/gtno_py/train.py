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
from gtno_py.gtno_model import IMPGTNO, GraphAttentionType, GraphHeterogenousAttentionType, NormType
from gtno_py.egno.egno_model import EGNO

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
    max_samples=5000,
    delta_frame=5000,
    num_timesteps=config["model"]["num_timesteps"],
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=MoleculeType.aspirin,
)
loader_train = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)

dataset_val = MD17DynamicsDataset(
    partition=DataPartition.val,
    max_samples=2000,
    delta_frame=5000,
    num_timesteps=config["model"]["num_timesteps"],
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=MoleculeType.aspirin,
)
loader_val = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

dataset_test = MD17DynamicsDataset(
    partition=DataPartition.test,
    max_samples=2000,
    delta_frame=5000,
    num_timesteps=config["model"]["num_timesteps"],
    data_dir="data/md17_npz/",
    split_dir="data/md17_egno_splits/",
    molecule_type=MoleculeType.aspirin,
)
loader_test = DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=False)

match config["model"]["model_type"]:
    case "gtno":
        model = IMPGTNO(
            lifting_dim=config["model"]["lifting_dim"],
            norm=NormType.RMS,
            activation=FFNActivation.RELU,
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            graph_attention_type=GraphAttentionType.UNIFIED_MHA,
            heterogenous_attention_type=GraphHeterogenousAttentionType.GHCNA,
            num_timesteps=config["model"]["num_timesteps"],
        ).to(device)
    case "egno":  # Default EGNO arguments
        model = EGNO(
            in_node_nf=2,
            in_edge_nf=2 + 3,
            hidden_nf=64,
            device="cuda",
            n_layers=5,
            with_v=True,
            flat=False,
            activation=nn.SiLU(),
            use_time_conv=True,
            num_modes=2,
            num_timesteps=8,
        ).to(device)
    case _:
        raise ValueError(f"Invalid model type: {config['model']['model_type']}")

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

scheduler: optim.lr_scheduler.LRScheduler
match config["scheduler"]["type"]:
    case "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"])
    case _:
        raise ValueError(f"Invalid scheduler: {config['scheduler']['type']}")

loss_fn = nn.MSELoss()


def train_step(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    _ = model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = tensordict.from_dict(batch).to(device)
        x_t: torch.Tensor = batch.pop("x_t")  # Pop them out to avoid tiling in the model

        optimizer.zero_grad()

        # Get predicted coordinates
        match config["model"]["model_type"]:
            case "gtno":
                pred_coords: torch.Tensor = model(batch)
            case "egno":
                # Same per-batch computations as EGNO
                batch_size = int(batch.batch_size[0])
                n_nodes = int(batch["x_0"].shape[1])

                # The velocity norm is already the last element of the velocity vector. We concat it with the normalised atomic numbers.
                z_normalised = batch["Z"] / batch["Z"].max().unsqueeze(-1)
                nodes: torch.Tensor = torch.cat([batch["v_0"], z_normalised], dim=-1)  # Shape [B, N, 5], 5 = 3 (x, y, z) + 1 (velocity norm) + 1 (normalised atomic number)

                edges: tuple[torch.LongTensor, torch.LongTensor] = dataset_train.get_edges(batch_size, batch["x_0"].shape[1])
                cfg = dataset_train.get_cfg(batch_size, batch["x_0"].shape[1], batch["cfg"])

                rows, cols = edges[0], edges[1]
                # loc_dist: we index on the N dimension (dim=1), so we do batch-wise indexing:
                #   batch["x_0"] has shape [B, N, 3].
                #   batch["x_0"][:, rows] -> shape [B, E, 3]
                #   batch["x_0"][:, cols] -> shape [B, E, 3]
                loc_dist = torch.sum((batch["x_0"][:, rows] - batch["x_0"][:, cols]) ** 2, dim=-1).unsqueeze(-1)  # Shape [batch, n_edges, 1]
                assert (
                    loc_dist.shape[1] == batch["edge_attr"].shape[1]
                ), f"Loc dist must have the same number of edges {loc_dist.shape[1]} as edge attributes {batch['edge_attr'].shape[1]}"
                edge_attr: torch.Tensor = torch.cat([batch["edge_attr"], loc_dist], dim=2).detach()

                # Replicate the logic from the original code to compute loc_mean:
                # 1) Take the mean over the node dimension (dim=1), shape => [B, 1, 4]
                # 2) Repeat for each node => [B, N, 4]
                # 3) add .view(-1, 4) to reshape => [B*N, 4] as in original EGNO
                loc_mean: torch.Tensor = batch["x_0"].mean(dim=1, keepdim=True).repeat(1, n_nodes, 1)  # shape [B, 1, 4]  # shape [B, N, 4]

                # To clearly align with the original EGNO, we specify the batch elements to deliver the forward here.
                pred_coords: torch.Tensor = model(batch["x_0"], nodes, edges, edge_attr, batch["v_0"], loc_mean=loc_mean)
            case _:
                raise ValueError(f"Invalid model type: {config['model']['model_type']}")

        # Get target coordinates and reshape to align with predictions
        target_coords: torch.Tensor = x_t
        assert pred_coords.shape == target_coords.shape, f"Predicted and target coordinates must have the same shape. Got {pred_coords.shape} and {target_coords.shape}"

        assert pred_coords.shape[-1] == target_coords.shape[-1], f"Predicted and target coordinates must have the same shape. Got {pred_coords.shape} and {target_coords.shape}"
        assert pred_coords.shape[-1] == 3, f"Predicted and target coordinates must have the last dimension of 3 (x, y, z). Got {pred_coords.shape}"
        assert target_coords.shape[-1] == 3, f"Predicted and target coordinates must have the last dimension of 3 (x, y, z). Got {target_coords.shape}"

        # Calculate MSE loss
        loss: torch.Tensor = loss_fn(pred_coords, target_coords)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"]["max_grad_norm"])
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.inference_mode()
def evaluate_step(model: nn.Module, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    model.eval()
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
date: str = datetime.datetime.now().strftime("%Y%m%d")
for epoch in range(num_epochs):
    train_loss = train_step(model, optimizer, loader_train)
    val_loss = evaluate_step(model, loader_val)
    eval_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_eval_loss:
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
