import torch
import torch.nn as nn
import torch.optim as optim
import tensordict
from Dataloaders.custom_dataloader import MD17Dataset
from torch.utils.data import DataLoader, Subset
from model import IMPGTNO
from modules.activations import FFNActivation
from model import NormType, GraphAttentionType
from tqdm import tqdm
from utils import get_data_split_indices_custom
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset = MD17Dataset("data/rmd17_cleaned/rmd17_aspirin.csv")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_indices, val_indices, test_indices = get_data_split_indices_custom(len(dataset))
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = IMPGTNO(
    node_feature_dim=7,
    edge_feature_dim=3,
    graph_feature_dim=1,
    lifting_dim=128,
    norm=NormType.RMS,
    activation=FFNActivation.RELU,
    num_layers=3,
    num_heads=4,
    graph_attention_type=GraphAttentionType.MHA,
).to(device)

optimizer_type = optim.AdamW
optimizer: optim.Optimizer
match optimizer_type:
    case optim.AdamW:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    case _:
        raise ValueError(f"Invalid optimizer: {optimizer}")

for batch in dataloader:
    from utils import pretty_print_graph_data

    pretty_print_graph_data(batch, print_node_features=True)
    break


def train_step(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = tensordict.from_dict(batch).to(device)
        optimizer.zero_grad()

        # Get predicted coordinates
        pred_coords = model(batch)

        # Get target coordinates and reshape to align with predictions
        target_coords = batch["coords"].view(pred_coords.shape)

        # Calculate MSE loss
        loss = torch.mean((pred_coords - target_coords) ** 2)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate_step(model: nn.Module, dataloader: DataLoader):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = tensordict.from_dict(batch).to(device)

        # Get predicted coordinates
        pred_coords = model(batch)

        # Get target coordinates and reshape to align with predictions
        target_coords = batch["coords"].view(pred_coords.shape)

        # Calculate MSE loss
        loss = torch.mean((pred_coords - target_coords) ** 2)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


num_epochs = 20
best_eval_loss = float("inf")
eval_losses: list[float] = []
for epoch in range(num_epochs):
    train_loss = train_step(model, optimizer, train_loader)
    val_loss = evaluate_step(model, val_loader)
    eval_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_eval_loss:
        best_eval_loss = val_loss
        date = datetime.datetime.now().strftime("%Y%m%d")
        torch.save(model.state_dict(), f"trained_models/best_model_{best_eval_loss:.4f}_{date}.pth")

test_loss = evaluate_step(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")

print(f"Best validation loss: {best_eval_loss:.4f}")
