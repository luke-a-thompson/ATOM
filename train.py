import torch
import torch.nn as nn
import torch.optim as optim
import tensordict
from custom_dataloader import MD17Dataset
from torch.utils.data import DataLoader
from model import IMPGTNO
from modules.activations import FFNActivation
from model import NormType, GraphAttentionType
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MD17Dataset("data/rmd17_cleaned/rmd17_aspirin.csv")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

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

# for batch in dataloader:
#     from utils import pretty_print_graph_data

#     pretty_print_graph_data(batch, print_node_features=True)
#     break

criterion = nn.L1Loss()


def train_step(model: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader):
    model.train()
    for batch in tqdm(dataloader, desc="Training"):
        batch = tensordict.from_dict(batch).to(device)
        print(batch)
        # break
        optimizer.zero_grad()
        loss = criterion(model(batch), batch["Coordinates"])
        loss.backward()
        optimizer.step()

    return loss


@torch.no_grad()
def evaluate_step(model: nn.Module, dataloader: DataLoader):
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tensordict.from_dict(batch).to(device)
        loss = criterion(model(batch), batch["Coordinates"])
        return loss


num_epochs = 50
best_eval_loss = float("inf")
for epoch in range(num_epochs):
    train_loss = train_step(model, optimizer, train_loader)
    val_loss = evaluate_step(model, val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_eval_loss:
        best_eval_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print(f"Best validation loss: {best_eval_loss:.4f}")
