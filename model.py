import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):
    """Wrap (X, y) numpy arrays for PyTorch."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """Simple MLP for window-level feature vectors."""
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_mlp(X_tr, y_tr, X_va, y_va, *, epochs=20, batch_size=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = NumpyDataset(X_tr, y_tr)
    ds_va = NumpyDataset(X_va, y_va)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    model = MLP(in_dim=X_tr.shape[1], n_classes=len(np.unique(y_tr))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_va = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += loss.item() * y.size(0)

        train_loss = running / len(ds_tr)
        va_acc = evaluate(model, dl_va, device)

        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={va_acc:.4f}")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, device
