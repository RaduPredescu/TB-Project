import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class TorchDataset(Dataset):
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
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device, loss_fn):
    model.eval()
    total_loss, total_n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)
    return total_loss / max(total_n, 1)


def train_mlp(
    X_tr, y_tr, X_va, y_va,
    *,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    seed=42,
    log_dir="runs",
    experiment_name="mlp_emg",
    log_weight_hist=False,
    sched_step=5,
    sched_factor=0.5
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ds_tr = TorchDataset(X_tr, y_tr)
    ds_va = TorchDataset(X_va, y_va)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    model = MLP(in_dim=X_tr.shape[1], n_classes=len(np.unique(y_tr))).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=sched_step,
        gamma=sched_factor
    )

    writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")

    # log hparams (simple)
    writer.add_text("hparams", f"epochs={epochs}, batch_size={batch_size}, lr={lr}, seed={seed}")

    best_va = -1.0
    best_state = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            # train stats
            running_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y).sum().item()
            running_total += y.numel()

            global_step += 1

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_acc = evaluate(model, dl_va, device)
        val_loss = evaluate_loss(model, dl_va, device, loss_fn)
        scheduler.step()
        # TensorBoard scalars
        current_lr = opt.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        if val_acc > best_va:
            best_va = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    writer.add_text("best", f"best_val_acc={best_va:.4f}")
    writer.close()

    return model, device

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)
