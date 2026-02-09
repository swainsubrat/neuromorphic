import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import pandas as pd

# ============================================================
# 0. OUTPUT
# ============================================================

def make_outdir(name):
    os.makedirs(name, exist_ok=True)
    return name

# ============================================================
# 1. DEVICE DATA (FULL LTP/LTD)
# ============================================================

DEFAULT_G_PLUS = np.array([
    5.30e-4, 5.33e-4, 5.36e-4, 5.39e-4, 5.44e-4, 5.445e-4, 5.455e-4,
    5.465e-4, 5.475e-4, 5.50e-4, 5.50e-4, 5.52e-4, 5.52e-4, 5.525e-4,
    5.54e-4, 5.545e-4, 5.555e-4, 5.56e-4, 5.58e-4, 5.58e-4,
    5.585e-4, 5.59e-4, 5.615e-4, 5.62e-4, 5.62e-4, 5.625e-4,
    5.625e-4, 5.63e-4, 5.645e-4, 5.645e-4, 5.65e-4, 5.665e-4,
    5.665e-4, 5.665e-4, 5.665e-4, 5.67e-4, 5.67e-4, 5.67e-4,
    5.67e-4, 5.68e-4, 5.68e-4, 5.685e-4, 5.685e-4, 5.685e-4,
    5.69e-4, 5.695e-4, 5.70e-4, 5.70e-4, 5.71e-4, 5.715e-4
])

DEFAULT_G_MINUS = np.array([
    4.455e-4, 4.35575e-4, 4.305e-4, 4.2315e-4, 4.16325e-4, 4.1055e-4,
    4.03725e-4, 3.99525e-4, 3.96375e-4, 3.94275e-4, 3.94275e-4,
    3.927e-4, 3.90075e-4, 3.885e-4, 3.8745e-4, 3.86925e-4,
    3.84825e-4, 3.843e-4, 3.81675e-4, 3.80625e-4, 3.801e-4,
    3.7905e-4, 3.78e-4, 3.78e-4, 3.77475e-4, 3.75375e-4,
    3.74325e-4, 3.73275e-4, 3.7275e-4, 3.7275e-4, 3.7275e-4,
    3.717e-4, 3.717e-4, 3.7065e-4, 3.696e-4, 3.6855e-4,
    3.68025e-4, 3.68025e-4, 3.675e-4, 3.66975e-4, 3.66975e-4,
    3.6645e-4, 3.6645e-4, 3.654e-4, 3.6435e-4, 3.63825e-4,
    3.633e-4, 3.633e-4, 3.612e-4, 3.58575e-4
])

G_PLUS = DEFAULT_G_PLUS
G_MINUS = DEFAULT_G_MINUS


def load_ltp_ltd_from_excel(sheet_name=None, excel_path="../LTP_LTD.xlsx"):
    if sheet_name is None:
        return DEFAULT_G_PLUS, DEFAULT_G_MINUS

    try:
        import openpyxl  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing optional dependency 'openpyxl'. Install it to read .xlsx files."
        ) from exc

    data = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    if data.shape[1] < 2:
        raise ValueError("Excel sheet must have at least two columns: LTP (col 0) and LTD (col 1).")
    g_plus = data.iloc[:, 0].astype(float).to_numpy()
    g_minus = data.iloc[:, 1].astype(float).to_numpy()
    return g_plus, g_minus

W_RAW = G_PLUS - G_MINUS
W_MONO = np.maximum.accumulate(W_RAW)
W_MIN, W_MAX = W_MONO.min(), W_MONO.max()
W_NORM = 2 * (W_MONO - W_MIN) / (W_MAX - W_MIN) - 1.0
PULSE_IDX = np.arange(len(W_NORM))

print(f"Device states: {len(W_NORM)}, W range: [{W_MIN:.6e}, {W_MAX:.6e}]")

device_interp = interp1d(PULSE_IDX, W_NORM, kind="cubic", fill_value="extrapolate")

# ============================================================
# 2. GLOBAL SCALING (CRITICAL FIX)
# ============================================================

WEIGHT_SCALE = 1e4        ### NEW: software amplification (try 1e3–1e5)
MAX_WEIGHT = 5.0          ### NEW: clip for stability

# ============================================================
# 3. UPDATE RULES
# ============================================================

class PulseBasedUpdater:
    def __init__(self, scale=1.0):
        self.scale = scale

    def step(self, w, delta):
        mag = torch.abs(delta)
        mag = mag / (mag.max() + 1e-8) * (len(PULSE_IDX) - 1)

        pulse_delta = device_interp(mag.cpu().numpy())
        pulse_delta = torch.tensor(pulse_delta, device=w.device)
        pulse_delta *= torch.sign(delta)

        # CHANGED: scale device update
        return torch.clamp(
            w + WEIGHT_SCALE * self.scale * pulse_delta,
            -MAX_WEIGHT, MAX_WEIGHT
        )


class AbstractUpdater:
    def __init__(self, beta):
        self.beta = beta

    def step(self, w, delta):
        # CHANGED: scaled abstract update
        return torch.clamp(
            w + WEIGHT_SCALE * delta * torch.exp(-self.beta * torch.abs(w)),
            -MAX_WEIGHT, MAX_WEIGHT
        )

# ============================================================
# 4. DATA
# ============================================================

def get_loaders(batch):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train = DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transform),
        batch_size=batch, shuffle=True
    )
    test = DataLoader(
        datasets.MNIST("../data", train=False, transform=transform),
        batch_size=1000
    )
    return train, test

# ============================================================
# 5. NETWORK
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

        # CHANGED: center weights away from zero
        nn.init.uniform_(self.fc1.weight, -0.5, 0.5)
        nn.init.uniform_(self.fc2.weight, -0.5, 0.5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# ============================================================
# 6. TRAIN / EVAL
# ============================================================

def train_epoch(model, loader, loss_fn, updater, mode, lr, device):
    model.train()
    loss_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                delta = -lr * p.grad
                if mode == "sgd":
                    p += delta
                else:
                    p.copy_(updater.step(p, delta))   ### CHANGED (safer)

        loss_sum += loss.item() * x.size(0)

    return loss_sum / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def compute_confusion(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return confusion_matrix(all_targets, all_preds)


def evaluate_with_loss(model, loader, loss_fn, device):
    model.eval()
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
    acc = 100.0 * correct / len(loader.dataset)
    avg_loss = loss_sum / len(loader.dataset)
    return avg_loss, acc


def set_publication_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", choices=["sgd", "pulse", "abstract"], default="pulse")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="../artifacts")
    parser.add_argument("--sheet", type=str, default=None, help="Sheet name in ../LTP_LTD.xlsx")
    args = parser.parse_args()

    outdir = make_outdir(args.out+"/"+args.sheet if args.sheet else args.out)
    set_publication_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    g_plus, g_minus = load_ltp_ltd_from_excel(args.sheet)
    global G_PLUS, G_MINUS, W_RAW, W_MONO, W_MIN, W_MAX, W_NORM, PULSE_IDX, device_interp
    G_PLUS, G_MINUS = g_plus, g_minus
    W_RAW = G_PLUS - G_MINUS
    W_MONO = np.maximum.accumulate(W_RAW)
    W_MIN, W_MAX = W_MONO.min(), W_MONO.max()
    W_NORM = 2 * (W_MONO - W_MIN) / (W_MAX - W_MIN) - 1.0
    PULSE_IDX = np.arange(len(W_NORM))
    device_interp = interp1d(PULSE_IDX, W_NORM, kind="cubic", fill_value="extrapolate")
    print(f"Device states: {len(W_NORM)}, W range: [{W_MIN:.6e}, {W_MAX:.6e}]")

    train_loader, test_loader = get_loaders(args.batch)
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.update == "pulse":
        updater = PulseBasedUpdater()
    elif args.update == "abstract":
        updater = AbstractUpdater(beta=args.beta)
    else:
        updater = None

    train_acc_log, test_acc_log = [], []
    train_loss_log, test_loss_log = [], []

    for ep in range(1, args.epochs + 1):
        if ep > 0:
            # lr_t = args.lr * (0.95 ** (ep - 20))
            lr_t = 0.0001
        else:
            lr_t = args.lr
        train_loss = train_epoch(model, train_loader, loss_fn,
                                 updater, args.update, lr_t, device)
        test_loss, test_acc = evaluate_with_loss(model, test_loader, loss_fn, device)
        _, train_acc = evaluate_with_loss(model, train_loader, loss_fn, device)

        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)
        train_acc_log.append(train_acc)
        test_acc_log.append(test_acc)

        print(
            f"[{args.update.upper()}] Epoch {ep:03d} | "
            f"Train Loss {train_loss:.4f} | Test Acc {test_acc:.2f}%"
        )

    metrics_df = pd.DataFrame({
        "epoch": np.arange(1, args.epochs + 1),
        "train_accuracy": train_acc_log,
        "test_accuracy": test_acc_log,
        "train_loss": train_loss_log,
        "test_loss": test_loss_log,
    })
    metrics_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    # Save confusion matrix as CSV (no .npy artifacts)
    cm = compute_confusion(model, test_loader, device)
    pd.DataFrame(cm).to_csv(os.path.join(outdir, "confusion_matrix.csv"), index=False)

    epochs = np.arange(1, args.epochs + 1)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(epochs, train_acc_log, label="Train", linestyle="-", marker="o", markersize=4)
    plt.plot(epochs, test_acc_log, label="Test", linestyle="--", marker="s", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_epoch.png"), dpi=600)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(epochs, train_loss_log, label="Train Loss", linestyle="-", marker="^", markersize=4)
    plt.plot(epochs, test_loss_log, label="Test Loss", linestyle=":", marker="d", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_vs_epoch.png"), dpi=600)

    plt.figure(figsize=(6.6, 6.0))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Final Epoch)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.grid(False)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=600)

if __name__ == "__main__":
    main()
