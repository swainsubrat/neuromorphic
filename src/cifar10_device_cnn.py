import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============================================================
# 0. OUTPUT
# ============================================================

def make_outdir(name):
    os.makedirs(name, exist_ok=True)
    return name

# ============================================================
# 1. DEVICE DATA (FULL LTP/LTD)
# ============================================================

G_PLUS = np.array([
    5.30e-4, 5.33e-4, 5.36e-4, 5.39e-4, 5.44e-4, 5.445e-4, 5.455e-4,
    5.465e-4, 5.475e-4, 5.50e-4, 5.50e-4, 5.52e-4, 5.52e-4, 5.525e-4,
    5.54e-4, 5.545e-4, 5.555e-4, 5.56e-4, 5.58e-4, 5.58e-4,
    5.585e-4, 5.59e-4, 5.615e-4, 5.62e-4, 5.62e-4, 5.625e-4,
    5.625e-4, 5.63e-4, 5.645e-4, 5.645e-4, 5.65e-4, 5.665e-4,
    5.665e-4, 5.665e-4, 5.665e-4, 5.67e-4, 5.67e-4, 5.67e-4,
    5.67e-4, 5.68e-4, 5.68e-4, 5.685e-4, 5.685e-4, 5.685e-4,
    5.69e-4, 5.695e-4, 5.70e-4, 5.70e-4, 5.71e-4, 5.715e-4
])

G_MINUS = np.array([
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

class AbstractUpdater:
    def __init__(self, beta):
        self.beta = beta

    def step(self, w, delta):
        return torch.clamp(
            w + WEIGHT_SCALE * delta * torch.exp(-self.beta * torch.abs(w)),
            -MAX_WEIGHT, MAX_WEIGHT
        )

# ============================================================
# 4. DATA (CIFAR-10)
# ============================================================

def get_loaders(batch):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train = DataLoader(
        datasets.CIFAR10("data", train=True, download=True, transform=transform_train),
        batch_size=batch, shuffle=True, num_workers=2, pin_memory=True
    )
    test = DataLoader(
        datasets.CIFAR10("data", train=False, download=True, transform=transform_test),
        batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )
    return train, test

# ============================================================
# 5. NETWORK (CNN)
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ============================================================
# 6. TRAIN / EVAL
# ============================================================

def train_epoch(model, loader, loss_fn, updater, mode, lr, device, log_interval=100):
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
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
                    p.copy_(updater.step(p, delta))

        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        seen += batch_size

        if log_interval and (batch_idx % log_interval == 0 or batch_idx == len(loader)):
            avg_loss = loss_sum / seen
            avg_acc = 100.0 * correct / seen
            print(
                f"  [Batch {batch_idx:04d}/{len(loader)}] "
                f"Loss {avg_loss:.4f} | Acc {avg_acc:.2f}%"
            )

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

# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", choices=["sgd", "abstract"], default="abstract")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="results_cifar10")
    parser.add_argument("--decay_start", type=int, default=20)
    parser.add_argument("--decay_gamma", type=float, default=0.95)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    outdir = make_outdir(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(args.batch)
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.update == "abstract":
        updater = AbstractUpdater(beta=args.beta)
    else:
        updater = None

    acc_log, loss_log = [], []

    for ep in range(1, args.epochs + 1):
        if ep > args.decay_start:
            lr_t = args.lr * (args.decay_gamma ** (ep - args.decay_start))
        else:
            lr_t = args.lr
        loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            updater,
            args.update,
            lr_t,
            device,
            log_interval=args.log_interval,
        )
        acc = evaluate(model, test_loader, device)

        acc_log.append(acc)
        loss_log.append(loss)

        print(f"[{args.update.upper()}] Epoch {ep:03d} | Loss {loss:.4f} | Acc {acc:.2f}%")

    np.save(os.path.join(outdir, "acc.npy"), np.array(acc_log))
    np.save(os.path.join(outdir, "loss.npy"), np.array(loss_log))

    plt.plot(acc_log)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(outdir, "accuracy.png"), dpi=300)

if __name__ == "__main__":
    main()
