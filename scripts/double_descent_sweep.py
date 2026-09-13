"""Synthetic label-noise classification sweep for reproducing double descent.

Generates a synthetic binary classification dataset with a controllable
label-noise fraction, sweeps a two-layer MLP's hidden width across the
interpolation threshold, trains multiple random seeds per width, and logs
per-epoch train/test error to CSV.

This script only builds the dataset + sweep + CSV logger -- it deliberately
does not plot anything yet. See scripts/plot_double_descent.py (once the CSV
schema below is confirmed) for that.

CSV schema (one row per width x seed x epoch)
----------------------------------------------
    width        int    hidden layer size for that run
    seed         int    random seed controlling init + data split/noise
    epoch        int    epoch index (0-based)
    train_error  float  fraction of misclassified training examples
    test_error   float  fraction of misclassified held-out examples

Usage:
    python scripts/double_descent_sweep.py --dry-run
    python scripts/double_descent_sweep.py --out results/double_descent.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Widths concentrated around the expected interpolation threshold: a 2-layer
# MLP with n_features inputs and 2 outputs has
# params(w) = w*(n_features + 1) + w*2 + 2 = w*(n_features + 3) + 2.
# With the defaults below (n_features=10, 500 training examples), that
# crosses 500 near w ~= 38 -- hence the denser spacing from 24 to 64.
DEFAULT_WIDTHS = [2, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 192, 256, 384, 512]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def make_noisy_dataset(
    n_samples: int, n_features: int, label_noise: float, test_size: float, seed: int
):
    """Synthetic binary classification data with a fraction of labels flipped."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        n_classes=2,
        flip_y=0.0,  # label noise is applied explicitly below instead
        random_state=seed,
    )
    if label_noise > 0:
        rng = np.random.default_rng(seed)
        flip_mask = rng.random(n_samples) < label_noise
        y = y.copy()
        y[flip_mask] = 1 - y[flip_mask]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


class MLP(nn.Module):
    """Two-layer MLP: Linear -> ReLU -> Linear (trained via autograd, not the
    manual backward pass from scripts/gradient_check.py)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def error_rate(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) != y).float().mean().item()


def train_one_run(
    width: int,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[dict]:
    torch.manual_seed(seed)
    model = MLP(X_train.shape[1], width, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    rows = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            train_err = error_rate(model(Xtr), ytr)
            test_err = error_rate(model(Xte), yte)
        rows.append(
            {
                "width": width,
                "seed": seed,
                "epoch": epoch,
                "train_error": train_err,
                "test_error": test_err,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/double_descent.csv"))
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--label-noise", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny run (2 widths, 1 seed, 5 epochs) to sanity-check the CSV schema",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.widths = args.widths[:2]
        args.seeds = args.seeds[:1]
        args.epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["width", "seed", "epoch", "train_error", "test_error"]

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in args.seeds:
            X_train, X_test, y_train, y_test = make_noisy_dataset(
                args.n_samples, args.n_features, args.label_noise, args.test_size, seed
            )
            for width in args.widths:
                rows = train_one_run(
                    width, seed, X_train, y_train, X_test, y_test, args.epochs, args.lr, device
                )
                writer.writerows(rows)
                last = rows[-1]
                print(
                    f"width={width:4d} seed={seed}  "
                    f"final train_error={last['train_error']:.3f}  "
                    f"test_error={last['test_error']:.3f}"
                )

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
