"""Modular-addition grokking experiment (mod 97).

Generates every (a, b) pair in Z_97 x Z_97 with label (a + b) mod 97, splits
40% train / 60% test, and trains a small embedding + MLP classifier with
AdamW at a high weight decay (1.0, matching Power et al. 2022's original
grokking setup) for a fixed number of full-batch gradient steps. Logs train
accuracy, test accuracy, and total weight L2 norm to CSV every 100 steps --
grokking shows up as train accuracy hitting ~100% almost immediately while
test accuracy stays near chance (1/97) for a long stretch, then jumps late.

CSV schema (one row every --log-every steps)
----------------------------------------------
    step            int    gradient step index (0-based)
    train_acc       float  fraction of train pairs classified correctly
    test_acc        float  fraction of test pairs classified correctly
    train_loss      float  mean cross-entropy loss on the train set
    test_loss       float  mean cross-entropy loss on the held-out set
    weight_l2_norm  float  L2 norm of all model parameters concatenated
    weight_decay    float  the run's AdamW weight decay (constant per run;
                           lets multiple runs be concatenated for comparison)
    train_frac      float  the run's train split fraction (same reason)
    seed            int    the run's random seed (data split + model init;
                           same reason -- see SEED_AUDIT.md)

Usage:
    python scripts/grokking_train.py --dry-run
    python scripts/grokking_train.py --steps 20000 --out results/grokking.csv
    python scripts/grokking_train.py --weight-decay 0.0 --out results/grokking_wd0.csv
    python scripts/grokking_train.py --train-frac 0.3 --out results/grokking_frac0.3.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODULUS = 97


def make_dataset(modulus: int, train_frac: float, seed: int):
    """Every (a, b) pair mod `modulus`, labeled (a+b) % modulus, split train/test."""
    a, b = np.meshgrid(np.arange(modulus), np.arange(modulus), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    y = (a + b) % modulus

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_train = int(len(y) * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    return (
        (a[train_idx], b[train_idx], y[train_idx]),
        (a[test_idx], b[test_idx], y[test_idx]),
    )


class GrokkingMLP(nn.Module):
    """Embed both operands, concatenate, small MLP -> modulus-way classifier."""

    def __init__(self, modulus: int, embed_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(modulus, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, modulus),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea, eb = self.embed(a), self.embed(b)
        return self.net(torch.cat([ea, eb], dim=-1))


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def weight_l2_norm(model: nn.Module) -> float:
    return torch.sqrt(sum((p**2).sum() for p in model.parameters())).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/grokking.csv"))
    parser.add_argument("--modulus", type=int, default=MODULUS)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny run (300 steps, log every 50) to sanity-check the CSV schema",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.steps = 300
        args.log_every = 50

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset(
        args.modulus, args.train_frac, args.seed
    )
    print(f"train examples = {len(y_tr)}, test examples = {len(y_te)}")

    a_tr = torch.tensor(a_tr, dtype=torch.long, device=device)
    b_tr = torch.tensor(b_tr, dtype=torch.long, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)
    a_te = torch.tensor(a_te, dtype=torch.long, device=device)
    b_te = torch.tensor(b_te, dtype=torch.long, device=device)
    y_te = torch.tensor(y_te, dtype=torch.long, device=device)

    model = GrokkingMLP(args.modulus, args.embed_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "train_acc",
        "test_acc",
        "train_loss",
        "test_loss",
        "weight_l2_norm",
        "weight_decay",
        "train_frac",
        "seed",
    ]

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(args.steps):
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(a_tr, b_tr), y_tr)
            loss.backward()
            opt.step()

            if step % args.log_every == 0 or step == args.steps - 1:
                model.eval()
                with torch.no_grad():
                    train_logits, test_logits = model(a_tr, b_tr), model(a_te, b_te)
                    train_acc = accuracy(train_logits, y_tr)
                    test_acc = accuracy(test_logits, y_te)
                    train_loss = F.cross_entropy(train_logits, y_tr).item()
                    test_loss = F.cross_entropy(test_logits, y_te).item()
                norm = weight_l2_norm(model)
                writer.writerow(
                    {
                        "step": step,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                        "weight_l2_norm": norm,
                        "weight_decay": args.weight_decay,
                        "train_frac": args.train_frac,
                        "seed": args.seed,
                    }
                )
                print(
                    f"step={step:6d}  train_acc={train_acc:.3f}  "
                    f"test_acc={test_acc:.3f}  |W|={norm:.2f}"
                )

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
