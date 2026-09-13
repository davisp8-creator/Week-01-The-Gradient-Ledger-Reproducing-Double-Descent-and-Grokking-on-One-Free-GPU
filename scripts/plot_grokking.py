"""Plot train/test accuracy, loss, and weight norm vs. step for grokking.

Reads results/grokking.csv (schema documented in scripts/grokking_train.py)
and draws three stacked panels sharing a step axis: train/test accuracy,
train/test loss (log scale -- loss often shows the generalizing solution
winning out before the hard accuracy metric flips), and weight L2 norm.
Marks the step where test accuracy first crosses 99% -- the grokking
transition -- as a vertical line on all three panels, so the loss dip and
weight-norm dip that drive it line up visually with the accuracy jump.

Usage:
    python scripts/plot_grokking.py
    python scripts/plot_grokking.py --csv results/grokking.csv --out results/grokking.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import pandas as pd

# Colors from the dataviz skill's reference palette (references/palette.md),
# light mode. Categorical hues assigned in fixed order: slot 1 for train
# accuracy, slot 2 for test accuracy; slot 3 for weight norm keeps the
# bottom panel visually distinct from both accuracy lines above it.
COLOR_TRAIN = "#2a78d6"  # slot 1, blue
COLOR_TEST = "#eb6834"  # slot 2, orange
COLOR_NORM = "#1baf7a"  # slot 3, aqua
COLOR_TRANSITION = "#898781"  # muted ink -- an annotation, not a data series
COLOR_CHANCE = "#c3c2b7"  # baseline/axis gray
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def find_grokking_step(df: pd.DataFrame, test_acc_threshold: float = 0.99) -> int | None:
    """First step where test accuracy crosses the given threshold."""
    grokked = df[df["test_acc"] >= test_acc_threshold]
    return int(grokked["step"].iloc[0]) if len(grokked) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/grokking.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/grokking.png"))
    parser.add_argument("--modulus", type=int, default=97, help="for the chance-level reference line")
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    transition_step = find_grokking_step(df, args.threshold)
    chance_level = 1.0 / args.modulus

    has_loss = "train_loss" in df.columns

    n_panels = 3 if has_loss else 2
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(7, 6.5 if n_panels == 2 else 8.5), sharex=True,
        gridspec_kw={"height_ratios": ([1.3, 1] if n_panels == 2 else [1.2, 1, 1])},
        facecolor=SURFACE,
    )
    ax_acc, ax_loss, ax_norm = (axes[0], axes[1], axes[2]) if has_loss else (axes[0], None, axes[1])

    # --- top panel: train/test accuracy ---
    ax_acc.set_facecolor(SURFACE)
    ax_acc.axhline(chance_level, color=COLOR_CHANCE, linestyle=":", linewidth=1, label="chance level")
    ax_acc.plot(df["step"], df["train_acc"], color=COLOR_TRAIN, linewidth=2, label="train accuracy")
    ax_acc.plot(df["step"], df["test_acc"], color=COLOR_TEST, linewidth=2, label="test accuracy")
    ax_acc.set_ylabel("accuracy", color=COLOR_TEXT_SECONDARY)
    ax_acc.set_title(
        "Grokking on modular addition (mod 97): accuracy, loss, and weight norm vs. step"
        if has_loss
        else "Grokking on modular addition (mod 97): accuracy and weight norm vs. step",
        color=COLOR_TEXT_PRIMARY, fontsize=12,
    )
    ax_acc.set_ylim(-0.05, 1.05)

    # --- middle panel: train/test loss (log scale) ---
    if has_loss:
        ax_loss.set_facecolor(SURFACE)
        ax_loss.plot(df["step"], df["train_loss"], color=COLOR_TRAIN, linewidth=2, label="train loss")
        ax_loss.plot(df["step"], df["test_loss"], color=COLOR_TEST, linewidth=2, label="test loss")
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("cross-entropy loss", color=COLOR_TEXT_SECONDARY)
        ax_loss.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, loc="upper right", fontsize=9)

    # --- bottom panel: weight L2 norm ---
    ax_norm.set_facecolor(SURFACE)
    ax_norm.plot(df["step"], df["weight_l2_norm"], color=COLOR_NORM, linewidth=2, label="weight L2 norm")
    ax_norm.set_ylabel("weight L2 norm", color=COLOR_TEXT_SECONDARY)
    ax_norm.set_xlabel("training step", color=COLOR_TEXT_SECONDARY)

    panels = (ax_acc, ax_loss, ax_norm) if has_loss else (ax_acc, ax_norm)
    if transition_step is not None:
        for ax in panels:
            ax.axvline(transition_step, color=COLOR_TRANSITION, linestyle="--", linewidth=1.5)
        ax_acc.plot([], [], color=COLOR_TRANSITION, linestyle="--", linewidth=1.5,
                    label=f"grokking transition (step = {transition_step})")

    for ax in panels:
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_AXIS)
        ax.tick_params(colors=COLOR_TEXT_MUTED)

    ax_acc.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, loc="center right", fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    if transition_step is not None:
        row = df[df["step"] == transition_step].iloc[0]
        print(
            f"\nGrokking transition at step {transition_step}: "
            f"test_acc={row['test_acc']:.3f}, weight_l2_norm={row['weight_l2_norm']:.2f}"
        )
        print(f"Weight norm peak: {df['weight_l2_norm'].max():.2f} at step {int(df.loc[df['weight_l2_norm'].idxmax(), 'step'])}")
        print(f"Weight norm trough: {df['weight_l2_norm'].min():.2f} at step {int(df.loc[df['weight_l2_norm'].idxmin(), 'step'])}")
    else:
        print(f"\nTest accuracy never reached {args.threshold:.0%} in this run.")


if __name__ == "__main__":
    main()
