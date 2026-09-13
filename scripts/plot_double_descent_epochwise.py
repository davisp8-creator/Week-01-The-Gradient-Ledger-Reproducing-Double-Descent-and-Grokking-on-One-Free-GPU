"""Plot train/test error vs. epoch for a few representative widths.

The main double-descent plot only looks at each run's final epoch. This
script uses the same CSV (every epoch is already logged) to show *how*
under-, near-threshold, and over-parameterized models get there: does a
narrow model plateau above zero train error, does a near-threshold model
overfit late, does a wide model sail to zero train error with a stable test
curve? Mean across seeds per width; one line per selected width per panel.

Usage:
    python scripts/plot_double_descent_epochwise.py
    python scripts/plot_double_descent_epochwise.py --widths 2 38 575
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import pandas as pd

SERIES_COLORS = [
    "#2a78d6",  # slot 1, blue
    "#eb6834",  # slot 2, orange
    "#1baf7a",  # slot 3, aqua
    "#eda100",  # slot 4, yellow
]
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def pick_representative_widths(widths: list[int], n: int = 4) -> list[int]:
    """n widths spread across the available range: smallest, largest, and
    evenly log-spaced points in between (deduplicated)."""
    widths = sorted(widths)
    if len(widths) <= n:
        return widths
    fractions = [i / (n - 1) for i in range(n)]
    idx = sorted({round(f * (len(widths) - 1)) for f in fractions})
    return [int(widths[i]) for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/double_descent.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/double_descent_epochwise.png"))
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=None,
        help="defaults to 4 widths spread across the sweep's range",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    available = sorted(df["width"].unique())
    widths = args.widths or pick_representative_widths(available)
    missing = [w for w in widths if w not in available]
    if missing:
        raise SystemExit(f"widths {missing} not present in {args.csv}; available: {available}")

    fig, (ax_train, ax_test) = plt.subplots(
        2, 1, figsize=(7, 6.5), sharex=True, facecolor=SURFACE
    )

    for width, color in zip(widths, SERIES_COLORS):
        sub = df[df["width"] == width].groupby("epoch")[["train_error", "test_error"]].mean()
        ax_train.plot(sub.index, sub["train_error"], color=color, linewidth=2, label=f"width = {width}")
        ax_test.plot(sub.index, sub["test_error"], color=color, linewidth=2, label=f"width = {width}")

    ax_train.set_ylabel("train error", color=COLOR_TEXT_SECONDARY)
    ax_test.set_ylabel("test error", color=COLOR_TEXT_SECONDARY)
    ax_test.set_xlabel("epoch", color=COLOR_TEXT_SECONDARY)
    ax_train.set_title(
        "Train/test error vs. epoch, by hidden width", color=COLOR_TEXT_PRIMARY, fontsize=12
    )

    for ax in (ax_train, ax_test):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_AXIS)
        ax.tick_params(colors=COLOR_TEXT_MUTED)

    ax_train.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, loc="upper right", fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    print(f"Widths plotted: {widths}")


if __name__ == "__main__":
    main()
