"""Compare grokking runs across a swept parameter (weight decay, train
fraction, ...): overlay test accuracy and weight L2 norm vs. step, one line
per run, and print each run's grokking transition step (or "never").

Works for any sweep dimension -- pass one --series per CSV produced by
scripts/grokking_train.py at a different setting.

Usage:
    python scripts/plot_grokking_sweep.py --out results/grokking_weight_decay_sweep.png \\
        --series results/grokking_wd0.csv="weight decay = 0" \\
        --series results/grokking_wd0.1.csv="weight decay = 0.1" \\
        --series results/grokking_wd0.5.csv="weight decay = 0.5" \\
        --series results/grokking.csv="weight decay = 1.0" \\
        --series results/grokking_wd2.csv="weight decay = 2.0"

    python scripts/plot_grokking_sweep.py --out results/grokking_train_frac_sweep.png \\
        --series results/grokking_frac0.2.csv="train frac = 0.2" \\
        --series results/grokking_frac0.3.csv="train frac = 0.3" \\
        --series results/grokking.csv="train frac = 0.4" \\
        --series results/grokking_frac0.5.csv="train frac = 0.5" \\
        --series results/grokking_frac0.6.csv="train frac = 0.6"
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
    "#e87ba4",  # slot 5, magenta
]
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def parse_series(spec: str) -> tuple[Path, str]:
    path_str, _, label = spec.partition("=")
    path = Path(path_str)
    return path, (label or path.stem)


def find_grokking_step(df: pd.DataFrame, threshold: float) -> int | None:
    grokked = df[df["test_acc"] >= threshold]
    return int(grokked["step"].iloc[0]) if len(grokked) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--series", action="append", type=parse_series, required=True, metavar="CSV=LABEL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Grokking sweep comparison")
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    if len(args.series) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} series are supported for color safety")

    fig, (ax_acc, ax_norm) = plt.subplots(
        2, 1, figsize=(7, 6.5), sharex=True, facecolor=SURFACE
    )

    print(f"{'label':28s} {'transition step':>16s}")
    print("-" * 46)
    for (csv_path, label), color in zip(args.series, SERIES_COLORS):
        df = pd.read_csv(csv_path)
        transition = find_grokking_step(df, args.threshold)
        ax_acc.plot(df["step"], df["test_acc"], color=color, linewidth=2, label=label)
        ax_norm.plot(df["step"], df["weight_l2_norm"], color=color, linewidth=2, label=label)
        print(f"{label:28s} {transition if transition is not None else 'never':>16}")

    ax_acc.set_ylabel("test accuracy", color=COLOR_TEXT_SECONDARY)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_title(args.title, color=COLOR_TEXT_PRIMARY, fontsize=12)
    ax_norm.set_ylabel("weight L2 norm", color=COLOR_TEXT_SECONDARY)
    ax_norm.set_xlabel("training step", color=COLOR_TEXT_SECONDARY)

    for ax in (ax_acc, ax_norm):
        ax.set_facecolor(SURFACE)
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
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
