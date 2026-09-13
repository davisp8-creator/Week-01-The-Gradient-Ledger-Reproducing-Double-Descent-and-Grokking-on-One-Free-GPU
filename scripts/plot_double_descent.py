"""Plot test error vs. hidden width from the double-descent sweep CSV.

Reads results/double_descent.csv (schema documented in
scripts/double_descent_sweep.py), takes each run's final-epoch train/test
error, and plots test error against hidden width with a shaded band across
seeds. Marks the smallest width where mean training error first reaches
zero across seeds -- the interpolation threshold.

Usage:
    python scripts/plot_double_descent.py
    python scripts/plot_double_descent.py --csv results/double_descent.csv --out results/double_descent.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import pandas as pd

# Colors from the dataviz skill's reference palette (references/palette.md),
# light mode.
COLOR_LINE = "#2a78d6"  # categorical slot 1 (blue) -- the mean test-error line
COLOR_BAND = "#2a78d6"  # same hue, low alpha, for the seed-range fill
COLOR_THRESHOLD = "#eb6834"  # categorical slot 2 (orange) -- distinct annotation
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def load_final_epoch(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    last_epoch = df.groupby(["width", "seed"])["epoch"].transform("max")
    return df[df["epoch"] == last_epoch].copy()


def find_interpolation_threshold(final_df: pd.DataFrame, tol: float = 0.01) -> int | None:
    """Smallest width whose mean (across seeds) train error is ~zero.

    Requiring *exactly* zero can lag well behind where models have
    effectively interpolated, if even one seed is still inching down --
    `tol` (default 1% mean training error) matches how the literature
    usually defines "reached the interpolation threshold".
    """
    by_width = final_df.groupby("width")["train_error"].mean().sort_index()
    zeroed = by_width[by_width <= tol]
    return int(zeroed.index[0]) if len(zeroed) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/double_descent.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/double_descent.png"))
    parser.add_argument(
        "--threshold-tol",
        type=float,
        default=0.01,
        help="mean train error below this counts as 'interpolated' (default 1%%)",
    )
    args = parser.parse_args()

    final_df = load_final_epoch(args.csv)
    summary = (
        final_df.groupby("width")["test_error"]
        .agg(mean="mean", min="min", max="max")
        .sort_index()
    )
    threshold = find_interpolation_threshold(final_df, tol=args.threshold_tol)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    widths = summary.index.to_numpy()
    ax.fill_between(
        widths,
        summary["min"],
        summary["max"],
        color=COLOR_BAND,
        alpha=0.18,
        linewidth=0,
        label="seed range (min-max)",
    )
    ax.plot(
        widths,
        summary["mean"],
        color=COLOR_LINE,
        linewidth=2,
        marker="o",
        markersize=5,
        label="mean test error",
    )

    if threshold is not None:
        ax.axvline(
            threshold,
            color=COLOR_THRESHOLD,
            linestyle="--",
            linewidth=1.5,
            label=f"interpolation threshold (width = {threshold})",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("hidden width", color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("test error", color=COLOR_TEXT_SECONDARY)
    ax.set_title(
        "Test error vs. hidden width (label-noise double descent)",
        color=COLOR_TEXT_PRIMARY,
        fontsize=12,
    )

    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_MUTED)

    ax.set_ylim(top=summary["max"].max() * 1.2)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, loc="upper right", fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    print("\nPer-width summary (final epoch, across seeds):")
    print(summary.round(3).to_string())
    if threshold is not None:
        print(
            f"\nInterpolation threshold (first width with mean train_error "
            f"<= {args.threshold_tol:.0%}): {threshold}"
        )
    else:
        print(f"\nNo width reached mean train_error <= {args.threshold_tol:.0%} in this sweep.")


if __name__ == "__main__":
    main()
