"""Plot test error vs. hidden width from double-descent sweep CSV(s).

Reads one or more CSVs written by scripts/double_descent_sweep.py, takes
each run's final-epoch train/test error, and plots mean test error (with a
+/-1 SEM band across seeds) against hidden width. With a single series, also
marks the smallest width where mean training error first drops to ~zero
across seeds -- the interpolation threshold. With multiple series (e.g. a
weight-decay or optimizer comparison), overlays one line per series instead
and skips the threshold annotation to keep the chart readable; each
series's own threshold is still printed to the console.

Usage:
    python scripts/plot_double_descent.py
    python scripts/plot_double_descent.py --series results/double_descent.csv=baseline

    # comparison overlay (repeat --series):
    python scripts/plot_double_descent.py --out results/double_descent_regularization.png \\
        --series results/double_descent_wd0.csv="weight decay = 0" \\
        --series results/double_descent_wd0.01.csv="weight decay = 0.01" \\
        --series results/double_descent_wd0.1.csv="weight decay = 0.1"
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import pandas as pd

# Colors from the dataviz skill's reference palette (references/palette.md),
# light mode. Categorical hues assigned in fixed order per series (slot 1,
# 2, 3, ...) -- safe for up to 4 series per the palette's adjacent-pair gate.
SERIES_COLORS = [
    "#2a78d6",  # slot 1, blue
    "#eb6834",  # slot 2, orange
    "#1baf7a",  # slot 3, aqua
    "#eda100",  # slot 4, yellow
]
COLOR_THRESHOLD = "#898781"  # muted ink -- an annotation, not a data series
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


def summarize(final_df: pd.DataFrame) -> pd.DataFrame:
    """Per-width mean and standard error of the mean, across seeds."""
    grouped = final_df.groupby("width")["test_error"]
    summary = grouped.agg(mean="mean", std="std", n="count").sort_index()
    summary["sem"] = summary["std"] / summary["n"] ** 0.5
    return summary


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


def parse_series(spec: str) -> tuple[Path, str]:
    path_str, _, label = spec.partition("=")
    path = Path(path_str)
    return path, (label or path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--series",
        action="append",
        type=parse_series,
        metavar="CSV[=LABEL]",
        help="repeatable; defaults to a single series from results/double_descent.csv",
    )
    parser.add_argument("--out", type=Path, default=Path("results/double_descent.png"))
    parser.add_argument(
        "--threshold-tol",
        type=float,
        default=0.01,
        help="mean train error below this counts as 'interpolated' (default 1%%)",
    )
    args = parser.parse_args()

    series = args.series or [(Path("results/double_descent.csv"), "mean test error")]
    if len(series) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} series are supported for color safety")

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    all_widths = set()
    global_max = 0.0
    for (csv_path, label), color in zip(series, SERIES_COLORS):
        final_df = load_final_epoch(csv_path)
        summary = summarize(final_df)
        widths = summary.index.to_numpy()
        all_widths.update(widths)
        global_max = max(global_max, (summary["mean"] + summary["sem"]).max())

        ax.fill_between(
            widths,
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(widths, summary["mean"], color=color, linewidth=2, marker="o", markersize=5, label=label)

        threshold = find_interpolation_threshold(final_df, tol=args.threshold_tol)
        print(f"\n[{label}] ({csv_path})")
        print(summary[["mean", "sem"]].round(4).to_string())
        if threshold is not None:
            print(
                f"Interpolation threshold (first width with mean train_error "
                f"<= {args.threshold_tol:.0%}): {threshold}"
            )
        else:
            print(f"No width reached mean train_error <= {args.threshold_tol:.0%}.")

        if len(series) == 1 and threshold is not None:
            ax.axvline(
                threshold,
                color=COLOR_THRESHOLD,
                linestyle="--",
                linewidth=1.5,
                label=f"interpolation threshold (width = {threshold})",
            )

    widths_sorted = sorted(all_widths)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths_sorted)
    ax.set_xticklabels([str(w) for w in widths_sorted], rotation=45 if len(widths_sorted) > 12 else 0)
    ax.set_xlabel("hidden width", color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("test error", color=COLOR_TEXT_SECONDARY)
    title = (
        "Test error vs. hidden width (label-noise double descent)"
        if len(series) == 1
        else "Test error vs. hidden width -- series comparison"
    )
    ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=12)

    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_MUTED)

    ax.set_ylim(top=global_max * 1.2)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, loc="upper right", fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
