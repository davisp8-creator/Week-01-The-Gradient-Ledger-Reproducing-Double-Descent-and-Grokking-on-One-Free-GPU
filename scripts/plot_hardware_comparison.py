"""Plot per-run duration, CPU vs. GPU, from two run_full_suite.py manifests.

Two stacked panels (log-scale y, since durations span ~1s to ~5min): the
quick checks + double-descent sweep on top, the grokking sweep below.
Grouped bars, one pair per run label.

Usage:
    python scripts/plot_hardware_comparison.py \\
        --manifest-a results/manifest_cpu.json --label-a "CPU (VM)" \\
        --manifest-b results/manifest_t4gpu.json --label-b "T4 GPU" \\
        --out results/hardware_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import numpy as np

COLOR_A = "#2a78d6"  # slot 1, blue
COLOR_B = "#eb6834"  # slot 2, orange
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"

QUICK_AND_DD = [
    "environment_verification",
    "gradient_check",
    "double_descent_baseline",
    "double_descent_wd0.01",
    "double_descent_wd0.1",
    "double_descent_sgd",
]
GROKKING = [
    "grokking_wd1.0_frac0.4_seed0",
    "grokking_wd0.0",
    "grokking_wd0.1",
    "grokking_wd0.5",
    "grokking_wd2.0",
    "grokking_frac0.2",
    "grokking_frac0.3",
    "grokking_frac0.5",
    "grokking_frac0.6",
    "grokking_seed1",
    "grokking_seed2",
    "grokking_seed3",
    "grokking_seed4",
]


def load_durations(manifest_path: Path) -> dict[str, float]:
    manifest = json.loads(manifest_path.read_text())
    return {r["label"]: r["duration_s"] for r in manifest["runs"]}


def plot_panel(ax, labels, dur_a, dur_b, label_a, label_b, ylabel):
    present = [l for l in labels if l in dur_a and l in dur_b]
    x = np.arange(len(present))
    width = 0.38
    a_vals = [dur_a[l] for l in present]
    b_vals = [dur_b[l] for l in present]

    ax.bar(x - width / 2, a_vals, width, color=COLOR_A, label=label_a)
    ax.bar(x + width / 2, b_vals, width, color=COLOR_B, label=label_b)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("grokking_", "").replace("double_descent_", "") for l in present],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color=COLOR_TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_TEXT_MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--manifest-b", type=Path, required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/hardware_comparison.png"))
    args = parser.parse_args()

    dur_a = load_durations(args.manifest_a)
    dur_b = load_durations(args.manifest_b)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8), facecolor=SURFACE)
    plot_panel(ax_top, QUICK_AND_DD, dur_a, dur_b, args.label_a, args.label_b, "duration (s, log scale)")
    plot_panel(ax_bottom, GROKKING, dur_a, dur_b, args.label_a, args.label_b, "duration (s, log scale)")

    ax_top.set_title("Quick checks + double descent", color=COLOR_TEXT_PRIMARY, fontsize=12, loc="left")
    ax_bottom.set_title("Grokking (20,000 steps each)", color=COLOR_TEXT_PRIMARY, fontsize=12, loc="left")
    ax_top.legend(frameon=False, labelcolor=COLOR_TEXT_SECONDARY, fontsize=9, loc="upper left")

    fig.suptitle(f"Per-run duration: {args.label_a} vs. {args.label_b}", color=COLOR_TEXT_PRIMARY, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    total_a = sum(dur_a.get(l, 0) for l in QUICK_AND_DD + GROKKING)
    total_b = sum(dur_b.get(l, 0) for l in QUICK_AND_DD + GROKKING)
    print(f"{args.label_a} total: {total_a:.1f}s | {args.label_b} total: {total_b:.1f}s | "
          f"overall speedup: {total_a / total_b:.2f}x")


if __name__ == "__main__":
    main()
