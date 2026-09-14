"""Run the entire experiment suite once, tagged by hardware, and record
per-run wall-clock timing plus hardware info -- so results from different
devices (this repo's CPU baseline vs. a Colab T4 GPU, say) are directly
comparable and never overwrite each other.

Every output file is suffixed with --tag: results/double_descent_<tag>.csv,
results/grokking_<tag>_wd0.5.csv, etc. A manifest at
results/manifest_<tag>.json records each run's parameters, duration, and
key derived result (interpolation threshold / grokking transition step),
plus a hardware snapshot from scripts/hardware_info.py. Feed that manifest
to scripts/generate_report.py to build report_<tag>.html.

This reruns the exact suite documented in report.html: 2 quick checks, 4
double-descent sweep variants, and 13 grokking runs (5 weight-decay values
+ 4 more train fractions + 4 more seeds, minus the shared baseline). On a
CPU this takes roughly an hour; expect it to be much faster on a GPU --
that speed difference is the point of rerunning it.

Usage:
    python scripts/run_full_suite.py --tag cpu
    python scripts/run_full_suite.py --tag t4gpu
    python scripts/run_full_suite.py --tag t4gpu --skip-grokking-seeds  # faster, drops axis 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardware_info  # noqa: E402
from plot_double_descent import find_interpolation_threshold, load_final_epoch, summarize  # noqa: E402
from plot_grokking_sweep import find_grokking_step  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], label: str, capture: bool = False) -> tuple[float, str | None]:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    start = time.perf_counter()
    if capture:
        proc = subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout)
        stdout = proc.stdout
    else:
        subprocess.run(cmd, check=True, cwd=ROOT)
        stdout = None
    duration = time.perf_counter() - start
    print(f"--- {label}: {duration:.1f}s ---")
    return duration, stdout


def dd_result(csv_path: Path) -> dict:
    final_df = load_final_epoch(csv_path)
    summary = summarize(final_df)
    threshold = find_interpolation_threshold(final_df)
    peak_width = int(summary["mean"].idxmax())
    return {
        "interpolation_threshold": threshold,
        "peak_test_error": round(float(summary["mean"].max()), 4),
        "peak_width": peak_width,
    }


def gk_result(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    transition = find_grokking_step(df, 0.99)
    final_test_acc = round(float(df.iloc[-1]["test_acc"]), 4)
    return {"transition_step": transition, "final_test_acc": final_test_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="e.g. cpu, t4gpu -- suffixes every output file")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-grokking-seeds", action="store_true", help="skip the 4 extra seed runs (axis 3)")
    parser.add_argument("--dd-epochs", type=int, default=1000, help="override for a quick smoke test")
    parser.add_argument("--gk-steps", type=int, default=20_000, help="override for a quick smoke test")
    args = parser.parse_args()

    py = sys.executable
    tag = args.tag
    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"tag": tag, "runs": []}

    def record(label: str, cmd: list[str], result_fn=None, csv_for_result: Path | None = None,
               capture: bool = False, **params):
        duration, stdout = run(cmd, label, capture=capture)
        entry = {"label": label, "duration_s": round(duration, 3), "params": params}
        if stdout is not None:
            entry["stdout"] = stdout
        if result_fn is not None and csv_for_result is not None:
            entry["result"] = result_fn(csv_for_result)
        manifest["runs"].append(entry)

    # -- quick checks (stdout captured for the report) --
    record("environment_verification", [py, "scripts/test_environment.py"], capture=True)
    record("gradient_check", [py, "scripts/gradient_check.py"], capture=True)

    # -- double descent: baseline + 2 regularized + 1 optimizer ablation --
    epochs = str(args.dd_epochs)
    dd = results / f"double_descent_{tag}.csv"
    record(
        "double_descent_baseline",
        [py, "scripts/double_descent_sweep.py", "--weight-decay", "0.0", "--epochs", epochs, "--out", str(dd)],
        result_fn=dd_result, csv_for_result=dd, optimizer="adam", weight_decay=0.0,
    )
    dd_wd01 = results / f"double_descent_{tag}_wd0.01.csv"
    record(
        "double_descent_wd0.01",
        [py, "scripts/double_descent_sweep.py", "--weight-decay", "0.01", "--epochs", epochs, "--out", str(dd_wd01)],
        result_fn=dd_result, csv_for_result=dd_wd01, optimizer="adam", weight_decay=0.01,
    )
    dd_wd1 = results / f"double_descent_{tag}_wd0.1.csv"
    record(
        "double_descent_wd0.1",
        [py, "scripts/double_descent_sweep.py", "--weight-decay", "0.1", "--epochs", epochs, "--out", str(dd_wd1)],
        result_fn=dd_result, csv_for_result=dd_wd1, optimizer="adam", weight_decay=0.1,
    )
    dd_sgd = results / f"double_descent_{tag}_sgd.csv"
    record(
        "double_descent_sgd",
        [py, "scripts/double_descent_sweep.py", "--optimizer", "sgd", "--lr", "0.1",
         "--momentum", "0.9", "--weight-decay", "0.0", "--epochs", epochs, "--out", str(dd_sgd)],
        result_fn=dd_result, csv_for_result=dd_sgd, optimizer="sgd", lr=0.1, momentum=0.9, weight_decay=0.0,
    )

    subprocess.run([py, "scripts/plot_double_descent.py", "--out", str(results / f"double_descent_{tag}.png"),
                     "--series", f"{dd}=mean test error"], check=True, cwd=ROOT)
    subprocess.run([py, "scripts/plot_double_descent.py",
                     "--out", str(results / f"double_descent_{tag}_regularization.png"),
                     "--series", f"{dd}=weight decay = 0",
                     "--series", f"{dd_wd01}=weight decay = 0.01",
                     "--series", f"{dd_wd1}=weight decay = 0.1"], check=True, cwd=ROOT)
    subprocess.run([py, "scripts/plot_double_descent.py",
                     "--out", str(results / f"double_descent_{tag}_optimizer.png"),
                     "--series", f"{dd}=Adam",
                     "--series", f"{dd_sgd}=SGD (momentum=0.9)"], check=True, cwd=ROOT)
    subprocess.run([py, "scripts/plot_double_descent_epochwise.py", "--csv", str(dd),
                     "--out", str(results / f"double_descent_{tag}_epochwise.png")], check=True, cwd=ROOT)

    # -- grokking: baseline + weight-decay axis + train-fraction axis + seed axis --
    steps = str(args.gk_steps)
    gk_base = results / f"grokking_{tag}.csv"
    record(
        "grokking_wd1.0_frac0.4_seed0",
        [py, "scripts/grokking_train.py", "--steps", steps, "--out", str(gk_base)],
        result_fn=gk_result, csv_for_result=gk_base, weight_decay=1.0, train_frac=0.4, seed=0,
    )

    wd_files = {1.0: gk_base}
    for wd in (0.0, 0.1, 0.5, 2.0):
        f = results / f"grokking_{tag}_wd{wd}.csv"
        record(
            f"grokking_wd{wd}",
            [py, "scripts/grokking_train.py", "--weight-decay", str(wd), "--steps", steps, "--out", str(f)],
            result_fn=gk_result, csv_for_result=f, weight_decay=wd, train_frac=0.4, seed=0,
        )
        wd_files[wd] = f

    frac_files = {0.4: gk_base}
    for frac in (0.2, 0.3, 0.5, 0.6):
        f = results / f"grokking_{tag}_frac{frac}.csv"
        record(
            f"grokking_frac{frac}",
            [py, "scripts/grokking_train.py", "--train-frac", str(frac), "--steps", steps, "--out", str(f)],
            result_fn=gk_result, csv_for_result=f, weight_decay=1.0, train_frac=frac, seed=0,
        )
        frac_files[frac] = f

    seed_files = {0: gk_base}
    if not args.skip_grokking_seeds:
        for seed in (1, 2, 3, 4):
            f = results / f"grokking_{tag}_seed{seed}.csv"
            record(
                f"grokking_seed{seed}",
                [py, "scripts/grokking_train.py", "--seed", str(seed), "--steps", steps, "--out", str(f)],
                result_fn=gk_result, csv_for_result=f, weight_decay=1.0, train_frac=0.4, seed=seed,
            )
            seed_files[seed] = f

    subprocess.run([py, "scripts/plot_grokking.py", "--csv", str(gk_base),
                     "--out", str(results / f"grokking_{tag}.png")], check=True, cwd=ROOT)

    wd_series = [x for wd, f in sorted(wd_files.items()) for x in ("--series", f"{f}=weight decay = {wd}")]
    subprocess.run([py, "scripts/plot_grokking_sweep.py",
                     "--out", str(results / f"grokking_{tag}_weight_decay_sweep.png"),
                     "--title", f"Grokking vs. weight decay ({tag})"] + wd_series, check=True, cwd=ROOT)

    frac_series = [x for frac, f in sorted(frac_files.items()) for x in ("--series", f"{f}=train frac = {frac}")]
    subprocess.run([py, "scripts/plot_grokking_sweep.py",
                     "--out", str(results / f"grokking_{tag}_train_frac_sweep.png"),
                     "--title", f"Grokking vs. train fraction ({tag})"] + frac_series, check=True, cwd=ROOT)

    if not args.skip_grokking_seeds:
        seed_series = [x for seed, f in sorted(seed_files.items()) for x in ("--series", f"{f}=seed = {seed}")]
        subprocess.run([py, "scripts/plot_grokking_sweep.py",
                         "--out", str(results / f"grokking_{tag}_seed_sweep.png"),
                         "--title", f"Grokking across seeds ({tag})"] + seed_series, check=True, cwd=ROOT)

    manifest["hardware"] = hardware_info.collect()
    manifest_path = results / f"manifest_{tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    total_s = sum(r["duration_s"] for r in manifest["runs"])
    print(f"\nWrote {manifest_path}")
    print(f"Total training wall-clock: {total_s:.1f}s ({total_s / 60:.1f} min) across {len(manifest['runs'])} runs")
    print(f"\nNext: python scripts/generate_report.py --tag {tag}")


if __name__ == "__main__":
    main()
