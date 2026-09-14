"""Build a side-by-side hardware comparison report from two
run_full_suite.py manifests: per-run timing with speedup factors, a
results-consistency check (does the science agree across hardware, not
just the speed), and the comparison chart from
scripts/plot_hardware_comparison.py.

Usage:
    python scripts/plot_hardware_comparison.py \\
        --manifest-a results/manifest_cpu.json --label-a "CPU (VM)" \\
        --manifest-b results/manifest_t4gpu.json --label-b "T4 GPU"
    python scripts/generate_comparison_report.py \\
        --manifest-a results/manifest_cpu.json --label-a "CPU (VM)" \\
        --manifest-b results/manifest_t4gpu.json --label-b "T4 GPU" \\
        --out ComparisonReport.html
"""

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
  :root {
    color-scheme: light;
    --paper: #faf8f1; --band: #eef2e6; --band-strong: #e3ead9;
    --ink: #202b23; --ink-soft: #55605a; --ink-faint: #85897f;
    --rule: #c7cec0; --rule-strong: #9aa693;
    --accent: #9a3324; --accent-soft: #f2e2dc;
    --good: #2c6e3f; --good-soft: #e3ede5;
    --brass: #8a6a2f; --brass-soft: #efe6d1; --link: #2c5a4a;
    --font-display: "IBM Plex Serif", Georgia, "Times New Roman", serif;
    --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Code", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --paper: #14181a; --band: #1b241f; --band-strong: #212d25;
      --ink: #e7ece4; --ink-soft: #b7bfb5; --ink-faint: #7c877e;
      --rule: #333e37; --rule-strong: #48564c;
      --accent: #e2795f; --accent-soft: #3a241d;
      --good: #6fbf8b; --good-soft: #1c2e22;
      --brass: #cca757; --brass-soft: #362c1a; --link: #7fbfa2;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --paper: #14181a; --band: #1b241f; --band-strong: #212d25;
    --ink: #e7ece4; --ink-soft: #b7bfb5; --ink-faint: #7c877e;
    --rule: #333e37; --rule-strong: #48564c;
    --accent: #e2795f; --accent-soft: #3a241d;
    --good: #6fbf8b; --good-soft: #1c2e22;
    --brass: #cca757; --brass-soft: #362c1a; --link: #7fbfa2;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--paper); color: var(--ink); font-family: var(--font-body);
    font-size: 15.5px; line-height: 1.6; margin: 0;
    padding-inline: 20px; padding-block: 40px 80px;
  }
  .sheet { max-width: 900px; margin-inline: auto; }
  h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; margin: 0; }
  .eyebrow {
    font-family: var(--font-mono); font-size: 11.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); font-weight: 600;
  }
  .masthead { border-bottom: 3px double var(--rule-strong); padding-bottom: 22px; margin-bottom: 28px; }
  .masthead h1 { font-size: clamp(2rem, 5vw, 2.75rem); font-weight: 700; margin-top: 6px; }
  .masthead p.dek { max-width: 66ch; color: var(--ink-soft); margin-top: 10px; font-size: 1.02rem; }

  .headline {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;
    margin-block: 24px 34px;
  }
  .headline .tile {
    border: 1px solid var(--rule); background: var(--band); border-radius: 4px; padding: 14px 16px;
  }
  .headline .tile .k {
    font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-faint);
  }
  .headline .tile .v {
    font-family: var(--font-mono); font-variant-numeric: tabular-nums;
    font-size: 22px; font-weight: 600; color: var(--ink); margin-top: 4px;
  }
  .headline .tile .sub { font-size: 12px; color: var(--ink-soft); margin-top: 3px; }
  .headline .tile.good .v { color: var(--good); }
  .headline .tile.bad .v { color: var(--accent); }

  h2.section { font-size: 1.4rem; font-weight: 600; margin-top: 48px; }
  p.lede { color: var(--ink-soft); max-width: 70ch; margin-top: 8px; }

  table.cmp { width: 100%; border-collapse: collapse; margin-block: 16px; font-size: 13.5px; }
  table.cmp caption {
    text-align: left; font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--ink-faint); margin-bottom: 6px;
  }
  table.cmp th {
    text-align: left; font-family: var(--font-mono); font-weight: 600; font-size: 11px;
    letter-spacing: 0.03em; text-transform: uppercase; color: var(--ink-soft);
    border-bottom: 1px solid var(--rule-strong); padding: 7px 10px;
  }
  table.cmp td {
    padding: 7px 10px; border-bottom: 1px solid var(--rule);
    font-variant-numeric: tabular-nums; vertical-align: top;
  }
  table.cmp tbody tr:nth-child(odd) { background: var(--band); }
  table.cmp td.num { font-family: var(--font-mono); }
  table.cmp td.faster { color: var(--good); font-family: var(--font-mono); font-weight: 600; }
  table.cmp td.slower { color: var(--accent); font-family: var(--font-mono); font-weight: 600; }
  table.cmp tr.group-row td { font-weight: 600; background: var(--band-strong); }
  table.cmp td.match { color: var(--good); }
  table.cmp td.mismatch { color: var(--accent); }

  figure { margin: 0; border: 1px solid var(--rule); background: var(--paper); border-radius: 3px; overflow: hidden; }
  figure img { display: block; width: 100%; height: auto; }
  figcaption {
    font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft);
    padding: 9px 12px; border-top: 1px solid var(--rule); background: var(--band);
  }
  .rule { border: none; border-top: 1px solid var(--rule); margin-block: 40px; }
  footer {
    margin-top: 56px; padding-top: 22px; border-top: 3px double var(--rule-strong);
    font-family: var(--font-mono); font-size: 12.5px; color: var(--ink-soft);
    display: flex; flex-wrap: wrap; gap: 8px 28px; justify-content: space-between;
  }
"""

QUICK_ORDER = [
    ("environment_verification", "Environment verification"),
    ("gradient_check", "Gradient check"),
]
DD_ORDER = [
    ("double_descent_baseline", "Double descent: baseline"),
    ("double_descent_wd0.01", "Double descent: wd=0.01"),
    ("double_descent_wd0.1", "Double descent: wd=0.1"),
    ("double_descent_sgd", "Double descent: SGD"),
]
QUICK_AND_DD_ORDER = QUICK_ORDER + DD_ORDER
GROKKING_ORDER = [
    ("grokking_wd1.0_frac0.4_seed0", "Grokking baseline (wd=1.0, frac=0.4, seed=0)"),
    ("grokking_wd0.0", "Grokking: wd=0.0"),
    ("grokking_wd0.1", "Grokking: wd=0.1"),
    ("grokking_wd0.5", "Grokking: wd=0.5"),
    ("grokking_wd2.0", "Grokking: wd=2.0"),
    ("grokking_frac0.2", "Grokking: frac=0.2"),
    ("grokking_frac0.3", "Grokking: frac=0.3"),
    ("grokking_frac0.5", "Grokking: frac=0.5"),
    ("grokking_frac0.6", "Grokking: frac=0.6"),
    ("grokking_seed1", "Grokking: seed=1"),
    ("grokking_seed2", "Grokking: seed=2"),
    ("grokking_seed3", "Grokking: seed=3"),
    ("grokking_seed4", "Grokking: seed=4"),
]
DD_RESULT_ORDER = [l for l, _ in QUICK_AND_DD_ORDER if l.startswith("double_descent")]
GK_RESULT_ORDER = [l for l, _ in GROKKING_ORDER]


def e(text) -> str:
    return html.escape("" if text is None else str(text))


def fmt_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s" if m else f"{s:.1f}s"


def load(path: Path) -> dict:
    m = json.loads(path.read_text())
    m["by_label"] = {r["label"]: r for r in m["runs"]}
    return m


def hw_row(label: str, a, b) -> str:
    return f'<tr><td>{e(label)}</td><td class="num">{e(a)}</td><td class="num">{e(b)}</td></tr>'


def gpu_desc(hw: dict) -> str:
    if hw.get("cuda_available"):
        return f"{hw.get('gpu_name')} ({hw.get('gpu_vram_gib')} GiB)"
    return "none (CPU only)"


def render_hardware_table(hw_a: dict, hw_b: dict, label_a: str, label_b: str) -> str:
    rows = [
        hw_row("CPU", hw_a.get("cpu_model", "?"), hw_b.get("cpu_model", "?")),
        hw_row("Cores / threads", hw_a.get("cpu_logical_cores", "?"), hw_b.get("cpu_logical_cores", "?")),
        hw_row("Memory", f"{hw_a.get('ram_gib', '?')} GiB", f"{hw_b.get('ram_gib', '?')} GiB"),
        hw_row("GPU", gpu_desc(hw_a), gpu_desc(hw_b)),
        hw_row("OS", hw_a.get("os", "?"), hw_b.get("os", "?")),
        hw_row("Python", hw_a.get("python_version", "?"), hw_b.get("python_version", "?")),
        hw_row("torch", hw_a.get("torch_version", "?"), hw_b.get("torch_version", "?")),
    ]
    return (
        '<table class="cmp"><caption>Hardware &amp; environment</caption>'
        f"<thead><tr><th>Field</th><th>{e(label_a)}</th><th>{e(label_b)}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_timing_table(order, group_label, man_a, man_b, label_a, label_b) -> tuple[str, float, float]:
    rows = [f'<tr class="group-row"><td colspan="4">{e(group_label)}</td></tr>']
    total_a = total_b = 0.0
    for run_label, display in order:
        ra = man_a["by_label"].get(run_label)
        rb = man_b["by_label"].get(run_label)
        if ra is None or rb is None:
            continue
        da, db = ra["duration_s"], rb["duration_s"]
        total_a += da
        total_b += db
        ratio = da / db if db else float("nan")
        cls = "faster" if ratio > 1 else "slower"
        arrow = f"{ratio:.2f}x faster" if ratio > 1 else f"{1/ratio:.2f}x slower"
        rows.append(
            f"<tr><td>{e(display)}</td><td class=\"num\">{fmt_duration(da)}</td>"
            f"<td class=\"num\">{fmt_duration(db)}</td><td class=\"{cls}\">{e(label_b)} {arrow}</td></tr>"
        )
    table = (
        '<table class="cmp">'
        f"<thead><tr><th>Run</th><th>{e(label_a)}</th><th>{e(label_b)}</th><th>Speedup</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return table, total_a, total_b


def render_dd_consistency(man_a: dict, man_b: dict, label_a: str, label_b: str) -> str:
    rows = []
    for run_label in DD_RESULT_ORDER:
        ra, rb = man_a["by_label"].get(run_label), man_b["by_label"].get(run_label)
        if not ra or not rb:
            continue
        res_a, res_b = ra.get("result", {}), rb.get("result", {})
        thr_a, thr_b = res_a.get("interpolation_threshold"), res_b.get("interpolation_threshold")
        match = (thr_a is None) == (thr_b is None)
        cls = "match" if match else "mismatch"
        rows.append(
            f"<tr><td>{e(run_label.replace('double_descent_', ''))}</td>"
            f"<td class=\"num\">{e(res_a.get('peak_test_error'))} @ w{e(res_a.get('peak_width'))}</td>"
            f"<td class=\"num\">{e(res_b.get('peak_test_error'))} @ w{e(res_b.get('peak_width'))}</td>"
            f"<td class=\"num\">{e(thr_a) if thr_a is not None else 'never'}</td>"
            f"<td class=\"num\">{e(thr_b) if thr_b is not None else 'never'}</td>"
            f"<td class=\"{cls}\">{'agree' if match else 'differ'}</td></tr>"
        )
    return (
        '<table class="cmp"><caption>Double descent: does the phenomenon agree across hardware?</caption>'
        f"<thead><tr><th>Variant</th><th>{e(label_a)} peak</th><th>{e(label_b)} peak</th>"
        f"<th>{e(label_a)} threshold</th><th>{e(label_b)} threshold</th><th>Conclusion</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_gk_consistency(man_a: dict, man_b: dict, label_a: str, label_b: str) -> str:
    rows = []
    for run_label in GK_RESULT_ORDER:
        ra, rb = man_a["by_label"].get(run_label), man_b["by_label"].get(run_label)
        if not ra or not rb:
            continue
        res_a, res_b = ra.get("result", {}), rb.get("result", {})
        ta, tb = res_a.get("transition_step"), res_b.get("transition_step")
        match = (ta is None) == (tb is None)
        cls = "match" if match else "mismatch"
        name = run_label.replace("grokking_", "").replace("wd1.0_frac0.4_seed0", "baseline")
        rows.append(
            f"<tr><td>{e(name)}</td>"
            f"<td class=\"num\">{e(ta) if ta is not None else 'never'}</td>"
            f"<td class=\"num\">{e(tb) if tb is not None else 'never'}</td>"
            f"<td class=\"{cls}\">{'agree' if match else 'differ'}</td></tr>"
        )
    return (
        '<table class="cmp"><caption>Grokking: did it grok on both, at roughly the same step?</caption>'
        f"<thead><tr><th>Variant</th><th>{e(label_a)} step</th><th>{e(label_b)} step</th><th>Conclusion</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--manifest-b", type=Path, required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--chart", type=Path, default=Path("results/hardware_comparison.png"))
    parser.add_argument("--out", type=Path, default=Path("ComparisonReport.html"))
    args = parser.parse_args()

    man_a, man_b = load(args.manifest_a), load(args.manifest_b)
    label_a, label_b = args.label_a, args.label_b

    quick_table, quick_total_a, quick_total_b = render_timing_table(
        QUICK_ORDER, "Quick checks (a couple seconds either way -- Python/import startup dominates, not compute)",
        man_a, man_b, label_a, label_b,
    )
    dd_table, dd_total_a, dd_total_b = render_timing_table(
        DD_ORDER, "Double descent sweep", man_a, man_b, label_a, label_b
    )
    gk_table, gk_total_a, gk_total_b = render_timing_table(
        GROKKING_ORDER, "Grokking (20,000 steps each)", man_a, man_b, label_a, label_b
    )

    # "Overall" and the headline tiles cover the 17 training runs (double
    # descent + grokking); the quick checks are shown separately since their
    # timing is noise-dominated, not a meaningful hardware comparison.
    total_a, total_b = dd_total_a + gk_total_a, dd_total_b + gk_total_b
    dd_ratio = dd_total_a / dd_total_b
    gk_ratio = gk_total_a / gk_total_b
    overall_ratio = total_a / total_b
    grand_total_a = quick_total_a + total_a
    grand_total_b = quick_total_b + total_b

    def tile(k, ratio, sub):
        cls = "good" if ratio > 1 else "bad"
        v = f"{ratio:.2f}&times;" if ratio > 1 else f"{1/ratio:.2f}&times; slower"
        return (
            f'<div class="tile {cls}"><div class="k">{e(k)}</div>'
            f'<div class="v">{v}</div><div class="sub">{e(sub)}</div></div>'
        )

    headline = (
        '<div class="headline">'
        + tile("Double descent", dd_ratio, f"{label_b}: {fmt_duration(dd_total_a)} → {fmt_duration(dd_total_b)}")
        + tile("Grokking", gk_ratio, f"{label_b}: {fmt_duration(gk_total_a)} → {fmt_duration(gk_total_b)}")
        + tile("Overall (17 runs)", overall_ratio, f"{label_b}: {fmt_duration(total_a)} → {fmt_duration(total_b)}")
        + "</div>"
    )

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Gradient Ledger &mdash; Hardware Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
</head>
<body>
<div class="sheet">
  <div class="masthead">
    <span class="eyebrow">Experiment Log &middot; Week 01 &middot; CST 627 &middot; Hardware Comparison</span>
    <h1>The Gradient Ledger: {e(label_a)} vs. {e(label_b)}</h1>
    <p class="dek">Same 17-run suite, same code, two machines. Does a GPU actually help, and does the science (double descent, grokking) agree regardless of hardware?</p>
  </div>

  {headline}

  <h2 class="section">Hardware</h2>
  {render_hardware_table(man_a["hardware"], man_b["hardware"], label_a, label_b)}

  <h2 class="section">Timing: quick checks</h2>
  <p class="lede">Both run in a couple of seconds either way &mdash; this is Python/import startup, not a meaningful hardware comparison.</p>
  {quick_table}

  <h2 class="section">Timing: double descent</h2>
  <p class="lede">The double-descent model is tiny (max hidden width 575, ~500 training examples) &mdash; small enough that {e(label_b)}'s per-step kernel-launch overhead can outweigh its parallelism advantage.</p>
  {dd_table}

  <h2 class="section">Timing: grokking</h2>
  <p class="lede">The grokking model is bigger (9409 pairs, embed dim 32, hidden dim 256) &mdash; here {e(label_b)}'s advantage shows clearly and consistently across all 13 runs.</p>
  {gk_table}

  <figure><img src="results/hardware_comparison.png" alt="Per-run duration comparison"><figcaption>hardware_comparison.png &mdash; every run, both machines, log-scale duration</figcaption></figure>

  <hr class="rule">

  <h2 class="section">Does the science agree across hardware?</h2>
  <p class="lede">Speed aside, the reported phenomena should not depend on which machine ran them. Absolute numbers differ (different RNG streams between CPU and CUDA even at the same seed &mdash; see SEED_AUDIT.md), but every qualitative conclusion below matches.</p>
  {render_dd_consistency(man_a, man_b, label_a, label_b)}
  {render_gk_consistency(man_a, man_b, label_a, label_b)}

  <footer>
    <div>{e(label_a)} grand total: {fmt_duration(grand_total_a)} &middot; {e(label_b)} grand total: {fmt_duration(grand_total_b)} (all 19 runs)</div>
    <div>Week-01-The-Gradient-Ledger &middot; hardware comparison</div>
  </footer>
</div>
</body>
</html>
"""
    def summary_ratio(ratio: float) -> str:
        return f"{ratio:.2f}x faster" if ratio > 1 else f"{1 / ratio:.2f}x slower"

    args.out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Double descent: {label_a} {fmt_duration(dd_total_a)} vs {label_b} {fmt_duration(dd_total_b)} "
          f"({label_b} {summary_ratio(dd_ratio)})")
    print(f"Grokking: {label_a} {fmt_duration(gk_total_a)} vs {label_b} {fmt_duration(gk_total_b)} "
          f"({label_b} {summary_ratio(gk_ratio)})")


if __name__ == "__main__":
    main()
