"""Build a ledger-style HTML report from a scripts/run_full_suite.py manifest.

Renders the same report.html design (device manifest, numbered entries,
per-run parameter/timing tables, embedded result PNGs) but populated from
whatever hardware actually ran the suite -- so a CPU run and a Colab T4
run each get their own correctly-labeled report instead of one being
hand-edited to describe the other.

Usage:
    python scripts/run_full_suite.py --tag t4gpu   # produces the manifest first
    python scripts/generate_report.py --tag t4gpu  # -> report_t4gpu.html
"""

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
  :root {
    color-scheme: light;
    --paper:        #faf8f1;
    --band:         #eef2e6;
    --band-strong:  #e3ead9;
    --ink:          #202b23;
    --ink-soft:     #55605a;
    --ink-faint:    #85897f;
    --rule:         #c7cec0;
    --rule-strong:  #9aa693;
    --accent:       #9a3324;
    --accent-soft:  #f2e2dc;
    --brass:        #8a6a2f;
    --brass-soft:   #efe6d1;
    --link:         #2c5a4a;
    --font-display: "IBM Plex Serif", Georgia, "Times New Roman", serif;
    --font-body:    "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --font-mono:    "IBM Plex Mono", ui-monospace, "Cascadia Code", Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --paper: #14181a; --band: #1b241f; --band-strong: #212d25;
      --ink: #e7ece4; --ink-soft: #b7bfb5; --ink-faint: #7c877e;
      --rule: #333e37; --rule-strong: #48564c;
      --accent: #e2795f; --accent-soft: #3a241d;
      --brass: #cca757; --brass-soft: #362c1a; --link: #7fbfa2;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --paper: #14181a; --band: #1b241f; --band-strong: #212d25;
    --ink: #e7ece4; --ink-soft: #b7bfb5; --ink-faint: #7c877e;
    --rule: #333e37; --rule-strong: #48564c;
    --accent: #e2795f; --accent-soft: #3a241d;
    --brass: #cca757; --brass-soft: #362c1a; --link: #7fbfa2;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--paper); color: var(--ink); font-family: var(--font-body);
    font-size: 15.5px; line-height: 1.6; margin: 0;
    padding-inline: 20px; padding-block: 40px 80px;
  }
  .sheet { max-width: 860px; margin-inline: auto; }
  h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; margin: 0; }
  .eyebrow {
    font-family: var(--font-mono); font-size: 11.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); font-weight: 600;
  }
  .masthead { border-bottom: 3px double var(--rule-strong); padding-bottom: 22px; margin-bottom: 22px; }
  .masthead h1 { font-size: clamp(2rem, 5vw, 2.75rem); font-weight: 700; margin-top: 6px; }
  .masthead p.dek { max-width: 62ch; color: var(--ink-soft); margin-top: 10px; font-size: 1.02rem; }
  .manifest {
    background: var(--band); border: 1px solid var(--rule); border-radius: 3px;
    padding: 16px 20px; margin-block: 26px 34px;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px 24px;
  }
  .manifest .field .k {
    font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-faint);
  }
  .manifest .field .v {
    font-family: var(--font-mono); font-variant-numeric: tabular-nums;
    font-size: 13.5px; color: var(--ink); margin-top: 2px;
  }
  .manifest .field .v.note { font-family: var(--font-body); color: var(--ink-soft); font-size: 13px; }
  .entry { position: relative; padding-left: 54px; margin-block: 40px; }
  .entry::before {
    content: attr(data-n); position: absolute; left: 0; top: 2px;
    font-family: var(--font-mono); font-weight: 600; font-size: 13px; color: var(--accent);
    border: 1px solid var(--accent); border-radius: 2px; width: 34px; height: 26px;
    display: flex; align-items: center; justify-content: center;
  }
  .entry h2 { font-size: 1.5rem; font-weight: 600; }
  .entry .file { font-family: var(--font-mono); font-size: 12.5px; color: var(--link); }
  .entry .lede { color: var(--ink-soft); max-width: 68ch; margin-top: 8px; }
  .stamp {
    float: right; border: 1px solid var(--brass); color: var(--brass); background: var(--brass-soft);
    font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.04em;
    padding: 5px 10px; border-radius: 2px; text-align: center; margin-left: 14px; margin-bottom: 8px;
  }
  .stamp .n { display: block; font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }
  table.runs { width: 100%; border-collapse: collapse; margin-block: 18px; font-size: 13.5px; }
  table.runs caption {
    text-align: left; font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--ink-faint); margin-bottom: 6px;
  }
  table.runs th {
    text-align: left; font-family: var(--font-mono); font-weight: 600; font-size: 11.5px;
    letter-spacing: 0.03em; text-transform: uppercase; color: var(--ink-soft);
    border-bottom: 1px solid var(--rule-strong); padding: 7px 10px;
  }
  table.runs td {
    padding: 7px 10px; border-bottom: 1px solid var(--rule);
    font-variant-numeric: tabular-nums; vertical-align: top;
  }
  table.runs tbody tr:nth-child(odd) { background: var(--band); }
  table.runs td.num, table.runs td.dur { font-family: var(--font-mono); }
  table.runs td.result { font-family: var(--font-body); color: var(--ink-soft); max-width: 30ch; }
  table.runs tr.baseline-row td { font-weight: 600; }
  table.runs td.never { color: var(--accent); font-family: var(--font-mono); }
  .frames { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; margin-block: 20px; }
  figure { margin: 0; border: 1px solid var(--rule); background: var(--paper); border-radius: 3px; overflow: hidden; }
  figure img { display: block; width: 100%; height: auto; }
  figcaption {
    font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft);
    padding: 9px 12px; border-top: 1px solid var(--rule); background: var(--band);
  }
  pre.console {
    font-family: var(--font-mono); font-size: 12px; color: var(--ink-soft);
    background: var(--band); border: 1px solid var(--rule); border-radius: 3px;
    padding: 12px 14px; overflow-x: auto; white-space: pre-wrap;
  }
  .rule { border: none; border-top: 1px solid var(--rule); margin-block: 40px; }
  footer {
    margin-top: 56px; padding-top: 22px; border-top: 3px double var(--rule-strong);
    font-family: var(--font-mono); font-size: 12.5px; color: var(--ink-soft);
    display: flex; flex-wrap: wrap; gap: 8px 28px; justify-content: space-between;
  }
  footer .total .n { color: var(--accent); font-weight: 600; font-size: 14px; }
  @media (max-width: 480px) {
    .entry { padding-left: 0; }
    .entry::before { position: static; margin-bottom: 8px; }
    .stamp { float: none; display: inline-block; }
  }
"""


def fmt_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m&nbsp;{s:.1f}s" if m else f"{s:.1f}s"


def e(text) -> str:
    return html.escape("" if text is None else str(text))


def find_run(runs: list[dict], label: str) -> dict | None:
    return next((r for r in runs if r["label"] == label), None)


def by_params(runs: list[dict], **want) -> list[dict]:
    return [r for r in runs if all(r["params"].get(k) == v for k, v in want.items())]


def render_manifest(hw: dict) -> str:
    gpu = f"{hw.get('gpu_name')} ({hw.get('gpu_vram_gib')} GiB VRAM)" if hw.get("cuda_available") else "none passed through — CPU only"
    ram = f"{hw.get('ram_gib')} GiB"
    fields = [
        ("CPU", hw.get("cpu_model", "unknown"), False),
        ("Cores / threads", str(hw.get("cpu_logical_cores", "?")), False),
        ("Memory", ram, False),
        ("GPU", gpu, True),
        ("OS", f"{hw.get('os', '?')} (build {hw.get('os_version', '?')})", True),
        ("Python", hw.get("python_version", "?"), False),
        ("torch", hw.get("torch_version", "?"), False),
        ("numpy / pandas", f"{hw.get('numpy_version', '?')} / {hw.get('pandas_version', '?')}", False),
        ("matplotlib / scikit-learn", f"{hw.get('matplotlib_version', '?')} / {hw.get('sklearn_version', '?')}", False),
    ]
    rows = "".join(
        f'<div class="field"><div class="k">{e(k)}</div><div class="v{" note" if is_note else ""}">{e(v)}</div></div>'
        for k, v, is_note in fields
    )
    return f'<div class="manifest">{rows}</div>'


def render_dd_table(runs: list[dict]) -> str:
    variants = [
        ("Baseline", "baseline-row", "double_descent_baseline", "Adam (lr 1e-3)"),
        ("Regularized", "", "double_descent_wd0.01", "Adam (lr 1e-3)"),
        ("Regularized", "", "double_descent_wd0.1", "Adam (lr 1e-3)"),
        ("Optimizer ablation", "", "double_descent_sgd", "SGD (lr 0.1, mom 0.9)"),
    ]
    rows = []
    for name, cls, label, opt_desc in variants:
        r = find_run(runs, label)
        if r is None:
            continue
        res = r.get("result", {})
        wd = r["params"].get("weight_decay")
        thr = res.get("interpolation_threshold")
        thr_str = f"interpolates at width {thr}" if thr is not None else "never interpolates (≤1% train err)"
        result_cell = f"peak test error {res.get('peak_test_error')} @ width {res.get('peak_width')}; {thr_str}"
        rows.append(
            f'<tr class="{cls}"><td>{e(name)}</td><td>{e(opt_desc)}</td><td class="num">{e(wd)}</td>'
            f'<td class="dur">{fmt_duration(r["duration_s"])}</td><td class="result">{e(result_cell)}</td></tr>'
        )
    return (
        '<table class="runs"><caption>Four sweep variants, same data/widths/seeds/epochs</caption>'
        "<thead><tr><th>Variant</th><th>Optimizer</th><th>Weight decay</th><th>Duration</th><th>Result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_gk_axis_table(caption: str, header: str, runs: list[dict], key: str, baseline_val) -> str:
    rows = []
    for r in sorted(runs, key=lambda r: r["params"][key]):
        val = r["params"][key]
        res = r.get("result", {})
        transition = res.get("transition_step")
        cls = "baseline-row" if val == baseline_val else ""
        if transition is not None:
            step_cell = f'<td class="num">{e(transition)}</td>'
            result = "grok"
        else:
            step_cell = '<td class="never">never</td>'
            acc = res.get("final_test_acc")
            result = f"never fully grokked (final test_acc {acc})" if acc is not None else "never grokked in budget"
        rows.append(
            f'<tr class="{cls}"><td class="num">{e(val)}</td><td class="dur">{fmt_duration(r["duration_s"])}</td>'
            f'{step_cell}<td class="result">{e(result)}</td></tr>'
        )
    return (
        f'<table class="runs"><caption>{e(caption)}</caption>'
        f"<thead><tr><th>{e(header)}</th><th>Duration</th><th>Transition step</th><th>Result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_frames(pairs: list[tuple[str, str]]) -> str:
    figs = "".join(
        f'<figure><img src="results/{e(src)}" alt="{e(caption)}"><figcaption>{e(src)} &mdash; {e(caption)}</figcaption></figure>'
        for src, caption in pairs
    )
    return f'<div class="frames">{figs}</div>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tag = args.tag
    manifest_path = args.manifest or ROOT / "results" / f"manifest_{tag}.json"
    out_path = args.out or ROOT / f"report_{tag}.html"
    manifest = json.loads(manifest_path.read_text())
    runs = manifest["runs"]
    hw = manifest["hardware"]

    env_run = find_run(runs, "environment_verification")
    grad_run = find_run(runs, "gradient_check")

    dd_runs = [r for r in runs if r["label"].startswith("double_descent")]
    gk_runs = [r for r in runs if r["label"].startswith("grokking")]
    gk_baseline = find_run(runs, "grokking_wd1.0_frac0.4_seed0")

    wd_axis = by_params(gk_runs, train_frac=0.4, seed=0)
    frac_axis = by_params(gk_runs, weight_decay=1.0, seed=0)
    seed_axis = by_params(gk_runs, weight_decay=1.0, train_frac=0.4)

    total_s = sum(r["duration_s"] for r in runs)
    gpu_desc = hw.get("gpu_name") if hw.get("cuda_available") else "CPU-only"

    seed_note = ""
    if len(seed_axis) > 1:
        steps = [r["result"]["transition_step"] for r in seed_axis if r.get("result", {}).get("transition_step") is not None]
        if steps:
            mean = sum(steps) / len(steps)
            std = (sum((s - mean) ** 2 for s in steps) / len(steps)) ** 0.5
            seed_note = (
                f'<p class="lede">Transition step across {len(steps)} seeds: mean {mean:.0f}, '
                f"std &asymp;{std:.0f}.</p>"
            )

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Gradient Ledger &mdash; {e(tag)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
</head>
<body>
<div class="sheet">
  <div class="masthead">
    <span class="eyebrow">Experiment Log &middot; Week 01 &middot; CST 627 &middot; tag: {e(tag)}</span>
    <h1>The Gradient Ledger</h1>
    <p class="dek">Auto-generated from scripts/run_full_suite.py + scripts/generate_report.py. Same suite as report.html, run on: <strong>{e(gpu_desc)}</strong>.</p>
  </div>
  {render_manifest(hw)}

  <div class="entry" data-n="00">
    <div class="stamp">DURATION<span class="n">{fmt_duration(env_run["duration_s"]) if env_run else "?"}</span></div>
    <h2>Environment verification</h2>
    <div class="file">scripts/test_environment.py</div>
    <pre class="console">{e(env_run.get("stdout", "").strip()) if env_run else ""}</pre>
  </div>

  <hr class="rule">

  <div class="entry" data-n="01">
    <div class="stamp">DURATION<span class="n">{fmt_duration(grad_run["duration_s"]) if grad_run else "?"}</span></div>
    <h2>Manual backprop gradient check</h2>
    <div class="file">scripts/gradient_check.py</div>
    <pre class="console">{e(grad_run.get("stdout", "").strip()) if grad_run else ""}</pre>
  </div>

  <hr class="rule">

  <div class="entry" data-n="02">
    <h2>Double-descent width sweep</h2>
    <div class="file">scripts/double_descent_sweep.py &middot; scripts/plot_double_descent.py</div>
    {render_dd_table(dd_runs)}
    {render_frames([
        (f"double_descent_{tag}.png", "baseline, mean ±1 SEM across 10 seeds"),
        (f"double_descent_{tag}_regularization.png", "weight decay comparison"),
        (f"double_descent_{tag}_optimizer.png", "Adam vs SGD"),
        (f"double_descent_{tag}_epochwise.png", "train/test error vs epoch by width"),
    ])}
  </div>

  <hr class="rule">

  <div class="entry" data-n="03">
    <h2>Grokking on modular addition (mod 97)</h2>
    <div class="file">scripts/grokking_train.py &middot; scripts/plot_grokking.py &middot; scripts/plot_grokking_sweep.py</div>
    {render_gk_axis_table("Axis 1 · weight decay (train_frac 0.4, seed 0)", "Weight decay", wd_axis, "weight_decay", 1.0)}
    {render_gk_axis_table("Axis 2 · train fraction (weight_decay 1.0, seed 0)", "Train fraction", frac_axis, "train_frac", 0.4)}
    {render_gk_axis_table("Axis 3 · random seed (train_frac 0.4, weight_decay 1.0)", "Seed", seed_axis, "seed", 0) if len(seed_axis) > 1 else ""}
    {seed_note}
    {render_frames([
        (f"grokking_{tag}.png", "baseline: accuracy, loss, weight norm"),
        (f"grokking_{tag}_weight_decay_sweep.png", "vs. weight decay"),
        (f"grokking_{tag}_train_frac_sweep.png", "vs. train fraction"),
    ] + ([(f"grokking_{tag}_seed_sweep.png", "across seeds")] if len(seed_axis) > 1 else []))}
  </div>

  <footer>
    <div>{len(runs)} training runs &middot; {e(gpu_desc)}</div>
    <div class="total">total wall-clock&nbsp;<span class="n">{fmt_duration(total_s)}</span></div>
    <div>Week-01-The-Gradient-Ledger &middot; tag {e(tag)}</div>
  </footer>
</div>
</body>
</html>
"""]

    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
