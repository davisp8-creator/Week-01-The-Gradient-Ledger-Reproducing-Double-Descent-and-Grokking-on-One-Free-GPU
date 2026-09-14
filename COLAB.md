# Running the suite on a Colab T4 GPU

This reruns the exact suite behind `VMreport.html` on a Colab T4 and
produces a separately-tagged report (`T4report.html`) plus tagged result
files (`results/*_t4gpu.*`) — nothing overwrites the CPU-VM results already
in the repo.

## 1. Runtime

Colab menu: **Runtime > Change runtime type > T4 GPU**.

## 2. Clone the repo and check the environment

```python
!git clone https://github.com/davisp8-creator/Week-01-The-Gradient-Ledger-Reproducing-Double-Descent-and-Grokking-on-One-Free-GPU.git
%cd Week-01-The-Gradient-Ledger-Reproducing-Double-Descent-and-Grokking-on-One-Free-GPU
!python scripts/test_environment.py
```

Don't `pip install -r requirements.txt` here — it pins a **CPU** build of
torch, and installing it would silently replace Colab's CUDA-enabled torch
with one that can't see the GPU. `test_environment.py` only installs a
package if it's actually missing, so it's safe to run as-is: it'll confirm
Colab's preinstalled torch already sees the T4.

Confirm the last check reports something like
`torch 2.14.0+cu121 - CUDA available - Tesla T4` before continuing — if it
says "CPU only" instead, the runtime wasn't switched to GPU in step 1.

## 3. Run the full suite

```python
!python scripts/run_full_suite.py --tag t4gpu
```

This runs all 17 training runs (4 double-descent variants, 13 grokking
runs) plus the two quick checks, exactly as documented in `VMreport.html`,
writing every output as `results/<name>_t4gpu.*` and a timing manifest to
`results/manifest_t4gpu.json`. Expect it to take well under the CPU
baseline's ~51 minutes for the grokking runs (much bigger matmuls, more to
gain from the GPU) — the double-descent runs use a tiny model on a tiny
dataset, so don't be surprised if the GPU isn't dramatically faster there,
or is even a bit slower once you count per-step kernel-launch overhead.

Add `--skip-grokking-seeds` to drop the 4 extra seed-variance runs (axis 3)
if you want a faster first pass.

## 4. Generate the report

```python
!python scripts/generate_report.py --tag t4gpu --out T4report.html
```

## 5. Push the results back to the repo

Results only exist in the Colab VM until you push them. You'll need a
[GitHub personal access token](https://github.com/settings/tokens) with
**Contents: Read and write** on this repo (fine-grained) or `repo` scope
(classic) — treat it as a secret, don't print or commit it, and revoke it
afterward if it was only for this one push.

```python
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add results/*_t4gpu* T4report.html
!git commit -m "Add T4 GPU experiment results"
!git remote set-url origin https://<YOUR_TOKEN>@github.com/davisp8-creator/Week-01-The-Gradient-Ledger-Reproducing-Double-Descent-and-Grokking-on-One-Free-GPU.git
!git push origin main
```

## What you'll get

- `results/manifest_t4gpu.json` — every run's parameters, duration, and
  derived result (interpolation threshold / grokking transition step),
  plus a hardware snapshot (GPU name/VRAM, CPU, RAM, OS, package versions).
- `results/double_descent_t4gpu*.csv` / `.png` and
  `results/grokking_t4gpu*.csv` / `.png` — the tagged result files.
- `T4report.html` — the same ledger-style report as `VMreport.html`, built
  from the manifest, with the T4's hardware called out in its own device
  manifest strip.

Open `VMreport.html` and `T4report.html` side by side afterward to compare
per-run durations directly.
