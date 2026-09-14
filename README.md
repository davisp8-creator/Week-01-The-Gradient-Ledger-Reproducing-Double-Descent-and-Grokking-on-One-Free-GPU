# CST 627 100: Deep Learning and Neural Networks

**Professor:** Chris Meehleib
**Week 01:** The Gradient Ledger: Reproducing Double Descent and Grokking on One Free GPU

## Scenario and Objective

You are a Machine Learning Engineer I at Northwind Diagnostics, a 30-person medical-imaging startup in Rochester. Your team lead, Dr. Priya Raman, has escalated a training anomaly: a small chest-film triage classifier's validation loss climbed for 40 epochs and then fell below its earlier best, and she suspects the backward pass is broken.

Sprint capacity is one free-tier Colab GPU session. Using Claude Code or Claude Co-Work, build a small experiment harness that:

1. Verifies your hand-written two-layer MLP's backprop against central finite differences with reported tolerances.
2. Reproduces a double-descent curve and a grokking curve on data you generate yourself: a modular-arithmetic task you write in Python, or a label-noise variant of a public dataset such as scikit-learn digits or an open Kaggle tabular set.

Drive the agent in plain English (for example, "sweep hidden width from 4 to 512, record train/test error, log every seed to CSV"), then iterate:

- Add seed control.
- Add a weight-norm trace.
- Plot the interpolation threshold.

Never accept code you cannot explain. Ask the agent to walk you through anything unclear before you record. Your deliverable answers Dr. Raman's actual question: **bug or phenomenon?**

## Tool Setup

- Install Node.js 18+ and Claude Code locally, or sign into Claude Co-Work.
- Create a Google Colab notebook and set the runtime type to a free T4 GPU.

## Code Generation

- Direct Claude to build a two-layer MLP that computes the backward pass manually without autograd.
- Instruct the agent to write a central finite-difference checker to verify your manual gradients against PyTorch autograd.
- Generate a synthetic dataset, train models across a sweep of hidden widths or epochs, log every random seed, and export the results to a CSV file.
- Plot the CSV data to visually reproduce the expected rise and drop of the double descent or grokking phenomenon.

## Output and Documentation

- Document all failed attempts, fixes, and prompt iterations directly in the notebook so a teammate can trace your reasoning.
- Export your finalized notebook, CSV logs, and PNG graphs to a GitHub repository or a shared folder link.
- Consult `Week_1_Assignment_Paddy_Davis.docx` to match any additional structural or naming conventions required for the final file upload.

## Video Demonstration

- Record an 8 to 12-minute video with your face clearly visible via a picture-in-picture overlay for the entire duration.
- Speak naturally in your own words; use an outline for structure but do not read verbatim from a script.
## Quick Start (5 Minutes)

### 3. Install Claude Code

Type this command and press Enter:

```bash
npm install -g @anthropic-ai/claude-code
```

If you see `npm: command not found`, install Node.js first from nodejs.org (download the LTS version), then try again.

### 4. Start Your First Session

Create a folder for your project, navigate to it, and launch Claude Code:

```bash
mkdir my-project
cd my-project
claude
```

Claude Code will open an interactive session. Describe what you want to build in plain English.

## Example Prompts

Try these prompts:

1. > Write a two-layer MLP in pure PyTorch tensors where I compute the backward pass manually (no autograd) for a cross-entropy loss. Then write a central finite-difference checker that compares my manual gradients to `torch.autograd` on a single minibatch and prints relative error per parameter tensor. Explain your step-size choice before you write it.
2. > My finite-difference check fails only on the ReLU layer with relative error around 1e-2 while other layers are 1e-7. Give me three candidate explanations ranked by likelihood and a one-line test that separates them.
3. > Build a script that generates a synthetic classification dataset with a controllable label-noise fraction, then sweeps hidden width over [4, 8, 16, 32, 64, 128, 256, 512], trains each width with 3 seeds, and writes width, seed, epoch, train_error, test_error to a CSV. Do not plot yet; show me the CSV schema first.
4. > Now add a plotting cell that draws test error versus hidden width with a shaded band across seeds, and mark the width where training error first reaches zero. Annotate that point on the axis.
5. > Write a modular-addition dataset generator for mod 97 with a 40% train split, and a small model that I can train for 20,000 steps with AdamW and weight decay 1.0. Log train and test accuracy plus total weight L2 norm every 100 steps to CSV.
6. > Walk me through lines 40 through 62 of the training loop you just wrote. I need to explain each one on camera; tell me what would break if I removed the seed-setting call.
7. > Review my repo and tell me every place where a result would change if the random seed changed, and whether I have recorded that seed.

Start simple, then add complexity. If something isn't right, tell Claude Code what to change.

## Troubleshooting

- **"command not found":** Make sure Node.js is installed (`node --version` should show a number). Restart your terminal after installing.
- **"permission denied":** On Mac/Linux, try `sudo npm install -g @anthropic-ai/claude-code`.
- **Claude Code shows errors:** Describe the error to Claude Code itself: "I'm getting this error: [paste error]. How do I fix it?" This is part of the learning process.
- **Need help?** Ask your instructor or post in the course discussion board. Include a screenshot of your terminal.

---

## Project Documentation

### Abstract

This repository implements a self-contained empirical harness for investigating two phenomena in overparameterized model training that appear pathological when viewed only through a classical bias–variance lens: **double descent**, in which test risk as a function of model capacity rises, peaks near the interpolation threshold, and then descends a second time in the overparameterized regime; and **grokking**, in which a network memorizes its training set almost immediately while generalization remains at chance level for an extended plateau before transitioning sharply to near-perfect test accuracy. Both effects are reproduced from first principles on synthetic data generated in-repo (a label-noise binary classification task and a modular-arithmetic task, respectively), rather than borrowed from a benchmark dataset, so that every controllable variable — sample size, label-noise rate, weight decay, optimizer, train/test split fraction, and random seed — is known and logged. A hand-derived backward pass for a two-layer MLP is additionally verified against both `torch.autograd` and central finite differences, establishing that the training dynamics observed downstream are not artifacts of an incorrect gradient computation. The full suite is re-executed on two distinct hardware targets (a CPU-only Windows/Hyper-V VM and a Colab Tesla T4 GPU) under an identical, hardware-tagged pipeline, producing directly comparable timing and correctness data.

### Repository Layout

| Path | Contents |
|---|---|
| `scripts/gradient_check.py` | Manual backprop correctness check: analytic gradients vs. `autograd` vs. float64 central finite differences, at `h ≈ 6×10⁻⁶` (the truncation/cancellation-error-balancing step size for float64). |
| `scripts/double_descent_sweep.py` | Generates a noisy synthetic binary classification dataset (`sklearn.make_classification` + explicit label flips) and trains a two-layer MLP across a hidden-width sweep centered on the analytic interpolation threshold, logging per-epoch train/test error to CSV. |
| `scripts/grokking_train.py` | Generates the full `(a, b) → (a+b) mod 97` addition table, trains an embedding + MLP classifier with AdamW, and logs train/test accuracy, loss, and weight L2 norm every 100 steps. |
| `scripts/plot_double_descent.py`, `plot_double_descent_epochwise.py` | Render test-error-vs-width curves (single or overlaid multi-series) and epoch-resolved views of the same runs; the former also locates the empirical interpolation threshold. |
| `scripts/plot_grokking.py`, `plot_grokking_sweep.py` | Render accuracy/weight-norm-vs-step curves for a single run or an overlay across a swept hyperparameter (weight decay, train fraction, seed), and locate each run's grokking transition step. |
| `scripts/hardware_info.py` | Cross-platform (Windows/Linux, CPU/CUDA) hardware and package-version snapshot, embedded in every manifest for reproducibility and cross-device comparison. |
| `scripts/run_full_suite.py` | Orchestrates the entire experiment battery (2 correctness checks, 4 double-descent variants, 13 grokking runs) under a single `--tag`, timing every subprocess and writing a consolidated `results/manifest_<tag>.json`. |
| `scripts/generate_report.py`, `generate_comparison_report.py` | Build the static HTML ledger reports from a manifest (single-device) or from two manifests side by side (cross-device comparison). |
| `scripts/test_environment.py` | Environment smoke test; verifies and, if needed, installs the packages pinned in `requirements.txt`. |
| `results/` | All CSV logs, PNG figures, and JSON manifests, each suffixed by hardware tag (`_t4gpu`) or left untagged for the CPU baseline. |
| `VMreport.html` | Ledger-style report for the CPU baseline suite. |
| `T4report.html` | Same report format, for the Colab T4 GPU rerun. |
| `ComparisonReport.html` | Side-by-side CPU-vs-T4 timing and results-consistency report. |
| `SEED_AUDIT.md` | Traces every source of stochasticity in the codebase to whether it is parameterized and recorded in output, per experiment. |
| `COLAB.md` | Step-by-step procedure for re-running the suite on a Colab T4 runtime and pushing tagged results back to this repository. |

### Experiment 1 — Gradient Correctness (`gradient_check.py`)

Before any downstream phenomenon can be attributed to the training dynamics themselves, the optimizer's gradient signal must be shown correct. A two-layer MLP (`Linear → ReLU → Linear → softmax cross-entropy`) is implemented with an explicit, hand-derived backward pass — no autograd graph is constructed for the primary computation. This manual gradient is cross-validated against two independent references: `torch.autograd` applied to an identical forward pass, and central finite differences of the loss surface computed in float64. All three estimators agree to a relative error below the `1×10⁻⁴` tolerance across every parameter tensor, which rules out an incorrect backward pass as the explanation for the anomaly described in the assignment scenario. A secondary, empirically-verified observation recorded in the script's docstring and in `SEED_AUDIT.md`: identically seeded CPU and CUDA runs of `torch.randn` draw from different RNG streams, so cross-device numerical agreement at the level of specific loss values should not be expected even when `torch.manual_seed` matches.

### Experiment 2 — Double Descent (`double_descent_sweep.py`, `plot_double_descent*.py`)

A synthetic binary classification task (1,000 samples, 10 features, 20% label noise, 50/50 train/test split) is used to sweep a two-layer MLP's hidden width across values spanning roughly 0.05× to 15× the analytic interpolation threshold — the width at which the model's parameter count equals the number of training examples. Ten random seeds are run per width, coupling model initialization and data resampling under a single integer (documented as a known limitation in `SEED_AUDIT.md`, since it means the reported seed-variance band conflates two distinct sources of variance rather than isolating either one).

**Empirical findings (CPU baseline, `results/manifest_cpu.json`):**

- With Adam and no weight decay, mean test error peaks at **34.4%** near width 77, close to the analytic threshold (width 64), then descends as width increases further into the overparameterized regime — the canonical double-descent signature.
- Introducing weight decay (0.01, 0.1) suppresses the second-descent structure sufficiently that the automatic threshold detector no longer identifies a clean crossing, and shifts peak test error downward slightly (29.0–29.8%), consistent with the literature's account of explicit regularization damping double descent.
- Switching the optimizer from Adam to SGD with momentum shifts the empirical threshold earlier (width 29 vs. 64) and produces a sharper, higher peak (37.1% at width 19), indicating that the interpolation crossing and its severity are optimizer-dependent, not solely a function of the width/sample-count ratio.

### Experiment 3 — Grokking (`grokking_train.py`, `plot_grokking*.py`)

The modular addition task `(a + b) mod 97` over all 9,409 ordered pairs is trained with AdamW at weight decay 1.0 (matching Power et al., 2022) for 20,000 full-batch gradient steps, logging accuracy and weight L2 norm every 100 steps. Three axes are swept independently against this baseline (weight decay ∈ {0, 0.1, 0.5, 1.0, 2.0}, train fraction ∈ {0.2, 0.3, 0.4, 0.5, 0.6}, and seed ∈ {0, 1, 2, 3, 4}), for 13 total grokking runs.

**Empirical findings (CPU baseline):**

- **Weight decay is necessary for generalization in this regime.** At wd = 0 and wd = 0.1, test accuracy never rises above chance within the step budget; at wd = 0.5 it reaches only 68.0%; at wd = 1.0 and 2.0 it reaches 100%, with higher decay grokking sooner (step 3,600 vs. 8,300) — consistent with grokking being driven by a slow weight-norm-reduction process that explicit regularization accelerates.
- **More training data grokks sooner.** Transition step falls monotonically from "never" at train fraction 0.2, to 14,900 steps at 0.3, 8,300 at 0.4, 4,900 at 0.5, and 3,100 at 0.6.
- **Transition timing is seed-sensitive but the outcome is not.** Across seeds 0–4 (all other hyperparameters fixed), every run eventually reaches 100% test accuracy, but the transition step varies from 5,300 to 9,000 — a reminder that a single-seed run cannot distinguish "this configuration groks" from "this configuration groks by step N."

### Hardware Comparison (`ComparisonReport.html`, `COLAB.md`)

The identical suite (via `run_full_suite.py --tag <cpu|t4gpu>`) was executed on a CPU-only Hyper-V VM (13th Gen Intel Core i9-13900HX) and a Colab Tesla T4 GPU (14.6 GiB VRAM), with results and figures kept in separate, tagged files so neither run overwrites the other. Aggregate training wall-clock was **~51.1 minutes on CPU** vs. **~26.7 minutes on the T4** — but this net ~1.9× speedup is not uniform across experiment types, which is the more informative result:

- **Grokking runs were ~4.5× faster on the T4** (2,639.7s → 592.5s aggregate): the embedding + MLP classifier's full-batch matmuls over 9,409 examples are large enough to amortize GPU kernel-launch overhead and saturate the device.
- **Double-descent runs were ~2.4× *slower* on the T4** (425.0s → 1,003.9s aggregate): the model and dataset are small enough that per-step kernel-launch latency dominates, and the many short, sequential training runs (10 seeds × ~10 widths × 4 variants) cannot hide that latency behind useful compute.

The practical implication — verified, not merely asserted — is that GPU acceleration is a function of arithmetic intensity per kernel launch, not merely "is a GPU present," and that a workload's suitability for GPU offload should be assessed per-experiment rather than assumed uniformly across a suite. `ComparisonReport.html` also includes a results-consistency check confirming the two devices agree on the scientific conclusions (thresholds, peaks, transition steps) despite the CPU/CUDA RNG-stream divergence noted in Experiment 1.

### Reproducing These Results

```bash
python scripts/test_environment.py          # verify/install pinned dependencies
python scripts/run_full_suite.py --tag cpu  # full battery, ~51 min on CPU
python scripts/generate_report.py --tag cpu --out VMreport.html
```

To reproduce the GPU run and regenerate the comparison report, follow `COLAB.md` end to end; it also documents the GitHub token handling required to push Colab-generated results back to this repository. See `SEED_AUDIT.md` before adding any new sweep script — it specifies the convention (seed as an explicit, first-class CSV column) that keeps every run traceable to the randomness that produced it.

