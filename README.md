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

