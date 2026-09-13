# Seed audit

Every place in this repo where a result would change if the random seed
changed, and whether that seed is recorded in the output. (Assignment
prompt: "Review my repo and tell me every place where a result would
change if the random seed changed, and whether I have recorded that seed.")

## scripts/double_descent_sweep.py -- fully recorded

`seed` is a first-class swept parameter and is written into every CSV row.
One seed value controls four things at once for a given run: the synthetic
dataset (`make_classification(random_state=seed)`), the label-noise flip
mask (`np.random.default_rng(seed)`), the train/test split
(`train_test_split(random_state=seed)`), and the model's weight
initialization (`torch.manual_seed(seed)`). Because all four are coupled to
one integer, this setup cannot separate "test error varies because of which
data was drawn" from "test error varies because of which weights were
drawn" -- both move together. That's an acceptable simplification for a
width sweep (we're averaging over both sources of variance at once), but it
means the seed-range band in the plots is not purely "model-init variance."

## scripts/grokking_train.py -- was NOT recorded, now fixed

`--seed` (default 0) feeds both `np.random.default_rng(seed)` (the
train/test split over all 9409 (a,b) pairs) and `torch.manual_seed(seed)`
(model init). Until this pass, the CSV output never recorded which seed
produced a given run -- a batch of grokking runs at different seeds would
have been untraceable back to their seeds from the CSV alone. Fixed by
adding a `seed` column (alongside the `weight_decay` and `train_frac`
columns added for the same reason) so any run's CSV is self-describing.

## scripts/gradient_check.py -- hardcoded, not parameterized or logged

`torch.manual_seed(0)` at module scope, plus a second `torch.Generator().manual_seed(0)`
used specifically for the CPU-side weight init (see the comment in that
file on why CUDA generators can't share a seed with CPU ones). Neither is
exposed as a CLI argument or printed in the output. This is low-stakes: the
script's purpose is a pass/fail correctness check (manual backprop vs.
autograd vs. finite differences), and that verdict is not expected to
depend on the seed -- only the specific loss/gradient values printed would
change. Worth noting anyway since "hardcoded 0" is itself a silent
assumption a reader can't see without opening the file.

## scripts/test_environment.py -- hardcoded, cosmetic only

`random_state=0` in the sklearn smoke test (`make_classification` /
`train_test_split`) only affects the diagnostic accuracy number printed
during environment verification. Not a scientific result; not worth
plumbing through a CLI flag.

## Cross-cutting: device changes results even at a fixed seed

Not a seed issue per se, but adjacent to it and empirically observed in
this project: running `scripts/gradient_check.py` with `torch.manual_seed(0)`
on this machine's CPU vs. on a Colab T4 GPU produced different loss values
(1.042441 vs. 1.225929), because `torch.randn(..., device="cuda")` draws
from a different RNG stream than the CPU path even when both are seeded
identically with `torch.manual_seed`. None of the scripts here call
`torch.use_deterministic_algorithms(True)`; low risk for these simple
MLPs, but worth knowing that "same seed" does not mean "same numbers"
across CPU and GPU.

## Recommendation

For any future sweep script, add the seed (and any other run-defining
hyperparameter that varies across files, like `grokking_train.py`'s
`weight_decay`/`train_frac`) as its own CSV column from the start, rather
than relying on filenames to encode it.
