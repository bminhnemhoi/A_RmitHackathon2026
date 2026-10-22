@AGENTS.md

## Claude Code notes
- Shell on the author's machine is Windows (PowerShell + Git Bash). Local torch is CPU-only; GPU training happens on Kaggle.
- Local data (not committed): put `train.csv`, `test.csv`, `sample_submission.csv` in `data/`.
- Before claiming a submission is ready, run the validator and the tests, and paste the real output.
- When asked to "improve the model", propose one change with a reason, compare on the same split, and log it.
