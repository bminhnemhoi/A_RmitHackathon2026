# RMIT Hackathon 2026 — Challenge 1: Multilingual Prompt-Attack Classification

Team **Fprinciples** · Kaggle competition [rmit-hackathon-2026](https://www.kaggle.com/competitions/rmit-hackathon-2026)
· Challenge 2 lives in [A_RmitHackathon2026_c2](https://github.com/bminhnemhoi/A_RmitHackathon2026_c2).

Classify prompts written in Portuguese, French, Italian and English as `BENIGN`, `JAILBREAK` or
`PROMPT_INJECTION`. Metric: macro-F1. Only approved backbones are allowed (see [docs/CONTEXT.md](docs/CONTEXT.md)).

## Status
| Item | Status |
|---|---|
| Valid first submission | ☐ |
| Best validation macro-F1 | _filled from experiments/log.csv_ |
| Final model | _filled after the deadline_ |

## Approach (plan)
1. Fine-tune approved multilingual encoders (mmBERT, XLM-R, wolf-defender) with class-weighted cross-entropy,
   head+tail truncation and a stratified (label × language) validation split.
2. Fit per-class probability multipliers on validation to maximise macro-F1 (rare `PROMPT_INJECTION`).
3. Blend softmax probabilities of diverse backbones; weights fitted on validation only.

Training notebooks are shared inside the team on Kaggle during the event and published here after the deadline
(Kaggle rule 6: no private sharing outside the team; public sharing belongs on Kaggle).

## Repository layout
```
docs/          CONTEXT (official brief) · PLAN · RESEARCH · SUBMISSIONS
src/c1/        metrics, class-multiplier search, ensembling
scripts/       validate_submission.py, py_to_ipynb.py, check_environment.py
experiments/   log.csv — one row per run, real numbers only
tests/         pytest
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q
# put Kaggle files in data/ (never committed)
python scripts/validate_submission.py --sample data/sample_submission.csv --submission outputs/submission.csv
```

## Team workflow
See [AGENTS.md](AGENTS.md) (rules for humans and AI agents) and [docs/PLAN.md](docs/PLAN.md).
