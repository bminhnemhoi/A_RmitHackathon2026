# C1 Plan — team Fprinciples (deadline 18:00 VN, 22/10/2026)

## Roles
| Who | Responsibility |
|---|---|
| **ML-1** (lead, Kaggle submission gatekeeper) | Data audit, split, final notebook `RMIT_<CAMPUS>_Fprinciples_C1`, ensemble + class multipliers, decides every submission slot |
| **ML-2** | Runs a second backbone in parallel on **their own Kaggle account** (separate GPU quota), shares outputs as notebook outputs |
| Web-1 / Web-2 | Work on C2; help C1 only on request |

## Timeline
| Time | Goal | Done when |
|---|---|---|
| 10:40–11:10 | Kaggle team `RMIT_<CAMPUS>_Fprinciples_C1` created, all 4 joined + rules accepted. GPU notebook with data + model attached | `!ls /kaggle/input` shows data + model |
| 11:10–11:45 | **S1** = `train_kaggle.py` with `mmbert-small`, 2 epochs → submit | Kaggle shows a score → **full C1 points locked** |
| 11:45–14:30 | ML-1: `xlm-roberta-large` (lr 1e-5, len 256, bs 16, 3 ep). ML-2: `wolf-defender-prompt-injection` (lr 2e-5, len 256, 3 ep), then `mmbert-base` | `val_probs.npy` + `test_probs.npy` + row in `experiments/log.csv` |
| 14:30 | **S2** = best single model by val macro-F1 | logged in SUBMISSIONS.md |
| 14:30–15:45 | Ensemble notebook: blend probs (weights fitted on val) + per-class multipliers → **S3** | val gain over best single ≥ 0.005 |
| 15:45–16:45 | S4 only if val improves clearly; S5–S6 stay in reserve | — |
| 16:45–17:15 | **Select 2 final submissions** on Kaggle (by validation, not public LB) | screenshot saved |
| after 18:00 | Push model code + results to GitHub | — |

## How to run a model on Kaggle (S1 recipe)
1. Kaggle → New Notebook → **File → Import notebook** → upload `notebooks/RMIT_SGS_Fprinciples_C1.ipynb`
   (fix the campus in the title). Rename the notebook exactly `RMIT_<CAMPUS>_Fprinciples_C1`.
2. **Add Input**: competition `rmit-hackathon-2026` + model `aisuko/mmbert-small` (Transformers/default).
3. Settings → Accelerator **GPU T4 x2** (or P100). Internet can stay off.
4. Edit `CFG` (model_slug, lr, batch_size, run_name). The first markdown cell must name the model used.
5. **Save Version → Save & Run All (Commit)**. When it finishes, the output tab has `submission.csv`.
6. Submit from the notebook's Output tab ("Submit to competition"); then log the slot in SUBMISSIONS.md.

Same split everywhere: `val_size=0.15`, `seed=42`, stratified by `label_language` → val probs from different
runs are comparable and can be blended. Do not change the split seed between runs that will be ensembled.

## Ensembling across accounts
- Each experiment notebook writes `val_probs.npy`, `val_labels.npy`, `val_ids.npy`, `test_probs.npy`, `run.json`.
- Share the notebook with teammates (Share → add collaborator), then in the final notebook **Add Input → Notebook output**.
- Blend with `src/c1/ensemble.py::fit_and_predict` (copy the functions into the notebook cell).
  Check `val_ids` are identical across runs before blending.

## Submission protocol (6 slots, no exceptions)
1. Validator passes: `python scripts/validate_submission.py --sample sample_submission.csv --submission submission.csv`.
2. Val macro-F1 + per-class F1 recorded in `experiments/log.csv`.
3. ML-1 approves in team chat → submit → fill the row in [SUBMISSIONS.md](SUBMISSIONS.md) with the public LB score.

## Risks
| Risk | Mitigation |
|---|---|
| Model collapses to all-BENIGN (seen with large models / high lr) | lr 1e-5 for xlmr-large, warmup 10 %, class-weighted CE, check pred distribution before submitting |
| T4 OOM | batch 8 + `grad_accum` 2, or max_len 192 |
| mmBERT / wolf-defender loading issues | script forces `sdpa` attention and pins RoPE theta on transformers 4.x — read the printed config line |
| GPU quota | each ML member uses their own account; keep runs ≤ 45 min |
| Public LB overfitting | choose finals by validation; public LB is only a sanity check |
