# C1 Context — Multilingual Prompt-Attack Classification

> Single source of truth for Challenge 1. Distilled from the official Kaggle pages
> (overview, data, rules, models) on 2026-10-22. If the Kaggle page changes, the Kaggle page wins —
> update this file and note the time.

## 1. Event facts

| Item | Value |
|---|---|
| Event | RMIT Hackathon 2026 — *Security × Generative AI × Low-Resource Languages* |
| Competition | https://www.kaggle.com/competitions/rmit-hackathon-2026 |
| Sample notebook | https://www.kaggle.com/code/aisuko/rmit-sgs-demo-c1 |
| Deadline | **22/10/2026 18:00 (Vietnam, UTC+7)** — same as Challenge 2 |
| Team | Fprinciples — 4 members, same venue |
| Kaggle team + notebook name | `RMIT_<CAMPUS>_Fprinciples_C1`, `<CAMPUS>` ∈ {SGS, HNI, MEL, MYS} (**confirm campus**) |
| Naming rule | letters only for `<university>` and `<campus>`; letters/numbers/`_`/`-` for `<team>` |
| Scoring note | Full Challenge 1 points on the **first valid submission**; after that, climb the leaderboard |
| Rules page hint | "Submit notebooks for higher score" — submit from the named Kaggle notebook |

## 2. Task

Classify each prompt into exactly one of:

| label | meaning |
|---|---|
| `BENIGN` | a normal, safe user request |
| `JAILBREAK` | an attempt to persuade the model to abandon its safety rules |
| `PROMPT_INJECTION` | an attempt to hijack the model's instructions |

Languages: Portuguese (`pt`), French (`fr`), Italian (`it`), English (`en`) — mostly non-English.

## 3. Data

| file | rows | columns |
|---|---:|---|
| `train.csv` | 14,000 | `id, text, language, label` |
| `test.csv` | 6,000 | `id, text, language` |
| `sample_submission.csv` | 6,000 | `id, label` |

- `id` like `t000006`. `text` length 10–18,550 chars, mean 294.
- Class balance (train): BENIGN 62%, JAILBREAK 31%, PROMPT_INJECTION 7%.
- Data licence: **CC BY-NC-SA 4.0** → never commit the data to GitHub.
- Kaggle input path is typically `/kaggle/input/rmit-hackathon-2026/` (verify in the notebook with `!ls /kaggle/input`).

## 4. Metric

**Macro-averaged F1** over the 3 classes (equal weight). PROMPT_INJECTION (7%) counts as much as
BENIGN (62%). All-BENIGN scores ≈ 0.26. Perfect = 1.0.

## 5. Allowed backbones (mandatory — no other pretrained backbone)

All provided as Kaggle Models by `aisuko`. **State the chosen model at the top of the notebook.**

| Model | Type | Kaggle | Suggested for |
|---|---|---|---|
| distilbert/distilbert-base-multilingual-cased | encoder | `aisuko/distilbert-base-multilingual-cased` | fastest baseline |
| jhu-clsp/mmBERT-small | encoder | `aisuko/mmbert-small` | fast, multilingual |
| jhu-clsp/mmBERT-base | encoder | `aisuko/mmbert-base` | strong multilingual |
| FacebookAI/xlm-roberta-base | encoder | `aisuko/xlm-roberta-base` | strong multilingual |
| FacebookAI/xlm-roberta-large | encoder | `aisuko/xlm-roberta-large` | best encoder, more VRAM |
| google/rembert | encoder | `aisuko/rembert` | multilingual (batch size 8) |
| protectai/deberta-v3-base-prompt-injection-v2 | encoder (safety-tuned) | `aisuko/deberta-v3-base-prompt-injection-v2` | known prompt injection |
| patronus-studio/wolf-defender-prompt-injection | encoder (safety-tuned) | `aisuko/wolf-defender-prompt-injection` | known prompt injection |
| ibm-granite/granite-3.3-2b-instruct | LLM | `aisuko/granite-3.3-2b-instruct` | LoRA fine-tuning |
| mistralai/Mistral-7B-Instruct-v0.3 | LLM | `aisuko/mistral-7b-instruct-v0.3` | LoRA fine-tuning |

URLs follow `https://www.kaggle.com/models/aisuko/<slug>/Transformers/default`.

## 6. Submission

- CSV `id,label`, **exactly 6,000 rows + header**, one row per test `id`.
- `label` ∈ {`BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`} exactly (case-sensitive).
- **Only 6 submissions in total** → every slot is logged in [SUBMISSIONS.md](SUBMISSIONS.md).
- Final ranking = **private leaderboard**; select final submissions by validation, not public LB.
- Always run `python scripts/validate_submission.py` before submitting.

## 7. Rules that matter (Kaggle base rules)

- One Kaggle account per person; join exactly one team; confirm team invite.
- **No private sharing of code/data outside the team.** Public sharing must be on Kaggle (forum/notebook).
  → This GitHub repo is public: before 18:00 it holds only scaffolding, docs and generic utilities.
  Training code is shared inside the team through Kaggle notebooks (collaborators). Push model code after 18:00.
- No hand-labelling of test data. Open-source code only under OSI licences.

## 8. Officially suggested approach

- Encoder: `AutoModelForSequenceClassification(num_labels=3)`.
- LLM: LoRA / QLoRA or a constrained classification head.
- Handle imbalance: class weights, focal loss, or resampling.
- Use the `language` column — per-language performance is uneven.
- Focus on robust generalisation, not public-LB overfitting.
