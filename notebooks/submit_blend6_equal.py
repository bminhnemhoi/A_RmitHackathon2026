# %% [markdown]
# # RMIT_SGS_Fprinciples_C1 — ensemble submission (blend6_equal)
# **Models used (all from the approved list):** `FacebookAI/xlm-roberta-large`, `jhu-clsp/mmBERT-base`, `jhu-clsp/mmBERT-small`, `patronus-studio/wolf-defender-prompt-injection`
#
# Each backbone was fine-tuned on `train.csv` with the same stratified 5-fold split (label x language, seed 42);
# test probabilities are the mean over the 5 fold models. This notebook blends the saved fold-mean test
# probabilities with **equal weights** (chosen on out-of-fold validation, not on the leaderboard) and writes
# `submission.csv`. Training code: `modal_jobs/train_modal.py`, `modal_jobs/train_lora.py` in the team repository.
#
# Runs: mmbert_small_s1, xlmrl_lr1e-5_ep5, wolf_lr2e-5_ep3, wolf_lr2e-5_ep4, mmbert_base_lr3e-5_ep3, mmbert_small_len768_ep3

# %%
import glob, json, os
import numpy as np, pandas as pd

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
sample = pd.read_csv(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)[0])
bundle = os.path.dirname(glob.glob("/kaggle/input/**/manifest.json", recursive=True)[0])
manifest = json.load(open(f"{bundle}/manifest.json"))
for m in manifest:
    print(f"{m['run']:<32} {m['backbone']:<48} OOF macro-F1 {m['oof_macro_f1']}")

# %%
probs = np.mean([np.load(f"{bundle}/{m['run']}__test_probs.npy") for m in manifest], axis=0)
assert probs.shape == (len(sample), 3)
sub = pd.DataFrame({"id": sample.id, "label": [LABELS[i] for i in probs.argmax(1)]})
assert len(sub) == 6000 and sub.label.isin(LABELS).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.label.value_counts(normalize=True))
