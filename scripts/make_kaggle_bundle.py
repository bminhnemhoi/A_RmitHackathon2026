"""Package an ensemble for Kaggle: a private dataset folder + a submission notebook that only blends saved probabilities.

Usage: python scripts/make_kaggle_bundle.py --blend blend6_equal --runs <run1> <run2> ... [--kaggle-user USER]
Creates outputs/kaggle_bundle/<blend>/ (dataset: per-run test_probs + manifest) and
notebooks/RMIT_SGS_Fprinciples_C1_<blend>.ipynb.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from py_to_ipynb import convert  # noqa: E402

MODEL_IDS = {
    "mmbert_small": "jhu-clsp/mmBERT-small", "mmbert_base": "jhu-clsp/mmBERT-base", "mmb_": "jhu-clsp/mmBERT-base",
    "xlmrl": "FacebookAI/xlm-roberta-large", "wolf": "patronus-studio/wolf-defender-prompt-injection",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3", "granite": "ibm-granite/granite-3.3-2b-instruct",
}


def model_of(run: str) -> str:
    return next((v for k, v in MODEL_IDS.items() if run.startswith(k)), "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--kaggle-user", default="KAGGLE_USERNAME")
    args = ap.parse_args()

    slug = f"fprinciples-c1-{args.blend.replace('_', '-')}"
    ds = REPO / "outputs" / "kaggle_bundle" / args.blend
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    manifest = []
    for r in args.runs:
        src = REPO / "outputs" / "runs" / r
        np.save(ds / f"{r}__test_probs.npy", np.load(src / "test_probs.npy"))
        summ = json.loads((src / "summary.json").read_text())
        oof = summ.get("oof_raw", summ.get("oof", {})).get("macro_f1")
        manifest.append({"run": r, "backbone": model_of(r), "oof_macro_f1": oof, "cfg": summ.get("cfg")})
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{args.kaggle_user}/{slug}", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, indent=1))

    backbones = sorted({m["backbone"] for m in manifest})
    py = f'''# %% [markdown]
# # RMIT_SGS_Fprinciples_C1 — ensemble submission ({args.blend})
# **Models used (all from the approved list):** {", ".join(f"`{b}`" for b in backbones)}
#
# Each backbone was fine-tuned on `train.csv` with the same stratified 5-fold split (label x language, seed 42);
# test probabilities are the mean over the 5 fold models. This notebook blends the saved fold-mean test
# probabilities with **equal weights** (chosen on out-of-fold validation, not on the leaderboard) and writes
# `submission.csv`. Training code: `modal_jobs/train_modal.py`, `modal_jobs/train_lora.py` in the team repository.
#
# Runs: {", ".join(m["run"] for m in manifest)}

# %%
import glob, json, os
import numpy as np, pandas as pd

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
sample = pd.read_csv(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)[0])
bundle = os.path.dirname(glob.glob("/kaggle/input/**/manifest.json", recursive=True)[0])
manifest = json.load(open(f"{{bundle}}/manifest.json"))
for m in manifest:
    print(f"{{m['run']:<32}} {{m['backbone']:<48}} OOF macro-F1 {{m['oof_macro_f1']}}")

# %%
probs = np.mean([np.load(f"{{bundle}}/{{m['run']}}__test_probs.npy") for m in manifest], axis=0)
assert probs.shape == (len(sample), 3)
sub = pd.DataFrame({{"id": sample.id, "label": [LABELS[i] for i in probs.argmax(1)]}})
assert len(sub) == 6000 and sub.label.isin(LABELS).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.label.value_counts(normalize=True))
'''
    py_path = REPO / "notebooks" / f"submit_{args.blend}.py"
    py_path.write_text(py, encoding="utf-8")
    nb_path = REPO / "notebooks" / f"RMIT_SGS_Fprinciples_C1_{args.blend}.ipynb"
    convert(str(py_path), str(nb_path))
    print("dataset folder:", ds)
    print("notebook:", nb_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
