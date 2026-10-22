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

# cfg["model"] key -> approved backbone id (mirrors MODELS in modal_jobs/train_modal.py and LLMS in train_lora.py;
# copied because importing those modules pulls in modal)
MODELS = {
    "distilbert": "distilbert/distilbert-base-multilingual-cased",
    "mmbert-small": "jhu-clsp/mmBERT-small",
    "mmbert-base": "jhu-clsp/mmBERT-base",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "rembert": "google/rembert",
    "wolf": "patronus-studio/wolf-defender-prompt-injection",
    "deberta-pi": "protectai/deberta-v3-base-prompt-injection-v2",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "granite": "ibm-granite/granite-3.3-2b-instruct",
}


def model_of(cfg: dict | None) -> str:
    key = (cfg or {}).get("model")
    assert key in MODELS, f"unknown backbone {key!r} — refusing to bundle a model outside the approved list"
    return MODELS[key]


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
        cfg = summ.get("cfg") or {}
        manifest.append({"run": r, "backbone": model_of(cfg), "oof_macro_f1": oof, "cfg": cfg,
                         "full_data": oof is None, "pseudo": cfg.get("pseudo") or None})
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # ship the exact training code so reviewers can audit how every probability file was produced
    for code in ("modal_jobs/train_modal.py", "src/c1/metrics.py", "src/c1/ensemble.py"):
        shutil.copy(REPO / code, ds / Path(code).name)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{args.kaggle_user}/{slug}", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, indent=1))

    backbones = sorted({m["backbone"] for m in manifest})
    pseudo_runs = [m for m in manifest if m["pseudo"]]
    full_runs = [m["run"] for m in manifest if m["full_data"]]
    if pseudo_runs:
        data_line = ("**Data:** the competition's `train.csv` only — no external data, no manual or LLM labels. "
                     + "; ".join(f"`{m['run']}` additionally trained on confident (p>={m['cfg'].get('pseudo_thr')}) "
                                 f"pseudo-labels for test-set TEXT produced by our own earlier ensemble (`{m['pseudo']}`) "
                                 "of approved models" for m in pseudo_runs) + ".")
    else:
        data_line = "**Data:** only the competition's `train.csv` (no external data, no pseudo-labels, no manual labels)."
    full_line = (f" Runs trained on 100% of `train.csv` (single model, no fold split): {', '.join(full_runs)}." if full_runs else "")
    py = f'''# %% [markdown]
# # RMIT_SGS_Fprinciples_C1 — ensemble submission ({args.blend})
# **Models used (all from the approved list):** {", ".join(f"`{b}`" for b in backbones)}
#
# {data_line}
# Each 5-fold run fully fine-tuned the backbone on `train.csv` with the same stratified split (label x language, seed 42)
# on cloud GPUs; its test probabilities are the mean over the 5 fold models.{full_line} This notebook blends the saved
# test probabilities with **equal weights** (chosen on out-of-fold validation, not on the leaderboard) and writes
# `submission.csv`. The exact training script is attached in the input dataset (`train_modal.py`) and printed below.
#
# Runs: {", ".join(m["run"] for m in manifest)}

# %%
# Run configurations and out-of-fold (5-fold CV) macro-F1 of every ensemble member
import glob, json, os
manifests = glob.glob("/kaggle/input/**/manifest.json", recursive=True)
assert len(manifests) == 1, f"attach exactly one bundle dataset, found {{manifests}}"
bundle = os.path.dirname(manifests[0])
for m in json.load(open(f"{{bundle}}/manifest.json")):
    c = m["cfg"] or {{}}
    oof = "n/a (100% train refit)" if m["oof_macro_f1"] is None else f"{{m['oof_macro_f1']:.4f}}"
    print(f"{{m['run']:<26}} {{m['backbone']:<46}} OOF={{oof}} lr={{c.get('lr')}} epochs={{c.get('epochs')}} "
          f"max_len={{c.get('max_len')}} batch={{c.get('batch_size')}} seed={{c.get('seed')}} pseudo={{c.get('pseudo') or '-'}}")

# %%
# Training code used for every run (fine-tuning loop, folds, tokenisation, outputs)
print(open(f"{{bundle}}/train_modal.py", encoding="utf-8").read())

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
