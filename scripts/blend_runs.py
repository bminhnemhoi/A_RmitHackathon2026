"""Blend several K-fold runs (outputs/runs/<run>/{oof,test}_probs.npy) using OOF predictions only.

Usage: python scripts/blend_runs.py --runs wolf_lr2e-5_ep3 xlmrl_lr1e-5_ep3 --name blend_wolf_xlmrl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from c1.ensemble import blend, fit_blend_weights  # noqa: E402
from c1.metrics import LABELS, apply_class_weights, fit_class_weights, report  # noqa: E402
from validate_submission import validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()

    run_dirs = [REPO / "outputs" / "runs" / r for r in args.runs]
    oofs = [np.load(d / "oof_probs.npy") for d in run_dirs]
    tests = [np.load(d / "test_probs.npy") for d in run_dirs]
    sample_path = run_dirs[0] / "_sample.csv"
    # labels/langs cached from train.csv in the same row order used for the folds
    labels_file = REPO / "outputs" / "train_labels.npy"
    langs_file = REPO / "outputs" / "train_langs.npy"
    y, langs = np.load(labels_file), np.load(langs_file, allow_pickle=True)

    mask = ~np.isnan(np.stack(oofs)).any(axis=(0, 2))
    w, blend_f1 = fit_blend_weights([o[mask] for o in oofs], y[mask], step=0.05)
    oof_b, test_b = blend(oofs, w), blend(tests, w)
    out = {"runs": args.runs, "blend_weights": w, "per_run_oof": {}}
    for r, o in zip(args.runs, oofs):
        out["per_run_oof"][r] = report(y[mask], o[mask].argmax(1))["macro_f1"]
    out["blend_raw"] = report(y[mask], oof_b[mask].argmax(1), langs[mask])
    cw = np.ones(3)
    if not args.no_class_weights:
        cw, _ = fit_class_weights(oof_b[mask], y[mask])
    out["class_multipliers"] = cw.tolist()
    out["blend_tuned"] = report(y[mask], apply_class_weights(oof_b[mask], cw), langs[mask])

    dst = REPO / "outputs" / "runs" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / "oof_probs.npy", oof_b)
    np.save(dst / "test_probs.npy", test_b)
    sample = pd.read_csv(sample_path)
    pred = apply_class_weights(test_b, cw)
    pd.DataFrame({"id": sample.id.astype(str).tolist(), "label": [LABELS[i] for i in pred]}).to_csv(dst / "submission.csv", index=False)
    errors = validate(sample_path, dst / "submission.csv")
    out["submission_valid"], out["submission_errors"] = not errors, errors
    (dst / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
