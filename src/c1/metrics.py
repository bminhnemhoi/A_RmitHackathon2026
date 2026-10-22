"""Metrics and macro-F1 post-processing shared by all C1 notebooks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, f1_score

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABELS)))))


def report(y_true: np.ndarray, y_pred: np.ndarray, languages: np.ndarray | None = None) -> dict:
    """Overall, per-class and (optionally) per-language macro-F1 — log these to experiments/log.csv."""
    out = {"macro_f1": macro_f1(y_true, y_pred)}
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))))
    out.update({f"f1_{label.lower()}": float(v) for label, v in zip(LABELS, per_class)})
    if languages is not None:
        for lang in sorted(set(languages)):
            mask = languages == lang
            out[f"f1_{lang}"] = macro_f1(y_true[mask], y_pred[mask])
    return out


def print_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))


def apply_class_weights(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.argmax(probs * weights[None, :], axis=1)


def fit_class_weights(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
    n_rounds: int = 3,
) -> tuple[np.ndarray, float]:
    """Coordinate search for per-class multipliers that maximise macro-F1 on a VALIDATION split.

    Never call this with test predictions. Class 0 (BENIGN) stays fixed at 1.0 as the reference.
    """
    grid = np.round(np.arange(0.3, 4.01, 0.05), 2) if grid is None else grid
    weights = np.ones(probs.shape[1])
    best = macro_f1(y_true, apply_class_weights(probs, weights))
    for _ in range(n_rounds):
        for c in range(1, probs.shape[1]):
            for w in grid:
                trial = weights.copy()
                trial[c] = w
                score = macro_f1(y_true, apply_class_weights(probs, trial))
                if score > best + 1e-6:
                    best, weights = score, trial
    return weights, best
