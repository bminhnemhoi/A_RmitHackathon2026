"""Blend saved probability files from several runs (val + test) into one submission."""
from __future__ import annotations

import itertools

import numpy as np

from .metrics import apply_class_weights, fit_class_weights, macro_f1


def blend(prob_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, prob_list))


def fit_blend_weights(val_probs: list[np.ndarray], y_val: np.ndarray, step: float = 0.1) -> tuple[list[float], float]:
    """Grid-search model weights on the validation split (all runs must share the SAME split)."""
    n = len(val_probs)
    if n == 1:
        return [1.0], macro_f1(y_val, val_probs[0].argmax(1))
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    best_w, best = [1.0 / n] * n, -1.0
    for combo in itertools.product(grid, repeat=n - 1):
        last = 1.0 - sum(combo)
        if last < -1e-9:
            continue
        w = list(combo) + [max(last, 0.0)]
        score = macro_f1(y_val, blend(val_probs, w).argmax(1))
        if score > best:
            best, best_w = score, w
    return best_w, best


def fit_and_predict(val_probs, y_val, test_probs):
    """Return (test_pred_ids, info). Blend weights and class multipliers are fitted on validation only."""
    w, blend_score = fit_blend_weights(val_probs, y_val)
    val_blend = blend(val_probs, w)
    class_w, final_score = fit_class_weights(val_blend, y_val)
    test_pred = apply_class_weights(blend(test_probs, w), class_w)
    info = {"blend_weights": w, "val_macro_f1_blend": blend_score,
            "class_weights": class_w.tolist(), "val_macro_f1_final": final_score}
    return test_pred, info
