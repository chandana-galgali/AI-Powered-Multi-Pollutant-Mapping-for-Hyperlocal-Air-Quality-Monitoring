"""Skill scores for a downscaled map against a reference (OSSE truth or in-situ)."""
from __future__ import annotations

import numpy as np

from .core import block_mean

HOTSPOT_LEVELS = ["Low", "Moderate", "High", "Hotspot"]
HOTSPOT_QUANTILES = [0.50, 0.80, 0.95]


def hotspot_thresholds(reference: np.ndarray, mask=None):
    vals = reference[mask] if mask is not None else reference.ravel()
    return np.quantile(vals, HOTSPOT_QUANTILES)


def classify(field: np.ndarray, thresholds) -> np.ndarray:
    return np.digitize(field, thresholds).astype(np.uint8)


def evaluate(pred: np.ndarray, truth: np.ndarray, mask=None, factor: int = 100,
             coarse_obs: np.ndarray | None = None, tolerance: float = 0.10) -> dict:
    """Return R², RMSE, MAE, bias, accuracy within ±tolerance, hotspot-class accuracy,
    hotspot (top 5 %) detection F1 and 1 km consistency with the satellite."""
    if mask is None:
        mask = np.ones_like(truth, dtype=bool)
    p, t = pred[mask].astype(np.float64), truth[mask].astype(np.float64)
    err = p - t
    ss_res = float((err ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum()) + 1e-30
    thr = hotspot_thresholds(truth, mask)
    cp, ct = classify(p, thr), classify(t, thr)
    hot_p, hot_t = cp == 3, ct == 3
    tp = float((hot_p & hot_t).sum())
    prec = tp / max(hot_p.sum(), 1)
    rec = tp / max(hot_t.sum(), 1)
    out = {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(ss_res / len(t))),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "within_tolerance": float((np.abs(err) <= tolerance * np.abs(t)).mean()),
        "hotspot_class_accuracy": float((cp == ct).mean()),
        "hotspot_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "tolerance": tolerance,
    }
    if coarse_obs is not None:
        bm = block_mean(pred, factor)
        ok = ~np.isnan(coarse_obs)
        out["coarse_consistency_rmse"] = float(np.sqrt(np.mean((bm[ok] - coarse_obs[ok]) ** 2)))
    return out
