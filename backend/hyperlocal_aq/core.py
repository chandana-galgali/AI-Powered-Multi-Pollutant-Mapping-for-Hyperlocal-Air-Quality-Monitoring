"""Building blocks shared by every downscaling model.

The flow for one pollutant and one date:

    coarse S5P grid (with cloud gaps) --gap_fill--> complete coarse grid
    Sentinel-2 bands --feature_stack--> 10 m predictors
    model.fit(predictors, coarse)  (trained only on coarse cells: no 10 m labels exist)
    model.predict(predictors) --residual_correct--> 10 m map whose 1 km means
                                                     match the satellite
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURE_NAMES = ["B2", "B3", "B4", "B8", "NDVI", "NDWI"]


def feature_stack(bands: np.ndarray) -> np.ndarray:
    """(4,H,W) reflectance B2,B3,B4,B8 -> (6,H,W) predictors incl. spectral indices."""
    b2, b3, b4, b8 = bands
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    return np.stack([b2, b3, b4, b8, ndvi, ndwi]).astype(np.float32)


def standardize(feats: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = feats.reshape(feats.shape[0], -1).mean(axis=1)
        std = feats.reshape(feats.shape[0], -1).std(axis=1) + 1e-6
    z = (feats - mean[:, None, None]) / std[:, None, None]
    return z.astype(np.float32), mean, std


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool the last two axes by `factor` (fine 10 m -> coarse 1 km)."""
    *lead, h, w = a.shape
    h2, w2 = h // factor, w // factor
    a = a[..., : h2 * factor, : w2 * factor]
    return a.reshape(*lead, h2, factor, w2, factor).mean(axis=(-3, -1))


def gap_fill(coarse: np.ndarray, power: float = 2.0, radius: int = 3) -> np.ndarray:
    """Fill cloud gaps (NaN) by inverse-distance weighting of nearby valid cells.

    Matches the gap-handling approach in the proposal ("interpolating from adjacent
    data points"); cells with no valid neighbour within `radius` get the
    global mean. For long gaps see timeseries.seasonal_fill.
    """
    out = coarse.copy()
    valid = ~np.isnan(coarse)
    if valid.all():
        return out
    vy, vx = np.nonzero(valid)
    vals = coarse[valid]
    for y, x in zip(*np.nonzero(~valid)):
        d = np.hypot(vy - y, vx - x)
        m = d <= radius
        if not m.any():
            out[y, x] = np.nanmean(coarse)
            continue
        wgt = 1.0 / d[m] ** power
        out[y, x] = float((wgt * vals[m]).sum() / wgt.sum())
    return out


def upsample_bilinear(coarse: np.ndarray, factor: int, shape=None) -> np.ndarray:
    """Bilinear interpolation between coarse-cell centres (edge-clamped)."""
    h, w = coarse.shape
    H, W = shape or (h * factor, w * factor)
    ys = (np.arange(H) + 0.5) / factor - 0.5
    xs = (np.arange(W) + 0.5) / factor - 0.5
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    return ndimage.map_coordinates(coarse.astype(np.float64),
                                   np.meshgrid(ys, xs, indexing="ij"),
                                   order=1, mode="nearest").astype(np.float32)


def residual_correct(pred_fine: np.ndarray, coarse_filled: np.ndarray, factor: int,
                     iterations: int = 10) -> np.ndarray:
    """Make the 10 m map consistent with the satellite at 1 km.

    The residual (observed - mean of prediction) in each cell is interpolated
    bilinearly and added back. The error roughly halves every iteration, so
    10 iterations bring the block means to within ~0.1 % of the observation
    while the result stays smooth (no blocky 1 km seams).
    """
    out = pred_fine.astype(np.float32).copy()
    for _ in range(iterations):
        resid = coarse_filled - block_mean(out, factor)
        out += upsample_bilinear(resid, factor, out.shape)
    return out


def pattern_gain(pred_fine: np.ndarray, target: np.ndarray, factor: int, footprint) -> float:
    """How strongly the model's standardised pattern expresses itself in the satellite's
    average anomaly pattern (regression of target on footprint(blockmean(pattern)))."""
    from .models import apply_footprint
    p = (pred_fine - pred_fine.mean()) / (pred_fine.std() + 1e-30)
    sb = apply_footprint(block_mean(p, factor), footprint)
    sb = sb - sb.mean()
    a = float((sb * target).sum() / ((sb * sb).sum() + 1e-30))
    return max(a, 0.0)


def footprint_correct(pred_fine: np.ndarray, coarse_filled: np.ndarray, factor: int, footprint,
                      gain: float, smooth_cells: float = 2.0, nonneg: bool = True,
                      water: np.ndarray | None = None) -> np.ndarray:
    """Turn the model's static 10 m pattern into today's map when cells are footprint averages.

    1. today's level + today's spatial spread x gain x standardised pattern, where `gain`
       comes from the whole season (pattern_gain), not from one noisy day;
    2. plus the remaining residual, smoothed over ~2 km, so today's regional gradients
       (wind, plumes) from the satellite are kept without forcing every 1 km block to a
       blurred value.
    """
    from scipy import ndimage
    from .models import apply_footprint
    p = (pred_fine - pred_fine.mean()) / (pred_fine.std() + 1e-30)
    if water is not None and water.any():
        # open water has no sources at 10 m: keep only the smooth (~300 m) part of the pattern there,
        # so waves and sun glint are not read as pollution
        p = np.where(water, ndimage.gaussian_filter(p, 30), p).astype(np.float32)
    level, spread = float(coarse_filled.mean()), float(coarse_filled.std())
    out = (level + spread * gain * p).astype(np.float32)
    r = coarse_filled - apply_footprint(block_mean(out, factor), footprint)
    if smooth_cells:
        r = ndimage.gaussian_filter(r, smooth_cells, mode="nearest")
    out = out + upsample_bilinear(r, factor, out.shape)
    # physical floor: not far below the lowest value the satellite saw here, never below zero
    floor = float(coarse_filled.min()) - 2.0 * spread
    if nonneg:
        floor = max(floor, 0.05 * max(level, 0.0))
    return np.maximum(out, floor)


def tile_windows(valid: np.ndarray, size: int, rng, n: int):
    """Random window origins (in coarse cells) with at least one valid cell."""
    h, w = valid.shape
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        if valid[y:y + size, x:x + size].sum() >= max(1, size * size // 2):
            out.append((y, x))
    return out
