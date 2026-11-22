"""End-to-end downscaling for one pollutant on one region."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .core import (feature_stack, footprint_correct, gap_fill, pattern_gain, residual_correct, standardize,
                   upsample_bilinear)
from .models import (FCNModel, ForestModel, LinearModel, MultiScaleLinearModel, fit_hybrid_cnn,
                     hybrid_predict, mean_anomaly_target)

METHOD_LABELS = {
    "bilinear": "Bilinear interpolation (baseline)",
    "lr": "Linear regression on Sentinel-2 bands (original method)",
    "mslr": "Multi-scale regression (bands + 30/90/270 m context)",
    "rf": "Random forest",
    "cnn": "Hybrid: multi-scale regression + CNN correction",
}


@dataclass
class DownscaleResult:
    pollutant: str
    coarse_obs: np.ndarray           # (h,w) with NaN gaps (target day)
    coarse_filled: np.ndarray        # (h,w)
    maps: dict = field(default_factory=dict)        # method -> (H,W) map at output resolution
    timings: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    gains: dict = field(default_factory=dict)      # pattern strength per method (footprint mode)


def water_mask(raw_feats):
    """Open water from Sentinel-2 (NDWI > 0.05), cleaned of isolated pixels."""
    from scipy import ndimage
    return ndimage.binary_opening(raw_feats[5] > 0.05, iterations=2)


def finish(pred, filled, factor, footprint, method, cnn_pretrained, gain=None, nonneg=True, water=None):
    """Turn a model's static prediction into today's map."""
    if footprint is not None:
        return footprint_correct(pred, filled, factor, footprint, gain, nonneg=nonneg, water=water)
    if method == "cnn" and cnn_pretrained:
        pred = _rescale_to_obs(pred, filled, factor)
    return residual_correct(pred, filled, factor)


def run(bands: np.ndarray, coarse_days: np.ndarray, pollutant: str, methods=("bilinear", "lr", "mslr", "cnn"),
        target_day: int = -1, factor: int = 100, cnn_steps: int = 500, cnn_model: FCNModel | None = None,
        verbose: bool = False, footprint=None) -> DownscaleResult:
    """bands: (4,H,W) Sentinel-2 reflectance; coarse_days: (D,h,w) S5P cells for D overpasses.

    Each method learns the relation between the Sentinel-2 predictors and the
    pollutant from the 1 km cells of all D overpasses, predicts every fine pixel,
    and the result is corrected so each 1 km block matches the target day's
    observation.
    """
    raw = feature_stack(bands)
    feats, mu, sd = standardize(raw)
    obs = coarse_days[target_day]
    filled = gap_fill(obs)
    res = DownscaleResult(pollutant, obs, filled)
    H, W = bands.shape[1:]
    water = water_mask(raw) if footprint is not None else None

    for m in methods:
        t0 = time.time()
        pred, model = fit_static(raw, feats, mu, sd, coarse_days, m, factor, cnn_steps, cnn_model, verbose,
                                 footprint)
        if m == "bilinear":
            res.maps[m] = upsample_bilinear(filled, factor, (H, W))
        else:
            gain = pattern_gain(pred, mean_anomaly_target(coarse_days), factor, footprint) \
                if footprint is not None else None
            res.maps[m] = finish(pred, filled, factor, footprint, m, cnn_model is not None, gain,
                                 nonneg=pollutant != "AER_AI", water=water)
            res.gains[m] = gain
            res.models[m] = model
        res.timings[m] = time.time() - t0
        if verbose:
            print(f"{pollutant} {m}: {res.timings[m]:.1f}s")
    res.feature_mean, res.feature_std = mu, sd
    return res


def fit_static(raw, feats, mu, sd, coarse_days, method, factor, cnn_steps=500, cnn_model=None, verbose=False,
               footprint=None):
    """Fit `method` on all overpasses and return its uncorrected fine-scale prediction.

    footprint: Sentinel-5P footprint in cells (config.S5P_FOOTPRINT_CELLS) for real
    data; None when cells are true block means (simulation, up-scaling test).

    The prediction depends only on the (fixed) Sentinel-2 composite, so it is a
    static spatial pattern; each day's map is this pattern residual-corrected to
    that day's Sentinel-5P observation (see maps_for_days). Bilinear returns None.
    """
    if method == "bilinear":
        return None, None
    if method in ("lr", "mslr", "rf"):
        cls = {"lr": LinearModel, "mslr": MultiScaleLinearModel, "rf": ForestModel}[method]
        model = cls().fit(feats, coarse_days, factor, footprint=footprint)
        return model.predict(feats), model
    if method == "cnn":
        if footprint is not None:
            # real Sentinel-5P: hybrid = multi-scale regression base + CNN correction
            if cnn_model is not None and cnn_model.meta.get("hybrid"):
                f = standardize(raw, cnn_model.feat_mean, cnn_model.feat_std)[0]
                return hybrid_predict(cnn_model, f, feats, coarse_days, factor, footprint), cnn_model
            model = fit_hybrid_cnn([feats], [coarse_days], factor, footprint, steps=cnn_steps, verbose=verbose)
            model.feat_mean, model.feat_std = mu, sd
            return hybrid_predict(model, feats, feats, coarse_days, factor, footprint), model
        # idealised cells (simulation): plain CNN with the window aggregation loss
        if cnn_model is not None:
            f = standardize(raw, cnn_model.feat_mean, cnn_model.feat_std)[0] \
                if cnn_model.feat_mean is not None else feats
            return cnn_model.predict(f), cnn_model
        wc = max(1, min(3, coarse_days.shape[1], coarse_days.shape[2]))
        model = FCNModel(in_ch=feats.shape[0]).fit(feats, coarse_days, factor, steps=cnn_steps, verbose=verbose,
                                                   window_cells=wc)
        model.feat_mean, model.feat_std = mu, sd
        return model.predict(feats), model
    raise ValueError(f"unknown method {method}")


def maps_for_days(bands, coarse_days, method, factor, cnn_steps=500, cnn_model=None, footprint=None):
    """Yield (day_index, fine_map) for every overpass, fitting the method once."""
    raw = feature_stack(bands)
    feats, mu, sd = standardize(raw)
    pred, _ = fit_static(raw, feats, mu, sd, coarse_days, method, factor, cnn_steps, cnn_model,
                         footprint=footprint)
    H, W = bands.shape[1:]
    gain = pattern_gain(pred, mean_anomaly_target(coarse_days), factor, footprint) \
        if (pred is not None and footprint is not None) else None
    water = water_mask(raw) if footprint is not None else None
    for d in range(coarse_days.shape[0]):
        filled = gap_fill(coarse_days[d])
        if pred is None:
            yield d, upsample_bilinear(filled, factor, (H, W))
        else:
            yield d, finish(pred, filled, factor, footprint, method, cnn_model is not None, gain, water=water)


def _rescale_to_obs(pred: np.ndarray, filled: np.ndarray, factor: int) -> np.ndarray:
    """Linear map of a pretrained prediction onto today's 1 km values (least squares on block means)."""
    from .core import block_mean
    bm = block_mean(pred, factor).ravel()
    y = filled.ravel()
    A = np.c_[bm, np.ones_like(bm)]
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    if not np.isfinite(a) or a <= 0:      # degenerate (e.g. a single cell): keep shape, match mean
        a, b = (y.mean() / (bm.mean() + 1e-30)), 0.0
    return (a * pred + b).astype(np.float32)
