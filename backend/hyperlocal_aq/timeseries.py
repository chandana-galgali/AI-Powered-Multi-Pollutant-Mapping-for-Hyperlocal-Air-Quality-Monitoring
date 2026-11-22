"""Temporal gap filling (from 'Decomposition Model.ipynb').

When a cell is cloudy for many consecutive overpasses, spatial IDW is not
enough. Each cell's time series is split into trend + seasonal components
(additive decomposition, as in the notebook) and missing values are filled
from them. Pure NumPy, no statsmodels needed.
"""
from __future__ import annotations

import numpy as np


def _moving_average(x: np.ndarray, period: int) -> np.ndarray:
    k = np.ones(period) / period
    pad = period // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[: len(x)]


def decompose(series: np.ndarray, period: int = 12):
    """Additive decomposition of a 1-D series with NaN gaps -> trend, seasonal."""
    s = series.astype(float).copy()
    idx = np.arange(len(s))
    ok = ~np.isnan(s)
    if ok.sum() < 2:
        return np.full_like(s, np.nanmean(s) if ok.any() else 0.0), np.zeros_like(s)
    s_interp = np.interp(idx, idx[ok], s[ok])
    trend = _moving_average(s_interp, period)
    detr = s - trend
    seasonal = np.zeros(period)
    for p in range(period):
        v = detr[p::period]
        v = v[~np.isnan(v)]
        seasonal[p] = v.mean() if len(v) else 0.0
    seasonal -= seasonal.mean()
    return trend, np.resize(seasonal, len(s))


def seasonal_fill(stack: np.ndarray, period: int = 12) -> np.ndarray:
    """stack: (T,h,w) time series of coarse grids with NaN gaps -> filled copy."""
    out = stack.copy()
    T, h, w = stack.shape
    for y in range(h):
        for x in range(w):
            s = stack[:, y, x]
            if np.isnan(s).any():
                trend, seas = decompose(s, period=min(period, max(2, T // 2)))
                fill = trend + seas
                out[:, y, x] = np.where(np.isnan(s), fill, s)
    return out
