"""Validation on real data.

1. Ground validation: downscaled maps vs. monitoring stations
   --------------------------------------------------------
   For each station, a tile around it is downscaled for every Sentinel-5P overpass
   in the period. Three satellite-side values are sampled at the station (3x3-pixel
   mean): the raw 1 km Sentinel-5P cell, bilinear interpolation and each
   downscaling method. They are compared with the station's overpass-time mean.
   Only days on which the satellite actually observed the station's cell (no
   cloud) are used, so every predictor is judged on the same pairs.

   Metrics (columns cannot be compared to surface µg/m³ directly, see docs/VALIDATION.md):
     pattern:     daily spatial Pearson r and Spearman ρ across stations (median over
                  days), pooled r, per-station temporal r (median);
     value:       leave-one-station-out linear calibration → RMSE, MAE, bias,
                  normalised mean bias, R² (µg/m³), on stations not used to calibrate;
     added value: skill score = 1 − RMSE(method) / RMSE(raw 1 km);
     decisions:   CPCB AQI category accuracy, hotspot precision/recall/F1
                  (top-quartile stations per day).

2. Up-scaling (scale-transfer) test: no ground data needed
   --------------------------------------------------------
   Real 1 km Sentinel-5P is averaged to 4 km, downscaled back to 1 km with the
   same methods (Sentinel-2 predictors averaged to 1 km), and compared with the
   real 1 km observations. "Detail r" correlates only the sub-4 km deviations,
   i.e. exactly the information the method had to add.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import gee
from .core import block_mean, gap_fill
from .ground import PROXY, Station
from .config import S5P_FOOTPRINT_CELLS
from .pipeline import maps_for_days

# CPCB National AQI breakpoints (upper bounds of Good, Satisfactory, Moderate, Poor, Very poor), µg/m³.
# Defined on 24 h (NO2, SO2, PM2.5) or 8 h (CO, O3) means; applied here to overpass-time means.
AQI_BREAKS = {
    "NO2": [40, 80, 180, 280, 400],
    "SO2": [40, 80, 380, 800, 1600],
    "CO": [1000, 2000, 10000, 17000, 34000],
    "O3": [50, 100, 168, 208, 748],
    "AER_AI": [30, 60, 90, 120, 250],   # PM2.5 proxy
}
AQI_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]


# ----------------------------------------------------------------------------
# Small statistics helpers (no SciPy needed at runtime)
# ----------------------------------------------------------------------------
def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3:
        return math.nan
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if den <= 1e-300 or np.ptp(x) <= 1e-12 * (abs(x).max() + 1e-300) or np.ptp(y) <= 1e-12 * (abs(y).max() + 1e-300):
        return math.nan
    return float((x * y).sum() / den)


def _rank(a):
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    for v in np.unique(a):            # average ties
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    return pearson(_rank(x), _rank(y)) if len(x) >= 3 else math.nan


def clean_json(o):
    """Replace NaN/inf with None recursively (NaN is not valid JSON)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating, np.integer)):
        return clean_json(o.item())
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    return o


def nanmedian(v):
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else math.nan


# ----------------------------------------------------------------------------
# Collocation
# ----------------------------------------------------------------------------
def station_bbox(st: Station, half_km=3.0):
    dlat = half_km / 110.7
    dlon = half_km / (111.32 * math.cos(math.radians(st.lat)))
    return [st.lon - dlon, st.lat - dlat, st.lon + dlon, st.lat + dlat]


def sample(field_, lat: "gee.Lattice", lon, la, radius=1):
    c = int((lon - lat.bounds[0]) / (lat.bounds[2] - lat.bounds[0]) * field_.shape[1])
    r = int((lat.bounds[3] - la) / (lat.bounds[3] - lat.bounds[1]) * field_.shape[0])
    if not (0 <= r < field_.shape[0] and 0 <= c < field_.shape[1]):
        return math.nan
    win = field_[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1]
    return float(np.nanmean(win))


def collocate(stations, pollutant, end_day, methods, n_days=60, window_days=120, res_m=10,
              cnn_model=None, cnn_steps=150, log=print):
    """Return records: dict(station, name, day, ground, coarse, bilinear, <method>...)."""
    records = []
    for k, st in enumerate(stations, 1):
        t0 = time.time()
        try:
            bands, coarse, lat, prov = gee.fetch_stack(pollutant, end_day, station_bbox(st), res_m=res_m,
                                                       n_days=n_days, window_days=window_days)
        except Exception as e:
            log(f"[{k}/{len(stations)}] {st.name}: no satellite data ({e})")
            continue
        days = prov["s5p_days"]
        rows = {}
        cy = int((lat.bounds[3] - st.lat) / (lat.bounds[3] - lat.bounds[1]) * coarse.shape[1])
        cx = int((st.lon - lat.bounds[0]) / (lat.bounds[2] - lat.bounds[0]) * coarse.shape[2])
        for d, day in enumerate(days):
            g = st.daily.get(day)
            v = coarse[d, cy, cx]
            if g is None or np.isnan(v):
                continue                     # station missing or satellite cell under cloud
            rows[d] = {"station": st.id, "name": st.name, "lat": st.lat, "lon": st.lon, "day": day,
                       "ground": g, "coarse": float(v)}
        if not rows:
            log(f"[{k}/{len(stations)}] {st.name}: no matching days")
            continue
        for m in ["bilinear"] + [m for m in methods if m != "bilinear"]:
            for d, fine in maps_for_days(bands, coarse, m, lat.factor, cnn_steps=cnn_steps,
                                         cnn_model=cnn_model if m == "cnn" else None,
                                         footprint=S5P_FOOTPRINT_CELLS):
                if d in rows:
                    rows[d][m] = sample(fine, lat, st.lon, st.lat)
        records.extend(rows.values())
        log(f"[{k}/{len(stations)}] {st.name}: {len(rows)} matched days ({time.time() - t0:.0f}s)")
    return records


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _loso_predictions(records, key):
    """Leave-one-station-out linear calibration predictor -> µg/m³."""
    stations = sorted({r["station"] for r in records})
    out = {}
    for s in stations:
        train = [r for r in records if r["station"] != s and r.get(key) == r.get(key)]
        test = [r for r in records if r["station"] == s and r.get(key) == r.get(key)]
        if len(train) < 5 or not test:
            continue
        x = np.array([r[key] for r in train])
        y = np.array([r["ground"] for r in train])
        a, b = np.polyfit(x, y, 1)
        for r in test:
            out[id(r)] = a * r[key] + b
    return out


def _cat(v, breaks):
    return int(np.searchsorted(breaks, v, side="left"))


def ground_metrics(records, pollutant, predictors):
    """Compute all metrics for each predictor ('coarse', 'bilinear', methods...)."""
    by_day = {}
    for r in records:
        by_day.setdefault(r["day"], []).append(r)
    by_station = {}
    for r in records:
        by_station.setdefault(r["station"], []).append(r)
    breaks = AQI_BREAKS.get(pollutant)
    out = {"n_pairs": len(records), "n_stations": len(by_station), "n_days": len(by_day),
           "proxy": pollutant in PROXY, "predictors": {}}
    base_rmse = None
    for key in predictors:
        recs = [r for r in records if r.get(key) == r.get(key) and key in r]
        if len(recs) < 5:
            continue
        # pattern
        sp_r, sp_rho, hot = [], [], []
        for day, rs in by_day.items():
            rs = [r for r in rs if key in r and r[key] == r[key]]
            if len(rs) >= 4:
                g = [r["ground"] for r in rs]
                p = [r[key] for r in rs]
                sp_r.append(pearson(p, g))
                sp_rho.append(spearman(p, g))
                k = max(1, len(rs) // 4)
                top_g = set(np.argsort(g)[-k:])
                top_p = set(np.argsort(p)[-k:])
                tp = len(top_g & top_p)
                hot.append(tp / k)          # precision = recall = F1 when both sets have size k
        tr = []
        for s, rs in by_station.items():
            rs = [r for r in rs if key in r and r[key] == r[key]]
            if len(rs) >= 8:
                tr.append(pearson([r[key] for r in rs], [r["ground"] for r in rs]))
        # value (calibrated, held-out stations)
        pred = _loso_predictions(recs, key)
        pairs = [(pred[id(r)], r["ground"]) for r in recs if id(r) in pred]
        m = {"daily_spatial_r_median": nanmedian(sp_r), "daily_spatial_spearman_median": nanmedian(sp_rho),
             "days_with_4plus_stations": len(sp_r),
             "pooled_r": pearson([r[key] for r in recs], [r["ground"] for r in recs]),
             "temporal_r_median": nanmedian(tr), "stations_with_8plus_days": len(tr),
             "hotspot_f1_top_quartile": float(np.mean(hot)) if hot else math.nan}
        if pairs:
            p, g = np.array(pairs).T
            e = p - g
            rmse = float(np.sqrt(np.mean(e ** 2)))
            m.update(rmse_ug_m3=rmse, mae_ug_m3=float(np.mean(np.abs(e))), bias_ug_m3=float(e.mean()),
                     normalised_mean_bias=float(e.sum() / g.sum()),
                     r2_calibrated=float(1 - np.sum(e ** 2) / np.sum((g - g.mean()) ** 2)),
                     n_calibrated=len(pairs))
            if breaks:
                m["aqi_category_accuracy"] = float(np.mean([_cat(a, breaks) == _cat(b, breaks) for a, b in pairs]))
            if key == "coarse":
                base_rmse = rmse
        out["predictors"][key] = m
    if base_rmse:
        for key, m in out["predictors"].items():
            if "rmse_ug_m3" in m:
                m["skill_vs_raw_1km"] = 1 - m["rmse_ug_m3"] / base_rmse
    return out


# ----------------------------------------------------------------------------
# Up-scaling test
# ----------------------------------------------------------------------------
def nan_block_mean(a, f, min_valid=0.5):
    """Block mean ignoring NaN; blocks with < min_valid coverage become NaN."""
    D, h, w = a.shape
    h2, w2 = h // f, w // f
    b = a[:, :h2 * f, :w2 * f].reshape(D, h2, f, w2, f)
    valid = (~np.isnan(b)).mean(axis=(2, 4))
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(b, axis=(2, 4))
    m[valid < min_valid] = np.nan
    return m


def upscaling_test(bands_1km, obs_1km, methods, k=4, cnn_steps=150):
    """bands_1km: (4,h,w) Sentinel-2 reflectance averaged to 1 km; obs_1km: (D,h,w) real S5P.
    Coarsens S5P to k km, downscales back to 1 km, scores against the real 1 km values."""
    D, h, w = obs_1km.shape
    h2, w2 = (h // k) * k, (w // k) * k
    bands = bands_1km[:, :h2, :w2]
    obs = obs_1km[:, :h2, :w2]
    coarse = nan_block_mean(obs, k)
    keep = [d for d in range(D) if (~np.isnan(coarse[d])).mean() >= 0.3]
    coarse, obs = coarse[keep], obs[keep]
    out = {"k_km": k, "days": len(keep), "grid_1km": [h2, w2], "methods": {}}
    for m in ["bilinear"] + [m for m in methods if m != "bilinear"]:
        rs, drs, ys, ps = [], [], [], []
        for d, fine in maps_for_days(bands, coarse, m, k, cnn_steps=cnn_steps):
            truth = obs[d]
            ok = ~np.isnan(truth) & ~np.isnan(np.repeat(np.repeat(coarse[d], k, 0), k, 1))
            if ok.sum() < 10:
                continue
            blk = np.repeat(np.repeat(gap_fill(coarse[d]), k, 0), k, 1)
            rs.append(pearson(fine[ok], truth[ok]))
            drs.append(pearson((fine - blk)[ok], (truth - blk)[ok]))
            ys.append(truth[ok])
            ps.append(fine[ok])
        if not ys:
            continue
        y, p = np.concatenate(ys), np.concatenate(ps)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        out["methods"][m] = {"spatial_r_median": nanmedian(rs), "detail_r_median": nanmedian(drs),
                             "rmse": rmse, "nrmse": rmse / float(np.mean(np.abs(y))), "n_cells": int(len(y))}
    base = out["methods"].get("bilinear", {}).get("rmse")
    if base:
        for m, v in out["methods"].items():
            v["skill_vs_bilinear"] = 1 - v["rmse"] / base
    return out


def fetch_upscaling_inputs(pollutant, end_day, bbox, n_days=30, window_days=90, s2_res_m=100):
    """Real S5P at 1 km and Sentinel-2 averaged to 1 km over bbox."""
    bands, coarse, lat, prov = gee.fetch_stack(pollutant, end_day, bbox, res_m=s2_res_m, n_days=n_days,
                                               window_days=window_days)
    bands_1km = block_mean(bands, lat.factor)
    return bands_1km, coarse, prov
