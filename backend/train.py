"""Train the CNN on live Sentinel data and save weights the API serves.

    python train.py --pollutant all                               # default: dry seasons 2022-23 … 2024-25
    python train.py --pollutant NO2 --seasons 2023-24,2024-25 --steps 1500
    python train.py --pollutant NO2 --date 2025-01-31             # single 90-day window instead

Default training period (see docs/VALIDATION.md, "Training period"):
dry seasons (1 Nov – end of Feb) of 2022-23, 2023-24 and 2024-25. This means one
Sentinel-5P processor version (v2.4), no COVID-19 lockdown months, no monsoon
cloud, and the 2025-26 season left unseen for validation.

Training uses every named 10 m area in hyperlocal_aq/service.py (≈450 km² of
Mumbai) and every usable Sentinel-5P overpass in each season. Weights go to
backend/models/<POLLUTANT>_cnn.json; commit them and the server uses them for
"CNN" requests. Without them, the server trains a small CNN per request (slower).

GPU alternative: hyperlocal_aq/models_keras.py (same architecture/JSON format).
Quick check of any method on one area:
    python train.py --evaluate --pollutant NO2 --area chembur_trombay --date 2025-01-31
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta

import numpy as np

from hyperlocal_aq import gee, pipeline, service
from hyperlocal_aq.config import POLLUTANTS, S5P_FOOTPRINT_CELLS
from hyperlocal_aq.core import block_mean, feature_stack, standardize
from hyperlocal_aq.models import (MultiScaleLinearModel, apply_footprint, fit_hybrid_cnn,
                                  hybrid_predict)
from hyperlocal_aq.validation import pearson


DEFAULT_SEASONS = "2022-23,2023-24,2024-25"


def season_window(season: str):
    """'2023-24' -> ('2024-02-29', 121): end day of February and window length from 1 November."""
    y1 = int(season.split("-")[0])
    start = date(y1, 11, 1)
    end = date(y1 + 1, 3, 1) - timedelta(days=1)
    return end.isoformat(), (end - start).days + 1


CACHE_DIR = os.environ.get("TRAIN_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))


def fetch(pollutant, end_day, area, n_days, window_days, label=""):
    """Download (or load from the local cache) one area/period. Re-runs don't spend Earth Engine quota."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{pollutant}_{area}_{label or end_day}.npz")
    t0 = time.time()
    if os.path.exists(path):
        z = np.load(path)
        bands, coarse = z["bands"].astype(np.float32), z["coarse"]
        src = "cache"
    else:
        bbox = service.AREAS[area]["bbox"]
        bands, coarse, lat, prov = gee.fetch_stack(pollutant, end_day, bbox, res_m=10, n_days=n_days,
                                                   window_days=window_days, s2_window_days=window_days,
                                                   progress=lambda s, f: None)
        np.savez_compressed(path, bands=bands.astype(np.float16), coarse=coarse,
                            bounds=np.array(lat.bounds), days=np.array(prov["s5p_days"]))
        src = "Earth Engine"
    print(f"  {label:8s} {area:16s} {bands.shape[2]}x{bands.shape[1]} px, {coarse.shape[0]} overpasses "
          f"({src}, {time.time() - t0:.0f}s)")
    return bands, coarse


def train(pollutant, windows, steps, n_days, out_dir):
    """windows: list of (label, end_day, window_days)."""
    # 10 m training areas only (the whole-city preset is far too large at 10 m)
    areas = [a for a, v in service.AREAS.items()
             if a != "mumbai" and service.area_km2(v["bbox"]) <= service.max_area_km2(10)]
    print(f"{pollutant}: downloading {len(areas)} areas x {len(windows)} period(s)")
    raws, coarses = [], []
    for label, end_day, wdays in windows:
        for a in areas:
            try:
                b, c = fetch(pollutant, end_day, a, n_days, wdays, label)
                raws.append(feature_stack(b))
                coarses.append(c)
            except Exception as e:
                print(f"  {label:8s} {a}: skipped ({e})")
    if not raws:
        raise SystemExit("No data downloaded.")
    # global feature normalisation (stored with the model, reused at inference)
    allf = np.concatenate([r.reshape(r.shape[0], -1) for r in raws], axis=1)
    mu, sd = allf.mean(1), allf.std(1) + 1e-6
    feats = [standardize(r, mu, sd)[0] for r in raws]
    del raws, allf
    print(f"  training hybrid (regression base + CNN correction) on {len(feats)} area-periods, {steps} steps")
    model = fit_hybrid_cnn(feats, coarses, factor=100, footprint=S5P_FOOTPRINT_CELLS, steps=steps,
                           verbose=True, log_every=max(1, steps // 20))
    losses = [l for _, l in model.history]
    q = max(1, len(losses) // 4)
    print(f"  loss, first quarter avg {np.mean(losses[:q]):.3f}  ->  last quarter avg {np.mean(losses[-q:]):.3f}")
    model.feat_mean, model.feat_std, model.res_m = mu, sd, 10
    model.meta = {"hybrid": True, "pollutant": pollutant, "periods": [w[0] for w in windows],
                  "trained_until": max(w[1] for w in windows), "areas": areas, "steps": steps,
                  "overpasses": [int(c.shape[0]) for c in coarses]}
    # Diagnostic: how well does each model's map, seen through the satellite footprint,
    # follow the spatial pattern of every overpass? Compared with multi-scale regression.
    def daily_r(pred, c):
        seen = apply_footprint(block_mean(pred, 100), S5P_FOOTPRINT_CELLS)
        rs = []
        for day_obs in c:
            ok = ~np.isnan(day_obs)
            if ok.sum() > 3:
                r = pearson(seen[ok].astype(np.float64), day_obs[ok].astype(np.float64))
                if r == r:            # days where all cells hold one value carry no pattern
                    rs.append(r)
        return rs
    r_cnn, r_ms = [], []
    for f, c in zip(feats, coarses):
        r_cnn += daily_r(hybrid_predict(model, f, f, c, 100, S5P_FOOTPRINT_CELLS), c)
        ms = MultiScaleLinearModel().fit(f, c, 100, footprint=S5P_FOOTPRINT_CELLS)
        r_ms += daily_r(ms.predict(f), c)
    model.meta["mean_daily_r_1km"] = float(np.mean(r_cnn)) if r_cnn else None
    model.meta["mean_daily_r_1km_mslr_baseline"] = float(np.mean(r_ms)) if r_ms else None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pollutant}_cnn.json")
    model.save(path)
    rc, rm = model.meta["mean_daily_r_1km"], model.meta["mean_daily_r_1km_mslr_baseline"]
    print(f"saved {path}")
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"  pattern match with Sentinel-5P (mean daily r over {len(r_cnn)} area-days): "
          f"hybrid {fmt(rc)}   regression alone {fmt(rm)}")
    if rc is not None and rm is not None and rc < rm - 0.02:
        print("  note: the CNN correction did not help here; the server will still work, "
              "but 'mslr' may be the better choice for this pollutant")


def evaluate(pollutant, day, area, methods):
    bbox = service.AREAS[area]["bbox"]
    bands, coarse, lat, prov = gee.fetch_stack(pollutant, day, bbox, res_m=10)
    print("data:", prov)
    res = pipeline.run(bands, coarse, pollutant, methods=methods, cnn_model=service.pretrained_cnn(pollutant, 10),
                       verbose=True, footprint=S5P_FOOTPRINT_CELLS)
    ok = ~np.isnan(res.coarse_obs)
    for m, mp in res.maps.items():
        seen = apply_footprint(block_mean(mp, 100), S5P_FOOTPRINT_CELLS)
        rmse = float(np.sqrt(np.mean((seen[ok] - res.coarse_obs[ok]) ** 2)))
        print(f"{pipeline.METHOD_LABELS[m]:55s} 1 km consistency RMSE {rmse:.3e}  "
              f"10 m range {np.percentile(mp, 1):.3e} – {np.percentile(mp, 99):.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pollutant", default="NO2", help="NO2, SO2, CO, HCHO, O3, AER_AI or 'all'")
    ap.add_argument("--seasons", default=DEFAULT_SEASONS,
                    help="dry seasons to train on, e.g. 2022-23,2023-24 (1 Nov - end Feb each)")
    ap.add_argument("--date", help="instead of --seasons: last day of a single 90-day window (YYYY-MM-DD)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--days", type=int, default=120, help="max overpasses per area and period")
    ap.add_argument("--out", default=service.MODELS_DIR)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--area", default="chembur_trombay")
    ap.add_argument("--methods", default="bilinear,lr,mslr,cnn")
    a = ap.parse_args()
    pols = list(POLLUTANTS) if a.pollutant == "all" else [a.pollutant.upper()]
    if a.evaluate and not a.date:
        ap.error("--evaluate needs --date")
    windows = [(a.date, a.date, 90)] if a.date else \
        [(s, *season_window(s)) for s in a.seasons.split(",")]
    for p in pols:
        if a.evaluate:
            evaluate(p, a.date, a.area, a.methods.split(","))
        else:
            train(p, windows, a.steps, a.days, a.out)


if __name__ == "__main__":
    main()
