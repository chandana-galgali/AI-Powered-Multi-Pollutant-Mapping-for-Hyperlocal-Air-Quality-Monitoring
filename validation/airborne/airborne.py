"""Column-to-column validation against NASA airborne NO2 maps (LISTOS 2018, New York).

The GCAS spectrometer on NASA's B-200 mapped the NO2 column below the aircraft
(~9 km altitude, i.e. effectively the tropospheric column) at ~250 m over New York
City during the Sentinel-5P overpass. This is an independent, high-resolution
measurement of the same quantity Sentinel-5P measures, so it is a direct answer key
for the downscaled maps.

Step 1 (no Earth Engine needed) — shrink each ~750 MB HDF5 file to a small .npz:
    python validation/airborne/airborne.py extract

Step 2 (Earth Engine credentials, like the server) — run the system on New York
tiles covered by each flight and compare at 250 m:
    python validation/airborne/airborne.py compare

The model is NOT trained or tuned on any airborne data: the CNN is the one trained
on Mumbai, and the regressions are fitted on Sentinel-5P/Sentinel-2 only. The aircraft
data is used only for scoring. Results go to validation/results/airborne_NO2.json
(`python validation/airborne/airborne.py summary` re-prints them without Earth Engine).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "listos2018")
RESULTS = os.path.join(ROOT, "validation", "results")
MOLEC_CM2_PER_MOL_M2 = 6.02214076e19
# TROPOMI pixel before 6 Aug 2019: 7 km along track x 3.5 km across → (rows, cols) in ~1 km cells
FOOTPRINT_2018 = (7, 4)


# ---------------------------------------------------------------------------
# Step 1: extract
# ---------------------------------------------------------------------------
def _read(f, path):
    d = f[path]
    return np.asarray(d[()] if d.shape == () else d[:])


def _text(arr):
    out = []
    for x in np.ravel(arr):
        out.append(x.decode(errors="replace") if isinstance(x, bytes) else str(x))
    return "\n".join(s.rstrip() for s in out)


def extract(_args):
    sys.path.insert(0, HERE)
    from h5read import h5lib
    files = sorted(glob.glob(os.path.join(DATA, "*.h5")))
    if not files:
        sys.exit(f"No .h5 files in {DATA}")
    for p in files:
        out = p[:-3] + ".npz"
        if os.path.exists(out):
            print("exists", os.path.basename(out))
            continue
        t0 = time.time()
        f = h5lib.File(p, "r")
        g, s, sc = "GEOLOCATION_PARAMETERS/", "SCIENCE_PRODUCTS/", "SCENE_INFORMATION/"
        rec = dict(
            lat=_read(f, g + "LAT").astype("f8"), lon=_read(f, g + "LON").astype("f8"),
            time_utc_h=_read(f, g + "TIME_STAMP").astype("f4"),
            sza=_read(f, g + "SZA").astype("f4"),
            cloud=_read(f, sc + "CLOUDFLAG").astype("i2"),
            vcol_below=_read(f, s + "NO2_VCOLBELOWAIRCRAFT").astype("f8"),
            slcol=_read(f, s + "NO2_SLCOL").astype("f8"),
            slcol_err=_read(f, s + "NO2_SLCOL_ERR").astype("f8"),
        )
        np.savez_compressed(out, **rec)
        with open(p[:-3] + "_README.txt", "w", encoding="utf-8") as fh:
            fh.write(_text(_read(f, "METADATA/README")) + "\n\nFLIGHT INFO\n" +
                     _text(_read(f, "METADATA/FlightInfo")) + "\n\nREFERENCE\n" +
                     _text(_read(f, "REFERENCE_INFORMATION/REFERENCE")) + "\n")
        print(f"wrote {os.path.basename(out)}  ({len(rec['lat'])} pixels, {time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Airborne data on a lattice
# ---------------------------------------------------------------------------
def load_flight(path, column=1, window_h=1.5, overpass_utc=17.75):
    """Clear-sky airborne pixels near the Sentinel-5P overpass (NYC ≈ 17:00–18:30 UTC)."""
    z = np.load(path)
    v = z["vcol_below"]
    v = v[:, column] if v.ndim == 2 else v
    ok = (np.isfinite(v) & (v > -9998) & (z["cloud"] == 0) &
          (np.abs(z["time_utc_h"] - overpass_utc) <= window_h) & np.isfinite(z["lat"]) & np.isfinite(z["lon"]))
    day = os.path.basename(path).split("_")[2]          # ..._AIRCRAFT_20180719_R0_PM...
    return dict(day=f"{day[:4]}-{day[4:6]}-{day[6:8]}", lat=z["lat"][ok], lon=z["lon"][ok], v=v[ok],
                t=z["time_utc_h"][ok], n_total=int(len(v)))


def grid_points(lon, lat, val, bounds, shape):
    """Mean of point values in each cell of a lattice (bounds = [lon0, lat0, lon1, lat1])."""
    H, W = shape
    c = np.floor((lon - bounds[0]) / (bounds[2] - bounds[0]) * W).astype(int)
    r = np.floor((bounds[3] - lat) / (bounds[3] - bounds[1]) * H).astype(int)
    m = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    idx = r[m] * W + c[m]
    s = np.bincount(idx, weights=val[m], minlength=H * W)
    n = np.bincount(idx, minlength=H * W)
    out = np.full(H * W, np.nan)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out.reshape(H, W), n.reshape(H, W)


def tile_bbox(lon_c, lat_c, km):
    dlat = km / 2 / 110.7
    dlon = km / 2 / (111.32 * math.cos(math.radians(lat_c)))
    return [lon_c - dlon, lat_c - dlat, lon_c + dlon, lat_c + dlat]


def pick_tiles(fl, km=8.0, n_max=6, min_cover=0.6):
    """Non-overlapping km x km tiles with the best airborne coverage at 250 m."""
    lat_mid = float(np.median(fl["lat"]))
    step_lat = km / 110.7
    step_lon = km / (111.32 * math.cos(math.radians(lat_mid)))
    lon0, lat0 = fl["lon"].min(), fl["lat"].min()
    gi = np.floor((fl["lon"] - lon0) / step_lon).astype(int)
    gj = np.floor((fl["lat"] - lat0) / step_lat).astype(int)
    cands = []
    for i, j in set(zip(gi.tolist(), gj.tolist())):
        bb = [lon0 + i * step_lon, lat0 + j * step_lat, lon0 + (i + 1) * step_lon, lat0 + (j + 1) * step_lat]
        sel = (gi == i) & (gj == j)
        n_cells = int(round(km * 1000 / 250)) ** 2
        g, n = grid_points(fl["lon"][sel], fl["lat"][sel], fl["v"][sel], bb, (int(round(km * 4)),) * 2)
        cover = float(np.isfinite(g).sum()) / n_cells
        if cover >= min_cover:
            cands.append((cover, (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))
    cands.sort(reverse=True)
    return [tile_bbox(lo, la, km) for _, lo, la in cands[:n_max]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def block_mean_nan(a, f):
    H, W = a.shape
    b = a[:H // f * f, :W // f * f].reshape(H // f, f, W // f, f)
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3))


def pearson(x, y):
    x, y = np.asarray(x, "f8").ravel(), np.asarray(y, "f8").ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return math.nan
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else math.nan


def detail(a, f=4):
    """Sub-1-km detail: each 250 m cell minus the mean of its 1 km block."""
    bm = block_mean_nan(a, f)
    return a[:bm.shape[0] * f, :bm.shape[1] * f] - np.kron(bm, np.ones((f, f)))


def score(pred250, air250):
    m = np.isfinite(air250)
    p = np.where(m, pred250, np.nan)
    out = {"r_250m": pearson(p, air250), "r_detail_sub1km": pearson(detail(p), detail(air250))}
    pb, ab = block_mean_nan(p, 4), block_mean_nan(air250, 4)
    out["r_1km"] = pearson(pb, ab)
    sx, sy = np.nanstd(p), np.nanstd(air250)
    out["spread_ratio"] = float(sx / sy) if sy > 0 else math.nan       # contrast vs aircraft
    out["mean_ratio"] = float(np.nanmean(p) / np.nanmean(air250))      # S5P vs aircraft level
    return out


# ---------------------------------------------------------------------------
# Step 2: compare
# ---------------------------------------------------------------------------
def compare(a):
    sys.path.insert(0, os.path.join(ROOT, "backend"))
    from hyperlocal_aq import gee, service
    from hyperlocal_aq.pipeline import run as run_pipeline

    methods = a.methods.split(",")
    cnn = service.pretrained_cnn("NO2", 10) if "cnn" in methods else None
    if "cnn" in methods and cnn is None:
        print("backend/models/NO2_cnn.json not found; the CNN will be trained online per tile.")
    footprint = tuple(int(x) for x in a.footprint.split(","))
    files = sorted(glob.glob(os.path.join(DATA, "*.npz")))
    if not files:
        sys.exit("Run `extract` first.")
    tiles_out = []
    for path in files:
        fl = load_flight(path, column=a.column, window_h=a.window_h)
        tiles = pick_tiles(fl, km=a.tile_km, n_max=a.tiles, min_cover=a.min_cover)
        print(f"{fl['day']}: {len(fl['v'])}/{fl['n_total']} clear pixels near overpass, {len(tiles)} tiles")
        for k, bb in enumerate(tiles, 1):
            t0 = time.time()
            try:
                bands, coarse, lat, prov = gee.fetch_stack("NO2", fl["day"], bb, res_m=10,
                                                           n_days=a.n_days, window_days=a.window_days)
            except Exception as e:  # noqa: BLE001
                print(f"  tile {k}: no satellite data ({e})")
                continue
            if prov["target_day"] != fl["day"]:
                print(f"  tile {k}: no Sentinel-5P overpass on {fl['day']} (closest {prov['target_day']}), skipped")
                continue
            res = run_pipeline(bands, coarse, "NO2", methods=tuple(["bilinear"] + [m for m in methods if m != "bilinear"]),
                               factor=lat.factor, cnn_steps=a.cnn_steps, cnn_model=cnn, footprint=footprint)
            f250 = 25                                   # 10 m → 250 m
            H250, W250 = lat.height // f250, lat.width // f250
            air, n = grid_points(fl["lon"], fl["lat"], fl["v"], lat.bounds, (H250, W250))
            obs = res.coarse_obs * MOLEC_CM2_PER_MOL_M2
            preds = {"raw_1km": np.kron(obs, np.ones((4, 4)))[:H250, :W250]}
            for m, fine in res.maps.items():
                preds[m] = block_mean_nan(fine * MOLEC_CM2_PER_MOL_M2, f250)[:H250, :W250]
            rec = {"day": fl["day"], "bbox": [round(x, 5) for x in lat.bounds],
                   "airborne_cells_250m": int(np.isfinite(air).sum()),
                   "airborne_mean_molec_cm2": float(np.nanmean(air)),
                   "s2": prov["s2"], "s5p_days_used": len(prov["s5p_days"]),
                   "scores": {k2: score(v2, air) for k2, v2 in preds.items()}}
            tiles_out.append(rec)
            sc = rec["scores"]
            print(f"  tile {k}: {rec['airborne_cells_250m']} cells  " +
                  "  ".join(f"{m} r={sc[m]['r_250m']:.2f}/{sc[m]['r_detail_sub1km']:.2f}" for m in sc) +
                  f"  ({time.time() - t0:.0f}s)")
    if not tiles_out:
        sys.exit("No tiles could be compared.")
    out = {"truth": "NASA GCAS airborne NO2 column below aircraft (WRF-Chem a priori if column=1), "
                    "LISTOS 2018 (New York), gridded to 250 m", "column": a.column,
           "pollutant": "NO2", "footprint_cells": footprint, "tile_km": a.tile_km,
           "overpass_window_utc_h": [17.75 - a.window_h, 17.75 + a.window_h],
           "cnn": "pretrained on Mumbai (no New York or airborne training)" if cnn else "trained online per tile",
           "tiles": tiles_out}
    write_results(out)


RESOLVED_R = 0.3   # tiles where the raw 1 km satellite itself tracks the aircraft pattern


def sign_test_p(wins, n):
    """One-sided binomial sign test: P(X >= wins) for X ~ Bin(n, 0.5)."""
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n) if n else math.nan


def summarize(tiles):
    """Median scores per method, wins against interpolation and a sign test."""
    if not tiles:
        return {}
    out = {}
    for m in tiles[0]["scores"]:
        s = {"tiles": len(tiles)}
        for k in tiles[0]["scores"][m]:
            v = [t["scores"][m][k] for t in tiles
                 if t["scores"][m].get(k) is not None and np.isfinite(t["scores"][m][k])]
            s[k + "_median"] = float(np.median(v)) if v else None
        out[m] = s
    for m, s in out.items():
        if m in ("bilinear", "raw_1km"):
            continue
        for k in ("r_250m", "r_detail_sub1km", "r_1km"):
            pairs = [(t["scores"][m][k], t["scores"]["bilinear"][k]) for t in tiles
                     if t["scores"][m].get(k) is not None and t["scores"]["bilinear"].get(k) is not None
                     and np.isfinite(t["scores"][m][k]) and np.isfinite(t["scores"]["bilinear"][k])]
            wins = sum(x > y for x, y in pairs)
            s[k + "_wins_vs_bilinear"] = f"{wins}/{len(pairs)}"
            s[k + "_sign_test_p"] = sign_test_p(wins, len(pairs))
            b, mm = out["bilinear"].get(k + "_median"), s.get(k + "_median")
            s[k + "_gain_vs_bilinear"] = (mm - b) if (b is not None and mm is not None) else None
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_results(out):
    tiles = out["tiles"]
    resolved = [t for t in tiles if (t["scores"]["raw_1km"]["r_250m"] or -1) >= RESOLVED_R]
    out["summary"] = summarize(tiles)
    out["summary_resolved"] = summarize(resolved)
    out["resolved_definition"] = f"tiles where raw 1 km Sentinel-5P has r >= {RESOLVED_R} with the aircraft map"
    out["flights"] = sorted({t["day"] for t in tiles})
    out["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "airborne_NO2.json")
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=1)
    fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"  # noqa: E731
    for label, summ in (("ALL TILES", out["summary"]), ("TILES THE SATELLITE RESOLVES", out["summary_resolved"])):
        n = next(iter(summ.values()))["tiles"] if summ else 0
        print(f"\n{label} ({n}) — median r at 250 m / sub-1 km detail / 1 km; wins vs interpolation (250 m)")
        for m, s in summ.items():
            print(f"  {m:9s} {fmt(s['r_250m_median'])} {fmt(s['r_detail_sub1km_median'])} {fmt(s['r_1km_median'])}"
                  f"  {s.get('r_250m_wins_vs_bilinear', '')}")
    print("wrote", path)


def resummarize(_a):
    """Recompute the summary of validation/results/airborne_NO2.json (no Earth Engine)."""
    path = os.path.join(RESULTS, "airborne_NO2.json")
    with open(path) as f:
        out = json.load(f)
    write_results(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    sub.add_parser("summary", help="recompute the summary from saved tiles (no Earth Engine)")
    c = sub.add_parser("compare")
    c.add_argument("--methods", default="bilinear,lr,mslr,cnn")
    c.add_argument("--tiles", type=int, default=6, help="tiles per flight")
    c.add_argument("--tile-km", type=float, default=8.0)
    c.add_argument("--min-cover", type=float, default=0.6, help="airborne coverage of a tile at 250 m")
    c.add_argument("--column", type=int, default=1, help="VCOLBELOW a-priori: 0 = NAM-CMAQ 12 km, 1 = WRF-Chem 4 km (default; README notes NAM-CMAQ under-estimates free-troposphere NO2)")
    c.add_argument("--window-h", type=float, default=1.5, help="hours around the 17:45 UTC overpass")
    c.add_argument("--footprint", default=",".join(map(str, FOOTPRINT_2018)))
    c.add_argument("--n-days", type=int, default=20)
    c.add_argument("--window-days", type=int, default=45)
    c.add_argument("--cnn-steps", type=int, default=150)
    a = ap.parse_args()
    {"extract": extract, "compare": compare, "summary": resummarize}[a.cmd](a)


if __name__ == "__main__":
    main()
