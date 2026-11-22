"""Request handling: validation, background jobs, caching, result packaging."""
from __future__ import annotations

import glob
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from . import gee, render
from .config import MUMBAI_BBOX, POLLUTANTS, S5P_FOOTPRINT_CELLS
from .export import geotiff_bytes
from .models import FCNModel
from .pipeline import METHOD_LABELS, run

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 1_600_000))      # memory guard (≈1.6 M output pixels)
RESOLUTIONS = (10, 20, 25, 40)
ONLINE_CNN_STEPS = int(os.environ.get("ONLINE_CNN_STEPS", 120))

# Named areas (lon0, lat0, lon1, lat1)
AREAS = {
    "chembur_trombay": {"label": "Chembur · Trombay · Mahul (refineries)", "bbox": [72.875, 18.985, 72.945, 19.065]},
    "somaiya":         {"label": "Vidyavihar · Somaiya campus and surroundings", "bbox": [72.880, 19.055, 72.920, 19.090]},
    "bkc_dharavi":     {"label": "BKC · Dharavi · Sion · Kurla", "bbox": [72.840, 19.025, 72.895, 19.085]},
    "andheri_airport": {"label": "Andheri · Airport · Powai", "bbox": [72.830, 19.080, 72.920, 19.140]},
    "south_mumbai":    {"label": "South Mumbai (Colaba to Worli)", "bbox": [72.805, 18.895, 72.860, 19.020]},
    "western_suburbs": {"label": "Bandra to Goregaon", "bbox": [72.815, 19.040, 72.870, 19.170]},
    "mumbai":          {"label": "All of Mumbai (coarser grid, 25–40 m)", "bbox": MUMBAI_BBOX},
}


def area_km2(bbox):
    lon0, lat0, lon1, lat1 = bbox
    return (lon1 - lon0) * 111.32 * np.cos(np.radians((lat0 + lat1) / 2)) * (lat1 - lat0) * 110.7


def max_area_km2(res_m):
    return MAX_PIXELS * (res_m / 1000) ** 2


def pick_resolution(bbox, requested=None):
    """Finest allowed resolution for the area (or the requested one if it fits)."""
    a = area_km2(bbox)
    options = [requested] if requested else list(RESOLUTIONS)
    for r in options:
        if a <= max_area_km2(r):
            return r
    raise ValueError(f"Area is {a:.0f} km²; the limit is {max_area_km2(max(RESOLUTIONS)):.0f} km² at "
                     f"{max(RESOLUTIONS)} m. Zoom in or pick a named area.")


# ----------------------------------------------------------------------------
# Pretrained CNN weights (backend/models/<POLLUTANT>_cnn.json)
# ----------------------------------------------------------------------------
_cnn_cache: dict = {}


def pretrained_cnn(pollutant: str, res_m: int):
    path = os.path.join(MODELS_DIR, f"{pollutant}_cnn.json")
    if not os.path.exists(path):
        return None
    if path not in _cnn_cache:
        _cnn_cache[path] = FCNModel.load(path)
    m = _cnn_cache[path]
    return m if (m.res_m == res_m and m.meta.get("hybrid")) else None


def available_methods():
    have = {os.path.basename(p).split("_cnn.json")[0] for p in glob.glob(os.path.join(MODELS_DIR, "*_cnn.json"))}
    return {k: {"label": v, **({"pretrained_for": sorted(have)} if k == "cnn" else {})}
            for k, v in METHOD_LABELS.items()}


# ----------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------
class Jobs:
    def __init__(self, workers=int(os.environ.get("JOB_WORKERS", 1)), keep=40):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: "OrderedDict[str, dict]" = OrderedDict()
        self.cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self.keep = keep
        self.lock = threading.Lock()

    def submit(self, params: dict) -> dict:
        key = cache_key(params)
        with self.lock:
            if key in self.cache:
                jid = self.cache[key]["job_id"]
                if jid in self.jobs:
                    return self.public(jid)
            jid = uuid.uuid4().hex[:12]
            self.jobs[jid] = {"id": jid, "state": "queued", "stage": "Queued", "progress": 0.0,
                              "params": params, "created": time.time()}
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)
        self.pool.submit(self._run, jid, key)
        return self.public(jid)

    def _run(self, jid, key):
        job = self.jobs.get(jid)
        if job is None:
            return

        def progress(stage, frac):
            job.update(stage=stage, progress=round(float(frac), 3))

        job["state"] = "running"
        t0 = time.time()
        try:
            result, files = execute(job["params"], progress)
            result["runtime_s"] = round(time.time() - t0, 1)
            job.update(state="done", stage="Done", progress=1.0, result=result, files=files)
            with self.lock:
                self.cache[key] = {"job_id": jid}
                while len(self.cache) > self.keep:
                    self.cache.popitem(last=False)
        except Exception as e:
            job.update(state="error", error=str(e), stage="Failed")

    def public(self, jid):
        j = self.jobs.get(jid)
        if j is None:
            return None
        out = {k: j[k] for k in ("id", "state", "stage", "progress", "params") if k in j}
        if j["state"] == "done":
            out["result"] = j["result"]
        if j["state"] == "error":
            out["error"] = j["error"]
        return out

    def file(self, jid, name):
        j = self.jobs.get(jid)
        return None if j is None or j.get("state") != "done" else j["files"].get(name)


def cache_key(p):
    return (p["pollutant"], p["method"], p["date"], tuple(round(v, 4) for v in p["bbox"]), p["res_m"])


def validate(body: dict) -> dict:
    pollutant = str(body.get("pollutant", "NO2")).upper()
    if pollutant not in POLLUTANTS:
        raise ValueError(f"Unknown pollutant '{pollutant}'.")
    method = body.get("method", "mslr")
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method '{method}'.")
    area = body.get("area")
    if area and area != "custom":
        if area not in AREAS:
            raise ValueError(f"Unknown area '{area}'.")
        bbox = AREAS[area]["bbox"]
        res = body.get("res_m") or AREAS[area].get("res_m")
    else:
        bbox = [float(v) for v in body.get("bbox", [])]
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("bbox must be [west, south, east, north].")
        res = body.get("res_m")
    res = pick_resolution(bbox, int(res) if res else None)
    if method == "cnn" and res != 10:
        raise ValueError("The CNN works at 10 m. Pick a smaller area (≤ "
                         f"{max_area_km2(10):.0f} km²) or another method.")
    if method == "rf" and area_km2(bbox) / (res / 1000) ** 2 > 700_000:
        raise ValueError("Random forest is limited to ~70 km² at 10 m on this server. Use a smaller area.")
    d = body.get("date") or (date.today() - timedelta(days=5)).isoformat()
    try:
        dd = date.fromisoformat(d)
    except ValueError:
        raise ValueError("date must be YYYY-MM-DD.")
    start = date.fromisoformat(POLLUTANTS[pollutant]["start_date"])
    if dd < start or dd > date.today():
        raise ValueError(f"{pollutant} data exists from {start} to today.")
    return {"pollutant": pollutant, "method": method, "date": d, "bbox": bbox, "res_m": res,
            "area": area or "custom"}


def execute(p: dict, progress):
    bands, coarse, lat, prov = gee.fetch_stack(p["pollutant"], p["date"], p["bbox"], res_m=p["res_m"],
                                               progress=progress)
    progress(f"Running {METHOD_LABELS[p['method']]}", 0.75)
    cnn = pretrained_cnn(p["pollutant"], p["res_m"]) if p["method"] == "cnn" else None
    res = run(bands, coarse, p["pollutant"], methods=(p["method"],), factor=lat.factor,
              cnn_steps=ONLINE_CNN_STEPS, cnn_model=cnn, footprint=footprint_for(lat))
    fine = res.maps[p["method"]]
    progress("Rendering map", 0.92)
    return package(p, fine, res.coarse_obs, res.coarse_filled, coarse.shape[0], lat, prov, cnn is not None)


def footprint_for(lat):
    """S5P footprint in units of the coarse (1 km) cells of this lattice."""
    return S5P_FOOTPRINT_CELLS


def package(p, fine, obs, filled, n_days, lat, prov, pretrained):
    from .core import block_mean
    from .models import apply_footprint
    vmin, vmax = float(np.percentile(fine, 1)), float(np.percentile(fine, 99.5))
    ok = ~np.isnan(obs)
    seen = apply_footprint(block_mean(fine, lat.factor), footprint_for(lat))   # what the satellite would see
    consistency = float(np.sqrt(np.mean((seen[ok] - obs[ok]) ** 2))) if ok.any() else None
    unit = POLLUTANTS[p["pollutant"]]["unit"]
    result = {
        "params": p, "unit": unit, "label": POLLUTANTS[p["pollutant"]]["label"],
        "method_label": METHOD_LABELS[p["method"]], "bounds": lat.bounds, "res_m": lat.res_m,
        "size": [lat.width, lat.height],
        "overlay": render.png_b64(render.colorize(fine, vmin, vmax), max_side=1600),
        "coarse_overlay": render.png_b64(render.coarse_rgba(obs, vmin, vmax)),
        "legend": {"min": vmin, "max": vmax, "palette": render.PALETTE.astype(int).tolist()},
        "stats": {"mean": float(fine.mean()), "p95": float(np.percentile(fine, 95)), "max": float(fine.max()),
                  "cloud_fraction": float(np.isnan(obs).mean()), "overpasses_used": int(n_days),
                  "satellite_consistency_rmse": consistency},
        "hotspots": render.hotspots(fine, lat.bounds, min_sep_px=max(10, 400 // lat.res_m)),
        "footprint_cells": list(footprint_for(lat)),
        "coarse_grid": [[None if np.isnan(v) else float(v) for v in row] for row in obs],
        "provenance": {**prov, "cnn": ("pretrained weights" if pretrained else "trained on this request")
                       if p["method"] == "cnn" else None},
    }
    desc = f"{p['pollutant']} {unit} downscaled to {lat.res_m} m, method={p['method']}, date={prov['target_day']}"
    files = {
        "geotiff": geotiff_bytes(fine, lat.bounds, description=desc),
        "coarse_geotiff": geotiff_bytes(obs, lat.bounds, nodata=float("nan"), description=desc + " (input 1 km)"),
    }
    return result, files
