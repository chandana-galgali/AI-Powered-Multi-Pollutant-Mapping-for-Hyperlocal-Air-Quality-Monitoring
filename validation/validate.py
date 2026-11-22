"""Validate the downscaling on real data.

Ground stations (CPCB / MPCB via OpenAQ, or a CSV export):
    export OPENAQ_API_KEY=...            # free: https://explore.openaq.org
    python validation/validate.py ground --pollutant NO2 --start 2025-11-01 --end 2026-02-28
    python validation/validate.py ground --pollutant NO2 --csv my_stations.csv --start ... --end ...

Up-scaling test (real Sentinel-5P only, no ground data needed):
    python validation/validate.py upscaling --pollutant all --end 2026-02-28

Both need Earth Engine credentials (same as the server). Results are written to
validation/results/*.json and shown by GET /api/validation and the dashboard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
RESULTS = os.path.join(HERE, "results")

from hyperlocal_aq import ground, service, validation  # noqa: E402
from hyperlocal_aq.config import MUMBAI_BBOX, POLLUTANTS  # noqa: E402

GROUND_POLLUTANTS = ["NO2", "SO2", "CO", "O3", "AER_AI"]   # HCHO has no routine ground monitoring


def refresh_docs():
    """Re-fill the auto-generated results tables in README.md and docs/VALIDATION.md."""
    try:
        import importlib.util
        path = os.path.join(HERE, "report.py")
        spec = importlib.util.spec_from_file_location("report", path)
        rep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rep)
        for doc in ("README.md", os.path.join("docs", "VALIDATION.md")):
            rep.fill(os.path.join(rep.ROOT, doc))
    except Exception as e:  # noqa: BLE001  (docs refresh must never break a validation run)
        print("README/VALIDATION tables not refreshed:", e)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    obj["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(validation.clean_json(obj), f, indent=1)
    print("wrote", path)
    refresh_docs()


def run_ground(a, pollutant):
    if pollutant not in ground.GROUND_PARAMETER:
        print(f"{pollutant}: no ground-station equivalent, skipped")
        return
    bbox = a.bbox or MUMBAI_BBOX
    if a.csv:
        stations = ground.load_csv(a.csv, pollutant, a.start, a.end)
        source = f"CSV {os.path.basename(a.csv)}"
    else:
        stations = ground.load_openaq(bbox, pollutant, a.start, a.end)
        source = "OpenAQ (CPCB/MPCB stations)"
    stations = [s for s in stations if s.lat and s.lon and
                bbox[0] <= s.lon <= bbox[2] and bbox[1] <= s.lat <= bbox[3]]
    print(f"{pollutant}: {len(stations)} stations with overpass-time data")
    if len(stations) < 3:
        print("Need at least 3 stations; skipped.")
        return
    window = (date.fromisoformat(a.end) - date.fromisoformat(a.start)).days + 1
    methods = a.methods.split(",")
    cnn = service.pretrained_cnn(pollutant, 10) if "cnn" in methods else None
    records = validation.collocate(stations, pollutant, a.end, methods, n_days=window, window_days=window,
                                   cnn_model=cnn, cnn_steps=a.cnn_steps)
    if not records:
        print("No station/satellite matches.")
        return
    preds = ["coarse", "bilinear"] + [m for m in methods if m != "bilinear"]
    res = validation.ground_metrics(records, pollutant, preds)
    res.update(pollutant=pollutant, source=source, period=[a.start, a.end],
               stations=[{"id": s.id, "name": s.name, "lat": s.lat, "lon": s.lon} for s in stations],
               cnn=("pretrained" if cnn else "trained per station tile") if "cnn" in methods else None,
               overpass_window_ist="13:00-15:00")
    save(f"ground_{pollutant}.json", res)
    with open(os.path.join(RESULTS, f"ground_{pollutant}_pairs.json"), "w") as f:
        json.dump(validation.clean_json(records), f)
    for k, v in res["predictors"].items():
        print(f"  {k:9s} spatial r {v['daily_spatial_r_median']:.2f}  pooled r {v['pooled_r']:.2f}  "
              f"RMSE {v.get('rmse_ug_m3', float('nan')):.1f} µg/m³  skill {v.get('skill_vs_raw_1km', float('nan')):.2f}")


def run_upscaling(a, pollutant):
    bbox = a.bbox or MUMBAI_BBOX
    bands_1km, obs, prov = validation.fetch_upscaling_inputs(pollutant, a.end, bbox, n_days=a.days)
    res = validation.upscaling_test(bands_1km, obs, a.methods.split(","), k=a.k, cnn_steps=a.cnn_steps)
    res.update(pollutant=pollutant, bbox=bbox, s5p_days=prov["s5p_days"], s2=prov["s2"])
    save(f"upscaling_{pollutant}.json", res)
    for k, v in res["methods"].items():
        print(f"  {k:9s} r {v['spatial_r_median']:.2f}  detail r {v['detail_r_median']:.2f}  "
              f"skill vs bilinear {v.get('skill_vs_bilinear', 0):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ground", "upscaling"])
    ap.add_argument("--pollutant", default="NO2", help="NO2, SO2, CO, HCHO, O3, AER_AI or all")
    ap.add_argument("--start", help="first day (ground)")
    ap.add_argument("--end", required=True, help="last day (YYYY-MM-DD)")
    ap.add_argument("--csv", help="station CSV instead of OpenAQ")
    ap.add_argument("--bbox", type=float, nargs=4, help="lon0 lat0 lon1 lat1 (default: Mumbai)")
    ap.add_argument("--methods", default="lr,mslr,cnn")
    ap.add_argument("--cnn-steps", type=int, default=150)
    ap.add_argument("--days", type=int, default=30, help="overpasses for the up-scaling test")
    ap.add_argument("--k", type=int, default=4, help="coarsening factor for the up-scaling test")
    a = ap.parse_args()
    if a.kind == "ground" and not a.start:
        ap.error("ground validation needs --start")
    pols = (GROUND_POLLUTANTS if a.kind == "ground" else list(POLLUTANTS)) if a.pollutant == "all" \
        else [a.pollutant.upper()]
    for p in pols:
        (run_ground if a.kind == "ground" else run_upscaling)(a, p)


if __name__ == "__main__":
    main()
