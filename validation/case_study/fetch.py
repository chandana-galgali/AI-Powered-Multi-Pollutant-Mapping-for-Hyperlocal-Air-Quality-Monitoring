"""Fetch NO2 maps for the Chembur-Trombay-Mahul case study from the live system.

Runs real jobs on the deployed server (no Earth Engine credentials needed locally) for
several dates in the held-out 2025-26 dry season, and saves each 10 m map plus its
1 km input to validation/case_study/data/.

    python validation/case_study/fetch.py
    python validation/case_study/fetch.py --server http://localhost:5000   # local backend instead
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
DATES = ["2025-11-20", "2025-12-05", "2025-12-20", "2026-01-05", "2026-01-20", "2026-02-05", "2026-02-20"]


def call(url, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_for_server(base):
    print("Waking the server (free hosting can take ~1 minute)...")
    for _ in range(30):
        try:
            code, raw = call(base + "/api/health", timeout=90)
            if code == 200:
                h = json.loads(raw)
                if not h.get("earth_engine", {}).get("connected"):
                    sys.exit(f"Server is up but not connected to Earth Engine: {h}")
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    sys.exit("Server did not respond.")


def run_job(base, body):
    code, raw = call(base + "/api/jobs", body)
    if code not in (200, 202):
        raise RuntimeError(f"submit failed ({code}): {raw[:300]}")
    job = json.loads(raw)
    t0 = time.time()
    while job.get("state") not in ("done", "error"):
        if time.time() - t0 > 900:
            raise RuntimeError("timed out")
        time.sleep(5)
        code, raw = call(f"{base}/api/jobs/{job['id']}")
        if code != 200:
            raise RuntimeError(f"poll failed ({code}): {raw[:300]}")
        job = json.loads(raw)
        print(f"   {job.get('stage', '')} {int(100 * job.get('progress', 0))}%", end="\r")
    print()
    if job["state"] == "error":
        raise RuntimeError(job.get("error"))
    return job


def read_tiff(data):
    import tifffile
    return tifffile.imread(io.BytesIO(data)).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="https://hyperlocal-air-quality.onrender.com")
    ap.add_argument("--pollutant", default="NO2")
    ap.add_argument("--method", default="cnn")
    ap.add_argument("--area", default="chembur_trombay")
    a = ap.parse_args()
    base = a.server.rstrip("/")
    os.makedirs(OUT, exist_ok=True)
    wait_for_server(base)
    seen = set()
    for d in DATES:
        print(f"{d}: running {a.pollutant} / {a.method} on {a.area}")
        try:
            job = run_job(base, {"pollutant": a.pollutant, "method": a.method, "date": d, "area": a.area})
        except Exception as e:  # noqa: BLE001
            print(f"   skipped: {e}")
            continue
        res = job["result"]
        day = res["provenance"]["target_day"]
        if day in seen:
            print(f"   same overpass as before ({day}), skipped")
            continue
        seen.add(day)
        code, tif = call(f"{base}/api/jobs/{job['id']}/geotiff", timeout=300)
        if code != 200:
            print(f"   GeoTIFF download failed ({code}), skipped")
            continue
        fine = read_tiff(tif)
        meta = {k: v for k, v in res.items() if k not in ("overlay", "coarse_overlay")}
        name = f"{a.pollutant}_{a.method}_{a.area}_{day}"
        np.savez_compressed(os.path.join(OUT, name + ".npz"), fine=fine,
                            coarse=np.array([[np.nan if v is None else v for v in row] for row in res["coarse_grid"]],
                                            dtype="float32"))
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"   saved {name} ({fine.shape[1]}x{fine.shape[0]} px at {res['res_m']} m, "
              f"{res['stats']['overpasses_used']} overpasses, {res.get('runtime_s')} s)")
    print(f"Done: {len(seen)} maps in {OUT}")


if __name__ == "__main__":
    main()
