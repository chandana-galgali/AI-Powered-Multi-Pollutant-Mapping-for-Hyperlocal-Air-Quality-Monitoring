"""Accuracy check of every downscaling method against a known 10 m field.

Real satellites provide no 10 m ground truth, so skill can't be scored on live
data. This script does it the standard way, with an observing-system
simulation: a 14 x 14 km stretch of east-central Mumbai with a known 10 m
pollution field. It is observed the way Sentinel-5P would see it (1 km means,
3 % noise, ~12 % cloud gaps, 8 overpasses). Every method sees only those 1 km
observations and is scored against the 10 m field.

    cd backend && python ../validation/run_validation.py --steps 700
Writes validation/simulation/metrics.json (served by GET /api/validation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))


def work(args):
    pollutant, steps = args
    import numpy as np
    from hyperlocal_aq import metrics, pipeline
    from hyperlocal_aq.scene import CLS, build_scene
    s = build_scene(seed=7, days=8, pollutants=[pollutant])
    s.bands = np.round(s.bands / 0.002) * 0.002          # 8-bit reflectance, as in the published results
    land = s.landcover != CLS["water"]
    res = pipeline.run(s.bands, s.coarse[pollutant], pollutant, methods=("bilinear", "lr", "mslr", "cnn"),
                       cnn_steps=steps)
    truth = s.truth[pollutant][-1]
    return pollutant, {m: metrics.evaluate(mp, truth, land, coarse_obs=res.coarse_obs)
                       for m, mp in res.maps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = {"_about": ("Simulated 14x14 km east-central Mumbai scene with a known 10 m field; "
                      "methods see only 1 km Sentinel-5P-like observations (8 overpasses, 3% noise, "
                      "~12% cloud gaps). hotspot_class_accuracy = share of land pixels in the right class "
                      "(low <50th pct, moderate, high >80th, hotspot >95th).")}
    with ProcessPoolExecutor(a.workers) as ex:
        for p, m in ex.map(work, [(p, a.steps) for p in ["NO2", "SO2", "CO", "HCHO", "O3", "AER_AI"]]):
            out[p] = m
            print(p, {k: round(v["hotspot_class_accuracy"], 3) for k, v in m.items()}, flush=True)
    os.makedirs(os.path.join(HERE, "simulation"), exist_ok=True)
    with open(os.path.join(HERE, "simulation", "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(HERE, "simulation", "metrics.json"))


if __name__ == "__main__":
    main()
