"""Airborne (NASA LISTOS) comparison helpers: gridding, tiles, scores, summary."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "airborne", os.path.join(HERE, "..", "..", "validation", "airborne", "airborne.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def _flight(seed=0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-74.1, -73.8, 60000)
    lat = rng.uniform(40.6, 40.9, 60000)
    return dict(lon=lon, lat=lat, v=1e16 * (1 + np.sin(lon * 80) + np.cos(lat * 60)))


def test_grid_and_tiles():
    fl = _flight()
    tiles = A.pick_tiles(fl, km=8, n_max=4, min_cover=0.5)
    assert 1 <= len(tiles) <= 4
    g, n = A.grid_points(fl["lon"], fl["lat"], fl["v"], tiles[0], (32, 32))
    assert np.isfinite(g).mean() > 0.9 and n.sum() > 0


def test_scores_reward_detail():
    fl = _flight()
    bb = A.pick_tiles(fl, km=8, n_max=1, min_cover=0.5)[0]
    air, _ = A.grid_points(fl["lon"], fl["lat"], fl["v"], bb, (32, 32))
    blocky = np.kron(A.block_mean_nan(air, 4), np.ones((4, 4)))
    good = A.score(air + np.random.default_rng(1).normal(0, 1e14, air.shape), air)
    flat = A.score(blocky, air)
    assert good["r_250m"] > flat["r_250m"] and good["r_detail_sub1km"] > 0.9
    assert np.isnan(flat["r_detail_sub1km"]) or flat["r_detail_sub1km"] < 0.1


def test_summary_and_sign_test():
    def t(r_cnn, r_bil):
        sc = {"r_250m": 0.3, "r_detail_sub1km": None, "r_1km": 0.4}
        return {"scores": {"raw_1km": dict(sc), "bilinear": {**sc, "r_250m": r_bil, "r_detail_sub1km": 0.0},
                           "cnn": {**sc, "r_250m": r_cnn, "r_detail_sub1km": 0.1}}}
    s = A.summarize([t(0.6, 0.5), t(0.4, 0.5), t(0.7, 0.5)])
    assert s["cnn"]["r_250m_wins_vs_bilinear"] == "2/3"
    assert abs(A.sign_test_p(2, 3) - 0.5) < 1e-12
    assert s["cnn"]["r_250m_gain_vs_bilinear"] > 0
