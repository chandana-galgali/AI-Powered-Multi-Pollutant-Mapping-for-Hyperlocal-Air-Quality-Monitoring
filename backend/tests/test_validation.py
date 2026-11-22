"""Tests for ground loading, collocation, metrics and the up-scaling test (Earth Engine mocked)."""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from hyperlocal_aq import gee, ground, validation
from hyperlocal_aq.core import block_mean
from hyperlocal_aq.scene import build_scene


# ---------------------------------------------------------------- units and loading
def test_unit_conversion():
    assert ground.to_ug_m3(10, "µg/m³", "no2") == 10
    assert math.isclose(ground.to_ug_m3(10, "ppb", "no2"), 10 * 46.0055 / 24.45)
    assert math.isclose(ground.to_ug_m3(1, "ppm", "co"), 1000 * 28.010 / 24.45)
    assert ground.to_ug_m3(1.2, "mg/m3", "co") == 1200
    assert ground.to_ug_m3(5, "furlongs", "no2") is None


def test_overpass_window_is_ist():
    utc = timezone.utc
    rows = [(datetime(2025, 1, 10, 7, 30, tzinfo=utc), 40.0),   # 13:00 IST -> kept
            (datetime(2025, 1, 10, 8, 30, tzinfo=utc), 60.0),   # 14:00 IST -> kept
            (datetime(2025, 1, 10, 2, 30, tzinfo=utc), 999.0)]  # 08:00 IST -> dropped
    assert ground.overpass_means(rows) == {"2025-01-10": 50.0}


def test_load_csv(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("station_id,station_name,lat,lon,datetime,parameter,value,unit\n"
                 "A,Chembur,19.036,72.896,2025-01-10T13:00:00+05:30,no2,40,µg/m³\n"
                 "A,Chembur,19.036,72.896,2025-01-10T14:00:00+05:30,NO2,20,ppb\n"
                 "A,Chembur,19.036,72.896,2025-01-10T13:00:00+05:30,pm25,99,µg/m³\n", encoding="utf-8")
    st = ground.load_csv(str(p), "NO2")
    assert len(st) == 1 and st[0].name == "Chembur"
    assert math.isclose(st[0].daily["2025-01-10"], (40 + 20 * 46.0055 / 24.45) / 2)


def test_openaq_parsing(monkeypatch):
    def fake_get(self, path, **params):
        if path == "/locations":
            return {"results": [{"id": 7, "name": "Sion - MPCB", "isMobile": False,
                                 "coordinates": {"latitude": 19.047, "longitude": 72.865},
                                 "provider": {"name": "CPCB"},
                                 "sensors": [{"id": 70, "parameter": {"name": "no2", "units": "µg/m³"}},
                                             {"id": 71, "parameter": {"name": "pm25", "units": "µg/m³"}}]}]}
        return {"results": [{"value": 30.0, "parameter": {"units": "µg/m³"},
                             "period": {"datetimeFrom": {"utc": "2025-01-10T07:30:00Z"}}},
                            {"value": 50.0, "parameter": {"units": "µg/m³"},
                             "period": {"datetimeFrom": {"utc": "2025-01-10T08:30:00Z"}}}]}
    monkeypatch.setattr(ground.OpenAQ, "get", fake_get)
    st = ground.load_openaq([72.7, 18.8, 73.0, 19.3], "NO2", "2025-01-01", "2025-01-31", api_key="x", log=lambda *_: None)
    assert len(st) == 1 and st[0].daily == {"2025-01-10": 40.0}


# ---------------------------------------------------------------- statistics
def test_stats_helpers():
    x = np.arange(10.0)
    assert math.isclose(validation.pearson(x, 2 * x + 1), 1.0)
    assert math.isclose(validation.spearman(x, x ** 3), 1.0)
    assert math.isnan(validation.pearson([1, 2], [1, 2]))


def test_ground_metrics_rewards_better_predictor():
    rng = np.random.default_rng(0)
    recs = []
    for d in range(20):
        for s in range(8):
            g = 30 + 10 * s + rng.normal(0, 3)
            recs.append({"station": f"s{s}", "day": f"2025-01-{d + 1:02d}", "ground": g,
                         "coarse": (35 + 5 * (s // 4) * 10) * 1e-6 + rng.normal(0, 4e-6),   # blocky, noisy
                         "mslr": g * 1e-6 + rng.normal(0, 1e-6)})                            # close to truth
    m = validation.ground_metrics(recs, "NO2", ["coarse", "mslr"])
    c, d = m["predictors"]["coarse"], m["predictors"]["mslr"]
    assert m["n_stations"] == 8 and m["n_days"] == 20
    assert d["daily_spatial_r_median"] > c["daily_spatial_r_median"]
    assert d["rmse_ug_m3"] < c["rmse_ug_m3"] and d["skill_vs_raw_1km"] > 0.3
    assert 0 <= d["aqi_category_accuracy"] <= 1


# ---------------------------------------------------------------- collocation with mocked Earth Engine
@pytest.fixture(scope="module")
def scene():
    return build_scene(seed=5, days=6, size_m=6000, pollutants=["NO2"])


def test_collocate_end_to_end(scene, monkeypatch):
    days = [(datetime(2025, 1, 1) + timedelta(days=i)).date().isoformat() for i in range(6)]
    g = scene.grid
    b = g.bounds

    def fake_fetch(pollutant, end_day, bbox, res_m=10, n_days=20, window_days=60, progress=None):
        lat = gee.Lattice(b, res_m=10)
        lat.width, lat.height, lat.cw, lat.ch = g.width, g.height, g.width // 100, g.height // 100
        lat.bounds = b
        return scene.bands, scene.coarse["NO2"], lat, {"s5p_days": days, "target_day": days[-1], "s2": "fake"}

    monkeypatch.setattr(gee, "fetch_stack", fake_fetch)
    rng = np.random.default_rng(1)
    stations = []
    for k in range(6):
        r, c = rng.integers(50, 550, 2)
        lon = b[0] + (c + 0.5) / g.width * (b[2] - b[0])
        lat_ = b[3] - (r + 0.5) / g.height * (b[3] - b[1])
        daily = {d: float(scene.truth["NO2"][i, r, c] * 4e5) for i, d in enumerate(days)}   # mol/m² -> "µg/m³"
        stations.append(ground.Station(f"s{k}", f"S{k}", lat_, lon, daily=daily))
    recs = validation.collocate(stations, "NO2", days[-1], ["lr", "mslr"], log=lambda *_: None)
    assert recs and all("mslr" in r and "coarse" in r for r in recs)
    m = validation.ground_metrics(recs, "NO2", ["coarse", "bilinear", "lr", "mslr"])
    assert set(m["predictors"]) >= {"coarse", "mslr"}
    assert m["predictors"]["mslr"]["pooled_r"] > 0


def test_upscaling_test(scene):
    bands_1km = block_mean(scene.bands, 100)
    obs = block_mean(scene.truth["NO2"], 10)              # pretend: 100 m "observations"
    obs = obs[:, :48, :48].copy()
    bands = block_mean(scene.bands, 10)[:, :48, :48]
    obs[0, :5, :5] = np.nan
    res = validation.upscaling_test(bands, obs, ["lr", "mslr"], k=4)
    assert res["days"] >= 5 and {"bilinear", "lr", "mslr"} <= set(res["methods"])
    for v in res["methods"].values():
        assert -1 <= v["spatial_r_median"] <= 1 and v["rmse"] > 0
    assert res["methods"]["mslr"]["detail_r_median"] > 0


def test_clean_json():
    import json
    out = validation.clean_json({"a": float("nan"), "b": [np.float32(1.5), float("inf")], "c": np.int64(3)})
    assert json.loads(json.dumps(out, allow_nan=False)) == {"a": None, "b": [1.5, None], "c": 3}


def test_multi_season_training(scene, tmp_path, monkeypatch):
    import train
    from hyperlocal_aq.models import FCNModel
    calls = []

    def fake_fetch(pollutant, end_day, bbox, res_m=10, n_days=20, window_days=60, s2_window_days=120, progress=None):
        calls.append((end_day, window_days))
        lat = gee.Lattice(bbox, res_m=10)
        return scene.bands, scene.coarse["NO2"], lat, {"s5p_days": [], "target_day": end_day, "s2": "fake"}

    monkeypatch.setattr(gee, "fetch_stack", fake_fetch)
    monkeypatch.setattr(train, "CACHE_DIR", str(tmp_path / "cache"))
    windows = [(s, *train.season_window(s)) for s in ["2022-23", "2023-24"]]
    train.train("NO2", windows, steps=3, n_days=120, out_dir=str(tmp_path))
    m = FCNModel.load(tmp_path / "NO2_cnn.json")
    assert m.meta["periods"] == ["2022-23", "2023-24"] and m.meta["trained_until"] == "2024-02-29"
    assert ("2023-02-28", 120) in calls and ("2024-02-29", 121) in calls
    assert m.feat_mean is not None and m.meta["hybrid"]
