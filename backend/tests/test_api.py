"""API tests with Earth Engine replaced by a fake that returns realistic arrays."""
import io
import time

import numpy as np
import pytest

import app as app_module
from hyperlocal_aq import gee, service
from hyperlocal_aq.scene import build_scene


@pytest.fixture(scope="module")
def client():
    scene = build_scene(seed=4, days=5, size_m=5000, pollutants=["NO2"])

    def fake_fetch(pollutant, end_day, bbox, res_m=10, progress=None, **kw):
        lat = gee.Lattice(bbox, res_m=res_m)
        H, W = lat.height, lat.width
        reps = (int(np.ceil(H / 500)), int(np.ceil(W / 500)))
        bands = np.tile(scene.bands, (1, *reps))[:, :H, :W]
        coarse = np.tile(scene.coarse["NO2"], (1, *reps))[:, :lat.ch, :lat.cw]
        if progress:
            progress("fake download", 0.5)
        return bands, coarse, lat, {"s2": "fake", "s5p_days": ["2025-01-01"] * coarse.shape[0],
                                    "target_day": "2025-01-05", "s5p_product": "fake"}

    gee.fetch_stack = fake_fetch
    gee.status = lambda: {"connected": True}
    app_module.gee.status = gee.status
    return app_module.app.test_client()


def wait(client, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise TimeoutError


def test_catalogue(client):
    assert set(client.get("/api/pollutants").json) == {"NO2", "SO2", "CO", "HCHO", "O3", "AER_AI"}
    assert "cnn" in client.get("/api/methods").json
    assert "mumbai" in client.get("/api/areas").json["areas"]


@pytest.mark.parametrize("method", ["bilinear", "lr", "mslr"])
def test_job_named_area(client, method):
    r = client.post("/api/jobs", json={"pollutant": "NO2", "method": method, "area": "somaiya",
                                       "date": "2025-01-05"})
    assert r.status_code == 202, r.json
    j = wait(client, r.json["id"])
    assert j["state"] == "done", j.get("error")
    res = j["result"]
    assert res["res_m"] == 10 and res["overlay"].startswith("data:image/png")
    if method != "bilinear":   # map, seen through the S5P footprint, stays close to the observation
        assert res["stats"]["satellite_consistency_rmse"] < 0.25 * res["stats"]["mean"]
    tif = client.get(f"/api/jobs/{j['id']}/geotiff")
    assert tif.status_code == 200 and tif.data[:2] in (b"II", b"MM")
    import tifffile
    arr = tifffile.imread(io.BytesIO(tif.data))
    assert arr.shape == (res["size"][1], res["size"][0])


def test_city_wide_uses_coarser_resolution(client):
    r = client.post("/api/jobs", json={"pollutant": "NO2", "method": "lr", "area": "mumbai", "date": "2025-01-05"})
    assert r.status_code == 202
    assert r.json["params"]["res_m"] == 25


def test_cnn_online_training(client, monkeypatch):
    monkeypatch.setattr(service, "ONLINE_CNN_STEPS", 5)
    r = client.post("/api/jobs", json={"pollutant": "NO2", "method": "cnn", "area": "somaiya", "date": "2025-01-04"})
    j = wait(client, r.json["id"])
    assert j["state"] == "done", j.get("error")
    assert j["result"]["provenance"]["cnn"] == "trained on this request"


def test_validation_errors(client):
    assert client.post("/api/jobs", json={"pollutant": "XYZ"}).status_code == 400
    assert client.post("/api/jobs", json={"bbox": [72.0, 18.0, 73.5, 19.5], "method": "lr"}).status_code == 400
    assert client.post("/api/jobs", json={"area": "mumbai", "method": "cnn"}).status_code == 400
    assert client.post("/api/jobs", json={"area": "somaiya", "date": "2010-01-01"}).status_code == 400


def test_validation_metrics(client):
    v = client.get("/api/validation").json
    assert "cnn" in v["simulation"]["NO2"]
    assert set(v) == {"simulation", "ground", "upscaling"}
