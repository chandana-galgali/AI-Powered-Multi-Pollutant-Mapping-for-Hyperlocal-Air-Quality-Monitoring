"""Fast tests on a small crop of the simulated scene (run: pytest -q)."""
import numpy as np
import pytest

from hyperlocal_aq import metrics, pipeline
from hyperlocal_aq.core import block_mean, gap_fill, residual_correct, upsample_bilinear
from hyperlocal_aq.models import FCNModel
from hyperlocal_aq.scene import CLS, build_scene
from hyperlocal_aq.timeseries import seasonal_fill


@pytest.fixture(scope="module")
def scene():
    return build_scene(seed=3, days=4, size_m=5000, pollutants=["NO2"])


def test_scene_shapes(scene):
    assert scene.bands.shape == (4, 500, 500)
    assert scene.truth["NO2"].shape == (4, 500, 500)
    assert scene.coarse["NO2"].shape == (4, 5, 5)
    assert 0 < np.nanmean(scene.truth["NO2"]) < 1e-3


def test_gap_fill_and_bilinear():
    c = np.arange(16, dtype=np.float32).reshape(4, 4)
    c[1, 2] = np.nan
    f = gap_fill(c)
    assert not np.isnan(f).any()
    up = upsample_bilinear(f, 10)
    assert up.shape == (40, 40)


def test_residual_correction_preserves_coarse_means():
    rng = np.random.default_rng(0)
    coarse = rng.random((3, 3)).astype(np.float32)
    fine = rng.random((300, 300)).astype(np.float32)
    out = residual_correct(fine, coarse, 100)
    assert np.abs(block_mean(out, 100) - coarse).max() < 2e-3


def test_fcn_gradients():
    rng = np.random.default_rng(1)
    m = FCNModel(seed=2)
    for L in m.layers:
        L["W"] = L["W"].astype(np.float64)
        L["b"] = L["b"].astype(np.float64)
    x = rng.normal(size=(1, 24, 24, 6))
    y, cache = m._forward(x, keep=True)
    dy = rng.normal(size=y.shape)
    g = m._backward(dy, cache)
    for li, idx in [(0, (3, 1)), (2, (50, 4)), (4, (7, 0))]:
        L, eps = m.layers[li], 1e-6
        L["W"][idx] += eps
        yp, _ = m._forward(x)
        L["W"][idx] -= 2 * eps
        ym, _ = m._forward(x)
        L["W"][idx] += eps
        num = ((yp - ym) * dy).sum() / (2 * eps)
        assert abs(num - g[li][0][idx]) < 1e-5 * max(1, abs(num))


def test_pipeline_end_to_end(scene):
    res = pipeline.run(scene.bands, scene.coarse["NO2"], "NO2",
                       methods=("bilinear", "lr", "mslr", "cnn"), cnn_steps=20)
    land = scene.landcover != CLS["water"]
    truth = scene.truth["NO2"][-1]
    for m, mp in res.maps.items():
        assert mp.shape == truth.shape and np.isfinite(mp).all()
        e = metrics.evaluate(mp, truth, land, coarse_obs=res.coarse_obs)
        assert e["r2"] > 0.0, (m, e)
        assert e["coarse_consistency_rmse"] < 0.2 * np.nanmean(res.coarse_obs)


def test_fcn_serialisation_roundtrip(tmp_path):
    m = FCNModel(seed=5)
    p = tmp_path / "m.json"
    m.save(p)
    m2 = FCNModel.load(p)
    x = np.random.default_rng(0).normal(size=(6, 120, 120)).astype(np.float32)
    assert np.allclose(m.predict(x), m2.predict(x), atol=1e-4)


def test_seasonal_fill():
    t = np.linspace(0, 6 * np.pi, 48)
    s = (np.sin(t)[:, None, None] + 2).repeat(2, 1).repeat(2, 2).astype(np.float32)
    s[[5, 6, 20], 0, 0] = np.nan
    f = seasonal_fill(s, period=16)
    assert not np.isnan(f).any()


def test_pretrained_cnn_path(scene, tmp_path):
    from hyperlocal_aq.core import feature_stack, standardize
    raw = feature_stack(scene.bands)
    feats, mu, sd = standardize(raw)
    m = FCNModel(seed=1).fit(feats, scene.coarse["NO2"], 100, steps=5)
    m.feat_mean, m.feat_std = mu, sd
    p = tmp_path / "NO2_cnn.json"
    m.save(p)
    loaded = FCNModel.load(p)
    assert np.allclose(loaded.feat_mean, mu) and loaded.res_m == 10
    res = pipeline.run(scene.bands, scene.coarse["NO2"], "NO2", methods=("cnn",), cnn_model=loaded)
    assert np.isfinite(res.maps["cnn"]).all()
    ok = ~np.isnan(res.coarse_obs)
    bm = block_mean(res.maps["cnn"], 100)
    assert np.abs(bm[ok] - res.coarse_obs[ok]).max() < 0.02 * np.nanmean(res.coarse_obs)
