"""Downscaling models.

All models learn the mapping  Sentinel-2 predictors -> pollutant  using only
1 km satellite cells as labels (no 10 m ground truth exists in real life).

* LinearModel   : the original GEE approach (regression on band means).
* MultiScaleLinearModel : regression on predictors + 30/90/270 m context.
* ForestModel   : Random Forest on cell-mean features (original RF notebook).
* FCNModel      : fully-convolutional CNN, trained with an *aggregation loss*:
                  the network predicts a 10 m map, the map is average-pooled to
                  1 km, and the loss compares the pooled map with the S5P cells.
                  Dilated convolutions give it ~310 m of spatial context, so
                  it can learn near-road decay, industrial footprints and so on.
                  Pure NumPy forward/backward (Adam) so it runs anywhere;
                  weights export to JSON for the TensorFlow.js front end and
                  match the Keras version in models_keras.py.

The original notebooks reshaped the 4 band values of a single pixel into a
2x2 "image", which gives a CNN no spatial context. They also re-fitted the
predictor scaler on the target, so the saved scaler held the target's
statistics. Both are fixed here.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .core import block_mean, tile_windows


# ----------------------------------------------------------------------------
# Regression baselines
# ----------------------------------------------------------------------------
def footprint_matrix(h, w, size):
    """Linear operator (h*w x h*w) of a box footprint on an h x w cell grid ('nearest' edges)."""
    from scipy import ndimage
    n = h * w
    M = np.empty((n, n), dtype=np.float64)
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        M[:, k] = ndimage.uniform_filter(e.reshape(h, w), size=size, mode="nearest").ravel()
    return M


def apply_footprint(cells, size):
    """Smooth (C?,h,w) cell grids with the satellite footprint (box, 'nearest' edges)."""
    from scipy import ndimage
    if size is None or tuple(size) == (1, 1):
        return cells
    if cells.ndim == 2:
        return ndimage.uniform_filter(cells, size=size, mode="nearest")
    return np.stack([ndimage.uniform_filter(c, size=size, mode="nearest") for c in cells])


class LinearModel:
    name = "Linear regression"

    def fit(self, feats_z, coarse_days, factor, footprint=None):
        """feats_z: (C,H,W) standardized predictors; coarse_days: (D,h,w) with NaN.
        footprint: satellite footprint in cells, e.g. (5, 4); the cell-mean predictors
        are smoothed the same way the instrument smooths the pollutant field."""
        cell_feats = apply_footprint(block_mean(feats_z, factor), footprint)   # (C,h,w)
        X, y = [], []
        for c in coarse_days:
            m = ~np.isnan(c)
            X.append(cell_feats[:, m].T)
            y.append(c[m])
        X = np.concatenate(X)
        y = np.concatenate(y)
        A = np.c_[X, np.ones(len(X))]
        ridge = 1e-3 * np.eye(A.shape[1])
        ridge[-1, -1] = 0
        self.coef_ = np.linalg.solve(A.T @ A + ridge * len(X), A.T @ y)
        return self

    def predict(self, feats_z):
        C = feats_z.shape[0]
        return (np.tensordot(self.coef_[:C], feats_z, axes=1) + self.coef_[C]).astype(np.float32)

    def to_dict(self):
        return {"type": "linear", "coef": self.coef_[:-1].tolist(), "intercept": float(self.coef_[-1])}


class MultiScaleLinearModel(LinearModel):
    """Linear regression on the predictors plus their 30 m / 90 m / 270 m
    neighbourhood means ("statistical correlation with spatial context").
    The model is linear, so the cell mean of the 10 m prediction equals the
    prediction from cell-mean features, and fitting on 1 km cells is exact."""
    name = "Multi-scale regression"
    SCALES = (3, 9, 27)

    @classmethod
    def expand(cls, feats_z):
        from scipy import ndimage
        ms = [feats_z] + [np.stack([ndimage.uniform_filter(c, s) for c in feats_z]) for s in cls.SCALES]
        return np.concatenate(ms).astype(np.float32)

    def fit(self, feats_z, coarse_days, factor, footprint=None):
        X = self.expand(feats_z)
        self.mu = X.reshape(X.shape[0], -1).mean(1)
        self.sd = X.reshape(X.shape[0], -1).std(1) + 1e-6
        return super().fit((X - self.mu[:, None, None]) / self.sd[:, None, None], coarse_days, factor,
                           footprint=footprint)

    def predict(self, feats_z):
        X = self.expand(feats_z)
        return super().predict((X - self.mu[:, None, None]) / self.sd[:, None, None])


class ForestModel:
    name = "Random forest"

    def __init__(self, n_estimators=150, seed=42):
        from sklearn.ensemble import RandomForestRegressor
        self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3,
                                        random_state=seed, n_jobs=-1)

    def fit(self, feats_z, coarse_days, factor, footprint=None):
        cell_feats = apply_footprint(block_mean(feats_z, factor), footprint)
        X, y = [], []
        for c in coarse_days:
            m = ~np.isnan(c)
            X.append(cell_feats[:, m].T)
            y.append(c[m])
        self.rf.fit(np.concatenate(X), np.concatenate(y))
        return self

    def predict(self, feats_z, chunk=400_000):
        C, H, W = feats_z.shape
        flat = feats_z.reshape(C, -1).T
        out = np.empty(flat.shape[0], dtype=np.float32)
        for i in range(0, len(flat), chunk):
            out[i:i + chunk] = self.rf.predict(flat[i:i + chunk])
        return out.reshape(H, W)


# ----------------------------------------------------------------------------
# Fully-convolutional network (NumPy implementation with manual gradients)
# ----------------------------------------------------------------------------
class FCNModel:
    name = "CNN (fully-convolutional)"
    # (kernel, dilation, out_channels, activation)
    ARCH = [(3, 1, 16, "relu"), (3, 2, 16, "relu"), (3, 4, 16, "relu"),
            (3, 8, 16, "relu"), (1, 1, 1, "linear")]

    def __init__(self, in_ch=6, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = []
        c = in_ch
        for k, d, co, act in self.ARCH:
            fan_in = k * k * c
            W = (rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, co))).astype(np.float32)
            if act == "linear":
                W *= 0.1
            self.layers.append({"k": k, "d": d, "act": act, "W": W, "b": np.zeros(co, np.float32)})
            c = co
        self.y_mean, self.y_std = 0.0, 1.0
        self.feat_mean = self.feat_std = None   # set when trained through pipeline.run
        self.res_m = 10
        self.meta = {}
        self.history = []

    @property
    def receptive_field(self):
        return 1 + sum((k - 1) * d for k, d, *_ in self.ARCH)

    # -- forward / backward ----------------------------------------------------
    @staticmethod
    def _conv(xp, W, k, d, H, Wd, C):
        """'same' dilated conv as a sum of k*k shifted matmuls (no im2col copy)."""
        out = None
        for idx in range(k * k):
            i, j = divmod(idx, k)
            part = xp[:, i * d:i * d + H, j * d:j * d + Wd, :] @ W[idx * C:(idx + 1) * C]
            out = part if out is None else out + part
        return out

    def _forward(self, x, keep=False):
        cache = []
        for L in self.layers:
            B, H, Wd, C = x.shape
            if L["k"] > 1:
                p = L["d"] * (L["k"] // 2)
                xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
                z = self._conv(xp, L["W"], L["k"], L["d"], H, Wd, C) + L["b"]
            else:
                xp = x
                z = x @ L["W"] + L["b"]
            out = np.maximum(z, 0) if L["act"] == "relu" else z
            if keep:
                cache.append((x.shape, xp, z))
            x = out
        return x[..., 0], cache

    def _backward(self, dy, cache):
        grads = []
        g = dy[..., None]
        for L, (xshape, xp, z) in zip(reversed(self.layers), reversed(cache)):
            if L["act"] == "relu":
                g = g * (z > 0)
            B, H, Wd, C = xshape
            k, d = L["k"], L["d"]
            g2 = g.reshape(-1, g.shape[-1])
            gb = g2.sum(axis=0)
            if k == 1:
                gW = xp.reshape(-1, C).T @ g2
                gx = g @ L["W"].T
            else:
                p = d * (k // 2)
                gW = np.empty_like(L["W"])
                gxp = np.zeros_like(xp)
                for idx in range(k * k):
                    i, j = divmod(idx, k)
                    xs = xp[:, i * d:i * d + H, j * d:j * d + Wd, :]
                    Wk = L["W"][idx * C:(idx + 1) * C]
                    gW[idx * C:(idx + 1) * C] = np.tensordot(xs, g, axes=([0, 1, 2], [0, 1, 2]))
                    gxp[:, i * d:i * d + H, j * d:j * d + Wd, :] += g @ Wk.T
                gx = gxp[:, p:p + H, p:p + Wd, :]
            grads.append((gW, gb))
            g = gx
        return list(reversed(grads))

    # -- training --------------------------------------------------------------
    def fit(self, feats_z, coarse_days, factor, steps=500, window_cells=3, batch=2,
            lr=3e-3, weight_decay=1e-4, seed=0, log_every=50, verbose=False,
            loss="centered", footprint=None):
        """Train with the aggregation loss on random windows of `window_cells`^2 cells.

        loss="plain":    mean over cells of (pooled prediction - S5P cell)^2
        loss="centered": same after subtracting each window's mean from both
        """
        rng = np.random.default_rng(seed)
        # Accept one region or a list of regions (feats_z, coarse_days pairs).
        regions = list(zip(feats_z, coarse_days)) if isinstance(feats_z, (list, tuple)) \
            else [(feats_z, coarse_days)]
        vals = np.concatenate([c[~np.isnan(c)] for _, c in regions])
        # Scale targets by the *spatial* spread within each overpass, not the day-to-day
        # spread (real Sentinel-5P levels swing far more between days than across a city,
        # which would otherwise leave almost no learning signal).
        anoms = np.concatenate([(c - np.nanmean(c, axis=(1, 2), keepdims=True))[~np.isnan(c)]
                                for _, c in regions])
        self.y_mean, self.y_std = float(vals.mean()), float(anoms.std() + 1e-12)
        data = []
        for f, c in regions:
            if c.shape[1] < window_cells or c.shape[2] < window_cells:
                continue
            data.append((np.ascontiguousarray(f.transpose(1, 2, 0)), c, ~np.isnan(c)))
        if not data:
            raise ValueError(f"Regions must be at least {window_cells}x{window_cells} km for CNN training.")
        weights = np.array([c.size for _, c, _ in data], float)
        weights /= weights.sum()
        m = [np.zeros_like(L["W"]) for L in self.layers] + [np.zeros_like(L["b"]) for L in self.layers]
        v = [np.zeros_like(a) for a in m]
        b1, b2, eps = 0.9, 0.999, 1e-8
        ws = window_cells * factor
        Fm = None
        if footprint is not None and tuple(footprint) != (1, 1):
            Fm = footprint_matrix(window_cells, window_cells, footprint)
        t0 = time.time()
        for step in range(1, steps + 1):
            xb, yb, mb = [], [], []
            for _ in range(batch):
                x_all, cd, valid_all = data[int(rng.choice(len(data), p=weights))]
                d = int(rng.integers(0, cd.shape[0]))
                win = tile_windows(valid_all[d], window_cells, rng, 1)
                if not win:
                    continue
                (cy, cx), = win
                xb.append(x_all[cy * factor:cy * factor + ws, cx * factor:cx * factor + ws])
                t = cd[d, cy:cy + window_cells, cx:cx + window_cells]
                mb.append(~np.isnan(t))
                yb.append(np.nan_to_num((t - self.y_mean) / self.y_std))
            if not xb:
                continue
            xb, yb, mb = np.stack(xb), np.stack(yb), np.stack(mb)
            # random flips for augmentation (physics is rotation-agnostic)
            if rng.random() < 0.5:
                xb, yb, mb = xb[:, :, ::-1], yb[:, :, ::-1], mb[:, :, ::-1]
            if rng.random() < 0.5:
                xb, yb, mb = xb[:, ::-1], yb[:, ::-1], mb[:, ::-1]
            xb = np.ascontiguousarray(xb)
            pred, cache = self._forward(xb, keep=True)
            pooled = block_mean(pred, factor)
            if Fm is not None:     # what the satellite would see of this 10 m field
                B, hh, ww = pooled.shape
                pooled = (pooled.reshape(B, -1) @ Fm.T).reshape(B, hh, ww)
            err = (pooled - yb) * mb
            if loss == "centered":
                # Remove each window's mean so the network learns *within-window
                # contrasts* (source signatures) rather than km-scale regional
                # gradients, which the residual correction restores later.
                cnt = np.maximum(mb.sum(axis=(1, 2), keepdims=True), 1)
                err = (err - err.sum(axis=(1, 2), keepdims=True) / cnt) * mb
            n_valid = max(mb.sum(), 1)
            loss = float((err ** 2).sum() / n_valid)
            dpooled = 2 * err / n_valid
            if Fm is not None:
                B, hh, ww = dpooled.shape
                dpooled = (dpooled.reshape(B, -1) @ Fm).reshape(B, hh, ww)
            dpred = np.repeat(np.repeat(dpooled, factor, axis=1), factor, axis=2) / (factor * factor)
            grads = self._backward(dpred.astype(np.float32), cache)
            # Adam update
            params = [L["W"] for L in self.layers] + [L["b"] for L in self.layers]
            gs = [g[0] for g in grads] + [g[1] for g in grads]
            lr_t = lr * (0.5 * (1 + np.cos(np.pi * step / steps)) * 0.9 + 0.1)  # cosine decay
            for i, (p, g) in enumerate(zip(params, gs)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                p -= (lr_t * mh / (np.sqrt(vh) + eps)).astype(np.float32)
                if i < len(self.layers):           # decoupled weight decay (AdamW) on weights
                    p *= np.float32(1 - lr_t * weight_decay)
            if step % log_every == 0 or step == 1:
                self.history.append((step, loss))
                if verbose:
                    print(f"  step {step:4d}  loss {loss:.4f}  ({time.time() - t0:.1f}s)")
        return self

    def predict(self, feats_z, tile=512):
        """Tiled inference with overlap so tiles join without seams."""
        C, H, W = feats_z.shape
        x = feats_z.transpose(1, 2, 0)
        halo = self.receptive_field // 2 + 1
        out = np.empty((H, W), dtype=np.float32)
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                ya, xa = max(0, y0 - halo), max(0, x0 - halo)
                yb, xb = min(H, y0 + tile + halo), min(W, x0 + tile + halo)
                p, _ = self._forward(np.ascontiguousarray(x[None, ya:yb, xa:xb]))
                out[y0:min(H, y0 + tile), x0:min(W, x0 + tile)] = \
                    p[0, y0 - ya:y0 - ya + min(tile, H - y0), x0 - xa:x0 - xa + min(tile, W - x0)]
        return out * self.y_std + self.y_mean

    # -- (de)serialisation -----------------------------------------------------
    def to_dict(self):
        return {
            "type": "fcn",
            "y_mean": self.y_mean, "y_std": self.y_std, "res_m": self.res_m,
            "feat_mean": None if self.feat_mean is None else [float(v) for v in self.feat_mean],
            "feat_std": None if self.feat_std is None else [float(v) for v in self.feat_std],
            "meta": self.meta,
            "layers": [{"k": L["k"], "d": L["d"], "act": L["act"],
                        "W": L["W"].round(6).tolist(), "b": L["b"].round(6).tolist()}
                       for L in self.layers],
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.layers = [{"k": L["k"], "d": L["d"], "act": L["act"],
                       "W": np.array(L["W"], np.float32), "b": np.array(L["b"], np.float32)}
                      for L in d["layers"]]
        obj.y_mean, obj.y_std, obj.history = d["y_mean"], d["y_std"], []
        obj.feat_mean = None if d.get("feat_mean") is None else np.array(d["feat_mean"], np.float32)
        obj.feat_std = None if d.get("feat_std") is None else np.array(d["feat_std"], np.float32)
        obj.res_m = d.get("res_m", 10)
        obj.meta = d.get("meta", {})
        return obj

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ----------------------------------------------------------------------------
# Hybrid: multi-scale regression base + CNN correction, trained on whole areas
# ----------------------------------------------------------------------------
def mean_anomaly_target(coarse_days: np.ndarray) -> np.ndarray:
    """Average over overpasses of each day's standardised spatial anomaly (cloudy cells ignored).

    Real Sentinel-5P varies far more from day to day than across a city, and each
    "1 km" cell is a ~5 km footprint average, so single days are mostly noise for
    learning the surface → pollution relationship. Averaging ~100 daily anomaly
    maps keeps the persistent pattern and cuts the noise by ~10x."""
    acc = np.zeros(coarse_days.shape[1:])
    n = np.zeros(coarse_days.shape[1:])
    for d in coarse_days:
        ok = ~np.isnan(d)
        if ok.sum() < 4:
            continue
        s = float(np.nanstd(d))
        if s <= 0:
            continue
        a = (d - np.nanmean(d)) / s
        acc[ok] += a[ok]
        n[ok] += 1
    t = acc / np.maximum(n, 1)
    t[n == 0] = 0.0
    return (t - t.mean()).astype(np.float32)


def regression_base(feats_z, coarse_days, factor, footprint, target=None):
    """Multi-scale regression pattern, standardised and scaled to the mean-anomaly target."""
    if target is None:
        target = mean_anomaly_target(coarse_days)
    b = MultiScaleLinearModel().fit(feats_z, coarse_days, factor, footprint=footprint).predict(feats_z)
    b = (b - b.mean()) / (b.std() + 1e-12)
    sb = apply_footprint(block_mean(b, factor), footprint)
    sb = sb - sb.mean()
    a = float((sb * target).sum() / ((sb * sb).sum() + 1e-12))
    return (a * b).astype(np.float32), target


def fit_hybrid_cnn(feats_list, coarse_list, factor=100, footprint=None, steps=600, lr=3e-3,
                   weight_decay=1e-4, seed=0, log_every=50, verbose=False, model=None):
    """Train the CNN correction on whole areas: footprint(blockmean(base + cnn)) ≈ target.

    Each step uses one whole area (chosen in proportion to its size) so the ~5 km
    footprint always fits inside what the network sees. Returns (model, bases)."""
    rng = np.random.default_rng(seed)
    model = model or FCNModel(in_ch=feats_list[0].shape[0], seed=seed)
    model.y_mean, model.y_std = 0.0, 1.0
    model.meta = {**getattr(model, "meta", {}), "hybrid": True}
    data = []
    for f, c in zip(feats_list, coarse_list):
        base, target = regression_base(f, c, factor, footprint)
        h, w = target.shape
        Fm = footprint_matrix(h, w, footprint) if footprint is not None else np.eye(h * w)
        data.append((np.ascontiguousarray(f.transpose(1, 2, 0))[None], block_mean(base, factor), target, Fm))
    sizes = np.array([d[2].size for d in data], float)
    prob = sizes / sizes.sum()
    P = [L["W"] for L in model.layers] + [L["b"] for L in model.layers]
    m1 = [np.zeros_like(p) for p in P]
    v1 = [np.zeros_like(p) for p in P]
    t0 = time.time()
    model.history = []
    for step in range(1, steps + 1):
        x, bb, target, Fm = data[int(rng.choice(len(data), p=prob))]
        fw, fh = rng.random() < 0.5, rng.random() < 0.5
        if fw:
            x = x[:, :, ::-1]
        if fh:
            x = x[:, ::-1]
        pred, cache = model._forward(np.ascontiguousarray(x), keep=True)
        bm = block_mean(pred, factor)[0]
        if fw:
            bm = bm[:, ::-1]
        if fh:
            bm = bm[::-1]
        h, w = bm.shape
        seen = (Fm @ (bm + bb).ravel()).reshape(h, w)
        err = (seen - seen.mean()) - target
        loss = float((err ** 2).mean())
        g = 2 * err / err.size
        g = g - g.mean()
        gbm = (Fm.T @ g.ravel()).reshape(h, w)
        if fw:
            gbm = gbm[:, ::-1]
        if fh:
            gbm = gbm[::-1]
        dpred = np.repeat(np.repeat(gbm[None], factor, 1), factor, 2) / (factor * factor)
        grads = model._backward(dpred.astype(np.float32), cache)
        gs = [g_[0] for g_ in grads] + [g_[1] for g_ in grads]
        lr_t = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        for i, (p, gr) in enumerate(zip(P, gs)):
            m1[i] = 0.9 * m1[i] + 0.1 * gr
            v1[i] = 0.999 * v1[i] + 0.001 * gr * gr
            p -= (lr_t * (m1[i] / (1 - 0.9 ** step)) / (np.sqrt(v1[i] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
            if i < len(model.layers):
                p *= np.float32(1 - lr_t * weight_decay)
        model.history.append((step, loss))
        if verbose and (step % log_every == 0 or step == 1):
            print(f"  step {step:4d}  loss {loss:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    return model


def hybrid_predict(model, feats_for_model, feats_z, coarse_days, factor, footprint):
    """Static hybrid pattern for an area: regression base fitted here + pretrained CNN correction."""
    base, _ = regression_base(feats_z, coarse_days, factor, footprint)
    return base + model.predict(feats_for_model)


MODEL_REGISTRY = {"lr": LinearModel, "mslr": MultiScaleLinearModel, "rf": ForestModel, "cnn": FCNModel}
