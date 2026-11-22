"""Turn rasters into map overlays (PNG) and hotspot lists for the frontend."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage

# Perceptually ordered air-quality palette (clean -> polluted)
PALETTE = np.array([
    [26, 152, 80], [102, 189, 99], [166, 217, 106], [217, 239, 139], [254, 224, 139],
    [253, 174, 97], [244, 109, 67], [215, 48, 39], [165, 0, 38], [103, 0, 31],
], dtype=np.float32)


def colorize(field: np.ndarray, vmin: float, vmax: float, alpha: int = 210, mask=None) -> np.ndarray:
    nan = np.isnan(field)
    t = np.clip((np.nan_to_num(field, nan=vmin) - vmin) / (vmax - vmin + 1e-30), 0, 1) * (len(PALETTE) - 1)
    i = np.floor(t).astype(int)
    j = np.minimum(i + 1, len(PALETTE) - 1)
    f = (t - i)[..., None]
    rgb = PALETTE[i] * (1 - f) + PALETTE[j] * f
    a = np.full(field.shape, alpha, np.uint8)
    a[nan] = 0
    if mask is not None:
        a[~mask] = 0
    return np.dstack([np.nan_to_num(rgb).astype(np.uint8), a])


def png_b64(rgba: np.ndarray, max_side: int | None = None) -> str:
    img = Image.fromarray(rgba)
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def coarse_rgba(coarse: np.ndarray, vmin, vmax) -> np.ndarray:
    return colorize(coarse, vmin, vmax, alpha=200)


def hotspots(field: np.ndarray, bounds, top: int = 10, min_sep_px: int = 40, mask=None):
    """Local maxima above the 95th percentile -> [{lat, lon, value, rank}]."""
    f = np.where(mask, field, np.nan) if mask is not None else field
    thr = np.nanpercentile(f, 95)
    mx = ndimage.maximum_filter(np.nan_to_num(f, nan=-np.inf), size=min_sep_px)
    peaks = np.argwhere((f == mx) & (f >= thr))
    vals = f[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(-vals)[:top]
    H, W = field.shape
    lon0, lat0, lon1, lat1 = bounds
    out = []
    for rank, k in enumerate(order, 1):
        r, c = peaks[k]
        out.append({"rank": rank, "value": float(vals[k]),
                    "lon": lon0 + (c + 0.5) / W * (lon1 - lon0),
                    "lat": lat1 - (r + 0.5) / H * (lat1 - lat0)})
    return out
