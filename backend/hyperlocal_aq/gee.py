"""Google Earth Engine data access: live Sentinel-2 and Sentinel-5P.

Both products are downloaded as NumPy arrays with `ee.data.computePixels` on one
lon/lat lattice, so every Sentinel-5P cell covers exactly `factor x factor`
Sentinel-2 pixels (factor = 1000 m / output resolution).

Authentication, in order of preference:
  1. GEE_SERVICE_ACCOUNT_JSON  – the full JSON key of a service account
                                  (what you paste into Render / Railway secrets);
  2. GEE_SERVICE_ACCOUNT + GEE_KEY_FILE – path to the JSON key file;
  3. your own credentials from `earthengine authenticate` (local development).
GEE_PROJECT is the Cloud project registered for Earth Engine.
"""
from __future__ import annotations

import io
import json
import math
import os
import threading
from datetime import date, datetime, timedelta, timezone

import numpy as np

from .config import (GEE_PROJECT, LANDSAT_BANDS, LANDSAT_COLLECTION, POLLUTANTS, S2_BANDS,
                     S2_COLLECTION)

_ee = None
_init_error = None
_lock = threading.Lock()
MAX_TILE = 1024          # pixels per side per computePixels request (stays under the 48 MB cap)


def ee_module():
    """Import and initialise earthengine-api once (thread-safe)."""
    global _ee, _init_error
    with _lock:
        if _ee is not None:
            return _ee
        import ee
        project = os.environ.get("GEE_PROJECT", GEE_PROJECT)
        sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
        sa, key = os.environ.get("GEE_SERVICE_ACCOUNT"), os.environ.get("GEE_KEY_FILE")
        try:
            if sa_json:
                info = json.loads(sa_json)
                creds = ee.ServiceAccountCredentials(info["client_email"], key_data=sa_json)
                project = os.environ.get("GEE_PROJECT", info.get("project_id", project))
            elif sa and key:
                creds = ee.ServiceAccountCredentials(sa, key)
            else:
                creds = None
            if creds is not None:
                ee.Initialize(creds, project=project)
            else:
                ee.Initialize(project=project)
        except Exception as e:  # keep the reason for /api/health
            _init_error = str(e)
            raise
        _ee, _init_error = ee, None
        return ee


def status() -> dict:
    try:
        ee_module()
        return {"connected": True}
    except Exception as e:  # pragma: no cover - depends on credentials
        return {"connected": False, "error": _init_error or str(e)}


# ----------------------------------------------------------------------------
# Lattice
# ----------------------------------------------------------------------------
class Lattice:
    """lon/lat lattice with `res_m` pixels, aligned to a 1 km super-grid."""

    def __init__(self, bbox, res_m: int = 10, coarse_m: int = 1000):
        if coarse_m % res_m:
            raise ValueError("resolution must divide 1000 m")
        lon0, lat0, lon1, lat1 = bbox
        lat_mid = (lat0 + lat1) / 2
        self.res_m = res_m
        self.dx = res_m / (111_320 * math.cos(math.radians(lat_mid)))
        self.dy = res_m / 110_700
        self.factor = coarse_m // res_m
        self.cw = max(1, math.ceil((lon1 - lon0) / (self.dx * self.factor)))
        self.ch = max(1, math.ceil((lat1 - lat0) / (self.dy * self.factor)))
        self.width, self.height = self.cw * self.factor, self.ch * self.factor
        self.lon0, self.lat1 = lon0, lat1
        self.bounds = [lon0, lat1 - self.height * self.dy, lon0 + self.width * self.dx, lat1]

    def grid(self, x0=0, y0=0, w=None, h=None, coarse=False):
        s = self.factor if coarse else 1
        return {
            "dimensions": {"width": w or (self.cw if coarse else self.width),
                           "height": h or (self.ch if coarse else self.height)},
            "affineTransform": {
                "scaleX": self.dx * s, "shearX": 0, "translateX": self.lon0 + x0 * self.dx * s,
                "shearY": 0, "scaleY": -self.dy * s, "translateY": self.lat1 - y0 * self.dy * s,
            },
            "crsCode": "EPSG:4326",
        }


def _compute(image, grid) -> np.ndarray:
    ee = ee_module()
    raw = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
    if isinstance(raw, (bytes, bytearray)):
        raw = np.load(io.BytesIO(raw))
    names = raw.dtype.names
    return np.stack([raw[n].astype(np.float32) for n in names]) if names else raw[None].astype(np.float32)


def download(image, lat: Lattice, coarse=False, progress=None) -> np.ndarray:
    """Download an image on the lattice, tiling to respect request limits."""
    W = lat.cw if coarse else lat.width
    H = lat.ch if coarse else lat.height
    tiles = [(x0, y0) for y0 in range(0, H, MAX_TILE) for x0 in range(0, W, MAX_TILE)]
    out = None
    for k, (x0, y0) in enumerate(tiles):
        w, h = min(MAX_TILE, W - x0), min(MAX_TILE, H - y0)
        t = _compute(image, lat.grid(x0, y0, w, h, coarse))
        if out is None:
            out = np.full((t.shape[0], H, W), np.nan, np.float32)
        out[:, y0:y0 + h, x0:x0 + w] = t
        if progress:
            progress((k + 1) / len(tiles))
    return out


# ----------------------------------------------------------------------------
# Collections
# ----------------------------------------------------------------------------
def _mask_s2(img):
    """Mask cloud, cloud shadow and cirrus with the L2A scene classification (SCL)."""
    scl = img.select("SCL")
    ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(ok)


def s2_composite(region, start: str, end: str, max_cloud=40):
    """Cloud-masked Sentinel-2 median composite with fallbacks
    (Sentinel-2 → Landsat-8 → multi-year Sentinel-2 median).
    Returns (image with B2,B3,B4,B8 as reflectance 0-1, description)."""
    ee = ee_module()
    s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)))
    n = s2.size().getInfo()
    if n > 0:
        img = s2.map(_mask_s2).median().select(S2_BANDS).divide(10000)
        # fill pixels that were cloudy in every scene from a longer window
        longer = (ee.ImageCollection(S2_COLLECTION).filterBounds(region)
                  .filterDate(ee.Date(start).advance(-1, "year"), end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                  .map(_mask_s2).median().select(S2_BANDS).divide(10000))
        return img.unmask(longer), f"Sentinel-2 L2A, {n} scenes {start} to {end}"
    l8 = ee.ImageCollection(LANDSAT_COLLECTION).filterBounds(region).filterDate(start, end)
    n = l8.size().getInfo()
    if n > 0:
        img = l8.median().select(LANDSAT_BANDS).multiply(0.0000275).add(-0.2).rename(S2_BANDS)
        return img, f"Landsat-8 C2 L2 fallback, {n} scenes"
    hist = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate("2019-01-01", end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)).map(_mask_s2))
    return hist.median().select(S2_BANDS).divide(10000), "Sentinel-2 multi-year median fallback"


def s5p_days(pollutant: str, region, end_day: str, n_days: int, window_days: int):
    """Daily Sentinel-5P means for up to n_days overpass days before end_day.
    Returns (image with one band per day, list of ISO dates)."""
    ee = ee_module()
    info = POLLUTANTS[pollutant]
    end = date.fromisoformat(end_day) + timedelta(days=1)
    start = end - timedelta(days=window_days)
    found = []
    for product in (info["collection"], info["collection"].replace("/OFFL/", "/NRTI/")):
        col = ee.ImageCollection(product).select(info["band"]).filterBounds(region) \
            .filterDate(start.isoformat(), end.isoformat())
        times = col.aggregate_array("system:time_start").getInfo()
        days = sorted({datetime.fromtimestamp(t / 1000, tz=timezone.utc).date().isoformat() for t in times})
        if days:
            found = days[-n_days:]
            break
    if not found:
        raise RuntimeError(f"No Sentinel-5P {pollutant} data between {start} and {end_day}.")
    imgs = []
    for d in found:
        d0 = ee.Date(d)
        imgs.append(col.filterDate(d0, d0.advance(1, "day")).mean().unmask(-9999).rename(f"d{d.replace('-', '')}"))
    return ee.Image.cat(imgs), found


def fetch_stack(pollutant: str, end_day: str, bbox, res_m: int = 10, n_days: int = 20,
                window_days: int = 60, s2_window_days: int = 120, progress=None):
    """Download everything needed to downscale one pollutant over bbox.

    Returns bands (4,H,W) reflectance, coarse (D,h,w) S5P with NaN gaps (last day =
    the requested date or the closest overpass before it), Lattice, provenance dict.
    """
    ee = ee_module()
    p = progress or (lambda *_: None)
    lat = Lattice(bbox, res_m=res_m)
    region = ee.Geometry.Rectangle(lat.bounds)
    end = date.fromisoformat(end_day)
    p("Finding Sentinel-5P overpasses", 0.05)
    s5p, days = s5p_days(pollutant, region, end_day, n_days, window_days)
    p(f"Downloading Sentinel-5P ({len(days)} days)", 0.12)
    coarse = download(s5p, lat, coarse=True)
    coarse[coarse <= -9998] = np.nan
    p("Building cloud-free Sentinel-2 composite", 0.2)
    s2, s2_desc = s2_composite(region, (end - timedelta(days=s2_window_days)).isoformat(),
                               (end + timedelta(days=1)).isoformat())
    bands = download(s2, lat, progress=lambda f: p(f"Downloading Sentinel-2 at {res_m} m", 0.25 + 0.45 * f))
    if np.isnan(bands).any():
        med = np.nanmedian(bands, axis=(1, 2))
        for b in range(bands.shape[0]):
            bands[b][np.isnan(bands[b])] = med[b]
    keep = [i for i in range(coarse.shape[0]) if (~np.isnan(coarse[i])).sum() >= max(1, coarse[i].size // 10)]
    if not keep:
        raise RuntimeError("Every Sentinel-5P overpass in the window is cloud-covered over this area. "
                           "Try a dry-season date (November–May) or a larger area.")
    coarse = coarse[keep]
    days = [days[i] for i in keep]
    return bands, coarse, lat, {"s2": s2_desc, "s5p_days": days, "target_day": days[-1],
                                "s5p_product": POLLUTANTS[pollutant]["collection"]}
