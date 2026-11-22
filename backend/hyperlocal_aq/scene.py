"""Simulated validation scene (an Observing System Simulation Experiment, OSSE).

Used only for accuracy measurement (validation/run_validation.py) and tests,
never by the live system. Real satellite data has no 10 m ground truth, so
skill is measured on a scene where the 10 m pollutant field is known and the
methods only see Sentinel-5P-like 1 km observations of it.

What it simulates:
  * a 14 x 14 km region of east-central Mumbai with approximate geography
    (Thane Creek, the harbour, Mithi river, Trombay hill, Mahul refineries,
    Tata Power Trombay, RCF, Deonar dumping ground, the airport, Dharavi,
    BKC, the Somaiya Vidyavihar campus and the main arterial roads);
  * Sentinel-2 L2A surface reflectance (B2, B3, B4, B8) at 10 m, derived from
    a land-cover map with realistic spectra, urban greenery and texture;
  * a 10 m truth field for each pollutant: background + sources (traffic,
    industry, dumping ground, ...) + dispersion with a wind-driven plume;
  * Sentinel-5P-like observations: the truth averaged over 1 km cells, with
    retrieval noise and cloud gaps, for several overpasses (days).

The geography is approximate and only for illustration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import VALIDATION_ORIGIN, VALIDATION_SIZE_M, FINE_RES_M, FACTOR

# ----------------------------------------------------------------------------
# Land-cover classes and Sentinel-2 surface reflectance (B2, B3, B4, B8)
# ----------------------------------------------------------------------------
CLASSES = [
    # name,           B2,    B3,    B4,    B8
    ("water",        0.055, 0.050, 0.030, 0.020),
    ("mangrove",     0.030, 0.050, 0.030, 0.270),
    ("forest",       0.030, 0.058, 0.034, 0.320),
    ("grass",        0.045, 0.075, 0.062, 0.280),
    ("residential",  0.085, 0.095, 0.102, 0.200),
    ("dense_urban",  0.100, 0.110, 0.126, 0.175),
    ("commercial",   0.132, 0.140, 0.146, 0.198),
    ("industrial",   0.140, 0.150, 0.160, 0.205),
    ("refinery",     0.175, 0.180, 0.185, 0.215),
    ("airport",      0.120, 0.130, 0.140, 0.170),
    ("bare_dump",    0.092, 0.102, 0.135, 0.168),
    ("major_road",   0.108, 0.118, 0.128, 0.140),
    ("minor_road",   0.100, 0.110, 0.120, 0.160),
]
CLS = {name: i for i, (name, *_r) in enumerate(CLASSES)}
SPECTRA = np.array([c[1:] for c in CLASSES], dtype=np.float32)

# ----------------------------------------------------------------------------
# Approximate geography (lon, lat)
# ----------------------------------------------------------------------------
WATER_POLYS = [
    # Thane Creek (east)
    [(72.950, 19.130), (72.990, 19.130), (72.990, 18.970), (72.938, 18.970),
     (72.944, 18.998), (72.951, 19.030), (72.946, 19.060), (72.952, 19.090),
     (72.947, 19.130)],
    # Mumbai harbour / Sewri mudflats (south-centre)
    [(72.860, 18.970), (72.890, 18.970), (72.886, 18.994), (72.876, 19.010),
     (72.866, 19.017), (72.860, 19.000)],
    # Mahim creek mouth (west)
    [(72.830, 19.032), (72.846, 19.040), (72.849, 19.046), (72.840, 19.052),
     (72.830, 19.060)],
]
LAKES = []  # none inside the test region
MANGROVE_POLYS = [
    [(72.930, 19.030), (72.951, 19.030), (72.946, 19.060), (72.952, 19.090),
     (72.947, 19.130), (72.933, 19.130), (72.936, 19.095), (72.928, 19.062)],
    [(72.858, 19.000), (72.866, 19.017), (72.876, 19.010), (72.879, 19.016),
     (72.868, 19.024), (72.855, 19.012)],
    [(72.842, 19.036), (72.852, 19.044), (72.850, 19.052), (72.842, 19.052)],
]
FOREST_POLYS = [
    # Trombay hill (BARC)
    [(72.906, 18.988), (72.938, 18.988), (72.944, 19.005), (72.940, 19.028),
     (72.920, 19.034), (72.906, 19.020)],
]
REFINERY_POLYS = [
    # BPCL / HPCL refineries at Mahul
    [(72.886, 18.996), (72.905, 18.993), (72.906, 19.016), (72.889, 19.017)],
    # Tata Power Trombay thermal station
    [(72.896, 18.982), (72.910, 18.982), (72.910, 18.993), (72.896, 18.993)],
]
INDUSTRIAL_POLYS = [
    # RCF Chembur fertiliser complex
    [(72.889, 19.019), (72.906, 19.019), (72.906, 19.035), (72.889, 19.035)],
    # Kurla / Saki Naka small industry
    [(72.876, 19.095), (72.890, 19.095), (72.890, 19.108), (72.876, 19.108)],
]
DUMP_POLYS = [
    # Deonar dumping ground
    [(72.917, 19.049), (72.934, 19.050), (72.935, 19.066), (72.918, 19.066)],
]
AIRPORT_POLYS = [
    [(72.850, 19.083), (72.884, 19.083), (72.886, 19.104), (72.852, 19.104)],
]
DENSE_POLYS = [
    # Dharavi
    [(72.848, 19.034), (72.862, 19.035), (72.863, 19.049), (72.850, 19.050)],
    # Kurla west / Jarimari
    [(72.872, 19.062), (72.882, 19.062), (72.882, 19.080), (72.872, 19.080)],
    # Govandi / Shivaji Nagar
    [(72.912, 19.050), (72.917, 19.049), (72.918, 19.072), (72.909, 19.072)],
]
COMMERCIAL_POLYS = [
    # Bandra-Kurla Complex
    [(72.858, 19.058), (72.872, 19.058), (72.872, 19.071), (72.858, 19.071)],
]
PARK_POLYS = [
    # K. J. Somaiya Vidyavihar campus (~65 acres, green campus)
    [(72.8970, 19.0700), (72.9025, 19.0700), (72.9025, 19.0752), (72.8970, 19.0752)],
    # Maharashtra Nature Park
    [(72.852, 19.049), (72.858, 19.049), (72.858, 19.054), (72.852, 19.054)],
]
SOMAIYA = PARK_POLYS[0]

# Major roads: (name, traffic weight, polyline)
ROADS = [
    ("Eastern Express Hwy", 1.00, [(72.862, 19.040), (72.882, 19.050), (72.905, 19.064),
                                    (72.918, 19.080), (72.928, 19.100), (72.934, 19.130)]),
    ("Western Express Hwy", 0.95, [(72.846, 19.050), (72.850, 19.075), (72.853, 19.100),
                                    (72.855, 19.130)]),
    ("Sion-Panvel Hwy",     0.90, [(72.874, 19.045), (72.900, 19.050), (72.925, 19.055),
                                    (72.950, 19.062)]),
    ("Eastern Freeway",     0.70, [(72.858, 18.970), (72.872, 19.016), (72.886, 19.032),
                                    (72.897, 19.046)]),
    ("LBS Marg",            0.70, [(72.870, 19.050), (72.884, 19.074), (72.904, 19.096),
                                    (72.918, 19.130)]),
    ("SCLR",                0.60, [(72.862, 19.068), (72.885, 19.070), (72.905, 19.068),
                                    (72.916, 19.073)]),
    ("Andheri-Kurla Rd",    0.55, [(72.877, 19.072), (72.872, 19.090), (72.866, 19.110),
                                    (72.862, 19.130)]),
    ("Dr Ambedkar Rd",      0.55, [(72.845, 18.970), (72.853, 19.012), (72.862, 19.040)]),
    ("Mahul Rd",            0.50, [(72.886, 19.035), (72.893, 19.018), (72.900, 18.994)]),
    ("Ghatkopar-Mahul Rd",  0.45, [(72.910, 19.085), (72.905, 19.052), (72.900, 19.035)]),
]

# Emission weights of each source type, per pollutant.
# Keys: major_road, minor_road, refinery, industrial, airport, dump,
#       dense_urban, commercial, residential, vegetation
SOURCE_WEIGHTS = {
    "NO2":    dict(major_road=1.00, minor_road=0.30, refinery=0.80, industrial=0.45, airport=0.55,
                   dump=0.10, dense_urban=0.25, commercial=0.30, residential=0.10, vegetation=0.0),
    "SO2":    dict(major_road=0.08, minor_road=0.02, refinery=1.00, industrial=0.55, airport=0.10,
                   dump=0.15, dense_urban=0.05, commercial=0.03, residential=0.01, vegetation=0.0),
    "CO":     dict(major_road=0.85, minor_road=0.30, refinery=0.35, industrial=0.25, airport=0.35,
                   dump=1.00, dense_urban=0.45, commercial=0.15, residential=0.10, vegetation=0.0),
    "HCHO":   dict(major_road=0.35, minor_road=0.10, refinery=0.70, industrial=0.30, airport=0.15,
                   dump=0.45, dense_urban=0.15, commercial=0.08, residential=0.05, vegetation=0.40),
    "O3":     dict(major_road=1.00, minor_road=0.30, refinery=0.80, industrial=0.45, airport=0.55,
                   dump=0.10, dense_urban=0.25, commercial=0.30, residential=0.10, vegetation=0.0),
    "AER_AI": dict(major_road=0.40, minor_road=0.15, refinery=0.40, industrial=0.50, airport=0.20,
                   dump=1.00, dense_urban=0.55, commercial=0.10, residential=0.10, vegetation=0.0),
}
# background, regional amplitude, local amplitude (in product units)
AMPLITUDES = {
    "NO2":    (4.0e-5, 7.0e-5, 1.3e-4),
    "SO2":    (1.2e-4, 4.0e-4, 6.0e-4),
    "CO":     (3.10e-2, 4.0e-3, 6.0e-3),
    "HCHO":   (1.1e-4, 7.0e-5, 9.0e-5),
    "O3":     (1.22e-1, -2.0e-3, -3.0e-3),   # ozone titration near NOx sources
    "AER_AI": (0.10, 0.60, 0.80),
}

LABELS = [
    ("Dharavi", 72.855, 19.042), ("BKC", 72.865, 19.064), ("Kurla", 72.881, 19.072),
    ("Sion", 72.862, 19.030), ("Chembur", 72.898, 19.054), ("Ghatkopar", 72.910, 19.092),
    ("Vidyavihar\n(Somaiya)", 72.8997, 19.0726), ("Mankhurd", 72.932, 19.045),
    ("Deonar", 72.926, 19.058), ("Trombay", 72.925, 19.010), ("Mahul", 72.896, 19.005),
    ("Wadala", 72.860, 19.015), ("Airport", 72.868, 19.094), ("Thane Creek", 72.960, 19.070),
    ("Harbour", 72.872, 18.992),
]


@dataclass
class Grid:
    """A regular lon/lat-aligned grid. Row 0 is the northern edge."""
    lon0: float
    lat0: float
    width: int
    height: int
    res_m: float = FINE_RES_M

    @property
    def m_per_deg_lon(self) -> float:
        lat_mid = self.lat0 + self.height * self.res_m / 110_700 / 2
        return 111_320 * math.cos(math.radians(lat_mid))

    @property
    def m_per_deg_lat(self) -> float:
        return 110_700.0

    @property
    def bounds(self):
        lon1 = self.lon0 + self.width * self.res_m / self.m_per_deg_lon
        lat1 = self.lat0 + self.height * self.res_m / self.m_per_deg_lat
        return [self.lon0, self.lat0, lon1, lat1]

    def to_px(self, lon, lat):
        """lon/lat -> fractional (col, row) pixel coordinates."""
        col = (np.asarray(lon) - self.lon0) * self.m_per_deg_lon / self.res_m
        lat1 = self.bounds[3]
        row = (lat1 - np.asarray(lat)) * self.m_per_deg_lat / self.res_m
        return col, row


@dataclass
class Scene:
    grid: Grid
    landcover: np.ndarray            # (H, W) uint8 class ids
    bands: np.ndarray                # (4, H, W) float32 reflectance B2,B3,B4,B8
    sources: dict                    # source-type -> (H, W) float32 intensity
    truth: dict = field(default_factory=dict)      # pollutant -> (days, H, W) float32
    coarse: dict = field(default_factory=dict)     # pollutant -> (days, h, w) float32 (NaN = cloud)
    days: int = 1
    winds: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# Raster helpers
# ----------------------------------------------------------------------------
def _poly_mask(grid: Grid, poly, cols, rows) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorised over all pixel centres."""
    px, py = grid.to_px([p[0] for p in poly], [p[1] for p in poly])
    inside = np.zeros(cols.shape, dtype=bool)
    n = len(px)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = px[i], py[i], px[j], py[j]
        cond = (yi > rows) != (yj > rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (rows - yi) / (yj - yi + 1e-12) + xi
        inside ^= cond & (cols < xint)
        j = i
    return inside


def _value_noise(shape, scale_px, rng, octaves=3) -> np.ndarray:
    """Smooth multi-octave value noise in [0, 1]."""
    h, w = shape
    out = np.zeros(shape, dtype=np.float32)
    amp, total = 1.0, 0.0
    s = scale_px
    for _ in range(octaves):
        gh, gw = max(2, int(h / s) + 2), max(2, int(w / s) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        z = ndimage.zoom(g, (h / (gh - 1), w / (gw - 1)), order=3)[:h, :w]
        out += amp * z
        total += amp
        amp *= 0.5
        s /= 2.0
    out /= total
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / (hi - lo + 1e-9), 0, 1)


def _draw_polyline(mask: np.ndarray, grid: Grid, line, value=True):
    px, py = grid.to_px([p[0] for p in line], [p[1] for p in line])
    h, w = mask.shape
    for k in range(len(px) - 1):
        n = int(max(abs(px[k + 1] - px[k]), abs(py[k + 1] - py[k])) * 2) + 2
        xs = np.linspace(px[k], px[k + 1], n)
        ys = np.linspace(py[k], py[k + 1], n)
        ci, ri = np.round(xs).astype(int), np.round(ys).astype(int)
        ok = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
        mask[ri[ok], ci[ok]] = value


# ----------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------
def build_scene(seed: int = 7, days: int = 8, size_m: int = VALIDATION_SIZE_M,
                pollutants=None) -> Scene:
    rng = np.random.default_rng(seed)
    n = size_m // FINE_RES_M
    grid = Grid(VALIDATION_ORIGIN[0], VALIDATION_ORIGIN[1], n, n)
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5

    lc = np.full((n, n), CLS["residential"], dtype=np.uint8)
    dense_noise = _value_noise((n, n), 120, rng)
    lc[dense_noise > 0.62] = CLS["dense_urban"]

    def paint(polys, cls):
        for p in polys:
            lc[_poly_mask(grid, p, cols, rows)] = cls

    paint(COMMERCIAL_POLYS, CLS["commercial"])
    paint(DENSE_POLYS, CLS["dense_urban"])
    paint(INDUSTRIAL_POLYS, CLS["industrial"])
    paint(REFINERY_POLYS, CLS["refinery"])
    paint(DUMP_POLYS, CLS["bare_dump"])
    paint(AIRPORT_POLYS, CLS["airport"])
    paint(FOREST_POLYS, CLS["forest"])
    paint(PARK_POLYS, CLS["grass"])

    # Small urban parks / playgrounds scattered through residential areas
    parks = _value_noise((n, n), 40, rng, octaves=2)
    lc[(parks > 0.93) & (lc == CLS["residential"])] = CLS["grass"]

    # Airport: grass between runways, two crossing runways
    ap = lc == CLS["airport"]
    lc[ap & (_value_noise((n, n), 30, rng, 2) > 0.45)] = CLS["grass"]
    rw = np.zeros((n, n), dtype=bool)
    _draw_polyline(rw, grid, [(72.853, 19.086), (72.884, 19.101)])
    _draw_polyline(rw, grid, [(72.860, 19.102), (72.880, 19.084)])
    rw = ndimage.binary_dilation(rw, iterations=3)
    lc[rw & ap] = CLS["airport"]

    # Mithi river (with mangrove fringe in its lower reach)
    river = np.zeros((n, n), dtype=bool)
    _draw_polyline(river, grid, [(72.892, 19.130), (72.884, 19.108), (72.872, 19.100),
                                 (72.858, 19.082), (72.856, 19.066), (72.849, 19.048),
                                 (72.840, 19.040)])
    river_w = ndimage.binary_dilation(river, iterations=3)
    fringe = ndimage.binary_dilation(river, iterations=9) & ~river_w
    lc[fringe & (rows > n * 0.45) & (lc != CLS["airport"])] = CLS["mangrove"]

    paint(MANGROVE_POLYS, CLS["mangrove"])
    for p in WATER_POLYS:
        lc[_poly_mask(grid, p, cols, rows)] = CLS["water"]
    lc[river_w] = CLS["water"]
    water = lc == CLS["water"]

    # Minor road network: warped grid in built-up areas
    warp_x = (_value_noise((n, n), 300, rng, 2) - 0.5) * 60
    warp_y = (_value_noise((n, n), 300, rng, 2) - 0.5) * 60
    spacing = 32.0  # px -> ~320 m blocks
    gx = np.abs(((cols + warp_x) % spacing) - spacing / 2)
    gy = np.abs(((rows + warp_y) % spacing) - spacing / 2)
    d_minor_px = spacing / 2 - np.maximum(gx, gy)  # distance to nearest grid line
    built = np.isin(lc, [CLS["residential"], CLS["dense_urban"], CLS["commercial"],
                         CLS["industrial"]])
    minor = (d_minor_px < 1.0) & built
    lc[minor] = CLS["minor_road"]

    # Major roads with traffic weights; distance field via EDT
    road_mask = np.zeros((n, n), dtype=bool)
    road_w = np.zeros((n, n), dtype=np.float32)
    for _name, wgt, line in ROADS:
        m = np.zeros((n, n), dtype=bool)
        _draw_polyline(m, grid, line)
        road_mask |= m
        road_w[m] = np.maximum(road_w[m], wgt)
    dist_px, (ir, ic) = ndimage.distance_transform_edt(~road_mask, return_indices=True)
    dist_major_m = dist_px * FINE_RES_M
    major_w = road_w[ir, ic]
    lc[(dist_px <= 2.0) & ~water] = CLS["major_road"]

    # ---------------------------------------------------------------- bands
    bands = SPECTRA[lc].transpose(2, 0, 1).copy()  # (4, H, W)
    # urban greenery: mix vegetation into built-up pixels
    veg_frac = 0.35 * _value_noise((n, n), 25, rng, 3) ** 1.6
    veg_frac[~np.isin(lc, [CLS["residential"], CLS["dense_urban"], CLS["commercial"]])] = 0
    veg_frac[lc == CLS["dense_urban"]] *= 0.35
    grass = SPECTRA[CLS["grass"]][:, None, None]
    bands = bands * (1 - veg_frac) + grass * veg_frac
    # rooftop / building-scale texture (30-60 m blocks) and sensor noise
    tex = ndimage.zoom(rng.random((n // 4 + 1, n // 4 + 1)), 4, order=0)[:n, :n]
    tex = 0.88 + 0.24 * tex.astype(np.float32)
    tex[water] = 1.0
    bands *= tex
    bands *= (1 + rng.normal(0, 0.025, size=bands.shape)).astype(np.float32)
    bands = np.clip(bands, 0.001, 0.6).astype(np.float32)

    # --------------------------------------------------------------- sources
    def cls_f(name):
        return (lc == CLS[name]).astype(np.float32)

    near_major = np.exp(-dist_major_m / 70.0) * major_w
    near_minor = np.exp(-np.maximum(d_minor_px, 0) * FINE_RES_M / 30.0) * built
    vegetation = np.isin(lc, [CLS["forest"], CLS["mangrove"], CLS["grass"]]).astype(np.float32)
    refinery = ndimage.gaussian_filter(cls_f("refinery"), 3)
    stacks = np.zeros((n, n), dtype=bool)  # point stacks at the power plant / refinery
    for lon, lat in [(72.903, 18.988), (72.897, 19.006), (72.893, 19.012)]:
        c, r = grid.to_px(lon, lat)
        if 0 <= r < n and 0 <= c < n:
            stacks[int(r), int(c)] = True
    stack_field = ndimage.gaussian_filter(stacks.astype(np.float32), 6)
    stack_field /= stack_field.max() + 1e-12
    sources = {
        "major_road": near_major.astype(np.float32),
        "minor_road": (0.6 * near_minor).astype(np.float32),
        "refinery": np.maximum(refinery, stack_field).astype(np.float32),
        "industrial": ndimage.gaussian_filter(cls_f("industrial"), 2),
        "airport": ndimage.gaussian_filter((ap & rw).astype(np.float32), 2) * 0.8
                    + ap.astype(np.float32) * 0.3,
        "dump": ndimage.gaussian_filter(cls_f("bare_dump"), 3),
        "dense_urban": ndimage.gaussian_filter(cls_f("dense_urban"), 2),
        "commercial": ndimage.gaussian_filter(cls_f("commercial"), 2),
        "residential": ndimage.gaussian_filter(cls_f("residential"), 2),
        "vegetation": ndimage.gaussian_filter(vegetation, 4),
    }

    scene = Scene(grid=grid, landcover=lc, bands=bands, sources=sources, days=days)
    # Each simulated overpass has its own wind (sea breeze from the W/SW at
    # the ~13:30 local overpass time) and its own cloud cover.
    scene.winds = []
    for _d in range(days):
        ang = math.radians(rng.normal(20, 25))     # direction the plume travels, from east
        speed = rng.uniform(0.6, 1.4)
        scene.winds.append((math.cos(ang) * speed, math.sin(ang) * speed))
    for p in (pollutants or SOURCE_WEIGHTS.keys()):
        simulate_pollutant(scene, p, rng)
    return scene


def simulate_pollutant(scene: Scene, pollutant: str, rng=None):
    rng = rng or np.random.default_rng(11)
    n = scene.grid.width
    w = SOURCE_WEIGHTS[pollutant]
    bg, a_reg, a_loc = AMPLITUDES[pollutant]
    local = sum(w[k] * scene.sources[k] for k in w)
    local = ndimage.gaussian_filter(local, 2.0)  # ~20 m mixing
    # Regional dispersion on a 100 m grid (cheap), then upsample.
    lo = local.reshape(n // 10, 10, n // 10, 10).mean(axis=(1, 3))
    truths, coarses = [], []
    for d in range(scene.days):
        ux, uy = scene.winds[d]
        near = ndimage.gaussian_filter(lo, 4)                 # ~400 m
        plume = ndimage.shift(ndimage.gaussian_filter(lo, 9), (-uy * 8, ux * 8), mode="nearest")
        far = ndimage.gaussian_filter(lo, 22)                 # ~2 km
        reg = 0.45 * near + 0.35 * plume + 0.20 * far
        reg /= np.percentile(reg, 99) + 1e-9
        reg_hi = ndimage.zoom(reg, 10, order=1)[:n, :n]
        loc = local / (np.percentile(local, 99.5) + 1e-9)
        day_scale = rng.uniform(0.85, 1.15)
        field_ = bg + day_scale * (a_reg * reg_hi + a_loc * loc)
        if pollutant == "O3":   # downwind photochemical production offsets titration
            field_ = field_ + 1.5e-3 * ndimage.zoom(
                ndimage.shift(ndimage.gaussian_filter(lo, 15), (-uy * 20, ux * 20), mode="nearest")
                / (lo.max() + 1e-9), 10, order=1)[:n, :n]
        field_ = field_.astype(np.float32)
        truths.append(field_)
        coarses.append(observe(field_, rng, pollutant))
    scene.truth[pollutant] = np.stack(truths)
    scene.coarse[pollutant] = np.stack(coarses)


def observe(truth: np.ndarray, rng, pollutant: str, factor: int = FACTOR,
            noise: float = 0.03, cloud_frac: float = 0.12) -> np.ndarray:
    """Sentinel-5P-like observation: 1 km cell means + noise + cloud gaps."""
    h, w = truth.shape[0] // factor, truth.shape[1] // factor
    cells = truth[: h * factor, : w * factor].reshape(h, factor, w, factor).mean(axis=(1, 3))
    if pollutant == "O3":     # total column: small absolute retrieval noise
        cells = cells + rng.normal(0, 3e-4, size=cells.shape)
    else:                     # relative retrieval noise
        cells = cells * (1 + rng.normal(0, noise, size=cells.shape))
    # clustered cloud mask
    cloud = ndimage.gaussian_filter(rng.random((h, w)), 1.2)
    thr = np.quantile(cloud, 1 - cloud_frac)
    cells = cells.astype(np.float32)
    cells[cloud > thr] = np.nan
    return cells


def landcover_rgb(scene: Scene) -> np.ndarray:
    """True-colour composite (B4, B3, B2) as uint8 RGB, gamma stretched."""
    rgb = scene.bands[[2, 1, 0]].transpose(1, 2, 0)
    rgb = np.clip(rgb / 0.22, 0, 1) ** (1 / 1.6)
    return (rgb * 255).astype(np.uint8)
