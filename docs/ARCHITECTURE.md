# Architecture

```
                 ┌──────────────────────── Google Earth Engine ────────────────────────┐
                 │ Sentinel-2 L2A (B2,B3,B4,B8)   fallbacks: Landsat-8 → MODIS → median  │
                 │ Sentinel-5P L3: NO2 · SO2 · CO · HCHO · O3 · aerosol index            │
                 └──────────────┬───────────────────────────────────────┬───────────────┘
                                │ computePixels (NumPy), one aligned    │
                                │ 10 m / 1 km lattice                   │
                ┌───────────────▼───────────────────────────────────────▼──────────────┐
backend/        │ hyperlocal_aq                                                        │
                │  gee.py        fetch_stack(pollutant, date, bbox, res) → S2 bands,   │
                │                S5P overpasses (computePixels, SCL cloud mask)       │
                │  service.py    validation, area/resolution limits, background jobs, │
                │                caching, GeoTIFF + PNG packaging                     │
                │  core.py       features (NDVI, NDWI), IDW gap-fill, bilinear,        │
                │                block means, residual correction                     │
                │  timeseries.py seasonal decomposition gap-fill (long cloud gaps)     │
                │  models.py     LinearModel · MultiScaleLinearModel · ForestModel ·   │
                │                FCNModel (NumPy CNN, aggregation loss)                │
                │  models_keras.py  same CNN in TensorFlow for GPU training           │
                │  metrics.py    R², RMSE, ±10 %, hotspot-class accuracy, F1           │
                │  scene.py      simulated scene with known 10 m field (validation)   │
                │  render.py     colour ramps → PNG overlays, hotspot extraction       │
                │ app.py  Flask REST API (+ serves dashboard) · train.py · tests/      │
                └───────────────┬──────────────────────────────────────────────────────┘
                                │ POST /api/jobs → poll /api/jobs/<id> → PNG overlays, GeoTIFF
frontend/       React + Leaflet dashboard (pollutant / date / area / method → map, hotspots, downloads)
validation/     accuracy study against a known 10 m field
gee/            Code Editor scripts (quick-look downscaling, training-data export)
```

## Why the aggregation loss

There is no 10 m pollution label anywhere: Sentinel-5P only tells us 1 km averages.
So the CNN produces a 10 m map, the map is average-pooled 100 × 100 → 1 km, and
the loss compares that with Sentinel-5P over many overpasses. Because surface
features (roads, refineries, vegetation, water) change between 10 m pixels and
the pooled output must match every block, the network has to learn which
surface patterns carry pollution. Dilated 3×3 convolutions (dilation 1, 2, 4, 8)
give each pixel ~310 m of context at ~16 k parameters.

## Request flow

1. `POST /api/jobs` validates the request and picks the finest resolution that fits the memory budget
   (10 m up to ~160 km², otherwise 20/25/40 m).
2. A background thread downloads the Sentinel-5P daily grids (up to 20 overpasses in 60 days,
   one request) and a cloud-masked Sentinel-2 median composite (120 days, tiled requests).
3. The chosen method is fitted on all overpasses through the Sentinel-5P footprint (below),
   predicts the requested day at 10 m and is matched to that day's observation. A pretrained CNN
   (`backend/models/*_cnn.json`) is used with its stored normalisation; otherwise a small CNN is
   trained on the request.
4. The result (PNG overlays, stats, hotspots, GeoTIFFs) is cached; identical requests are instant.

## The Sentinel-5P footprint (important)

Earth Engine serves Sentinel-5P on a ~1.1 km grid, but the instrument's ground pixel is about
5.5 km × 3.5 km. Each "1 km" value is really an average over a ~5 × 4 km footprint, so
neighbouring cells largely repeat the same measurement. Treating cells as independent 1 km truths
has two bad effects:
* models learn from differences between neighbouring cells that are mostly noise;
* forcing every 1 km block of the 10 m map to equal a blurred value erases real 1–5 km detail.

So the system models what the satellite actually sees, `footprint(blockmean(map))`, with a 5 × 4-cell
box (`config.S5P_FOOTPRINT_CELLS`):
* regressions are fitted on footprint-smoothed predictors;
* the "CNN" method is a **hybrid**: the multi-scale regression gives the base pattern and a CNN learns a
  correction. It is trained on **whole areas** (so the footprint always fits inside what the network
  sees) against the **season-average anomaly map** (each overpass standardised, then averaged over ~100
  days), which cuts day-to-day noise ~10× (`models.fit_hybrid_cnn`);
* each day's map = today's level + today's spatial spread × pattern strength (fitted once per season, not
  per noisy day) × the model's standardised pattern, plus the remaining residual smoothed over ~2 km so
  today's wind-driven gradients from the satellite are kept (`core.footprint_correct`);
* over open water (Sentinel-2 NDWI > 0.05) only the smooth (~300 m) part of the pattern is kept, so waves
  and glint are not read as pollution; values are floored at zero for gases.

### What real data showed (NO₂, Mumbai, dry season 2024-25)

Trained on the first ~80 days of each of 6 areas and scored on the **last 30 days (never seen)**: mean
daily correlation between the map, seen through the footprint, and Sentinel-5P:

| Pattern | r on unseen days |
|---|---|
| Regression on raw 1 km cells (old approach) | 0.15 |
| Multi-scale regression, footprint-aware | 0.20 |
| CNN trained on 5 km windows | ≈ 0.00 (failed: no signal inside one footprint) |
| **Hybrid regression + CNN, whole areas** | **0.24** |
| Satellite's own average pattern, upsampled (no model) | 0.25 |

At the satellite's own ~5 km scale, "what the satellite usually shows here" (persistent winds, plumes)
is as good as any surface-based model. That's why each day's map takes its large-scale pattern from the
satellite itself, and the models only supply the 10 m detail. That detail is judged against independent
NASA airborne NO₂ column maps and by the up-scaling test (README, *Validation*; docs/VALIDATION.md). Ground
stations measure a different quantity (surface air, not the column) and turned out not to be usable for this.

In a simulation with a realistic footprint (5 × 4 km smoothing, day-to-day swings of ×0.3–2.5, 15 %
noise), this raised hotspot-class accuracy from 0.52 (interpolation) to 0.70 (multi-scale regression)
and 0.71–0.73 (CNN), and R² from 0.24 to ~0.75. The strict block correction scored below
interpolation in the same test.

The strict per-block correction (`core.residual_correct`, `pred + bilinear(obs − blockmean(pred))`
iterated) is still used where cells really are block means: the simulated validation scene and the
up-scaling test.

## Bugs fixed from the hackathon notebooks

| Notebook | Issue | Fix |
|---|---|---|
| Backend.ipynb (CNN) | 4 band values of one pixel reshaped into a 2×2 "image": no spatial context | Fully-convolutional net on real 10 m neighbourhoods |
| Backend.ipynb | `scaler.fit_transform(target)` re-fitted the predictor scaler, so the saved mean/scale were the target's | Separate predictor and target normalisation |
| Backend / UNet / RF | Prediction = one constant per tile (16 tiles), so not actually 10 m | Per-pixel prediction + residual correction |
| GEE Codes.docx | `NO2_column_number_density` band with `L3_NO2`; no intercept; S2 DN not scaled | Tropospheric band, intercept, reflectance scaling |
| NO2_Prediction.ipynb | 27 images, 5 hand-typed labels; accuracy computed on training data | Labels come from S5P; evaluation against held-out 10 m truth (OSSE) |
| SIH_Backend.ipynb | `addBands(sentinel1)` with an ImageCollection → export failed | Not needed (S1 dropped); computePixels path |
