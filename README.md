# 🌍 AI-Powered Multi-Pollutant Mapping for Hyperlocal Air Quality Monitoring

> **1st Place & ₹1.5L Grant Winner – Climate Hackathon 2024**  
> *Issued by the U.S. Consulate General Mumbai & Ekonnect Knowledge Foundation*

![Spatial Resolution](https://img.shields.io/badge/Resolution_Enhancement-1000m_→_10m-brightgreen)
![Pollutants](https://img.shields.io/badge/Pollutants-NO₂_SO₂_CO_HCHO_O₃_Aerosol-blue)
![Domain](https://img.shields.io/badge/Domain-Geospatial_AI_%7C_Remote_Sensing-orange)

---

## 📌 Project Overview

Urban air pollution monitoring often suffers from a critical data scarcity issue: satellite instruments like **Sentinel-5P** provide essential atmospheric pollution data, but at a coarse resolution (~1000m), making it difficult to pinpoint neighborhood-level hotspots.

This project implements a **domain-agnostic "Super-Resolution" AI pipeline** that fuses coarse-resolution Sentinel-5P atmospheric data with high-resolution **Sentinel-2** surface imagery. Powered by **Convolutional Neural Networks (CNNs)** and statistical correlation techniques, the system downscales pollution maps to **10m spatial resolution**, enabling data-driven, hyperlocal environmental action.

**How it works:**
1. Sentinel-5P gives the concentration of each pollutant in ~1 km blocks.
2. Sentinel-2 gives the surface at 10 m: blue, green, red and near-infrared bands, plus NDVI (vegetation) and NDWI (water).
3. The model learns how each pollutant correlates with those surface signals across the 1 km blocks of many overpasses.
4. It applies that relationship to every 10 m pixel.
5. A final step matches the 10 m map to that day's satellite reading as the satellite actually sees it. Sentinel-5P's real
   ground pixel is ~5.5 × 3.5 km, so the map, averaged over that footprint, must reproduce the observation
   (details: docs/ARCHITECTURE.md).

**Live system:** pick a pollutant, a date and an area of Mumbai, and the server fetches the real Sentinel data from Google Earth Engine and returns the 10 m map, hotspots and a GeoTIFF. Add the deployed URL here.

---

## ⚡ Key Features

* **100x Resolution Enhancement:** 1 km Sentinel-5P → 10 m maps (25–40 m for the whole city in one go), consistent with the satellite at 1 km.
* **Live Multi-Source Data Fusion:** Sentinel-5P (TROPOMI, OFFL with NRTI fallback) and cloud-masked Sentinel-2 L2A composites, pulled on demand from Google Earth Engine on one aligned 10 m / 1 km grid; Landsat-8 fallback.
* **Six Pollutants:** NO₂, SO₂, CO, HCHO, O₃ and the UV aerosol index.
* **Model Choice:** linear regression on Sentinel-2 bands (the original method), multi-scale regression, random forest, and a **hybrid regression + CNN** whose CNN is trained with an aggregation loss through the real Sentinel-5P footprint (~5.5 × 3.5 km).
* **Cloud-Gap Handling:** inverse-distance filling of cloudy blocks; seasonal decomposition for long gaps.
* **Actionable Output:** ranked 10 m hotspots and downloadable GeoTIFFs for GIS tools. Named areas include the Somaiya Vidyavihar campus (*“Towards Green Campuses”*).
* **Extensible & Domain-Agnostic Architecture:** the same downscaling core can be retrained for crop health, flood risk or urban heat islands.

---

## 📅 Training and validation period

| | Period |
|---|---|
| **Training** | dry seasons (1 Nov – end of Feb) **2022-23, 2023-24, 2024-25** |
| **Validation (held out)** | dry season **2025-26** (1 Nov 2025 – 28 Feb 2026) |

Why this window:
1. **One data version.** The Sentinel-5P NO₂ processor was upgraded several times; version changes raised
   tropospheric NO₂ by ~38 % (v1.4) and ~14 % (v2.2), and mixing versions creates artificial jumps
   (Zhang et al., EGUsphere 2023; van Geffen et al., AMT 2022). From November 2022 on, data come from the
   current processor (v2.4, released July 2022).
2. **No COVID-19 distortion.** Mumbai's NO₂ fell by ~40–50 % during the 2020 lockdown (ESA, April 2020);
   2020–21 would be an outlier, not today's city.
3. **Dry season only.** Monsoon cloud (Jun–Sep) leaves few usable overpasses and blurs Sentinel-2
   composites; November–February has the clearest skies and the strongest, most stable spatial
   pollution patterns (winter inversions trap emissions near their sources).
4. **Recent and multi-year.** Three seasons give enough overpasses while reflecting today's roads, construction and industry.
5. **Honest test.** The 2025-26 season is never used in training, so validation on it is out-of-sample.

Sentinel-5P has only flown since late 2017, so earlier years don't exist for this sensor. Maps can be made for any date since
2018, but they are most reliable for dry-season dates; monsoon maps rest on few cloud-free overpasses.

---

## 📊 Validation

Three checks, all from publicly available data (details in **[docs/VALIDATION.md](docs/VALIDATION.md)**):

1. **Ground stations:** the 10 m maps are compared with Mumbai's CPCB/MPCB monitors (via OpenAQ) at the
   Sentinel-5P overpass time. Reported: spatial correlation across stations, calibrated RMSE in µg/m³
   (leave-one-station-out), **skill score vs. the raw 1 km satellite value**, CPCB AQI-band accuracy and hotspot F1.
   `python validation/validate.py ground --pollutant NO2 --start 2025-11-01 --end 2026-02-28`
2. **Up-scaling test on real data:** real Sentinel-5P is averaged to 4 km, downscaled back to 1 km and compared
   with the real 1 km values. This measures added detail without any ground data.
   `python validation/validate.py upscaling --pollutant all --end 2026-02-28`
3. **Simulated scene with a known 10 m field:** checks whether a method can recover 10 m detail at all.
   Hotspot-class accuracy (share of pixels in the right class):

| Pollutant | Interpolation | Linear regression (original) | Multi-scale regression | **CNN** |
|---|---|---|---|---|
| NO₂    | 70.3 % | 69.1 % | 76.1 % | **76.2 %** |
| SO₂    | 75.9 % | 45.0 % | 51.9 % | **77.3 %** |
| CO     | 63.2 % | 63.1 % | 65.0 % | **67.1 %** |
| HCHO   | 75.3 % | 64.1 % | 72.7 % | **78.9 %** |
| O₃     | 67.6 % | 64.1 % | 66.6 % | **68.2 %** |
| Aerosol index | 66.9 % | 72.7 % | **78.7 %** | 76.9 % |

These are for an idealised satellite whose cells are true 1 km averages. On real Sentinel-5P data, see
"What real data showed" in docs/ARCHITECTURE.md. With a realistic ~5 × 4 km Sentinel-5P
footprint (the system's default for real data), the footprint-aware methods reach 70–73 % versus 52 % for
interpolation (R² ≈ 0.75 vs 0.24); see docs/ARCHITECTURE.md.

Results of 1 and 2 appear in the dashboard's *How accurate is this?* panel once the scripts have been run.

---

## 🛠️ Tech Stack

* **Core & ML:** Python, NumPy, SciPy, scikit-learn, CNN (NumPy implementation + TensorFlow/Keras for GPU training)
* **Geospatial & Remote Sensing:** Google Earth Engine (GEE) API, Sentinel-5P, Sentinel-2, Landsat-8
* **Backend & API:** Flask, REST APIs, background jobs, GeoTIFF export
* **Frontend Dashboard:** React.js, Leaflet
* **Deployment:** Docker, Render (blueprint included)

---

## 🗂️ Repository layout

```
backend/
  app.py              REST API (also serves the built dashboard)
  train.py            train the CNN on live Sentinel data → backend/models/
  hyperlocal_aq/      gee.py (data) · core.py · models.py · pipeline.py · service.py (jobs)
                      metrics.py · render.py · export.py (GeoTIFF) · timeseries.py
                      ground.py (OpenAQ/CSV stations) · validation.py (collocation, metrics) · scene.py
  tests/              pytest (Earth Engine mocked)
frontend/             React + Leaflet dashboard (Vite)
validation/           validate.py (ground stations, up-scaling test), run_validation.py (simulation), results/
gee/                  Earth Engine Code Editor scripts
notebooks/legacy/     original hackathon notebooks
docs/                 ARCHITECTURE.md · DEPLOY.md · VALIDATION.md
Dockerfile · render.yaml · docker-compose.yml
```

## 🚀 Run locally

```bash
# backend
cd backend
pip install -r requirements.txt
earthengine authenticate                 # once, with the Google account that owns the EE project
export GEE_PROJECT=<your-cloud-project-id>
python app.py                            # http://localhost:5000/api/health
pytest -q                                # 25 tests

# frontend (another terminal)
cd frontend
npm install
npm run dev                              # http://localhost:5173 (proxies /api to :5000)
```

Production deployment (Render, Railway, Cloud Run): see **[docs/DEPLOY.md](docs/DEPLOY.md)**.

### API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | server and Earth Engine status |
| GET | `/api/pollutants` · `/api/methods` · `/api/areas` | catalogues and limits |
| POST | `/api/jobs` | `{"pollutant":"NO2","method":"mslr","date":"2025-01-15","area":"chembur_trombay"}` or `"bbox":[w,s,e,n]` → job id |
| GET | `/api/jobs/<id>` | progress, then map overlays, stats, hotspots, data provenance |
| GET | `/api/jobs/<id>/geotiff` | 10 m result (`coarse_geotiff` for the 1 km input) |
| GET | `/api/validation` | simulation, ground-station and up-scaling results |

---