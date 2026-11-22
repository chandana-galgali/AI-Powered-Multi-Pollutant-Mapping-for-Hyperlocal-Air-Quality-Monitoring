# Validation

**Training period:** dry seasons 2022-23, 2023-24, 2024-25 (`backend/train.py` default).
**Validation period:** dry season 2025-26, which is never used in training. See the README section
"Training and validation period" for the reasons (single Sentinel-5P processor version, no COVID-19
lockdown, no monsoon cloud, held-out test season).

Three complementary checks, from most controlled to most real. All three run without building any sensors.

| Check | Data | Answers |
|---|---|---|
| 1. Simulated scene | known 10 m field | Can the method recover 10 m detail at all? |
| 2. Up-scaling test | real Sentinel-5P + Sentinel-2 | Does the learned surface relationship add real detail? |
| 3. Ground stations | CPCB / MPCB monitors (via OpenAQ or CSV) | Does the 10 m map agree with what people breathe better than the raw satellite? |

## 1. Simulated scene: `python validation/run_validation.py`

A 14 × 14 km stretch of east-central Mumbai with a known 10 m pollution field, observed the way
Sentinel-5P would see it (1 km means, 3 % noise, ~12 % cloud gaps, 8 overpasses). The methods see only
the 1 km observations. Metrics: hotspot-class accuracy, hotspot F1, R², RMSE. Results are in `validation/simulation/metrics.json`.

## 2. Up-scaling test: `python validation/validate.py upscaling --pollutant all --end 2026-02-28`

Needs only Earth Engine access. For up to 30 overpasses over Mumbai:

1. Take real Sentinel-5P at 1 km, and Sentinel-2 (fetched at 100 m) averaged to 1 km.
2. Average Sentinel-5P to 4 km (`--k 4`); blocks under 50 % cloud-free are dropped.
3. Downscale 4 km → 1 km with each method, exactly as the live system does 1 km → 10 m.
4. Compare with the real 1 km values.

| Metric | Meaning |
|---|---|
| `spatial_r_median` | correlation with the real 1 km field (median over days) |
| `detail_r_median` | correlation of the deviations from each 4 km block mean. This is the only part the method had to supply, so it's the honest measure of added detail |
| `nrmse` | RMSE / mean |
| `skill_vs_bilinear` | 1 − RMSE / RMSE(interpolation) |

Assumption: the relation learned between 4 km and 1 km also holds between 1 km and 10 m. This
"scale invariance" is standard in downscaling studies but isn't guaranteed, which is why check 3 exists.

## 3. Ground stations: `python validation/validate.py ground --pollutant NO2 --start 2025-11-01 --end 2026-02-28`

**Data.** OpenAQ redistributes Mumbai's continuous monitors (CPCB / MPCB). Get a free API key at
https://explore.openaq.org and `export OPENAQ_API_KEY=...`. Alternatively pass `--csv file.csv`
(format documented in `backend/hyperlocal_aq/ground.py`), e.g. converted from the CPCB CCR portal.

**Matching.**
- *Time:* Sentinel-5P crosses Mumbai around 13:30 local solar time, so station hourly means from
  13:00–15:00 IST are averaged per day.
- *Place:* each station gets its own 6 × 6 km tile, downscaled at 10 m for every overpass. Values are
  sampled at the station as a 3 × 3-pixel (30 m) mean.
- *Fairness:* only days when the satellite actually observed the station's 1 km cell (no cloud) are
  used, and every predictor is scored on the same station-days.
- *Units:* ppb/ppm/mg/m³ are converted to µg/m³ at 25 °C.

**Predictors compared:** the raw Sentinel-5P 1 km cell value (baseline), interpolation, and each method.

**Why calibration is needed.** Sentinel-5P measures the *column* (mol/m², all the gas above a point);
stations measure *surface concentration* (µg/m³). Correlations need no conversion. For values in
µg/m³, a linear calibration is fitted on all stations except one and tested on the one left out,
repeated for every station (leave-one-station-out). Nothing is scored on data it was calibrated on.

| Metric | Meaning |
|---|---|
| `daily_spatial_r_median`, `daily_spatial_spearman_median` | across stations on the same day: does the map get the *pattern* across the city right? The core metric for downscaling |
| `pooled_r` | all station-days together |
| `temporal_r_median` | per station across days: does the map follow day-to-day changes? |
| `rmse_ug_m3`, `mae_ug_m3`, `bias_ug_m3`, `normalised_mean_bias`, `r2_calibrated` | calibrated error on held-out stations |
| `skill_vs_raw_1km` | **1 − RMSE / RMSE(raw 1 km)**. Above 0 means the 10 m map beats the satellite as it is. This is the headline number |
| `aqi_category_accuracy` | share of readings in the correct CPCB AQI band (breakpoints are defined on 8/24 h means and applied here to overpass-time means) |
| `hotspot_f1_top_quartile` | each day, do the top-25 % stations by map match the top-25 % by measurement? |

**Pollutant notes.** HCHO isn't monitored at ground stations, so it's covered only by checks 1–2. The aerosol index is compared
with PM2.5 as a proxy (flagged in the output). It's a different quantity, so expect weaker agreement.

**Limits to keep in mind.**
- Mumbai has a few dozen stations at most, so daily spatial r rests on small samples. That's why the median over many days is reported, and why a dry-season period (Nov–May) with many cloud-free overpasses gives the most reliable numbers.
- Stations often sit near roads or in compounds; a 30 m sample is the closest the map can get.
- A column-to-surface relationship that varies with boundary-layer height adds scatter the method can't remove.

Results go to `validation/results/` (`ground_<POL>.json`, plus `ground_<POL>_pairs.json` with every
matched reading), are served by `GET /api/validation`, and appear in the dashboard's
"How accurate is this?" panel. Commit the result files so the deployed site shows them.
