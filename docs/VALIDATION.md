# Validation

**Training period:** dry seasons 2022-23, 2023-24, 2024-25 (`backend/train.py` default).
**Validation period:** dry season 2025-26, which is never used in training. See the README section
"Training and validation period" for the reasons (single Sentinel-5P processor version, no COVID-19
lockdown, no monsoon cloud, held-out test season).

Four complementary checks. None needs any sensors to be built.

| Check | Data | Answers |
|---|---|---|
| 1. Simulated scene | known 10 m field | Can the method recover 10 m detail at all? |
| 2. Up-scaling test | real Sentinel-5P + Sentinel-2 | Does the learned surface relationship add real detail? |
| 3. Ground stations | CPCB / MPCB monitors (via OpenAQ or CSV) | Does the map agree with surface readings? (different quantity, see the findings) |
| **4. Airborne column maps** | NASA GCAS NO₂ columns at 250 m (LISTOS 2018, New York) | **Does the map match an independent high-resolution measurement of the same quantity?** |

**Why check 4 is the reference.** Sentinel-5P measures the column, i.e. all the gas above a point; stations measure the
air at nose height. A column map can only be checked properly against a column measurement at higher
resolution. Only aircraft imaging spectrometers provide one over a whole city. None has flown over Mumbai, so
check 4 runs the system unchanged on New York, where NASA flew during Sentinel-5P overpasses.

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

**Findings (2025-26 season).** In every pollutant tested, the median daily correlation across stations is
about 0.1 or lower, *including for the raw satellite*, and skill against the raw satellite is about 0 for every
method (numbers: README, *Validation*, check 4). Co-located stations often disagree with each other.
Conclusion: surface stations cannot score a column map in Mumbai, which is why check 4 exists. Caveat about
the files: the CO AQI-band accuracy of 100 % is meaningless because every reading is in the *Good* band.

## 4. Airborne column maps: `python validation/airborne/airborne.py compare`

**Data.** NASA LISTOS 2018, *LISTOS NASA Aircraft Remote Sensing Data*
(DOI 10.5067/Suborbital/LISTOS/AircraftRemoteSensing_NASAAircraft_Data_1, Earthdata collection
`C3880706941-LARC_CLOUD`). Download with a free NASA Earthdata account. The afternoon GCAS NO₂ files
(`LISTOS-GCAS-NO2_NASA-AIRCRAFT_2018MMDD_R0_PM_250MX250M.h5`, 0.5–0.8 GB each) go in
`validation/airborne/listos2018/`, which git ignores. <!-- AUTO:airborne_flights -->
Used so far: 2 Jul, 19 Jul, 20 Jul, 6 Aug, 15 Aug, 16 Aug, 24 Aug, 28 Aug, 29 Aug, 6 Sep 2018. 5 Aug had no Sentinel-5P overpass in Earth Engine.
<!-- /AUTO:airborne_flights -->

**Steps.**
1. `airborne.py extract` reduces each HDF5 file to a small `.npz` with these fields: latitude, longitude, time, cloud flag and
   `NO2_VCOLBELOWAIRCRAFT`. That is the column below the ~9 km flight altitude, effectively the tropospheric column. The
   default is the WRF-Chem 4 km a-priori version (`--column 1`); NASA's README notes the NAM-CMAQ version
   under-estimates free-troposphere NO₂. The two versions correlate at r = 0.97–1.0. The reader `h5read.py` falls
   back to pure-Python `pyfive` where compiled HDF5 libraries are blocked.
2. Keep pixels flagged cloud-free and within ±1.5 h of the ~17:45 UTC overpass.
3. Pick up to 6 non-overlapping 8 × 8 km tiles per flight, each with ≥ 60 % aircraft coverage at 250 m.
4. Run the live pipeline at 10 m on each tile for the flight date. It uses Sentinel-5P OFFL, a Sentinel-2 L2A composite and the
   2018 Sentinel-5P footprint (7 × 3.5 km, i.e. 7 × 4 cells; the pixel shrank to 5.5 × 3.5 km in August 2019). The CNN is
   the Mumbai-trained model.
5. Average each map to the 250 m grid and compare with the aircraft grid of the same tile.

The aircraft data is used **only for scoring**. Nothing is trained or tuned on it.

| Metric | Meaning |
|---|---|
| `r_250m` | spatial correlation with the aircraft map at 250 m |
| `r_1km` | the same after averaging both to 1 km |
| `r_detail_sub1km` | correlation of each 250 m cell's deviation from its 1 km block mean. This is the detail downscaling must add; it is undefined for the raw 1 km satellite |
| `spread_ratio` | map contrast / aircraft contrast (std) |
| `mean_ratio` | map level / aircraft level |
| `*_wins_vs_bilinear`, `*_sign_test_p` | tiles where a method beats interpolation, and a one-sided sign test |

`summary` covers all tiles. `summary_resolved` covers only tiles where the raw satellite itself tracks the aircraft
(r ≥ 0.3); that selection is independent of the downscaling methods.

<!-- AUTO:airborne_doc -->
**Results** (10 flights, 60 tiles):

| | All 60 tiles: r 250 m / 1 km | 30 resolved tiles: r 250 m / 1 km / detail | Beats interpolation (all · resolved) |
|---|---|---|---|
| Sentinel-5P as is (1 km) | 0.28 / 0.48 | 0.55 / 0.67 / – | – |
| Interpolation (bilinear) | 0.29 / 0.50 | 0.57 / 0.70 / 0.09 | – |
| Linear regression | 0.30 / 0.51 | 0.45 / 0.65 / 0.03 | 28/60 · 7/30 |
| Multi-scale regression | 0.27 / 0.49 | 0.46 / 0.63 / 0.02 | 24/60 · 4/30 |
| **Hybrid regression + CNN (Mumbai-trained)** | 0.28 / 0.53 | 0.53 / 0.73 / 0.06 | 33/60 · 13/30 |

Sign tests (CNN vs interpolation): 250 m p = 0.26 (all), 0.82 (resolved); sub-1 km detail 33/60, p = 0.26.
<!-- /AUTO:airborne_doc -->

**Limits.**
- The test covers one city (New York) and one pollutant (NO₂), with an older Sentinel-5P processor (v1.2).
- On the 7 low-NO₂ tiles, where columns are close to instrument noise, no method finds a pattern.
- The maps are smoother than the aircraft (spread ratio 0.4–0.65), and sub-1 km detail is weak.
- NASA notes that parks and bright buildings can leave small artifacts in the aircraft retrieval. Our model also uses surface
  brightness, so part of the agreement at the finest scale may be shared artifact.

`python validation/airborne/airborne.py summary` re-prints the saved results without Earth Engine.

