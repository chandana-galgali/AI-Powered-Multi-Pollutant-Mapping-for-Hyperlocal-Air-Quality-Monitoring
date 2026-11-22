"""Central configuration: pollutant catalogue, area of interest and grid settings."""

# Sentinel-5P (TROPOMI) Level-3 collections on Google Earth Engine.
# Band names follow the GEE catalogue (tropospheric NO2 is the product used for
# surface-relevant mapping; the old JS script used the total column by mistake).
POLLUTANTS = {
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
        "start_date": "2018-06-28",
        "unit": "mol/m²",
        "label": "Nitrogen dioxide (tropospheric column)",
    },
    "SO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band": "SO2_column_number_density",
        "start_date": "2018-07-10",
        "unit": "mol/m²",
        "label": "Sulphur dioxide",
    },
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "start_date": "2018-06-28",
        "unit": "mol/m²",
        "label": "Carbon monoxide",
    },
    "HCHO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_HCHO",
        "band": "tropospheric_HCHO_column_number_density",
        "start_date": "2018-10-02",
        "unit": "mol/m²",
        "label": "Formaldehyde (tropospheric column)",
    },
    "O3": {
        "collection": "COPERNICUS/S5P/OFFL/L3_O3",
        "band": "O3_column_number_density",
        "start_date": "2018-07-10",
        "unit": "mol/m²",
        "label": "Ozone (total column)",
    },
    "AER_AI": {
        "collection": "COPERNICUS/S5P/OFFL/L3_AER_AI",
        "band": "absorbing_aerosol_index",
        "start_date": "2018-07-04",
        "unit": "index",
        "label": "Absorbing aerosol index (UV)",
    },
}

# Sentinel-2 predictors (surface reflectance, harmonized collection).
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ["B2", "B3", "B4", "B8"]

# Fallback optical sources when no cloud-free Sentinel-2 scene is available
# (mirrors the fallback chain in the original LR notebook).
LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
MODIS_COLLECTION = "MODIS/061/MCD43A4"
MODIS_BANDS = [
    "Nadir_Reflectance_Band3",  # blue
    "Nadir_Reflectance_Band4",  # green
    "Nadir_Reflectance_Band1",  # red
    "Nadir_Reflectance_Band2",  # NIR
]

GEE_PROJECT = None   # set the GEE_PROJECT environment variable to your Cloud project ID
# Bounding box of the ward shapefile (from SIH_Backend.ipynb output).
MUMBAI_BBOX = [72.7794, 18.8937, 72.9747, 19.2683]

# Grid settings
# Sentinel-5P ground pixel is ~5.5 km (along track, ~N-S) x 3.5 km (across track) since
# Aug 2019. Earth Engine's L3 grid (~1.1 km) oversamples it, so each "1 km" cell is
# really a ~5 x 4 km footprint average. Models are fitted and corrected through this
# footprint (rows, cols in 1 km cells) instead of treating cells as independent.
S5P_FOOTPRINT_CELLS = (5, 4)

FINE_RES_M = 10       # output resolution (Sentinel-2)
COARSE_RES_M = 1000   # Sentinel-5P L3 grid on GEE (~1.1 km)
FACTOR = COARSE_RES_M // FINE_RES_M  # 100

# Validation scene (validation/run_validation.py): 14 km x 14 km of east-central Mumbai
# (Dharavi, BKC, Kurla, Chembur, Trombay/Mahul, Deonar, Ghatkopar, airport).
VALIDATION_ORIGIN = (72.835, 18.985)   # lon, lat of the south-west corner
VALIDATION_SIZE_M = 14000
