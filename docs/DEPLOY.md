# Deploying the live system

The app is one Docker container: Flask serves the API and the built React dashboard. It needs:

1. **An Earth Engine service account key**, so the server can read Sentinel data without your personal login.
2. **A host that runs Docker.** Render is set up below (`render.yaml`); Railway, Google Cloud Run, Fly.io or any VM work with the same `Dockerfile`.

## 1. Earth Engine service account (≈5 minutes, once)

First create a Google Cloud project and register it for noncommercial Earth Engine use
(https://console.cloud.google.com/earth-engine). Its **project ID** is what `GEE_PROJECT` refers to below.

1. Open Google Cloud Console → **IAM & Admin → Service Accounts**, with your project selected.
2. **Create service account**, e.g. name `aq-server`.
3. Grant it these roles:
   - **Earth Engine Resource Viewer** (`roles/earthengine.viewer`)
   - **Service Usage Consumer** (`roles/serviceusage.serviceUsageConsumer`)
4. Open the account → **Keys → Add key → Create new key → JSON**. A `.json` file downloads.
   **Never commit this file.** `.gitignore` already blocks `secrets/` and `*.json.key`.
5. Make sure the **Earth Engine API** is enabled for the project (APIs & Services → Library → "Google Earth Engine API").

Test it locally before deploying:

```bash
cd backend
pip install -r requirements.txt
export GEE_PROJECT=<your-project-id>
export GEE_SERVICE_ACCOUNT_JSON="$(cat /path/to/key.json)"     # PowerShell: $env:GEE_SERVICE_ACCOUNT_JSON = Get-Content key.json -Raw
python app.py
# another terminal:
curl http://localhost:5000/api/health        # → "earth_engine": {"connected": true}
```

## 2. Deploy on Render

1. Push the repository to GitHub.
2. Render dashboard → **New → Blueprint** → choose the repository. Render reads `render.yaml`.
3. When asked, fill in the two secret variables:
   - `GEE_SERVICE_ACCOUNT_JSON`: paste the **entire** contents of the key file
   - `GEE_PROJECT`: `<your-project-id>`
4. Deploy. The first build takes ~5 minutes (it installs Node + Python and builds React).
5. Open `https://<service-name>.onrender.com/api/health`: `earth_engine.connected` must be `true`.
   Then open the root URL. That's the live dashboard.

**Plan / memory.** `render.yaml` asks for *Standard* (2 GB). The free and Starter plans have 512 MB;
they work if you set `MAX_PIXELS=900000`, which lowers the area per request (≈90 km² at 10 m, and the
whole-city map drops to 40 m). The free plan also sleeps after 15 minutes idle, so the first request
after that waits ~1 minute.

### Other hosts

| Host | How |
|---|---|
| Railway | New project → Deploy from GitHub repo (detects `Dockerfile`) → add the two variables. |
| Google Cloud Run | `gcloud run deploy aq --source . --region asia-south1 --memory 2Gi --set-env-vars GEE_PROJECT=<your-project-id> --set-secrets GEE_SERVICE_ACCOUNT_JSON=<secret>:latest` |
| Any VM | `docker build -t aq . && docker run -p 80:10000 -e GEE_PROJECT=... -e GEE_SERVICE_ACCOUNT_JSON="$(cat key.json)" aq` |

## 3. Pretrained CNN weights (recommended)

Without weights, choosing *CNN* trains a small network on each request (slow on a small server).
Train once on real data and commit the weights:

```bash
cd backend
python train.py --pollutant all        # dry seasons 2022-23, 2023-24, 2024-25; can take several hours
git add models/*_cnn.json
```

Training downloads ~450 km² of Mumbai at 10 m for each dry season (Nov–Feb) of 2022-23 … 2024-25
and every usable Sentinel-5P overpass. It needs ~2 GB RAM. Change the period with `--seasons`.
For GPU training, use `hyperlocal_aq/models_keras.py` (same architecture and JSON format).

## 4. Validation results (recommended)

Run the real-data validation once and commit `validation/results/` so the deployed dashboard shows it
(see `docs/VALIDATION.md`):

```bash
export OPENAQ_API_KEY=...        # free key from https://explore.openaq.org
python validation/validate.py ground --pollutant all --start 2025-11-01 --end 2026-02-28
python validation/validate.py upscaling --pollutant all --end 2026-02-28
git add validation/results
```

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GEE_SERVICE_ACCOUNT_JSON` | – | full JSON key (production) |
| `GEE_SERVICE_ACCOUNT`, `GEE_KEY_FILE` | – | alternative: email + path to key file |
| `GEE_PROJECT` | `<your-project-id>` | Earth Engine Cloud project |
| `MAX_PIXELS` | 1 600 000 | largest output map per request (memory guard) |
| `ONLINE_CNN_STEPS` | 120 | training steps when no pretrained CNN exists |
| `JOB_WORKERS` | 1 | maps computed in parallel |
| `CORS_ORIGIN` | `*` | if the frontend is hosted elsewhere |
