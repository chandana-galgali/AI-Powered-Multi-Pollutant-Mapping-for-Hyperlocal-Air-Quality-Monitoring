Pretrained CNN weights live here as `<POLLUTANT>_cnn.json` (e.g. `NO2_cnn.json`).

Create them with live data, then commit them:

    cd backend
    python train.py --pollutant all        # dry seasons 2022-23 … 2024-25

The server picks them up automatically; without them, "CNN" requests train a
small network on the fly.
