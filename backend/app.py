"""Flask REST API for hyperlocal multi-pollutant downscaling with live Sentinel data.

Run locally:   earthengine authenticate   (once)
               GEE_PROJECT=<your-ee-project> python app.py
Production:    see docs/DEPLOY.md (service-account key in GEE_SERVICE_ACCOUNT_JSON)

Endpoints
  GET  /api/health                     server + Earth Engine status
  GET  /api/pollutants                 pollutant catalogue
  GET  /api/methods                    methods (+ which pollutants have pretrained CNN weights)
  GET  /api/areas                      named areas of Mumbai and limits
  POST /api/jobs                       start a downscaling job
       body: {"pollutant": "NO2", "method": "mslr", "date": "2025-01-15",
              "area": "chembur_trombay"}     or   "bbox": [w, s, e, n]  (+ optional "res_m": 10|20|25)
  GET  /api/jobs/<id>                  progress / result (map overlays, stats, hotspots, provenance)
  GET  /api/jobs/<id>/geotiff          10 m result as GeoTIFF   (…/coarse_geotiff for the 1 km input)
  GET  /api/validation                 accuracy: simulation, ground stations, up-scaling test
"""
from __future__ import annotations

import json
import os

from flask import Flask, Response, jsonify, request, send_from_directory

from hyperlocal_aq import gee, service
from hyperlocal_aq.validation import clean_json
from hyperlocal_aq.config import POLLUTANTS

STATIC = os.environ.get("FRONTEND_DIST", os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "..", "validation")
app = Flask(__name__, static_folder=None)
jobs = service.Jobs()


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.get("/api/health")
def health():
    return jsonify(status="ok", earth_engine=gee.status(), version="2.0.0")


@app.get("/api/pollutants")
def pollutants():
    return jsonify({k: {kk: v[kk] for kk in ("label", "unit", "band", "collection", "start_date")}
                    for k, v in POLLUTANTS.items()})


@app.get("/api/methods")
def methods():
    return jsonify(service.available_methods())


@app.get("/api/areas")
def areas():
    return jsonify(areas=service.AREAS, max_pixels=service.MAX_PIXELS,
                   max_area_km2={r: round(service.max_area_km2(r)) for r in service.RESOLUTIONS})


@app.post("/api/jobs")
def create_job():
    try:
        params = service.validate(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    st = gee.status()
    if not st["connected"]:
        return jsonify(error="The server can't reach Google Earth Engine: " + st.get("error", "")), 503
    return jsonify(jobs.submit(params)), 202


@app.get("/api/jobs/<jid>")
def get_job(jid):
    j = jobs.public(jid)
    return (jsonify(clean_json(j)), 200) if j else (jsonify(error="Job not found (the server may have restarted)."), 404)


@app.get("/api/jobs/<jid>/<name>")
def job_file(jid, name):
    if name not in ("geotiff", "coarse_geotiff"):
        return jsonify(error="unknown file"), 404
    data = jobs.file(jid, name)
    if data is None:
        return jsonify(error="Result not ready or expired."), 404
    p = jobs.jobs[jid]["params"]
    fn = f"{p['pollutant']}_{p['method']}_{p['date']}_{'1km' if name == 'coarse_geotiff' else str(p['res_m']) + 'm'}.tif"
    return Response(data, mimetype="image/tiff", headers={"Content-Disposition": f"attachment; filename={fn}"})


@app.get("/api/validation")
def validation():
    """simulation: known 10 m field · ground: CPCB/MPCB stations · upscaling: real S5P 4 km → 1 km."""
    out = {"simulation": None, "ground": {}, "upscaling": {}}
    sim = os.path.join(VALIDATION_DIR, "simulation", "metrics.json")
    if os.path.exists(sim):
        with open(sim) as f:
            out["simulation"] = json.load(f)
    res = os.path.join(VALIDATION_DIR, "results")
    for name in sorted(os.listdir(res)) if os.path.isdir(res) else []:
        if name.endswith(".json") and not name.endswith("_pairs.json"):
            kind, _, pol = name[:-5].partition("_")
            if kind in ("ground", "upscaling"):
                with open(os.path.join(res, name)) as f:
                    out[kind][pol] = json.load(f)
    return jsonify(clean_json(out))


# Serve the built React app (frontend/dist)
@app.get("/")
@app.get("/<path:path>")
def spa(path="index.html"):
    if path.startswith("api/"):
        return jsonify(error="not found"), 404
    if os.path.isdir(STATIC):
        full = os.path.join(STATIC, path)
        return send_from_directory(STATIC, path if os.path.isfile(full) else "index.html")
    return jsonify(message="API is running. Build the frontend (cd frontend && npm run build) "
                           "or run `npm run dev` on port 5173."), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
