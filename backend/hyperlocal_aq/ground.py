"""Ground-station data: OpenAQ (CPCB / MPCB stations) or a CSV export.

Stations report surface concentrations; Sentinel-5P reports column amounts.
This module only loads and harmonises the station data:
  * converts ppb / ppm / mg/m³ to µg/m³;
  * keeps the hours around the Sentinel-5P overpass (≈13:30 local solar time,
    13:00–15:00 IST by default);
  * returns one value per station per day.

OpenAQ needs a free API key (https://explore.openaq.org → account → API key),
passed as OPENAQ_API_KEY.

CSV format (one row per hourly reading; extra columns are ignored):
    station_id,station_name,lat,lon,datetime,parameter,value,unit
    MH001,Chembur,19.0360,72.8956,2025-01-10T13:00:00+05:30,no2,48.2,µg/m³
datetime must include a UTC offset (or end with Z); the hour is the start of the averaging hour.
"""
from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

OPENAQ_URL = "https://api.openaq.org/v3"
IST = timezone(timedelta(hours=5, minutes=30))

# Sentinel-5P pollutant -> ground parameter (OpenAQ name). The aerosol index has
# no direct ground equivalent; PM2.5 is used as a proxy and flagged as such.
GROUND_PARAMETER = {"NO2": "no2", "SO2": "so2", "CO": "co", "O3": "o3", "AER_AI": "pm25"}
PROXY = {"AER_AI"}
MOLAR_MASS = {"no2": 46.0055, "so2": 64.066, "co": 28.010, "o3": 47.998}
MOLAR_VOLUME_L = 24.45   # at 25 °C, 1 atm (the convention used by CPCB/US-EPA)


def to_ug_m3(value: float, unit: str, parameter: str) -> float | None:
    u = (unit or "").strip().lower().replace("µ", "u").replace("μ", "u").replace("³", "3")
    if u in ("ug/m3", "ugm3", "ug/m^3"):
        return value
    if u in ("mg/m3", "mg/m^3"):
        return value * 1000.0
    mw = MOLAR_MASS.get(parameter)
    if mw is None:
        return None
    if u == "ppb":
        return value * mw / MOLAR_VOLUME_L
    if u == "ppm":
        return value * 1000.0 * mw / MOLAR_VOLUME_L
    return None


@dataclass
class Station:
    id: str
    name: str
    lat: float
    lon: float
    provider: str = ""
    daily: dict = field(default_factory=dict)      # ISO date -> µg/m³ (overpass-window mean)


def overpass_means(hourly, hours=(13, 14)):
    """hourly: iterable of (aware datetime of hour start, µg/m³) -> {date: mean} over IST hours."""
    buckets = defaultdict(list)
    for t, v in hourly:
        if v is None or v < 0:
            continue
        lt = t.astimezone(IST)
        if lt.hour in hours:
            buckets[lt.date().isoformat()].append(v)
    return {d: sum(v) / len(v) for d, v in buckets.items() if v}


# ----------------------------------------------------------------------------
# OpenAQ
# ----------------------------------------------------------------------------
class OpenAQ:
    def __init__(self, api_key: str | None = None, pause: float = 1.1):
        self.key = api_key or os.environ.get("OPENAQ_API_KEY")
        if not self.key:
            raise RuntimeError("Set OPENAQ_API_KEY (free key from https://explore.openaq.org).")
        self.pause = pause   # stay under the free-tier rate limit (60 requests/minute)

    def get(self, path, **params):
        url = f"{OPENAQ_URL}{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(5):
            req = urllib.request.Request(url, headers={"X-API-Key": self.key, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    time.sleep(self.pause)
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAQ {e.code} for {path}: {e.read()[:200]!r}") from e
        raise RuntimeError(f"OpenAQ request failed: {path}")

    def stations(self, bbox, parameter):
        """Fixed monitoring stations in bbox measuring `parameter` -> [(Station, sensor_id)]."""
        out, page = [], 1
        while True:
            js = self.get("/locations", bbox=",".join(f"{v:.4f}" for v in bbox), limit=1000, page=page)
            for loc in js.get("results", []):
                if loc.get("isMobile"):
                    continue
                c = loc.get("coordinates") or {}
                for s in loc.get("sensors", []):
                    if (s.get("parameter") or {}).get("name") == parameter:
                        st = Station(str(loc["id"]), loc.get("name") or str(loc["id"]), c.get("latitude"),
                                     c.get("longitude"), (loc.get("provider") or {}).get("name", ""))
                        out.append((st, s["id"], s["parameter"].get("units", "")))
            if len(js.get("results", [])) < 1000:
                break
            page += 1
        return out

    def hourly(self, sensor_id, start, end, units, parameter):
        """Hourly means between two dates -> [(aware datetime, µg/m³)]."""
        rows, page = [], 1
        while True:
            js = self.get(f"/sensors/{sensor_id}/hours", datetime_from=f"{start}T00:00:00Z",
                          datetime_to=f"{end}T23:59:59Z", limit=1000, page=page)
            res = js.get("results", [])
            for r in res:
                t = ((r.get("period") or {}).get("datetimeFrom") or {}).get("utc")
                u = (r.get("parameter") or {}).get("units", units)
                if t is None or r.get("value") is None:
                    continue
                v = to_ug_m3(float(r["value"]), u, parameter)
                if v is not None:
                    rows.append((datetime.fromisoformat(t.replace("Z", "+00:00")), v))
            if len(res) < 1000:
                break
            page += 1
        return rows


def load_openaq(bbox, pollutant, start, end, api_key=None, log=print):
    param = GROUND_PARAMETER[pollutant]
    client = OpenAQ(api_key)
    found = client.stations(bbox, param)
    log(f"OpenAQ: {len(found)} stations measuring {param} in the area")
    stations = []
    for st, sid, units in found:
        try:
            st.daily = overpass_means(client.hourly(sid, start, end, units, param))
        except RuntimeError as e:
            log(f"  {st.name}: skipped ({e})")
            continue
        if st.daily:
            stations.append(st)
            log(f"  {st.name}: {len(st.daily)} days")
    return stations


# ----------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------
def load_csv(path, pollutant, start=None, end=None):
    param = GROUND_PARAMETER[pollutant]
    meta, hourly = {}, defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("parameter", "").strip().lower().replace(".", "") != param:
                continue
            try:
                v = to_ug_m3(float(row["value"]), row.get("unit", "ug/m3"), param)
                t = datetime.fromisoformat(row["datetime"].strip().replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            if v is None:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=IST)          # CPCB exports are in IST
            sid = row["station_id"].strip()
            meta[sid] = (row.get("station_name", sid).strip(), float(row["lat"]), float(row["lon"]))
            hourly[sid].append((t, v))
    stations = []
    for sid, rows in hourly.items():
        name, lat, lon = meta[sid]
        st = Station(sid, name, lat, lon, "csv", overpass_means(rows))
        if start or end:
            st.daily = {d: v for d, v in st.daily.items() if (not start or d >= start) and (not end or d <= end)}
        if st.daily:
            stations.append(st)
    return stations
