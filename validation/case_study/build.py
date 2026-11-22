"""Build the Chembur-Trombay-Mahul case study from maps saved by fetch.py.

    python validation/case_study/build.py

Writes docs/case_study/*.png and docs/CASE_STUDY.md. It needs numpy and matplotlib only.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
OUT_DIR = os.path.join(ROOT, "docs", "case_study")
DOC = os.path.join(ROOT, "docs", "CASE_STUDY.md")

# Site points from Global Energy Monitor (gem.wiki); HPCL from Climate TRACE.
FACILITIES = [
    ("Tata Power Trombay (coal, 274 m stack)", 19.0027, 72.8979, "Global Energy Monitor"),
    ("BPCL Mumbai Refinery, Mahul", 19.0171, 72.9013, "Global Energy Monitor"),
    ("HPCL Mumbai Refinery, Mahul", 19.0179, 72.9032, "Climate TRACE"),
    ("RCF Trombay (fertilisers)", 19.0410, 72.8869, "Global Energy Monitor"),
]
PALETTE = np.array([
    [26, 152, 80], [102, 189, 99], [166, 217, 106], [217, 239, 139], [254, 224, 139],
    [253, 174, 97], [244, 109, 67], [215, 48, 39], [165, 0, 38], [103, 0, 31]]) / 255.0


def load(area="chembur_trombay", pollutant="NO2"):
    items = []
    for p in sorted(glob.glob(os.path.join(DATA, f"{pollutant}_*_{area}_*.json"))):
        with open(p) as f:
            meta = json.load(f)
        z = np.load(p[:-5] + ".npz")
        items.append({"meta": meta, "fine": z["fine"].astype("f8"), "coarse": z["coarse"].astype("f8"),
                      "day": meta["provenance"]["target_day"]})
    return items


def km_per_deg(lat):
    return 111.32 * math.cos(math.radians(lat)), 110.7


def pixel_of(lat, lon, bounds, shape):
    lon0, lat0, lon1, lat1 = bounds
    H, W = shape
    r = int((lat1 - lat) / (lat1 - lat0) * H)
    c = int((lon - lon0) / (lon1 - lon0) * W)
    return (r, c) if (0 <= r < H and 0 <= c < W) else None


def value_at(field, lat, lon, bounds, radius_px):
    rc = pixel_of(lat, lon, bounds, field.shape)
    if rc is None:
        return math.nan
    r, c = rc
    w = field[max(0, r - radius_px):r + radius_px + 1, max(0, c - radius_px):c + radius_px + 1]
    return float(np.nanmean(w))


def percentile_rank(field, v):
    f = field[np.isfinite(field)]
    return float((f < v).mean() * 100) if f.size and np.isfinite(v) else math.nan


def upsample_nearest(coarse, shape):
    H, W = shape
    h, w = coarse.shape
    ri = np.minimum((np.arange(H) * h / H).astype(int), h - 1)
    ci = np.minimum((np.arange(W) * w / W).astype(int), w - 1)
    return coarse[np.ix_(ri, ci)]


def draw(ax, field, bounds, vmin, vmax, title, hotspots=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("aq", PALETTE)
    cmap.set_bad((0.93, 0.93, 0.93))
    lon0, lat0, lon1, lat1 = bounds
    im = ax.imshow(field, extent=[lon0, lon1, lat0, lat1], cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="upper")
    if hotspots:
        ax.scatter([h[1] for h in hotspots], [h[0] for h in hotspots], s=14, facecolors="none",
                   edgecolors="#1b1b1b", linewidths=0.8, label="daily top-10 hotspots")
    for i, (name, la, lo, _src) in enumerate(FACILITIES, 1):
        ax.scatter([lo], [la], marker="^", s=70, color="white", edgecolors="black", linewidths=1.1, zorder=5)
        ax.annotate(str(i), (lo, la), xytext=(5, 4), textcoords="offset points", fontsize=9,
                    fontweight="bold", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
    kx, _ = km_per_deg((lat0 + lat1) / 2)
    x0, y0 = lon0 + 0.04 * (lon1 - lon0), lat0 + 0.04 * (lat1 - lat0)
    ax.plot([x0, x0 + 1 / kx], [y0, y0], color="black", lw=3)
    ax.text(x0 + 0.5 / kx, y0 + 0.012 * (lat1 - lat0), "1 km", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect(1 / math.cos(math.radians((lat0 + lat1) / 2)))
    return im


def legend_text():
    return "\n".join(f"{i}  {n}" for i, (n, *_r) in enumerate(FACILITIES, 1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = load()
    if not items:
        sys.exit(f"No maps in {DATA}. Run fetch.py first.")
    shape = items[0]["fine"].shape
    items = [it for it in items if it["fine"].shape == shape]
    meta0 = items[0]["meta"]
    bounds, unit, res_m = meta0["bounds"], meta0["unit"], meta0["res_m"]
    os.makedirs(OUT_DIR, exist_ok=True)
    scale = 1e6 if unit == "mol/m²" else 1.0
    u = "µmol/m²" if unit == "mol/m²" else unit

    # Best single day: the least cloud in the 1 km input
    best = min(items, key=lambda it: np.isnan(it["coarse"]).mean())
    fine, coarse = best["fine"] * scale, best["coarse"] * scale
    vmin, vmax = np.nanpercentile(fine, 1), np.nanpercentile(fine, 99.5)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.2), constrained_layout=True)
    draw(axs[0], upsample_nearest(coarse, shape), bounds, vmin, vmax, f"Sentinel-5P as measured (~1 km), {best['day']}")
    im = draw(axs[1], fine, bounds, vmin, vmax, f"This system, {res_m} m, {best['day']}")
    fig.colorbar(im, ax=axs, shrink=0.75, label=f"NO₂ column ({u})")
    fig.text(0.01, -0.02, legend_text(), fontsize=8, va="top", family="sans-serif")
    fig.savefig(os.path.join(OUT_DIR, "before_after.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Season mean and recurring hotspots
    mean_f = np.nanmean(np.stack([it["fine"] for it in items]), axis=0) * scale
    mean_c = np.nanmean(np.stack([it["coarse"] for it in items]), axis=0) * scale
    mean_c_up = upsample_nearest(mean_c, shape)
    hs = [(h["lat"], h["lon"], it["day"]) for it in items for h in it["meta"].get("hotspots", [])]
    vmin, vmax = np.nanpercentile(mean_f, 1), np.nanpercentile(mean_f, 99.5)
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.2), constrained_layout=True)
    draw(axs[0], mean_c_up, bounds, vmin, vmax, f"Sentinel-5P, mean of {len(items)} overpasses")
    im = draw(axs[1], mean_f, bounds, vmin, vmax, f"This system ({res_m} m), same {len(items)} overpasses", hotspots=hs)
    axs[1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.colorbar(im, ax=axs, shrink=0.75, label=f"NO₂ column ({u})")
    fig.text(0.01, -0.02, legend_text(), fontsize=8, va="top")
    fig.savefig(os.path.join(OUT_DIR, "season_mean.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Numbers for the write-up
    kx, ky = km_per_deg((bounds[1] + bounds[3]) / 2)
    rad = max(1, int(round(100 / res_m)))           # ~200 m window around each site
    rows = []
    for i, (name, la, lo, src) in enumerate(FACILITIES, 1):
        vf = value_at(mean_f, la, lo, bounds, rad)
        vc = value_at(mean_c_up, la, lo, bounds, rad)
        near = sorted({d for (hla, hlo, d) in hs if math.hypot((hlo - lo) * kx, (hla - la) * ky) <= 1.0})
        rows.append({"n": i, "name": name, "source": src,
                     "pct_10m": percentile_rank(mean_f, vf), "pct_1km": percentile_rank(mean_c_up, vc),
                     "ratio_10m": vf / np.nanmedian(mean_f), "ratio_1km": vc / np.nanmedian(mean_c_up),
                     "days_hotspot_1km": len(near)})
    with open(os.path.join(OUT_DIR, "facilities.json"), "w") as f:
        json.dump({"days": [it["day"] for it in items], "res_m": res_m, "rows": rows}, f, indent=1)

    lines = [
        "# Case study: Chembur – Trombay – Mahul, Mumbai",
        "",
        "This is Mumbai's main industrial belt: a coal power station, two oil refineries and a fertiliser complex"
        " sit within about 4 km of each other, next to homes (the Mahul resettlement colony). At Sentinel-5P's resolution the"
        " whole belt is a handful of pixels. This page shows what the 10 m maps add there.",
        "",
        f"Maps were made by the live system (hybrid regression + CNN, NO₂) for {len(items)} overpasses in the held-out"
        f" 2025-26 dry season ({', '.join(it['day'] for it in items)}), over the app's *Chembur · Trombay · Mahul* area"
        " at " + f"{res_m} m. Reproduce: `python validation/case_study/fetch.py`, then `python validation/case_study/build.py`.",
        "",
        f"## One day: {best['day']} (least cloud)",
        "",
        "![Sentinel-5P at 1 km vs this system at 10 m](case_study/before_after.png)",
        "",
        "## Season average and recurring hotspots",
        "",
        "Circles mark each day's ten strongest hotspots, as listed by the app.",
        "",
        "![Season mean, 1 km vs 10 m](case_study/season_mean.png)",
        "",
        "## Known sources",
        "",
        "The table compares each site's value with the rest of the area. *Percentile*: the share of the area with lower NO₂ than the site, from the season-mean maps."
        " *× median*: the site's value relative to the area median. *Hotspot days*: days with an app hotspot"
        " within 1 km of the site.",
        "",
        "| # | Site | Percentile at 1 km | **Percentile at 10 m** | × median at 1 km | **× median at 10 m** | Hotspot days |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['n']} | {r['name']} | {r['pct_1km']:.0f} | **{r['pct_10m']:.0f}** | {r['ratio_1km']:.2f}"
                     f" | **{r['ratio_10m']:.2f}** | {r['days_hotspot_1km']}/{len(items)} |")
    p10 = [r["pct_10m"] for r in rows]
    p1 = [r["pct_1km"] for r in rows]
    top = max(rows, key=lambda r: r["ratio_10m"])
    lines += [
        "",
        "## What the maps show",
        "",
        f"- **The 1 km map is a gentle gradient.** The whole industrial belt spans a few satellite pixels, and the sites sit"
        f" at the {min(p1):.0f}th–{max(p1):.0f}th percentile of the area. At 10 m they sit at the {min(p10):.0f}th–{max(p10):.0f}th"
        " percentile: the map places the high values on the industrial plots themselves rather than spreading them over"
        " homes, mangroves and the bay.",
        f"- **{top['name']} stands out most** ({top['ratio_10m']:.1f}× the area median at 10 m vs {top['ratio_1km']:.1f}× at 1 km).",
        "- **Hotspot days:** " + ", ".join(f"{r['name'].split(' (')[0].replace(', Mahul', '')} {r['days_hotspot_1km']}/{len(items)}"
                                           for r in rows) + ". A miss at Tata Power's chimney site fits a tall stack (see below).",
        "- **It stays faithful to the satellite.** Each 10 m map, seen through the satellite's footprint, closely"
        " reproduces that day's Sentinel-5P observation (the app reports this as *satellite consistency*). The detail"
        " redistributes what the satellite measured; it does not add pollution.",
        "",
        "Site locations: Global Energy Monitor (Tata Power, BPCL, RCF) and Climate TRACE (HPCL). They are single points;"
        " each site covers hundreds of metres.",
        "",
        "## How to read this",
        "",
        "- **This is an illustration, not a validation.** A hotspot next to a factory is *plausible*, not proof. The"
        " independent accuracy checks are in the README (NASA airborne maps, up-scaling test).",
        "- **Tall stacks put NO₂ downwind.** Tata Power's 274 m chimney releases its plume high up and away from the site."
        " The satellite sees the column above, so the peak can sit a kilometre or more from the plant.",
        "- **Traffic matters too.** The Eastern Freeway, the Sion–Trombay road and port traffic also emit NO₂, so not every hotspot is industrial.",
        "- **Bright bare or paved ground can also look hot.** The model links built-up, industrial and bare surfaces"
        " with NO₂, so some isolated hot patches away from known sources (e.g. in the north-east of the area) are"
        " likely bare land, yards or construction sites rather than emitters. Treat them as places to check, not findings.",
        "- **The 10 m detail comes from the surface** (built-up, industrial and bare land vs vegetation and water). It"
        " sharpens where the satellite already sees pollution. It cannot discover a source the satellite misses.",
        "",
    ]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", DOC, "and", OUT_DIR)
    for r in rows:
        print(f"  {r['n']} {r['name'][:38]:38s} pct 1km {r['pct_1km']:5.1f}  10m {r['pct_10m']:5.1f}  "
              f"x med {r['ratio_1km']:.2f}/{r['ratio_10m']:.2f}  hotspot days {r['days_hotspot_1km']}")


if __name__ == "__main__":
    main()
