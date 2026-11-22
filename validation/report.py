"""Fill the results tables in README.md and docs/VALIDATION.md from validation/results/*.json.

    python validation/report.py

Text between <!-- AUTO:name --> and <!-- /AUTO:name --> markers is regenerated. Everything
else is left alone, so after re-running any validation, run this and the numbers update.
"""
from __future__ import annotations

import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(HERE, "results")
SHORT = {"NO2": "NO₂", "SO2": "SO₂", "CO": "CO", "HCHO": "HCHO", "O3": "O₃", "AER_AI": "Aerosol index"}
ORDER = ["NO2", "SO2", "CO", "HCHO", "O3", "AER_AI"]
# afternoon GCAS NO2 flights of LISTOS 2018 that overlap Sentinel-5P operations
LISTOS_PM_FLIGHTS = ["2018-07-02", "2018-07-19", "2018-07-20", "2018-08-05", "2018-08-06", "2018-08-15",
                     "2018-08-16", "2018-08-24", "2018-08-28", "2018-08-29", "2018-09-06"]
KNOWN_NO_OVERPASS = ["2018-08-05"]   # no Sentinel-5P NO2 over New York in Earth Engine that day
NAMES = {"raw_1km": "Sentinel-5P as is (1 km)", "bilinear": "Interpolation (bilinear)", "lr": "Linear regression",
         "mslr": "Multi-scale regression", "cnn": "**Hybrid regression + CNN**"}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def f2(v, signed=False):
    """Two decimals, typographic minus, no '-0.00' / '+0.00'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if abs(v) < 0.005:
        return "0.00"
    t = f"{v:+.2f}" if signed else f"{v:.2f}"
    return t.replace("-", "−")


def pct(v):
    return "–" if v is None else f"{100 * v:.0f}"


def pval(p):
    if p is None:
        return "–"
    return "< 0.01" if p < 0.01 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def wins(s, key="r_250m"):
    w = s.get(f"{key}_wins_vs_bilinear") or "0/0"
    a, b = (int(x) for x in w.split("/"))
    return a, b


def verdict(s, key):
    """'better than' / 'on par with' / 'slightly worse than' interpolation, from wins, sign test and median gain."""
    w, n = wins(s, key)
    p = s.get(f"{key}_sign_test_p")
    g = s.get(f"{key}_gain_vs_bilinear") or 0.0
    if p is not None and p < 0.05 and g >= 0.05:
        return "better than"
    if p is not None and p < 0.05 and g > 0:
        return "slightly better than"
    if p is not None and p <= 0.1 and g > 0:
        return "slightly better than"
    if g < -0.02 and w < n / 2:
        return "slightly worse than"
    return "on par with"


def nice_day(d):
    from datetime import date
    x = date.fromisoformat(d)
    return f"{x.day} {x.strftime('%b')}"


# ---------------------------------------------------------------- airborne
def airborne_parts():
    d = load("airborne_NO2.json")
    if not d:
        return None
    S, R = d["summary"], d.get("summary_resolved") or {}
    tiles = d["tiles"]
    flights = d.get("flights") or sorted({t["day"] for t in tiles})
    skipped = sorted(set(d.get("flights_without_overpass") or []) | set(KNOWN_NO_OVERPASS))
    n_all, n_res = S["cnn"]["tiles"], (R.get("cnn") or {}).get("tiles", 0)
    w_all, _ = wins(S["cnn"])
    w_res, _ = wins(R["cnn"]) if R else (0, 0)
    w_det, n_det = wins(S["cnn"], "r_detail_sub1km")
    unres = [t for t in tiles if (t["scores"]["raw_1km"]["r_250m"] or -1) < 0.3]
    lo = [t["airborne_mean_molec_cm2"] / 1e15 for t in unres]
    remaining = [x for x in LISTOS_PM_FLIGHTS if x not in flights and x not in skipped]
    return dict(d=d, S=S, R=R, flights=flights, skipped=skipped, n_all=n_all, n_res=n_res, w_all=w_all,
                w_res=w_res, w_det=w_det, n_det=n_det, unres=unres, lo=lo, remaining=remaining)


def flights_sentence(a):
    """Main clause plus extra sentences about skipped flights and the a-priori version."""
    s = f"{len(a['flights'])} afternoon flights ({', '.join(nice_day(x) for x in a['flights'])} 2018)."
    if a["skipped"]:
        s += f" {', '.join(nice_day(x) for x in a['skipped'])} had no Sentinel-5P overpass in Earth Engine."
    req = a["d"].get("column", 1)
    alt = sorted({t["day"] for t in a["d"]["tiles"] if t.get("column_used", req) != req})
    if alt:
        s += (f" For {', '.join(nice_day(x) for x in alt)} NASA's WRF-Chem column is empty, so the NAM-CMAQ version "
              "of the aircraft column is used.")
    return s


def block_headline():
    a, rows = airborne_parts(), []
    if a:
        c, b = a["R"].get("cnn", {}), a["R"].get("bilinear", {})
        S = a["S"]["cnn"]
        w250, n = wins(S, "r_250m")
        w1, _ = wins(S, "r_1km")
        rows.append(f"| **1. NASA airborne NO₂ maps (New York)** | independent column maps at 250 m | With no New York training, "
                    f"the Mumbai-trained CNN is **{verdict(S, 'r_250m')} interpolation at 250 m** ({w250}/{n} tiles) and "
                    f"**{verdict(S, 'r_1km')} it at 1 km** ({w1}/{n} tiles, r {f2(a['S']['bilinear']['r_1km_median'])} → "
                    f"{f2(S['r_1km_median'])}) |")
    sk = [load(f"upscaling_{p}.json") for p in ORDER]
    sk = [x["methods"]["cnn"]["skill_vs_bilinear"] for x in sk if x]
    if sk:
        pos = sum(v > 0 for v in sk)
        rows.append(f"| **2. Up-scaling test (Mumbai, {len(sk)} pollutants)** | real Sentinel-5P at 1 km | CNN adds detail "
                    f"for **{'every pollutant' if pos == len(sk) else f'{pos} of {len(sk)} pollutants'}**: skill "
                    f"**{min(sk):+.2f} to {max(sk):+.2f}** over interpolation on a held-out season |")
    rows.append("| 3. Simulated scene | known 10 m field | CNN recovers hotspot classes best for 5 of 6 pollutants |")
    g = [load(f"ground_{p}.json") for p in ORDER]
    g = [x for x in g if x]
    if g:
        rs = [x["predictors"]["coarse"]["daily_spatial_r_median"] for x in g]
        rows.append(f"| 4. Surface monitoring stations | CPCB / MPCB monitors | no method, including the raw satellite, "
                    f"tracks station readings (r ≤ {max(0.1, max(rs)):.1f}). Documented below, not used as an accuracy claim |")
    return "| Check | Truth | Headline result |\n|---|---|---|\n" + "\n".join(rows)


def shows_bullets(a):
    S, R = a["S"], a["R"]
    c, b, cr, br = S["cnn"], S["bilinear"], R.get("cnn", {}), R.get("bilinear", {})
    w250, n = wins(c, "r_250m")
    w1, _ = wins(c, "r_1km")
    v250, v1 = verdict(c, "r_250m"), verdict(c, "r_1km")
    out = [
        f"- **At 250 m the Mumbai-trained CNN is {v250} interpolation** in New York: it wins {w250} of {n} tiles "
        f"(sign test p = {pval(c.get('r_250m_sign_test_p'))}), median r {f2(c['r_250m_median'])} vs "
        f"{f2(b['r_250m_median'])}. Its 10 m detail does not transfer reliably to a different city and season.",
        f"- **At 1 km it is {v1} interpolation:** {w1} of {n} tiles (p = {pval(c.get('r_1km_sign_test_p'))}), "
        f"median r {f2(c['r_1km_median'])} vs {f2(b['r_1km_median'])}.",
    ]
    if cr and br:
        out.append(f"- **Where the satellite already matches the aircraft well** ({a['n_res']} tiles), interpolation is hard to "
                   f"beat: r at 250 m is {f2(cr.get('r_250m_median'))} for the CNN vs {f2(br.get('r_250m_median'))} for "
                   f"interpolation, and {f2(cr.get('r_1km_median'))} vs {f2(br.get('r_1km_median'))} at 1 km. On some of these "
                   "tiles the surface-based detail moved NO₂ to the wrong places.")
    out += [
        "- **Regression alone does worse than interpolation,** in New York as in Mumbai.",
        f"- **The level is right:** Sentinel-5P reads ~{f2(S['raw_1km'].get('mean_ratio_median'))}× the aircraft column, "
        "consistent with the known low bias of the 2018 product.",
        "- **More data changed the picture.** The first 3 flights alone (18 tiles) suggested a clear win, but the full "
        f"{len(a['flights'])} flights do not. That is why the larger sample is the one reported.",
    ]
    return out


def block_airborne():
    a = airborne_parts()
    if not a:
        return "_Not run yet: see `validation/airborne/airborne.py`._"
    S, R = a["S"], a["R"]
    head = ("| Method | r at 250 m (all {n} tiles) | r at 1 km (all) | **r at 250 m ({m} resolved tiles)** | "
            "r at 1 km (resolved) | Beats interpolation (all / resolved) |\n|---|---|---|---|---|---|").format(
        n=a["n_all"], m=a["n_res"])
    rows = []
    for k in ["raw_1km", "bilinear", "lr", "mslr", "cnn"]:
        if k not in S:
            continue
        bold = k == "cnn"
        cell = (lambda v: f"**{v}**") if bold else (lambda v: v)
        wa = S[k].get("r_250m_wins_vs_bilinear", "–")
        wr = (R.get(k) or {}).get("r_250m_wins_vs_bilinear", "–")
        rows.append(f"| {NAMES[k]} | {cell(f2(S[k]['r_250m_median']))} | {cell(f2(S[k]['r_1km_median']))} | "
                    f"{cell(f2((R.get(k) or {}).get('r_250m_median')))} | {cell(f2((R.get(k) or {}).get('r_1km_median')))} | "
                    f"{'–' if k in ('raw_1km', 'bilinear') else cell(f'{wa} / {wr}')} |")
    c, b = R.get("cnn", {}), R.get("bilinear", {})
    pa = S["cnn"].get("r_250m_sign_test_p")
    pd = S["cnn"].get("r_detail_sub1km_sign_test_p")
    lo = f"{min(a['lo']):.1f}–{max(a['lo']):.1f} × 10¹⁵ molecules/cm²" if a["lo"] else "low columns"
    sr = (S["cnn"].get("spread_ratio_median"), c.get("spread_ratio_median"))
    mr = S["raw_1km"].get("mean_ratio_median")
    return "\n".join([
        f"I ran the system unchanged on New York for {flights_sentence(a)} It ran on {a['n_all']} tiles of 8 × 8 km "
        "covered by the aircraft at 10 m, and the maps were averaged to the aircraft's 250 m grid. **The CNN is the one trained on Mumbai; no New York "
        "or aircraft data was used to train or tune anything.** The aircraft data is used only for scoring.",
        "", head, *rows, "",
        "*Resolved tiles* are those where the raw 1 km satellite itself tracks the aircraft pattern (r ≥ 0.3). The selection "
        "depends only on the satellite, not on any downscaling method.",
        "", "What it shows:",
        *shows_bullets(a),
        "", "Limits, stated plainly:",
        f"- **On {len(a['unres'])} of {a['n_all']} tiles the raw satellite itself does not track the aircraft** (r < 0.3), "
        "mostly clean or patchy days. No method can add a pattern the satellite does not contain.",
        f"- **Detail inside each 1 km cell is weakly predicted** (r ≈ {f2(c.get('r_detail_sub1km_median'))} vs "
        f"{f2(b.get('r_detail_sub1km_median'))} for interpolation, resolved tiles). The maps are also smoother than reality, with ~{pct(sr[0])}–{pct(sr[1])} % of the aircraft's contrast.",
        f"- **Sub-1 km detail:** the CNN wins {a['w_det']}/{a['n_det']} tiles (p = {pval(pd)}), i.e. no reliable gain.",
        "- **A different city and year** (New York, 2018, an older Sentinel-5P processor with a 7 × 3.5 km pixel). This "
        "tests the **method**, not Mumbai's maps specifically.",
        "- **Shared surface artifacts are possible:** NASA notes that parks and bright buildings can leave small artifacts "
        "in the aircraft retrieval, and our model also uses surface brightness.",
    ])


TRAIN_MORE = ("- **Train on more cities and seasons.** In New York the Mumbai-trained model's fine detail did not transfer "
              "reliably; training on several cities (and the summer season) is the most direct way to make it general.")


def block_airborne_next():
    a = airborne_parts()
    if not a or not a["remaining"]:
        return TRAIN_MORE + ("\n- **Other airborne campaigns over different cities** that overlap Sentinel-5P, to test "
                             "any retrained model independently.")
    return "\n".join([
        f"- **More New York flights.** {len(a['remaining'])} more LISTOS afternoon flights overlap Sentinel-5P "
        f"({', '.join(nice_day(x) for x in a['remaining'])} 2018). Adding them grows the tile count and tightens the statistics.",
        "- **Other airborne campaigns over different cities** that overlap Sentinel-5P, to test transfer beyond one city.",
    ])


def block_airborne_doc():
    a = airborne_parts()
    if not a:
        return "_Not run yet._"
    S, R = a["S"], a["R"]
    rows = []
    for k in ["raw_1km", "bilinear", "lr", "mslr", "cnn"]:
        s, r = S.get(k, {}), R.get(k, {})
        name = NAMES[k].replace("**", "")
        name = f"**{name} (Mumbai-trained)**" if k == "cnn" else name
        w = "–" if k in ("raw_1km", "bilinear") else f"{s.get('r_250m_wins_vs_bilinear')} · {r.get('r_250m_wins_vs_bilinear')}"
        det = "–" if k == "raw_1km" else f2(r.get("r_detail_sub1km_median"))
        rows.append(f"| {name} | {f2(s.get('r_250m_median'))} / {f2(s.get('r_1km_median'))} | "
                    f"{f2(r.get('r_250m_median'))} / {f2(r.get('r_1km_median'))} / {det} | {w} |")
    return "\n".join([
        f"**Results** ({len(a['flights'])} flights, {a['n_all']} tiles):", "",
        f"| | All {a['n_all']} tiles: r 250 m / 1 km | {a['n_res']} resolved tiles: r 250 m / 1 km / detail | "
        "Beats interpolation (all · resolved) |", "|---|---|---|---|", *rows, "",
        f"Sign tests (CNN vs interpolation): 250 m p = {pval(S['cnn'].get('r_250m_sign_test_p'))} (all), "
        f"{pval(R['cnn'].get('r_250m_sign_test_p'))} (resolved); sub-1 km detail {a['w_det']}/{a['n_det']}, "
        f"p = {pval(S['cnn'].get('r_detail_sub1km_sign_test_p'))}.",
    ])


def block_airborne_flights():
    a = airborne_parts()
    if not a:
        return "None used yet."
    s = f"Used so far: {', '.join(nice_day(x) for x in a['flights'])} 2018."
    if a["skipped"]:
        s += f" {', '.join(nice_day(x) for x in a['skipped'])} had no Sentinel-5P overpass in Earth Engine."
    if a["remaining"]:
        s += f" Still available: {', '.join(nice_day(x) for x in a['remaining'])}."
    return s


# ---------------------------------------------------------------- up-scaling and ground
def block_upscaling():
    rows = []
    for p in ORDER:
        d = load(f"upscaling_{p}.json")
        if not d:
            continue
        m = d["methods"]
        rows.append(f"| {SHORT[p]} | {f2(m['bilinear']['detail_r_median'])} | **{f2(m['cnn']['detail_r_median'])}** | "
                    f"**{f2(m['cnn']['skill_vs_bilinear'], True)}** | {f2(m['lr']['skill_vs_bilinear'], True)} |")
    if not rows:
        return "_Not run yet._"
    return ("| Pollutant | Detail r: interpolation | **Detail r: CNN** | **CNN skill vs interpolation** | "
            "Linear regression skill |\n|---|---|---|---|---|\n" + "\n".join(rows))


def block_ground():
    rows, missing = [], []
    for p in ORDER:
        if p == "HCHO":
            continue
        d = load(f"ground_{p}.json")
        if not d:
            missing.append(SHORT[p])
            continue
        pr = d["predictors"]
        rows.append(f"| {SHORT[p]}{' (vs PM2.5)' if d.get('proxy') else ''} | {d['n_stations']} / {d['n_pairs']} | "
                    f"{f2(pr['coarse']['daily_spatial_r_median'])} | {f2(pr['bilinear']['daily_spatial_r_median'])} | "
                    f"{f2(pr['cnn']['daily_spatial_r_median'])} | {f2(pr['cnn']['skill_vs_raw_1km'], True)} |")
    out = ("| Pollutant | Stations / readings | Sentinel-5P as is | Interpolation | CNN | CNN skill vs raw satellite |\n"
           "|---|---|---|---|---|---|\n" + "\n".join(rows))
    if missing:
        out += f"\n\nNot run: {', '.join(missing)}. HCHO has no ground monitors."
    return out


BLOCKS = {"headline": block_headline, "airborne": block_airborne, "airborne_next": block_airborne_next,
          "airborne_doc": block_airborne_doc, "airborne_flights": block_airborne_flights,
          "upscaling": block_upscaling, "ground": block_ground}


def fill(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    n = 0
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- AUTO:{name} -->\n)(.*?)(\n<!-- /AUTO:{name} -->)", re.S)
        if pat.search(s):
            s = pat.sub(lambda m: m.group(1) + fn() + m.group(3), s)
            n += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"{os.path.relpath(path, ROOT)}: {n} blocks updated")


if __name__ == "__main__":
    for p in ("README.md", os.path.join("docs", "VALIDATION.md")):
        fill(os.path.join(ROOT, p))
