"""New York corn silage: the Silage tab's data and county table (no Streamlit import).

NASS publishes NY corn-for-silage area harvested, production (tons) and yield at state, district and county level. The rows live
in ny_silage_snapshot.json (build_ny_silage.py) in the NASS tab's row format, so NYCrop treats silage like any crop: reported
counties are used as published and the withheld ones are estimated from their history, scaled to NASS's totals.

Silage has no planted-acres line and NASS has no current-year silage forecast. The current year is DERIVED from the grain corn
numbers the NASS tab already carries: silage acres = corn planted (all purposes) - corn harvested for grain - the typical
"other" (planted acres that were neither grain nor silage: abandoned, grazed, failed), averaged over the last five years.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ny_asd_data as nd
import ny_ho

SNAPSHOT_FILE = Path(__file__).with_name("ny_silage_snapshot.json")
HIST = 5                                                  # years behind the typical "other" and the default yield


def load_snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build(corn: nd.NYCrop, fips_map: dict | None = None) -> tuple:
    """(silage NYCrop, meta). meta: other_avg (typical planted acres that were neither grain nor silage), fc_acres (derived current-
    year silage acres, or None), generated (snapshot date). Returns (None, {}) when the snapshot file is missing."""
    snap = load_snapshot()
    if not snap.get("rows"):
        return None, {}
    frame = nd.snapshot_to_frame(snap)
    base = nd.NYCrop("Silage", frame, fips_map)
    gaps = []
    for y in range(nd.FORECAST_YEAR - HIST, nd.FORECAST_YEAR):
        p, g, s = corn.state_total("planted", y), corn.state_total("harvested", y), base.state_total("harvested", y)
        if p and g is not None and s is not None:
            gaps.append(p - g - s)
    other_avg = float(np.mean(gaps)) if gaps else 0.0
    pl, gh = corn.forecast_state("planted"), corn.forecast_state("harvested")
    fc = max(0.0, pl - gh - other_avg) if (pl and gh) else None
    meta = {"other_avg": other_avg, "fc_acres": fc, "generated": snap.get("generated", ""), "planted_fc": pl, "grain_fc": gh}
    if not fc:
        return base, meta
    row = {"kind": "f", "stat": "harvested", "req_year": nd.FORECAST_YEAR, "year": str(nd.FORECAST_YEAR), "county_name": None,
           "county_ansi": None, "asd_desc": None, "asd_code": None, "practice": nd.ALL_PRACTICES, "ref_period": "YEAR",
           "short_desc": "CORN, SILAGE - ACRES HARVESTED (derived from corn planted less grain harvested)", "value": f"{fc:.0f}"}
    return nd.NYCrop("Silage", pd.concat([frame, pd.DataFrame([row])], ignore_index=True), fips_map), meta


def default_yield(silage: nd.NYCrop) -> float | None:
    """Olympic average (tons/acre) of the last five published state silage yields; None when there are none."""
    ys = [silage.state_total("yield", y) for y in range(nd.FORECAST_YEAR - HIST, nd.FORECAST_YEAR)]
    ys = [v for v in ys if v]
    return round(nd._olympic(ys), 1) if ys else None


def is_forecast(silage: nd.NYCrop, year: int) -> bool:
    return year == nd.FORECAST_YEAR and silage.reported_counties(year) == 0


def state_history(corn: nd.NYCrop, silage: nd.NYCrop) -> pd.DataFrame:
    """[year, planted, grain, silage, other, forecast] oldest first; the current year uses the forecast values."""
    rows = []
    for y in sorted(nd.YEARS):
        fc = y == nd.FORECAST_YEAR and silage.reported_counties(y) == 0
        p = corn.forecast_state("planted") if fc else corn.state_total("planted", y)
        g = corn.forecast_state("harvested") if fc else corn.state_total("harvested", y)
        s = silage.forecast_state("harvested") if fc else silage.state_total("harvested", y)
        if p and g is not None and s is not None:
            rows.append({"year": y, "planted": p, "grain": g, "silage": s, "other": max(0.0, p - g - s), "forecast": fc})
    return pd.DataFrame(rows, columns=["year", "planted", "grain", "silage", "other", "forecast"])


def county_table(corn: nd.NYCrop, silage: nd.NYCrop, geo: dict, year: int, corn_yield: float | None = None,
                 silage_yield: float | None = None) -> pd.DataFrame:
    """One row per NY county for the Silage tab: silage acres / tons / yield, corn planted (all purposes), grain acres and bushels,
    silage share of corn acres, milk cows, and whether any silage or corn figure is an estimate (withheld county). For the current
    year every figure is a forecast: each district's forecast split across its counties (NYCrop.county_forecast)."""
    names = {f["id"]: f["properties"]["NAME"] for f in geo["counties"]["features"]}
    fm = silage.fips_map() or corn.fips_map()
    if is_forecast(silage, year):
        cf, sf = corn.county_forecast(float(corn_yield or 0.0)), silage.county_forecast(float(silage_yield or 0.0))
        pl, gh, gb = cf["planted"], cf["harvested"], cf["production"]
        sa, stn = sf["harvested"], sf["production"]
        est = {}
    else:
        def comp(obj, stat):
            c = obj.completed(stat, year)
            return dict(zip(c["fips"], c["Value"])), dict(zip(c["fips"], c["is_estimated"]))
        pl, pl_e = comp(corn, "planted")
        gh, _ = comp(corn, "harvested")
        gb, _ = comp(corn, "production")
        sa, sa_e = comp(silage, "harvested")
        stn, _ = comp(silage, "production")
        est = {f: bool(pl_e.get(f) or sa_e.get(f)) for f in names}
    look = ny_ho.fips_lookup(geo)
    cows = {look[ny_ho.norm(n)]: v for n, v in ny_ho.DAIRY_COWS.items() if ny_ho.norm(n) in look}
    rows = []
    for f, n in names.items():
        a, p = sa.get(f), pl.get(f)
        rows.append({"fips": f, "County": n, "District": fm.get(f, ("", ""))[0],
                     "Silage acres": a, "Silage tons": stn.get(f),
                     "Silage yield": (stn[f] / a) if (stn.get(f) and a) else None,
                     "Corn planted": p, "Grain acres": gh.get(f), "Grain bu": gb.get(f),
                     # silage and planted acres are estimated separately, so a county can come out a hair over 100%; cap it
                     "Silage share": min(100.0, a / p * 100.0) if (a is not None and p) else None,
                     "Milk cows": cows.get(f), "Estimated": bool(est.get(f, False))})
    return pd.DataFrame(rows)
