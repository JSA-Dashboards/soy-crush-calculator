"""Regenerate ny_livestock_census.json: New York county livestock from the Census of Agriculture.

Run locally:  set NASS_API_KEY=<your Quick Stats key>  then  python build_ny_livestock.py
NASS publishes county livestock only in the Census (every 5 years: ..., 2017, 2022; next is 2027, released 2029),
so this file is static between Census releases and the app never calls Quick Stats for it.
"""
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import ny_asd_figs as nf

OUT = Path(__file__).with_name("ny_livestock_census.json")
API = "https://quickstats.nass.usda.gov/api/api_GET/?"

# app label -> (inventory short_desc, operations short_desc)
COMMODITIES = {
    "Dairy Cows": ("CATTLE, COWS, MILK - INVENTORY", "CATTLE, COWS, MILK - OPERATIONS WITH INVENTORY"),
    "All Cattle & Calves": ("CATTLE, INCL CALVES - INVENTORY", "CATTLE, INCL CALVES - OPERATIONS WITH INVENTORY"),
    "Hogs & Pigs": ("HOGS - INVENTORY", "HOGS - OPERATIONS WITH INVENTORY"),
}


def query(key: str, **params):
    params.update(key=key, format="JSON")
    url = API + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)["data"]
        except urllib.error.HTTPError as e:
            if e.code == 400:      # Quick Stats answers an empty result with 400
                return []
            time.sleep(3)
    raise RuntimeError("Quick Stats did not answer: " + params.get("short_desc", ""))


def parse(v: str):
    """(D) = withheld -> None, (Z) = under half a unit -> 0."""
    s = v.strip().replace(",", "")
    if s == "(Z)":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def main() -> int:
    key = os.environ.get("NASS_API_KEY")
    if not key:
        print("Set NASS_API_KEY first (Quick Stats key; see your NASS API notes).")
        return 1
    geo = nf.load_geo()
    ids = {f["id"] for f in geo["counties"]["features"]}
    out = {"generated": datetime.date.today().isoformat(), "source": "USDA NASS Census of Agriculture (Quick Stats)",
           "reference": "Dec 31", "comm": {}}
    years = set()
    for label, (inv_sd, ops_sd) in COMMODITIES.items():
        out["comm"][label] = {}
        for stat, sd in (("inventory", inv_sd), ("farms", ops_sd)):
            state, county = {}, {}
            for lvl in ("STATE", "COUNTY"):
                rows = query(key, short_desc=sd, agg_level_desc=lvl, state_alpha="NY", source_desc="CENSUS",
                             domain_desc="TOTAL")
                time.sleep(0.4)
                for r in rows:
                    y = str(int(r["year"]))
                    years.add(int(y))
                    v = parse(r["Value"])
                    if lvl == "STATE":
                        state[y] = v
                    else:
                        f = "36" + r["county_code"]
                        if f in ids:
                            county.setdefault(y, {})[f] = v
            out["comm"][label][stat] = {"state": state, "county": county}
            print(f"{label} / {stat}: {len(state)} state years, "
                  f"{sum(len(c) for c in county.values())} county values")
    out["years"] = sorted(years)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.name}: {OUT.stat().st_size / 1024:.0f} KB, Census years {out['years']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
