"""Builds ny_silage_snapshot.json: NASS QuickStats New York corn-silage rows (county + state) in the NASS tab's row format.

NASS publishes silage area harvested, production (tons) and yield (tons/acre) for NY at state, ASD and county level; the
county lines include an "OTHER (COMBINED) COUNTIES" row for withheld counties, which is what NYCrop.completed() needs to
estimate them. The Snowflake NASS cache and the bundled corn snapshot only hold grain corn, so silage ships as its own file.

Run:  NASS_API_KEY=... python build_ny_silage.py          (the key comes from the environment, never from a file in the repo)
Rebuild after NASS's January state estimates (Crop Production Annual) and again when county estimates land (Feb-Mar).
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import ny_asd_data as nd

OUT = Path(__file__).with_name("ny_silage_snapshot.json")
BASE = "https://quickstats.nass.usda.gov/api/api_GET/"
LINES = {"CORN, SILAGE - ACRES HARVESTED": "harvested",
         "CORN, SILAGE - PRODUCTION, MEASURED IN TONS": "production",
         "CORN, SILAGE - YIELD, MEASURED IN TONS / ACRE": "yield"}


def fetch(key: str, agg: str) -> list:
    # source_desc SURVEY, like every NASS-tab query: in Census years (2017, 2022) NASS also carries a Census of Agriculture row for
    # the same year and line, which would otherwise double the state figure.
    q = {**nd.CROPS_BASE, "key": key, "format": "JSON", "state_alpha": nd.STATE, "commodity_desc": "CORN",
         "util_practice_desc": "SILAGE", "agg_level_desc": agg, "domain_desc": "TOTAL", "year__GE": str(min(nd.YEARS))}
    with urllib.request.urlopen(BASE + "?" + urllib.parse.urlencode(q), timeout=120) as r:
        return json.loads(r.read()).get("data", [])


def build(key: str) -> dict:
    rows = []
    for agg, kind in (("COUNTY", "c"), ("STATE", "s")):
        for r in fetch(key, agg):
            stat = LINES.get(r["short_desc"])
            if stat is None or (kind == "s" and r["reference_period_desc"] != "YEAR"):
                continue
            rows.append([kind, stat, int(r["year"]), r["year"], r.get("county_name"), r.get("county_ansi"), r.get("asd_desc"),
                         r.get("asd_code"), r.get("prodn_practice_desc"), r["reference_period_desc"], r["short_desc"], r["Value"]])
    return {"generated": date.today().isoformat(), "source": "USDA NASS QuickStats, NY corn for silage",
            "cols": nd.ROW_COLS, "rows": rows}


if __name__ == "__main__":
    k = os.environ.get("NASS_API_KEY")
    if not k:
        sys.exit("Set NASS_API_KEY in the environment.")
    snap = build(k)
    OUT.write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
    n_c = sum(1 for r in snap["rows"] if r[0] == "c")
    print(f"wrote {OUT.name}: {len(snap['rows'])} rows ({n_c} county, {len(snap['rows']) - n_c} state), generated {snap['generated']}")
