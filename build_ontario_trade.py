"""Regenerate ontario_trade.json: US soybean exports to Canada by month and customs district (Census Bureau).

Needs a Census API key in the environment:  set CENSUS_API_KEY=...   then   python build_ontario_trade.py
(`--stdout` prints the JSON instead of writing the file, for running on a server and capturing the output.)

US export data is the only monthly, bean-specific import signal for Ontario: Statistics Canada's provincial trade tables stop at
broad commodity groups. Census reports where each shipment left the US, so the customs districts Detroit, Buffalo and Cleveland
are the Ontario gateways. Census publishes a month about five weeks after it ends, so the newest month is always a little old.
HS10 lines: 1201900005 (soybean seeds used as oil stock, kg) and 1201900095 (soybeans except seed, tonnes); sowing seed is left out.
"""
import datetime
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_name("ontario_trade.json")
URL = "https://api.census.gov/data/timeseries/intltrade/exports/hs"
CODES = ["1201900005", "1201900095"]
CANADA = "1220"
START = "2015-01"


def query(key: str, **params):
    params["key"] = key
    url = URL + "?" + urllib.parse.urlencode(params)
    err = ""
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                body = r.read()
            return json.loads(body) if body.strip() else []      # an empty answer (HTTP 204) means no rows
        except Exception as e:
            err = str(e).replace(key, "***")
            time.sleep(2 + 2 * i)
    raise RuntimeError(err[:200])


def kt(qty: str, unit: str) -> float:
    q = float(qty or 0)
    return q / 1e6 if unit == "KG" else (q / 1e3 if unit == "T" else 0.0)


def main() -> int:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        print("Set CENSUS_API_KEY first.", file=sys.stderr)
        return 1
    months: dict = {}
    names: dict = {}
    for code in CODES:
        rows = query(key, get="E_COMMODITY,DISTRICT,DIST_NAME,ALL_VAL_MO,QTY_1_MO,UNIT_QY1", COMM_LVL="HS10", E_COMMODITY=code,
                     CTY_CODE=CANADA, time=f"from {START} to {datetime.date.today().year}-12")
        if not rows:
            raise RuntimeError(f"no rows for {code}")
        hdr = rows[0]
        for r in rows[1:]:
            d = dict(zip(hdr, r))
            m = months.setdefault(d["time"], {"total": 0.0, "usd": 0.0, "d": {}})
            if d["DISTRICT"] == "-":                              # the all-districts row is the month's total
                m["total"] += kt(d["QTY_1_MO"], d["UNIT_QY1"])
                m["usd"] += float(d["ALL_VAL_MO"] or 0)
            else:
                names[d["DISTRICT"]] = d["DIST_NAME"].title()
                m["d"][d["DISTRICT"]] = round(m["d"].get(d["DISTRICT"], 0.0) + kt(d["QTY_1_MO"], d["UNIT_QY1"]), 3)
        print(f"{code}: done", file=sys.stderr)
    for m in months.values():
        m["total"], m["usd"] = round(m["total"], 3), round(m["usd"])
    out = {"generated": datetime.date.today().isoformat(), "source": "US Census Bureau international trade API, exports by "
           "customs district, Canada, HS10 " + " + ".join(CODES), "unit": "thousand tonnes (kt)", "districts": names,
           "months": dict(sorted(months.items()))}
    text = json.dumps(out, separators=(",", ":"))
    if "--stdout" in sys.argv:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        last = max(months)
        print(f"wrote {OUT.name}: {len(months)} months {min(months)} to {last}, {len(text) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
