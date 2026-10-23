"""Regenerate ny_lanes.json: road miles and drive hours from every New York county centroid to each buyer.

Run locally (needs internet, no keys):  python build_ny_lanes.py
Source: the public OSRM demo server (OpenStreetMap road network, shortest-time driving routes, car profile).
The app reads the saved file, so Streamlit Cloud never calls OSRM. Rebuild after adding or moving a destination.
"""
import datetime
import json
import sys
import urllib.request
from pathlib import Path

import ny_asd_figs as nf
import ny_ho

OUT = Path(__file__).with_name("ny_lanes.json")
METERS_PER_MILE = 1609.344

# key, display name, short label, how to find the coordinates, crosses the border?
DESTS = [
    ("union_springs", "Du Mond Grain - Union Springs", "Union Springs", ("facility", "Union Springs"), False),
    ("york", "Du Mond Grain - York", "York", ("facility", "York"), False),
    ("hamilton", "Bunge - Hamilton, Ontario", "Hamilton ON", ("competitor", "Bunge Hamilton (Ontario, Canada)"), True),
    ("oswego", "DeLong - Port of Oswego export", "Oswego", ("competitor", "DeLong - Port of Oswego Grain Export Center"), False),
    ("perdue_bainbridge", "Perdue - Bainbridge, PA", "Perdue Bainbridge", ("competitor", "Perdue Agribusiness - Bainbridge, PA"), False),
    ("perdue_lancaster", "Perdue - Lancaster Co., PA", "Perdue Lancaster", ("competitor", "Perdue Agribusiness - Lancaster Co., PA"), False),
]


def dest_point(kind: str, name: str):
    if kind == "facility":
        f = next(x for x in nf.FACILITIES if x["name"] == name)
        return f["lon"], f["lat"]
    c = next(x for x in ny_ho.COMPETITORS if x["name"] == name)
    return c["lon"], c["lat"]


def main() -> int:
    geo = nf.load_geo()
    names = {f["id"]: f["properties"]["NAME"] for f in geo["counties"]["features"]}
    fips = [f for f in geo["centroids"] if f in names]
    origins = [(geo["centroids"][f][0], geo["centroids"][f][1]) for f in fips]
    dests = [dest_point(*d[3]) for d in DESTS]

    coords = ";".join(f"{lon:.5f},{lat:.5f}" for lon, lat in origins + dests)
    n_o = len(origins)
    src = ";".join(str(i) for i in range(n_o))
    dst = ";".join(str(n_o + i) for i in range(len(dests)))
    url = (f"https://router.project-osrm.org/table/v1/driving/{coords}"
           f"?sources={src}&destinations={dst}&annotations=distance,duration")
    req = urllib.request.Request(url, headers={"User-Agent": "jsa-soy-crush-calculator lane build"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    if d.get("code") != "Ok":
        print("OSRM error:", d.get("code"), d.get("message"))
        return 1

    out = {
        "built": datetime.date.today().isoformat(),
        "source": "OSRM public demo (OpenStreetMap), car profile, county centroid to destination",
        "dests": [{"key": k, "name": nm, "short": sh, "lon": dest_point(*loc)[0], "lat": dest_point(*loc)[1],
                   "cross_border": cb} for k, nm, sh, loc, cb in DESTS],
        "origins": {},
    }
    worst = 0.0
    for i, f in enumerate(fips):
        snap = (d["sources"][i].get("distance") or 0) / METERS_PER_MILE
        worst = max(worst, snap)
        miles, hours = {}, {}
        for j, (key, *_rest) in enumerate(DESTS):
            m, s = d["distances"][i][j], d["durations"][i][j]
            miles[key] = None if m is None else round(m / METERS_PER_MILE, 1)
            hours[key] = None if s is None else round(s / 3600, 2)
        out["origins"][f] = {"name": names[f], "lon": origins[i][0], "lat": origins[i][1],
                             "snap_mi": round(snap, 1), "miles": miles, "hours": hours}
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    missing = [(v["name"], k) for v in out["origins"].values() for k, m in v["miles"].items() if m is None]
    print(f"wrote {OUT.name}: {len(fips)} counties x {len(DESTS)} destinations, "
          f"{OUT.stat().st_size / 1024:.0f} KB; largest centroid-to-road snap {worst:.1f} mi; unroutable pairs: {len(missing)}")
    if missing:
        print(missing[:10])
    return 0


if __name__ == "__main__":
    sys.exit(main())
