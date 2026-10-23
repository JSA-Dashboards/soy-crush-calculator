"""Regenerate the Draw Territory data assets from the live NASS cache:

    ny_nass_snapshot.json  NY rows for corn/soybeans (county + state + stocks) and NY livestock series;
                           the fallback used when the live cache can't be reached
    ny_asd_geo.json        static NY county outlines, dissolved ASD district polygons, label centroids
                           and the county -> district map (so the app needs no geopandas / big GeoJSON)

Run with the SNOWFLAKE_* environment variables set (account, user, role, warehouse and
SNOWFLAKE_PRIVATE_KEY_PATH + SNOWFLAKE_PRIVATE_KEY_PWD, or SNOWFLAKE_PASSWORD):

    python build_ny_assets.py

Needs shapely (local only; the app itself does not).
"""
import datetime
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import snowflake.connector

import dt_nass
import ny_asd_data as nd

HERE = Path(__file__).parent
GEO_FILE = HERE / "ny_asd_geo.json"
COUNTY_GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def _private_key():
    path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "")
    if not path and not pem.strip():
        return None
    from cryptography.hazmat.primitives import serialization
    raw = open(path, "rb").read() if path else pem.replace("\\n", "\n").encode()
    pwd = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PWD") or None
    key = serialization.load_pem_private_key(raw, password=pwd.encode() if pwd else None)
    return key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _connect():
    kw = dict(account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
              role=os.environ.get("SNOWFLAKE_ROLE") or None,
              warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or None,
              database="JSA", schema="NASS_CACHE", login_timeout=30)
    pkey = _private_key()
    if pkey is not None:
        kw["private_key"] = pkey
    else:
        kw["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    return snowflake.connector.connect(**kw)


def _rnd(coords, nd_=4):
    if isinstance(coords, (int, float)):
        return round(float(coords), nd_)
    return [_rnd(c, nd_) for c in coords]


def _poly_centroid_area(coords):
    x, y = coords[:, 0], coords[:, 1]
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = 0.5 * a.sum()
    area = abs(A)
    if area < 1e-10:
        return float(x.mean()), float(y.mean()), area
    return (float(((x[:-1] + x[1:]) * a).sum() / (6 * A)),
            float(((y[:-1] + y[1:]) * a).sum() / (6 * A)), area)


def _county_centroid(geom):
    if geom["type"] == "Polygon":
        return _poly_centroid_area(np.array(geom["coordinates"][0]))
    best, best_area = None, 0
    for part in geom["coordinates"]:
        c = np.array(part[0])
        a = abs(0.5 * (c[:-1, 0] * c[1:, 1] - c[1:, 0] * c[:-1, 1]).sum())
        if a > best_area:
            best, best_area = c, a
    return _poly_centroid_area(best)


def build_geo(fips_map: dict) -> dict:
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    with urllib.request.urlopen(COUNTY_GEOJSON, timeout=60) as r:
        allgeo = json.load(r)
    ny = [f for f in allgeo["features"] if f["properties"]["STATE"] == nd.STATE_FIPS]
    counties, centroids, groups = [], {}, {}
    for f in ny:
        fips = f["properties"]["STATE"] + f["properties"]["COUNTY"]
        g = {"type": f["geometry"]["type"], "coordinates": _rnd(f["geometry"]["coordinates"])}
        counties.append({"type": "Feature", "id": fips, "properties": {"NAME": f["properties"]["NAME"]}, "geometry": g})
        cx, cy, area = _county_centroid(g)
        centroids[fips] = [round(cx, 4), round(cy, 4), round(area, 6)]
        entry = fips_map.get(fips)
        if entry:
            groups.setdefault(tuple(entry), []).append(shape(g).buffer(0))
    districts = []
    for (name, code), geoms in sorted(groups.items(), key=lambda kv: kv[0][1]):
        u = unary_union(geoms)
        c = u.centroid
        geom = mapping(u)
        districts.append({"type": "Feature", "properties": {
            "District": name, "DistrictCode": code,
            "centroid_lon": round(c.x, 4), "centroid_lat": round(c.y, 4)},
            "geometry": {"type": geom["type"], "coordinates": _rnd(geom["coordinates"])}})
    return {"counties": {"type": "FeatureCollection", "features": counties},
            "districts": {"type": "FeatureCollection", "features": districts},
            "centroids": centroids, "fips_map": {k: list(v) for k, v in fips_map.items()}}


def main() -> None:
    conn = _connect()
    cur = conn.cursor()

    def run(sql):
        cur.execute(sql)
        return cur.fetchall()

    snap = {"generated": datetime.date.today().isoformat(), "crops": {}, "livestock": {}}
    for crop in nd.CROPS:
        rows = nd.fetch_crop_rows(crop, run)
        snap["crops"][crop] = nd.frame_to_snapshot(rows)
        print(f"{crop}: {len(rows)} NY rows")

    for comm, mp in dt_nass.LIVESTOCK.items():
        cur.execute("SELECT data FROM nass_cache WHERE cache_key = %s", (dt_nass.cache_key(dt_nass.state_params(mp)),))
        row = cur.fetchone()
        recs = []
        if row:
            raw = row[0] if isinstance(row[0], (dict, list)) else json.loads(row[0])
            recs = [r for r in (raw.get("data", []) if isinstance(raw, dict) else raw) if r.get("state_alpha") == nd.STATE]
        df = dt_nass.state_series(recs, mp)
        snap["livestock"][comm] = [[int(r.year), float(r.value)] for r in df.itertuples()]
        print(f"{comm}: {len(df)} years")

    # complete county -> ASD map: every NY county row anywhere in the ASD dashboard's cache
    rows = run("""
        SELECT f.value:county_ansi::string, f.value:asd_code::string, f.value:asd_desc::string, COUNT(*)
        FROM JSA.NASS_CACHE.nass_cache c, LATERAL FLATTEN(input => c.data:data) f
        WHERE c.dashboard = 'rma-production-map' AND c.params:agg_level_desc::string = 'COUNTY'
          AND f.value:state_alpha::string = 'NY' AND f.value:county_ansi::string <> ''
          AND f.value:asd_code::string <> '' AND f.value:county_ansi::string NOT IN ('998','999','000')
        GROUP BY 1, 2, 3""")
    best: dict = {}
    for ansi, code, desc, n in rows:
        fips = nd.STATE_FIPS + str(ansi).zfill(3)
        if fips not in best or n > best[fips][2]:
            best[fips] = (str(desc).strip().title(), str(code).strip(), n)
    fips_map = {f: (v[0], v[1]) for f, v in best.items()}
    print("county -> district map:", len(fips_map), "counties,", len({v for v in fips_map.values()}), "districts")
    conn.close()

    snap["fips_map"] = {k: list(v) for k, v in fips_map.items()}
    nd.SNAPSHOT_FILE.write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {nd.SNAPSHOT_FILE.name} ({nd.SNAPSHOT_FILE.stat().st_size / 1024:.0f} KB)")

    geo = build_geo(fips_map)
    GEO_FILE.write_text(json.dumps(geo, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {GEO_FILE.name} ({GEO_FILE.stat().st_size / 1024:.0f} KB): "
          f"{len(geo['counties']['features'])} counties, {len(geo['districts']['features'])} districts")


if __name__ == "__main__":
    main()
