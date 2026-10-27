"""Regenerate ontario_data.json and ontario_geo.json for the Draw Territory > Ontario sub-tab.

Run locally (internet, shapely; no keys):  python build_ontario_data.py
Sources, all open data:
- Statistics Canada table 32-10-0359-01 (Ontario corn for grain and soybeans, annual): area, yield, production, price.
- Ontario Data Catalogue (OMAFRA), Open Government Licence - Ontario: supply and disposition of soybeans and grain corn
  by crop year (Oct-Sep), and estimated area and production by county, 2004-2025.
- Statistics Canada 2021 census division cartographic boundaries, matched to OMAFRA's county names.
The app reads the saved files, so Streamlit Cloud never calls these services.
"""
import csv
import datetime
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import orient

HERE = Path(__file__).parent
UA = {"User-Agent": "Mozilla/5.0 (jsa-soy-crush-calculator Ontario build)"}

STATCAN_ZIP = "https://www150.statcan.gc.ca/n1/tbl/csv/32100359-eng.zip"
ODC = "https://data.ontario.ca/dataset"
FILES = {
    "sd_soy": f"{ODC}/402e1e3c-5a77-415b-8351-4b3be113253f/resource/1de3d22e-8179-403a-b142-7602d28f22c3/download/sd_soybeans.xlsx",
    "sd_corn": f"{ODC}/d2090f6d-6002-4272-8817-bd510afba60d/resource/6d817172-14ba-40a1-8fc0-43a8c72bf3cb/download/sd_corn.xlsx",
    "cty_soy": f"{ODC}/e30dc044-5f75-4f33-b63e-6326f8769bea/resource/3a2e7395-17fd-462b-bcfb-1c2e8aa53dbb/download/ctysoy.xlsx",
    "cty_corn": f"{ODC}/e30dc044-5f75-4f33-b63e-6326f8769bea/resource/5f0973f1-6228-4b54-9e5d-9f0d252e0331/download/ctygcorn.xlsx",
}
CD_LAYER = "https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Cartographic_boundary_files/MapServer/4/query"

# OMAFRA county name -> 2021 census division id
COUNTY_CD = {
    "Brant": "3529", "Chatham-Kent": "3536", "Elgin": "3534", "Essex": "3537", "Haldimand-Norfolk": "3528",
    "Hamilton": "3525", "Lambton": "3538", "Middlesex": "3539", "Niagara": "3526", "Oxford": "3532",
    "Bruce": "3541", "Dufferin": "3522", "Grey": "3542", "Halton": "3524", "Huron": "3540", "Peel": "3521",
    "Perth": "3531", "Simcoe": "3543", "Waterloo": "3530", "Wellington": "3523", "Durham": "3518",
    "Haliburton": "3546", "Hastings": "3512", "Kawartha Lakes": "3516", "Muskoka": "3544",
    "Northumberland": "3514", "Parry Sound": "3549", "Peterborough": "3515", "Prince Edward": "3513", "York": "3519",
    "Frontenac": "3510", "Lanark": "3509", "Leeds and Grenville": "3507", "Lennox and Addington": "3511",
    "Ottawa": "3506", "Prescott and Russell": "3502", "Renfrew": "3547", "Stormont, Dundas and Glengarry": "3501",
    "Algoma": "3557", "Cochrane": "3556", "Greater Sudbury": "3553", "Kenora": "3560", "Manitoulin": "3551",
    "Nipissing": "3548", "Rainy River": "3559", "Sudbury": "3552", "Timiskaming": "3554", "Thunder Bay": "3558",
}
REGION = re.compile(r"^\s*(southern|western|central|eastern|northern)\s+ontario(\s+region)?\s*$", re.I)
SUFFIX = re.compile(r"\s+(United Counties|Counties|County|Regional Municipality|Division|District)$", re.I)


def clean_name(label: str) -> str:
    """Older sheets say 'Brant County' or 'Durham Regional Municipality'; newer ones just 'Brant' / 'Durham'."""
    if REGION.match(label):
        return re.sub(r"\s+region$", "", label.strip(), flags=re.I)
    return SUFFIX.sub("", label.strip())
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS = {"m": M[1:-1]}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def get(url: str, params: dict | None = None, tries: int = 6) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                data = r.read()
            if data[:1] in (b"<",) and b"Application Error" in data[:2000]:
                raise RuntimeError("server error page")
            return data
        except Exception as e:                     # the StatCan map server answers 500 now and then
            if i == tries - 1:
                raise
            print("   retry", i + 1, type(e).__name__)
            time.sleep(3 + 2 * i)


# ── minimal XLSX reader (stdlib only) ─────────────────────────────────────────
def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(blob: bytes) -> dict:
    z = zipfile.ZipFile(io.BytesIO(blob))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(t.text or "" for t in si.iter(M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    out = {}
    for s in wb.find("m:sheets", NS):
        path = "xl/" + rels[s.get(RID)].replace("/xl/", "").lstrip("/")
        rows = []
        for r in ET.fromstring(z.read(path)).iter(M + "row"):
            row = {}
            for c in r.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is not None:
                    row[_col(c.get("r"))] = ss[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(row)
        out[s.get("name")] = rows
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── 1. Statistics Canada: Ontario annual series ───────────────────────────────
STATCAN_MEASURES = {
    "Seeded area (acres)": "seeded_ac", "Harvested area (acres)": "harvested_ac",
    "Average yield (bushels per acre)": "yield_bu", "Production (metric tonnes)": "prod_t",
    "Production (bushels)": "prod_bu", "Average farm price (dollars per bushel)": "price_bu",
    "Total farm value (dollars)": "value",
}


def statcan() -> dict:
    z = zipfile.ZipFile(io.BytesIO(get(STATCAN_ZIP)))
    out = {"corn": {}, "soy": {}}
    crops = {"Corn for grain": "corn", "Soybeans": "soy"}
    for r in csv.DictReader(io.TextIOWrapper(z.open("32100359.csv"), encoding="utf-8-sig")):
        if r["GEO"] != "Ontario" or r["Type of crop"] not in crops or int(r["REF_DATE"]) < 1980:
            continue
        key = STATCAN_MEASURES.get(r["Harvest disposition"])
        v = num(r["VALUE"])
        if key and v is not None:
            out[crops[r["Type of crop"]]].setdefault(r["REF_DATE"], {})[key] = v
    return out


# ── 2. OMAFRA supply and disposition ──────────────────────────────────────────
SD_LABELS = {
    "soy": {"Area harvested ('000 acres)": "area", "Beginning stocks": "beginning", "Production": "production",
            "Total supply": "supply", "Milling": "milling", "Export": "export", "Seed requirements": "seed",
            "Feed, waste and dockage": "feed", "Total domestic disappearance": "domestic", "Ending stocks": "ending"},
    "corn": {"Area harvested ('000 acres)": "area", "Total beginning stocks": "beginning", "Production": "production",
             "Total Imports": "imports", "Total supply": "supply", "Total exports": "exports",
             "Human Consumption -estimated": "human", "Industrial use - including ethanol -": "industrial",
             "Seed requirements": "seed", "Feed, waste and dockage": "feed", "Total domestic disappearance": "domestic",
             "Total ending stocks": "ending"},
}


def sd(blob: bytes, crop: str) -> dict:
    sh = read_xlsx(blob)["English"]
    cy = {i: int(re.search(r"(\d{4})", v).group(1)) for i, v in next(r for r in sh if r.get(1, "").startswith("October")).items()
          if re.search(r"(\d{4})", v)}
    out: dict = {}
    for r in sh:
        label = (r.get(0) or "").strip()
        key = next((k for lab, k in SD_LABELS[crop].items() if label.startswith(lab)), None)
        if key and key not in out:
            out[key] = {str(cy[i]): num(r.get(i)) for i in cy if num(r.get(i)) is not None}
    return out


# ── 3. OMAFRA county files ────────────────────────────────────────────────────
def county(blob: bytes, bu_per_tonne: float) -> dict:
    """{year: {"counties": {name: [seeded ac, harvested ac, yield bu/ac, '000 bu, '000 t]}, "regions": {...}, "ontario": [...]}}"""
    out = {}
    for name, rows in read_xlsx(blob).items():
        m = re.match(r"(\d{4})", name)
        if not m:
            continue
        hdr = next((r for r in rows if any("seeded" in str(v).lower() for v in r.values())), None)
        if hdr is None:
            continue
        cols = {}
        for i, v in hdr.items():
            t = str(v).lower()
            if "seeded" in t:
                cols["seeded"] = i
            elif "harvested" in t:
                cols["harvested"] = i
            elif "yield" in t:
                cols["yield"] = i
            elif "000 bu" in t:
                cols["kbu"] = i
            elif "tonnes" in t:
                cols["kt"] = i
        if not {"seeded", "harvested", "kt"} <= set(cols):
            raise RuntimeError(f"unexpected county sheet layout in {name}: {hdr}")
        rec = {"counties": {}, "regions": {}, "ontario": None}
        for r in rows:
            label = clean_name(r.get(0) or "")
            vals = [num(r.get(cols[k])) if k in cols else None for k in ("seeded", "harvested", "yield", "kbu", "kt")]
            if not label or vals[4] is None or label.lower().startswith(("soy", "grain", "corn")):
                continue
            if "kbu" not in cols:                  # the 2015 sheets give hundredweight: derive bushels from tonnes
                vals[3] = vals[4] * bu_per_tonne
                vals[2] = vals[3] * 1000 / vals[1] if vals[1] else None
            vals = [None if v is None else round(v, 2) for v in vals]
            if label.lower() == "ontario":
                rec["ontario"] = vals
            elif REGION.match(label):
                rec["regions"][label] = vals
            else:
                rec["counties"][label] = vals
        out[m.group(1)] = rec
    return out


# ── 4. census-division geometry ───────────────────────────────────────────────
def tidy_geom(geom, tol: float, min_area: float):
    """Simplify, then drop islands and lakes smaller than min_area (square degrees); always keep the largest part."""
    simple = geom.simplify(tol, preserve_topology=True)
    parts = list(simple.geoms) if simple.geom_type == "MultiPolygon" else [simple]
    big = [p for p in parts if p.area >= min_area] or [max(parts, key=lambda p: p.area)]
    keep = [orient(Polygon(p.exterior, [i for i in p.interiors if Polygon(i).area >= min_area]), -1.0) for p in big]
    return keep[0] if len(keep) == 1 else MultiPolygon(keep)       # Plotly's renderer needs CLOCKWISE exteriors


def geometry() -> dict:
    attrs = json.loads(get(CD_LAYER, {"where": "PRUID='35'", "outFields": "CDUID,CDNAME", "returnGeometry": "false", "f": "json"}))
    cds = {f["attributes"]["CDUID"]: f["attributes"]["CDNAME"] for f in attrs["features"]}
    by_cd = {v: k for k, v in COUNTY_CD.items()}
    feats, cents = [], {}
    cache = Path(os.environ["GEO_CACHE_DIR"]) if os.environ.get("GEO_CACHE_DIR") else None
    for uid in sorted(cds):
        raw = cache / f"{uid}.geojson" if cache else None
        if raw is not None and raw.exists():
            gj = json.loads(raw.read_text(encoding="utf-8"))
        else:
            gj = json.loads(get(CD_LAYER, {"where": f"CDUID='{uid}'", "outFields": "CDUID", "returnGeometry": "true",
                                           "outSR": "4326", "f": "geojson"}))
            if raw is not None:
                raw.write_text(json.dumps(gj), encoding="utf-8")
        geom = shape(gj["features"][0]["geometry"])
        north = geom.centroid.y > 46.3
        simple = tidy_geom(geom, 0.04 if north else 0.012, 0.02 if north else 0.002)
        c = simple.representative_point() if not simple.contains(simple.centroid) else simple.centroid
        name = by_cd.get(uid, cds[uid])
        feats.append({"type": "Feature", "id": uid, "properties": {"NAME": name},
                      "geometry": json.loads(json.dumps(mapping(simple)), parse_float=lambda s: round(float(s), 3))})
        cents[uid] = [round(c.x, 3), round(c.y, 3), round(simple.area, 4)]
        print(f"   {uid} {name:32s} {len(json.dumps(feats[-1])) / 1024:6.1f} KB")
    return {"counties": {"type": "FeatureCollection", "features": feats}, "centroids": cents,
            "cd_name": cds, "county_cd": COUNTY_CD}


def main() -> int:
    print("StatCan Ontario series ...")
    sc = statcan()
    print("OMAFRA supply and disposition ...")
    sds = {"soy": sd(get(FILES["sd_soy"]), "soy"), "corn": sd(get(FILES["sd_corn"]), "corn")}
    print("OMAFRA county files ...")
    cty = {"soy": county(get(FILES["cty_soy"]), 36.7437), "corn": county(get(FILES["cty_corn"]), 39.3680)}
    missing = sorted({n for c in cty.values() for y in c.values() for n in y["counties"]} - set(COUNTY_CD))
    if missing:
        print("County names with no census division:", missing)
        return 1
    data = {"generated": datetime.date.today().isoformat(),
            "sources": {"statcan": "Statistics Canada table 32-10-0359-01", "omafra": "Ontario Data Catalogue (OMAFRA), "
                        "Open Government Licence - Ontario", "geometry": "Statistics Canada 2021 census division cartographic boundaries"},
            "statcan": sc, "sd": sds, "county": cty}
    (HERE / "ontario_data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print("Census division geometry ...")
    geo = geometry()
    (HERE / "ontario_geo.json").write_text(json.dumps(geo, separators=(",", ":")), encoding="utf-8")
    for n in ("ontario_data.json", "ontario_geo.json"):
        print(f"wrote {n}: {(HERE / n).stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
