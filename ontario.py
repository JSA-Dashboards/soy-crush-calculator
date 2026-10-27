"""Ontario corn and soybean supply, use and county production (no Streamlit import).

Data (ontario_data.json / ontario_geo.json, built by build_ontario_data.py):
- Statistics Canada table 32-10-0359-01: Ontario annual area, yield and production, including the latest model-based estimate.
- OMAFRA supply and disposition by crop year (October-September). The soybean file has no import line.
- OMAFRA county area and production, 2004-2025, matched to census divisions for the map.
All volumes are in thousand tonnes (kt) unless a name says otherwise.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).with_name("ontario_data.json")
GEO_FILE = Path(__file__).with_name("ontario_geo.json")

CROPS = {"Soybeans": "soy", "Corn": "corn"}
BU_PER_T = {"soy": 36.7437, "corn": 39.3680}
SCALE_GAP = 0.03          # scale counties to the provincial total when they miss it by more than this

# (label, key) lines of the OMAFRA balance sheet. "firm" uses are what Ontario must supply itself; exports are flexible.
LINES = {
    "soy": {"supply": [("Beginning stocks", "beginning"), ("Production", "production")],
            "firm": [("Processing (milling)", "milling"), ("Feed, waste & dockage", "feed"), ("Seed", "seed")],
            "exports": ("Exports", "export"), "ending": ("Ending stocks", "ending")},
    "corn": {"supply": [("Beginning stocks", "beginning"), ("Production", "production"), ("Imports", "imports")],
             "firm": [("Human consumption", "human"), ("Industrial use incl. ethanol", "industrial"),
                      ("Feed, waste & dockage", "feed"), ("Seed", "seed")],
             "exports": ("Exports", "exports"), "ending": ("Ending stocks", "ending")},
}
USE_COLORS = {"milling": "#5e7164", "human": "#a78bfa", "industrial": "#f59e0b", "feed": "#0693e3", "seed": "#9ca3af",
              "export": "#16a34a", "exports": "#16a34a", "ending": "#cbd5cf"}

# Demand and origination nodes. conf = how well the entry is sourced; the capacity notes are dated, so treat them as pointers.
NODE_KINDS = {
    "crush": {"label": "Soybean / canola crusher", "color": "#7c3aed", "symbol": "pentagon", "size": 19},
    "ethanol": {"label": "Ethanol plant (corn)", "color": "#ea580c", "symbol": "diamond", "size": 17},
    "ny": {"label": "Du Mond Grain plant", "color": "#fbbf24", "symbol": "star", "size": 22},
}
NODES = [
    dict(name="Bunge - Hamilton", kind="crush", town="Hamilton", lat=43.2702, lon=-79.8495, conf="medium",
         cap="Your notes: ~3,000 t/day, soy or canola. USDA FAS has listed ~1.05 Mt/yr canola + soy (older table); "
             "Bunge ETGO (Hamilton + Becancour QC) ~2 Mt/yr combined.",
         note="Flex plant: soy or canola. The 25% surtax is the hurdle for US beans coming in."),
    dict(name="ADM - Windsor", kind="crush", town="Windsor", lat=42.3149, lon=-83.0364, conf="low",
         cap="3,600 t/day of oilseeds in a 2015 trade report, about half soybeans then; a canola line was to become a "
             "flex line. Current capacity not confirmed.",
         note="Sits on the Detroit border, so US beans reach it without long hauls."),
    dict(name="Suncor St. Clair ethanol", kind="ethanol", town="Mooretown (Lambton Co.)", lat=42.8945, lon=-82.4736,
         conf="medium", cap="~400 million litres/yr; Suncor says ~40 million bu of corn a year (about a fifth of Ontario's crop).",
         note="Canada's largest ethanol plant."),
    dict(name="IGPC Ethanol - Aylmer", kind="ethanol", town="Aylmer (Elgin Co.)", lat=42.7686, lon=-80.9831, conf="low",
         cap="~150 million litres/yr and ~15 million bu of corn in a 2009 report; a doubling to 378 million litres was "
             "announced later, completion not confirmed.",
         note=""),
    dict(name="Greenfield Global - Chatham", kind="ethanol", town="Chatham (Chatham-Kent)", lat=42.4048, lon=-82.1910,
         conf="low", cap="Capacity not confirmed in the sources checked.", note=""),
    dict(name="Greenfield Global - Johnstown", kind="ethanol", town="Johnstown (Leeds and Grenville)", lat=44.7253,
         lon=-75.4936, conf="low", cap="~20 million bu of corn (2008 report); ~250 million litres (2020 release), part "
                                       "since moved to high-purity alcohol.", note="Eastern Ontario corn demand."),
]


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_geo() -> dict:
    return json.loads(GEO_FILE.read_text(encoding="utf-8"))


def crop_year_label(y: int) -> str:
    return f"{y % 100:02d}/{(y + 1) % 100:02d}"


# ── supply and disposition ────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def sd_table(crop: str) -> pd.DataFrame:
    """One row per crop year (index = the October start year), kt. `complete` = every use line is published."""
    raw = load()["sd"][crop]
    df = pd.DataFrame({k: {int(y): v for y, v in d.items()} for k, d in raw.items()}).sort_index()
    use_keys = [k for _, k in LINES[crop]["firm"][:-1]] + [LINES[crop]["exports"][1], LINES[crop]["ending"][1]]
    df["complete"] = df[[k for k in use_keys if k in df]].notna().all(axis=1)
    if crop == "corn":
        df["imports"] = df["imports"].fillna(0.0)
    return df


def statcan(crop: str, year: int) -> dict:
    return load()["statcan"][crop].get(str(year), {})


def production_kt(crop: str, year: int):
    t = statcan(crop, year).get("prod_t")
    return None if t is None else t / 1000.0


def latest_statcan_year(crop: str) -> int:
    return max(int(y) for y in load()["statcan"][crop])


def defaults(crop: str) -> dict:
    """Scenario defaults from the published years: recent averages of each use line."""
    df = sd_table(crop)
    done = df[df["complete"]]
    keys = [k for _, k in LINES[crop]["firm"]]
    out = {k: float(done[k].tail(3).mean()) for k in keys if k in done}
    if crop == "soy":
        out["milling"] = float(done["milling"].tail(2).mean())       # processing stepped up in 2019/20; use the recent run
    out["exports"] = float(done[LINES[crop]["exports"][1]].tail(3).mean())
    out["target"] = float(done["ending"].tail(5).mean())
    out["imports"] = float(done["imports"].tail(3).mean()) if crop == "corn" else 0.0
    out["begin"] = float(done["ending"].iloc[-1])
    out["last_complete"] = int(done.index[-1])
    return out


def balance(crop: str, begin: float, production: float, a: dict) -> dict:
    """One crop year. Firm uses and the carryout target come first; what is left can be exported, and a negative
    remainder is the import need (Ontario cannot feed its own processors and hold the target stocks)."""
    firm = sum(a[k] for _, k in LINES[crop]["firm"] if k in a)
    supply = begin + production + (a.get("imports", 0.0) if crop == "corn" else 0.0)
    net = supply - firm - a["target"]
    exportable = max(net, 0.0)
    exports = min(exportable, a["exports"])
    return {"begin": begin, "production": production, "supply": supply, "firm": firm, "net": net,
            "import_need": max(-net, 0.0), "exportable": exportable, "exports": exports,
            "ending": a["target"] + (exportable - exports), "typical_exports": a["exports"],
            "export_gap": a["exports"] - exportable}


def headroom(crop: str, b: dict, a: dict) -> float:
    """Processing (or total firm use) the year's supply can cover while still holding the carryout target."""
    other = b["firm"] - (a["milling"] if crop == "soy" else 0.0)
    return b["supply"] - a["target"] - other


def chain(crop: str, years: tuple, a: dict, begin: float, prods: dict | None = None) -> list:
    """Run `years` (harvest years) in order; each year's ending stocks open the next. `prods` overrides production (kt)."""
    out = []
    for y in years:
        p = (prods or {}).get(y)
        p = production_kt(crop, y) if p is None else p
        b = balance(crop, begin, p, a)
        b["year"] = y
        out.append(b)
        begin = b["ending"]
    return out


def to_mbu(crop: str, kt: float) -> float:
    return kt * 1000.0 * BU_PER_T[crop] / 1e6


# ── county production ─────────────────────────────────────────────────────────
def county_years(crop: str) -> list:
    return sorted((int(y) for y in load()["county"][crop]), reverse=True)


def county_gap(crop: str, year: int) -> float:
    """Share of the StatCan provincial production the county rows miss (positive = missing)."""
    rec = load()["county"][crop][str(year)]
    tot = production_kt(crop, year)
    got = sum(v[4] for v in rec["counties"].values())
    return (tot - got) / tot if tot else 0.0


def county_table(crop: str, year: int, scale: bool) -> pd.DataFrame:
    """[fips, County, Harvested acres, Yield (bu/ac), Production (kt), Production (M bu), Share]. With `scale`, production is
    scaled up pro rata so the counties add up to StatCan's provincial figure."""
    rec = load()["county"][crop][str(year)]
    names = {n: u for n, u in load_geo()["county_cd"].items()}
    k = 1.0
    if scale:
        got = sum(v[4] for v in rec["counties"].values())
        tot = production_kt(crop, year)
        k = tot / got if got and tot else 1.0
    rows = []
    for name, (seeded, harv, yld, kbu, kt) in rec["counties"].items():
        rows.append({"fips": names[name], "County": name, "Seeded acres": seeded, "Harvested acres": harv,
                     "Yield (bu/ac)": yld, "Production (kt)": kt * k, "Production (M bu)": (kbu or 0.0) * k / 1000.0})
    df = pd.DataFrame(rows)
    df["Share"] = df["Production (kt)"] / df["Production (kt)"].sum() if len(df) else np.nan
    return df.sort_values("Production (kt)", ascending=False).reset_index(drop=True)
