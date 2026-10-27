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

import ny_ho

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
    "ethanol": ny_ho.ETHANOL_STYLE,                       # one marker for Ontario's plants and New York's
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
    """Share of StatCan's provincial production the published county rows miss (positive = missing). Measured in bushels: OMAFRA's
    tonnes column for soybeans is 14-19% too low in 2023-2025 against its own bushel and yield columns, so it is not used."""
    rec = load()["county"][crop][str(year)]
    tot = production_kt(crop, year)
    got = sum((v[3] or 0.0) for v in rec["counties"].values()) / (BU_PER_T[crop])
    return (tot - got) / tot if tot else 0.0


def county_table(crop: str, year: int, prod_kt=None, harv_ac=None) -> pd.DataFrame:
    """Presentation table for the county detail: reconciled (or forecast) values, largest first."""
    f = county_frame(crop, year, prod_kt, harv_ac)
    out = pd.DataFrame({"County": f["County"], "Region": f["Region"], "Harvested acres": f["harv"], "Yield (bu/ac)": f["yield"],
                        "Production (M bu)": f["prod_bu"] / 1e6, "Production (kt)": f["prod_bu"] / BU_PER_T[crop] / 1000.0,
                        "Share": f["prod_bu"] / f["prod_bu"].sum(), "Estimated": f["is_estimated"]})
    return out.sort_values("Production (M bu)", ascending=False).reset_index(drop=True)


# ── how fresh is the crop estimate, and what does it mean historically ───────
# Great Ontario Yield Tour (Farms.com Risk Management): provincial yield forecast made in August, (corn, soy) bu/ac.
TOUR = {2016: (149.0, 43.0), 2017: (163.2, 45.7), 2018: (179.5, 50.2), 2019: (173.2, 43.6), 2020: (183.3, 53.2),
        2021: (191.5, 51.7), 2022: (186.6, 50.3), 2023: (203.0, 53.5), 2024: (194.75, 54.12), 2025: (178.93, 46.61),
        2026: (206.0, 55.0)}
TOUR_SOURCE = "Great Ontario Yield Tour results pages, riskmanagement.farms.com (provincial estimate announced in August)"
TRADE_JUNE = {"soy": 4300.0, "corn": 10200.0}      # Farmtario / Jerry Klassen, 2026-06-27 (before the yield tour)


def tour_table(crop: str) -> pd.DataFrame:
    """Tour forecast against StatCan's figure for each year (the latest year is StatCan's model estimate, not final)."""
    i = 0 if crop == "corn" else 1
    rows = []
    for y, v in sorted(TOUR.items()):
        sc = statcan(crop, y).get("yield_bu")
        if sc:
            rows.append({"year": y, "tour": v[i], "statcan": sc, "ratio": sc / v[i], "final": y < latest_statcan_year(crop)})
    return pd.DataFrame(rows)


def tour_factor(crop: str) -> float:
    """Median StatCan-to-tour yield ratio over the years StatCan has finalised: the tour's bias."""
    t = tour_table(crop)
    return float(t[t["final"]]["ratio"].median())


def estimates(crop: str) -> list:
    """Estimates of the latest crop, newest official first: [{label, when, yield_bu, kt, note}]."""
    y = latest_statcan_year(crop)
    sc = statcan(crop, y)
    ac = sc.get("harvested_ac") or 0.0
    out = [{"label": "StatCan model-based estimate", "when": "Sep 16", "yield_bu": sc.get("yield_bu"),
            "kt": sc["prod_t"] / 1000.0, "note": "Satellite + weather + crop-insurance model; final survey figure due Dec 4."}]
    if y in TOUR:
        raw = TOUR[y][0 if crop == "corn" else 1]
        for label, yld, note in (
                ("Yield tour, calibrated to StatCan", raw * tour_factor(crop),
                 f"Tour yield x {tour_factor(crop):.3f}, its median StatCan-to-tour ratio since 2016."),
                ("Yield tour, as published", raw, "Field counts in August, scaled to StatCan's harvested area.")):
            out.append({"label": label, "when": "Aug 27", "yield_bu": yld, "kt": yld * ac / BU_PER_T[crop] / 1000.0, "note": note})
    if crop in TRADE_JUNE and y == 2026:
        out.append({"label": "Trade estimate (Farmtario)", "when": "Jun 27", "yield_bu": None, "kt": TRADE_JUNE[crop],
                    "note": "Market commentary before the tour and StatCan's model; not a survey."})
    return out


def history(crop: str, since: int = 1990) -> pd.DataFrame:
    rows = [{"year": int(y), "kt": v["prod_t"] / 1000.0, "yield_bu": v.get("yield_bu"), "harvested_ac": v.get("harvested_ac")}
            for y, v in load()["statcan"][crop].items() if int(y) >= since and "prod_t" in v]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def rank_info(crop: str, kt: float) -> dict:
    """Where a production figure sits among the years since 1990 (the latest year excluded)."""
    h = history(crop)
    prior = h[h["year"] < latest_statcan_year(crop)]
    avg5, avg10 = prior["kt"].tail(5).mean(), prior["kt"].tail(10).mean()
    return {"rank": int((prior["kt"] > kt).sum()) + 1, "of": len(prior) + 1, "vs_avg5": kt / avg5 - 1, "vs_avg10": kt / avg10 - 1,
            "max_year": int(prior.loc[prior["kt"].idxmax(), "year"]), "max_kt": float(prior["kt"].max()),
            "avg5": float(avg5), "avg10": float(avg10)}


def farm_stocks(crop: str) -> pd.DataFrame:
    """Latest Dec 31, Mar 31 and Jul 31 farm stocks (kt) against a year earlier and the prior five-year average."""
    s = load()["stocks"][crop]
    last = max(s)
    rows = []
    for mm, lab in (("12", "Dec 31"), ("03", "Mar 31"), ("07", "Jul 31")):
        ys = sorted(int(k[:4]) for k in s if k.endswith("-" + mm))
        y = max(yy for yy in ys if f"{yy}-{mm}" <= last)
        cur = s[f"{y}-{mm}"]
        prev5 = [s[f"{yy}-{mm}"] for yy in range(y - 5, y) if f"{yy}-{mm}" in s]
        rows.append({"date": f"{lab} {y}", "kt": cur, "year_ago": s.get(f"{y - 1}-{mm}"), "avg5": sum(prev5) / len(prev5) if prev5 else None})
    df = pd.DataFrame(rows)
    df["vs_year_ago"] = df["kt"] / df["year_ago"] - 1
    df["vs_avg5"] = df["kt"] / df["avg5"] - 1
    return df


def backcast(crop: str, production_kt: float, since: int = 2019) -> pd.DataFrame:
    """Run a production figure through each past crop year's own opening stocks, firm use and carryout: what would the
    crop have meant for exports then? Also the carryout if exports had been what they actually were."""
    df = sd_table(crop)
    done = df[df["complete"] & (df.index >= since)]
    firm_keys = [k for _, k in LINES[crop]["firm"] if k in done]
    exp_key, end_key = LINES[crop]["exports"][1], LINES[crop]["ending"][1]
    rows = []
    for y, r in done.iterrows():
        imp = r["imports"] if crop == "corn" else 0.0
        avail = r["beginning"] + production_kt + imp - sum(r[k] for k in firm_keys)
        rows.append({"year": int(y), "label": crop_year_label(int(y)), "begin": r["beginning"], "firm": sum(r[k] for k in firm_keys),
                     "ending": r[end_key], "implied_exports": avail - r[end_key], "actual_exports": r[exp_key],
                     "carryout_same_exports": avail - r[exp_key], "actual_production": r["production"]})
    return pd.DataFrame(rows)


# ── county map logic, modeled on the NASS Production tab ──────────────────────
# NASS: counties NASS withholds are estimated from olympic-average shares and scaled to reconcile with the state total, the current
# year (no county data yet) is allocated from historical shares, and the map has metric / change / district-or-county views.
# Here: published years are reconciled to OMAFRA's regional subtotals and then to StatCan's provincial figure; the forecast year is
# allocated from each county's olympic-average share of the last five years; regions (OMAFRA's five) stand in for ASD districts.
METRICS = ["Production (bu)", "Harvested Acres", "Yield (bu/ac)"]
CHANGE_OPTS = ["Current Year", "vs Prior Year", "vs Selected Year", "vs 3-Yr Avg"]
REGIONS = ["Southern Ontario", "Western Ontario", "Central Ontario", "Eastern Ontario", "Northern Ontario"]
RECONCILE_TOL = 0.01          # leave a total alone when it is already within this of its target


def region_of() -> dict:
    return load()["region_of"]


def published_years(crop: str) -> list:
    return sorted(int(y) for y in load()["county"][crop])


def is_forecast(crop: str, year: int) -> bool:
    return year not in published_years(crop)


def _reconciled(crop: str, year: int) -> pd.DataFrame:
    """One published year: [fips, County, Region, seeded, harv, prod_bu, adj, is_estimated]. Works in bushels (OMAFRA's tonnes column is
    unreliable for soybeans in 2023-2025). Counties are scaled up to their regional subtotal when they fall short of it, then everything is
    scaled to StatCan's provincial production when it is off by more than RECONCILE_TOL; `adj` is the combined factor - 1."""
    rec = load()["county"][crop][str(year)]
    names = load_geo()["county_cd"]
    reg = region_of()
    rows = []
    for name, (seeded, harv, _yld, kbu, _kt) in rec["counties"].items():
        rows.append({"fips": names[name], "County": name, "Region": reg.get(name, "Other"), "seeded": seeded or 0.0,
                     "harv": harv or 0.0, "prod_bu": (kbu or 0.0) * 1000.0, "f": 1.0})
    df = pd.DataFrame(rows)
    for r, sub in df.groupby("Region"):
        row = rec["regions"].get(r)
        if row and row[3] and 0 < sub["prod_bu"].sum() < row[3] * 1000.0 * (1 - RECONCILE_TOL):
            df.loc[sub.index, "f"] = row[3] * 1000.0 / sub["prod_bu"].sum()
    after = (df["prod_bu"] * df["f"]).sum()
    target = (production_kt(crop, year) or 0.0) * 1000.0 * BU_PER_T[crop]
    if target and after > 0 and abs(target / after - 1) > RECONCILE_TOL:
        df["f"] = df["f"] * (target / after)
    df["prod_bu"] = df["prod_bu"] * df["f"]
    df["adj"] = df["f"] - 1.0
    df["is_estimated"] = False
    return df.drop(columns=["f"])


def _olympic(values: list) -> float:
    v = sorted(values)
    if len(v) >= 4:
        v = v[1:-1]
    return float(sum(v) / len(v)) if v else 0.0


def shares(crop: str, year: int, n: int = 5) -> pd.DataFrame:
    """Each county's olympic-average share of the province over the last n published years before `year`: [fips, County, Region, prod, acres]."""
    yrs = [y for y in published_years(crop) if y < year][-n:]
    frames = [_reconciled(crop, y) for y in yrs]
    keys = pd.concat([f[["fips", "County", "Region"]] for f in frames]).drop_duplicates("fips")
    out = []
    for _, k in keys.iterrows():
        ps, ac = [], []
        for f in frames:
            tp, ta = f["prod_bu"].sum(), f["harv"].sum()
            row = f[f["fips"] == k["fips"]]
            ps.append(float(row["prod_bu"].iloc[0]) / tp if len(row) else 0.0)
            ac.append(float(row["harv"].iloc[0]) / ta if len(row) else 0.0)
        out.append({**k.to_dict(), "prod": _olympic(ps), "acres": _olympic(ac)})
    df = pd.DataFrame(out)
    df["prod"] /= df["prod"].sum()
    df["acres"] /= df["acres"].sum()
    return df


def _forecast(crop: str, year: int, prod_kt, harv_ac) -> pd.DataFrame:
    p = prod_kt if prod_kt is not None else production_kt(crop, year)
    a = harv_ac if harv_ac is not None else statcan(crop, year).get("harvested_ac")
    sh = shares(crop, year)
    df = sh[["fips", "County", "Region"]].copy()
    df["seeded"] = 0.0
    df["prod_bu"] = p * 1000.0 * BU_PER_T[crop] * sh["prod"]
    df["harv"] = (a or 0.0) * sh["acres"]
    df["adj"] = 0.0
    df["is_estimated"] = True
    return df


def county_frame(crop: str, year: int, prod_kt=None, harv_ac=None) -> pd.DataFrame:
    """County values for a year: reconciled published data, or the share-based forecast when the year has no county data."""
    df = _forecast(crop, year, prod_kt, harv_ac) if is_forecast(crop, year) else _reconciled(crop, year)
    df["yield"] = df["prod_bu"] / df["harv"].where(df["harv"] > 0)
    return df


def _unit(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """County rows as they are, or summed to OMAFRA's regions (yield = bushels / acres)."""
    if view == "County":
        out = df.rename(columns={"fips": "key", "County": "Name"})
        return out[["key", "Name", "Region", "prod_bu", "harv", "yield", "adj", "is_estimated"]].copy()
    g = df.groupby("Region").agg(prod_bu=("prod_bu", "sum"), harv=("harv", "sum"), is_estimated=("is_estimated", "any"),
                                 adj=("adj", "mean")).reset_index()
    g["yield"] = g["prod_bu"] / g["harv"].where(g["harv"] > 0)
    g["key"], g["Name"] = g["Region"], g["Region"]
    return g[["key", "Name", "Region", "prod_bu", "harv", "yield", "adj", "is_estimated"]]


_COL = {"Production (bu)": "prod_bu", "Harvested Acres": "harv", "Yield (bu/ac)": "yield"}


def county_view(crop: str, metric: str, year: int, change: str, comp=None, view: str = "County", prod_kt=None, harv_ac=None) -> pd.DataFrame:
    """[key, Name, Region, Value, raw, prod_bu, harv, yield, share, is_estimated, adj]: Value is the mapped number (a % change for
    the change views), raw the current figure, share the unit's share of the province's production."""
    col = _COL[metric]
    cur = _unit(county_frame(crop, year, prod_kt, harv_ac), view)
    cur["share"] = cur["prod_bu"] / cur["prod_bu"].sum() if cur["prod_bu"].sum() else 0.0
    cur["raw"] = cur[col]
    if change == "Current Year":
        cur["Value"] = cur["raw"]
        return cur.dropna(subset=["Value"]).reset_index(drop=True)
    if change == "vs 3-Yr Avg":
        pool = [y for y in range(year - 3, year) if y >= 2004]
    elif change == "vs Prior Year":
        pool = [year - 1]
    else:
        pool = [comp]
    bases = [_unit(county_frame(crop, y), view).set_index("key")[col] for y in pool if y is not None and y >= 2004]
    if not bases:
        return cur.iloc[0:0].assign(Value=np.nan)
    base = pd.concat(bases, axis=1).mean(axis=1)
    cur["base"] = cur["key"].map(base)
    cur["Value"] = (cur["raw"] - cur["base"]) / cur["base"].where(cur["base"] > 0) * 100.0
    return cur.dropna(subset=["Value"]).reset_index(drop=True)
