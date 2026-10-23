"""NY NASS helpers for the Draw Territory tab (no Streamlit import).

Cache keys must hash identically to usda-nass-etl (jobs/domestic_production.py for crops,
jobs/livestock_inventory.py for livestock) or the lookup silently misses.
"""
import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

MIN_CACHED_YEAR = "1980"
COUNTY_FIRST_YEAR = 2015
SNAPSHOT_FILE = Path(__file__).with_name("dt_ny_snapshot.json")

CROPS: dict = {
    "Corn": {
        "Planted Acres":   {"commodity_desc": "CORN", "statisticcat_desc": "AREA PLANTED",  "unit_desc": "ACRES"},
        "Harvested Acres": {"commodity_desc": "CORN", "statisticcat_desc": "AREA HARVESTED", "unit_desc": "ACRES", "util_practice_desc": "GRAIN"},
        "Yield (Bu/Ac)":   {"commodity_desc": "CORN", "statisticcat_desc": "YIELD",          "unit_desc": "BU / ACRE", "util_practice_desc": "GRAIN"},
        "Production (Bu)": {"commodity_desc": "CORN", "statisticcat_desc": "PRODUCTION",     "unit_desc": "BU", "util_practice_desc": "GRAIN"},
    },
    "Soybeans": {
        "Planted Acres":   {"commodity_desc": "SOYBEANS", "statisticcat_desc": "AREA PLANTED",  "unit_desc": "ACRES"},
        "Harvested Acres": {"commodity_desc": "SOYBEANS", "statisticcat_desc": "AREA HARVESTED", "unit_desc": "ACRES"},
        "Yield (Bu/Ac)":   {"commodity_desc": "SOYBEANS", "statisticcat_desc": "YIELD",          "unit_desc": "BU / ACRE"},
        "Production (Bu)": {"commodity_desc": "SOYBEANS", "statisticcat_desc": "PRODUCTION",     "unit_desc": "BU"},
    },
}

# The ETL caches livestock as ONE row per short_desc (all states, all years, all domains);
# reference_period_desc here is applied locally after the row is read.
LIVESTOCK: dict = {
    "Dairy Cows":          {"Inventory (Head)": {"short_desc": "CATTLE, COWS, MILK - INVENTORY",   "reference_period_desc": "FIRST OF JAN"}},
    "All Cattle & Calves": {"Inventory (Head)": {"short_desc": "CATTLE, INCL CALVES - INVENTORY",  "reference_period_desc": "FIRST OF JAN"}},
    "Hogs & Pigs":         {"Inventory (Head)": {"short_desc": "HOGS - INVENTORY",                 "reference_period_desc": "FIRST OF DEC"}},
}

ALL: dict = {**CROPS, **LIVESTOCK}


def cache_key(params: dict) -> str:
    clean = {k: str(v) for k, v in params.items() if k not in {"key", "format"}}
    canon = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"api_GET|{canon}".encode()).hexdigest()


def state_params(mp: dict) -> dict:
    if "short_desc" in mp:
        return {"short_desc": mp["short_desc"], "agg_level_desc": "STATE"}
    base = {k: v for k, v in mp.items() if k != "reference_period_desc"}
    return {**base, "source_desc": "SURVEY", "domain_desc": "TOTAL", "freq_desc": "ANNUAL",
            "agg_level_desc": "STATE", "year__GE": MIN_CACHED_YEAR}


def county_params(mp: dict, year: int):
    """None for series the ETL does not cache at county level (livestock, planted acres)."""
    if "short_desc" in mp:
        return None
    base = {k: v for k, v in mp.items() if k != "reference_period_desc"}
    return {**base, "source_desc": "SURVEY", "domain_desc": "TOTAL", "freq_desc": "ANNUAL",
            "agg_level_desc": "COUNTY", "year": str(year)}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def state_series(records: list, mp: dict, state: str = "NY") -> pd.DataFrame:
    empty = pd.DataFrame(columns=["year", "value"])
    df = pd.DataFrame(records)
    if df.empty or "Value" not in df.columns:
        return empty
    if "state_alpha" in df.columns:
        df = df[df["state_alpha"] == state]
    if "short_desc" in mp:
        for col, want in (("source_desc", "SURVEY"), ("domain_desc", "TOTAL"),
                          ("reference_period_desc", mp["reference_period_desc"])):
            if col in df.columns:
                df = df[df[col] == want]
    elif "reference_period_desc" in df.columns:
        yr = df[df["reference_period_desc"] == "YEAR"]
        if not yr.empty:
            df = yr
    if df.empty:
        return empty
    out = pd.DataFrame({"year": pd.to_numeric(df["year"], errors="coerce"), "value": _num(df["Value"])}).dropna()
    if out.empty:
        return empty
    out["year"] = out["year"].astype(int)
    return out.drop_duplicates("year", keep="last").sort_values("year").reset_index(drop=True)


COUNTY_COLS = ["State", "County", "fips", "Value", "asd_desc", "asd_code"]


def county_frame(records: list, state: str = "NY") -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty or "Value" not in df.columns:
        return pd.DataFrame()
    if "state_alpha" in df.columns:
        df = df[df["state_alpha"] == state]
    if "county_ansi" in df.columns:
        df = df[~df["county_ansi"].isin({"998", "000", "999"})]
    if "county_name" in df.columns:
        df = df[~df["county_name"].str.strip().str.lower().str.startswith("other")]
    df = df.copy()
    if df.empty:
        return pd.DataFrame()
    df["Value"] = _num(df["Value"])
    df = df.dropna(subset=["Value"])
    if df.empty:
        return pd.DataFrame()
    if "state_fips_code" in df.columns and "county_ansi" in df.columns:
        df["fips"] = df["state_fips_code"].str.zfill(2) + df["county_ansi"].str.zfill(3)
    else:
        df["fips"] = ""
    df["State"] = state
    df["County"] = df["county_name"].str.strip().str.title() if "county_name" in df.columns else ""
    key_cols = ["State", "County", "fips"]
    if "prodn_practice_desc" in df.columns:
        all_p = df[df["prodn_practice_desc"] == "ALL PRODUCTION PRACTICES"].copy()
        rest = df[~df["fips"].isin(all_p["fips"].unique())].copy()
        if not rest.empty:
            rest = rest.loc[rest.groupby(key_cols, sort=False)["Value"].idxmax()]
        df = pd.concat([all_p, rest], ignore_index=True)
    for c in ("asd_desc", "asd_code"):
        if c not in df.columns:
            df[c] = None
    return df[COUNTY_COLS].reset_index(drop=True)


# ── Bundled snapshot (fallback when the live cache is unreachable) ───────────

def load_snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def snap_state(snap: dict, comm: str, metric: str) -> pd.DataFrame:
    rows = snap.get("state", {}).get(comm, {}).get(metric, [])
    return pd.DataFrame(rows, columns=["year", "value"])


def snap_county(snap: dict, comm: str, metric: str, year: int) -> pd.DataFrame:
    rows = snap.get("county", {}).get(comm, {}).get(metric, {}).get(str(year), [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["fips", "County", "Value", "asd_desc", "asd_code"])
    df["State"] = "NY"
    return df[COUNTY_COLS]


def build_snapshot(fetch) -> dict:
    """fetch(params) -> list of NY records, or None. Builds the snapshot payload."""
    this_year = datetime.date.today().year
    state: dict = {}
    county: dict = {}
    for comm, metrics in ALL.items():
        for metric, mp in metrics.items():
            recs = fetch(state_params(mp))
            df = state_series(recs, mp) if recs else pd.DataFrame(columns=["year", "value"])
            state.setdefault(comm, {})[metric] = [[int(r.year), float(r.value)] for r in df.itertuples()]
            if comm not in CROPS:
                continue
            for year in range(COUNTY_FIRST_YEAR, this_year + 1):
                p = county_params(mp, year)
                crecs = fetch(p) if p else None
                cdf = county_frame(crecs) if crecs else pd.DataFrame()
                if cdf.empty:
                    continue
                county.setdefault(comm, {}).setdefault(metric, {})[str(year)] = [
                    [r.fips, r.County, float(r.Value), r.asd_desc, r.asd_code] for r in cdf.itertuples()
                ]
    return {"generated": datetime.date.today().isoformat(), "state": state, "county": county}
