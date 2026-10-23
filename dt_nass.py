"""NY livestock helpers for the Draw Territory tab (no Streamlit import).

The ETL caches livestock as ONE row per short_desc (all states, years and domains); the
reference period / domain / source are applied locally. Keys must hash identically to
usda-nass-etl jobs/livestock_inventory.py or the lookup silently misses.
"""
import hashlib
import json

import pandas as pd

from ny_asd_data import SNAPSHOT_FILE

LIVESTOCK: dict = {
    "Dairy Cows":          {"short_desc": "CATTLE, COWS, MILK - INVENTORY",  "reference_period_desc": "FIRST OF JAN"},
    "All Cattle & Calves": {"short_desc": "CATTLE, INCL CALVES - INVENTORY", "reference_period_desc": "FIRST OF JAN"},
    "Hogs & Pigs":         {"short_desc": "HOGS - INVENTORY",                "reference_period_desc": "FIRST OF DEC"},
}


def cache_key(params: dict) -> str:
    clean = {k: str(v) for k, v in params.items() if k not in {"key", "format"}}
    canon = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"api_GET|{canon}".encode()).hexdigest()


def state_params(mp: dict) -> dict:
    return {"short_desc": mp["short_desc"], "agg_level_desc": "STATE"}


def state_series(records: list, mp: dict, state: str = "NY") -> pd.DataFrame:
    empty = pd.DataFrame(columns=["year", "value"])
    df = pd.DataFrame(records)
    if df.empty or "Value" not in df.columns:
        return empty
    if "state_alpha" in df.columns:
        df = df[df["state_alpha"] == state]
    for col, want in (("source_desc", "SURVEY"), ("domain_desc", "TOTAL"),
                      ("reference_period_desc", mp["reference_period_desc"])):
        if col in df.columns:
            df = df[df[col] == want]
    if df.empty:
        return empty
    val = pd.to_numeric(df["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    out = pd.DataFrame({"year": pd.to_numeric(df["year"], errors="coerce"), "value": val}).dropna()
    if out.empty:
        return empty
    out["year"] = out["year"].astype(int)
    return out.drop_duplicates("year", keep="last").sort_values("year").reset_index(drop=True)


def load_snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def snap_series(snap: dict, comm: str) -> pd.DataFrame:
    rows = snap.get("livestock", {}).get(comm, [])
    return pd.DataFrame(rows, columns=["year", "value"])
