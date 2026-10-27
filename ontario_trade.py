"""US soybean shipments to Canada, as the monthly signal for Ontario's imports (no Streamlit import).

Data (ontario_trade.json, built by build_ontario_trade.py): Census Bureau exports to Canada, HS10 commercial soybeans, by customs
district of exit. Detroit, Buffalo and Cleveland are the Ontario gateways; Ogdensburg (Ontario or Quebec), Pembina and the rest
are left out of the Ontario figure. Volumes are thousand tonnes (kt). Crop year here runs September to August, the US marketing year;
the OMAFRA tables elsewhere on the page run October to September, so the match is one month off.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).with_name("ontario_trade.json")
GATEWAYS = {"38": "Detroit", "09": "Buffalo", "41": "Cleveland"}
BU_PER_T = 36.7437


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def crop_year(month: str) -> int:
    y, m = int(month[:4]), int(month[5:])
    return y if m >= 9 else y - 1


@lru_cache(maxsize=1)
def monthly() -> pd.DataFrame:
    """One row per month: total to Canada, the Ontario-gateway sum, each gateway, and the crop year and Sep-Aug month number."""
    rows = []
    for m, v in load()["months"].items():
        ont = sum(v["d"].get(k, 0.0) for k in GATEWAYS)
        row = {"month": m, "total": v["total"], "ontario": ont, "cy": crop_year(m), "pos": (int(m[5:]) - 9) % 12}
        row.update({name: v["d"].get(k, 0.0) for k, name in GATEWAYS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def latest_month() -> str:
    return max(load()["months"])


def crop_years() -> pd.DataFrame:
    """Per Sep-Aug crop year: total, Ontario-gateway kt, months published, whether the year is complete, Ontario share."""
    m = monthly()
    g = m.groupby("cy").agg(total=("total", "sum"), ontario=("ontario", "sum"), months=("month", "count"))
    g["complete"] = g["months"] == 12
    g["share"] = g["ontario"] / g["total"]
    return g


def typical(latest: int, n: int = 3) -> float:
    """Average Ontario-gateway kt over the n complete crop years before `latest`."""
    g = crop_years()
    prior = g[(g.index < latest) & g["complete"]]
    return float(prior["ontario"].tail(n).mean())


def monthly_pivot(latest: int, n_hist: int = 5) -> tuple:
    """(this and the last two crop years by Sep-Aug month position, the MEDIAN of the n_hist complete years before them; a median
    because 2018/19, the trade-war year, shipped almost 4x a normal year and would drag an average up)."""
    m = monthly()
    pv = m.pivot(index="pos", columns="cy", values="ontario").sort_index()
    shown = [c for c in (latest - 2, latest - 1, latest) if c in pv.columns]
    hist = [c for c in pv.columns if c < latest - 2][-n_hist:]
    return pv[shown], pv[hist].median(axis=1), hist


def span_total(start: str, end: str) -> tuple:
    """(all-Canada kt, Ontario-gateway kt) over the inclusive month range."""
    m = monthly()
    s = m[(m["month"] >= start) & (m["month"] <= end)]
    return float(s["total"].sum()), float(s["ontario"].sum())


def to_mbu(kt: float) -> float:
    return kt * 1000.0 * BU_PER_T / 1e6


def month_label(month: str) -> str:
    import datetime
    return datetime.date(int(month[:4]), int(month[5:]), 1).strftime("%b %Y")
