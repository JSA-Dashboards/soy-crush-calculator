"""Freight lanes and delivered-cost spreads from New York counties to the buyers (no Streamlit import).

Miles come from ny_lanes.json (build_ny_lanes.py: OSRM road miles, county centroid to buyer). Everything else is an
assumption you can change on the page: truck rate, load size, cross-border extras, Canada's surtax, and what each buyer
bids. Cost to a buyer = haul + border + surtax (cents per bushel); net-back = the buyer's bid basis minus that cost.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

LANES_FILE = Path(__file__).with_name("ny_lanes.json")

# $/loaded mile, bu per load, minimum $/load, cross-border rate multiplier (empty backhaul), border $/load
BENCHMARK = {"rate": 4.25, "bu_load": 833.0, "min_load": 150.0, "xb_mult": 1.30, "border": 250.0}
# Your 2026-10-22 notes: 3 cents a bushel for 152 miles = 0.02 cents per bushel-mile, with no border costs.
NOTES = {"rate": 0.17, "bu_load": 833.0, "min_load": 0.0, "xb_mult": 1.0, "border": 0.0}
PRESETS = {"Benchmark trucking": BENCHMARK, "Your notes (0.02 c/bu-mile)": NOTES}

TARIFF_CHOICES = [0.0, 0.15, 0.25, 0.50]      # Canada's surtax tiers on US goods (15 / 25 / 50%); soybeans sit at 25%
DEST_COLORS = {"union_springs": "#d97706", "york": "#16a34a", "hamilton": "#dc2626", "oswego": "#2563eb",
               "perdue_bainbridge": "#7c3aed", "perdue_lancaster": "#0d9488"}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(LANES_FILE.read_text(encoding="utf-8"))


def dests() -> list:
    return load()["dests"]


def haul_cents(miles: float, a: dict, cross_border: bool = False) -> float:
    """Truck freight in cents per bushel: miles x rate (x the cross-border multiplier), never below the minimum load."""
    per_load = max(miles * a["rate"] * (a["xb_mult"] if cross_border else 1.0), a["min_load"])
    return per_load / a["bu_load"] * 100.0


def border_cents(a: dict) -> float:
    return a["border"] / a["bu_load"] * 100.0


def surtax_cents(customs_usd: float, rate: float) -> float:
    return customs_usd * rate * 100.0


def lane_table(a: dict, tariff: float, futures_usd: float, bid_c: dict) -> pd.DataFrame:
    """One row per county x buyer: miles, hours, haul / border / surtax cents, cost, bid, net-back, spread vs Union Springs.
    bid_c = {buyer key: bid basis in cents vs futures}. The customs value for the surtax is futures + Hamilton's bid."""
    rows = []
    ds = dests()
    for fips, o in load()["origins"].items():
        for d in ds:
            m = o["miles"][d["key"]]
            if m is None:
                continue
            xb = d["cross_border"]
            haul = haul_cents(m, a, xb)
            border = border_cents(a) if xb else 0.0
            sur = surtax_cents(futures_usd + bid_c.get(d["key"], 0.0) / 100.0, tariff) if xb else 0.0
            cost = haul + border + sur
            rows.append({"fips": fips, "County": o["name"], "key": d["key"], "Buyer": d["name"], "short": d["short"],
                         "miles": m, "hours": o["hours"][d["key"]], "haul_c": haul, "border_c": border,
                         "surtax_c": sur, "cost_c": cost, "bid_c": bid_c.get(d["key"], 0.0),
                         "netback_c": bid_c.get(d["key"], 0.0) - cost})
    df = pd.DataFrame(rows)
    base = df[df["key"] == "union_springs"].set_index("fips")["netback_c"]
    df["vs_us_c"] = df["netback_c"] - df["fips"].map(base)
    return df


def winners(df: pd.DataFrame) -> pd.DataFrame:
    """Best buyer per county: the highest net-back."""
    return df.loc[df.groupby("fips")["netback_c"].idxmax()].reset_index(drop=True)
