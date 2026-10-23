"""New York county and ASD livestock from the Census of Agriculture (no Streamlit import).

Mirrors the NYCrop API in ny_asd_data.py so the Livestock page can follow the NASS Production page.
Data: ny_livestock_census.json (built by build_ny_livestock.py). NASS publishes county livestock only in the
Census, every 5 years, as of Dec 31. Counties NASS withholds -(D)- are estimated: the state Census total minus the
published counties is shared among the withheld counties in proportion to their farm counts (never withheld).
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).with_name("ny_livestock_census.json")

COMMODITIES = ["Dairy Cows", "All Cattle & Calves", "Hogs & Pigs"]
METRICS = ["Inventory (head)", "Farms with Inventory", "Head per Farm"]
RATIO_METRICS = {"Head per Farm"}
CHANGE_OPTS = ["Current Year", "vs Prior Census", "vs Selected Year"]
_STAT = {"Inventory (head)": "inventory", "Farms with Inventory": "farms", "Head per Farm": "per_farm"}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def years() -> list:
    return sorted(load()["years"], reverse=True)


def stat_of(metric: str) -> str:
    return _STAT[metric]


def _pct(cur, base):
    if cur is None or base is None or pd.isna(cur) or pd.isna(base) or base <= 0:
        return None
    return (cur - base) / base * 100.0


class NYLivestock:
    def __init__(self, comm: str, geo: dict):
        self.comm = comm
        self.data = load()["comm"][comm]
        self.names = {f["id"]: f["properties"]["NAME"] for f in geo["counties"]["features"]}
        self.dist = {f: (v[0], v[1]) for f, v in geo["fips_map"].items()}     # fips -> (district name, ASD code)
        self.years = years()

    # ── county values ─────────────────────────────────────────────────────────
    def state_value(self, stat: str, year: int):
        if stat == "per_farm":
            inv, farms = self.state_value("inventory", year), self.state_value("farms", year)
            return inv / farms if inv and farms else None
        return self.data[stat]["state"].get(str(year))

    def _published(self, stat: str, year: int) -> dict:
        return self.data[stat]["county"].get(str(year), {})

    def inventory_filled(self, year: int) -> tuple:
        """({fips: head}, {fips: estimated?}): withheld counties filled from the state residual by farm count."""
        pub = self._published("inventory", year)
        farms = self._published("farms", year)
        vals = {f: v for f, v in pub.items() if v is not None}
        est = {f: False for f in vals}
        withheld = [f for f, v in pub.items() if v is None]
        state = self.state_value("inventory", year)
        if withheld and state is not None:
            resid = state - sum(vals.values())
            w = {f: (farms.get(f) or 0.0) for f in withheld}
            tot = sum(w.values())
            if resid > 0 and tot > 0:
                for f in withheld:
                    if w[f] > 0:
                        vals[f] = resid * w[f] / tot
                        est[f] = True
        return vals, est

    def county_values(self, stat: str, year: int) -> pd.DataFrame:
        """[fips, County, District, DistrictCode, Value, is_estimated]; counties NASS lists no data for are absent."""
        inv, inv_est = self.inventory_filled(year)
        farms = {f: v for f, v in self._published("farms", year).items() if v is not None}
        rows = []
        if stat == "inventory":
            src, est = inv, inv_est
        elif stat == "farms":
            src, est = farms, {}
        else:
            src = {f: inv[f] / farms[f] for f in inv if farms.get(f)}
            est = inv_est
        for f, v in src.items():
            d = self.dist.get(f, (None, None))
            rows.append({"fips": f, "County": self.names.get(f, f), "District": d[0], "DistrictCode": d[1],
                         "Value": v, "is_estimated": bool(est.get(f, False))})
        return pd.DataFrame(rows, columns=["fips", "County", "District", "DistrictCode", "Value", "is_estimated"])

    def reported_counties(self, year: int) -> int:
        """Counties with a published (not withheld) inventory."""
        return sum(1 for v in self._published("inventory", year).values() if v is not None)

    def county_view(self, metric: str, year: int, change: str, comp: int | None = None) -> pd.DataFrame:
        stat = _STAT[metric]
        cur = self.county_values(stat, year)
        if change == "Current Year" or cur.empty:
            return cur
        base_year = self._base_year(year, change, comp)
        base = self.county_values(stat, base_year) if base_year else pd.DataFrame()
        bmap = dict(zip(base["fips"], base["Value"])) if not base.empty else {}
        out = cur.copy()
        out["Value"] = [_pct(v, bmap.get(f)) for f, v in zip(out["fips"], out["Value"])]
        return out.dropna(subset=["Value"]).reset_index(drop=True)

    def _base_year(self, year: int, change: str, comp: int | None):
        if change == "vs Selected Year":
            return comp
        older = [y for y in self.years if y < year]
        return older[0] if older else None

    # ── ASD districts ─────────────────────────────────────────────────────────
    def district_series(self, metric: str, year: int) -> tuple:
        """({district: value}, {district: any county estimated}) — sums counties; Head per Farm = head / farms."""
        stat = _STAT[metric]
        cur = self.county_values("inventory" if stat == "per_farm" else stat, year)
        cur = cur[cur["District"].notna()]
        if cur.empty:
            return {}, {}
        sums = cur.groupby("District")["Value"].sum()
        est = cur.groupby("District")["is_estimated"].any().to_dict()
        if stat == "per_farm":
            fm = self.county_values("farms", year)
            fm = fm[fm["District"].notna()].groupby("District")["Value"].sum()
            return {d: sums[d] / fm[d] for d in sums.index if fm.get(d)}, est
        return sums.to_dict(), est

    def district_view(self, metric: str, year: int, change: str, comp: int | None = None) -> tuple:
        """(view_df [District, Value], current_df [District, Value], estimated districts)."""
        cur, est = self.district_series(metric, year)
        cur_df = pd.DataFrame({"District": list(cur), "Value": list(cur.values())})
        if change == "Current Year" or cur_df.empty:
            return cur_df, cur_df, {d for d, e in est.items() if e}
        base_year = self._base_year(year, change, comp)
        base = self.district_series(metric, base_year)[0] if base_year else {}
        rows = [{"District": d, "Value": _pct(v, base.get(d))} for d, v in cur.items()]
        view = pd.DataFrame(rows).dropna(subset=["Value"]).reset_index(drop=True)
        return view, cur_df, {d for d, e in est.items() if e}
