"""New York NASS production data and ASD/county estimation (no Streamlit import).

Port of the NASS Production tab logic in rma-production-map, restricted to New York and
reading the shared usda-nass-etl cache (jobs/rma_map.py) instead of calling NASS live.
Cache keys must hash identically to that job list or lookups silently miss.

County values NASS withholds are lumped into an "OTHER COUNTIES" row. Missing counties are
estimated from each county's olympic-average historical share of the state total, scaled by a
district performance multiplier, then reconciled so named + estimated counties add up to NASS.
"""
import hashlib
import json
import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATE = "NY"
STATE_FIPS = "36"
YEARS = list(range(2026, 2014, -1))          # same list as the ASD dashboard's NASS_YEARS
FORECAST_YEAR = 2026
FIRST_HIST_YEAR = 2015
BENCHMARK_YEAR = 2023                        # most complete county year, for the coverage KPI
SNAPSHOT_FILE = Path(__file__).with_name("ny_nass_snapshot.json")

METRICS = ["Production (bu)", "Total Supply (bu)", "Planted Acres", "Harvested Acres",
           "% Harvested", "Yield (bu/ac)", "Prevent Plant Acres"]
CHANGE_OPTS = ["Current Year", "vs Prior Year", "vs Selected Year", "vs 3-Yr Avg"]
METRIC_TO_STAT = {
    "Production (bu)": "production", "Total Supply (bu)": "production",
    "Planted Acres": "planted", "Harvested Acres": "harvested",
    "% Harvested": "pct_harvested", "Yield (bu/ac)": "yield",
    "Prevent Plant Acres": "prevent_plant",
}
RATIO_METRICS = ("Yield (bu/ac)", "% Harvested")
ESTIMABLE = ("production", "planted", "harvested", "yield")

# Mirrors jobs/rma_map.py STAT_BASE / CROP_STAT_PARAMS / CROPS_BASE.
STAT_BASE = {
    "production": {"statisticcat_desc": "PRODUCTION", "unit_desc": "BU"},
    "planted":    {"statisticcat_desc": "AREA PLANTED", "unit_desc": "ACRES"},
    "harvested":  {"statisticcat_desc": "AREA HARVESTED", "unit_desc": "ACRES"},
    "yield":      {"statisticcat_desc": "YIELD", "unit_desc": "BU / ACRE"},
}
CROP_STAT_PARAMS = {
    "Corn": {
        "production": {"commodity_desc": "CORN", "util_practice_desc": "GRAIN"},
        "planted":    {"commodity_desc": "CORN"},
        "harvested":  {"commodity_desc": "CORN", "util_practice_desc": "GRAIN"},
        "yield":      {"commodity_desc": "CORN", "util_practice_desc": "GRAIN"},
    },
    "Soybeans": {
        "production": {"commodity_desc": "SOYBEANS"},
        "planted":    {"commodity_desc": "SOYBEANS"},
        "harvested":  {"commodity_desc": "SOYBEANS"},
        "yield":      {"commodity_desc": "SOYBEANS"},
    },
}
CROPS = list(CROP_STAT_PARAMS)
CROPS_BASE = {"source_desc": "SURVEY", "sector_desc": "CROPS"}
ALL_PRACTICES = "ALL PRODUCTION PRACTICES"
_BAD_ANSI = {"998", "000", "999"}


def cache_key(params: dict) -> str:
    clean = {k: str(v) for k, v in params.items() if k not in {"key", "format"}}
    canon = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"api_GET|{canon}".encode()).hexdigest()


def requests_for(crop: str) -> list:
    """[(kind, stat, year, params)] for every cached row this tab can use.
    kind: c = county (domain TOTAL, incl. OTHER COUNTIES), s = state (final YEAR),
          f = current-year state forecast (any reference period), k = Sep 1 stocks."""
    out = []
    for stat in STAT_BASE:
        q = {**STAT_BASE[stat], **CROP_STAT_PARAMS[crop][stat]}
        for y in sorted(YEARS):
            out.append(("c", stat, y, {**CROPS_BASE, "agg_level_desc": "COUNTY",
                                       "domain_desc": "TOTAL", "year": str(y), **q}))
            out.append(("s", stat, y, {**CROPS_BASE, "agg_level_desc": "STATE", "domain_desc": "TOTAL",
                                       "reference_period_desc": "YEAR", "year": str(y), **q}))
        out.append(("f", stat, FORECAST_YEAR, {**CROPS_BASE, "agg_level_desc": "STATE",
                                               "domain_desc": "TOTAL", "year": str(FORECAST_YEAR), **q}))
    commodity = CROP_STAT_PARAMS[crop]["production"]["commodity_desc"]
    for y in sorted(YEARS):
        out.append(("k", "stocks", y, {**CROPS_BASE, "statisticcat_desc": "STOCKS", "unit_desc": "BU",
                                       "agg_level_desc": "STATE", "domain_desc": "TOTAL",
                                       "reference_period_desc": "FIRST OF SEP", "year": str(y),
                                       "commodity_desc": commodity}))
    return out


ROW_COLS = ["kind", "stat", "req_year", "year", "county_name", "county_ansi", "asd_desc", "asd_code",
            "practice", "ref_period", "short_desc", "value"]


def fetch_crop_rows(crop: str, run_query) -> pd.DataFrame:
    """One server-side-filtered query for every NY row this crop needs.
    run_query(sql) -> list of row tuples. Raises on connection/query failure."""
    reqs = requests_for(crop)
    by_key = {cache_key(p): (kind, stat, y) for kind, stat, y, p in reqs}
    inlist = ",".join("'%s'" % k for k in by_key)          # sha256 hex only
    sql = f"""
        SELECT c.cache_key, f.value:year::string, f.value:county_name::string, f.value:county_ansi::string,
               f.value:asd_desc::string, f.value:asd_code::string, f.value:prodn_practice_desc::string,
               f.value:reference_period_desc::string, f.value:short_desc::string, f.value:"Value"::string
        FROM JSA.NASS_CACHE.nass_cache c, LATERAL FLATTEN(input => c.data:data) f
        WHERE c.cache_key IN ({inlist}) AND f.value:state_alpha::string = '{STATE}'"""
    rows = []
    for r in run_query(sql):
        kind, stat, y = by_key[r[0]]
        rows.append([kind, stat, y, *r[1:]])
    return pd.DataFrame(rows, columns=ROW_COLS)


def frame_to_snapshot(df: pd.DataFrame) -> dict:
    return {"cols": ROW_COLS, "rows": df[ROW_COLS].where(df[ROW_COLS].notna(), None).values.tolist()}


def snapshot_to_frame(obj: dict) -> pd.DataFrame:
    return pd.DataFrame(obj["rows"], columns=obj["cols"])


def load_snapshot() -> dict:
    try:
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass(frozen=True)
class Adj:
    """Per-session adjustments: NY high-oleic share carved out of soybeans (0-1), and total supply
    (production + Sep 1 stocks, spread over counties in proportion to production)."""
    ho: float = 0.0
    ts: bool = False


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _olympic(vals):
    v = sorted(vals)
    return float(np.mean(v[1:-1] if len(v) >= 4 else v))


class NYCrop:
    """All NY rows for one crop, with the ASD dashboard's loaders as methods."""

    def __init__(self, crop: str, rows: pd.DataFrame, fips_map: dict | None = None):
        self.crop = crop
        self._static_fips_map = {k: tuple(v) for k, v in (fips_map or {}).items()}
        df = rows.copy()
        df["req_year"] = df["req_year"].astype(int)
        df["val"] = _num(df["value"])
        df["County"] = df["county_name"].fillna("").str.strip().str.title()
        df["ANSI"] = df["county_ansi"].fillna("").astype(str).str.strip().str.zfill(3)
        df["fips"] = STATE_FIPS + df["ANSI"]
        df["District"] = df["asd_desc"].fillna("").str.strip().str.title()
        df["DistrictCode"] = df["asd_code"].fillna("").astype(str).str.strip()
        df["practice"] = df["practice"].fillna("")
        df["is_pp"] = df["short_desc"].fillna("").str.upper().str.contains("PREVENT")
        self.df = df
        self._groups = {k: g for k, g in df.groupby(["kind", "stat", "req_year"])}
        self._memo: dict = {}

    # ── helpers ───────────────────────────────────────────────────────────────
    def _rows(self, kind: str, stat: str, year: int) -> pd.DataFrame:
        base = "planted" if stat == "prevent_plant" else stat
        g = self._groups.get((kind, base, year))
        if g is None:
            return pd.DataFrame(columns=self.df.columns)
        if base == "planted":
            return g[g["is_pp"]] if stat == "prevent_plant" else g[~g["is_pp"]]
        return g

    def _memoize(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def _ho(self, stat: str, adj: Adj) -> float:
        if adj.ho > 0 and self.crop == "Soybeans" and stat in ("production", "planted", "harvested"):
            return 1.0 - adj.ho
        return 1.0

    def stocks(self, year: int) -> float | None:
        """Sep 1 NY stocks (total, excluding the on-/off-farm splits)."""
        g = self._groups.get(("k", "stocks", year))
        if g is None:
            return None
        sd = g["short_desc"].fillna("").str.upper()
        tot = g[~sd.str.contains("ON FARM|OFF FARM", na=False)]["val"].dropna()
        return float(tot.sum()) if len(tot) else None

    def _ts_factor(self, year: int, adj: Adj) -> float:
        if not adj.ts:
            return 1.0
        stocks = self.stocks(year)
        prod = self._state_raw("production", year, ho=self._ho("production", adj))
        if prod and prod > 0 and stocks and stocks > 0:
            return (prod + stocks) / prod
        return 1.0

    def _scale(self, stat: str, year: int, adj: Adj) -> float:
        f = self._ho(stat, adj)
        if stat == "production":
            f *= self._ts_factor(year, adj)
        return f

    # ── state level ───────────────────────────────────────────────────────────
    def _state_raw(self, stat: str, year: int, ho: float = 1.0):
        g = self._rows("s", stat, year)
        if g.empty:
            return None
        pref = g[g["practice"] == ALL_PRACTICES]
        g = pref if not pref.empty else g
        v = g["val"].fillna(0).max()
        return float(v) * ho

    def state_total(self, stat: str, year: int, adj: Adj = Adj()):
        """Official NY total (None when NASS has not published it)."""
        if stat == "pct_harvested":
            h = self.state_total("harvested", year, adj)
            p = self.state_total("planted", year, adj)
            return h / p * 100 if h is not None and p else None
        v = self._state_raw(stat, year, ho=self._ho(stat, adj))
        if v is None:
            return None
        return v * (self._ts_factor(year, adj) if stat == "production" else 1.0)

    def state_series(self, stat: str, adj: Adj = Adj()) -> dict:
        return {y: self.state_total(stat, y, adj) for y in sorted(YEARS)}

    def forecast_state(self, stat: str, adj: Adj = Adj()):
        """Current-year NY value from any reference period (preferring NASS's latest in-season YEAR row)."""
        g = self._groups.get(("f", stat, FORECAST_YEAR))
        if g is None or g.empty:
            return None
        g = g[~g["is_pp"]] if stat == "planted" else g
        g = g.dropna(subset=["val"])
        if g.empty:
            return None
        # The YEAR row moves with each monthly report; the Jun/Mar Acreage rows are frozen snapshots.
        pref = {"YEAR": 0, "YEAR - JUN ACREAGE": 1, "YEAR - MAR ACREAGE": 2}
        g = g.assign(_p=g["ref_period"].map(lambda x: pref.get(x, 99))).sort_values("_p")
        return float(g.iloc[0]["val"]) * self._ho(stat, adj)

    # ── county level ──────────────────────────────────────────────────────────
    def county_raw(self, stat: str, year: int, adj: Adj = Adj()) -> pd.DataFrame:
        """County rows INCLUDING the OTHER COUNTIES catch-all (the estimation input)."""
        def build():
            g = self._rows("c", stat, year)
            if g.empty:
                return pd.DataFrame()
            out = g[["County", "fips", "ANSI", "District", "DistrictCode", "practice"]].copy()
            out["State"] = STATE
            out["prodn_practice_desc"] = out.pop("practice")
            out["Value_num"] = g["val"].fillna(0) * self._scale(stat, year, adj)
            return out.reset_index(drop=True)
        return self._memoize(("raw", stat, year, adj), build)

    def named_county(self, stat: str, year: int, adj: Adj = Adj()) -> pd.DataFrame:
        """Reported counties only: [State, County, fips, Value, District, DistrictCode]."""
        cols = ["State", "County", "fips", "Value", "District", "DistrictCode"]
        if stat == "pct_harvested":
            pl = self.named_county("planted", year, adj)
            hv = self.named_county("harvested", year, adj)
            if pl.empty or hv.empty:
                return pd.DataFrame(columns=cols)
            m = pl.merge(hv[["fips", "Value"]].rename(columns={"Value": "Harv"}), on="fips", how="inner")
            m["Value"] = m["Harv"] / m["Value"].replace(0, np.nan) * 100
            return m[cols].dropna(subset=["Value"]).reset_index(drop=True)

        def build():
            raw = self.county_raw(stat, year, adj)
            if raw.empty:
                return pd.DataFrame(columns=cols)
            named = raw[~raw["County"].str.lower().str.startswith("other") & ~raw["ANSI"].isin(_BAD_ANSI)]
            if named.empty:
                return pd.DataFrame(columns=cols)
            pref = named[named["prodn_practice_desc"] == ALL_PRACTICES]
            rest = named[~named["fips"].isin(pref["fips"].unique())]
            if not rest.empty:
                rest = rest.loc[rest.groupby("fips")["Value_num"].idxmax()]
            out = pd.concat([pref, rest], ignore_index=True)
            out = out.loc[out.groupby("fips")["Value_num"].idxmax()]
            out = out.rename(columns={"Value_num": "Value"})
            return out[cols].reset_index(drop=True)
        return self._memoize(("named", stat, year, adj), build)

    def reported_counties(self, year: int) -> int:
        return int(self.named_county("production", year)["fips"].nunique())

    def fips_map(self) -> dict:
        """{fips: (District, DistrictCode)}: the full map built from the whole ASD cache when given,
        else whatever this crop's own county rows reveal."""
        if self._static_fips_map:
            return self._static_fips_map

        def build():
            d = self.df[(self.df["kind"] == "c") & (self.df["District"] != "")
                        & ~self.df["ANSI"].isin(_BAD_ANSI)]
            out = {}
            for fips, g in d.groupby("fips"):
                m = g.groupby(["District", "DistrictCode"]).size().sort_values(ascending=False)
                out[fips] = tuple(m.index[0])
            return out
        return self._memoize(("fips_map",), build)

    # ── Tier-1 estimation ─────────────────────────────────────────────────────
    def hist_shares(self, stat: str, hist_years: tuple, adj: Adj = Adj()) -> pd.DataFrame:
        def build():
            rows = []
            for yr in hist_years:
                st_val = self.state_total(stat, yr, adj)
                if not st_val or st_val <= 0:
                    continue
                df = self.county_raw(stat, yr, adj)
                if df.empty:
                    continue
                named = df[(~df["ANSI"].isin(_BAD_ANSI))
                           & (~df["County"].str.lower().str.startswith("other", na=False))]
                named = named[named["prodn_practice_desc"] == ALL_PRACTICES]
                if named.empty:
                    continue
                named = named.loc[named.groupby("fips")["Value_num"].idxmax()]
                for r in named.itertuples():
                    rows.append({"fips": r.fips, "County": r.County, "District": r.District,
                                 "DistrictCode": r.DistrictCode, "year": yr,
                                 "share": r.Value_num / st_val})
            if not rows:
                return pd.DataFrame()
            hist = pd.DataFrame(rows)
            out = []
            for (fips, county), g in hist.groupby(["fips", "County"]):
                out.append({"fips": fips, "County": county,
                            "hist_share": _olympic(g["share"].tolist()), "n_obs": len(g),
                            "District": g["District"].mode().iloc[0],
                            "DistrictCode": g["DistrictCode"].mode().iloc[0]})
            return pd.DataFrame(out)
        return self._memoize(("hs", stat, hist_years, adj), build)

    def completed(self, stat: str, year: int, adj: Adj = Adj()) -> pd.DataFrame:
        """Reported + estimated counties: [fips, County, District, DistrictCode, Value, is_estimated].
        Named + estimated sum (mean for yield) to the NASS state total."""
        out_cols = ["fips", "County", "District", "DistrictCode", "Value", "is_estimated"]

        def build():
            raw = self.county_raw(stat, year, adj)
            if raw.empty:
                return pd.DataFrame(columns=out_cols)
            state_total = self.state_total(stat, year, adj)
            if state_total is None:
                return pd.DataFrame(columns=out_cols)
            is_other = raw["County"].str.lower().str.startswith("other", na=False)
            other_tot = raw[is_other]["Value_num"].sum()
            named = raw[~is_other & ~raw["ANSI"].isin(_BAD_ANSI)].copy()
            named = named[named["prodn_practice_desc"] == ALL_PRACTICES]
            if named.empty:
                return pd.DataFrame(columns=out_cols)
            named = named.loc[named.groupby("fips")["Value_num"].idxmax()]
            named["is_estimated"] = False
            named["Value"] = named["Value_num"]
            keep = named[out_cols]
            is_yield = stat == "yield"
            if (not is_yield and other_tot <= 0):
                return keep.reset_index(drop=True)
            hist_yrs = tuple(range(max(year - 8, FIRST_HIST_YEAR), year))
            shares = self.hist_shares(stat, hist_yrs, adj)
            if shares.empty:
                return keep.reset_index(drop=True)
            missing = shares[~shares["fips"].isin(set(named["fips"]))].copy()
            if missing.empty:
                return keep.reset_index(drop=True)
            hs_map = dict(zip(shares["fips"], shares["hist_share"]))
            dev = {r.fips: r.Value_num / (hs_map[r.fips] * state_total)
                   for r in named.itertuples()
                   if r.fips in hs_map and hs_map[r.fips] > 0 and state_total}
            nr = named.assign(ratio=named["fips"].map(dev)).dropna(subset=["ratio"])
            dist_ratio = nr.groupby("District")["ratio"].mean().to_dict()
            state_ratio = float(np.mean(list(dev.values()))) if dev else 1.0
            missing["mult"] = missing["District"].map(lambda d: dist_ratio.get(d, state_ratio))
            missing["raw_est"] = missing["hist_share"] * state_total * missing["mult"]
            if not is_yield:
                raw_sum = missing["raw_est"].sum()
                missing["Value"] = missing["raw_est"] * ((other_tot / raw_sum) if raw_sum > 0 else 1.0)
            else:
                missing["Value"] = missing["raw_est"]
            missing["is_estimated"] = True
            return pd.concat([keep, missing[out_cols]], ignore_index=True)
        return self._memoize(("comp", stat, year, adj), build)

    def asd_hist_shares(self, stat: str, hist_years: tuple, adj: Adj = Adj()) -> dict:
        yearly: dict = {}
        for yr in hist_years:
            cdf = self.completed(stat, yr, adj)
            if cdf.empty:
                continue
            st_val = self.state_total(stat, yr, adj)
            if not st_val or st_val <= 0:
                continue
            ratio = stat in ("yield", "pct_harvested")
            dv = cdf.groupby("District")["Value"].mean() if ratio else cdf.groupby("District")["Value"].sum()
            for dist, dval in dv.items():
                yearly.setdefault(dist, []).append(float(dval) / st_val)
        return {d: _olympic(v) for d, v in yearly.items() if v}

    # ── current-year forecast ─────────────────────────────────────────────────
    def default_yield_sources(self, adj: Adj = Adj()) -> dict:
        out = {"usda": None, "ly": None}
        cur = self.state_total("yield", FORECAST_YEAR, adj)
        if cur and cur > 0:
            out["usda"] = round(cur, 1)
        for yr in (FORECAST_YEAR - 1, FORECAST_YEAR - 2):
            v = self.state_total("yield", yr, adj)
            if v:
                out["ly"] = round(v, 1)
                break
        return out

    def asd_forecast(self, yield_est: float, adj: Adj = Adj()) -> tuple:
        """(stats {stat: {district: value}}, state_totals {stat: value}) for FORECAST_YEAR."""
        hist = tuple(range(FORECAST_YEAR - 5, FORECAST_YEAR))
        st_pl = self.forecast_state("planted", adj) or 0.0
        st_hv = self.forecast_state("harvested", adj) or 0.0
        st_pr = yield_est * st_hv if (yield_est and st_hv) else 0.0
        stk = self.stocks(FORECAST_YEAR) or 0.0
        st_sup = st_pr + (stk if (adj.ts and st_pr) else 0.0)
        sh_pl = self.asd_hist_shares("planted", hist, adj)
        sh_hv = self.asd_hist_shares("harvested", hist, adj)
        sh_pr = self.asd_hist_shares("production", hist, adj)

        def _norm(sh: dict) -> dict:
            # Per-district olympic averages don't sum to 1; rescale so districts add to the state figure.
            tot = sum(sh.values())
            return {d: v / tot for d, v in sh.items()} if tot > 0 else sh
        sh_pl, sh_hv, sh_pr = _norm(sh_pl), _norm(sh_hv), _norm(sh_pr)
        stats: dict = {k: {} for k in ("planted", "harvested", "pct_harvested", "production", "yield")}
        for d in set(sh_pl) | set(sh_hv) | set(sh_pr):
            apl = st_pl * sh_pl[d] if (d in sh_pl and st_pl) else None
            ahv = st_hv * sh_hv[d] if (d in sh_hv and st_hv) else None
            apr = st_pr * sh_pr[d] if (d in sh_pr and st_pr) else None
            stats["planted"][d] = apl
            stats["harvested"][d] = ahv
            stats["production"][d] = (st_sup * sh_pr[d]) if (d in sh_pr and st_sup) else None
            if apl and ahv and apl > 0:
                stats["pct_harvested"][d] = ahv / apl * 100
            if ahv and apr and ahv > 0:
                stats["yield"][d] = apr / ahv
        totals = {"planted": st_pl, "harvested": st_hv, "production": st_sup,
                  "yield": yield_est or 0.0, "pct_harvested": st_hv / st_pl * 100 if st_pl else 0.0}
        return stats, totals

    # ── aggregation for the views ─────────────────────────────────────────────
    def district_series(self, stat: str, year: int, adj: Adj) -> tuple:
        """({district: value}, {district: any_estimated}) for one stat/year."""
        if stat == "pct_harvested":
            pl, hv = self.completed("planted", year, adj), self.completed("harvested", year, adj)
            if pl.empty or hv.empty:
                return {}, {}
            p = pl.groupby("District")["Value"].sum()
            h = hv.groupby("District")["Value"].sum()
            vals = {d: float(h[d] / p[d] * 100) for d in p.index if d in h.index and p[d] > 0}
            est = {d: bool(pl[pl["District"] == d]["is_estimated"].any()
                           or hv[hv["District"] == d]["is_estimated"].any()) for d in vals}
            return vals, est
        if stat in ESTIMABLE:
            cdf = self.completed(stat, year, adj)
            if cdf.empty:
                return {}, {}
            ratio = stat == "yield"
            vals, est = {}, {}
            for dist, g in cdf.groupby("District"):
                if not dist:
                    continue
                vals[dist] = float(g["Value"].mean() if ratio else g["Value"].sum())
                est[dist] = bool(g["is_estimated"].any())
            return vals, est
        nc = self.named_county(stat, year, adj)
        if nc.empty:
            return {}, {}
        fm = self.fips_map()
        nc = nc.assign(District=nc["fips"].map(lambda f: fm.get(f, ("", ""))[0])).query("District != ''")
        vals = {d: float(g["Value"].sum()) for d, g in nc.groupby("District")}
        return vals, {d: False for d in vals}

    def district_view(self, metric: str, year: int, change_view: str, comp_year, adj: Adj = Adj(),
                      fc_stats: dict | None = None) -> tuple:
        """(view_df [District, Value], current_abs_df, estimated_districts). View = absolute value or
        the % change vs the comparison period, computed on district totals."""
        stat = METRIC_TO_STAT[metric]
        years = {year}
        if change_view == "vs Prior Year":
            years.add(year - 1)
        elif change_view == "vs Selected Year" and comp_year:
            years.add(comp_year)
        elif change_view == "vs 3-Yr Avg":
            years |= {y for y in (year - 1, year - 2, year - 3) if y >= FIRST_HIST_YEAR}
        vals, est = {}, {}
        for y in sorted(years):
            if y == FORECAST_YEAR and fc_stats:
                fv = {d: v for d, v in fc_stats.get(stat, {}).items() if v is not None}
                vals[y], est[y] = fv, {d: True for d in fv}
            else:
                vals[y], est[y] = self.district_series(stat, y, adj)
        cur = vals.get(year, {})
        if change_view == "Current Year":
            view = dict(cur)
        else:
            if change_view == "vs Prior Year":
                base = vals.get(year - 1, {})
            elif change_view == "vs Selected Year" and comp_year:
                base = vals.get(comp_year, {})
            else:
                maps = [vals[y] for y in (year - 1, year - 2, year - 3) if y >= FIRST_HIST_YEAR and y in vals]
                keys = {d for m in maps for d in m}
                base = {d: sum(m[d] for m in maps if d in m) / sum(1 for m in maps if d in m) for d in keys}
            view = {d: (v - base[d]) / abs(base[d]) * 100 for d, v in cur.items() if v and base.get(d)}
        mk = lambda d: pd.DataFrame(list(d.items()), columns=["District", "Value"]).dropna(subset=["Value"])
        return mk(view), mk(cur), {d for d, e in est.get(year, {}).items() if e}

    def county_view(self, metric: str, year: int, change_view: str, comp_year, adj: Adj = Adj()) -> pd.DataFrame:
        """[County, fips, Value, District, is_estimated] for the county map and ranking.
        Current Year on an estimable stat uses completed (reported + estimated) counties."""
        cols = ["County", "fips", "Value", "District", "DistrictCode", "is_estimated"]
        stat = METRIC_TO_STAT[metric]
        if change_view == "Current Year" and stat in ESTIMABLE:
            c = self.completed(stat, year, adj)
            out = c if not c.empty else pd.DataFrame(columns=cols)
        else:
            cur = self.named_county(stat, year, adj)
            if change_view == "Current Year" or cur.empty:
                out = cur.assign(is_estimated=False)
            else:
                if change_view == "vs Prior Year":
                    frames = [self.named_county(stat, year - 1, adj)]
                elif change_view == "vs Selected Year" and comp_year:
                    frames = [self.named_county(stat, comp_year, adj)]
                else:
                    frames = [self.named_county(stat, y, adj) for y in (year - 1, year - 2, year - 3)
                              if y >= FIRST_HIST_YEAR]
                frames = [f for f in frames if not f.empty]
                if not frames:
                    return pd.DataFrame(columns=cols)
                base = pd.concat(frames).groupby("fips")["Value"].mean().rename("Base")
                m = cur.merge(base, on="fips", how="inner")
                m["Value"] = (m["Value"] - m["Base"]) / m["Base"].replace(0, np.nan) * 100
                m["is_estimated"] = False
                out = m.dropna(subset=["Value"])
        out = out.copy()
        fm = self.fips_map()
        if len(out):
            out["District"] = [fm.get(f, (d, c))[0] for f, d, c in zip(out["fips"], out["District"], out["DistrictCode"])]
            out["DistrictCode"] = [fm.get(f, (d, c))[1] for f, d, c in zip(out["fips"], out["District"], out["DistrictCode"])]
        return out[cols].reset_index(drop=True)

    def state_view_value(self, metric: str, year: int, change_view: str, comp_year, adj: Adj = Adj()):
        """Official NY value for the KPI: absolute, or % change vs the comparison period."""
        stat = METRIC_TO_STAT[metric]
        cur = self.state_total(stat, year, adj)
        if change_view == "Current Year" or cur is None:
            return cur
        if change_view == "vs Prior Year":
            base = self.state_total(stat, year - 1, adj)
        elif change_view == "vs Selected Year" and comp_year:
            base = self.state_total(stat, comp_year, adj)
        else:
            ys = [self.state_total(stat, y, adj) for y in (year - 1, year - 2, year - 3) if y >= FIRST_HIST_YEAR]
            ys = [v for v in ys if v is not None]
            base = float(np.mean(ys)) if ys else None
        return (cur - base) / abs(base) * 100 if base else None


