"""Crush history data loading and seasonal alignment.

Pulls ZS + ZM + ZL settlement histories from Massive and computes a daily
gross processing margin series for each contract-year combination.

Contract month pairings (beans → meal / oil):
  Mar, May, Jul, Aug, Sep  → same month for all three products
  Nov beans (ZSX)          → Dec meal (ZMZ) + Dec oil (ZLZ)
    (ZM and ZL have no November contract; Dec is the nearest available after Nov)

Massive settlement history starts ~2021-09-02, giving roughly 4 usable
prior contract years.  The DTE-based alignment converts each year's series
onto a common "days-to-expiration" axis so seasonal overlays line up
regardless of weekends and holidays.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

from massive_api import MassiveApiError, get_settlement_history

# ── Crush month definitions ───────────────────────────────────────────────────

MONTH_LETTER = {
    "F": "Jan", "H": "Mar", "K": "May", "N": "Jul",
    "Q": "Aug", "U": "Sep", "X": "Nov",
}

# Crush months the user can choose: bean month → (zs_letter, zm_letter, zl_letter)
# Nov beans pair with Dec meal/oil because ZM/ZL have no November contract.
CRUSH_MONTHS: dict[str, tuple[str, str, str]] = {
    "Mar": ("H", "H", "H"),
    "May": ("K", "K", "K"),
    "Jul": ("N", "N", "N"),
    "Aug": ("Q", "Q", "Q"),
    "Sep": ("U", "U", "U"),
    "Nov": ("X", "Z", "Z"),
}

YEAR_COLORS = ["#0693e3", "#e8833a", "#5aa469", "#b05fb0", "#9aa5b1", "#c0392b"]
AVG_COLOR   = "#111111"


def _ticker(product: str, month_letter: str, year_digit: int) -> str:
    return f"{product}{month_letter}{year_digit % 10}"


def tickers_for_month(crush_month: str, base_year: int, n_years: int = 5) -> list[dict]:
    """Return ticker sets for each year going back n_years from base_year.

    Each entry: {year, zs, zm, zl, label}
    """
    zs_m, zm_m, zl_m = CRUSH_MONTHS[crush_month]
    rows = []
    for back in range(n_years - 1, -1, -1):  # oldest first
        yr = base_year - back
        rows.append({
            "year":  yr,
            "zs":    _ticker("ZS", zs_m, yr),
            "zm":    _ticker("ZM", zm_m, yr),
            "zl":    _ticker("ZL", zl_m, yr),
            "label": f"{crush_month} '{str(yr)[2:]}",
        })
    return rows


def _fetch_one(ticker: str, api_key: str) -> pd.Series:
    try:
        return get_settlement_history(ticker, api_key)
    except (MassiveApiError, Exception):
        return pd.Series(dtype=float)


def fetch_all_histories(
    ticker_sets: list[dict],
    api_key: str,
    max_workers: int = 8,
) -> dict[str, pd.Series]:
    """Fetch settlement history for every unique ticker in parallel."""
    unique = {t for row in ticker_sets for t in (row["zs"], row["zm"], row["zl"])}
    out: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique))) as pool:
        futures = {pool.submit(_fetch_one, t, api_key): t for t in unique}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def daily_crush(
    zs_hist: pd.Series,
    zm_hist: pd.Series,
    zl_hist: pd.Series,
    meal_tons: float,
    oil_lbs: float,
) -> pd.Series:
    """Compute daily GPM series ($/bu) from three settlement series.

    Massive prices: ZS ¢/bu, ZM $/ton, ZL ¢/lb.
    GPM = ZM * meal_tons + (ZL/100) * oil_lbs - ZS/100
    """
    df = pd.DataFrame({"ZS": zs_hist, "ZM": zm_hist, "ZL": zl_hist}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    crush = df["ZM"] * meal_tons + (df["ZL"] / 100.0) * oil_lbs - (df["ZS"] / 100.0)
    return crush.sort_index()


def to_dte(series: pd.Series) -> pd.Series:
    """Convert a date-indexed price series to DTE-indexed (days to expiration).

    Uses the last date in the series as a proxy for contract expiration.
    Returned index: negative integers (e.g. -365 = 365 days before expiry, 0 = expiry day).
    """
    if series.empty:
        return series
    expiry = series.index.max()
    dte_index = pd.Index([(d - expiry).days for d in series.index])
    return pd.Series(series.values, index=dte_index).sort_index()


def build_seasonal_data(
    crush_month: str,
    api_key: str,
    base_year: int,
    n_years: int,
    cme_meal_tons: float,
    cme_oil_lbs: float,
    ext_meal_tons: float,
    ext_oil_lbs: float,
) -> dict[str, dict]:
    """Load and process all data for the seasonal chart.

    Returns a dict keyed by year-label, each value:
        {
          "cme_dte":  pd.Series (DTE → GPM $/bu),
          "ext_dte":  pd.Series (DTE → GPM $/bu),
          "expiry":   date (last date in ZS series),
          "year":     int,
        }
    """
    ticker_sets = tickers_for_month(crush_month, base_year, n_years)
    hist = fetch_all_histories(ticker_sets, api_key)

    result: dict[str, dict] = {}
    for row in ticker_sets:
        zs = hist.get(row["zs"], pd.Series(dtype=float))
        zm = hist.get(row["zm"], pd.Series(dtype=float))
        zl = hist.get(row["zl"], pd.Series(dtype=float))

        cme_series = daily_crush(zs, zm, zl, cme_meal_tons, cme_oil_lbs)
        ext_series = daily_crush(zs, zm, zl, ext_meal_tons, ext_oil_lbs)

        if cme_series.empty:
            continue

        expiry = cme_series.index.max()
        result[row["label"]] = {
            "cme_dte": to_dte(cme_series),
            "ext_dte": to_dte(ext_series),
            "expiry":  expiry,
            "year":    row["year"],
        }
    return result


def dte_to_calendar(dte_index, anchor_expiry: date) -> list[date]:
    """Convert DTE integers back to calendar dates using a fixed anchor expiry."""
    from datetime import timedelta
    return [anchor_expiry + timedelta(days=int(d)) for d in dte_index]


def year_average(
    series_list: list[pd.Series],
    window_days: int = 500,
) -> pd.Series:
    """Mean across years at each DTE point (points where at least half the years are present)."""
    if not series_list:
        return pd.Series(dtype=float)
    grid = pd.RangeIndex(-window_days, 1)
    aligned = {}
    for i, s in enumerate(series_list):
        clean = s[~s.index.duplicated(keep="last")].sort_index()
        aligned[i] = clean.reindex(grid).interpolate(limit_area="inside")
    frame = pd.DataFrame(aligned)
    required = max(2, (len(series_list) + 1) // 2)
    return frame.mean(axis=1, skipna=True)[frame.count(axis=1) >= required]
