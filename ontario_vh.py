"""Weekly satellite crop condition for Ontario cropland, from NOAA STAR's Vegetation Health product (no Streamlit import).

Why NOAA and not Statistics Canada: StatCan's public weekly NDVI table (32-10-0383-01, from its Crop Condition Assessment
Program) stops at 2024-05-27, and the earlier table 32-10-0453-01 is archived at 2021-10-04. NOAA STAR publishes the same kind of
series for Canadian provinces, cropland pixels only, weekly from 1982 to the latest week:
  SMN = smoothed NDVI (greenness), SMT = smoothed brightness temperature (K), VCI = vegetation condition index (0-100, this
  week's NDVI against its own historical min-max), TCI = thermal condition index, VHI = vegetation health index (0.5 VCI + 0.5 TCI).
NOAA labels the product experimental, and this is a province-wide average, not a county or field measure.

`python ontario_vh.py` refreshes the bundled ontario_vh.json used when the live service cannot be reached.
"""
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php"
SNAPSHOT = Path(__file__).with_name("ontario_vh.json")
COLS = ["SMN", "SMT", "VCI", "TCI", "VHI"]
LABEL = {"SMN": "NDVI (greenness)", "VCI": "VCI (vegetation condition)", "TCI": "TCI (thermal condition)",
         "VHI": "VHI (vegetation health)"}
FIRST_YEAR = 1982
# Windows fixed in advance on crop stage (not searched for the best fit): soybeans pod set and fill, corn silking and grain fill.
WINDOWS = {"soy": (29, 35), "corn": (27, 35)}


def parse(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{4}),\s*(\d+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)", line)
        if m and float(m.group(3)) > -0.5:                 # -1 marks weeks that have not happened yet
            rows.append([int(m.group(1)), int(m.group(2))] + [float(m.group(i)) for i in range(3, 8)])
    return pd.DataFrame(rows, columns=["year", "week"] + COLS)


def fetch(timeout: float = 12.0) -> pd.DataFrame:
    """Live series for Ontario cropland, 1982 to now. Raises on any failure so a failure is never cached."""
    params = {"provinceID": 9, "country": "CAN", "adminVHversion": "GC_current", "yearlyTag": "weekly", "type": "Mean",
              "TagCropland": "crop", "year1": FIRST_YEAR, "year2": datetime.date.today().year}
    r = requests.get(URL, params=params, timeout=timeout, headers={"User-Agent": "jsa-soy-crush-calculator"})
    r.raise_for_status()
    df = parse(r.text)
    if df.empty or "Ontario" not in r.text[:300]:
        raise RuntimeError("NOAA STAR returned no Ontario series")
    return df


def snapshot() -> pd.DataFrame:
    df = pd.DataFrame(json.loads(SNAPSHOT.read_text(encoding="utf-8"))["rows"], columns=["year", "week"] + COLS)
    return df.astype({"year": int, "week": int})


def save_snapshot(df: pd.DataFrame) -> None:
    SNAPSHOT.write_text(json.dumps({"generated": datetime.date.today().isoformat(), "source": "NOAA STAR Vegetation Health, "
                                    "province-averaged, cropland only, GC_current", "rows": df.round(3).values.tolist()},
                                   separators=(",", ":")), encoding="utf-8")


def week_start(year: int, week: int) -> datetime.date:
    """Approximate first day of NOAA's week (weeks count in sevens from Jan 1)."""
    return datetime.date(int(year), 1, 1) + datetime.timedelta(days=7 * (int(week) - 1))


def last_week(df: pd.DataFrame) -> tuple:
    y = int(df["year"].max())
    return y, int(df[df["year"] == y]["week"].max())


def matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Weeks down, years across."""
    return df.pivot(index="week", columns="year", values=col)


def window_mean(df: pd.DataFrame, year: int, window: tuple, col: str = "SMN"):
    a, b = window
    s = df[(df["year"] == year) & df["week"].between(a, b)]
    return float(s[col].mean()) if len(s) == b - a + 1 else None


def season_table(df: pd.DataFrame, crop: str, year: int) -> pd.DataFrame:
    """The crop window's mean for each index in `year` against every other year: value, average, rank (1 = best), peak NDVI."""
    win = WINDOWS[crop]
    rows = []
    for col in ("SMN", "VCI", "VHI", "TCI"):
        vals = {y: window_mean(df, y, win, col) for y in sorted(df["year"].unique())}
        vals = {y: v for y, v in vals.items() if v is not None}
        if year not in vals:
            continue
        others = [v for y, v in vals.items() if y != year]
        rows.append({"index": col, "this_year": vals[year], "average": float(np.mean(others)), "rank": int(sum(v > vals[year] for v in others)) + 1,
                     "of": len(vals), "years": f"{min(vals)}-{max(vals)}"})
    return pd.DataFrame(rows)


def peak(df: pd.DataFrame, year: int) -> tuple:
    s = df[(df["year"] == year) & df["week"].between(20, 42)]
    i = s["SMN"].idxmax()
    return float(s.loc[i, "SMN"]), int(s.loc[i, "week"])


def yield_model(df: pd.DataFrame, crop: str, yields: dict, year: int, col: str = "SMN") -> dict:
    """yield = a + b x (year - 2000) + c x (crop-window mean of `col`), fitted on the years before `year`.
    Leave-one-out error is reported against the same model without the satellite term, because that comparison is the
    honest measure of what the index adds."""
    win = WINDOWS[crop]
    yrs = [y for y in sorted(yields) if y < year and y >= FIRST_YEAR and window_mean(df, y, win, col) is not None]
    x = np.array([window_mean(df, y, win, col) for y in yrs])
    y_ = np.array([yields[y] for y in yrs])
    t = np.array([y - 2000 for y in yrs], dtype=float)
    n = len(yrs)

    def fit(mask, with_x):
        a = np.c_[np.ones(mask.sum()), t[mask]] if not with_x else np.c_[np.ones(mask.sum()), t[mask], x[mask]]
        return np.linalg.lstsq(a, y_[mask], rcond=None)[0]

    e1, e0 = [], []
    for i in range(n):
        m = np.arange(n) != i
        b1, b0 = fit(m, True), fit(m, False)
        e1.append(y_[i] - (b1[0] + b1[1] * t[i] + b1[2] * x[i]))
        e0.append(y_[i] - (b0[0] + b0[1] * t[i]))
    full, base = fit(np.ones(n, bool), True), fit(np.ones(n, bool), False)
    fitted = full[0] + full[1] * t + full[2] * x
    x_now = window_mean(df, year, win, col)
    out = {"n": n, "years": yrs, "x": x, "yield": y_, "slope": float(full[2]), "rmse_loo": float(np.sqrt(np.mean(np.square(e1)))),
           "rmse_trend_loo": float(np.sqrt(np.mean(np.square(e0)))), "r2": float(1 - ((y_ - fitted) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()),
           "trend_year": float(base[0] + base[1] * (year - 2000)), "x_now": x_now, "window": win, "col": col,
           "anomaly": y_ - (base[0] + base[1] * t)}
    out["trend_slope"] = float(np.polyfit(x, out["anomaly"], 1)[0])
    out["predicted"] = None if x_now is None else float(full[0] + full[1] * (year - 2000) + full[2] * x_now)
    out["skill"] = 1 - out["rmse_loo"] / out["rmse_trend_loo"]
    return out


if __name__ == "__main__":
    frame = fetch()
    save_snapshot(frame)
    ly, lw = last_week(frame)
    print(f"wrote {SNAPSHOT.name}: {len(frame)} weekly rows, {frame['year'].min()}-{ly}, last week {lw} (about {week_start(ly, lw):%b %d})")
    sys.exit(0)
