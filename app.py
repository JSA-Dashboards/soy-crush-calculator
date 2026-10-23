"""Soybean Crush Calculator — CME Board Crush vs. Extruder Plant GPM.

Standard CME / hexane solvent extraction yields per bushel:
  Meal:  44 lbs → 0.022 short tons
  Oil:   11 lbs
  GPM  = ZM($/ton) × 0.022 + ZL($/lb) × 11 - ZS($/bu)

Extruder (client default, editable in sidebar):
  365.5 lbs oil / ton meal, 44.38 bu / ton meal  →  8.235 lbs oil/bu, 0.022532 t meal/bu
  GPM  = ZM × 0.022532 + ZL × 8.235 - ZS

Massive prices:  ZS → ¢/bu   ZM → $/ton   ZL → ¢/lb
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from massive_api import MassiveApiError, get_futures_curve

from crush_history import (  # noqa: E402
    CRUSH_MONTHS, YEAR_COLORS, AVG_COLOR,
    build_seasonal_data, dte_to_calendar, year_average,
)

# ── Brand constants ───────────────────────────────────────────────────────────

JPSI_DARK  = "#32373c"
JPSI_BLUE  = "#0693e3"
NEG_RED    = "#d64545"
POS_GREEN  = "#2e7d32"
LOGO_URL   = "https://www.jpsi.com/wp-content/themes/gate39media/img/logo-full.png"
FAVICON    = "https://www.jpsi.com/wp-content/uploads/2019/04/cropped-Favicon-1-192x192.png"

# ── Crush yield constants ─────────────────────────────────────────────────────

CME_MEAL_TONS = 44.0 / 2000.0   # 0.022 t/bu
CME_OIL_LBS   = 11.0            # lbs/bu

DEFAULT_OIL_LBS_PER_TON = 365.5
DEFAULT_BU_PER_TON       = 44.38

# ── CME contract sizes ────────────────────────────────────────────────────────

ZS_BU    = 5_000     # bushels per ZS contract
ZM_TONS  = 100       # short tons per ZM contract
ZL_LBS   = 60_000    # lbs per ZL contract

# ── Position persistence ──────────────────────────────────────────────────────

_POSITIONS_FILE = Path(__file__).parent / "hedge_positions.json"

_POS_DEFAULTS = [
    {"Leg": "ZS", "Month": "", "Qty": -10, "Entry": 1350.00,
     "Note": "Short beans (buy crush)"},
    {"Leg": "ZM", "Month": "", "Qty": 11,  "Entry": 370.00,
     "Note": "Long meal"},
    {"Leg": "ZL", "Month": "", "Qty": 9,   "Entry": 70.00,
     "Note": "Long oil (CME ratio)"},
]


def _load_positions() -> list[dict]:
    try:
        return json.loads(_POSITIONS_FILE.read_text())
    except Exception:
        return [row.copy() for row in _POS_DEFAULTS]


def _save_positions(rows: list[dict]) -> None:
    try:
        _POSITIONS_FILE.write_text(json.dumps(rows, indent=2))
    except Exception:
        pass


# ── Crush position persistence ────────────────────────────────────────────────

_CRUSH_POS_FILE = Path(__file__).parent / "crush_position.json"
_POS_N = 11  # months stored (Oct → Aug); display sliced to 6 or 11 via toggle


def _pos_month_labels(n: int = _POS_N) -> list[str]:
    d = date.today().replace(day=1)
    labels = []
    for _ in range(n):
        labels.append(d.strftime("%b '%y"))
        # Advance one month safely
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return labels


_CRUSH_POS_DEFAULTS: dict = {
    # Location: Union Springs
    "us_bean_opening":  0.0,
    "us_capacity":      1_000_000.0,
    "us_bean_arrive":   [0.0] * _POS_N,
    # Location: York
    "york_bean_opening": 0.0,
    "york_capacity":     350_000.0,
    "york_bean_arrive":  [0.0] * _POS_N,
    # Crush schedule per plant
    "us_crush_bu":   [0.0] * _POS_N,
    "york_crush_bu": [0.0] * _POS_N,
    # Product balances
    "meal_opening_tons": 0.0,
    "meal_sales_tons":   [0.0] * _POS_N,
    "oil_opening_lbs":   0.0,
    "oil_sales_lbs":     [0.0] * _POS_N,
}


def _load_crush_pos() -> dict:
    try:
        sf = _sf_load_crush_pos()
        data = sf if sf is not None else json.loads(_CRUSH_POS_FILE.read_text())
        # Migrate old single-location format → two-location format
        if "bean_opening" in data and "us_bean_opening" not in data:
            data["us_bean_opening"] = data.pop("bean_opening", 0.0)
            old_p = data.pop("bean_priced",   [0.0] * _POS_N)
            old_u = data.pop("bean_unpriced", [0.0] * _POS_N)
            data["us_bean_arrive"] = [p + u for p, u in zip(old_p, old_u)]
        # Migrate priced+unpriced → combined arrive if old two-column format
        if "us_bean_priced" in data and "us_bean_arrive" not in data:
            data["us_bean_arrive"] = [
                p + u for p, u in zip(data.pop("us_bean_priced", [0.0] * _POS_N),
                                       data.pop("us_bean_unpriced", [0.0] * _POS_N))
            ]
        if "york_bean_priced" in data and "york_bean_arrive" not in data:
            data["york_bean_arrive"] = [
                p + u for p, u in zip(data.pop("york_bean_priced", [0.0] * _POS_N),
                                       data.pop("york_bean_unpriced", [0.0] * _POS_N))
            ]
        # Migrate combined crush_bu → per-plant keys
        if "crush_bu" in data and "us_crush_bu" not in data:
            data["us_crush_bu"]   = data.pop("crush_bu")
            data["york_crush_bu"] = [0.0] * _POS_N
        # Drop the old combined key if it somehow still exists
        data.pop("crush_bu", None)
        # Fill any missing keys from defaults
        for k, v in _CRUSH_POS_DEFAULTS.items():
            if k not in data:
                data[k] = list(v) if isinstance(v, list) else v
            elif isinstance(v, list) and len(data[k]) != _POS_N:
                data[k] = (data[k] + [0.0] * _POS_N)[:_POS_N]
        return data
    except Exception:
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in _CRUSH_POS_DEFAULTS.items()}


def _save_crush_pos(data: dict) -> None:
    try:
        _CRUSH_POS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass
    _sf_save_crush_pos(data)


# ── Snowflake persistence ─────────────────────────────────────────────────────

def _sf_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or ""
    except Exception:
        return ""


def _sf_private_key():
    pem  = _sf_secret("SNOWFLAKE_PRIVATE_KEY")
    path = _sf_secret("SNOWFLAKE_PRIVATE_KEY_PATH")
    if not pem and not path:
        return None
    from cryptography.hazmat.primitives import serialization
    raw = open(path, "rb").read() if path else pem.replace("\\n", "\n").encode()
    pwd = _sf_secret("SNOWFLAKE_PRIVATE_KEY_PWD") or None
    key = serialization.load_pem_private_key(raw, password=pwd.encode() if pwd else None)
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _sf_connect():
    import snowflake.connector
    kw = dict(
        account=_sf_secret("SNOWFLAKE_ACCOUNT"),
        user=_sf_secret("SNOWFLAKE_USER"),
        role=_sf_secret("SNOWFLAKE_ROLE") or None,
        warehouse=_sf_secret("SNOWFLAKE_WAREHOUSE") or None,
        database=_sf_secret("SNOWFLAKE_DATABASE") or "JSA",
        schema="PUBLIC",
        login_timeout=30,
    )
    pkey = _sf_private_key()
    if pkey is not None:
        kw["private_key"] = pkey
    else:
        kw["password"] = _sf_secret("SNOWFLAKE_PASSWORD")
    return snowflake.connector.connect(**kw)


_SF_DDL = (
    "CREATE TABLE IF NOT EXISTS JSA.PUBLIC.CRUSH_POSITION "
    "(key VARCHAR(50) NOT NULL, data VARCHAR(16777216), "
    "updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())"
)


def _sf_load_crush_pos() -> dict | None:
    if not _sf_secret("SNOWFLAKE_ACCOUNT"):
        return None
    try:
        conn = _sf_connect()
        cur  = conn.cursor()
        cur.execute(_SF_DDL)
        cur.execute(
            "SELECT data FROM JSA.PUBLIC.CRUSH_POSITION WHERE key = 'default' LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _sf_save_crush_pos(data: dict) -> None:
    if not _sf_secret("SNOWFLAKE_ACCOUNT"):
        return
    try:
        conn = _sf_connect()
        cur  = conn.cursor()
        cur.execute(_SF_DDL)
        cur.execute(
            "MERGE INTO JSA.PUBLIC.CRUSH_POSITION AS tgt "
            "USING (SELECT 'default' AS key, %s AS data) AS src ON tgt.key = src.key "
            "WHEN MATCHED THEN UPDATE SET data = src.data, updated_at = CURRENT_TIMESTAMP() "
            "WHEN NOT MATCHED THEN INSERT (key, data, updated_at) "
            "VALUES (src.key, src.data, CURRENT_TIMESTAMP())",
            (json.dumps(data),),
        )
        conn.close()
    except Exception:
        pass


# ── NASS cache helpers (Draw Territory) ──────────────────────────────────────

def _nass_cache_key(params: dict) -> str:
    ignored = {"key", "format"}
    clean = {k: str(v) for k, v in params.items() if k not in ignored}
    canon = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"api_GET|{canon}".encode()).hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)
def _dt_fetch_ny(
    commodity_desc: str,
    statisticcat_desc: str,
    unit_desc: str,
    reference_period_desc: str = "YEAR",
    year__GE: str = "1980",
    util_practice_desc: str = "",
    class_desc: str = "",
) -> pd.DataFrame:
    """Fetch NY state data from JSA.NASS_CACHE. Returns year/value DataFrame."""
    params: dict = {
        "commodity_desc": commodity_desc,
        "statisticcat_desc": statisticcat_desc,
        "unit_desc": unit_desc,
        "reference_period_desc": reference_period_desc,
        "source_desc": "SURVEY",
        "agg_level_desc": "STATE",
        "year__GE": year__GE,
    }
    if util_practice_desc:
        params["util_practice_desc"] = util_practice_desc
    if class_desc:
        params["class_desc"] = class_desc
    ck = _nass_cache_key(params)
    try:
        conn = _sf_connect()
        cur  = conn.cursor()
        cur.execute(
            "SELECT data FROM JSA.NASS_CACHE.nass_cache WHERE cache_key = %s LIMIT 1",
            (ck,),
        )
        row = cur.fetchone()
        conn.close()
    except Exception:
        return pd.DataFrame()
    if not row:
        return pd.DataFrame()
    try:
        raw = row[0] if isinstance(row[0], (dict, list)) else json.loads(row[0])
        records = raw.get("data", []) if isinstance(raw, dict) else raw
        df = pd.DataFrame(records)
        if "state_alpha" in df.columns:
            df = df[df["state_alpha"] == "NY"]
        df["year"] = pd.to_numeric(df.get("year", pd.Series(dtype=float)), errors="coerce")
        df["value"] = pd.to_numeric(
            df.get("Value", pd.Series(dtype=str)).astype(str).str.replace(",", ""),
            errors="coerce",
        )
        return (
            df.dropna(subset=["year", "value"])[["year", "value"]]
            .sort_values("year").reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


# ── Draw Territory commodity/metric config ────────────────────────────────────

_DT_CROPS: dict = {
    "Corn": {
        "Planted Acres":   {"commodity_desc": "CORN", "statisticcat_desc": "AREA PLANTED",  "unit_desc": "ACRES",     "source_desc": "SURVEY"},
        "Harvested Acres": {"commodity_desc": "CORN", "statisticcat_desc": "AREA HARVESTED","unit_desc": "ACRES",     "source_desc": "SURVEY", "util_practice_desc": "GRAIN"},
        "Yield (Bu/Ac)":   {"commodity_desc": "CORN", "statisticcat_desc": "YIELD",         "unit_desc": "BU / ACRE", "source_desc": "SURVEY", "util_practice_desc": "GRAIN"},
        "Production (Bu)": {"commodity_desc": "CORN", "statisticcat_desc": "PRODUCTION",    "unit_desc": "BU",        "source_desc": "SURVEY", "util_practice_desc": "GRAIN"},
    },
    "Soybeans": {
        "Planted Acres":   {"commodity_desc": "SOYBEANS", "statisticcat_desc": "AREA PLANTED",  "unit_desc": "ACRES",     "source_desc": "SURVEY"},
        "Harvested Acres": {"commodity_desc": "SOYBEANS", "statisticcat_desc": "AREA HARVESTED","unit_desc": "ACRES",     "source_desc": "SURVEY"},
        "Yield (Bu/Ac)":   {"commodity_desc": "SOYBEANS", "statisticcat_desc": "YIELD",         "unit_desc": "BU / ACRE", "source_desc": "SURVEY"},
        "Production (Bu)": {"commodity_desc": "SOYBEANS", "statisticcat_desc": "PRODUCTION",    "unit_desc": "BU",        "source_desc": "SURVEY"},
    },
}

_DT_LIVESTOCK: dict = {
    "Dairy Cows": {
        "Inventory (Head)": {"commodity_desc": "CATTLE", "class_desc": "COWS",        "util_practice_desc": "MILK", "statisticcat_desc": "INVENTORY", "unit_desc": "HEAD", "source_desc": "SURVEY", "reference_period_desc": "FIRST OF JAN"},
    },
    "All Cattle & Calves": {
        "Inventory (Head)": {"commodity_desc": "CATTLE", "class_desc": "ALL CLASSES", "statisticcat_desc": "INVENTORY", "unit_desc": "HEAD", "source_desc": "SURVEY", "reference_period_desc": "FIRST OF JAN"},
    },
    "Hogs & Pigs": {
        "Inventory (Head)": {"commodity_desc": "HOGS", "statisticcat_desc": "INVENTORY","unit_desc": "HEAD", "source_desc": "SURVEY", "reference_period_desc": "FIRST OF DEC"},
    },
}

_DT_ALL_COMMS: dict = {**_DT_CROPS, **_DT_LIVESTOCK}

# Light-card palette for map / table (matches JPSI light theme)
_DT_CARD = "#ffffff"
_DT_BORD = "#e5e7eb"
_DT_TXT  = "#111827"
_DT_SEC  = "#6b7280"


@st.cache_data(ttl=1800, show_spinner=False)
def _dt_load_county_ny(commodity: str, metric_label: str, year: int) -> pd.DataFrame:
    """Load NY county NASS data from JSA.NASS_CACHE using the same key as the domestic-production-dashboard ETL."""
    mp = _DT_ALL_COMMS.get(commodity, {}).get(metric_label)
    if mp is None:
        return pd.DataFrame()
    # County params: strip reference_period_desc, add county-specific fields
    params = {k: v for k, v in mp.items() if k != "reference_period_desc"}
    params.update({
        "domain_desc": "TOTAL",
        "freq_desc": "ANNUAL",
        "agg_level_desc": "COUNTY",
        "year": str(year),
    })
    ck = _nass_cache_key(params)
    try:
        conn = _sf_connect()
        cur  = conn.cursor()
        cur.execute("SELECT data FROM JSA.NASS_CACHE.nass_cache WHERE cache_key = %s LIMIT 1", (ck,))
        row = cur.fetchone()
        conn.close()
    except Exception:
        return pd.DataFrame()
    if not row:
        return pd.DataFrame()
    try:
        raw = row[0] if isinstance(row[0], (dict, list)) else json.loads(row[0])
        records = raw.get("data", []) if isinstance(raw, dict) else raw
        df = pd.DataFrame(records)
        # Filter to NY
        if "state_alpha" in df.columns:
            df = df[df["state_alpha"] == "NY"]
        # Drop aggregate/other rows
        if "county_ansi" in df.columns:
            df = df[~df["county_ansi"].isin({"998", "000", "999"})]
        if "county_name" in df.columns:
            df = df[~df["county_name"].str.strip().str.lower().str.startswith("other")]
        if df.empty:
            return pd.DataFrame()
        # Numeric value
        df["Value"] = pd.to_numeric(
            df["Value"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=["Value"])
        # FIPS
        if "state_fips_code" in df.columns and "county_ansi" in df.columns:
            df["fips"] = df["state_fips_code"].str.zfill(2) + df["county_ansi"].str.zfill(3)
        else:
            df["fips"] = ""
        df["State"]  = df["state_alpha"].str.strip() if "state_alpha" in df.columns else "NY"
        df["County"] = df["county_name"].str.strip().str.title() if "county_name" in df.columns else ""
        # Prefer ALL PRODUCTION PRACTICES
        key_cols = ["State", "County", "fips"]
        if "prodn_practice_desc" in df.columns:
            all_p = df[df["prodn_practice_desc"] == "ALL PRODUCTION PRACTICES"].copy()
            rest  = df[~df["fips"].isin(all_p["fips"].unique())].copy()
            if not rest.empty:
                rest = rest.loc[rest.groupby(key_cols, sort=False)["Value"].idxmax()]
            df = pd.concat([all_p, rest], ignore_index=True)
        extra = [c for c in ["asd_desc", "asd_code"] if c in df.columns]
        return df[key_cols + ["Value"] + extra].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def _dt_load_geojson() -> dict:
    import urllib.request as _req
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with _req.urlopen(url, timeout=30) as r:
        return json.load(r)


def _dt_centroid(feat: dict):
    geom = feat["geometry"]
    rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    best  = max(rings, key=lambda r: (
        (max(p[0] for p in r) - min(p[0] for p in r)) *
        (max(p[1] for p in r) - min(p[1] for p in r))
    ))
    return (sum(p[0] for p in best) / len(best), sum(p[1] for p in best) / len(best))


def _dt_county_fig(df: pd.DataFrame, geo: dict, metric_label: str) -> go.Figure:
    is_yield = "Yield" in metric_label
    unit     = " Bu/Ac" if is_yield else (" Ac" if "Acres" in metric_label else (" Head" if "Head" in metric_label else " Bu"))
    fmt      = ".1f" if is_yield else ",.0f"
    NY_FIPS  = "36"
    geo_sub  = {"type": "FeatureCollection", "features": [f for f in geo["features"] if f["properties"]["STATE"] == NY_FIPS]}
    fig = go.Figure(go.Choropleth(
        geojson=geo_sub, locations=df["fips"], z=df["Value"],
        featureidkey="id",
        colorscale="RdYlGn" if is_yield else "YlOrRd",
        marker_line_width=0.5, marker_line_color=_DT_BORD,
        colorbar=dict(title=dict(text=unit.strip(), font=dict(size=11, color=_DT_SEC)),
                      tickfont=dict(size=10, color=_DT_SEC), thickness=12, len=0.65, x=1.0),
        hovertemplate=f"<b>%{{text}}</b><br>{metric_label}: %{{z:{fmt}}}{unit}<extra></extra>",
        text=df["County"] + ", NY",
    ))
    fig.update_geos(scope="usa", projection_type="albers usa", fitbounds="geojson",
                    showlakes=False, showland=True, landcolor="#f8f9fa",
                    showcoastlines=False, showframe=False)
    fig.update_layout(paper_bgcolor=_DT_CARD, plot_bgcolor=_DT_CARD,
                      margin=dict(l=0, r=0, t=0, b=0), height=480, font=dict(color=_DT_TXT))
    return fig


def _dt_asd_fig(df: pd.DataFrame, geo: dict, metric_label: str) -> go.Figure:
    if "asd_desc" not in df.columns or df["asd_desc"].isna().all():
        return _dt_county_fig(df, geo, metric_label)
    is_yield = "Yield" in metric_label
    unit     = " Bu/Ac" if is_yield else (" Ac" if "Acres" in metric_label else (" Head" if "Head" in metric_label else " Bu"))
    fmt      = ".1f" if is_yield else ",.0f"
    agg_fn   = "mean" if is_yield else "sum"
    dist_agg = (df.groupby(["asd_desc", "asd_code"], as_index=False)["Value"]
                  .agg(agg_fn).rename(columns={"Value": "DistValue"}))
    merged   = df.merge(dist_agg, on=["asd_desc", "asd_code"], how="left")
    NY_FIPS  = "36"
    geo_sub  = {"type": "FeatureCollection", "features": [f for f in geo["features"] if f["properties"]["STATE"] == NY_FIPS]}
    fig = go.Figure(go.Choropleth(
        geojson=geo_sub, locations=merged["fips"], z=merged["DistValue"],
        featureidkey="id",
        colorscale="RdYlGn" if is_yield else "YlOrRd",
        marker_line_width=0.4, marker_line_color=_DT_BORD,
        colorbar=dict(title=dict(text=unit.strip(), font=dict(size=11, color=_DT_SEC)),
                      tickfont=dict(size=10, color=_DT_SEC), thickness=12, len=0.65, x=1.0),
        hovertemplate=f"<b>%{{text}}</b><br>District {metric_label}: %{{z:{fmt}}}{unit}<extra></extra>",
        text=merged["asd_desc"],
    ))
    # District labels from county centroids
    cent_map: dict = {}
    for feat in geo_sub["features"]:
        fips5 = feat["properties"]["STATE"] + feat["properties"]["COUNTY"]
        cx, cy = _dt_centroid(feat)
        cent_map[fips5] = (cx, cy)
    merged["_cx"] = merged["fips"].map(lambda f: cent_map.get(f, (None, None))[0])
    merged["_cy"] = merged["fips"].map(lambda f: cent_map.get(f, (None, None))[1])
    dc = merged.dropna(subset=["_cx"]).groupby("asd_desc", as_index=False).agg(
        lon=("_cx","mean"), lat=("_cy","mean"), val=("DistValue","first"))
    if not dc.empty:
        lbl_fn = (lambda v: f"{v:.1f}") if is_yield else (lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v:,.0f}")
        fig.add_trace(go.Scattergeo(
            lon=dc["lon"], lat=dc["lat"],
            text=dc["asd_desc"].str.replace(" District","",case=False,regex=False) + "<br>" + dc["val"].apply(lbl_fn),
            textfont=dict(size=8, color=_DT_TXT), mode="text",
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_geos(scope="usa", projection_type="albers usa", fitbounds="geojson",
                    showlakes=False, showland=True, landcolor="#f8f9fa",
                    showcoastlines=False, showframe=False)
    fig.update_layout(paper_bgcolor=_DT_CARD, plot_bgcolor=_DT_CARD,
                      margin=dict(l=0, r=0, t=0, b=0), height=480, font=dict(color=_DT_TXT))
    return fig


def _extract_ch_report_image(img_bytes: bytes, mime: str) -> dict:
    """Send a CH Basis report image/screenshot to Claude Vision and extract
    per-location, per-month to-arrive figures (BASIS POSITION + UNPRICED PURCHASES).
    Returns a dict compatible with _parse_ch_report output, plus 'us_by_month'
    and 'york_by_month' dicts keyed by lowercase month labels like 'nov_26'."""
    try:
        import base64
        import anthropic  # installed via requirements.txt
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not found in Streamlit secrets."}
        client = anthropic.Anthropic(api_key=api_key)
        img_b64 = base64.standard_b64encode(img_bytes).decode()
        prompt = (
            "This is a CH Basis report from Du Mond Grain. "
            "Extract the following and return ONLY a JSON object with no explanation:\n\n"
            "{\n"
            '  "company_owned_inventory": <integer from COMPANY OWNED INVENTORY row>,\n'
            '  "total_stock": <integer from TOTAL STOCK row>,\n'
            '  "union_springs_total": <integer total from Union Springs BASIS POSITION row>,\n'
            '  "york_total": <integer total from York Elevator BASIS POSITION row>,\n'
            '  "us_by_month": { "nearby": N, "nov_26": N, "dec_26": N, "jan_27": N, '
            '"feb_27": N, "mar_27": N, "apr_27": N, "later": N },\n'
            '  "york_by_month": { "nearby": N, "nov_26": N, "dec_26": N, "jan_27": N, '
            '"feb_27": N, "mar_27": N, "apr_27": N, "later": N }\n'
            "}\n\n"
            "For us_by_month and york_by_month: add BASIS POSITION + UNPRICED PURCHASES "
            "for that location and month column. Use 0 for columns that are blank."
        )
        resp = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as _exc:
        return {"error": str(_exc)}


# Month-label → position-tab month index  (0 = current month = Oct '26)
_REPORT_MONTH_IDX: dict[str, int] = {
    "nearby": 0,
    "nov_26": 1, "nov26": 1,
    "dec_26": 2, "dec26": 2,
    "jan_27": 3, "jan27": 3,
    "feb_27": 4, "feb27": 4,
    "mar_27": 5, "mar27": 5,
    "apr_27": 6, "apr27": 6,
    "may_27": 7, "may27": 7,
    "jun_27": 8, "jun27": 8,
    "jul_27": 9, "jul27": 9,
    "aug_27": 10, "aug27": 10,
    "later": 10,   # lump "later" into last slot
}


def _parse_ch_report(text: str) -> dict:
    """Extract key inventory totals from a Du Mond-style CH Basis report."""
    result: dict = {}
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if re.match(r"TOTAL\s+STOCK", s):
            nums = re.findall(r"[\d,]+", s)
            if nums:
                result["total_stock"] = int(nums[0].replace(",", ""))
        if "COMPANY OWNED INVENTORY" in s:
            nums = re.findall(r"[\d,]+", s)
            if nums:
                result["company_owned_inventory"] = int(nums[-1].replace(",", ""))
        if re.match(r"UNPRICED PURCHASES", s) and "HTA" not in s:
            nums = re.findall(r"[\d,]+", s)
            if nums:
                result["unpriced_purchases"] = int(nums[0].replace(",", ""))
        if re.match(r"BASIS POSITION\s+[\d,]", s) and "HEDGE" not in s and " BY " not in s:
            nums = re.findall(r"[\d,]+", s)
            if nums:
                result["basis_position"] = int(nums[0].replace(",", ""))
        # Location-specific rows (basis position or unpriced by location)
        if re.search(r"UNION\s+SPRINGS", s, re.IGNORECASE):
            nums = re.findall(r"[\d,]+", s)
            if nums:
                result["union_springs_total"] = int(nums[0].replace(",", ""))
        if re.search(r"YORK", s, re.IGNORECASE) and "NEW YORK" not in s.upper():
            nums = re.findall(r"[\d,]+", s)
            if nums:
                result["york_total"] = int(nums[0].replace(",", ""))
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def gpm(zm: float, zl_cents: float, zs_cents: float, meal_tons: float, oil_lbs: float) -> float:
    return zm * meal_tons + (zl_cents / 100.0) * oil_lbs - (zs_cents / 100.0)


def ext_yields(oil_per_ton: float, bu_per_ton: float) -> tuple[float, float]:
    return 1.0 / bu_per_ton, oil_per_ton / bu_per_ton


def align_curves(zs: pd.DataFrame, zm: pd.DataFrame, zl: pd.DataFrame) -> pd.DataFrame:
    def tag(df, col):
        d = df.copy()
        d["month"] = pd.to_datetime(d["expiration"]).dt.to_period("M").astype(str)
        return d.rename(columns={"price": col, "ticker": f"tkr_{col}"})[
            ["month", f"tkr_{col}", col]
        ]
    m = tag(zs, "ZS").merge(tag(zm, "ZM"), on="month").merge(tag(zl, "ZL"), on="month")
    return m.dropna(subset=["ZS", "ZM", "ZL"]).sort_values("month").reset_index(drop=True)


def plotly_defaults(fig, height: int = 420):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Source Sans Pro", color=JPSI_DARK),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    )
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#e5e7eb")
    fig.update_xaxes(gridcolor="#f0f0f0")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Soy Crush Calculator · JPSI",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide the Streamlit Community Cloud viewer badge (the profile avatar that links
# to the creator's other apps) for a clean, client-facing footer.
# The Streamlit Cloud badge (creator avatar + logo) is drawn by Cloud's outer page,
# outside this iframe, so CSS here can't reach it; add the rule to the parent
# document instead (same-origin). No-op when run locally.
_BADGE_JS = """<script>(function(){try{var w=window;while(w.parent&&w.parent!==w){try{void w.parent.document;w=w.parent;}catch(e){break;}}var d=w.document;if(d.getElementById('jsa-hide-cloud-badge'))return;var s=d.createElement('style');s.id='jsa-hide-cloud-badge';s.textContent="[class*='_profileContainer_'],[class*='_viewerBadge_']{display:none !important;}";d.head.appendChild(s);}catch(e){}})();</script>"""
try:
    st.html(_BADGE_JS, unsafe_allow_javascript=True)
except TypeError:  # older Streamlit without st.html JS support
    import streamlit.components.v1 as _stc
    _stc.html(_BADGE_JS, height=0)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=EB+Garamond:wght@400;500;600&display=swap');
  html, body, [class*="css"], .stApp, button, input, select, textarea,
  table, td, th, .stMarkdown, .stMetricLabel, .stMetricValue {{
    font-family: 'Source Sans Pro', system-ui, -apple-system, sans-serif !important;
  }}
  table td, table th {{ font-variant-numeric: tabular-nums; }}
  header[data-testid="stHeader"] {{ display: none !important; }}
  #MainMenu {{ visibility: hidden !important; }}
  footer {{ visibility: hidden !important; }}
  .block-container {{
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1280px;
  }}
  .stApp {{ background-color: #ffffff; }}

  .dash-header {{
    background: #ffffff;
    border-bottom: 3px solid {JPSI_BLUE};
    padding: 18px 8px 14px 8px;
    margin: -0.75rem 0 16px 0;
    display: flex; align-items: center; gap: 20px;
  }}
  .dash-header-logo img {{ height: 52px; display: block; }}
  .dash-header-text {{ flex: 1; text-align: center; }}
  .dash-header-text h1 {{
    margin: 0; color: {JPSI_DARK} !important;
    font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em;
  }}
  .dash-header-text .subtitle {{ color: #6b7280; font-size: 0.85rem; margin: 3px 0 0 0; }}

  .sh {{ font-size: 1.0rem; font-weight: 700; color: {JPSI_DARK};
         border-left: 4px solid {JPSI_BLUE}; padding: 2px 0 2px 10px;
         margin: 18px 0 8px 0; }}

  .callout {{
    background: rgba(6,147,227,0.05); border: 1px solid #d6e9f7;
    border-left: 4px solid {JPSI_BLUE}; border-radius: 8px;
    padding: 10px 16px; margin: 4px 0 14px 0;
    font-size: 0.9rem; color: {JPSI_DARK};
  }}
  .callout strong {{ color: {JPSI_BLUE}; }}

  [data-testid="stMetricLabel"] {{
    font-size: 0.75rem !important; color: #6b7280 !important;
    font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase;
  }}
  [data-testid="stMetricValue"] {{
    font-size: 1.3rem !important; font-weight: 700 !important; color: {JPSI_DARK} !important;
  }}
  [data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

  .sheet-wrap {{
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(50,55,60,0.10);
    border: 1px solid #e5e7eb; background: #fff; margin-bottom: 12px;
  }}
  hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 14px 0; }}
  .kpi-card {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px;
               padding:14px 18px; height:100%; }}
  .kpi-label {{ color:#6b7280; font-size:11px; font-weight:700;
                text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; }}
  .kpi-value {{ color:{JPSI_DARK}; font-size:26px; font-weight:700; line-height:1.1; }}
  .kpi-year  {{ color:#6b7280; font-size:11px; margin-top:5px; }}
  .kpi-delta {{ font-size:13px; margin-top:3px; font-weight:600; }}
  .kpi-pos   {{ color:#16a34a; }}
  .kpi-neg   {{ color:#dc2626; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th {{
    background: #f6f8fa; color: {JPSI_DARK};
    font-weight: 700; padding: 8px 12px; text-align: left;
    border-bottom: 2px solid #e5e7eb;
  }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(6,147,227,0.03); }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="dash-header">'
    f'<div class="dash-header-logo"><img src="{LOGO_URL}" alt="JSA"></div>'
    f'<div class="dash-header-text">'
    f'<h1>Soybean Crush Calculator</h1>'
    f'<div class="subtitle">CME Board (Hexane) vs. Extruder Plant GPM &nbsp;·&nbsp;'
    f' Commodity &amp; Ag Risk Management Specialists</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:14px 8px;border-bottom:3px solid {JPSI_BLUE};'
        f'margin:-1rem -1rem 20px -1rem;background:rgba(6,147,227,0.06);">'
        f'<img src="{LOGO_URL}" style="height:32px;margin-bottom:6px;" alt="JSA">'
        f'<h3 style="margin:0;color:{JPSI_DARK};font-size:0.9rem;font-weight:600;">Soy Crush Calculator</h3>'
        f'<small style="color:#6b7280;font-size:0.78rem;">Extruder vs. CME Board</small>'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        api_key: str = st.secrets.get("MASSIVE_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = st.text_input("Massive API Key", type="password")

    st.markdown(f'<div class="sh" style="margin-top:12px;">Extruder Yields</div>', unsafe_allow_html=True)
    st.caption("Edit to match your plant's actual output")

    oil_per_ton = st.number_input("Lbs oil per ton of meal",
        value=DEFAULT_OIL_LBS_PER_TON, min_value=100.0, max_value=600.0, step=1.0, format="%.1f")
    bu_per_ton  = st.number_input("Bushels of beans per ton of meal",
        value=DEFAULT_BU_PER_TON, min_value=30.0, max_value=60.0, step=0.01, format="%.2f")

    ext_meal_tons, ext_oil_lbs = ext_yields(oil_per_ton, bu_per_ton)

    st.markdown("---")
    st.markdown(f"""
<div style="font-size:0.85rem;color:{JPSI_DARK};">
  <div style="font-weight:700;margin-bottom:6px;">Extruder (calculated)</div>
  <div>Meal: <b>{ext_meal_tons:.5f}</b> t/bu &nbsp;({ext_meal_tons*2000:.2f} lbs)</div>
  <div>Oil: &nbsp;<b>{ext_oil_lbs:.4f}</b> lbs/bu</div>
  <div style="font-weight:700;margin:10px 0 6px 0;">CME Standard (hexane)</div>
  <div>Meal: <b>{CME_MEAL_TONS:.4f}</b> t/bu &nbsp;(44.00 lbs)</div>
  <div>Oil: &nbsp;<b>{CME_OIL_LBS:.2f}</b> lbs/bu</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    n_contracts = st.slider("Contracts to load", min_value=3, max_value=14, value=8)

# ── Auth gate ─────────────────────────────────────────────────────────────────

if not api_key:
    st.info("Enter your Massive API key in the sidebar to load live prices.")
    st.stop()

today = date.today()


@st.cache_data(ttl=300, show_spinner=False)
def _load_curves(key: str, n: int, as_of: str):
    d = date.fromisoformat(as_of)
    return (
        get_futures_curve("ZS", key, d, n_contracts=n),
        get_futures_curve("ZM", key, d, n_contracts=n),
        get_futures_curve("ZL", key, d, n_contracts=n),
    )


try:
    with st.spinner("Fetching ZS / ZM / ZL curves…"):
        zs_df, zm_df, zl_df = _load_curves(api_key, n_contracts, today.isoformat())
except MassiveApiError as exc:
    st.error(f"Massive API error: {exc}")
    st.stop()

if zs_df.empty or zm_df.empty or zl_df.empty:
    st.warning("No prices returned — check your API key or try refreshing.")
    st.stop()

aligned = align_curves(zs_df, zm_df, zl_df)
if aligned.empty:
    st.warning("No months common across ZS / ZM / ZL — increase 'Contracts to load'.")
    st.stop()

aligned["cme_crush"]   = aligned.apply(lambda r: gpm(r.ZM, r.ZL, r.ZS, CME_MEAL_TONS, CME_OIL_LBS), axis=1)
aligned["ext_crush"]   = aligned.apply(lambda r: gpm(r.ZM, r.ZL, r.ZS, ext_meal_tons, ext_oil_lbs), axis=1)
aligned["delta_cents"] = (aligned["cme_crush"] - aligned["ext_crush"]) * 100
front = aligned.iloc[0]

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

tab_live, tab_seasonal, tab_curve, tab_hedge, tab_position, tab_territory = st.tabs([
    "📊  Live Summary",
    "📅  Seasonal History",
    "🌊  Crush Curve",
    "🔧  Hedge Converter",
    "📋  Position",
    "🗺️  Draw Territory",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

with tab_live:
    st.markdown(
        f'<div class="sh">Nearest Aligned Month — {front["month"]}'
        f' <span style="font-size:0.8rem;font-weight:400;color:#6b7280;">'
        f'ZS {front.tkr_ZS} · ZM {front.tkr_ZM} · ZL {front.tkr_ZL}</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ZS — Soybeans", f"{front.ZS:.2f} ¢/bu",   f"${front.ZS/100:.4f}/bu")
    c2.metric("ZM — Meal",     f"${front.ZM:.2f}/ton")
    c3.metric("ZL — Oil",      f"{front.ZL:.3f} ¢/lb",   f"${front.ZL/100:.4f}/lb")
    c4.metric("CME Board Crush",  f"{front.cme_crush*100:.2f} ¢/bu", f"${front.cme_crush:.4f}/bu")
    c5.metric("Extruder Crush",   f"{front.ext_crush*100:.2f} ¢/bu", f"${front.ext_crush:.4f}/bu",
              delta_color="inverse")

    sf_color = NEG_RED if front.delta_cents > 0 else POS_GREEN
    st.markdown(
        f'<div class="callout">Extruder shortfall vs. CME board: '
        f'<strong style="color:{sf_color};">{-front.delta_cents:+.2f} ¢/bu</strong>'
        f' &nbsp;(${-front.delta_cents/100:+.4f}/bu)</div>',
        unsafe_allow_html=True,
    )

    # Component breakdown
    st.markdown('<div class="sh">Component Breakdown — Front Month</div>', unsafe_allow_html=True)
    meal_cme  = front.ZM * CME_MEAL_TONS
    oil_cme   = (front.ZL / 100) * CME_OIL_LBS
    meal_ext  = front.ZM * ext_meal_tons
    oil_ext   = (front.ZL / 100) * ext_oil_lbs
    bean_cost = front.ZS / 100

    def dc(v):
        c = POS_GREEN if v >= 0 else NEG_RED
        return f'<span style="color:{c};font-weight:600;">{v:+.2f} ¢</span>'

    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Component</th>
    <th style="text-align:right;">CME Board (hexane)</th>
    <th style="text-align:right;">Extruder Plant</th>
    <th style="text-align:right;">Difference</th>
  </tr></thead>
  <tbody>
    <tr><td>Meal revenue</td>
        <td style="text-align:right;">${meal_cme:.4f}/bu</td>
        <td style="text-align:right;">${meal_ext:.4f}/bu</td>
        <td style="text-align:right;">{dc((meal_ext-meal_cme)*100)}</td></tr>
    <tr><td>Oil revenue</td>
        <td style="text-align:right;">${oil_cme:.4f}/bu</td>
        <td style="text-align:right;">${oil_ext:.4f}/bu</td>
        <td style="text-align:right;">{dc((oil_ext-oil_cme)*100)}</td></tr>
    <tr style="background:#f9fafb;"><td>Bean cost</td>
        <td style="text-align:right;color:{NEG_RED};">−${bean_cost:.4f}/bu</td>
        <td style="text-align:right;color:{NEG_RED};">−${bean_cost:.4f}/bu</td>
        <td style="text-align:right;color:#9ca3af;">—</td></tr>
    <tr style="font-weight:700;border-top:2px solid #e5e7eb;"><td>GPM</td>
        <td style="text-align:right;color:{JPSI_BLUE};">${front.cme_crush:.4f}/bu</td>
        <td style="text-align:right;color:{JPSI_BLUE};">${front.ext_crush:.4f}/bu</td>
        <td style="text-align:right;">{dc((front.ext_crush-front.cme_crush)*100)}</td></tr>
  </tbody>
</table></div>
""", unsafe_allow_html=True)

    # Yield assumptions
    st.markdown('<div class="sh">Yield Assumptions</div>', unsafe_allow_html=True)
    total_cme = CME_MEAL_TONS * 2000 + CME_OIL_LBS
    total_ext = ext_meal_tons * 2000 + ext_oil_lbs
    oil_delta = ext_oil_lbs - CME_OIL_LBS
    meal_delta = (ext_meal_tons - CME_MEAL_TONS) * 2000

    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th></th>
    <th style="text-align:right;">CME Standard (hexane)</th>
    <th style="text-align:right;">Extruder Plant</th>
    <th style="text-align:right;">Δ</th>
  </tr></thead>
  <tbody>
    <tr><td>Oil extracted (lbs/bu)</td>
        <td style="text-align:right;">{CME_OIL_LBS:.2f}</td>
        <td style="text-align:right;">{ext_oil_lbs:.4f}</td>
        <td style="text-align:right;color:{NEG_RED};font-weight:600;">{oil_delta:+.3f} lbs</td></tr>
    <tr><td>Meal produced (lbs/bu)</td>
        <td style="text-align:right;">{CME_MEAL_TONS*2000:.2f}</td>
        <td style="text-align:right;">{ext_meal_tons*2000:.2f}</td>
        <td style="text-align:right;color:{POS_GREEN};font-weight:600;">{meal_delta:+.3f} lbs</td></tr>
    <tr><td>Meal (tons/bu)</td>
        <td style="text-align:right;">{CME_MEAL_TONS:.5f}</td>
        <td style="text-align:right;">{ext_meal_tons:.5f}</td>
        <td style="text-align:right;color:#9ca3af;">—</td></tr>
    <tr><td>Oil share of total lbs</td>
        <td style="text-align:right;">{CME_OIL_LBS/total_cme*100:.1f}%</td>
        <td style="text-align:right;">{ext_oil_lbs/total_ext*100:.1f}%</td>
        <td style="text-align:right;color:#9ca3af;">—</td></tr>
    <tr><td style="color:#6b7280;font-size:0.85rem;">Extraction method</td>
        <td style="text-align:right;color:#6b7280;font-size:0.85rem;">Hexane solvent</td>
        <td style="text-align:right;color:#6b7280;font-size:0.85rem;">Mechanical / expeller</td>
        <td></td></tr>
  </tbody>
</table></div>
<p style="font-size:0.82rem;color:#6b7280;margin-top:-6px;">
Extruder/expeller plants leave residual oil in the meal (higher-fat expeller cake),
so they extract significantly less oil per bushel but produce slightly more meal by weight.
</p>
""", unsafe_allow_html=True)

    # Sensitivity expander
    with st.expander("Sensitivity — Oil Price vs. Crush Shortfall"):
        gap_per_c = (CME_OIL_LBS - ext_oil_lbs) / 100
        oil_explain = (CME_OIL_LBS - ext_oil_lbs) * front.ZL / 100
        st.markdown(f"""
<div class="callout">
Each <strong>1 ¢/lb move in ZL (soy oil)</strong> changes the shortfall by
<strong>{gap_per_c:.4f} ¢/bu</strong>
({CME_OIL_LBS:.2f} − {ext_oil_lbs:.4f} = {CME_OIL_LBS - ext_oil_lbs:.4f} lbs ÷ 100).
At current ZL <strong>{front.ZL:.2f} ¢/lb</strong>, oil extraction alone explains
<strong>{oil_explain:.2f} ¢/bu</strong> of the <strong>{front.delta_cents:.2f} ¢/bu</strong> total.
</div>""", unsafe_allow_html=True)
        oil_range = list(range(25, 76))
        shortfalls = [
            (gpm(front.ZM, z, front.ZS, CME_MEAL_TONS, CME_OIL_LBS)
             - gpm(front.ZM, z, front.ZS, ext_meal_tons, ext_oil_lbs)) * 100
            for z in oil_range
        ]
        fig_s = go.Figure()
        fig_s.add_scatter(x=oil_range, y=shortfalls, mode="lines",
                          line=dict(color=NEG_RED, width=2.5),
                          fill="tozeroy", fillcolor="rgba(214,69,69,0.10)")
        fig_s.add_vline(x=front.ZL, line_dash="dash", line_color=JPSI_BLUE,
                        annotation_text=f"ZL {front.ZL:.2f}¢", annotation_font_color=JPSI_BLUE,
                        annotation_position="top right")
        fig_s.update_layout(xaxis_title="Soy Oil (¢/lb)", yaxis_title="Shortfall (¢/bu)",
                            height=260, margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor="#fff", paper_bgcolor="#fff",
                            font=dict(family="Source Sans Pro"), showlegend=False,
                            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig_s, use_container_width=True)

    with st.expander("Raw Massive API curves"):
        r1, r2, r3 = st.columns(3)
        r1.markdown("**ZS — Soybeans (¢/bu)**")
        r1.dataframe(zs_df.rename(columns={"price": "¢/bu"}), use_container_width=True, hide_index=True)
        r2.markdown("**ZM — Meal ($/ton)**")
        r2.dataframe(zm_df.rename(columns={"price": "$/ton"}), use_container_width=True, hide_index=True)
        r3.markdown("**ZL — Oil (¢/lb)**")
        r3.dataframe(zl_df.rename(columns={"price": "¢/lb"}), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SEASONAL HISTORY
# ─────────────────────────────────────────────────────────────────────────────

with tab_seasonal:
    st.markdown(f"""
<div class="callout">
<strong>Seasonal overlay</strong> — for a selected crush month, each line shows the daily gross
processing margin leading up to contract expiration, one line per year.
X-axis is aligned on <strong>days to expiration</strong> of the bean contract so seasonal
patterns line up regardless of year.  Massive history starts ~Sep 2021; older contracts
may have shorter windows.
</div>""", unsafe_allow_html=True)

    # Controls row
    sc1, sc2, sc3, sc4 = st.columns([2, 2, 2, 2])
    with sc1:
        crush_month = st.selectbox(
            "Crush month",
            list(CRUSH_MONTHS.keys()),
            index=list(CRUSH_MONTHS.keys()).index("Nov"),
            help="Bean contract month to chart. Nov beans pair with Dec meal/oil.",
        )
    with sc2:
        n_years = st.slider("Years back", min_value=2, max_value=5, value=4)
    with sc3:
        show_avg = st.checkbox("Show average line", value=True)
    with sc4:
        crush_type = st.radio("Crush type", ["CME Board", "Extruder", "Both"],
                              horizontal=True, index=2)

    zs_m, zm_m, zl_m = CRUSH_MONTHS[crush_month]
    zm_label = "Dec" if zm_m == "Z" else crush_month
    pairing_note = (f"Nov beans (ZSX) → Dec meal (ZMZ) + Dec oil (ZLZ)"
                    if crush_month == "Nov" else
                    f"{crush_month} beans (ZS{zs_m}) → {crush_month} meal (ZM{zm_m}) + {crush_month} oil (ZL{zl_m})")

    st.caption(f"Contract pairing: {pairing_note}")

    @st.cache_data(ttl=6*3600, show_spinner="Loading crush history…")
    def _load_seasonal(month: str, key: str, base_yr: int, n: int,
                       cm_tons: float, cm_oil: float, em_tons: float, em_oil: float):
        return build_seasonal_data(month, key, base_yr, n,
                                   cm_tons, cm_oil, em_tons, em_oil)

    with st.spinner(f"Pulling {n_years} years of {crush_month} crush history…"):
        seasonal_data = _load_seasonal(
            crush_month, api_key, today.year, n_years,
            CME_MEAL_TONS, CME_OIL_LBS, ext_meal_tons, ext_oil_lbs,
        )

    if not seasonal_data:
        st.warning("No historical data returned for this crush month — try a different month or check your API key.")
    else:
        labels = list(seasonal_data.keys())
        current_label = labels[-1]  # most recent year

        # Anchor expiry: use current year's ZS expiry for calendar x-axis labels
        anchor_expiry = seasonal_data[current_label]["expiry"]

        # Window: DTE range to display
        window = 400  # days before expiry

        def make_seasonal_fig(series_key: str, crush_label: str, color_offset: int = 0) -> go.Figure:
            fig = go.Figure()
            all_series = []
            for i, (label, d) in enumerate(seasonal_data.items()):
                s = d[series_key] * 100   # convert to ¢/bu
                s = s[s.index >= -window]
                if s.empty:
                    continue
                all_series.append(s)
                x_dates = dte_to_calendar(s.index, anchor_expiry)
                is_current = (label == current_label)
                color = YEAR_COLORS[i % len(YEAR_COLORS)]
                fig.add_scatter(
                    x=x_dates, y=s.values,
                    name=label,
                    mode="lines",
                    line=dict(color=color, width=2.5 if is_current else 1.5,
                              dash="solid" if is_current else "solid"),
                    opacity=1.0 if is_current else 0.7,
                    hovertemplate=f"{label}<br>%{{y:.2f}} ¢/bu<extra></extra>",
                )

            if show_avg and len(all_series) >= 2:
                avg = year_average(all_series) * 100
                if not avg.empty:
                    x_avg = dte_to_calendar(avg.index, anchor_expiry)
                    fig.add_scatter(
                        x=x_avg, y=avg.values, name=f"Avg ({len(all_series)} yr)",
                        mode="lines",
                        line=dict(color=AVG_COLOR, width=2, dash="dash"),
                        hovertemplate="Avg<br>%{y:.2f} ¢/bu<extra></extra>",
                    )

            # Expiry marker — use add_shape/add_annotation instead of add_vline to
            # avoid a Plotly bug where _mean(x) calls sum([Timestamp,Timestamp]) → error
            expiry_str = (anchor_expiry.strftime("%Y-%m-%d")
                          if hasattr(anchor_expiry, "strftime") else str(anchor_expiry))
            fig.add_shape(type="line", x0=expiry_str, x1=expiry_str, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(dash="dot", color="#c0392b", width=1.5))
            fig.add_annotation(x=expiry_str, y=0.98, xref="x", yref="paper",
                               text=f"{crush_month} expiry (est.)",
                               showarrow=False, xanchor="left", yanchor="top",
                               font=dict(color="#c0392b", size=11))

            fig.update_layout(
                title=dict(text=f"{crush_month} Crush — {crush_label} (¢/bu)", x=0,
                           font=dict(size=13, color=JPSI_DARK)),
            )
            plotly_defaults(fig, height=420)
            fig.update_xaxes(dtick="M1", tickformat="%b '%y")
            return fig

        if crush_type in ("CME Board", "Both"):
            fig_cme = make_seasonal_fig("cme_dte", "CME Board (hexane)")
            st.plotly_chart(fig_cme, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

        if crush_type in ("Extruder", "Both"):
            fig_ext = make_seasonal_fig("ext_dte", "Extruder Plant")
            st.plotly_chart(fig_ext, use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

        if crush_type == "Both":
            # Overlay comparison: current year only, both crush types
            st.markdown('<div class="sh">Current Year: CME Board vs. Extruder</div>', unsafe_allow_html=True)
            cur = seasonal_data[current_label]
            cme_cur = (cur["cme_dte"] * 100)[cur["cme_dte"].index >= -window]
            ext_cur = (cur["ext_dte"] * 100)[cur["ext_dte"].index >= -window]

            fig_both = go.Figure()
            if not cme_cur.empty:
                fig_both.add_scatter(
                    x=dte_to_calendar(cme_cur.index, anchor_expiry), y=cme_cur.values,
                    name=f"CME Board ({current_label})", mode="lines",
                    line=dict(color=JPSI_BLUE, width=2.5),
                )
            if not ext_cur.empty:
                fig_both.add_scatter(
                    x=dte_to_calendar(ext_cur.index, anchor_expiry), y=ext_cur.values,
                    name=f"Extruder ({current_label})", mode="lines",
                    line=dict(color="#f6821f", width=2.5),
                )
            if not cme_cur.empty and not ext_cur.empty:
                delta = (cme_cur - ext_cur).reindex(cme_cur.index)
                fig_both.add_scatter(
                    x=dte_to_calendar(delta.index, anchor_expiry), y=delta.values,
                    name="Shortfall (CME − Ext)", mode="lines",
                    line=dict(color=NEG_RED, dash="dot", width=1.5),
                    yaxis="y2",
                )
            fig_both.update_layout(
                yaxis2=dict(title="Shortfall (¢/bu)", overlaying="y", side="right",
                            showgrid=False, tickformat="+.1f"),
            )
            plotly_defaults(fig_both, height=380)
            fig_both.update_xaxes(dtick="M1", tickformat="%b '%y")
            st.plotly_chart(fig_both, use_container_width=True,
                            config={"displaylogo": False})

        # Summary stats table
        st.markdown('<div class="sh">Season Summary Stats</div>', unsafe_allow_html=True)
        rows_html = ""
        for label, d in seasonal_data.items():
            s = d["cme_dte"] * 100
            s = s[s.index >= -window]
            if s.empty:
                continue
            se = d["ext_dte"] * 100
            se = se[se.index >= -window]
            rows_html += (
                f'<tr{"style=\"font-weight:700;\"" if label == current_label else ""}>'
                f"<td>{label}</td>"
                f"<td style='text-align:right;color:{JPSI_BLUE};'>{s.mean():.1f} ¢</td>"
                f"<td style='text-align:right;'>{s.min():.1f} ¢</td>"
                f"<td style='text-align:right;'>{s.max():.1f} ¢</td>"
                f"<td style='text-align:right;color:#f6821f;'>{se.mean():.1f} ¢</td>"
                f"<td style='text-align:right;color:{NEG_RED};'>{(s-se.reindex(s.index)).mean():.1f} ¢</td>"
                f"<td style='text-align:right;color:#6b7280;'>{len(s)}</td>"
                f"</tr>"
            )
        st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Contract</th>
    <th style="text-align:right;color:{JPSI_BLUE};">CME Avg</th>
    <th style="text-align:right;">CME Min</th>
    <th style="text-align:right;">CME Max</th>
    <th style="text-align:right;color:#f6821f;">Ext Avg</th>
    <th style="text-align:right;color:{NEG_RED};">Avg Shortfall</th>
    <th style="text-align:right;color:#6b7280;">Sessions</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CRUSH CURVE (forward curve)
# ─────────────────────────────────────────────────────────────────────────────

with tab_curve:
    st.markdown(f"""
<div class="callout">
<strong>Forward crush curve</strong> — current live crush values across all available aligned
contract months.  The curve shows what a processor can lock in at today's futures prices
for each delivery period.  CME board uses hexane-standard yields; Extruder uses your
plant's actual parameters.
</div>""", unsafe_allow_html=True)

    # Main forward curve chart
    st.markdown('<div class="sh">Crush Spread — Forward Curve</div>', unsafe_allow_html=True)

    fig_fc = go.Figure()
    fig_fc.add_bar(
        x=aligned["month"],
        y=(aligned["cme_crush"] * 100).round(2),
        name="CME Board (hexane)",
        marker_color=JPSI_BLUE,
        offsetgroup=0,
        hovertemplate="%{x}<br>CME: %{y:.2f} ¢/bu<extra></extra>",
    )
    fig_fc.add_bar(
        x=aligned["month"],
        y=(aligned["ext_crush"] * 100).round(2),
        name="Extruder Plant",
        marker_color="#f6821f",
        offsetgroup=1,
        hovertemplate="%{x}<br>Extruder: %{y:.2f} ¢/bu<extra></extra>",
    )
    fig_fc.add_scatter(
        x=aligned["month"],
        y=aligned["delta_cents"].round(2),
        name="Shortfall (CME − Ext)",
        mode="lines+markers",
        line=dict(color=NEG_RED, dash="dot", width=2),
        marker=dict(size=7, color=NEG_RED),
        yaxis="y2",
        hovertemplate="%{x}<br>Shortfall: %{y:.2f} ¢/bu<extra></extra>",
    )
    fig_fc.update_layout(
        barmode="group",
        yaxis=dict(title="Gross Processing Margin (¢/bu)", gridcolor="#f0f0f0",
                   zeroline=True, zerolinecolor="#e5e7eb"),
        yaxis2=dict(title="CME − Extruder shortfall (¢/bu)",
                    overlaying="y", side="right", showgrid=False, tickformat="+.1f"),
    )
    plotly_defaults(fig_fc, height=420)
    st.plotly_chart(fig_fc, use_container_width=True,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # Component stacked chart
    st.markdown('<div class="sh">Component Breakdown — Forward Curve</div>', unsafe_allow_html=True)
    st.caption("Stacked bars show revenue from meal and oil vs. the bean cost, for each contract month.")

    months    = aligned["month"].tolist()
    meal_revs = (aligned["ZM"] * CME_MEAL_TONS * 100).round(2)
    oil_revs  = (aligned["ZL"] / 100 * CME_OIL_LBS * 100).round(2)
    bean_costs= (aligned["ZS"] / 100 * 100).round(2)   # already ¢

    fig_comp = go.Figure()
    fig_comp.add_bar(x=months, y=meal_revs, name="Meal revenue (CME)",
                     marker_color="#5aa469", offsetgroup=0,
                     hovertemplate="%{x}<br>Meal: %{y:.2f} ¢/bu<extra></extra>")
    fig_comp.add_bar(x=months, y=oil_revs,  name="Oil revenue (CME)",
                     marker_color="#0693e3", offsetgroup=0, base=meal_revs,
                     hovertemplate="%{x}<br>Oil: %{y:.2f} ¢/bu<extra></extra>")
    fig_comp.add_scatter(x=months, y=bean_costs, name="Bean cost",
                         mode="lines+markers",
                         line=dict(color=NEG_RED, width=2),
                         marker=dict(size=7),
                         hovertemplate="%{x}<br>Beans: %{y:.2f} ¢/bu<extra></extra>")

    fig_comp.update_layout(
        barmode="stack",
        yaxis=dict(title="¢ per bushel of beans", gridcolor="#f0f0f0"),
    )
    plotly_defaults(fig_comp, height=380)
    st.plotly_chart(fig_comp, use_container_width=True,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # Full detail table
    st.markdown('<div class="sh">Contract Detail</div>', unsafe_allow_html=True)
    rows_html = ""
    for _, r in aligned.iterrows():
        sf = r["delta_cents"]
        sfc = NEG_RED if sf > 0 else POS_GREEN
        rows_html += (
            f'<tr>'
            f"<td style='font-weight:600;color:{JPSI_DARK};'>{r['month']}</td>"
            f"<td style='text-align:right;'>{r.ZS:.2f} ¢</td>"
            f"<td style='text-align:right;'>${r.ZM:.2f}</td>"
            f"<td style='text-align:right;'>{r.ZL:.3f} ¢</td>"
            f"<td style='text-align:right;color:{JPSI_BLUE};font-weight:600;'>{r.cme_crush*100:.2f} ¢</td>"
            f"<td style='text-align:right;color:#f6821f;font-weight:600;'>{r.ext_crush*100:.2f} ¢</td>"
            f"<td style='text-align:right;color:{sfc};font-weight:600;'>{-sf:+.2f} ¢</td>"
            f"</tr>"
        )
    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Month</th>
    <th style="text-align:right;">Beans (¢/bu)</th>
    <th style="text-align:right;">Meal ($/ton)</th>
    <th style="text-align:right;">Oil (¢/lb)</th>
    <th style="text-align:right;color:{JPSI_BLUE};">CME Crush</th>
    <th style="text-align:right;color:#f6821f;">Ext. Crush</th>
    <th style="text-align:right;">Shortfall</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)

    # CSV export
    export_df = aligned[["month", "ZS", "ZM", "ZL"]].copy()
    export_df["cme_crush_cents"] = (aligned["cme_crush"] * 100).round(3)
    export_df["ext_crush_cents"] = (aligned["ext_crush"] * 100).round(3)
    export_df["shortfall_cents"] = (-aligned["delta_cents"]).round(3)
    export_df.columns = ["Month", "ZS (¢/bu)", "ZM ($/ton)", "ZL (¢/lb)",
                         "CME Crush (¢/bu)", "Ext Crush (¢/bu)", "Shortfall (¢/bu)"]
    st.download_button(
        "⬇ Download CSV",
        export_df.to_csv(index=False).encode(),
        f"soy_crush_curve_{today.isoformat()}.csv",
        "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HEDGE CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

with tab_hedge:
    st.markdown(f"""
<div class="callout">
<strong>Hedge Converter</strong> — Enter your live ZS / ZM / ZL futures positions below.
The calculator shows how a CME board-crush hedge maps to your extruder's actual exposure,
what gaps exist, and marks your book to current market prices.
<br><br>
<strong>Sign convention:</strong> negative qty = short (selling), positive = long (buying).
A standard <em>long crush</em> position = short ZS + long ZM + long ZL.
</div>""", unsafe_allow_html=True)

    # ── Position blotter ──────────────────────────────────────────────────────

    st.markdown('<div class="sh">Position Blotter</div>', unsafe_allow_html=True)
    st.caption("Edit qty and entry directly in the table. Changes are saved for your next session.")

    if "positions" not in st.session_state:
        st.session_state["positions"] = _load_positions()

    pos_df = pd.DataFrame(st.session_state["positions"])

    edited = st.data_editor(
        pos_df,
        column_config={
            "Leg":   st.column_config.SelectboxColumn("Leg", options=["ZS", "ZM", "ZL"], required=True, width="small"),
            "Month": st.column_config.TextColumn("Month", help='e.g. "Nov \'26"', width="small"),
            "Qty":   st.column_config.NumberColumn("Qty (contracts)", help="+long / -short", step=1, format="%d"),
            "Entry": st.column_config.NumberColumn("Entry price", help="¢/bu for ZS, $/ton for ZM, ¢/lb for ZL",
                                                    min_value=0.0, format="%.3f"),
            "Note":  st.column_config.TextColumn("Note", width="large"),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="blotter_editor",
    )

    bc1, bc2 = st.columns([1, 5])
    if bc1.button("💾 Save positions", type="primary"):
        saved = edited.to_dict("records")
        st.session_state["positions"] = saved
        _save_positions(saved)
        st.success("Positions saved.")
    if bc2.button("↺ Reset to defaults"):
        st.session_state["positions"] = [row.copy() for row in _POS_DEFAULTS]
        _save_positions(st.session_state["positions"])
        st.rerun()

    # ── Aggregate positions ───────────────────────────────────────────────────

    rows = edited.dropna(subset=["Leg", "Qty"]).to_dict("records")

    def _agg(leg: str) -> tuple[float, float]:
        """Returns (net_qty, weighted_avg_entry) for a given leg."""
        subset = [r for r in rows if r.get("Leg") == leg and r.get("Qty")]
        if not subset:
            return 0.0, 0.0
        qty_total = sum(r["Qty"] for r in subset)
        if qty_total == 0:
            return 0.0, float(subset[0].get("Entry") or 0)
        wavg = sum(r["Qty"] * (r.get("Entry") or 0) for r in subset) / qty_total
        return qty_total, wavg

    zs_qty, zs_entry = _agg("ZS")
    zm_qty, zm_entry = _agg("ZM")
    zl_qty, zl_entry = _agg("ZL")

    bu_hedged = zs_qty * ZS_BU  # positive = long beans, negative = short beans (long crush)

    # ── KPI strip ─────────────────────────────────────────────────────────────

    st.markdown('<div class="sh">Position Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    crush_dir = "Long crush" if zs_qty < 0 else ("Short crush" if zs_qty > 0 else "No ZS")
    k1.metric("Beans hedged", f"{abs(bu_hedged):,.0f} bu",
              f"{abs(zs_qty):.0f} ZS contracts · {crush_dir}")

    # Crush locked (using entry prices vs live)
    has_entries = zs_entry > 0 and zm_entry > 0 and zl_entry > 0
    if has_entries:
        cme_locked   = gpm(zm_entry, zl_entry, zs_entry, CME_MEAL_TONS, CME_OIL_LBS)
        ext_locked   = gpm(zm_entry, zl_entry, zs_entry, ext_meal_tons, ext_oil_lbs)
        live_cme_gpm = gpm(front.ZM, front.ZL, front.ZS, CME_MEAL_TONS, CME_OIL_LBS)
        live_ext_gpm = gpm(front.ZM, front.ZL, front.ZS, ext_meal_tons, ext_oil_lbs)
        k2.metric("CME crush locked", f"{cme_locked*100:.2f} ¢/bu",
                  f"Live {live_cme_gpm*100:.2f} ¢")
        k3.metric("Extruder crush locked", f"{ext_locked*100:.2f} ¢/bu",
                  f"Live {live_ext_gpm*100:.2f} ¢")
        k4.metric("Entry shortfall (locked)",
                  f"{(cme_locked - ext_locked)*100:.2f} ¢/bu",
                  "CME − Extruder at entry")
    else:
        k2.metric("CME crush locked", "—", "Enter entry prices")
        k3.metric("Extruder crush locked", "—", "")
        k4.metric("Entry shortfall", "—", "")

    # Mark-to-market total P&L
    if has_entries and abs(zs_qty) > 0:
        pnl_zs = (front.ZS - zs_entry) / 100 * zs_qty * ZS_BU
        pnl_zm = (front.ZM - zm_entry) * zm_qty * ZM_TONS
        pnl_zl = (front.ZL - zl_entry) / 100 * zl_qty * ZL_LBS
        total_pnl = pnl_zs + pnl_zm + pnl_zl
        pnl_color = POS_GREEN if total_pnl >= 0 else NEG_RED
        k5.metric("Mark-to-market P&L",
                  f"${total_pnl:+,.0f}",
                  f"ZS {pnl_zs:+,.0f} / ZM {pnl_zm:+,.0f} / ZL {pnl_zl:+,.0f}")
    else:
        k5.metric("Mark-to-market P&L", "—", "Enter positions + entry prices")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Hedge gap analysis ────────────────────────────────────────────────────

    st.markdown('<div class="sh">Hedge Adequacy — CME Board vs. Extruder</div>', unsafe_allow_html=True)
    st.caption(
        f"Based on {abs(bu_hedged):,.0f} bu hedged ({abs(zs_qty):.0f} ZS contracts × {ZS_BU:,} bu). "
        f"CME contract sizes: ZS {ZS_BU:,} bu · ZM {ZM_TONS} tons · ZL {ZL_LBS:,} lbs"
    )

    # Ideal targets (signs: for long crush, ZS<0, ZM>0, ZL>0)
    factor = -bu_hedged  # positive when long crush (short ZS)
    zm_cme_ideal = factor * CME_MEAL_TONS / ZM_TONS
    zl_cme_ideal = factor * CME_OIL_LBS  / ZL_LBS
    zm_ext_ideal = factor * ext_meal_tons / ZM_TONS
    zl_ext_ideal = factor * ext_oil_lbs  / ZL_LBS

    # Gaps
    zm_vs_cme = zm_qty - zm_cme_ideal
    zl_vs_cme = zl_qty - zl_cme_ideal
    zm_vs_ext = zm_qty - zm_ext_ideal
    zl_vs_ext = zl_qty - zl_ext_ideal

    def _gap_td(v: float) -> str:
        if abs(v) < 0.05:
            return f'<td style="text-align:right;color:{POS_GREEN};font-weight:600;">✓ flat</td>'
        c = NEG_RED if abs(v) > 1 else "#e8833a"
        label = f"{v:+.2f}"
        return f'<td style="text-align:right;color:{c};font-weight:600;">{label}</td>'

    def _qty_td(v: float, ideal: float) -> str:
        c = JPSI_BLUE
        return (f'<td style="text-align:right;color:{c};">{v:+.2f}</td>'
                f'<td style="text-align:right;color:#6b7280;">{ideal:.2f}</td>')

    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Leg</th>
    <th style="text-align:right;">Current</th>
    <th style="text-align:right;color:{JPSI_BLUE};">CME Ideal</th>
    <th style="text-align:right;">Gap vs. CME</th>
    <th style="text-align:right;color:#f6821f;">Ext. Ideal</th>
    <th style="text-align:right;">Gap vs. Ext.</th>
    <th style="text-align:right;">Gap $ value (live)</th>
  </tr></thead>
  <tbody>
    <tr>
      <td style="font-weight:700;">ZS</td>
      <td style="text-align:right;color:{JPSI_BLUE};">{zs_qty:+.0f}</td>
      <td style="text-align:right;color:#6b7280;">—</td>
      <td style="text-align:right;color:#6b7280;">—</td>
      <td style="text-align:right;color:#6b7280;">—</td>
      <td style="text-align:right;color:#6b7280;">—</td>
      <td style="text-align:right;color:#6b7280;">—</td>
    </tr>
    <tr>
      <td style="font-weight:700;">ZM</td>
      <td style="text-align:right;color:{JPSI_BLUE};">{zm_qty:+.2f}</td>
      <td style="text-align:right;color:#6b7280;">{zm_cme_ideal:+.2f}</td>
      {_gap_td(zm_vs_cme)}
      <td style="text-align:right;color:#f6821f;">{zm_ext_ideal:+.2f}</td>
      {_gap_td(zm_vs_ext)}
      <td style="text-align:right;">${zm_vs_ext * ZM_TONS * front.ZM:+,.0f}</td>
    </tr>
    <tr>
      <td style="font-weight:700;">ZL</td>
      <td style="text-align:right;color:{JPSI_BLUE};">{zl_qty:+.2f}</td>
      <td style="text-align:right;color:#6b7280;">{zl_cme_ideal:+.2f}</td>
      {_gap_td(zl_vs_cme)}
      <td style="text-align:right;color:#f6821f;">{zl_ext_ideal:+.2f}</td>
      {_gap_td(zl_vs_ext)}
      <td style="text-align:right;">${zl_vs_ext * ZL_LBS * front.ZL / 100:+,.0f}</td>
    </tr>
  </tbody>
</table></div>""", unsafe_allow_html=True)

    # ── Adjustment recommendations ────────────────────────────────────────────

    st.markdown('<div class="sh">To Convert CME Hedge → Extruder Hedge</div>', unsafe_allow_html=True)

    zm_adj = zm_ext_ideal - zm_cme_ideal   # per full set of contracts
    zl_adj = zl_ext_ideal - zl_cme_ideal

    adj_rows = []
    if abs(zl_adj) >= 0.05:
        action = "SELL" if zl_adj < 0 else "BUY"
        adj_rows.append(
            f"<li><strong>ZL:</strong> {action} <strong>{abs(zl_adj):.2f} contracts</strong> "
            f"({zl_cme_ideal:+.2f} CME → {zl_ext_ideal:+.2f} Ext) — "
            f"oil yield {CME_OIL_LBS:.1f} lbs/bu → {ext_oil_lbs:.3f} lbs/bu</li>"
        )
    if abs(zm_adj) >= 0.05:
        action = "BUY" if zm_adj > 0 else "SELL"
        adj_rows.append(
            f"<li><strong>ZM:</strong> {action} <strong>{abs(zm_adj):.2f} contracts</strong> "
            f"({zm_cme_ideal:+.2f} CME → {zm_ext_ideal:+.2f} Ext) — "
            f"meal yield {CME_MEAL_TONS*2000:.1f} lbs/bu → {ext_meal_tons*2000:.2f} lbs/bu</li>"
        )

    if adj_rows:
        adj_value_zl = abs(zl_adj) * ZL_LBS * front.ZL / 100
        adj_value_zm = abs(zm_adj) * ZM_TONS * front.ZM
        st.markdown(f"""
<div class="callout">
Starting from a <strong>CME board-crush hedge</strong> on {abs(zs_qty):.0f} ZS contracts,
to hedge your extruder's actual output you need to adjust:
<ul style="margin:8px 0 4px 0;">
{"".join(adj_rows)}
</ul>
<strong>Combined adjustment notional:</strong>
ZL ~${adj_value_zl:,.0f} · ZM ~${adj_value_zm:,.0f}
(using live front-month prices)
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="callout" style="border-left-color:#2e7d32;">'
            'Position is aligned to extruder ratios — no adjustment needed.</div>',
            unsafe_allow_html=True,
        )

    # ── Ratio reference table ─────────────────────────────────────────────────

    with st.expander("Hedge ratio reference — per ZS contract (5,000 bu)"):
        r_zm_cme = ZS_BU * CME_MEAL_TONS / ZM_TONS
        r_zl_cme = ZS_BU * CME_OIL_LBS  / ZL_LBS
        r_zm_ext = ZS_BU * ext_meal_tons / ZM_TONS
        r_zl_ext = ZS_BU * ext_oil_lbs  / ZL_LBS

        st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Leg</th>
    <th style="text-align:right;color:{JPSI_BLUE};">CME Standard (hexane)</th>
    <th style="text-align:right;color:#f6821f;">Extruder Plant</th>
    <th style="text-align:right;">Δ per ZS contract</th>
    <th style="text-align:right;">Standard lot (10 ZS)</th>
  </tr></thead>
  <tbody>
    <tr><td>ZM contracts</td>
        <td style="text-align:right;color:{JPSI_BLUE};">{r_zm_cme:.4f}</td>
        <td style="text-align:right;color:#f6821f;">{r_zm_ext:.4f}</td>
        <td style="text-align:right;">{r_zm_ext-r_zm_cme:+.4f}</td>
        <td style="text-align:right;">CME {r_zm_cme*10:.2f} → Ext {r_zm_ext*10:.2f}</td></tr>
    <tr><td>ZL contracts</td>
        <td style="text-align:right;color:{JPSI_BLUE};">{r_zl_cme:.4f}</td>
        <td style="text-align:right;color:#f6821f;">{r_zl_ext:.4f}</td>
        <td style="text-align:right;color:{NEG_RED};">{r_zl_ext-r_zl_cme:+.4f}</td>
        <td style="text-align:right;color:{NEG_RED};">CME {r_zl_cme*10:.2f} → Ext {r_zl_ext*10:.2f}</td></tr>
    <tr><td style="color:#6b7280;">Meal yield (lbs/bu)</td>
        <td style="text-align:right;color:#6b7280;">{CME_MEAL_TONS*2000:.1f}</td>
        <td style="text-align:right;color:#6b7280;">{ext_meal_tons*2000:.2f}</td>
        <td style="text-align:right;color:{POS_GREEN};">{(ext_meal_tons-CME_MEAL_TONS)*2000:+.2f} lbs</td>
        <td></td></tr>
    <tr><td style="color:#6b7280;">Oil yield (lbs/bu)</td>
        <td style="text-align:right;color:#6b7280;">{CME_OIL_LBS:.1f}</td>
        <td style="text-align:right;color:#6b7280;">{ext_oil_lbs:.3f}</td>
        <td style="text-align:right;color:{NEG_RED};">{ext_oil_lbs-CME_OIL_LBS:+.3f} lbs</td>
        <td></td></tr>
  </tbody>
</table></div>""", unsafe_allow_html=True)

    # ── Price sensitivity / stress ────────────────────────────────────────────

    st.markdown('<div class="sh">Price Sensitivity</div>', unsafe_allow_html=True)
    st.caption("Slide to stress-test the locked crush GPM and position P&L at different price shocks.")

    sc1, sc2, sc3 = st.columns(3)
    zs_shock = sc1.slider("ZS shock (¢/bu)", min_value=-150, max_value=150, value=0, step=5)
    zm_shock = sc2.slider("ZM shock ($/ton)", min_value=-50, max_value=50, value=0, step=1)
    zl_shock = sc3.slider("ZL shock (¢/lb)", min_value=-20, max_value=20, value=0, step=1)

    s_zs = front.ZS + zs_shock
    s_zm = front.ZM + zm_shock
    s_zl = front.ZL + zl_shock

    s_cme = gpm(s_zm, s_zl, s_zs, CME_MEAL_TONS, CME_OIL_LBS)
    s_ext = gpm(s_zm, s_zl, s_zs, ext_meal_tons, ext_oil_lbs)

    sc1b, sc2b, sc3b, sc4b = st.columns(4)
    sc1b.metric("Stressed CME crush",  f"{s_cme*100:.2f} ¢/bu",
                f"{(s_cme - gpm(front.ZM, front.ZL, front.ZS, CME_MEAL_TONS, CME_OIL_LBS))*100:+.2f} ¢")
    sc2b.metric("Stressed Ext. crush", f"{s_ext*100:.2f} ¢/bu",
                f"{(s_ext - gpm(front.ZM, front.ZL, front.ZS, ext_meal_tons, ext_oil_lbs))*100:+.2f} ¢")

    if has_entries and abs(zs_qty) > 0:
        s_pnl_zs = (s_zs - zs_entry) / 100 * zs_qty * ZS_BU
        s_pnl_zm = (s_zm - zm_entry) * zm_qty * ZM_TONS
        s_pnl_zl = (s_zl - zl_entry) / 100 * zl_qty * ZL_LBS
        s_total   = s_pnl_zs + s_pnl_zm + s_pnl_zl
        sc3b.metric("Stressed P&L", f"${s_total:+,.0f}",
                    f"ZS {s_pnl_zs:+,.0f} / ZM {s_pnl_zm:+,.0f}")
        sc4b.metric("ZL stressed P&L", f"${s_pnl_zl:+,.0f}",
                    f"Oil-leg slippage: ${(zl_vs_ext * ZL_LBS * (s_zl - (zl_entry or s_zl)) / 100):+,.0f}")
    else:
        sc3b.metric("Stressed P&L", "—", "Enter entry prices")
        sc4b.metric("ZL slippage P&L", "—", "")

    # Heatmap: ZS shock vs ZL shock on extruder GPM
    st.markdown('<div class="sh">Extruder GPM Heatmap — ZS vs ZL Price Shock</div>', unsafe_allow_html=True)
    st.caption("Each cell shows stressed extruder GPM (¢/bu). ZM held at live price.")

    zs_range = list(range(-100, 105, 25))
    zl_range = list(range(-15, 16, 5))
    heat_z = []
    for zl_s in reversed(zl_range):
        row_vals = []
        for zs_s in zs_range:
            g = gpm(front.ZM + zm_shock, front.ZL + zl_s, front.ZS + zs_s,
                    ext_meal_tons, ext_oil_lbs) * 100
            row_vals.append(round(g, 1))
        heat_z.append(row_vals)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_z,
        x=[f"{v:+d}¢" for v in zs_range],
        y=[f"{v:+d}¢" for v in reversed(zl_range)],
        colorscale=[[0, NEG_RED], [0.5, "#ffffff"], [1, POS_GREEN]],
        text=heat_z, texttemplate="%{text:.1f}",
        colorbar=dict(title="¢/bu", len=0.8),
        hovertemplate="ZS %{x} / ZL %{y}<br>Ext GPM: %{z:.1f} ¢/bu<extra></extra>",
    ))
    fig_heat.update_layout(
        xaxis_title="ZS shock (¢/bu)", yaxis_title="ZL shock (¢/lb)",
        height=340, margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        font=dict(family="Source Sans Pro", color=JPSI_DARK),
    )
    st.plotly_chart(fig_heat, use_container_width=True,
                    config={"displaylogo": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — POSITION
# ─────────────────────────────────────────────────────────────────────────────

with tab_position:
    st.markdown(f"""
<div class="callout">
<strong>Crush Position</strong> — Monthly supply schedule, crush plan, and meal/oil balance
for your plant.  Opening inventory cascades month-over-month from closing.
Meal and oil produced are calculated automatically from your crush schedule and
extruder yield parameters set in the sidebar.
</div>""", unsafe_allow_html=True)

    # ── Load / init state ─────────────────────────────────────────────────────

    if "crush_pos" not in st.session_state:
        st.session_state["crush_pos"] = _load_crush_pos()

    cp = st.session_state["crush_pos"]
    pm_labels = _pos_month_labels()  # 11 month labels starting this month

    _ext_col, _ = st.columns([3, 5])
    _show_ext = _ext_col.checkbox(
        "📅  Expand to extended view (through Aug)",
        value=False,
        key="crush_extended",
    )
    n_show = 11 if _show_ext else 6

    # ── Quick import from CH Basis report ────────────────────────────────────

    with st.expander("📋  Import from CH Basis Report"):
        _imp_tab_img, _imp_tab_txt = st.tabs(["📷  Upload image / screenshot", "📝  Paste text"])

        with _imp_tab_img:
            st.caption(
                "Upload a screenshot or PDF of the CH Basis report. "
                "Claude Vision reads the Basis Position + Unpriced Purchases columns "
                "per location and month and fills in the to-arrive schedule automatically."
            )
            _has_key = bool(st.secrets.get("ANTHROPIC_API_KEY", ""))
            if not _has_key:
                st.warning(
                    "Add `ANTHROPIC_API_KEY` to your Streamlit Cloud secrets to enable "
                    "image extraction. The key is available in the JSA 1Password vault."
                )
            _img_file = st.file_uploader(
                "CH Basis report image (PNG, JPG) or PDF",
                type=["png", "jpg", "jpeg", "pdf"],
                key="ch_img_upload",
                disabled=not _has_key,
            )
            if _img_file and st.button("Extract from image", key="ch_img_btn", disabled=not _has_key):
                _mime_map = {
                    "image/png": "image/png",
                    "image/jpeg": "image/jpeg",
                    "image/jpg": "image/jpeg",
                    "application/pdf": "application/pdf",
                }
                _mime = _mime_map.get(_img_file.type, "image/png")
                with st.spinner("Reading with Claude Vision — a few seconds…"):
                    _ex = _extract_ch_report_image(_img_file.read(), _mime)

                if "error" in _ex:
                    st.error(f"Extraction failed: {_ex['error']}")
                else:
                    # Populate opening inventory from location totals
                    if "union_springs_total" in _ex:
                        cp["us_bean_opening"] = float(_ex["union_springs_total"])
                    if "york_total" in _ex:
                        cp["york_bean_opening"] = float(_ex["york_total"])

                    # Populate monthly to-arrive from extracted breakdown
                    def _apply_by_month(key: str, loc_dict: dict) -> None:
                        arr = list(cp[key])
                        for _mkey, _bu in (loc_dict or {}).items():
                            _idx = _REPORT_MONTH_IDX.get(_mkey.lower().replace("-", "_"))
                            if _idx is not None and _idx < _POS_N:
                                arr[_idx] = arr[_idx] + float(_bu or 0)
                        cp[key] = arr

                    # Reset arrive arrays before applying
                    cp["us_bean_arrive"]   = [0.0] * _POS_N
                    cp["york_bean_arrive"] = [0.0] * _POS_N
                    _apply_by_month("us_bean_arrive",   _ex.get("us_by_month", {}))
                    _apply_by_month("york_bean_arrive", _ex.get("york_by_month", {}))

                    st.session_state["crush_pos"] = cp
                    _save_crush_pos(cp)

                    # Show what was extracted
                    _preview = {
                        "US Opening (bu)":   f"{cp['us_bean_opening']:,.0f}",
                        "York Opening (bu)": f"{cp['york_bean_opening']:,.0f}",
                        "US To-Arrive total": f"{sum(cp['us_bean_arrive']):,.0f}",
                        "York To-Arrive total": f"{sum(cp['york_bean_arrive']):,.0f}",
                    }
                    st.success("Extracted and applied! " +
                               " · ".join(f"{k}: {v}" for k, v in _preview.items()))
                    st.rerun()

        with _imp_tab_txt:
            st.caption(
                "Paste the full text of your CH Basis report (e.g. from Du Mond / GrainBridge). "
                "The parser extracts Company Owned Inventory, Basis Position, Unpriced Purchases, "
                "and per-location totals for Union Springs and York."
            )
            report_text = st.text_area(
                "Paste report text here", height=180, key="ch_report_paste",
                placeholder="TOTAL STOCK          217,211   ...\nUNION SPRINGS (100)  444,147...\nYORK ELEVATOR (300)  106,284..."
            )
            if st.button("Auto-fill from report", key="ch_parse_btn"):
                parsed = _parse_ch_report(report_text)
                if parsed:
                    if "union_springs_total" in parsed:
                        cp["us_bean_opening"] = float(parsed["union_springs_total"])
                    elif "company_owned_inventory" in parsed:
                        cp["us_bean_opening"] = float(parsed["company_owned_inventory"])
                    elif "total_stock" in parsed:
                        cp["us_bean_opening"] = float(parsed["total_stock"])
                    if "york_total" in parsed:
                        cp["york_bean_opening"] = float(parsed["york_total"])
                    st.session_state["crush_pos"] = cp
                    _save_crush_pos(cp)
                    detail = " · ".join(
                        f"{k.replace('_', ' ').title()}: {v:,}" for k, v in parsed.items()
                    )
                    st.success(f"Parsed: {detail}")
                else:
                    st.warning("No recognizable data found — paste the full report text.")

    with st.expander("📥  Restore saved position from file"):
        st.caption(
            "Upload a `crush_position.json` file you previously downloaded to restore all "
            "inventory, supply, and crush schedule data."
        )
        _upload = st.file_uploader(
            "Upload crush_position.json", type="json", key="crush_pos_upload",
        )
        if _upload is not None:
            try:
                _uploaded_data = json.loads(_upload.read().decode())
                # Run through the same migration + fill logic
                if "crush_bu" in _uploaded_data and "us_crush_bu" not in _uploaded_data:
                    _uploaded_data["us_crush_bu"]   = _uploaded_data.pop("crush_bu")
                    _uploaded_data["york_crush_bu"] = [0.0] * _POS_N
                _uploaded_data.pop("crush_bu", None)
                for _k, _v in _CRUSH_POS_DEFAULTS.items():
                    if _k not in _uploaded_data:
                        _uploaded_data[_k] = list(_v) if isinstance(_v, list) else _v
                    elif isinstance(_v, list) and len(_uploaded_data[_k]) != _POS_N:
                        _uploaded_data[_k] = (_uploaded_data[_k] + [0.0] * _POS_N)[:_POS_N]
                st.session_state["crush_pos"] = _uploaded_data
                _save_crush_pos(_uploaded_data)
                st.success("Position restored from file — data populated above.")
                st.rerun()
            except Exception as _e:
                st.error(f"Could not parse file: {_e}")

    # ── Soybean Supply by Location ────────────────────────────────────────────

    st.markdown('<div class="sh">Soybean Supply by Location</div>', unsafe_allow_html=True)
    st.caption(
        "Enter to-arrive quantities (priced basis contracts and unpriced/DP grain) "
        "by delivery month for each location. Opening inventory = current on-hand bushels."
    )

    _loc1, _loc2 = st.columns(2)

    # ── Union Springs ──────────────────────────────────────────────────────────
    with _loc1:
        st.markdown(
            f'<div style="font-weight:700;font-size:1rem;color:{JPSI_DARK};'
            f'border-left:3px solid {JPSI_BLUE};padding-left:8px;margin-bottom:8px;">'
            f'📍 Union Springs</div>',
            unsafe_allow_html=True,
        )
        us_opening_val = st.number_input(
            "Opening inventory (bu)", value=float(cp.get("us_bean_opening", 0.0)),
            step=1000.0, format="%.0f", key="us_opening_in",
        )
        cp["us_bean_opening"] = us_opening_val
        us_cap_val = st.number_input(
            "Storage capacity (bu)", value=float(cp.get("us_capacity", 1_000_000.0)),
            step=10_000.0, format="%.0f", key="us_capacity_in",
        )
        cp["us_capacity"] = us_cap_val

        us_df = pd.DataFrame({
            "Month":          pm_labels[:n_show],
            "To-Arrive (bu)": [float(v) for v in cp["us_bean_arrive"][:n_show]],
        })
        us_edited = st.data_editor(
            us_df,
            column_config={
                "Month":          st.column_config.TextColumn("Month", disabled=True, width="small"),
                "To-Arrive (bu)": st.column_config.NumberColumn("To-Arrive (bu)", format="%,.0f", min_value=0.0),
            },
            hide_index=True, use_container_width=True, key="us_supply_editor", num_rows="fixed",
        )
        cp["us_bean_arrive"] = us_edited["To-Arrive (bu)"].tolist() + cp["us_bean_arrive"][n_show:]

    # ── York ──────────────────────────────────────────────────────────────────
    with _loc2:
        st.markdown(
            f'<div style="font-weight:700;font-size:1rem;color:{JPSI_DARK};'
            f'border-left:3px solid #f6821f;padding-left:8px;margin-bottom:8px;">'
            f'📍 York</div>',
            unsafe_allow_html=True,
        )
        york_opening_val = st.number_input(
            "Opening inventory (bu)", value=float(cp.get("york_bean_opening", 0.0)),
            step=1000.0, format="%.0f", key="york_opening_in",
        )
        cp["york_bean_opening"] = york_opening_val
        york_cap_val = st.number_input(
            "Storage capacity (bu)", value=float(cp.get("york_capacity", 350_000.0)),
            step=10_000.0, format="%.0f", key="york_capacity_in",
        )
        cp["york_capacity"] = york_cap_val

        york_df = pd.DataFrame({
            "Month":          pm_labels[:n_show],
            "To-Arrive (bu)": [float(v) for v in cp["york_bean_arrive"][:n_show]],
        })
        york_edited = st.data_editor(
            york_df,
            column_config={
                "Month":          st.column_config.TextColumn("Month", disabled=True, width="small"),
                "To-Arrive (bu)": st.column_config.NumberColumn("To-Arrive (bu)", format="%,.0f", min_value=0.0),
            },
            hide_index=True, use_container_width=True, key="york_supply_editor", num_rows="fixed",
        )
        cp["york_bean_arrive"] = york_edited["To-Arrive (bu)"].tolist() + cp["york_bean_arrive"][n_show:]

    # ── Crush schedule (per plant) ────────────────────────────────────────────

    st.markdown('<div class="sh">Crush Schedule</div>', unsafe_allow_html=True)

    crush_df = pd.DataFrame({
        "Month":                  pm_labels[:n_show],
        "Union Springs (bu)":     [float(v) for v in cp["us_crush_bu"][:n_show]],
        "York (bu)":              [float(v) for v in cp["york_crush_bu"][:n_show]],
    })
    crush_edited = st.data_editor(
        crush_df,
        column_config={
            "Month":              st.column_config.TextColumn("Month", disabled=True, width="small"),
            "Union Springs (bu)": st.column_config.NumberColumn(
                "Union Springs (bu)", format="%,.0f", min_value=0.0),
            "York (bu)":          st.column_config.NumberColumn(
                "York (bu)", format="%,.0f", min_value=0.0),
        },
        hide_index=True, use_container_width=True, key="crush_schedule_editor", num_rows="fixed",
    )
    cp["us_crush_bu"]   = crush_edited["Union Springs (bu)"].tolist() + cp["us_crush_bu"][n_show:]
    cp["york_crush_bu"] = crush_edited["York (bu)"].tolist() + cp["york_crush_bu"][n_show:]
    comb_crush_bu = [cp["us_crush_bu"][_i] + cp["york_crush_bu"][_i] for _i in range(_POS_N)]

    # ── Per-location supply cascades ─────────────────────────────────────────

    comb_arrive = [cp["us_bean_arrive"][_i] + cp["york_bean_arrive"][_i] for _i in range(_POS_N)]

    us_open_arr    = [0.0] * _POS_N
    us_closing_arr = [0.0] * _POS_N
    york_open_arr    = [0.0] * _POS_N
    york_closing_arr = [0.0] * _POS_N
    us_open_arr[0]   = us_opening_val
    york_open_arr[0] = york_opening_val
    for _i in range(_POS_N):
        if _i > 0:
            us_open_arr[_i]   = us_closing_arr[_i - 1]
            york_open_arr[_i] = york_closing_arr[_i - 1]
        us_closing_arr[_i]   = us_open_arr[_i]   + cp["us_bean_arrive"][_i]   - cp["us_crush_bu"][_i]
        york_closing_arr[_i] = york_open_arr[_i]  + cp["york_bean_arrive"][_i] - cp["york_crush_bu"][_i]

    b_opening = [us_open_arr[_i] + york_open_arr[_i] for _i in range(_POS_N)]
    b_avail   = [b_opening[_i] + comb_arrive[_i]     for _i in range(_POS_N)]
    b_closing = [us_closing_arr[_i] + york_closing_arr[_i] for _i in range(_POS_N)]

    hdr = "".join(f"<th style='text-align:right;'>{lbl}</th>" for lbl in pm_labels[:n_show])

    def _sc(v):
        return POS_GREEN if float(v) >= 0 else NEG_RED

    def _cap_color_us(v):
        cap = cp.get("us_capacity", 1_000_000.0)
        fv = float(v)
        if fv < 0:
            return NEG_RED
        pct = fv / cap if cap > 0 else 0.0
        return NEG_RED if pct > 1.0 else ("#e8833a" if pct > 0.8 else POS_GREEN)

    def _cap_color_york(v):
        cap = cp.get("york_capacity", 350_000.0)
        fv = float(v)
        if fv < 0:
            return NEG_RED
        pct = fv / cap if cap > 0 else 0.0
        return NEG_RED if pct > 1.0 else ("#e8833a" if pct > 0.8 else POS_GREEN)

    _bal_rows = [
        ("Opening (bu)",           b_opening[:n_show],             JPSI_DARK, "", None),
        ("  ↳ Union Springs",      us_open_arr[:n_show],           "#6b7280", "font-size:0.85rem;", None),
        ("  ↳ York",               york_open_arr[:n_show],         "#6b7280", "font-size:0.85rem;", None),
        ("To-Arrive (bu)",         comb_arrive[:n_show],           JPSI_DARK, "border-top:1px solid #f0f0f0;", None),
        ("  ↳ Union Springs",      cp["us_bean_arrive"][:n_show],  "#6b7280", "font-size:0.85rem;", None),
        ("  ↳ York",               cp["york_bean_arrive"][:n_show], "#6b7280", "font-size:0.85rem;", None),
        ("Total Available (bu)",   b_avail[:n_show],               JPSI_DARK, "border-top:2px solid #e5e7eb;font-weight:700;", None),
        ("Crush Planned (bu)",     comb_crush_bu[:n_show],         NEG_RED,   "", None),
        ("  ↳ Union Springs",      cp["us_crush_bu"][:n_show],     "#6b7280", "font-size:0.85rem;", None),
        ("  ↳ York",               cp["york_crush_bu"][:n_show],   "#6b7280", "font-size:0.85rem;", None),
        ("Closing Inventory (bu)", b_closing[:n_show],             JPSI_DARK, "font-weight:700;background:#f6f8fa;", _sc),
        ("  ↳ Union Springs",      us_closing_arr[:n_show],        "#6b7280", "font-size:0.85rem;background:#f6f8fa;", _cap_color_us),
        ("  ↳ York",               york_closing_arr[:n_show],      "#6b7280", "font-size:0.85rem;background:#f6f8fa;", _cap_color_york),
    ]

    rows_html = ""
    for _label, _vals, _color, _extra, _cfn in _bal_rows:
        if _cfn is not None:
            _tds = "".join(
                f'<td style="text-align:right;{_extra};color:{_cfn(v)};">{float(v):,.0f}</td>'
                for v in _vals
            )
        else:
            _tds = "".join(
                f'<td style="text-align:right;{_extra}">{float(v):,.0f}</td>'
                for v in _vals
            )
        _fw = "400" if _label.startswith("  ↳") else "600"
        rows_html += (
            f'<tr><td style="{_extra}font-weight:{_fw};color:{_color};">{_label}</td>{_tds}</tr>'
        )

    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr><th>Combined Supply Balance</th>{hdr}</tr></thead>
  <tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)

    # ── Crush Production ──────────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Crush Production (Extruder Yields)</div>', unsafe_allow_html=True)

    meal_produced = [bu * ext_meal_tons for bu in comb_crush_bu]  # tons
    oil_produced  = [bu * ext_oil_lbs   for bu in comb_crush_bu]  # lbs

    prod_rows_html = ""
    for _lbl, _bu, _mt, _ol in zip(pm_labels[:n_show], comb_crush_bu[:n_show],
                                    meal_produced[:n_show], oil_produced[:n_show]):
        prod_rows_html += (
            f"<tr>"
            f"<td style='font-weight:600;'>{_lbl}</td>"
            f"<td style='text-align:right;'>{_bu:,.0f}</td>"
            f"<td style='text-align:right;color:{JPSI_BLUE};'>{_mt:,.1f}</td>"
            f"<td style='text-align:right;color:#f6821f;'>{_ol:,.0f}</td>"
            f"<td style='text-align:right;color:#6b7280;'>{_ol/7.524:,.0f}</td>"
            f"</tr>"
        )
    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Month</th>
    <th style="text-align:right;">Crush (bu)</th>
    <th style="text-align:right;color:{JPSI_BLUE};">Meal Produced (tons)</th>
    <th style="text-align:right;color:#f6821f;">Oil Produced (lbs)</th>
    <th style="text-align:right;color:#6b7280;">Oil (approx. gal)</th>
  </tr></thead>
  <tbody>{prod_rows_html}</tbody>
</table></div>
<p style="font-size:0.81rem;color:#6b7280;margin-top:-6px;">
Meal = crush × {ext_meal_tons:.5f} t/bu · Oil = crush × {ext_oil_lbs:.4f} lbs/bu ·
Gal ≈ lbs ÷ 7.524 (crude soy oil density)
</p>""", unsafe_allow_html=True)

    # ── Meal Balance ──────────────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Meal Balance (short tons)</div>', unsafe_allow_html=True)

    _mc1, _mc2 = st.columns([2, 4])
    meal_opening_val = _mc1.number_input(
        "Meal opening inventory (tons)",
        value=float(cp.get("meal_opening_tons", 0.0)),
        min_value=0.0, step=10.0, format="%.1f",
        key="meal_opening_in",
        help="Current on-hand meal inventory in short tons.",
    )
    cp["meal_opening_tons"] = meal_opening_val
    _mc2.caption(
        f"Enter current meal in storage. Produced = crush × {ext_meal_tons:.5f} t/bu. "
        f"Closing = Opening + Produced − Sales."
    )

    meal_sales_df = pd.DataFrame({
        "Month":              pm_labels[:n_show],
        "Meal Sales (tons)":  [float(v) for v in cp["meal_sales_tons"][:n_show]],
    })
    meal_edited = st.data_editor(
        meal_sales_df,
        column_config={
            "Month": st.column_config.TextColumn("Month", disabled=True, width="small"),
            "Meal Sales (tons)": st.column_config.NumberColumn(
                "Sales (tons)", format="%,.1f", min_value=0.0),
        },
        hide_index=True, use_container_width=True,
        key="meal_sales_editor", num_rows="fixed",
    )
    cp["meal_sales_tons"] = meal_edited["Meal Sales (tons)"].tolist() + cp["meal_sales_tons"][n_show:]

    # Cascade meal balance
    m_opening = [0.0] * _POS_N
    m_closing = [0.0] * _POS_N
    m_opening[0] = meal_opening_val
    for _i in range(_POS_N):
        if _i > 0:
            m_opening[_i] = m_closing[_i - 1]
        m_closing[_i] = m_opening[_i] + meal_produced[_i] - cp["meal_sales_tons"][_i]

    # Meal balance table
    meal_bal_rows = [
        ("Opening (tons)",    m_opening[:n_show],            JPSI_DARK, "", None),
        ("Produced (tons)",   meal_produced[:n_show],         JPSI_BLUE, "", None),
        ("Sales (tons)",      cp["meal_sales_tons"][:n_show], NEG_RED,   "", None),
        ("Closing (tons)",    m_closing[:n_show],             JPSI_DARK, "font-weight:700;background:#f6f8fa;", _sc),
    ]
    meal_rows_html = ""
    for lbl, vals, color, extra_style, cfn in meal_bal_rows:
        if cfn is not None:
            tds = "".join(
                f'<td style="text-align:right;{extra_style};color:{cfn(v)};">{v:,.1f}</td>'
                for v in vals
            )
        else:
            tds = "".join(
                f'<td style="text-align:right;{extra_style}">{v:,.1f}</td>'
                for v in vals
            )
        meal_rows_html += (
            f'<tr><td style="{extra_style}font-weight:600;color:{color};">{lbl}</td>{tds}</tr>'
        )
    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr><th>Meal Balance</th>{hdr}</tr></thead>
  <tbody>{meal_rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)

    # ── Oil Balance ───────────────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Oil Balance (lbs)</div>', unsafe_allow_html=True)

    _oc1, _oc2 = st.columns([2, 4])
    oil_opening_val = _oc1.number_input(
        "Oil opening inventory (lbs)",
        value=float(cp.get("oil_opening_lbs", 0.0)),
        step=1000.0, format="%.0f",
        key="oil_opening_in",
        help="Current on-hand oil inventory in lbs.",
    )
    cp["oil_opening_lbs"] = oil_opening_val
    _oc2.caption(
        f"Enter current oil in storage. Produced = crush × {ext_oil_lbs:.4f} lbs/bu. "
        f"Closing = Opening + Produced − Sales."
    )

    oil_sales_df = pd.DataFrame({
        "Month":            pm_labels[:n_show],
        "Oil Sales (lbs)":  [float(v) for v in cp["oil_sales_lbs"][:n_show]],
    })
    oil_edited = st.data_editor(
        oil_sales_df,
        column_config={
            "Month": st.column_config.TextColumn("Month", disabled=True, width="small"),
            "Oil Sales (lbs)": st.column_config.NumberColumn(
                "Sales (lbs)", format="%,.0f", min_value=0.0),
        },
        hide_index=True, use_container_width=True,
        key="oil_sales_editor", num_rows="fixed",
    )
    cp["oil_sales_lbs"] = oil_edited["Oil Sales (lbs)"].tolist() + cp["oil_sales_lbs"][n_show:]

    # Cascade oil balance
    o_opening = [0.0] * _POS_N
    o_closing = [0.0] * _POS_N
    o_opening[0] = oil_opening_val
    for _i in range(_POS_N):
        if _i > 0:
            o_opening[_i] = o_closing[_i - 1]
        o_closing[_i] = o_opening[_i] + oil_produced[_i] - cp["oil_sales_lbs"][_i]

    # Oil balance table
    oil_bal_rows = [
        ("Opening (lbs)",   o_opening[:n_show],            JPSI_DARK, "", None),
        ("Produced (lbs)",  oil_produced[:n_show],          "#f6821f",  "", None),
        ("Sales (lbs)",     cp["oil_sales_lbs"][:n_show],   NEG_RED,   "", None),
        ("Closing (lbs)",   o_closing[:n_show],             JPSI_DARK, "font-weight:700;background:#f6f8fa;", _sc),
    ]
    oil_rows_html = ""
    for lbl, vals, color, extra_style, cfn in oil_bal_rows:
        if cfn is not None:
            tds = "".join(
                f'<td style="text-align:right;{extra_style};color:{cfn(v)};">{v:,.0f}</td>'
                for v in vals
            )
        else:
            tds = "".join(
                f'<td style="text-align:right;{extra_style}">{v:,.0f}</td>'
                for v in vals
            )
        oil_rows_html += (
            f'<tr><td style="{extra_style}font-weight:600;color:{color};">{lbl}</td>{tds}</tr>'
        )
    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr><th>Oil Balance</th>{hdr}</tr></thead>
  <tbody>{oil_rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)

    # ── Monthly P&L at live prices ────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Projected Crush P&L — at Live Front-Month Prices</div>',
                unsafe_allow_html=True)
    st.caption(
        f"Meal @ ${front.ZM:.2f}/ton · Oil @ {front.ZL:.3f} ¢/lb · Beans @ {front.ZS:.2f} ¢/bu. "
        f"Uses current live prices for all months — does not account for forward-curve differences."
    )

    pnl_rows_html = ""
    pnl_total_gpm = 0.0
    any_crush = any(v > 0 for v in comb_crush_bu[:n_show])
    if any_crush:
        for _i in range(n_show):
            _bu = comb_crush_bu[_i]
            if _bu <= 0:
                continue
            _meal_rev  = meal_produced[_i] * front.ZM
            _oil_rev   = oil_produced[_i] * (front.ZL / 100.0)
            _bean_cost = _bu * (front.ZS / 100.0)
            _gpm_total = _meal_rev + _oil_rev - _bean_cost
            _gpm_bu    = _gpm_total / _bu
            pnl_total_gpm += _gpm_total
            _gpm_color = POS_GREEN if _gpm_total >= 0 else NEG_RED
            pnl_rows_html += (
                f"<tr>"
                f"<td style='font-weight:600;'>{pm_labels[_i]}</td>"
                f"<td style='text-align:right;'>{_bu:,.0f}</td>"
                f"<td style='text-align:right;color:#5aa469;'>${_meal_rev:,.0f}</td>"
                f"<td style='text-align:right;color:#f6821f;'>${_oil_rev:,.0f}</td>"
                f"<td style='text-align:right;color:{NEG_RED};'>−${_bean_cost:,.0f}</td>"
                f"<td style='text-align:right;color:{_gpm_color};font-weight:700;'>${_gpm_total:,.0f}</td>"
                f"<td style='text-align:right;'>{_gpm_bu*100:.2f} ¢</td>"
                f"</tr>"
            )
        # Totals row
        total_bu   = sum(v for v in comb_crush_bu[:n_show] if v > 0)
        total_meal = sum(mt * front.ZM for mt, bu in zip(meal_produced[:n_show], comb_crush_bu[:n_show]) if bu > 0)
        total_oil  = sum(ol * (front.ZL / 100.0) for ol, bu in zip(oil_produced[:n_show], comb_crush_bu[:n_show]) if bu > 0)
        total_bean = sum(bu * (front.ZS / 100.0) for bu in comb_crush_bu[:n_show] if bu > 0)
        total_color = POS_GREEN if pnl_total_gpm >= 0 else NEG_RED
        pnl_rows_html += (
            f"<tr style='font-weight:700;border-top:2px solid #e5e7eb;background:#f6f8fa;'>"
            f"<td>Total</td>"
            f"<td style='text-align:right;'>{total_bu:,.0f}</td>"
            f"<td style='text-align:right;color:#5aa469;'>${total_meal:,.0f}</td>"
            f"<td style='text-align:right;color:#f6821f;'>${total_oil:,.0f}</td>"
            f"<td style='text-align:right;color:{NEG_RED};'>−${total_bean:,.0f}</td>"
            f"<td style='text-align:right;color:{total_color};'>${pnl_total_gpm:,.0f}</td>"
            f"<td style='text-align:right;'>{pnl_total_gpm/total_bu*100:.2f} ¢</td>"
            f"</tr>"
        )
        st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Month</th>
    <th style="text-align:right;">Crush (bu)</th>
    <th style="text-align:right;color:#5aa469;">Meal Revenue</th>
    <th style="text-align:right;color:#f6821f;">Oil Revenue</th>
    <th style="text-align:right;color:{NEG_RED};">Bean Cost</th>
    <th style="text-align:right;color:{JPSI_BLUE};">Gross Margin</th>
    <th style="text-align:right;">Margin/bu</th>
  </tr></thead>
  <tbody>{pnl_rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)
    else:
        st.info("Enter crush quantities in the Supply Schedule table above to see projected P&L.")

    # ── Hedge Positions ───────────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Hedge Positions — ZS / ZM / ZL</div>', unsafe_allow_html=True)
    st.caption(
        "Shared with the Hedge Converter tab — changes here are reflected there and vice versa. "
        "Sign convention: negative Qty = short, positive = long. "
        "Long crush = short ZS (beans) + long ZM (meal) + long ZL (oil)."
    )

    if "positions" not in st.session_state:
        st.session_state["positions"] = _load_positions()

    _hp_df = pd.DataFrame(st.session_state["positions"])
    _hp_edited = st.data_editor(
        _hp_df,
        column_config={
            "Leg":   st.column_config.SelectboxColumn(
                "Leg", options=["ZS", "ZM", "ZL"], required=True, width="small"),
            "Month": st.column_config.TextColumn(
                "Month", help='e.g. "Nov \'26"', width="small"),
            "Qty":   st.column_config.NumberColumn(
                "Qty (contracts)", help="+long / −short", step=1, format="%d"),
            "Entry": st.column_config.NumberColumn(
                "Entry price", min_value=0.0, format="%.3f"),
            "Note":  st.column_config.TextColumn("Note", width="large"),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="pos_tab_hedge_editor",
    )

    _hb1, _hb2 = st.columns([1, 5])
    if _hb1.button("💾  Save hedge", type="primary", key="save_hedge_from_pos_btn"):
        _saved_hp = _hp_edited.to_dict("records")
        st.session_state["positions"] = _saved_hp
        _save_positions(_saved_hp)
        st.success("Hedge positions saved.")
    if _hb2.button("↺  Reset to defaults", key="reset_hedge_from_pos_btn"):
        st.session_state["positions"] = [row.copy() for row in _POS_DEFAULTS]
        _save_positions(st.session_state["positions"])
        st.rerun()

    # Coverage analysis
    st.markdown('<div class="sh" style="margin-top:1rem;">Hedge Coverage vs. Crush Plan</div>',
                unsafe_allow_html=True)

    _hp_rows = _hp_edited.dropna(subset=["Leg", "Qty"]).to_dict("records")

    def _leg_contracts(leg: str) -> float:
        return sum(r.get("Qty", 0) or 0 for r in _hp_rows if r.get("Leg") == leg)

    _zs_c = _leg_contracts("ZS")
    _zm_c = _leg_contracts("ZM")
    _zl_c = _leg_contracts("ZL")

    _zs_bu  = _zs_c * ZS_BU
    _zm_ton = _zm_c * ZM_TONS
    _zl_lbs = _zl_c * ZL_LBS

    _total_crush_bu  = sum(comb_crush_bu[:n_show])
    _total_meal_tons = sum(meal_produced[:n_show])
    _total_oil_lbs   = sum(oil_produced[:n_show])

    def _cov_pct(hedged_abs: float, exposure: float) -> str:
        if exposure == 0:
            return "<span style='color:#6b7280;'>—</span>"
        pct = hedged_abs / exposure * 100
        color = POS_GREEN if pct >= 80 else ("#e8833a" if pct >= 40 else NEG_RED)
        return f'<span style="color:{color};font-weight:700;">{pct:.1f}%</span>'

    def _contracts_fmt(qty: float, unit: int, unit_label: str) -> str:
        if qty == 0:
            return "<span style='color:#6b7280;'>—</span>"
        total = qty * unit
        sign = "+" if qty > 0 else ""
        return (f"{sign}{qty:,.0f} contracts "
                f"({sign}{total:,.0f} {unit_label})")

    _cov_data = [
        ("ZS — Beans",
         f"{_total_crush_bu:,.0f} bu planned crush",
         _contracts_fmt(_zs_c, ZS_BU, "bu"),
         _cov_pct(abs(_zs_bu), _total_crush_bu)),
        ("ZM — Meal",
         f"{_total_meal_tons:,.1f} tons planned production",
         _contracts_fmt(_zm_c, ZM_TONS, "tons"),
         _cov_pct(abs(_zm_ton), _total_meal_tons)),
        ("ZL — Oil",
         f"{_total_oil_lbs:,.0f} lbs planned production",
         _contracts_fmt(_zl_c, ZL_LBS, "lbs"),
         _cov_pct(abs(_zl_lbs), _total_oil_lbs)),
    ]

    _cov_html = "".join(
        f"<tr>"
        f"<td style='font-weight:600;'>{leg}</td>"
        f"<td style='text-align:right;color:#6b7280;'>{exp}</td>"
        f"<td style='text-align:right;'>{hdg}</td>"
        f"<td style='text-align:right;'>{cov}</td>"
        f"</tr>"
        for leg, exp, hdg, cov in _cov_data
    )

    st.markdown(f"""
<div class="sheet-wrap"><table>
  <thead><tr>
    <th>Leg</th>
    <th style="text-align:right;color:#6b7280;">Exposure (crush plan)</th>
    <th style="text-align:right;">Hedged</th>
    <th style="text-align:right;">Coverage</th>
  </tr></thead>
  <tbody>{_cov_html}</tbody>
</table></div>
<p style="font-size:0.81rem;color:#6b7280;margin-top:-6px;">
Coverage = |contracts × contract size| ÷ planned crush/production quantity.
<span style="color:{POS_GREEN};font-weight:600;">Green ≥ 80%</span> ·
<span style="color:#e8833a;font-weight:600;">Orange ≥ 40%</span> ·
<span style="color:{NEG_RED};font-weight:600;">Red &lt; 40%</span>.
Enter crush quantities above to see exposure totals.
</p>""", unsafe_allow_html=True)

    # ── Save / Download ───────────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    _sv1, _sv2, _sv3 = st.columns([2, 2, 4])
    if _sv1.button("💾  Save position", type="primary", key="save_crush_pos_btn"):
        st.session_state["crush_pos"] = cp
        _save_crush_pos(cp)
        st.success("Position saved.")
    _sv2.download_button(
        label="⬇️  Download position",
        data=json.dumps(cp, indent=2).encode(),
        file_name="crush_position.json",
        mime="application/json",
        key="download_crush_pos_btn",
    )
    _sv3.caption(
        "Download saves a backup to your computer. "
        "On Streamlit Cloud the file resets on each reboot — "
        "use **Restore saved position from file** at the top to reload it."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — DRAW TERRITORY (New York)
# ─────────────────────────────────────────────────────────────────────────────

with tab_territory:
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{JPSI_DARK} 0%,#2a4a2a 100%);
     border-radius:8px;padding:16px 20px;margin-bottom:20px;">
  <div style="font-size:1.2rem;font-weight:700;color:#fff;">🗺️ Draw Territory — New York State</div>
  <div style="font-size:0.875rem;color:#ccc;margin-top:4px;">
    USDA NASS crop production + livestock inventory. Sourced from the JSA shared NASS cache.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Top filter bar ────────────────────────────────────────────────────────
    _fc1, _fc2, _fc3, _fc4 = st.columns([2, 3, 1, 0.5])
    with _fc1:
        _dt_comm = st.selectbox("Commodity", list(_DT_ALL_COMMS.keys()), key="dt_comm")
    with _fc2:
        _dt_yr_range = st.slider(
            "Historical range", 1980, 2024, (1990, 2024), step=1, key="dt_yr_range",
        )
    with _fc3:
        _dt_map_yr = st.selectbox("Map year", list(range(2024, 2013, -1)), key="dt_map_yr")
    with _fc4:
        st.markdown("<div style='margin-top:26px'></div>", unsafe_allow_html=True)
        if st.button("🔄", help="Clear cached data and reload", use_container_width=True, key="dt_refresh"):
            st.cache_data.clear()
            st.rerun()

    _dt_is_crop   = _dt_comm in _DT_CROPS
    _dt_comm_cfg  = _DT_ALL_COMMS[_dt_comm]
    _dt_mlist     = list(_dt_comm_cfg.keys())

    # ── Color palette ─────────────────────────────────────────────────────────
    _DT_COLORS = {
        "Corn":                "#d97706",
        "Soybeans":            JPSI_BLUE,
        "Dairy Cows":          "#5ba5af",
        "All Cattle & Calves": JPSI_DARK,
        "Hogs & Pigs":         "#8b5cf6",
    }
    _DT_FILLS = {
        "Corn":                "rgba(217,119,6,0.12)",
        "Soybeans":            "rgba(6,147,227,0.12)",
        "Dairy Cows":          "rgba(91,165,175,0.12)",
        "All Cattle & Calves": "rgba(50,55,60,0.12)",
        "Hogs & Pigs":         "rgba(139,92,246,0.12)",
    }
    _trend_color = _DT_COLORS.get(_dt_comm, JPSI_BLUE)
    _fill_color  = _DT_FILLS.get(_dt_comm, "rgba(6,147,227,0.12)")

    # ── Load state-level timeseries for all metrics ───────────────────────────
    _dt_series: dict = {}
    for _mlbl, _mp in _dt_comm_cfg.items():
        _dt_series[_mlbl] = _dt_fetch_ny(
            commodity_desc=_mp["commodity_desc"],
            statisticcat_desc=_mp["statisticcat_desc"],
            unit_desc=_mp["unit_desc"],
            reference_period_desc=_mp.get("reference_period_desc", "YEAR"),
            util_practice_desc=_mp.get("util_practice_desc", ""),
            class_desc=_mp.get("class_desc", ""),
        )

    # ── Value formatter (matches domestic-production-dashboard) ───────────────
    def _dt_fmt(v: float, metric: str) -> str:
        if "Yield" in metric or "/Ac" in metric:
            return f"{v:.1f}"
        if v >= 1_000_000_000:
            return f"{v/1_000_000_000:.2f}B"
        if v >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:,.0f}"

    # ── KPI cards ─────────────────────────────────────────────────────────────
    _kpi_cols = st.columns(len(_dt_mlist), gap="small")
    for _ki, _mlbl in enumerate(_dt_mlist):
        _mdf = _dt_series[_mlbl]
        _lbl_short = _mlbl.split("(")[0].strip()
        with _kpi_cols[_ki]:
            if _mdf.empty:
                st.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{_lbl_short}</div>"
                    f"<div class='kpi-value' style='font-size:16px;color:#9ca3af;'>No data</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                _lat_yr = int(_mdf["year"].max())
                _lat_v  = float(_mdf.loc[_mdf["year"] == _lat_yr, "value"].iloc[0])
                _prev_r = _mdf[_mdf["year"] == _lat_yr - 1]
                _prev_v = float(_prev_r["value"].iloc[0]) if not _prev_r.empty else None
                _dhtml  = ""
                if _prev_v and _prev_v != 0:
                    _pct  = (_lat_v - _prev_v) / _prev_v * 100
                    _cls  = "kpi-pos" if _pct >= 0 else "kpi-neg"
                    _sign = "▲" if _pct >= 0 else "▼"
                    _dhtml = f"<div class='kpi-delta {_cls}'>{_sign} {abs(_pct):.1f}% vs {_lat_yr - 1}</div>"
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-label'>{_lbl_short}</div>"
                    f"<div class='kpi-value'>{_dt_fmt(_lat_v, _mlbl)}</div>"
                    f"{_dhtml}"
                    f"<div class='kpi-year'>{_lat_yr} — New York State</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend charts ──────────────────────────────────────────────────────────
    def _dt_trend_fig(df: pd.DataFrame, title: str, yr_range: tuple) -> go.Figure:
        fig = go.Figure()
        _mdf = df[df["year"].between(yr_range[0], yr_range[1])].copy() if not df.empty else df
        _is_yld = "Yield" in title or "/Ac" in title
        _tick   = ".1f" if _is_yld else ",.0f"
        if _mdf.empty:
            fig.add_annotation(text="No data — check NASS cache",
                               showarrow=False, font=dict(size=12, color="#9ca3af"))
        else:
            fig.add_trace(go.Scatter(
                x=_mdf["year"].astype(int), y=_mdf["value"],
                mode="lines+markers",
                line=dict(color=_trend_color, width=2.5),
                marker=dict(size=4, color=_trend_color),
                fill="tozeroy", fillcolor=_fill_color,
                name=title,
                hovertemplate=f"<b>%{{x}}</b><br>{title}: %{{y:{_tick}}}<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color=JPSI_DARK), x=0),
            height=320, paper_bgcolor="white", plot_bgcolor="white",
            showlegend=False, margin=dict(l=55, r=14, t=44, b=36),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#6b7280"),
                       linecolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", tickformat=_tick,
                       tickfont=dict(size=9, color="#6b7280"), linecolor="#e5e7eb"),
            hovermode="x unified",
        )
        return fig

    if _dt_is_crop:
        _tr1, _tr2 = st.columns(2)
        _tr3, _tr4 = st.columns(2)
        _trcols = [_tr1, _tr2, _tr3, _tr4]
        for _ti, _mlbl in enumerate(_dt_mlist):
            with _trcols[_ti]:
                st.plotly_chart(
                    _dt_trend_fig(_dt_series[_mlbl], f"{_dt_comm} — {_mlbl} (NY)", _dt_yr_range),
                    use_container_width=True,
                )
    else:
        _mlbl0 = _dt_mlist[0]
        st.plotly_chart(
            _dt_trend_fig(_dt_series[_mlbl0], f"{_dt_comm} — {_mlbl0} (NY)", _dt_yr_range),
            use_container_width=True,
        )

    # ── County / ASD section ──────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sh">County / ASD Map</div>', unsafe_allow_html=True)

    _ctrl_a, _ctrl_b = st.columns([5, 2])
    with _ctrl_a:
        _prod_dflt = next((m for m in _dt_mlist if "Production" in m), _dt_mlist[0])
        _dt_cmetric = st.radio(
            "County metric", _dt_mlist,
            index=_dt_mlist.index(_prod_dflt),
            horizontal=True, label_visibility="collapsed",
            key=f"dt_cmetric_{_dt_comm}",
        )
    with _ctrl_b:
        _dt_cyr = st.selectbox("Year", list(range(2024, 2013, -1)), key="dt_cyr",
                               index=list(range(2024, 2013, -1)).index(_dt_map_yr))

    with st.spinner("Loading county data from NASS cache..."):
        _dt_cdf = _dt_load_county_ny(_dt_comm, _dt_cmetric, _dt_cyr)

    if _dt_cdf.empty:
        st.info(
            f"County-level data for **{_dt_comm} — {_dt_cmetric} ({_dt_cyr})** is not yet "
            "in the NASS cache. State-level trend data is shown above.",
            icon="ℹ️",
        )
    else:
        # ASD view toggle (only if data has ASD fields)
        _has_asd = "asd_desc" in _dt_cdf.columns and _dt_cdf["asd_desc"].notna().any()
        _asd_view = False
        if _has_asd:
            _view_mode = st.radio(
                "View", ["County", "ASD Districts"],
                horizontal=True, label_visibility="collapsed",
                key="dt_view_mode",
            )
            _asd_view = _view_mode == "ASD Districts"

        # Summary KPI cards
        _is_yld_c = "Yield" in _dt_cmetric
        _summ_val = _dt_cdf["Value"].mean() if _is_yld_c else _dt_cdf["Value"].sum()
        _n_ctys   = _dt_cdf["fips"].nunique()
        _unit_lbl = ("Bu/Ac" if _is_yld_c
                     else ("Ac" if "Acres" in _dt_cmetric
                           else ("Head" if "Head" in _dt_cmetric else "Bu")))
        _val_str  = f"{_summ_val:,.1f}" if _is_yld_c else f"{_summ_val:,.0f}"
        _kc1, _kc2, _ = st.columns([1, 1, 4])
        _card_css = ("background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;"
                     "padding:12px 16px;")
        _lbl_css  = "font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;"
        _val_css  = "font-size:22px;font-weight:700;color:#111827;margin-top:4px;"
        _unit_css = "font-size:13px;color:#6b7280;"
        with _kc1:
            st.markdown(
                f"<div style='{_card_css}'><div style='{_lbl_css}'>{'Avg Yield' if _is_yld_c else 'Total'}</div>"
                f"<div style='{_val_css}'>{_val_str} <span style='{_unit_css}'>{_unit_lbl}</span></div></div>",
                unsafe_allow_html=True,
            )
        with _kc2:
            st.markdown(
                f"<div style='{_card_css}'><div style='{_lbl_css}'>Counties Reporting</div>"
                f"<div style='{_val_css}'>{_n_ctys:,}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        # Map
        try:
            with st.spinner("Rendering map..."):
                _dt_geo = _dt_load_geojson()
                if _asd_view:
                    _map_fig = _dt_asd_fig(_dt_cdf, _dt_geo, _dt_cmetric)
                else:
                    _map_fig = _dt_county_fig(_dt_cdf, _dt_geo, _dt_cmetric)
            st.plotly_chart(_map_fig, use_container_width=True)
        except Exception as _me:
            st.warning(f"Map render failed: {_me}")

        # Data table
        with st.expander("Data table", expanded=False):
            if _asd_view and _has_asd:
                _agg_m = "mean" if _is_yld_c else "sum"
                _tbl_c = (
                    _dt_cdf.groupby(
                        [c for c in ["asd_code", "asd_desc"] if c in _dt_cdf.columns],
                        as_index=False,
                    )["Value"].agg(_agg_m)
                    .rename(columns={"asd_desc": "District", "asd_code": "Code"})
                )
                _tbl_c = _tbl_c.sort_values("Code") if "Code" in _tbl_c.columns else _tbl_c
            else:
                _tbl_c = _dt_cdf[["County", "Value"]].copy()
                _tbl_c.columns = ["County", _dt_cmetric]
                _tbl_c = _tbl_c.sort_values(_dt_cmetric, ascending=False)
            st.dataframe(_tbl_c.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Livestock overview (shown when a crop is selected) ────────────────────
    if _dt_is_crop:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sh">Livestock Inventory — New York</div>', unsafe_allow_html=True)
        _lv_data = {
            "Dairy Cows":          _dt_fetch_ny("CATTLE", "INVENTORY", "HEAD",
                                                reference_period_desc="FIRST OF JAN",
                                                class_desc="COWS", util_practice_desc="MILK"),
            "All Cattle & Calves": _dt_fetch_ny("CATTLE", "INVENTORY", "HEAD",
                                                reference_period_desc="FIRST OF JAN",
                                                class_desc="ALL CLASSES"),
            "Hogs & Pigs":         _dt_fetch_ny("HOGS",   "INVENTORY", "HEAD",
                                                reference_period_desc="FIRST OF DEC"),
        }
        _lv_c1, _lv_c2, _lv_c3 = st.columns(3)
        for (_lname, _ldf), _lcol in zip(_lv_data.items(), [_lv_c1, _lv_c2, _lv_c3]):
            _lcolor = _DT_COLORS[_lname]
            _lfill  = _DT_FILLS[_lname]
            _lref   = "Dec 1" if _lname == "Hogs & Pigs" else "Jan 1"
            with _lcol:
                _lfig = go.Figure()
                _ldf2 = _ldf[_ldf["year"].between(_dt_yr_range[0], _dt_yr_range[1])] if not _ldf.empty else _ldf
                if _ldf2.empty:
                    _lfig.add_annotation(text="No data", showarrow=False, font=dict(size=11, color="#9ca3af"))
                else:
                    _lfig.add_trace(go.Scatter(
                        x=_ldf2["year"].astype(int), y=_ldf2["value"],
                        mode="lines+markers",
                        line=dict(color=_lcolor, width=2.5),
                        marker=dict(size=4, color=_lcolor),
                        fill="tozeroy", fillcolor=_lfill,
                        hovertemplate=f"<b>%{{x}}</b><br>Inventory: %{{y:,.0f}}<extra></extra>",
                    ))
                _lfig.update_layout(
                    title=dict(text=f"{_lname} — {_lref} (NY)", font=dict(size=12, color=JPSI_DARK), x=0),
                    height=280, paper_bgcolor="white", plot_bgcolor="white",
                    showlegend=False, margin=dict(l=55, r=14, t=40, b=32),
                    xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#6b7280"), linecolor="#e5e7eb"),
                    yaxis=dict(gridcolor="#e5e7eb", tickformat=",.0f",
                               tickfont=dict(size=9, color="#6b7280"), linecolor="#e5e7eb"),
                    hovermode="x unified",
                )
                st.plotly_chart(_lfig, use_container_width=True)
        st.caption("Livestock: USDA NASS annual surveys. Jan 1 cattle/dairy, Dec 1 hogs.")
