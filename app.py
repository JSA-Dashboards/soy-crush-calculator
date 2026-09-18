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

import json
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

tab_live, tab_seasonal, tab_curve, tab_hedge = st.tabs([
    "📊  Live Summary",
    "📅  Seasonal History",
    "🌊  Crush Curve",
    "🔧  Hedge Converter",
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
