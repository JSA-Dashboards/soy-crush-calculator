"""Draw Territory tab: the ASD dashboard's NASS Production drill-down for New York, plus NY livestock.

Data: live JSA.NASS_CACHE (one NY-filtered query per crop), falling back to the bundled
ny_nass_snapshot.json. Logic lives in ny_asd_data.py, figures in ny_asd_figs.py.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import draw_ontario
import draw_silage
import dt_nass
import ny_asd_data as nd
import ny_asd_figs as nf
import ny_freight as nfr
import ny_ho
import ny_livestock as nl

_CTX: dict = {}
_LIVE_TTL = 6 * 3600
_STAT_LABEL = {"planted": "Planted Acres (000 ac)", "harvested": "Harvested Acres (000 ac)",
               "pct_harvested": "% Harvested", "yield": "Yield (bu/ac)", "production": "Production (M bu)"}
_STATS = ["planted", "harvested", "pct_harvested", "yield", "production"]
_LS_COLORS = {"Dairy Cows": ("#5ba5af", "rgba(91,165,175,0.12)"),
              "All Cattle & Calves": ("#32373c", "rgba(50,55,60,0.12)"),
              "Hogs & Pigs": ("#8b5cf6", "rgba(139,92,246,0.12)")}


# ── data access ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _geo() -> dict:
    return nf.load_geo()


@st.cache_resource(show_spinner=False)
def _snapshot() -> dict:
    return nd.load_snapshot()


def _fips_map() -> dict:
    return _snapshot().get("fips_map") or _geo()["fips_map"]


@st.cache_resource(ttl=_LIVE_TTL, show_spinner=False)
def _live_crop(crop: str) -> nd.NYCrop:
    """One NY-filtered query. Raises on any failure, so a failure is never cached."""
    conn = _CTX["connect"]()
    try:
        cur = conn.cursor()

        def run(sql):
            cur.execute(sql)
            return cur.fetchall()
        rows = nd.fetch_crop_rows(crop, run)
    finally:
        conn.close()
    if rows.empty:
        raise RuntimeError("the NASS cache returned no New York rows")
    return nd.NYCrop(crop, rows, _fips_map())


@st.cache_resource(show_spinner=False)
def _snap_crop(crop: str) -> nd.NYCrop:
    return nd.NYCrop(crop, nd.snapshot_to_frame(_snapshot()["crops"][crop]), _fips_map())


def _get_crop(crop: str):
    """(NYCrop, error). error is None when the live cache served the data."""
    try:
        return _live_crop(crop), None
    except Exception as e:
        return _snap_crop(crop), f"{type(e).__name__}: {str(e)[:160]}"


@st.cache_data(ttl=_LIVE_TTL, show_spinner=False)
def _live_livestock() -> dict:
    keys = {dt_nass.cache_key(dt_nass.state_params(mp)): comm for comm, mp in dt_nass.LIVESTOCK.items()}
    inlist = ",".join("'%s'" % k for k in keys)
    conn = _CTX["connect"]()
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT c.cache_key, f.value:year::string, f.value:"Value"::string, f.value:reference_period_desc::string,
                   f.value:domain_desc::string, f.value:source_desc::string
            FROM JSA.NASS_CACHE.nass_cache c, LATERAL FLATTEN(input => c.data:data) f
            WHERE c.cache_key IN ({inlist}) AND f.value:state_alpha::string = '{nd.STATE}'""")
        rows = cur.fetchall()
    finally:
        conn.close()
    recs: dict = {c: [] for c in dt_nass.LIVESTOCK}
    for k, y, v, rp, dom, src in rows:
        recs[keys[k]].append({"state_alpha": nd.STATE, "year": y, "Value": v, "reference_period_desc": rp,
                              "domain_desc": dom, "source_desc": src})
    out = {c: dt_nass.state_series(r, dt_nass.LIVESTOCK[c]) for c, r in recs.items()}
    if all(df.empty for df in out.values()):
        raise RuntimeError("the NASS cache returned no New York livestock rows")
    return out


def _livestock():
    try:
        return _live_livestock(), None
    except Exception as e:
        snap = _snapshot()
        return {c: dt_nass.snap_series(snap, c) for c in dt_nass.LIVESTOCK}, f"{type(e).__name__}: {str(e)[:160]}"


def _missing_secrets() -> str:
    sec = _CTX["secret"]
    miss = [n for n in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER") if not sec(n)]
    if not (sec("SNOWFLAKE_PRIVATE_KEY") or sec("SNOWFLAKE_PRIVATE_KEY_PATH") or sec("SNOWFLAKE_PASSWORD")):
        miss.append("SNOWFLAKE_PRIVATE_KEY")
    return ", ".join(miss)


def _source_note(err):
    gen = _snapshot().get("generated", "n/a")
    if err:
        miss = _missing_secrets()
        st.warning(f"**Live NASS cache not connected** — showing the bundled NY snapshot ({gen}).  \n`{err}`"
                   + (f"  \nMissing Snowflake settings in this app's secrets: **{miss}**" if miss else ""),
                   icon="⚠️")
    else:
        st.caption("Source: JSA NASS cache (live).")


# ── formatting ────────────────────────────────────────────────────────────────
def _fmt_val(metric: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if metric == "Yield (bu/ac)":
        return f"{v:.1f} bu/ac"
    if metric == "% Harvested":
        return f"{v:.1f}%"
    if "Acres" in metric:
        if abs(v) >= 500e3:
            return f"{v / 1e6:,.2f}M ac"
        return f"{v / 1e3:,.1f}K ac" if abs(v) >= 1e3 else f"{v:,.0f} ac"
    div, unit = nf.auto_bu(abs(v))
    return f"{v / div:,.2f} {unit}" if div > 1 else f"{v:,.0f} bu"


def _hist_fmt(stat: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return {"production": f"{v / 1e6:,.1f}", "planted": f"{v / 1e3:,.0f}", "harvested": f"{v / 1e3:,.0f}",
            "pct_harvested": f"{v:.1f}%", "yield": f"{v:.1f}"}.get(stat, f"{v:,.0f}")


def _card(label: str, value: str, sub: str = "") -> str:
    return (f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>"
            f"{sub}</div>")


# ── tab ───────────────────────────────────────────────────────────────────────
@st.fragment
def render(sf_connect, sf_secret, dark: str, blue: str, zs_cents: float | None = None) -> None:
    _CTX.update(connect=sf_connect, secret=sf_secret, dark=dark, blue=blue, zs=zs_cents)
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{dark} 0%,#2a4a2a 100%);border-radius:8px;padding:16px 20px;margin-bottom:14px;">
  <div style="font-size:1.2rem;font-weight:700;color:#fff;">🗺️ Draw Territory — New York State</div>
  <div style="font-size:0.875rem;color:#ccc;margin-top:4px;">
    USDA NASS production by ASD district and county, with estimates for counties NASS withholds, plus livestock inventory.
  </div>
</div>""", unsafe_allow_html=True)
    t_nass, t_ho, t_sil, t_frt, t_on, t_live = st.tabs(["NASS Production", "High-Oleic & Competition", "Silage",
                                                        "Freight & Spreads", "Ontario", "Livestock"])
    with t_nass:
        _nass_tab()
    with t_ho:
        _ho_tab()
    with t_sil:
        _silage_tab()
    with t_frt:
        _freight_tab()
    with t_on:
        _ontario_tab()
    with t_live:
        _livestock_tab()


def _nass_tab() -> None:
    geo = _geo()
    fm = _fips_map()
    codes = {v[0]: v[1] for v in fm.values()}

    c1, c2, c3, c4 = st.columns([1, 0.75, 1.8, 0.55])
    crop = c1.selectbox("Crop", nd.CROPS, key="ny_crop")
    year = c2.selectbox("Year", nd.YEARS, index=1, key="ny_year")
    c3.selectbox("State Drill-Down", ["NY  —  New York"], key="ny_state", disabled=True)
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True, key="ny_refresh"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with st.expander("Adjustments", expanded=False):
        ho = st.number_input(
            "New York high-oleic soybeans (% of NY soybean acres)", min_value=0.0, max_value=100.0, value=0.0,
            step=1.0, format="%.0f", key="ny_ho",
            help="USDA does not separate high-oleic soybeans. Enter your estimate of the share of New York soybean "
                 "acres that is high-oleic; it is carved out of NY soybean production, planted and harvested acres "
                 "(yield unchanged). 0 = no adjustment. The same share is applied to every year.")
        st.caption("High-oleic beans in New York mostly go to on-farm / dairy-feed use rather than crush or export, "
                   "so the carve-out leaves commodity soybean supply.")

    st.markdown("<p style='color:#5f7267;font-size:0.78rem;margin:4px 0 2px 0;'>Metric</p>", unsafe_allow_html=True)
    metric = st.radio("Metric", nd.METRICS, horizontal=True, key="ny_metric", label_visibility="collapsed")
    st.markdown("<p style='color:#5f7267;font-size:0.78rem;margin:4px 0 2px 0;'>View</p>", unsafe_allow_html=True)
    change = st.radio("Change View", nd.CHANGE_OPTS, horizontal=True, key="ny_change", label_visibility="collapsed")
    comp = None
    if change == "vs Selected Year":
        cc, _ = st.columns([0.55, 2.6])
        comp = cc.selectbox("Compare to Year", [y for y in nd.YEARS if y != year], key="ny_comp_year")

    adj = nd.Adj(ho=(ho / 100.0 if crop == "Soybeans" else 0.0), ts=(metric == "Total Supply (bu)"))
    stat = nd.METRIC_TO_STAT[metric]
    ratio = metric in nd.RATIO_METRICS
    cobj, err = _get_crop(crop)
    _source_note(err)
    if crop == "Soybeans" and adj.ho > 0:
        st.caption(f"New York soybean production, planted and harvested acres exclude {adj.ho:.0%} high-oleic "
                   "(your estimate under Adjustments; yield unchanged).")
    if adj.ts:
        st.caption("Total supply = Sep 1 stocks + production for the same year. NASS publishes stocks by state only, "
                   "so counties and districts get New York's stocks in proportion to their share of its production.")
        if not cobj.stocks(year):
            st.info(f"NASS has not published New York {crop.lower()} grain stocks for {year}; showing production only.")

    is_fc = year == nd.FORECAST_YEAR and cobj.reported_counties(year) == 0
    fc_stats: dict = {}
    fc_totals: dict = {}
    fc_county: dict = {}
    if is_fc:
        fc_stats, fc_totals = _forecast_block(cobj, crop, year, adj)
        if fc_totals:                                      # each district's forecast split across its counties
            fc_county = cobj.county_forecast(float(fc_totals.get("yield") or 0.0), adj)

    state_val = (fc_totals.get(stat) if is_fc and fc_totals else cobj.state_view_value(metric, year, change, comp, adj))
    if state_val is None and not is_fc:
        if stat == "prevent_plant":
            st.warning(f"NASS does not publish {crop.lower()} prevent-plant acres for New York.")
        else:
            st.warning(f"NASS has no {year} {crop.lower()} {metric.lower()} for New York yet.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    bench = cobj.reported_counties(nd.BENCHMARK_YEAR)
    curn = cobj.reported_counties(year)
    pct_rep = curn / bench * 100 if bench else 0.0
    k1, k2, k3, k4 = st.columns(4)
    abs_val = fc_totals.get(stat) if is_fc and fc_totals else cobj.state_total(stat, year, adj)
    k1.metric(f"{year} {metric}", _fmt_val(metric, abs_val))
    if change == "Current Year":
        if is_fc:
            k2.metric("County Coverage", "Forecast", help=f"County data not yet published for {year}")
            k3.metric("Counties Reporting", "—")
        else:
            k2.metric("County Coverage", f"{pct_rep:.0f}%",
                      help=f"{curn:,} counties reporting vs {bench:,} in {nd.BENCHMARK_YEAR} benchmark")
            k3.metric("Counties Reporting", f"{curn:,}")
        k4.metric("ASD Districts", f"{len(codes)}")
    else:
        cv_all = cobj.county_view(metric, year, change, comp, adj, fc_county)
        vals = cv_all["Value"].dropna()
        if is_fc:
            k2.metric("County Coverage", "Forecast", help=f"County data not yet published for {year}")
        else:
            k2.metric("County Coverage", f"{pct_rep:.0f}%", help=f"{curn:,} of {bench:,} counties reporting")
        k3.metric("Counties Above Prior", f"{int((vals > 0).sum()):,} ▲")
        k4.metric("Counties Below Prior", f"{int((vals < 0).sum()):,} ▼")

    # ── map view ──────────────────────────────────────────────────────────────
    mv1, mv2, mv3 = st.columns([1, 1.7, 1.1])
    mv1.radio("Map View", ["ASD District", "County"], horizontal=True, key="ny_map_view")
    map_view = st.session_state["ny_map_view"]
    mk = ny_ho.markers_for(crop)                           # the markers follow the crop: beans and corn each get their own set
    with mv2:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        show_comp = st.checkbox(f"Show competition on the map ({mk['comp_label']})", key="ny_comp_nass")
    with mv3:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        show_eth = st.checkbox("Show ethanol plants" + ("" if mk["ethanol"] else " (corn only)"), value=True, key="ny_ethanol_nass",
                               disabled=not mk["ethanol"],
                               help="New York's corn-ethanol plant, Western New York Energy at Medina. Shown for corn only.")
    show_eth = show_eth and mk["ethanol"]
    comp_cats = mk["cats"] if show_comp and mk["cats"] else None

    if map_view == "ASD District":
        view_df, cur_df, est_d = cobj.district_view(metric, year, change, comp, adj, fc_stats if is_fc else None)
        n_est = 0
        comp_df = pd.DataFrame()
        if stat in nd.ESTIMABLE and not is_fc:
            comp_df = cobj.completed(stat, year, adj)
            n_est = int(comp_df["is_estimated"].sum()) if not comp_df.empty else 0
        fig = nf.district_fig(view_df, cur_df, geo, crop, year, metric, change, est_d, comp_cats, ethanol=show_eth)
        if fig is None:
            st.info("ASD district map not available for this selection.")
        else:
            fig.update_layout(dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key="ny_district_map",
                            config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
        if not cur_df.empty:
            t = cur_df.copy()
            t["ASD"] = t["District"].map(codes).fillna("")
            t = t.sort_values("ASD").reset_index(drop=True)
            tot = t["Value"].mean() if ratio else t["Value"].sum()
            t[metric] = t["Value"].apply(lambda v: _fmt_val(metric, v))
            t["% of State"] = (t["Value"] / tot * 100).round(1).astype(str) if not ratio and tot > 0 else "—"
            cols = ["District", "ASD", metric, "% of State"]
            if not comp_df.empty:
                e = (comp_df[comp_df["is_estimated"]].groupby("District")["is_estimated"].count()
                     .rename("Est. Counties").reset_index())
                t = t.merge(e, on="District", how="left")
                t["Est. Counties"] = t["Est. Counties"].fillna(0).astype(int).apply(lambda n: str(n) if n else "—")
                cols.append("Est. Counties")
            st.markdown(f"<p style='color:#5f7267;font-size:0.82rem;font-weight:600;margin:6px 0 4px 0;'>"
                        f"New York {year} {crop} — by ASD district</p>", unsafe_allow_html=True)
            tc, kc = st.columns([2.4, 1])
            with tc:
                st.dataframe(t[cols], use_container_width=True, hide_index=True)
            with kc:
                m1, m2 = st.columns(2)
                m1.metric("State Avg" if ratio else "State Total", _fmt_val(metric, tot))
                m2.metric("Districts", f"{len(t)}")
                if n_est > 0:
                    st.caption("Est = some counties not published by NASS. Values are estimated from each county's "
                               "historical share of state output, adjusted for current district performance, and "
                               "scaled to reconcile with the NASS state total.")
                if not ratio and not is_fc and abs_val and abs(tot / abs_val - 1) > 0.01:
                    st.caption(f"District total ({_fmt_val(metric, tot)}) differs from the NASS state figure "
                               f"({_fmt_val(metric, abs_val)}): NASS publishes its county and state estimates separately.")
    else:
        cdf = cobj.county_view(metric, year, change, comp, adj, fc_county)
        fig = nf.county_fig(cdf, geo, crop, year, metric, change, comp_cats, ethanol=show_eth, forecast=is_fc and bool(fc_county))
        if fig is None:
            st.info("County map not available for this selection (county estimates are published after harvest)."
                    if is_fc else "County map not available for this selection.")
        else:
            fig.update_layout(dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key="ny_county_map",
                            config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
            if is_fc and fc_county:
                st.caption(f"{year} forecast: NASS has no {year} county data until after harvest, so each ASD district's forecast "
                           "(from the yield above) is split across its counties by their average share of the district over the last "
                           "five years. Counties add up to the district forecasts and the state forecast; it is not county data.")
            n_cty_est = int(cdf["is_estimated"].sum())
            if n_cty_est > 0:
                st.caption(f"Est = {n_cty_est} counties not published by NASS. Production is estimated using each "
                           "county's historical share of state output, adjusted for current district performance, "
                           "and scaled to reconcile with the NASS state total.")

    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)
    _history_section(cobj, crop, year, metric, change, comp, adj, is_fc, fc_stats, fc_totals, fm, codes, fc_county)


def _forecast_block(cobj: nd.NYCrop, crop: str, year: int, adj: nd.Adj) -> tuple:
    key = f"ny_fc_yield_{crop}"
    src = cobj.default_yield_sources(adj)
    usda, ly = src["usda"], src["ly"]
    if key not in st.session_state:
        st.session_state[key] = float(usda or ly or 0.0)
    if usda:
        st.info(f"**{year} Forecast** — USDA's latest {crop.lower()} yield estimate for New York is **{usda:g} bu/ac**"
                + (f" (last year {ly:g})" if ly else "") + "; it is the default below. Change it to run your own "
                "estimate: ASD district production is allocated from historical district shares.")
    else:
        st.info(f"**{year} Forecast** — USDA has not yet published a {crop.lower()} yield for New York, so the "
                + (f"default below is last year's ({ly:g} bu/ac). " if ly else "") +
                "Enter your own estimate to project ASD district production.")
    f1, f2, _ = st.columns([1.2, 1.4, 2.4])
    y_in = f1.number_input(f"{crop} Yield Est. (bu/ac)", min_value=0.0, max_value=500.0, step=0.5, format="%.1f", key=key)
    with f2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if usda and abs(y_in - usda) > 0.05:
            st.button(f"Reset to USDA ({usda:g})", key=f"ny_fc_reset_{crop}",
                      on_click=lambda k=key, v=float(usda): st.session_state.__setitem__(k, v))
    if y_in > 0:
        return cobj.asd_forecast(y_in, adj)
    st.info("Enter a yield estimate above to generate the ASD district forecast.")
    return {}, {}


def _history_section(cobj, crop, year, metric, change, comp, adj, is_fc, fc_stats, fc_totals, fm, codes, fc_county=None) -> None:
    stat = nd.METRIC_TO_STAT[metric]
    ratio = metric in nd.RATIO_METRICS
    geo = _geo()
    state_name = "New York"
    h1, h2 = st.columns([1.4, 2.6])
    scope = h1.radio("View", ["State", "ASD District", "County"], horizontal=True, key="ny_scope")
    dist = county = None
    with h2:
        if scope == "ASD District":
            dist = st.selectbox("District", sorted(codes), key="ny_hist_district")
            label = f"{state_name} — {dist}"
        elif scope == "County":
            names = sorted(f["properties"]["NAME"] for f in geo["counties"]["features"])
            county = st.selectbox("County", names, key="ny_hist_county")
            label = f"{state_name} — {county} County"
        else:
            label = state_name

    def value_for(s: str, y: int):
        """Scope value for one stat/year (None when unavailable)."""
        if is_fc and y == nd.FORECAST_YEAR:
            if scope == "State":
                return fc_totals.get(s)
            if scope == "ASD District":
                return fc_stats.get(s, {}).get(dist)
            return None
        if scope == "State":
            return cobj.state_total(s, y, adj)
        if scope == "ASD District":
            vals, _ = cobj.district_series(s, y, adj)
            return vals.get(dist)
        nc = cobj.named_county(s, y, adj)
        r = nc[nc["County"].str.lower() == county.lower()] if not nc.empty else nc
        return float(r["Value"].iloc[0]) if not r.empty else None

    n_chart = st.radio("Chart history (years)", [5, 10], horizontal=True, key="ny_chart_yrs", index=1)
    yrs = sorted([y for y in nd.YEARS if y <= year][:n_chart])
    vals = [value_for(stat, y) for y in yrs]
    mx = max((v for v in vals if v), default=0)
    if metric in ("Production (bu)", "Total Supply (bu)"):
        unit, ylab = nf.auto_bu(mx)
    elif "Acres" in metric:
        unit, ylab = nf.auto_ac(mx)
    elif metric == "Yield (bu/ac)":
        unit, ylab = 1, "bu/ac"
    else:
        unit, ylab = 1, "%"
    st.plotly_chart(nf.history_bar({metric: [v / unit if v else 0 for v in vals]}, yrs,
                                   f"{label} — {crop} {metric}", ylab),
                    use_container_width=True, key="ny_hist_chart", config={"displayModeBar": False})
    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:#5f7267;font-size:0.82rem;font-weight:600;margin:0 0 6px 0;letter-spacing:0.04em;'>"
                f"📅 HISTORICAL SUMMARY — {label} | {year - 5}–{year}</p>", unsafe_allow_html=True)
    n_tbl = st.radio("Table history (years)", [5, 10], horizontal=True, key="ny_tbl_yrs", index=0)
    hyears = [y for y in range(year - (n_tbl - 1), year + 1) if y >= nd.YEARS[-1]]
    unit_lbl = {"Production (bu)": "M bu", "Total Supply (bu)": "M bu", "Planted Acres": "M ac",
                "Harvested Acres": "M ac", "Prevent Plant Acres": "M ac", "% Harvested": "%",
                "Yield (bu/ac)": "bu/ac"}[metric]
    div = 1 if ratio else 1e6
    title_tail = f"({unit_lbl}) | {min(hyears)}–{max(hyears)}"
    est_by_dist: dict = {}

    def county_rows(y: int) -> pd.DataFrame:
        if is_fc and y == nd.FORECAST_YEAR:
            return pd.DataFrame()
        if stat in nd.ESTIMABLE:
            return cobj.completed(stat, y, adj)
        return cobj.named_county(stat, y, adj).assign(is_estimated=False)

    fig = None
    dist_county: dict = {}
    dist_totals: dict = {}
    if scope == "State":
        rows = {state_name: {y: (fc_totals.get(stat) if is_fc and y == nd.FORECAST_YEAR else cobj.state_total(stat, y, adj))
                             for y in hyears}}
        fig = nf.heatmap_table(rows, hyears, f"{crop} — {metric} {title_tail}", unit_lbl, div, ratio, fmt=".1f",
                               first_col="State")
    elif scope == "ASD District":
        d_yr: dict = {}
        status: dict = {}
        for y in hyears:
            if is_fc and y == nd.FORECAST_YEAR:
                for d, v in fc_stats.get(stat, {}).items():
                    d_yr.setdefault(d, {})[y] = v
                    status.setdefault(d, {})[y] = True
                dist_totals[y] = fc_totals.get(stat)
                continue
            vals_y, est_y = cobj.district_series(stat, y, adj)
            for d, v in vals_y.items():
                d_yr.setdefault(d, {})[y] = v
                status.setdefault(d, {})[y] = est_y.get(d, False)
            cdf = county_rows(y)
            if not cdf.empty:
                dist_totals[y] = float(cdf["Value"].mean() if ratio else cdf["Value"].sum())
            for d, g in (cdf.groupby("District") if not cdf.empty else []):
                for r in g.itertuples():
                    dist_county.setdefault(d, {}).setdefault(r.County, {})[y] = float(r.Value)
                    est_by_dist.setdefault(d, {}).setdefault(r.County, {})[y] = bool(r.is_estimated)
        fig = nf.heatmap_table(d_yr, hyears, f"New York {crop} — {metric} by ASD District {title_tail}", unit_lbl, div,
                               ratio, total=dist_totals if not ratio else None, fmt=".1f", cell_status=status,
                               total_label="NY Total", first_col="District")
    else:
        c_yr: dict = {}
        for y in hyears:
            nc = cobj.named_county(stat, y, adj)
            for r in nc.itertuples():
                c_yr.setdefault(r.County, {})[y] = float(r.Value) if pd.notna(r.Value) else None
        fig = nf.heatmap_table(c_yr, hyears, f"New York {crop} — {metric} by County {title_tail}", unit_lbl, div, ratio,
                               fmt=".1f", total_label="NY Total", first_col="County")
    st.plotly_chart(fig, use_container_width=True, key="ny_heatmap", config={"displayModeBar": False})

    if scope == "ASD District" and dist_county:
        st.caption("▼ Expand any district below to view county detail  ·  Top 2 = green  ·  Bottom 2 = red per row  ·  "
                   "Olympic Avg drops single highest and lowest year.")
        d_yr_vals = {d: {y: v for y, v in m.items()} for d, m in d_yr.items()}
        for dname in sorted(dist_county):
            cd = dist_county[dname]
            with st.expander(f"▼  {dname}  —  {len(cd)} counties", expanded=False):
                est_map = est_by_dist.get(dname, {})
                cs = {cn: {y: est_map.get(cn, {}).get(y, False) for y in hyears} for cn in cd}
                top_cs = {y: any(est_map.get(cn, {}).get(y, False) for cn in cd) for y in hyears}
                cfig = nf.heatmap_table(
                    cd, hyears, f"{dname} — County Detail ({unit_lbl})", unit_lbl, div, ratio, fmt=".1f",
                    top_row=d_yr_vals.get(dname, {}), top_row_label=f"{dname} District Total", cell_status=cs,
                    top_row_cell_status=top_cs, total_label="NY Total", first_col="County")
                st.plotly_chart(cfig, use_container_width=True, key=f"ny_cty_tbl_{dname}",
                                config={"displayModeBar": False})
    else:
        st.caption("Top 2 = green  ·  Bottom 2 = red per row  ·  Olympic Avg drops single highest and lowest year.")

    # ── detail expander: the five stats by year, with the map's change view ───
    hist = {y: {s: (fc_totals.get(s) if is_fc and y == nd.FORECAST_YEAR and scope == "State" else value_for(s, y))
                for s in _STATS} for y in hyears}
    with st.expander(f"📋 {label} Detail — {metric} ({min(hyears)}–{max(hyears)})", expanded=False):
        delta: dict = {y: {} for y in hyears}
        if change == "vs Prior Year":
            for i, y in enumerate(hyears[1:], 1):
                for s in _STATS:
                    c, p = hist[y].get(s), hist[hyears[i - 1]].get(s)
                    delta[y][s] = (c - p) / abs(p) * 100 if c and p else None
        elif change == "vs 3-Yr Avg":
            for s in _STATS:
                base = [hist[y].get(s) for y in (year - 3, year - 2, year - 1) if hist.get(y, {}).get(s)]
                avg = sum(base) / len(base) if base else None
                for y in hyears:
                    c = hist[y].get(s)
                    delta[y][s] = (c - avg) / abs(avg) * 100 if c and avg else None
        elif change == "vs Selected Year" and comp in hist:
            for y in hyears:
                for s in _STATS:
                    c, b = hist[y].get(s), hist[comp].get(s)
                    delta[y][s] = (c - b) / abs(b) * 100 if c and b and y != comp else None
        rows = []
        for s in _STATS:
            row = {"": _STAT_LABEL[s]}
            for y in hyears:
                dv = delta[y].get(s)
                dtxt = f" ({'+' if dv >= 0 else ''}{dv:.1f}%)" if dv is not None and change != "Current Year" else ""
                row[str(y)] = f"{_hist_fmt(s, hist[y].get(s))}{dtxt}"
            rows.append(row)
        df = pd.DataFrame(rows).set_index("")
        est_years = {y for y in hyears if is_fc and y == nd.FORECAST_YEAR}
        if est_years:
            df = df.rename(columns={str(y): f"{y} (Est)" for y in est_years})
        st.dataframe(df, use_container_width=True)
        if est_years:
            st.caption("Est = forecast from your yield estimate and historical district shares.")

    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)

    # ── county ranking + table ────────────────────────────────────────────────
    cv = cobj.county_view(metric, year, change, comp, adj, fc_county)
    if not cv.empty:
        st.plotly_chart(nf.ranking_chart(cv, crop, year, metric, change,
                                         by_district=st.session_state.get("ny_map_view") == "ASD District"),
                        use_container_width=True, key="ny_ranking")
        with st.expander("County Data Table — New York", expanded=False):
            tbl = cv[["County", "Value", "District", "is_estimated"]].dropna(subset=["Value"]) \
                .sort_values("Value", ascending=False)
            col = f"% Change ({metric})" if change != "Current Year" else metric
            show = pd.DataFrame({"County": tbl["County"], "District": tbl["District"],
                                 col: [f"{v:+.1f}%" if change != "Current Year" else _fmt_val(metric, v)
                                       for v in tbl["Value"]],
                                 "Estimated": ["Est" if e else "" for e in tbl["is_estimated"]]})
            st.dataframe(show, use_container_width=True, hide_index=True)


# ── silage ────────────────────────────────────────────────────────────────────
def _silage_tab() -> None:
    cobj, err = _get_crop("Corn")
    _source_note(err)
    draw_silage.render(_geo(), cobj, _fips_map())


# ── high-oleic density + competition ──────────────────────────────────────────
def _ho_inputs(cobj, year: int) -> tuple:
    """(acres by fips, bushels by fips, is_forecast, yield) feeding the HO model. The forecast year has no NASS county data (county
    estimates come after harvest), so it uses the same county forecast as the NASS tab's county map: each district's forecast split
    across its counties by five-year shares. The yield is the one the NASS Production tab uses (USDA's latest unless changed there)."""
    if year == nd.FORECAST_YEAR and cobj.reported_counties(year) == 0:
        src = cobj.default_yield_sources()
        y_in = float(st.session_state.get("ny_fc_yield_Soybeans") or src["usda"] or src["ly"] or 0.0)
        fc = cobj.county_forecast(y_in)
        return fc["harvested"], fc["production"], True, y_in
    ac, pr = cobj.completed("harvested", year), cobj.completed("production", year)
    return dict(zip(ac["fips"], ac["Value"])), dict(zip(pr["fips"], pr["Value"])), False, 0.0


def _ho_tab() -> None:
    geo = _geo()
    cobj, err = _get_crop("Soybeans")
    _source_note(err)
    st.markdown(
        "<div class='callout'><strong>High-oleic (HO) soybeans are identity-preserved and contract-driven.</strong> "
        "Seed and grower contracts lock up 18-24 months ahead and processors place contracts the September before the "
        "crop, so HO acres are effectively unavailable for conventional origination where they concentrate. "
        "This map shows where that blocked zone sits, who else is buying, and what is left to originate.</div>",
        unsafe_allow_html=True)

    years = list(nd.YEARS)                                 # the forecast year first; the default is the latest actual year
    c1, c2, c3 = st.columns([1, 1.2, 3])
    year = c1.selectbox("Soybean year", years, index=1, key="ny_ho_year",
                        format_func=lambda y: f"{y} (forecast)" if y == nd.FORECAST_YEAR and cobj.reported_counties(y) == 0 else str(y),
                        help="The forecast year has no NASS county data yet, so NASS's state forecast is split across counties by "
                             "their five-year average shares.")
    share = c2.number_input(
        "NY high-oleic share of soy acres (%)", min_value=0.0, max_value=100.0, value=ny_ho.STATE_HO_SHARE * 100,
        step=1.0, format="%.0f", key="ny_ho_share",
        help="Your estimate is closer to 60% than the one-third NYCSGA cited in early 2025. The county map is scaled so "
             "the acre-weighted state share equals this number.")
    c3.radio("Map", nf.HO_VIEWS, horizontal=True, key="ny_ho_view")
    view = st.session_state["ny_ho_view"]

    with st.expander("Model settings", expanded=False):
        m1, m2, m3 = st.columns(3)
        radius = m1.slider("Roaster / mill reach (miles)", 10, 80, 35, 5, key="ny_ho_radius",
                           help="How quickly a roaster's or feed mill's pull fades with distance from a county.")
        dairy_w = m2.slider("Weight on dairy density", 0.0, 1.0, 0.3, 0.05, key="ny_ho_dairy",
                            help="Share of the score from milk-cow density (Jan 2023); the rest comes from roaster/mill proximity.")
        zone_w = m3.slider("Weight your zone calls", 0.0, 1.0, 0.5, 0.05, key="ny_ho_zone",
                           help="0 = pure proximity + dairy model. 1 = your blocked / conventional-pool calls scale each "
                                "county's score fully (blocked x1.0, partly blocked x0.85, pool x0.25, other x0.6).")
        m4, m5, _m6 = st.columns(3)
        feed_scale = m4.slider("Feed-mill weight (x the registry's)", 0.0, 1.5, 1.0, 0.1, key="ny_ho_feed",
                               help="Scales the pull of the feed mills (CNY Feeds, registry weight 3.0). 1 = as listed, 0 = leave "
                                    "the feed mills out of the model. The registry weights are assumptions, not data.")
        include_pending = m5.checkbox("Count planned and unconfirmed sites", value=False, key="ny_ho_pending",
                                      help="Off: a site that is planned or unconfirmed (the CNY Feeds Avon mill, opening summer "
                                           "2027, and York Milling) adds no pull. Turn on to see the map once they run.")
        st.caption("This is a transparent proxy, not observed data: county-level HO trait data does not exist publicly. "
                   "Seed companies track variety placement by county and may share aggregates under NDA; that would replace this.")

    acres_by, bu_by, is_fc, y_fc = _ho_inputs(cobj, year)
    if not acres_by:
        st.warning(f"No county soybean data for {year}.")
        return
    ylab = f"{year} forecast" if is_fc else str(year)
    if is_fc:
        st.info(f"**{year} forecast.** NASS has no {year} county data (county estimates come after harvest), so New York's state "
                f"forecast ({sum(acres_by.values()):,.0f} harvested acres at {y_fc:g} bu/ac) is split across counties by each county's "
                "average share of the last five years. The county high-oleic shares are the model's, so this says nothing about where "
                "high-oleic beans were actually planted. The yield is USDA's latest unless you change it on the NASS Production tab "
                "(Soybeans, 2026).")
    tbl = ny_ho.ho_table(geo, acres_by, bu_by, share / 100.0, float(radius), float(dairy_w), float(zone_w), include_pending,
                         float(feed_scale))
    base = tbl[tbl["Soy acres"] > 0]
    tot_ac, ho_ac = base["Soy acres"].sum(), base["HO acres"].sum()
    conv_bu = base["Conv bu"].sum()
    open_bu = base[base["Zone"].isin(["open", "other"])]["Conv bu"].sum()
    blocked_ac = base[base["Zone"] == "blocked"]["Soy acres"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{ylab} NY soybean acres", f"{tot_ac / 1e3:,.0f}K")
    k2.metric("Est. high-oleic acres", f"{ho_ac / 1e3:,.0f}K", f"{ho_ac / tot_ac:.0%} of acres", delta_color="off", delta_arrow="off")
    k3.metric("Conventional pool", f"{conv_bu / 1e6:,.2f}M bu", f"{tot_ac - ho_ac:,.0f} acres", delta_color="off", delta_arrow="off")
    k4.metric("Outside blocked zone", f"{open_bu / 1e6:,.2f}M bu",
              f"{open_bu / conv_bu:.0%} of the pool" if conv_bu else None, delta_color="off", delta_arrow="off",
              help="Conventional bushels in your conventional-pool counties and in counties you haven't assessed "
                   "(excludes the blocked ring and the partly blocked Wyoming / Steuben).")

    o1, o2 = st.columns([2.2, 2.8])
    show_zones = o1.checkbox("Outline your blocked zone and conventional pool", value=True, key="ny_ho_zones")
    ho_cats = ny_ho.ho_marker_cats()                       # crushers, roasters, elevators and the feed mills the model uses; no corn markers
    cats = o2.multiselect("Show competition", ho_cats, default=ho_cats,
                          format_func=lambda k: ny_ho.CATEGORIES[k]["label"], key="ny_ho_cats")
    fig = nf.ho_fig(tbl, geo, view, ylab, share / 100.0, cats, show_zones)
    fig.update_layout(dragmode=False)
    st.plotly_chart(fig, use_container_width=True, key="ny_ho_map",
                    config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
    _, off = nf.competition_points(geo, cats)
    st.caption("Hover any marker for details; click a legend entry to hide or show that group. Stars are the Du Mond Grain "
               "plants." + (f" {off} competitor location(s) fall off the map frame (Perdue's Pennsylvania plants, to the south)."
                            if off else "") + " Markers placed at a town or county centre are flagged in the hover.")

    st.button(f"Use {share:.0f}% as the high-oleic adjustment on the NASS Production tab", key="ny_ho_apply",
              on_click=lambda v=float(share): st.session_state.__setitem__("ny_ho", v),
              help="Carves this share out of New York soybean production, planted and harvested acres there.")

    st.markdown("<hr style='border-color:#d7e2dc;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Competition and demand nodes</div>', unsafe_allow_html=True)
    prec_txt = {"address": "street address", "town": "town centre", "county": "county centre", "none": "not placed on the map"}
    shown = 0
    for key, meta in ny_ho.CATEGORIES.items():
        if key not in cats:
            continue
        items = [c for c in ny_ho.COMPETITORS if c["cat"] == key]
        if not items:
            continue
        lines = []
        for c in items:
            where = c["town"] + (f", {c['county']} Co." if c["county"] and "as listed" not in c["county"] else
                                 (f" ({c['county']})" if c["county"] else ""))
            tag = {"planned": " *(planned)*", "unconfirmed": " *(unconfirmed)*"}.get(c.get("status", ""), "")
            eff = ny_ho.node_weight(c, include_pending, feed_scale)
            wt = (f" · model weight {eff:g}" if eff else
                  f" · weight {c['pull']:g}, not counted while {c['status']}" if c["pull"] and c.get("status") in ny_ho.PENDING else "")
            lines.append(f"- **{c['name']}**{tag} — {where} *({prec_txt[c['prec']]}{wt})*  \n  {c['note']}")
        st.markdown(f"<span style='color:{meta['color']};font-weight:700;'>● {meta['label']}</span>", unsafe_allow_html=True)
        st.markdown("\n".join(lines))
        shown += len(items)
    if not shown:
        st.caption("No competition groups selected.")
    st.caption("Model weight = the assumed pull (0-3) of that node as high-oleic demand in the density model; no weight = "
               "not used. Out-of-state and export nodes are shown for context only. The map and this list show crushers, "
               "roasters, elevators and feed mills. "
               "Planned and unconfirmed sites add no weight until they open (switch in Model settings).")

    with st.expander("County detail (model inputs and results)", expanded=False):
        det = tbl[tbl["Soy acres"] > 0].sort_values("Conv bu", ascending=False)
        st.dataframe(pd.DataFrame({
            "County": det["County"], "Your zone": det["Zone"].map(ny_ho.ZONE_LABEL),
            "Soy acres": det["Soy acres"].round(0), "Dairy cows (Jan 2023)": det["Dairy cows"],
            "Roaster pull": det["Roaster pull"].round(2), "Est. HO share": (det["HO share"] * 100).round(0),
            "Est. HO acres": det["HO acres"].round(0), "Conventional acres": det["Conv acres"].round(0),
            "Conventional bu": det["Conv bu"].round(0)}), use_container_width=True, hide_index=True,
            column_config={"Est. HO share": st.column_config.NumberColumn(format="%d%%")})
        st.caption("Counties not in your dairy list (e.g. Oswego, Otsego, Yates, Warren) are scored as zero dairy. Acres and "
                   "bushels are NASS harvested acres and production, with NASS-withheld counties estimated as on the "
                   "NASS Production tab.")

# ── freight lanes and spreads ─────────────────────────────────────────────────
_NOTES_PRICE = 10.48            # the soybean price behind the notes' $2.62 surtax (25% of $10.48)
_FR_KEYS = {"rate": "ny_fr_rate", "bu_load": "ny_fr_bu", "min_load": "ny_fr_min", "xb_mult": "ny_fr_xb",
            "border": "ny_fr_border"}


def _fr_apply_preset() -> None:
    a = nfr.PRESETS[st.session_state["ny_fr_preset"]]
    st.session_state.update({k: a[f] for f, k in _FR_KEYS.items()})


def _cd(v) -> str:
    """Cents per bushel, shown as dollars from $1 up."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.0f}¢" if abs(v) < 100 else f"{'-' if v < 0 else ''}${abs(v) / 100:.2f}"


def _soy_bushels() -> tuple:
    """({fips: soybean bu}, {fips: conventional bu}, year label): NASS soybeans with the High-Oleic tab's settings. The label reads
    "2026 forecast" when the HO tab is on the forecast year (counties split from NASS's state forecast)."""
    cobj, _err = _get_crop("Soybeans")
    ss = st.session_state
    years = [y for y in nd.YEARS if y != nd.FORECAST_YEAR]
    year = ss.get("ny_ho_year", years[0])
    acres_by, bu_by, is_fc, _y = _ho_inputs(cobj, year)
    ylab = f"{year} forecast" if is_fc else str(year)
    if not acres_by:
        return {}, {}, ylab
    tbl = ny_ho.ho_table(_geo(), acres_by, bu_by,
                         ss.get("ny_ho_share", ny_ho.STATE_HO_SHARE * 100) / 100.0, float(ss.get("ny_ho_radius", 35)),
                         float(ss.get("ny_ho_dairy", 0.3)), float(ss.get("ny_ho_zone", 0.5)),
                         bool(ss.get("ny_ho_pending", False)), float(ss.get("ny_ho_feed", 1.0)))
    return (dict(zip(tbl["fips"], tbl["Soy bu"].fillna(0.0))), dict(zip(tbl["fips"], tbl["Conv bu"].fillna(0.0))), ylab)


def _freight_tab() -> None:
    geo = _geo()
    lanes = nfr.load()
    names = {d["key"]: d for d in nfr.dests()}
    zs = _CTX.get("zs")
    live_price = round(zs / 100.0, 2) if zs else _NOTES_PRICE
    ss = st.session_state
    ss.setdefault("ny_fr_preset", list(nfr.PRESETS)[0])
    if "ny_fr_rate" not in ss:
        _fr_apply_preset()
    ss.setdefault("ny_fr_price", live_price)

    st.markdown(
        "<div class='callout'><strong>Where does each county's soybean go, and what does it cost to get it there?</strong> "
        "Road miles from each county to the Du Mond plants, Bunge Hamilton (Canada), DeLong's Oswego export terminal and "
        "Perdue's Pennsylvania plants, turned into a delivered cost (truck + border, plus Canada's surtax if you switch one on: "
        "US soybeans are not on the surtax list that took effect September 8, 2026) and, once you enter "
        "what each buyer bids, a net-back to the farm gate and the territory each buyer can draw.</div>",
        unsafe_allow_html=True)

    a1, a2, a3 = st.columns([1.7, 1, 1.3])
    a1.radio("Freight assumptions", list(nfr.PRESETS), horizontal=True, key="ny_fr_preset", on_change=_fr_apply_preset,
             help="Benchmark trucking uses current truck rates (roughly $4-5 per loaded mile). 'Your notes' uses the "
                  "0.02 cents per bushel-mile behind the 3 cents for 152 miles in the research notes, with no border costs.")
    tariff = a2.selectbox("Canada surtax on US soybeans (what-if)", nfr.TARIFF_CHOICES, index=0, key="ny_fr_tariff",
                          format_func=lambda v: f"{v:.0%}", help="Soybeans are NOT on the United States Surtax Order (2026) "
                                                                "schedules (15%, 25% and 50% lists, 335 tariff items, none in "
                                                                "chapter 12), so the default is 0%. Checked 2026-10-27 against the "
                                                                "order's text. Pick 15, 25 or 50% to see what a surtax would do.")
    a3.number_input("Soybean price for a surtax what-if ($/bu)", min_value=0.0, max_value=40.0, step=0.05, format="%.2f",
                    key="ny_fr_price", help="Customs value = this futures price plus Hamilton's bid; only matters if you switch a "
                                            f"surtax on. Defaults to the live front-month ZS (${live_price:.2f}).")
    with st.expander("Freight and border assumptions", expanded=False):
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.number_input("Truck rate ($/loaded mile)", 0.0, 20.0, step=0.05, format="%.2f", key="ny_fr_rate")
        f2.number_input("Bushels per load", 300.0, 1200.0, step=10.0, format="%.0f", key="ny_fr_bu",
                        help="About 50,000 lb net payload / 60 lb per bushel = 833.")
        f3.number_input("Minimum charge ($/load)", 0.0, 1000.0, step=10.0, format="%.0f", key="ny_fr_min",
                        help="Short hauls are priced at a minimum, not by the mile.")
        f4.number_input("Cross-border rate multiplier", 1.0, 3.0, step=0.05, format="%.2f", key="ny_fr_xb",
                        help="Loads to Canada price in an empty backhaul and border delay. My assumption, not from the notes.")
        f5.number_input("Border cost ($/load)", 0.0, 2000.0, step=25.0, format="%.0f", key="ny_fr_border",
                        help="Broker, ACI eManifest filing and unpaid waiting time at the bridge. My assumption.")
    with st.expander("What each buyer bids (cents per bushel vs futures)", expanded=False):
        st.caption("Leave at 0 for a pure freight comparison. Enter each buyer's delivered bid basis to see net-backs and "
                   "the territory each can draw. Hamilton's bid also feeds the surtax customs value.")
        bc = st.columns(3)
        for i, d in enumerate(nfr.dests()):
            bc[i % 3].number_input(d["name"], -400.0, 400.0, 0.0, 5.0, format="%.0f", key=f"ny_fr_bid_{d['key']}")

    a = {f: float(ss[k]) for f, k in _FR_KEYS.items()}
    bids = {d["key"]: float(ss.get(f"ny_fr_bid_{d['key']}", 0.0)) for d in nfr.dests()}
    price = float(ss["ny_fr_price"])
    df = nfr.lane_table(a, float(tariff), price, bids)
    soy_bu, conv_bu, soy_year = _soy_bushels()

    # ── lane card for one origin ──────────────────────────────────────────────
    st.markdown('<div class="sh">Lane card</div>', unsafe_allow_html=True)
    o1, _ = st.columns([1, 3])
    county = o1.selectbox("Origin county", sorted(v["name"] for v in lanes["origins"].values()),
                          index=sorted(v["name"] for v in lanes["origins"].values()).index("Seneca"), key="ny_fr_origin")
    lane = df[df["County"] == county].set_index("key")
    us, ham = lane.loc["union_springs"], lane.loc["hamilton"]
    best = lane["netback_c"].idxmax()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("To Du Mond Union Springs", _cd(us["cost_c"]), f"{us['miles']:.0f} road miles", delta_color="off", delta_arrow="off")
    k2.metric("To Bunge Hamilton", _cd(ham["cost_c"]), f"{ham['miles']:.0f} road miles", delta_color="off", delta_arrow="off",
              help="Haul + border + Canada surtax.")
    k3.metric("In-state advantage", _cd(ham["cost_c"] - us["cost_c"]), "Hamilton vs Union Springs",
              delta_color="off", delta_arrow="off", help="Hamilton's cost minus Union Springs' cost, per bushel.")
    k4.metric("Best net-back", lane.loc[best, "short"], f"{lane.loc[best, 'netback_c']:+.0f}¢ vs futures", delta_color="off",
              delta_arrow="off", help="The buyer with the highest bid minus delivered cost. At zero bids that is the cheapest haul.")
    t = lane.reset_index()
    show = pd.DataFrame({"Buyer": t["Buyer"], "Road miles": t["miles"].round(0), "Drive (hrs)": t["hours"].round(1),
                         "Haul (¢/bu)": t["haul_c"].round(1), "Border (¢/bu)": t["border_c"].round(1),
                         "Surtax (¢/bu)": t["surtax_c"].round(1), "Cost (¢/bu)": t["cost_c"].round(1),
                         "Bid (¢)": t["bid_c"].round(0), "Net-back (¢)": t["netback_c"].round(1),
                         "vs Union Springs (¢)": t["vs_us_c"].round(1)})
    lc1, lc2 = st.columns([1.85, 1])
    lc1.dataframe(show, use_container_width=True, hide_index=True)
    bar = go.Figure()
    for col, nm, color in (("haul_c", "Haul", "#5e7164"), ("border_c", "Border + backhaul", "#f59e0b"),
                           ("surtax_c", "Surtax", "#dc2626")):
        bar.add_trace(go.Bar(y=t["short"], x=t[col], name=nm, orientation="h", marker_color=color,
                             hovertemplate=f"{nm}: %{{x:.1f}}¢<extra></extra>"))
    bar.update_layout(barmode="stack", height=max(300, 44 * len(t) + 90), paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE,
                      font=dict(color=nf.TEXT, family="Arial"), margin=dict(l=10, r=10, t=40, b=64),
                      title=dict(text=f"{county} County: cost to each buyer (¢/bu)", font=dict(size=13, color=nf.ACCENT)),
                      legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center", yanchor="top", font=dict(size=10)),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=10)), xaxis=dict(gridcolor=nf.BORDER))
    lc2.plotly_chart(bar, use_container_width=True, key="ny_fr_bar", config={"displayModeBar": False})
    st.caption("Origin = the county's centre point; real lanes run from a farm or elevator, so towns near a plant are closer "
               "than the county figure. Road miles from OpenStreetMap routing (OSRM), built "
               f"{lanes['built']}. Cost = haul + border + surtax; net-back = bid minus cost.")

    # ── map ───────────────────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#d7e2dc;margin:10px 0'>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns([1.9, 1.5, 1.6])
    mode = m1.radio("Map", ["Draw territory (best net-back)", "Delivered cost to a buyer", "In-state advantage over Hamilton"],
                    key="ny_fr_map", horizontal=False)
    buyer = None
    if mode.startswith("Delivered"):
        buyer = m2.selectbox("Buyer", [d["key"] for d in nfr.dests()], key="ny_fr_map_buyer",
                             format_func=lambda k: names[k]["name"])
    show_comp = m3.checkbox("Show crushers and export terminals", value=True, key="ny_fr_comp")
    comp_cats = ["crush"] if show_comp else None
    cost_note = f"{tariff:.0%} surtax on a ${price:.2f} bean" if tariff else "no surtax"
    if mode.startswith("Draw"):
        win = nfr.winners(df).copy()
        win["hover"] = [
            f"<b>{r.County}</b><br>Best net-back: {r.Buyer}<br>{r.miles:.0f} road miles · cost {_cd(r.cost_c)} · "
            f"net-back {r.netback_c:+.0f}¢<br>Conventional soybeans: {conv_bu.get(r.fips, 0) / 1e3:,.0f}K bu"
            for r in win.itertuples()]
        flat = all(v == 0 for v in bids.values())
        fig = nf.territory_fig(geo, win, nfr.DEST_COLORS, "Draw territory: the buyer each county nets most from",
                               ("Pure freight comparison (every buyer's bid at 0)" if flat else "Net-back = bid minus delivered cost")
                               + f" · {cost_note} · labels = cost to the winner", comp_cats)
    elif mode.startswith("Delivered"):
        sub = df[df["key"] == buyer].copy()
        sub["Value"] = sub["cost_c"]
        sub["hover"] = [f"<b>{r.County}</b><br>{r.miles:.0f} road miles ({r.hours:.1f} hrs)<br>Haul {r.haul_c:.0f}¢ · "
                        f"border {r.border_c:.0f}¢ · surtax {r.surtax_c:.0f}¢<br><b>Cost {_cd(r.cost_c)}/bu</b>"
                        for r in sub.itertuples()]
        fig = nf.lane_value_fig(geo, sub, f"Delivered cost to {names[buyer]['name']}", cost_note, "Cost<br>(¢/bu)",
                                comp_cats=comp_cats)
    else:
        us_c = df[df["key"] == "union_springs"].set_index("fips")["cost_c"]
        hm = df[df["key"] == "hamilton"].copy()
        hm["Value"] = hm["cost_c"] - hm["fips"].map(us_c)
        hm["hover"] = [f"<b>{r.County}</b><br>Hamilton {_cd(r.cost_c)} vs Union Springs {_cd(us_c[r.fips])}<br>"
                       f"<b>In-state advantage {_cd(r.Value)}/bu</b>" for r in hm.itertuples()]
        fig = nf.lane_value_fig(geo, hm, "In-state advantage: Bunge Hamilton cost minus Du Mond Union Springs cost",
                                cost_note, "Advantage<br>(¢/bu)", cscale=nf.SAGE_SEQ, comp_cats=comp_cats)
    st.plotly_chart(fig, use_container_width=True, key="ny_fr_map_chart",
                    config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
    st.caption("Hover a county for the breakdown. Stars are the Du Mond plants. Hamilton, Ontario and Perdue's Pennsylvania "
               "plants are off the map frame; their lanes still count.")

    # ── territory bushels ─────────────────────────────────────────────────────
    st.markdown('<div class="sh">Soybean bushels by buyer territory</div>', unsafe_allow_html=True)
    conv = st.checkbox("Count conventional bushels only (after the High-Oleic tab's carve-out)", value=True, key="ny_fr_conv")
    bu = conv_bu if conv else soy_bu
    w = nfr.winners(df)
    w["bu"] = w["fips"].map(bu).fillna(0.0)
    tot_bu = w["bu"].sum()
    grp = (w.groupby(["key", "Buyer"]).agg(Counties=("fips", "count"), bu=("bu", "sum"), miles=("miles", "mean"),
                                           cost=("cost_c", "mean")).reset_index().sort_values("bu", ascending=False))
    tbl = pd.DataFrame({"Buyer": grp["Buyer"], "Counties won": grp["Counties"], f"{soy_year} bushels (K)": (grp["bu"] / 1e3).round(0),
                        "% of NY": (grp["bu"] / tot_bu * 100).round(1) if tot_bu else 0.0,
                        "Avg road miles": grp["miles"].round(0), "Avg cost (¢/bu)": grp["cost"].round(1)})
    b1, b2 = st.columns([2.2, 1])
    b1.dataframe(tbl, use_container_width=True, hide_index=True)
    dm = w[w["key"].isin(["union_springs", "york"])]["bu"].sum()
    b2.metric("Du Mond's territory", f"{dm / 1e6:,.2f}M bu", f"{dm / tot_bu:.0%} of NY" if tot_bu else None,
              delta_color="off", delta_arrow="off", help="Bushels in the counties where Union Springs or York has the best net-back.")
    st.caption(f"{soy_year} NASS soybean production by county ("
               + ("conventional share only, from the High-Oleic tab's model and settings" if conv else "all soybeans")
               + "). With every bid at 0, a buyer 'wins' a county only on freight.")

    # ── surtax sensitivity ────────────────────────────────────────────────────
    st.markdown('<div class="sh">Surtax what-if (selected origin; soybeans are not on the list today)</div>', unsafe_allow_html=True)
    base_cost = ham["haul_c"] + ham["border_c"]
    prices = [price * 0.8, price, price * 1.2]
    grid = pd.DataFrame(
        {f"${p:.2f}/bu" + (" (now)" if i == 1 else ""): [
            round(base_cost + nfr.surtax_cents(p + bids["hamilton"] / 100.0, t_) - us["cost_c"], 0) for t_ in nfr.TARIFF_CHOICES]
         for i, p in enumerate(prices)},
        index=[f"{t_:.0%} surtax" for t_ in nfr.TARIFF_CHOICES])
    g1, g2 = st.columns([1.9, 1.5])
    g1.dataframe(grid.style.format("{:,.0f}¢"), use_container_width=True)
    g2.caption(f"In-state advantage in ¢/bu for {county} County beans (Hamilton cost minus Union Springs cost) at each surtax "
               "tier and soybean price. The surtax is a percentage of the price, so every move in beans moves the advantage; "
               "at 0% what is left is the freight difference alone.")

    # ── notes check ───────────────────────────────────────────────────────────
    with st.expander("Check against your notes (sections eight and nine)", expanded=False):
        def mi(c, k):
            return next(v["miles"][k] for v in lanes["origins"].values() if v["name"] == c)
        sen = df[df["County"] == "Seneca"].set_index("key")
        bench = nfr.lane_table(nfr.BENCHMARK, float(tariff), price, bids)
        bsen = bench[bench["County"] == "Seneca"].set_index("key")
        nsen = nfr.lane_table(nfr.NOTES, float(tariff), _NOTES_PRICE, {})
        nsen = nsen[nsen["County"] == "Seneca"].set_index("key")
        b0 = nfr.lane_table(nfr.BENCHMARK, 0.0, price, bids)
        b0 = b0[b0["County"] == "Seneca"].set_index("key")
        adv0 = b0.loc["hamilton", "cost_c"] - b0.loc["union_springs", "cost_c"]
        rows = [
            ("Seneca to Hamilton, miles", "~152", f"{mi('Seneca', 'hamilton'):.0f}", "County centre point, by road"),
            ("Seneca to Union Springs, miles", "~11", f"{mi('Seneca', 'union_springs'):.0f}",
             "Cayuga Lake forces a detour from Seneca's centre; Seneca towns on the lake are nearer"),
            ("Union Springs to Hamilton, miles", "~162", f"{mi('Cayuga', 'hamilton'):.0f}", "Cayuga County centre point"),
            ("Freight to Hamilton", "~3¢", f"{_cd(sen.loc['hamilton', 'haul_c'] + sen.loc['hamilton', 'border_c'])}",
             f"Notes preset gives {_cd(nsen.loc['hamilton', 'haul_c'])}; benchmark trucking with border costs gives "
             f"{_cd(bsen.loc['hamilton', 'haul_c'] + bsen.loc['hamilton', 'border_c'])}"),
            ("Surtax on soybeans", "$2.62 (25% of $10.48)", _cd(sen.loc["hamilton", "surtax_c"]),
             "Soybeans are not on Canada's 2026 surtax schedules, so this is 0 unless you switch a what-if on"),
            ("All-in cost to Hamilton", "$2.66", _cd(sen.loc["hamilton", "cost_c"]), "Haul + border + surtax"),
            ("In-state advantage", "$2.63", _cd(sen.loc["hamilton", "cost_c"] - sen.loc["union_springs", "cost_c"]),
             "Hamilton cost minus Union Springs cost"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Item", "Your notes", "This model (now)", "Comment"]),
                     use_container_width=True, hide_index=True)
        hb = nfr.haul_cents(152, nfr.BENCHMARK, True) + nfr.border_cents(nfr.BENCHMARK)
        hd = nfr.haul_cents(152, nfr.BENCHMARK, False)
        st.markdown(
            f"- **Freight is the one number I'd challenge.** 3¢ for 152 miles is 0.02¢ per bushel-mile. The notes say "
            f"'twenty cents per bushel-mile', which would be $30 a bushel, so that is probably a mis-transcription. At "
            f"roughly $4-5 per loaded mile (USDA AMS truck-rate benchmarks) a 152-mile haul costs about {_cd(hd)} per bushel "
            f"in New York and about {_cd(hb)} to Hamilton with the border and empty backhaul, 20-40 times the notes' figure.\n"
            f"- **The premise behind the $2.63 does not hold for soybeans.** Canada's United States Surtax Order (2026) lists 335 "
            f"tariff items (15%: 21, 25%: 172, 50%: 142): dairy, molasses and syrups, doughs and mixes, steel, machinery, "
            f"apparel and the like. None is in chapter 12, so US soybeans (1201) pay no surtax; the March 2025 order was "
            f"repealed on September 1, 2025. Checked against the order's text on 2026-10-27. Without it the in-state advantage "
            f"to Hamilton is the freight and border difference alone, about {_cd(adv0)} a bushel for Seneca County at "
            f"benchmark trucking, not $2.63. The surtax rows above are a what-if.\n"
            f"- **Mileage:** I measure from each county's centre point, so a county's lanes can differ from a town-to-plant "
            f"figure by 15 miles or more.\n- **Surtax base:** modeled on the futures price plus Hamilton's bid, the customs "
            f"value for goods sold into Canada.")

    with st.expander("County detail (all counties, all buyers)", expanded=False):
        wide_c = df.pivot(index="County", columns="short", values="cost_c").round(0)
        wide_m = df.pivot(index="County", columns="short", values="miles").round(0)
        wide = wide_c.add_suffix(" ¢").join(wide_m.add_suffix(" mi"))
        wn = nfr.winners(df).set_index("County")
        wide.insert(0, "Best buyer", wn["short"])
        st.dataframe(wide.reset_index(), use_container_width=True, hide_index=True)
        st.caption("Cost in cents per bushel (haul + border + surtax) and road miles from the county centre point.")


# ── Ontario: what it grows, uses and might need ───────────────────────────────
def _ontario_tab() -> None:
    try:
        soy_bu, conv_bu, ny_year = _soy_bushels()
        ny_total, ny_conv = float(sum(soy_bu.values())), float(sum(conv_bu.values()))
    except Exception:
        ny_total = ny_conv = ny_year = None
    ss = st.session_state
    draw_ontario.render(ny_total, ny_conv, ny_year, float(ss.get("ny_fr_price", _NOTES_PRICE)),
                        float(ss.get("ny_fr_tariff", 0.0)), _geo(), _ny_pair, list(nd.YEARS))


def _ny_pair(crop_label: str, year: int, metric: str, change: str, comp, view: str, show_comp: bool) -> dict:
    """The New York half of the Ontario + New York map, built the way the NASS Production tab builds its maps: the values (so both can
    share a colour range), a figure builder that takes that range, its unit for the subtitle and a note. The forecast year has ASD
    districts only."""
    geo = _geo()
    cobj, _err = _get_crop(crop_label)
    ho = float(st.session_state.get("ny_ho", 0.0)) / 100.0 if crop_label == "Soybeans" else 0.0
    adj = nd.Adj(ho=ho, ts=False)
    mk = ny_ho.markers_for(crop_label)                     # beans and corn each get their own markers
    comp_cats = mk["cats"] if show_comp and mk["cats"] else None
    show_eth = show_comp and mk["ethanol"]
    is_fc = year == nd.FORECAST_YEAR and cobj.reported_counties(year) == 0
    note = ""
    fc_stats: dict = {}
    fc_county: dict = {}
    y_in = 0.0
    if is_fc:
        src = cobj.default_yield_sources(adj)
        y_in = float(st.session_state.get(f"ny_fc_yield_{crop_label}") or src["usda"] or src["ly"] or 0.0)
        if y_in > 0:
            fc_stats, _ = cobj.asd_forecast(y_in, adj)
            fc_county = cobj.county_forecast(y_in, adj)
    if view == "County" and (not is_fc or fc_county):
        cdf = cobj.county_view(metric, year, change, comp, adj, fc_county)
        if cdf is not None and not cdf.empty:
            n_est = int(cdf["is_estimated"].sum())
            if is_fc:
                note = (f"New York {year}: county estimates come after harvest, so the counties are each ASD district's forecast "
                        f"({y_in:g} bu/ac, set on the NASS Production tab) split by average county shares.")
            elif n_est:
                note = f"New York: {n_est} counties NASS withholds are estimated (Est)."
            return {"values": [v for v in cdf["Value"] if pd.notna(v)], "note": note, "unit": "NASS counties",
                    "fig": lambda z: nf.county_fig(cdf, geo, crop_label, year, metric, change, comp_cats, zrange=z,
                                                   ethanol=show_eth, forecast=is_fc)}
        return {"values": [], "fig": None, "note": ""}
    if is_fc:
        note = (f"New York {year}: county estimates come after harvest, so this shows the NASS ag districts from the forecast yield "
                f"({y_in:g} bu/ac, set on the NASS Production tab).")
    view_df, cur_df, est_d = cobj.district_view(metric, year, change, comp, adj, fc_stats if is_fc else None)
    if view_df is None or view_df.empty:
        return {"values": [], "fig": None, "note": note}
    return {"values": [v for v in view_df["Value"] if pd.notna(v)], "note": note, "unit": "NASS ag districts",
            "fig": lambda z: nf.district_fig(view_df, cur_df, geo, crop_label, year, metric, change, est_d, comp_cats, zrange=z,
                                             ethanol=show_eth)}


# ── livestock: Census county / ASD view, then the annual state survey ─────────
_LS_UNITS ={"Inventory (head)": ("K head", 1e3, ".1f"), "Farms with Inventory": ("farms", 1, ",.0f"),
             "Head per Farm": ("head/farm", 1, ".0f")}


@st.cache_resource(show_spinner=False)
def _ls_obj(comm: str) -> nl.NYLivestock:
    return nl.NYLivestock(comm, _geo())


def _fmt_ls(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}"


def _livestock_tab() -> None:
    geo = _geo()
    codes = {v[0]: v[1] for v in _fips_map().values()}

    c1, c2, c3 = st.columns([1.1, 0.75, 2.35])
    comm = c1.selectbox("Livestock", nl.COMMODITIES, key="ny_ls_comm")
    year = c2.selectbox("Census Year", nl.years(), key="ny_ls_year")
    c3.selectbox("State Drill-Down", ["NY  —  New York"], key="ny_ls_state", disabled=True)

    st.markdown("<p style='color:#5f7267;font-size:0.78rem;margin:4px 0 2px 0;'>Metric</p>", unsafe_allow_html=True)
    metric = st.radio("Metric", nl.METRICS, horizontal=True, key="ny_ls_metric", label_visibility="collapsed")
    st.markdown("<p style='color:#5f7267;font-size:0.78rem;margin:4px 0 2px 0;'>View</p>", unsafe_allow_html=True)
    change = st.radio("Change View", nl.CHANGE_OPTS, horizontal=True, key="ny_ls_change", label_visibility="collapsed",
                      format_func=lambda o: "Census Year" if o == "Current Year" else o)
    comp = None
    if change == "vs Selected Year":
        cc, _ = st.columns([0.55, 2.6])
        comp = cc.selectbox("Compare to Census", [y for y in nl.years() if y != year], key="ny_ls_comp_year")

    L = _ls_obj(comm)
    stat = nl.stat_of(metric)
    ratio = metric in nl.RATIO_METRICS
    st.caption("USDA Census of Agriculture, inventory as of Dec 31. NASS publishes county livestock only in the Census "
               "(every five years); the annual survey below is state level.")

    state_val = L.state_value(stat, year)
    n_listed = len(L.county_values("inventory", year))
    n_rep = L.reported_counties(year)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{year} {metric}", _fmt_ls(state_val))
    if change == "Current Year":
        k2.metric("County Coverage", f"{n_rep / n_listed * 100:.0f}%" if n_listed else "—",
                  help=f"{n_rep} of {n_listed} counties have a published inventory; NASS withholds the rest to protect "
                       "individual operations, and they are estimated here.")
        k3.metric("Counties Reporting", f"{n_rep:,}")
        k4.metric("ASD Districts", f"{len(codes)}")
    else:
        cv_all = L.county_view(metric, year, change, comp)
        vals = cv_all["Value"].dropna()
        k2.metric("County Coverage", f"{n_rep / n_listed * 100:.0f}%" if n_listed else "—")
        k3.metric("Counties Above Prior", f"{int((vals > 0).sum()):,} ▲")
        k4.metric("Counties Below Prior", f"{int((vals < 0).sum()):,} ▼")

    # ── map view ──────────────────────────────────────────────────────────────
    mv1, mv2 = st.columns([1, 2])
    mv1.radio("Map View", ["ASD District", "County"], horizontal=True, key="ny_ls_map_view")
    map_view = st.session_state["ny_ls_map_view"]
    with mv2:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        show_comp = st.checkbox("Show competition on the map (roasters, mills, dealers, crushers)", key="ny_comp_ls")
    comp_cats = list(ny_ho.CATEGORIES) if show_comp else None
    no_base = change != "Current Year" and L._base_year(year, change, comp) is None

    if map_view == "ASD District":
        view_df, cur_df, est_d = L.district_view(metric, year, change, comp)
        fig = None if no_base else nf.district_fig(view_df, cur_df, geo, comm, year, metric, change, est_d, comp_cats,
                                                   src="Census")
        if fig is None:
            st.info(f"No earlier Census to compare {year} with." if no_base else "ASD district map not available for "
                    "this selection.")
        else:
            fig.update_layout(dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key="ny_ls_district_map",
                            config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
        if not cur_df.empty:
            t = cur_df.copy()
            t["ASD"] = t["District"].map(codes).fillna("")
            t = t.sort_values("ASD").reset_index(drop=True)
            tot = state_val if ratio else t["Value"].sum()
            t[metric] = t["Value"].apply(_fmt_ls)
            t["% of State"] = (t["Value"] / tot * 100).round(1).astype(str) if not ratio and tot else "—"
            cols = ["District", "ASD", metric, "% of State"]
            cv_est = L.county_values(stat if stat != "per_farm" else "inventory", year)
            n_est = int(cv_est["is_estimated"].sum()) if not cv_est.empty else 0
            if n_est:
                e = (cv_est[cv_est["is_estimated"]].groupby("District")["is_estimated"].count()
                     .rename("Est. Counties").reset_index())
                t = t.merge(e, on="District", how="left")
                t["Est. Counties"] = t["Est. Counties"].fillna(0).astype(int).apply(lambda n: str(n) if n else "—")
                cols.append("Est. Counties")
            st.markdown(f"<p style='color:#5f7267;font-size:0.82rem;font-weight:600;margin:6px 0 4px 0;'>"
                        f"New York {year} {comm} — by ASD district</p>", unsafe_allow_html=True)
            tc, kc = st.columns([2.4, 1])
            with tc:
                st.dataframe(t[cols], use_container_width=True, hide_index=True)
            with kc:
                m1, m2 = st.columns(2)
                m1.metric("State Avg" if ratio else "State Total", _fmt_ls(tot))
                m2.metric("Districts", f"{len(t)}")
                if n_est:
                    st.caption("Est = some counties withheld by NASS. The state Census total minus the published "
                               "counties is shared among the withheld ones in proportion to their farm counts.")
    else:
        cdf = L.county_view(metric, year, change, comp)
        fig = None if no_base else nf.county_fig(cdf, geo, comm, year, metric, change, comp_cats, src="Census")
        if fig is None:
            st.info(f"No earlier Census to compare {year} with." if no_base else "County map not available for this "
                    "selection.")
        else:
            fig.update_layout(dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key="ny_ls_county_map",
                            config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
            n_cty_est = int(cdf["is_estimated"].sum())
            if n_cty_est > 0:
                st.caption(f"Est = {n_cty_est} counties withheld by NASS. The state Census total minus the published "
                           "counties is shared among the withheld ones in proportion to their farm counts. Counties "
                           "NASS lists no data for show as No data.")

    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)
    _ls_history(L, comm, year, metric, change, comp, codes)
    st.markdown("<hr style='border-color:#d7e2dc;margin:14px 0 8px 0'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5f7267;font-size:0.82rem;font-weight:600;margin:0 0 6px 0;letter-spacing:0.04em;'>"
                "📈 ANNUAL STATE SURVEY — NEW YORK</p>", unsafe_allow_html=True)
    _ls_annual()


def _ls_history(L, comm, year, metric, change, comp, codes) -> None:
    stat = nl.stat_of(metric)
    ratio = metric in nl.RATIO_METRICS
    unit_lbl, div, fmt = _LS_UNITS[metric]
    state_name = "New York"
    h1, h2 = st.columns([1.4, 2.6])
    scope = h1.radio("View", ["State", "ASD District", "County"], horizontal=True, key="ny_ls_scope")
    dist = county = None
    with h2:
        if scope == "ASD District":
            dist = st.selectbox("District", sorted(codes), key="ny_ls_hist_district")
            label = f"{state_name} — {dist}"
        elif scope == "County":
            county = st.selectbox("County", sorted(L.names.values()), key="ny_ls_hist_county")
            label = f"{state_name} — {county} County"
        else:
            label = state_name

    hyears = sorted(y for y in L.years if y <= year)

    def value_for(y: int):
        if scope == "State":
            return L.state_value(stat, y)
        if scope == "ASD District":
            return L.district_series(metric, y)[0].get(dist)
        cv = L.county_values(stat, y)
        r = cv[cv["County"] == county]
        return float(r["Value"].iloc[0]) if not r.empty else None

    vals = [value_for(y) for y in hyears]
    st.plotly_chart(nf.history_bar({metric: [v / div if v else 0 for v in vals]}, hyears,
                                   f"{label} — {comm} {metric}", unit_lbl, fmt=fmt),
                    use_container_width=True, key="ny_ls_hist_chart", config={"displayModeBar": False})
    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)

    st.markdown(f"<p style='color:#5f7267;font-size:0.82rem;font-weight:600;margin:0 0 6px 0;letter-spacing:0.04em;'>"
                f"📅 CENSUS HISTORY — {label} | {min(hyears)}–{max(hyears)}</p>", unsafe_allow_html=True)
    title_tail = f"({unit_lbl}) | {min(hyears)}–{max(hyears)}"
    ly = "% vs Prior Census"
    dist_county: dict = {}
    d_yr: dict = {}
    if scope == "State":
        rows = {state_name: {y: L.state_value(stat, y) for y in hyears}}
        fig = nf.heatmap_table(rows, hyears, f"{comm} — {metric} {title_tail}", unit_lbl, div, ratio, fmt=fmt,
                               first_col="State", ly_label=ly)
    elif scope == "ASD District":
        status: dict = {}
        totals: dict = {}
        est_by_dist: dict = {}
        for y in hyears:
            vy, ey = L.district_series(metric, y)
            for d, v in vy.items():
                d_yr.setdefault(d, {})[y] = v
                status.setdefault(d, {})[y] = ey.get(d, False)
            cdf = L.county_values(stat, y)
            if not cdf.empty:
                totals[y] = float(cdf["Value"].sum()) if not ratio else L.state_value(stat, y)
            for r in cdf[cdf["District"].notna()].itertuples():
                dist_county.setdefault(r.District, {}).setdefault(r.County, {})[y] = float(r.Value)
                est_by_dist.setdefault(r.District, {}).setdefault(r.County, {})[y] = bool(r.is_estimated)
        fig = nf.heatmap_table(d_yr, hyears, f"New York {comm} — {metric} by ASD District {title_tail}", unit_lbl, div,
                               ratio, total=totals if not ratio else None, fmt=fmt, cell_status=status,
                               total_label="NY Total", first_col="District", ly_label=ly)
    else:
        c_yr: dict = {}
        for y in hyears:
            for r in L.county_values(stat, y).itertuples():
                c_yr.setdefault(r.County, {})[y] = float(r.Value)
        fig = nf.heatmap_table(c_yr, hyears, f"New York {comm} — {metric} by County {title_tail}", unit_lbl, div, ratio,
                               fmt=fmt, total_label="NY Total", first_col="County", ly_label=ly)
    st.plotly_chart(fig, use_container_width=True, key="ny_ls_heatmap", config={"displayModeBar": False})

    if scope == "ASD District" and dist_county:
        st.caption("▼ Expand any district below to view county detail  ·  Top 2 = green  ·  Bottom 2 = red per row  ·  "
                   "E = includes a withheld county estimated by Census share  ·  Olympic Avg drops the highest and lowest year.")
        for dname in sorted(dist_county):
            cd = dist_county[dname]
            with st.expander(f"▼  {dname}  —  {len(cd)} counties", expanded=False):
                est_map = est_by_dist.get(dname, {})
                cs = {cn: {y: est_map.get(cn, {}).get(y, False) for y in hyears} for cn in cd}
                top_cs = {y: any(est_map.get(cn, {}).get(y, False) for cn in cd) for y in hyears}
                cfig = nf.heatmap_table(
                    cd, hyears, f"{dname} — County Detail ({unit_lbl})", unit_lbl, div, ratio, fmt=fmt,
                    top_row=d_yr.get(dname, {}), top_row_label=f"{dname} District Total", cell_status=cs,
                    top_row_cell_status=top_cs, total_label="NY Total", first_col="County", ly_label=ly)
                st.plotly_chart(cfig, use_container_width=True, key=f"ny_ls_cty_tbl_{dname}",
                                config={"displayModeBar": False})
    else:
        st.caption("Top 2 = green  ·  Bottom 2 = red per row  ·  Olympic Avg drops the highest and lowest Census year.")
    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)

    cv = L.county_view(metric, year, change, comp)
    if not cv.empty:
        st.plotly_chart(nf.ranking_chart(cv, comm, year, metric, change,
                                         by_district=st.session_state.get("ny_ls_map_view") == "ASD District",
                                         src="Census"),
                        use_container_width=True, key="ny_ls_ranking")
        with st.expander("County Data Table — New York", expanded=False):
            tbl = cv[["County", "Value", "District", "is_estimated"]].dropna(subset=["Value"]) \
                .sort_values("Value", ascending=False)
            col = f"% Change ({metric})" if change != "Current Year" else metric
            show = pd.DataFrame({"County": tbl["County"], "District": tbl["District"],
                                 col: [f"{v:+.1f}%" if change != "Current Year" else _fmt_ls(v) for v in tbl["Value"]],
                                 "Estimated": ["Est" if e else "" for e in tbl["is_estimated"]]})
            st.dataframe(show, use_container_width=True, hide_index=True)


def _ls_annual() -> None:
    series, err = _livestock()
    _source_note(err)
    ty = pd.Timestamp.today().year
    lo, hi = st.slider("Historical range", 1980, ty, (1990, ty), step=1, key="ny_ls_range")
    cols = st.columns(len(series), gap="small")
    for col, (comm, df) in zip(cols, series.items()):
        ref = "Dec 1" if comm == "Hogs & Pigs" else "Jan 1"
        if df.empty:
            col.markdown(_card(comm, "<span style='font-size:16px;color:#9ca3af;'>No data</span>"), unsafe_allow_html=True)
            continue
        ly = int(df["year"].max())
        v = float(df.loc[df["year"] == ly, "value"].iloc[0])
        p = df[df["year"] == ly - 1]
        sub = ""
        if not p.empty and p["value"].iloc[0]:
            pc = (v - float(p["value"].iloc[0])) / float(p["value"].iloc[0]) * 100
            sub = f"<div class='kpi-delta {'kpi-pos' if pc >= 0 else 'kpi-neg'}'>{'▲' if pc >= 0 else '▼'} {abs(pc):.1f}% vs {ly - 1}</div>"
        col.markdown(_card(f"{comm} — {ref}", f"{v / 1e3:,.1f}K" if v < 1e6 else f"{v / 1e6:.2f}M",
                           sub + f"<div class='kpi-year'>{ly} — New York State</div>"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(len(series), gap="small")
    for col, (comm, df) in zip(cols, series.items()):
        color, fill = _LS_COLORS[comm]
        ref = "Dec 1" if comm == "Hogs & Pigs" else "Jan 1"
        d = df[df["year"].between(lo, hi)] if not df.empty else df
        fig = go.Figure()
        if d.empty:
            fig.add_annotation(text="No data", showarrow=False, font=dict(size=11, color="#9ca3af"))
        else:
            fig.add_trace(go.Scatter(x=d["year"].astype(int), y=d["value"], mode="lines+markers",
                                     line=dict(color=color, width=2.5), marker=dict(size=4, color=color),
                                     fill="tozeroy", fillcolor=fill,
                                     hovertemplate="<b>%{x}</b><br>Inventory: %{y:,.0f}<extra></extra>"))
        fig.update_layout(title=dict(text=f"{comm} — {ref} (NY)", font=dict(size=12, color=_CTX["dark"]), x=0),
                          height=300, paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
                          margin=dict(l=55, r=14, t=40, b=32), hovermode="x unified",
                          xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#6b7280"), linecolor="#e5e7eb"),
                          yaxis=dict(gridcolor="#e5e7eb", tickformat=",.0f", tickfont=dict(size=9, color="#6b7280"),
                                     linecolor="#e5e7eb"))
        col.plotly_chart(fig, use_container_width=True, key=f"ny_ls_ann_{comm}")
    st.caption("Livestock: USDA NASS annual surveys. Jan 1 cattle/dairy, Dec 1 hogs. NASS publishes New York hog "
               "inventory through 2023.")
