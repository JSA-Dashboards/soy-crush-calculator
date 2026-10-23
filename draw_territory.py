"""Draw Territory tab: the ASD dashboard's NASS Production drill-down for New York, plus NY livestock.

Data: live JSA.NASS_CACHE (one NY-filtered query per crop), falling back to the bundled
ny_nass_snapshot.json. Logic lives in ny_asd_data.py, figures in ny_asd_figs.py.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dt_nass
import ny_asd_data as nd
import ny_asd_figs as nf

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
def render(sf_connect, sf_secret, dark: str, blue: str) -> None:
    _CTX.update(connect=sf_connect, secret=sf_secret, dark=dark, blue=blue)
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{dark} 0%,#2a4a2a 100%);border-radius:8px;padding:16px 20px;margin-bottom:14px;">
  <div style="font-size:1.2rem;font-weight:700;color:#fff;">🗺️ Draw Territory — New York State</div>
  <div style="font-size:0.875rem;color:#ccc;margin-top:4px;">
    USDA NASS production by ASD district and county, with estimates for counties NASS withholds, plus livestock inventory.
  </div>
</div>""", unsafe_allow_html=True)
    t_nass, t_live = st.tabs(["NASS Production", "Livestock"])
    with t_nass:
        _nass_tab()
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
    if is_fc:
        fc_stats, fc_totals = _forecast_block(cobj, crop, year, adj)

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
        cv_all = cobj.county_view(metric, year, change, comp, adj)
        vals = cv_all["Value"].dropna()
        k2.metric("County Coverage", f"{pct_rep:.0f}%", help=f"{curn:,} of {bench:,} counties reporting")
        k3.metric("Counties Above Prior", f"{int((vals > 0).sum()):,} ▲")
        k4.metric("Counties Below Prior", f"{int((vals < 0).sum()):,} ▼")

    # ── map view ──────────────────────────────────────────────────────────────
    st.radio("Map View", ["ASD District", "County"], horizontal=True, key="ny_map_view")
    map_view = st.session_state["ny_map_view"]

    if map_view == "ASD District":
        view_df, cur_df, est_d = cobj.district_view(metric, year, change, comp, adj, fc_stats if is_fc else None)
        n_est = 0
        comp_df = pd.DataFrame()
        if stat in nd.ESTIMABLE and not is_fc:
            comp_df = cobj.completed(stat, year, adj)
            n_est = int(comp_df["is_estimated"].sum()) if not comp_df.empty else 0
        fig = nf.district_fig(view_df, cur_df, geo, crop, year, metric, change, est_d)
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
        cdf = cobj.county_view(metric, year, change, comp, adj)
        fig = nf.county_fig(cdf, geo, crop, year, metric, change)
        if fig is None:
            st.info("County map not available for this selection (county estimates are published after harvest)."
                    if is_fc else "County map not available for this selection.")
        else:
            fig.update_layout(dragmode=False)
            st.plotly_chart(fig, use_container_width=True, key="ny_county_map",
                            config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
            n_cty_est = int(cdf["is_estimated"].sum())
            if n_cty_est > 0:
                st.caption(f"Est = {n_cty_est} counties not published by NASS. Production is estimated using each "
                           "county's historical share of state output, adjusted for current district performance, "
                           "and scaled to reconcile with the NASS state total.")

    st.markdown("<hr style='border-color:#d7e2dc;margin:8px 0'>", unsafe_allow_html=True)
    _history_section(cobj, crop, year, metric, change, comp, adj, is_fc, fc_stats, fc_totals, fm, codes)


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


def _history_section(cobj, crop, year, metric, change, comp, adj, is_fc, fc_stats, fc_totals, fm, codes) -> None:
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
    cv = cobj.county_view(metric, year, change, comp, adj)
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


# ── livestock ─────────────────────────────────────────────────────────────────
def _livestock_tab() -> None:
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
        col.plotly_chart(fig, use_container_width=True, key=f"ny_ls_{comm}")
    st.caption("Livestock: USDA NASS annual surveys. Jan 1 cattle/dairy, Dec 1 hogs. NASS publishes New York hog "
               "inventory through 2023.")
