"""Draw Territory > Ontario: what Ontario grows, what it uses, and what it might need.

The question behind this page is demand: does Ontario have enough soybeans (or corn) for its own processors, and how big
could its call on outside beans be compared with New York's crop? Logic is in ontario.py, figures in ontario_figs.py.
"""
import time

import numpy as np
import pandas as pd
import streamlit as st

import ny_asd_figs as nf
import ontario as on
import ontario_figs as of
import ontario_trade as ot
import ontario_vh as ovh

_SOY_TIP = ("Ontario's own processing ('milling' in OMAFRA's table) has run about 3 Mt a year since 2019/20 and exports "
            "another 1.1-1.3 Mt, so a short crop is felt in exports first and then in imports.")
_CONF_TXT = {"high": "high confidence: EIA and trade sources", "medium": "medium confidence",
             "low": "low confidence: dated or partial source"}


def _mbu(crop: str, kt: float) -> str:
    return f"{on.to_mbu(crop, kt):,.1f}M bu"


def _kt(v: float) -> str:
    return f"{v:,.0f} kt"


def _k(crop: str, name: str) -> str:
    return f"on_{crop}_{name}"


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _vh_live() -> pd.DataFrame:
    return ovh.fetch()


@st.cache_resource(show_spinner=False)
def _vh_backoff() -> dict:
    return {"until": 0.0, "error": ""}


def _vh() -> tuple:
    """(weekly series, error). The live NOAA series when reachable; otherwise the bundled copy and the reason. After a
    failure the live call is not retried for ten minutes, so an outage cannot slow every interaction."""
    state = _vh_backoff()
    if time.time() < state["until"]:
        return ovh.snapshot(), state["error"]
    try:
        return _vh_live(), None
    except Exception as e:
        state.update(until=time.time() + 600, error=f"{type(e).__name__}: {str(e)[:140]}")
        return ovh.snapshot(), state["error"]


def _set_production(crop: str, kt: float) -> None:
    st.session_state[_k(crop, "p_latest")] = float(round(kt))


def _apply_estimate(crop: str) -> None:
    """Selecting a crop estimate sets the latest-year production in the balance's assumptions."""
    picked = {e["label"]: e for e in on.estimates(crop)}[st.session_state[_k(crop, "est")]]
    st.session_state[_k(crop, "p_latest")] = float(round(picked["kt"]))


def _zr(values: list, cfg: dict, cap: float = 100.0) -> tuple:
    """Colour range for a set of values: zero up to the largest, or symmetric (capped at +/-cap %) for the change views."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return (0.0, 1.0)
    if cfg["diverging"]:
        m = min(max(abs(min(vals)), abs(max(vals)), 1.0), cap)
        return (-m, m)
    return (0.0, max(vals))


def _one_map(crop: str, crop_label: str, latest: int, p_latest: float, st_latest: dict, ny_geo: dict, ny_pair, ny_years) -> None:
    """Ontario and New York on one map, always shown, coloured on one scale and driven by one set of controls."""
    st.markdown('<div class="sh">Ontario and New York on one map</div>', unsafe_allow_html=True)
    years = [y for y in (ny_years or list(range(latest, 2014, -1))) if y >= 2015]
    c1, c2, c3 = st.columns([0.7, 2.0, 2.4])
    year = c1.selectbox("Year", years, index=min(1, len(years) - 1), key="on_pair_year")
    metric = c2.radio("Metric", on.METRICS, horizontal=True, key="on_pair_metric")
    change = c3.radio("View", on.CHANGE_OPTS, horizontal=True, key="on_pair_change")
    comp = None
    if change == "vs Selected Year":
        cc, _ = st.columns([0.7, 3])
        comp = cc.selectbox("Compare to Year", [y for y in years if y != year], key="on_pair_comp")
    d1, d2 = st.columns([1.5, 4.5])
    view = d1.radio("Map view", ["County", "District"], horizontal=True, key="on_pair_view",
                    help="District = OMAFRA's five regions for Ontario and the NASS ag districts for New York.")
    plants = d2.checkbox("Show plants and competitors (crushers, ethanol plants, Du Mond, NY roasters and mills)", value=True,
                         key="on_pair_plants")

    is_fc = on.is_forecast(crop, year)
    prod_kt = float(st.session_state.get(_k(crop, "p_latest"), p_latest))
    acres = st_latest.get("harvested_ac")
    on_view = "County" if view == "County" else "Region"
    ov = on.county_view(crop, metric, year, change, comp, on_view, prod_kt if is_fc else None, acres if is_fc else None)
    ny = ny_pair(crop_label, year, metric, change, comp, "County" if view == "County" else "ASD District", plants) if ny_pair else None

    cfg = nf.view_cfg(metric, change)
    on_vals = list(ov["Value"]) if not ov.empty else []
    ny_vals = list(ny["values"]) if ny else []
    zr = _zr(on_vals + ny_vals, cfg)                       # one range for both, so the two are directly comparable
    on_sub = "OMAFRA counties" if view == "County" else "OMAFRA regions"
    if is_fc:
        on_sub = f"{year} forecast, {prod_kt:,.0f} kt split by 5-yr county share"
    elif not ov.empty and abs(ov["adj"].mean()) > on.RECONCILE_TOL:
        on_sub += f", scaled {ov['adj'].mean():+.1%} to StatCan"
    has_ny = bool(ny and ny.get("fig") is not None)
    sub = f"Ontario: {on_sub}" + (f"  ·  New York: {ny['unit']}" if has_ny else "")
    view_label = metric if change == "Current Year" else f"{change} — {metric}"
    kinds = [k for k in on.NODE_KINDS if k != "ny"] if plants else []      # the New York map already draws the Du Mond plants

    base = ny["fig"](zr) if has_ny else None
    if ov.empty:
        st.info(f"No earlier year to compare Ontario {year} with." if change != "Current Year" else "No Ontario data for this selection.")
        fig = base
    else:
        fig = of.combined_map(ov, metric, change, f"{crop_label} {year} — {view_label} | Ontario and New York", sub, kinds, base, ny_geo,
                              on_view, zr)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key="on_pair_map",
                        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
    notes = []
    if not has_ny:
        notes.append("No New York data for this selection.")
    if is_fc:
        notes.append(f"Ontario {year}: no county data yet, so the selected provincial estimate is split by each county's olympic-average "
                     "share of production and acres over the last five years (the NASS tab's forecast method).")
    elif not ov.empty and abs(ov["adj"].mean()) > on.RECONCILE_TOL:
        notes.append("Ontario county values are reconciled to OMAFRA's regional subtotals and then to StatCan's provincial total "
                     f"({on.production_kt(crop, year):,.0f} kt).")
    if ny and ny.get("note"):
        notes.append(ny["note"])
    notes.append("Hover a county for the detail. Ontario and New York share the crop, year, metric, view and one colour scale.")
    st.caption(" ".join(notes))


def render(ny_soy_bu, ny_conv_bu, ny_year, price_usd: float, tariff: float, ny_geo: dict, ny_pair=None, ny_years=None) -> None:
    st.markdown(
        "<div class='callout'><strong>What might Ontario need?</strong> Ontario grows about 4 Mt of soybeans and 10 Mt of corn, "
        "but its processors, feed trade and ethanol plants use a lot of both. This page lays out Ontario's supply and use by "
        "crop year, runs a two-year balance with the assumptions you can change, and sets any shortfall against New York's "
        "crop and the freight lane to Hamilton. Data: Statistics Canada, OMAFRA, NOAA and the US Census, all open.</div>",
        unsafe_allow_html=True)

    crop_label = st.radio("Crop", list(on.CROPS), horizontal=True, key="on_crop")
    crop = on.CROPS[crop_label]
    noun = "soybean" if crop == "soy" else "corn"
    a0 = on.defaults(crop)
    latest = on.latest_statcan_year(crop)
    prev = latest - 1
    p_latest, p_prev = on.production_kt(crop, latest), on.production_kt(crop, prev)
    st_latest = on.statcan(crop, latest)

    _one_map(crop, crop_label, latest, p_latest, st_latest, ny_geo, ny_pair, ny_years)

    # ── headline numbers ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{latest} crop (StatCan model)", _kt(p_latest), f"{(p_latest / p_prev - 1) * 100:+.1f}% vs {prev}",
              delta_color="off", help=f"{st_latest.get('yield_bu', 0):.1f} bu/ac, {st_latest.get('harvested_ac', 0) / 1e6:.2f}M acres "
                                      "harvested. Model-based estimate; the final survey figure is due 2026-12-04.")
    if crop == "soy":
        k2.metric("Processing run-rate", _kt(a0["milling"]), f"{_mbu(crop, a0['milling'])} a year", delta_color="off",
                  help="Average of the last two published crop years (OMAFRA 'milling').")
    else:
        firm = sum(a0[k] for _, k in on.LINES[crop]["firm"])
        k2.metric("Domestic use (food, ethanol, feed)", _kt(firm), f"{_mbu(crop, firm)} a year", delta_color="off",
                  help="Average of the last three published crop years.")
    k3.metric("Typical exports", _kt(a0["exports"]), f"{_mbu(crop, a0['exports'])} a year", delta_color="off",
              help="Average of the last three published crop years.")
    k4.metric(f"Last carryout ({on.crop_year_label(a0['last_complete'])})", _kt(a0["begin"]), f"{_mbu(crop, a0['begin'])}",
              delta_color="off", help="Ending stocks of the last crop year OMAFRA has published in full.")
    if crop == "soy":
        st.caption(_SOY_TIP)

    st.plotly_chart(of.sd_chart(crop), use_container_width=True, key="on_sd_chart", config={"displayModeBar": False})
    with st.expander("Supply and use table (kt)", expanded=False):
        df = on.sd_table(crop)
        lines = on.LINES[crop]
        cols = [("Beginning stocks", "beginning"), ("Production", "production")] + \
               ([("Imports", "imports")] if crop == "corn" else []) + [("Total supply", "supply")] + lines["firm"] + \
               [lines["exports"], lines["ending"]]
        tbl = pd.DataFrame({lab: df[k] for lab, k in cols if k in df}).tail(10)
        tbl.index = [on.crop_year_label(y) + ("" if df.loc[y, "complete"] else " (in progress)") for y in tbl.index]
        st.dataframe(tbl.T.round(0).style.format("{:,.0f}", na_rep="—"), use_container_width=True)
        st.caption("OMAFRA supply and disposition, crop years run October to September. The year in progress shows supply only. "
                   + ("This file has no soybean import line, so Ontario's soybean imports are not in it." if crop == "soy" else ""))

    # ── how fresh is the crop estimate ────────────────────────────────────────
    st.markdown('<div class="sh">How fresh is the crop estimate?</div>', unsafe_allow_html=True)
    ests = on.estimates(crop)
    st.caption("StatCan's September 16 model is the latest official figure and its final survey number comes December 4 (there is "
               "no July update). The yield tour and trade estimate are independent cross-checks.")
    est_df = pd.DataFrame([{"Estimate": e["label"], "Date": e["when"], "Yield (bu/ac)": float("nan") if e["yield_bu"] is None else e["yield_bu"],
                            f"{latest} production (kt)": e["kt"], "M bu": on.to_mbu(crop, e["kt"]),
                            "vs StatCan model": e["kt"] / ests[0]["kt"] - 1, "How it is built": e["note"]} for e in ests])
    st.dataframe(est_df.style.format({"Yield (bu/ac)": "{:.1f}", f"{latest} production (kt)": "{:,.0f}", "M bu": "{:.1f}",
                                      "vs StatCan model": "{:+.1%}"}, na_rep="—"), use_container_width=True, hide_index=True)
    st.selectbox(f"Use this estimate for the {latest} crop in the balance below", [e["label"] for e in ests],
                 key=_k(crop, "est"), on_change=_apply_estimate, args=(crop,))

    fs = on.farm_stocks(crop)
    st.dataframe(pd.DataFrame({"Ontario farm stocks": fs["date"], "kt": fs["kt"], "A year earlier": fs["year_ago"],
                               "Prior 5-yr average": fs["avg5"], "vs a year ago": fs["vs_year_ago"],
                               "vs 5-yr average": fs["vs_avg5"]}).style.format(
        {"kt": "{:,.0f}", "A year earlier": "{:,.0f}", "Prior 5-yr average": "{:,.0f}", "vs a year ago": "{:+.0%}",
         "vs 5-yr average": "{:+.0%}"}), use_container_width=True, hide_index=True)
    mar = fs[fs["date"].str.startswith("Mar")].iloc[0]
    st.caption(f"StatCan farm stocks (table 32-10-0007-01, through {fs['date'].iloc[-1]}): Ontario {crop_label.lower()} on farms at "
               f"{mar['date']} were {mar['kt']:,.0f} kt, {mar['vs_year_ago']:+.0%} on a year earlier and {mar['vs_avg5']:+.0%} on the "
               "prior five-year average. Commercial stocks are not published by province.")
    with st.expander("How the August yield tour has compared with StatCan", expanded=False):
        tt = on.tour_table(crop)
        fin = tt[tt["final"]]
        factor = on.tour_factor(crop)
        mae = (fin["ratio"] / factor - 1).abs().mean()
        st.plotly_chart(of.tour_chart(crop), use_container_width=True, key="on_tour_chart", config={"displayModeBar": False})
        verdict = ("has run high" if factor < 0.97 else "has been close to unbiased")
        st.markdown(f"Across {len(fin)} years ({fin['year'].min()}-{fin['year'].max()}) StatCan's final yield has been **{factor:.2f}x** "
                    f"the tour's (median), so the tour {verdict} for {crop_label.lower()}. After scaling by that ratio the typical miss "
                    f"is about **{mae:.1%}**. The {latest} tour is compared with StatCan's model, not a final figure.")
        st.caption(on.TOUR_SOURCE + ". Yields are the tour's provincial estimate; the tour is a private sample, not a survey.")

    # ── the crop in history ───────────────────────────────────────────────────
    st.markdown(f'<div class="sh">The {latest} crop in history</div>', unsafe_allow_html=True)
    sel_kt = float(st.session_state.get(_k(crop, "p_latest"), p_latest))
    ri = on.rank_info(crop, sel_kt)
    hist = on.history(crop)
    ac_latest = st_latest.get("harvested_ac") or 0.0
    yld_sel = sel_kt * 1000 * on.BU_PER_T[crop] / ac_latest if ac_latest else None
    avg_yld = hist[hist["year"] < latest]["yield_bu"].tail(5).mean()
    st.plotly_chart(of.history_chart(crop, sel_kt), use_container_width=True, key="on_hist_chart", config={"displayModeBar": False})
    st.markdown(f"At {sel_kt:,.0f} kt ({on.to_mbu(crop, sel_kt):,.0f}M bu) the {latest} {noun} crop would be the "
                f"**{_ord(ri['rank'])} largest of {ri['of']}** since 1990: {ri['vs_avg5']:+.1%} on the last five years' average and "
                f"{ri['vs_avg10']:+.1%} on the ten-year average. The record is {ri['max_year']}'s {ri['max_kt']:,.0f} kt."
                + (f" The implied yield is {yld_sel:.1f} bu/ac against {avg_yld:.1f} over the last five years." if yld_sel else ""))

    # ── satellite crop condition ──────────────────────────────────────────────
    st.markdown('<div class="sh">Satellite crop condition</div>', unsafe_allow_html=True)
    vh, vh_err = _vh()
    vy, vw = ovh.last_week(vh)
    if vh_err:
        st.warning(f"NOAA's service could not be reached, so this shows the bundled copy through week {vw} of {vy}.  \n`{vh_err}`",
                   icon="⚠️")
    st.caption(f"NOAA STAR Vegetation Health, Ontario cropland only, weekly, through week {vw} of {vy} (about "
               f"{ovh.week_start(vy, vw):%b %d}). Statistics Canada's own weekly NDVI table (CCAP) stopped in May 2024, so this is the "
               "nearest current series. A province-wide average, not a county or field measure; NOAA labels it experimental.")
    vh_col = st.radio("Index", ["SMN", "VCI", "VHI", "TCI"], horizontal=True, key="on_vh_metric", format_func=lambda c: ovh.LABEL[c])
    st.plotly_chart(of.vh_chart(vh, vh_col, vy), use_container_width=True, key="on_vh_chart", config={"displayModeBar": False})

    wa, wb = ovh.WINDOWS[crop]
    season = ovh.season_table(vh, crop, vy)
    if not season.empty:
        end = ovh.week_start(vy, wb) + pd.Timedelta(days=6)
        st.markdown(f"**{noun.capitalize()} crop window: weeks {wa} to {wb} (about {ovh.week_start(vy, wa):%b %d} to {end:%b %d})**")

        def fmt(col, v):
            return f"{v:.3f}" if col == "SMN" else f"{v:.1f}"
        st.dataframe(pd.DataFrame({
            "Index": [ovh.LABEL[c] for c in season["index"]],
            str(vy): [fmt(c, v) for c, v in zip(season["index"], season["this_year"])],
            "Average": [fmt(c, v) for c, v in zip(season["index"], season["average"])],
            "Rank (1 = best)": [f"{r} of {n}" for r, n in zip(season["rank"], season["of"])]}),
            use_container_width=True, hide_index=True)
        pk, pkw = ovh.peak(vh, vy)
        st.caption(f"Peak NDVI this season: {pk:.3f} in week {pkw}; last year {ovh.peak(vh, vy - 1)[0]:.3f}. Rank compares the same "
                   f"window in every year since {season['years'].iloc[0][:4]}.")

    yields = {int(y): v["yield_bu"] for y, v in on.load()["statcan"][crop].items() if "yield_bu" in v}
    mod = ovh.yield_model(vh, crop, yields, latest)
    if mod["predicted"] is not None:
        ac_now = st_latest.get("harvested_ac") or 0.0
        sat_kt = mod["predicted"] * ac_now / on.BU_PER_T[crop] / 1000.0
        tour = next((e for e in ests if e["label"].startswith("Yield tour, calibrated")), None)
        y1, y2, y3, y4 = st.columns(4)
        y1.metric("Yield trend", f"{mod['trend_year']:.1f} bu/ac", delta_color="off")
        y2.metric("Trend + satellite", f"{mod['predicted']:.1f} bu/ac", f"{sat_kt:,.0f} kt", delta_color="off", delta_arrow="off")
        y3.metric("StatCan model", f"{st_latest.get('yield_bu', 0):.1f} bu/ac", f"{ests[0]['kt']:,.0f} kt", delta_color="off", delta_arrow="off")
        if tour:
            y4.metric("Yield tour (calibrated)", f"{tour['yield_bu']:.1f} bu/ac", f"{tour['kt']:,.0f} kt", delta_color="off", delta_arrow="off")
        order = np.argsort(mod["x"])[::-1]
        top, bottom = mod["anomaly"][order[:10]].mean(), mod["anomaly"][order[-10:]].mean()
        rank_row = season[season["index"] == "SMN"].iloc[0] if not season.empty else None
        st.markdown(
            f"Using only the {vy} satellite reading and the yield trend, Ontario {noun} yield would be about **{mod['predicted']:.1f} "
            f"bu/ac** (trend alone {mod['trend_year']:.1f}), which is {sat_kt:,.0f} kt on StatCan's harvested area. "
            f"**The signal is weak:** over {mod['n']} past years the satellite term cuts the leave-one-out error from "
            f"{mod['rmse_trend_loo']:.1f} to {mod['rmse_loo']:.1f} bu/ac ({mod['skill']:.0%} better), and the window was fixed in advance "
            "rather than searched for the best fit. Years with a top-10 window finished "
            f"{top:+.1f} bu/ac against trend on average, the 10 weakest {bottom:+.1f}. "
            + (f"The {vy} window ranks {rank_row['rank']} of {rank_row['of']}: a good year, not a record, so the satellite does not "
               "back a record-yield call." if rank_row is not None and rank_row["rank"] > 3 else ""))
        st.button("Use the trend + satellite estimate in the balance below", on_click=_set_production, args=(crop, sat_kt),
                  key=_k(crop, "use_sat"), help="Sets the latest crop in the assumptions to this figure.")
        with st.expander("How the satellite reading maps to yield", expanded=False):
            st.plotly_chart(of.vh_scatter(mod, noun.capitalize(), vy), use_container_width=True, key="on_vh_scatter",
                            config={"displayModeBar": False})
            st.caption(f"Each dot is a year: mean NDVI over weeks {wa}-{wb} against that year's StatCan yield minus the linear yield "
                       "trend. NDVI saturates when crops are healthy, so it is better at flagging stress years than at ranking good ones.")
    elif not season.empty:
        st.info("The crop window has not finished yet, so there is no yield cross-check.")

    # ── what US shipments say ─────────────────────────────────────────────────
    if crop == "soy":
        st.markdown('<div class="sh">What US shipments say</div>', unsafe_allow_html=True)
        cyt = ot.crop_years()
        tcy = int(cyt.index.max())
        row, prior = cyt.loc[tcy], cyt.loc[tcy - 1]
        typ = ot.typical(tcy)
        lm = ot.latest_month()
        cy_lab = f"{tcy}/{str(tcy + 1)[2:]}" + ("" if row["complete"] else " to date")
        all_m = ot.monthly()
        late_now = all_m[(all_m["cy"] == tcy) & (all_m["pos"] >= 7)]["ontario"].sum()
        late_prev = all_m[(all_m["cy"] == tcy - 1) & (all_m["pos"] >= 7)]["ontario"].sum()
        ja_all, ja_ont = ot.span_total(f"{tcy + 1}-01", f"{tcy + 1}-08")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric(f"US soybeans to Canada, {cy_lab}", f"{row['total']:,.0f} kt", f"{ot.to_mbu(row['total']):.1f}M bu · "
                  f"{row['total'] / prior['total'] - 1:+.0%} on a year earlier", delta_color="off", delta_arrow="off")
        t2.metric("Through the Ontario gateways", f"{row['ontario']:,.0f} kt", f"{row['share']:.0%} of the total", delta_color="off",
                  delta_arrow="off", help="Customs districts Detroit, Buffalo and Cleveland.")
        t3.metric("Typical (3 prior years)", f"{typ:,.0f} kt", f"{ot.to_mbu(typ):.1f}M bu", delta_color="off", delta_arrow="off")
        t4.metric(f"Latest month ({ot.month_label(lm)})", f"{all_m.iloc[-1]['ontario']:.1f} kt", delta_color="off")
        st.plotly_chart(of.trade_chart(tcy), use_container_width=True, key="on_trade_chart", config={"displayModeBar": False})
        st.markdown(
            f"US soybeans reaching Canada through the Ontario gateways were **{row['ontario']:,.0f} kt** in {cy_lab}, "
            f"{row['ontario'] / typ - 1:+.0%} on the three-year typical of {typ:,.0f} kt: no surge in imports. The year was "
            f"back-loaded, though: April to August ran {late_now:,.0f} kt against {late_prev:,.0f} kt a year earlier, as farm stocks "
            f"ran down. Trade analysts had projected about 500 kt of Ontario imports for January to August; shipments were "
            f"{ja_ont:,.0f} kt, {ja_ont / 500:.0%} of that. That fits the balance below: at the processing run-rate Ontario's own "
            "crop covers its crushers, and imports rise only if processing runs harder. OMAFRA's table has no import line, so typical "
            "imports are probably already inside its processing history and only a change from typical would move the balance. "
            "Census publishes September "
            "around November 5, the first month after Canada's September 8 counter-tariffs. Soybeans are not on that list, so "
            "September is a plain read on demand, not a tariff test.")
        st.caption(f"Census Bureau exports to Canada, HS 1201900005 + 1201900095, by customs district of exit, through {ot.month_label(lm)}. "
                   "Detroit, Buffalo and Cleveland are treated as the Ontario gateways (88% to 99% of the total; Ogdensburg and Pembina "
                   "excluded). Shipments are dated at US exit, not arrival, and the crop year runs September to August, one month off "
                   "the OMAFRA tables.")

    # ── what Ontario needs ────────────────────────────────────────────────────
    st.markdown('<div class="sh">What Ontario needs</div>', unsafe_allow_html=True)
    st.caption("Processors, feed and seed come first, then a carryout; whatever is left can be exported, and a negative "
               "remainder is the import need. Each year's ending stocks open the next.")
    key = lambda n: f"on_{crop}_{n}"                                      # noqa: E731

    def reset() -> None:
        for k in [k for k in st.session_state if k.startswith(f"on_{crop}_")]:
            st.session_state.pop(k)

    with st.expander("Assumptions (recent published averages; edit any)", expanded=False):
        st.button("Reset to recent averages", on_click=reset, key=f"on_reset_{crop}")
        spec = [("begin", f"Beginning stocks, {on.crop_year_label(prev)} (kt)")] + \
               [(k, f"{lab} (kt)") for lab, k in on.LINES[crop]["firm"] if k in a0] + \
               ([("imports", "Typical imports (kt)")] if crop == "corn" else []) + \
               [("exports", "Typical exports (kt)"), ("target", "Carryout target (kt)")]
        cols = st.columns(4)
        a = dict(a0)
        for i, (k, lab) in enumerate(spec):
            a[k] = cols[i % 4].number_input(lab, 0.0, 30000.0, float(round(a0[k])), 25.0, format="%.0f", key=key(k))
        pc = st.columns(4)
        st.session_state.setdefault(key("p_prev"), float(round(p_prev)))
        st.session_state.setdefault(key("p_latest"), float(round(p_latest)))
        prods = {prev: pc[0].number_input(f"{prev} crop (kt)", 0.0, 30000.0, step=25.0, format="%.0f", key=key("p_prev"),
                                          help="StatCan's figure; change it to test a short or big crop."),
                 latest: pc[1].number_input(f"{latest} crop (kt)", 0.0, 30000.0, step=25.0, format="%.0f", key=key("p_latest"),
                                            help="Set by the estimate you pick above; edit it to test your own number.")}
    res = on.chain(crop, (prev, latest), a, a["begin"], prods)

    cards = st.columns(len(res))
    for col, r in zip(cards, res):
        lab = on.crop_year_label(r["year"])
        if r["net"] < 0:
            col.metric(f"{lab}: import need", _kt(r["import_need"]), _mbu(crop, r["import_need"]), delta_color="off", delta_arrow="off",
                       help="Supply falls short of firm domestic use plus the carryout target.")
        else:
            col.metric(f"{lab}: left to export", _kt(r["exportable"]), f"{_mbu(crop, r['exportable'])} · typical {_kt(r['typical_exports'])}",
                       delta_color="off", delta_arrow="off",
                       help="Supply minus firm domestic use minus the carryout target.")

    rows = [("Beginning stocks", "begin"), ("Production", "production")]
    if crop == "corn":
        rows.append(("Imports (typical)", None))
    rows += [("Total supply", "supply"), ("Firm domestic use", "firm"), ("Carryout target", None),
             ("Left after use and carryout", "net"), ("Typical exports", "typical_exports"),
             ("Exports vs typical", None), ("Ending stocks", "ending")]
    table = {}
    for r in res:
        col = []
        for lab, k in rows:
            if lab == "Imports (typical)":
                v = a["imports"]
            elif lab == "Carryout target":
                v = a["target"]
            elif lab == "Exports vs typical":
                v = -r["export_gap"]
            else:
                v = r[k]
            col.append(v)
        table[on.crop_year_label(r["year"])] = col
    st.dataframe(pd.DataFrame(table, index=[lab for lab, _ in rows]).style.format("{:+,.0f}", subset=pd.IndexSlice[
        ["Left after use and carryout", "Exports vs typical"], :]).format("{:,.0f}", subset=pd.IndexSlice[
        [r for r, _ in rows if r not in ("Left after use and carryout", "Exports vs typical")], :]), use_container_width=True)

    r1, r2 = res
    if crop == "soy":
        head = on.headroom(crop, r1, a)
        msg = (f"**{on.crop_year_label(r1['year'])}:** {_kt(r1['production'])} of crop plus {_kt(r1['begin'])} of carryover is "
               f"{_kt(r1['supply'])}. Processors at {_kt(a['milling'])}, feed and waste {_kt(a['feed'])} and a {_kt(a['target'])} "
               f"carryout leave **{r1['net']:+,.0f} kt** ({on.to_mbu(crop, r1['net']):+,.1f}M bu). ")
        msg += (f"Ontario would have to import about {_mbu(crop, r1['import_need'])}. " if r1["net"] < 0 else
                f"Exports would run about {_kt(r1['export_gap'])} below the typical {_kt(a['exports'])}. ")
        msg += (f"Its own crop covers processing up to **{_kt(head)}**; above that, the shortfall comes from imports, "
                f"mostly US beans. **{on.crop_year_label(r2['year'])}:** the larger crop leaves {r2['net']:+,.0f} kt "
                + ("(an import need)." if r2["net"] < 0 else f"and exports recover to about {_kt(r2['exports'])}."))
    else:
        msg = (f"**Corn is in surplus.** {on.crop_year_label(r1['year'])} leaves {_kt(r1['net'])} after food, ethanol, feed and a "
               f"{_kt(a['target'])} carryout, against typical exports of {_kt(a['exports'])}; {on.crop_year_label(r2['year'])} leaves "
               f"{_kt(r2['net'])}. Ontario does not need outside corn; the question is where its surplus goes, and the "
               "ethanol plants and feed trade are the demand to watch.")
    st.markdown(msg)

    # ── the latest crop on past demand ────────────────────────────────────────
    st.markdown(f'<div class="sh">The {latest} crop on past demand</div>', unsafe_allow_html=True)
    bc = on.backcast(crop, prods[latest])
    st.caption("The same crop run through each past crop year's own opening stocks, firm domestic use and carryout: what it "
               "would have left to export then. It separates how much of the picture is crop size and how much is demand.")
    bc_df = pd.DataFrame({"Crop year": bc["label"], "Opening stocks": bc["begin"], "Firm domestic use": bc["firm"],
                          "Carryout": bc["ending"], f"Exports a {prods[latest]:,.0f} kt crop supports": bc["implied_exports"],
                          "Exports that year": bc["actual_exports"], "Carryout if exports as then": bc["carryout_same_exports"]})
    st.dataframe(bc_df.style.format("{:,.0f}", subset=[c for c in bc_df.columns if c != "Crop year"]),
                 use_container_width=True, hide_index=True)
    imp, act = bc["implied_exports"], bc["actual_exports"]
    msg = (f"Through the demand of {bc['label'].iloc[0]} to {bc['label'].iloc[-1]}, a {prods[latest]:,.0f} kt {noun} crop "
           f"would have supported **{imp.min():,.0f} to {imp.max():,.0f} kt of exports (median {imp.median():,.0f} kt)**, against "
           f"{act.min():,.0f} to {act.max():,.0f} kt actually shipped. ")
    if crop == "soy":
        msg += (f"Those years opened with {bc['begin'].min():,.0f} to {bc['begin'].max():,.0f} kt of stocks; the balance above opens "
                f"{on.crop_year_label(r2['year'])} with {r2['begin']:,.0f} kt, and Ontario farm stocks at {mar['date']} were "
                f"{mar['vs_avg5']:+.0%} on average, so the export room is probably smaller than the table suggests.")
    else:
        msg += ("Corn has had room to export in every one of those years; a crop this size adds to it.")
    st.markdown(msg)

    # ── set against New York ──────────────────────────────────────────────────
    if crop == "soy":
        st.markdown('<div class="sh">Against New York</div>', unsafe_allow_html=True)
        need_kt = max(r1["import_need"], r2["import_need"])
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Ontario typical exports", _mbu(crop, a["exports"]), "what its crop used to ship out", delta_color="off", delta_arrow="off")
        n2.metric("Ontario processing", _mbu(crop, a["milling"]), "a year", delta_color="off", delta_arrow="off")
        if ny_soy_bu:
            n3.metric(f"NY soybean crop ({ny_year})", f"{ny_soy_bu / 1e6:,.1f}M bu",
                      f"{ny_conv_bu / 1e6:,.1f}M bu conventional pool", delta_color="off", delta_arrow="off",
                      help="NASS production; the conventional pool is the High-Oleic tab's model.")
        sur = price_usd * tariff * 100
        n4.metric("Canada surtax on US beans", "None" if not tariff else f"{sur:,.0f}¢/bu (what-if)",
                  "not on the Sep 8, 2026 list" if not tariff else f"{tariff:.0%} on ${price_usd:.2f}", delta_color="off",
                  delta_arrow="off", help="Checked 2026-10-27 against the United States Surtax Order (2026): its 335 tariff items "
                                          "(15%, 25%, 50%) include dairy, molasses, doughs, steel and machinery but nothing in chapter "
                                          "12, so US soybeans (1201) pay no surtax. A what-if from the Freight & Spreads tab shows here.")
        if ny_soy_bu:
            ex_bu = on.to_mbu(crop, a["exports"]) * 1e6
            msg = (f"Ontario's typical exports alone are **{ex_bu / ny_soy_bu:.1f}x** New York's whole soybean crop, and each "
                   f"100 kt of Ontario shortfall is {on.to_mbu(crop, 100):.1f}M bu, or {on.to_mbu(crop, 100) * 1e6 / ny_soy_bu:.0%} of it. ")
            if need_kt > 0:
                msg += (f"The modeled import need of {_kt(need_kt)} would be {on.to_mbu(crop, need_kt) * 1e6 / ny_soy_bu:.0%} of New "
                        "York's crop. ")
            typ_imp = ot.typical(ot.crop_years().index.max())
            imp_bu = ot.to_mbu(typ_imp) * 1e6
            msg += (f"Ontario's normal imports of US beans (about {typ_imp:,.0f} kt, {ot.to_mbu(typ_imp):.1f}M bu) equal "
                    f"{imp_bu / ny_soy_bu:.0%} of New York's whole crop"
                    + (f" and {imp_bu / ny_conv_bu:.1f}x its conventional pool" if ny_conv_bu else "")
                    + ", and Buffalo is one of the two main gateways, so New York beans are relevant to that flow. The Midwest "
                    "still supplies most of it. With no surtax on soybeans, a US bean competes with an Ontario one on price, freight "
                    "and the border alone.")
            st.markdown(msg)

    # ── county table for the paired maps' year ────────────────────────────────
    st.markdown("<hr style='border-color:#d7e2dc;margin:10px 0'>", unsafe_allow_html=True)
    tbl_year = int(st.session_state.get("on_pair_year", latest - 1))
    tbl_fc = on.is_forecast(crop, tbl_year)
    with st.expander(f"Ontario county table, {tbl_year}" + (" (forecast allocation)" if tbl_fc else ""), expanded=False):
        ct = on.county_table(crop, tbl_year, float(st.session_state.get(_k(crop, "p_latest"), p_latest)) if tbl_fc else None,
                             st_latest.get("harvested_ac") if tbl_fc else None)
        st.dataframe(ct.assign(Share=(ct["Share"] * 100).round(1), Estimated=ct["Estimated"].map({True: "Est", False: ""}))
                     .round({"Harvested acres": 0, "Yield (bu/ac)": 1, "Production (M bu)": 2, "Production (kt)": 1}),
                     use_container_width=True, hide_index=True,
                     column_config={"Share": st.column_config.NumberColumn("Share of province (%)", format="%.1f")})
        st.caption("Reconciled (or forecast) values as drawn on the map; Est marks the forecast allocation.")

    with st.expander("Demand nodes and what we know about them", expanded=False):
        for kind, meta in on.NODE_KINDS.items():
            items = [n for n in on.NODES if n["kind"] == kind]
            if not items:
                continue
            st.markdown(f"<span style='color:{meta['color']};font-weight:700;'>● {meta['label']}</span>", unsafe_allow_html=True)
            st.markdown("\n".join(f"- **{n['name']}** — {n['town']}" + (" — **idle**" if n.get("status") == "idle" else "")
                                  + f" *({_CONF_TXT[n['conf']]})*  \n  {n['cap']}" + (f" {n['note']}" if n["note"] else "")
                                  for n in items))
        st.caption("Capacity figures come from dated trade reports and company releases; confirm with the plants or the "
                   "Canadian Oilseed Processors Association before relying on them. Most Ontario markers sit at the town, "
                   "not the plant gate; the two New York ethanol plants are placed at their sites.")

    with st.expander("Data notes", expanded=False):
        st.markdown(
            "- **Sources:** Statistics Canada table 32-10-0359-01 (provincial production, including the model-based estimate "
            "released mid-September and the final survey figure due 2026-12-04); OMAFRA supply and disposition and county files on "
            "the Ontario Data Catalogue (Open Government Licence - Ontario); Statistics Canada census division boundaries.\n"
            "- **'Milling'** is OMAFRA's processing line: crushing, food and other mill use, not just Hamilton and Windsor.\n"
            "- **No import line for soybeans** in the OMAFRA file, so the import need here is computed, not observed. Trade "
            "analysts have projected roughly 500 kt of Ontario soybean imports for January to August 2026, which is a useful cross-check.\n"
            "- **Fresher data:** StatCan's September 16 model is the newest official production figure (there is no July "
            "update); farm stocks come from table 32-10-0007-01 (Dec 31, Mar 31 and Jul 31, farm stocks only, no commercial). "
            "The Great Ontario Yield Tour and the Farmtario trade estimate are typed in from published articles and will not "
            "update themselves.\n"
            "- **Import signal:** US Census export data (soybeans to Canada, by customs district); Statistics Canada's provincial "
            "trade tables do not go to soybeans. It measures US shipments, not Ontario's arrivals, and misses any non-US beans.\n"
            "- **Satellite series:** NOAA STAR Vegetation Health (province average over cropland pixels, weekly, 1982 to now; "
            "fetched live, bundled copy as fallback). StatCan's public NDVI tables stopped (32-10-0453-01 archived at 2021-10-04; "
            "32-10-0383-01 ends 2024-05-27), although StatCan's own September model still uses its CCAP series internally.\n"
            "- **Vintages differ:** OMAFRA's table lists 3,840 kt for the 2025 soybean crop; StatCan's current figure is "
            f"{on.production_kt('soy', 2025):,.0f} kt. This page uses StatCan for production and OMAFRA for use.\n"
            "- **County data:** OMAFRA's soybean tonnes column is 14-19% too low for 2023-2025 against its own bushel and yield "
            "columns, so the maps use bushels, which tie to StatCan within about 2%. Published years are reconciled to OMAFRA's regional "
            "subtotals and then to StatCan's provincial total; the forecast year is allocated from each county's olympic-average "
            "share of the last five years. The 2015 sheet's Ontario total row is wrong, so totals come from StatCan.\n"
            "- **Not covered yet:** livestock feed demand by county and canola. The corn feed line in the balance already "
            "carries Ontario's feed use.")
