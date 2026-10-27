"""Draw Territory > Ontario: what Ontario grows, what it uses, and what it might need.

The question behind this page is demand: does Ontario have enough soybeans (or corn) for its own processors, and how big
could its call on outside beans be compared with New York's crop? Logic is in ontario.py, figures in ontario_figs.py.
"""
import pandas as pd
import streamlit as st

import ontario as on
import ontario_figs as of

_SOY_TIP = ("Ontario's own processing ('milling' in OMAFRA's table) has run about 3 Mt a year since 2019/20 and exports "
            "another 1.1-1.3 Mt, so a short crop is felt in exports first and then in imports.")
_CONF_TXT = {"medium": "medium confidence", "low": "low confidence: dated or partial source"}


def _mbu(crop: str, kt: float) -> str:
    return f"{on.to_mbu(crop, kt):,.1f}M bu"


def _kt(v: float) -> str:
    return f"{v:,.0f} kt"


def _k(crop: str, name: str) -> str:
    return f"on_{crop}_{name}"


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _apply_estimate(crop: str) -> None:
    """Selecting a crop estimate sets the latest-year production in the balance's assumptions."""
    picked = {e["label"]: e for e in on.estimates(crop)}[st.session_state[_k(crop, "est")]]
    st.session_state[_k(crop, "p_latest")] = float(round(picked["kt"]))


def render(ny_soy_bu, ny_conv_bu, ny_year, price_usd: float, tariff: float, ny_geo: dict) -> None:
    st.markdown(
        "<div class='callout'><strong>What might Ontario need?</strong> Ontario grows about 4 Mt of soybeans and 10 Mt of corn, "
        "but its processors, feed trade and ethanol plants use a lot of both. This page lays out Ontario's supply and use by "
        "crop year, runs a two-year balance with the assumptions you can change, and sets any shortfall against New York's "
        "crop and Canada's surtax. Data: Statistics Canada and OMAFRA, both open.</div>", unsafe_allow_html=True)

    crop_label = st.radio("Crop", list(on.CROPS), horizontal=True, key="on_crop")
    crop = on.CROPS[crop_label]
    noun = "soybean" if crop == "soy" else "corn"
    a0 = on.defaults(crop)
    latest = on.latest_statcan_year(crop)
    prev = latest - 1
    p_latest, p_prev = on.production_kt(crop, latest), on.production_kt(crop, prev)
    st_latest = on.statcan(crop, latest)

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
        n4.metric("Surtax on US beans", f"{sur:,.0f}¢/bu", f"{tariff:.0%} on ${price_usd:.2f}", delta_color="off",
                  delta_arrow="off", help="Settings from the Freight & Spreads tab. Your notes put soybeans on the surtax list "
                                          "at 25%; I have not confirmed it applies to beans bought for crushing.")
        if ny_soy_bu:
            ex_bu = on.to_mbu(crop, a["exports"]) * 1e6
            msg = (f"Ontario's typical exports alone are **{ex_bu / ny_soy_bu:.1f}x** New York's whole soybean crop, and each "
                   f"100 kt of Ontario shortfall is {on.to_mbu(crop, 100):.1f}M bu, or {on.to_mbu(crop, 100) * 1e6 / ny_soy_bu:.0%} of it. ")
            if need_kt > 0:
                msg += (f"The modeled import need of {_kt(need_kt)} would be {on.to_mbu(crop, need_kt) * 1e6 / ny_soy_bu:.0%} of New "
                        "York's crop. ")
            msg += ("New York can only ever be a marginal supplier to Ontario's crushers: the swing is in the Midwest, and "
                    "the surtax is what separates a US bean from an Ontario one.")
            st.markdown(msg)

    # ── map ───────────────────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#d7e2dc;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Where it is grown, and where it is used</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns([0.7, 1.9, 1.5, 1.8])
    year = m1.selectbox("Year", on.county_years(crop), key="on_map_year")
    metric = m2.radio("Map", ["Production (kt)", "Harvested acres", "Yield (bu/ac)"], horizontal=True, key="on_map_metric")
    scale = m3.checkbox("Scale counties to the StatCan total", value=True, key="on_scale",
                        help="Since 2023 the soybean county rows add up to 14-17% less than the province. This scales them "
                             "up pro rata; it only changes production, not acres or yield.")
    kinds = m4.multiselect("Show", list(on.NODE_KINDS), default=list(on.NODE_KINDS), key="on_nodes",
                           format_func=lambda k: on.NODE_KINDS[k]["label"])
    gap = on.county_gap(crop, year)
    cdf = on.county_table(crop, year, scale)
    sub = "OMAFRA county estimates on census-division boundaries"
    if abs(gap) > on.SCALE_GAP:
        sub += (f" · counties scaled up {gap / (1 - gap):.0%} to reach StatCan's {on.production_kt(crop, year):,.0f} kt"
                if scale else f" · county rows are {gap:.0%} short of StatCan's provincial total")
    fig = of.county_map(cdf, metric, f"Ontario {crop_label.lower()} {year}: {metric.lower()}", sub, kinds, ny_geo)
    st.plotly_chart(fig, use_container_width=True, key="on_map",
                    config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
    st.caption("Hover a marker for the plant and what we know about it. Grey counties across the lake are New York, with the "
               "Du Mond plants as stars. Northern Ontario is off the frame (there is almost no crop there).")
    with st.expander("County table", expanded=False):
        show = cdf.assign(**{"Share": (cdf["Share"] * 100).round(1)}).drop(columns=["fips"])
        st.dataframe(show.round({"Seeded acres": 0, "Harvested acres": 0, "Yield (bu/ac)": 1, "Production (kt)": 1,
                                 "Production (M bu)": 2}), use_container_width=True, hide_index=True,
                     column_config={"Share": st.column_config.NumberColumn("Share of province (%)", format="%.1f")})

    with st.expander("Demand nodes and what we know about them", expanded=False):
        for kind, meta in on.NODE_KINDS.items():
            items = [n for n in on.NODES if n["kind"] == kind]
            if not items:
                continue
            st.markdown(f"<span style='color:{meta['color']};font-weight:700;'>● {meta['label']}</span>", unsafe_allow_html=True)
            st.markdown("\n".join(f"- **{n['name']}** — {n['town']} *({_CONF_TXT[n['conf']]})*  \n  {n['cap']}"
                                  + (f" {n['note']}" if n["note"] else "") for n in items))
        st.caption("Capacity figures come from dated trade reports and company releases; confirm with the plants or the "
                   "Canadian Oilseed Processors Association before relying on them. Markers sit at the town, not the plant gate.")

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
            "- **Vintages differ:** OMAFRA's table lists 3,840 kt for the 2025 soybean crop; StatCan's current figure is "
            f"{on.production_kt('soy', 2025):,.0f} kt. This page uses StatCan for production and OMAFRA for use.\n"
            "- **County gap:** the soybean county rows stopped adding up after 2022; corn county rows tie to StatCan in every year. "
            "The 2015 sheet's Ontario total row is wrong, so totals here come from StatCan.\n"
            "- **Not covered yet:** livestock feed demand by county and canola. The corn feed line in the balance already "
            "carries Ontario's feed use.")
