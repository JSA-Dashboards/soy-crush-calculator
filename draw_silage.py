"""Draw Territory > Silage: how much of New York's corn acreage goes to silage instead of grain, and where.

Silage stays on the farm and feeds the dairy herd, so it is acreage a grain merchandiser cannot originate. NASS reports it
directly (state, district and county), so unlike the high-oleic tab this is observed data: reported counties are used as
published and withheld counties are estimated the way the NASS tab does it. The current year is derived from the grain corn
forecast. Logic is in ny_silage.py, the map in ny_asd_figs.county_fig, the markers are the corn set from ny_ho.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ny_asd_data as nd
import ny_asd_figs as nf
import ny_ho
import ny_silage as si

# radio label -> (county_table column, view_cfg metric key, map title)
VIEWS = {
    "Silage acres": ("Silage acres", "Silage Acres", "Corn silage acres harvested"),
    "Share of corn acres": ("Silage share", "Silage Share of Corn Acres", "Silage as a share of corn acres planted"),
    "Silage tons": ("Silage tons", "Silage Production (tons)", "Corn silage production (tons)"),
    "Silage yield": ("Silage yield", "Silage Yield (tons/ac)", "Corn silage yield (tons per acre)"),
    "Grain corn (bu)": ("Grain bu", "Grain Corn Production (bu)", "Corn for grain production (bu)"),
    "Milk cows": ("Milk cows", "Milk Cows (Jan 2023)", "Milk cows by county (Jan 2023)"),
}


def _kac(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v / 1e3:,.0f}K ac"


def _history_chart(h: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    pat = ["/" if f else "" for f in h["forecast"]]                  # the forecast year is hatched
    xs = list(h["year"].astype(str) + np.where(h["forecast"], " (fcst)", ""))
    for key, name, color in (("grain", "Corn for grain", "#5e7164"), ("silage", "Corn for silage", "#d97706"),
                             ("other", "Neither (abandoned, grazed, failed)", "#cbd5cf")):
        fig.add_trace(go.Bar(x=xs, y=h[key] / 1e3, name=name,
                             marker=dict(color=color, line=dict(width=0),
                                         pattern=dict(shape=pat, fgcolor="rgba(255,255,255,0.6)", bgcolor=color)),
                             hovertemplate=name + ": %{y:,.0f}K ac<extra></extra>"))
    fig.update_layout(barmode="stack", height=360, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE,
                      font=dict(color=nf.TEXT, family="Arial"), margin=dict(l=50, r=10, t=44, b=60),
                      title=dict(text="New York corn acres planted, by use (K acres)", font=dict(size=14, color=nf.ACCENT)),
                      legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=11)),
                      xaxis=dict(type="category", tickmode="array", tickvals=xs, tickfont=dict(color=nf.MUTED, size=10),
                                 gridcolor=nf.BORDER),
                      yaxis=dict(tickformat=",.0f", gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)))
    return fig


def render(geo: dict, corn: nd.NYCrop, fips_map: dict | None = None) -> None:
    sil, meta = si.build(corn, fips_map)
    if sil is None:
        st.warning("The silage data file (ny_silage_snapshot.json) is missing. Build it with build_ny_silage.py.")
        return
    st.markdown(
        "<div class='callout'><strong>Silage corn never reaches the grain market.</strong> It is chopped and fed on the dairy farm, so "
        "the corn acres NASS counts as planted split into grain (what a merchandiser can originate) and silage. Roughly half of "
        "New York's corn is silage, concentrated where the cows are. This tab shows how much, where, and what is left as grain.</div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns([1, 4])
    year = c1.selectbox("Year", list(nd.YEARS), index=1, key="ny_sil_year",
                        format_func=lambda y: f"{y} (forecast)" if si.is_forecast(sil, y) else str(y))
    view = c2.radio("Map", list(VIEWS), horizontal=True, key="ny_sil_view")
    col, metric, title = VIEWS[view]
    is_fc = si.is_forecast(sil, year)
    ylab = f"{year} forecast" if is_fc else str(year)

    corn_y = sil_y = None
    if is_fc:
        if not meta.get("fc_acres"):
            st.warning(f"NASS has no {year} corn planted and grain-harvested forecast yet, so {year} silage cannot be derived.")
            return
        yc = corn.default_yield_sources()
        corn_y = float(st.session_state.get("ny_fc_yield_Corn") or yc["usda"] or yc["ly"] or 0.0)
        d = si.default_yield(sil) or 18.0
        f1, _f2 = st.columns([1.3, 3])
        sil_y = f1.number_input("Silage yield est. (tons/ac)", min_value=0.0, max_value=60.0, value=float(d), step=0.1,
                                format="%.1f", key="ny_sil_yield",
                                help="Default = the olympic average of the last five NASS state silage yields.")
        st.info(f"**{year} silage forecast (derived).** NASS has no {year} silage estimate yet, so silage acres = corn planted "
                f"{meta['planted_fc'] / 1e3:,.0f}K − corn harvested for grain {meta['grain_fc'] / 1e3:,.0f}K (NASS forecast) − the "
                f"typical {meta['other_avg'] / 1e3:,.0f}K that is neither (five-year average) = **{meta['fc_acres'] / 1e3:,.0f}K acres**. "
                f"Tons = acres × the yield above. Each district's figure is split across its counties by five-year shares, the NASS "
                f"tab's method. The grain side uses the corn yield from the NASS Production tab ({corn_y:g} bu/ac).")

    tbl = si.county_table(corn, sil, geo, year, corn_y, sil_y)
    if is_fc:
        pl, gr, sv = meta["planted_fc"], meta["grain_fc"], meta["fc_acres"]
    else:
        pl, gr, sv = corn.state_total("planted", year), corn.state_total("harvested", year), sil.state_total("harvested", year)
    if sv is None or not pl:
        st.warning(f"No New York silage data for {year}.")
        return
    other = max(0.0, pl - (gr or 0.0) - sv)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{ylab} corn planted", _kac(pl), help="All purposes: grain, silage and anything not harvested.")
    k2.metric("Corn for grain", _kac(gr), f"{(gr or 0) / pl:.0%} of acres", delta_color="off", delta_arrow="off")
    k3.metric("Corn for silage", _kac(sv), f"{sv / pl:.0%} of acres", delta_color="off", delta_arrow="off")
    k4.metric("Neither", _kac(other), f"{other / pl:.1%} of acres", delta_color="off", delta_arrow="off",
              help="Planted acres that were neither harvested for grain nor for silage: abandoned, grazed or failed.")

    o1, o2, _o3 = st.columns([2.4, 1.2, 1.4])
    mk = ny_ho.markers_for("Corn")
    show_comp = o1.checkbox(f"Show competition ({mk['comp_label']})", value=True, key="ny_sil_comp")
    show_eth = o2.checkbox("Show ethanol plants", value=True, key="ny_sil_eth")
    comp_cats = mk["cats"] if show_comp and mk["cats"] else None

    cur = tbl[["County", "fips", col, "Estimated"]].rename(columns={col: "Value", "Estimated": "is_estimated"})
    cur = cur.dropna(subset=["Value"])
    if col == "Milk cows":                                 # the herd is a published figure for every county, never an estimate
        cur["is_estimated"] = False
    fig = nf.county_fig(cur, geo, "Corn", year, metric, "Current Year", comp_cats, ethanol=show_eth, forecast=is_fc, label_est=True)
    if fig is None:
        st.info("No county values for this selection.")
    else:
        old = fig.layout.title.text or ""
        sub = old.split("<br>", 1)[1] if "<br>" in old else ""
        fig.update_layout(title_text=f"New York {title}, {ylab}<br>{sub}", dragmode=False)
        st.plotly_chart(fig, use_container_width=True, key="ny_sil_map",
                        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})
    notes = []
    n_est = int(cur["is_estimated"].sum())
    if n_est and not is_fc:
        notes.append(f"* = {n_est} counties NASS withholds, estimated from their share of the district's silage over earlier years "
                     "and scaled to NASS's combined-counties figure.")
    ssum, sstate = tbl["Silage acres"].sum(), sil.state_total("harvested", year) if not is_fc else sv
    if sstate and abs(ssum / sstate - 1) > 0.01:
        notes.append(f"County silage acres add to {ssum / 1e3:,.0f}K against NASS's state figure of {sstate / 1e3:,.0f}K: NASS publishes "
                     "its county and state estimates separately, and a county NASS has never reported cannot be estimated.")
    if col == "Milk cows":
        notes.append("Milk cows are the January 2023 county figures from your notes; silage follows the herd.")
    notes.append("Hover a county for the figure; hover a marker for the site. Silage tons are tons as fed, not bushels.")
    st.caption(" ".join(notes))

    st.markdown('<div class="sh">Corn acres by use, all years</div>', unsafe_allow_html=True)
    st.plotly_chart(_history_chart(si.state_history(corn, sil)), use_container_width=True, key="ny_sil_hist",
                    config={"displayModeBar": False})
    st.caption("Planted minus grain minus silage is the thin grey slice. The silage share has sat near half of New York's corn acres "
               "for a decade; it moves with the dairy herd and with how good the grain crop looks.")

    with st.expander(f"County table, {ylab}", expanded=False):
        show = tbl.sort_values("Silage acres", ascending=False, na_position="last").assign(
            Estimated=lambda d: d["Estimated"].map({True: "Est", False: ""}))
        show = show[["County", "District", "Silage acres", "Silage share", "Silage tons", "Silage yield", "Corn planted",
                     "Grain acres", "Grain bu", "Milk cows", "Estimated"]]
        st.dataframe(show.round({"Silage acres": 0, "Silage share": 1, "Silage tons": 0, "Silage yield": 1, "Corn planted": 0,
                                 "Grain acres": 0, "Grain bu": 0}),
                     use_container_width=True, hide_index=True,
                     column_config={"Silage share": st.column_config.NumberColumn("Share of corn acres (%)", format="%.1f"),
                                    "Silage yield": st.column_config.NumberColumn("Silage yield (tons/ac)", format="%.1f")})

    with st.expander("Data notes", expanded=False):
        st.markdown(
            "- **Source:** USDA NASS QuickStats, New York corn for silage (survey, not the Census of Agriculture): area harvested, "
            "production in tons and yield, at state and county level, 2015 on. Data file last changed "
            f"{meta.get('generated') or 'n/a'}. The JSA droplet re-checks NASS (`build_ny_silage.py`) daily in January-March, when the "
            "state and then county silage estimates appear, and on Tuesdays the rest of the year, and commits the file only when "
            "NASS's figures changed. The Snowflake NASS cache only holds grain corn.\n"
            "- **Grain and planted acres** come from the same corn data as the NASS tab (planted is all purposes).\n"
            "- **Withheld counties** are estimated the NASS tab's way. County estimates for the earliest year (2015) cannot be filled "
            "in because there is no earlier year to take shares from.\n"
            "- **The current year is derived**, not published: silage = planted - grain harvested - the typical acreage that is "
            "neither. Check it against NASS's silage estimate in the January Crop Production report.\n"
            "- **Share of corn acres** is silage acres over all-purpose planted acres, capped at 100% because the two are estimated "
            "separately.")
