"""Plotly figures for the Ontario sub-tab (no Streamlit import)."""
import math

import pandas as pd
import plotly.graph_objects as go

import ny_asd_figs as nf
import ontario as on

VIEW = {"lon": [-83.9, -73.9], "lat": [41.6, 46.5]}      # southern Ontario on its own
LINE = dict(color="#8fa6bb", width=1.1)


def _frame(ny_geo: dict | None = None) -> dict:
    """Map frame: southern Ontario alone, or widened south and east to take in New York when the two share a map."""
    if not ny_geo:
        return VIEW
    v = nf.geo_view(ny_geo)
    return {"lon": [VIEW["lon"][0], max(VIEW["lon"][1], v["lon"][1])], "lat": [min(VIEW["lat"][0], v["lat"][0]), VIEW["lat"][1]]}


def _ratio(frame: dict) -> float:
    def merc(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return (merc(frame["lat"][1]) - merc(frame["lat"][0])) / math.radians(frame["lon"][1] - frame["lon"][0])


def map_height(width: int = 960, frame: dict | None = None) -> int:
    return int(width * _ratio(frame or VIEW)) + 64 + 120


def sd_chart(crop: str) -> go.Figure:
    """Disposition by crop year as stacked bars (firm uses, exports, ending stocks) against total supply."""
    df = on.sd_table(crop)
    done = df[df["complete"]]
    lines = on.LINES[crop]
    x = [on.crop_year_label(y) for y in done.index]
    fig = go.Figure()
    stack = [(lab, key) for lab, key in lines["firm"] if key in done and (done[key].fillna(0) > 0).any()]
    stack += [lines["exports"], lines["ending"]]
    for lab, key in stack:
        fig.add_trace(go.Bar(x=x, y=done[key], name=lab, marker_color=on.USE_COLORS.get(key, "#9ca3af"),
                             marker_line_width=0, hovertemplate=f"{lab}: %{{y:,.0f}} kt<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=done["supply"], name="Total supply", mode="lines+markers",
                             line=dict(color="#32373c", width=2.2), marker=dict(size=6),
                             hovertemplate="Total supply: %{y:,.0f} kt<extra></extra>"))
    cur = df[~df["complete"]]
    if not cur.empty:
        y = int(cur.index[-1])
        fig.add_trace(go.Scatter(x=[on.crop_year_label(y)], y=[cur["supply"].iloc[-1]], mode="markers",
                                 name="Supply, year in progress",
                                 marker=dict(size=11, symbol="diamond-open", color="#dc2626", line=dict(width=2.2)),
                                 hovertemplate="Supply so far: %{y:,.0f} kt (use not yet published)<extra></extra>"))
    fig.update_layout(barmode="stack", height=470, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE,
                      font=dict(color=nf.TEXT, family="Arial"), margin=dict(l=60, r=20, t=50, b=120),
                      title=dict(text="Ontario supply and use by crop year (kt, October-September)",
                                 font=dict(size=14, color=nf.ACCENT)),
                      legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", yanchor="top", font=dict(size=11)),
                      xaxis=dict(tickfont=dict(color=nf.MUTED, size=10), gridcolor=nf.BORDER, tickangle=-45),
                      yaxis=dict(tickformat=",.0f", gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)))
    return fig


def _add_ontario(fig: go.Figure, vdf: pd.DataFrame, metric: str, change: str, kinds: list, view: str, zrange: tuple | None,
                 showscale: bool) -> None:
    """Draw Ontario counties (or OMAFRA regions) onto fig, coloured by one metric with the same scales, labels and hover as the NASS
    maps. vdf comes from ontario.county_view: [key, Name, Region, Value, raw, prod_bu, harv, yield, share, is_estimated, ...]."""
    geo = on.load_geo()
    coll = geo["counties"] if view == "County" else geo["regions"]
    cents = geo["centroids"] if view == "County" else geo["region_centroids"]
    all_ids = [f["id"] for f in coll["features"]]
    cfg = nf.view_cfg(metric, change)
    val = dict(zip(vdf["key"], vdf["Value"]))
    hover = {}
    for _, r in vdf.iterrows():
        head = f"<b>{r['Name']}</b>" + (" (Est)" if r["is_estimated"] else "")
        body = (f"<br>{r['Region']}" if view == "County" else "")
        body += (f"<br>Production: {r['prod_bu'] / 1e6:,.2f}M bu · {r['share']:.1%} of Ontario"
                 f"<br>Harvested: {r['harv']:,.0f} ac" + (f" · {r['yield']:.1f} bu/ac" if pd.notna(r["yield"]) else ""))
        if change != "Current Year":
            body += f"<br><b>{change}: {r['Value']:+.1f}%</b>"
        hover[r["key"]] = head + body
    fig.add_trace(go.Choropleth(geojson=coll, featureidkey="id", locations=all_ids, z=[0] * len(all_ids),
                                colorscale=[[0, nf.PANEL], [1, nf.PANEL]], showscale=False, marker=dict(line=LINE),
                                hoverinfo="skip"))
    z = [val.get(f) for f in all_ids]
    nums = [v for v in z if v is not None and not pd.isna(v)]
    z_min, z_max = zrange or nf._z_range(cfg, nums)
    fig.add_trace(go.Choropleth(
        geojson=coll, featureidkey="id", locations=all_ids, z=z, zmin=z_min, zmax=z_max, colorscale=cfg["cscale"],
        marker=dict(line=LINE if view == "County" else dict(color="white", width=2.2)),
        text=[hover.get(f, "") for f in all_ids], hovertemplate="%{text}<extra></extra>", showscale=showscale,
        colorbar=dict(title=dict(text=cfg["clabel"], font=dict(color=nf.TEXT, size=12)), thickness=18,
                      tickfont=dict(color=nf.TEXT, size=11), len=0.34, x=0.01, xanchor="left", y=0.02, yanchor="bottom",
                      bgcolor="rgba(255,255,255,0.85)")))
    for kind in kinds:                                     # plants first, so the value labels draw on top of them
        meta = on.NODE_KINDS[kind]
        for idle in (False, True):                         # idle plants get their own hollow marker and legend entry
            pts = [n for n in on.NODES if n["kind"] == kind and (n.get("status") == "idle") == idle]
            if not pts:
                continue
            text = [f"<b>{p['name']}</b><br>{p['town']}" + ("<br><i>Idle</i>" if idle else "")
                    + (f"<br>{nf._wrap(p['cap'])}" if p["cap"] else "")
                    + (f"<br><i>Confidence in this entry: {p['conf']}</i>" if p["conf"] else "") for p in pts]
            fig.add_trace(go.Scattergeo(
                lon=[p["lon"] for p in pts], lat=[p["lat"] for p in pts], mode="markers",
                name=meta["label"] + (", idle" if idle else ""),
                marker=dict(symbol=meta["symbol"] + ("-open" if idle else ""), size=meta["size"], color=meta["color"],
                            line=dict(color=meta["color"] if idle else "white", width=3 if idle else 1.4)),
                text=text, hovertemplate="%{text}<extra></extra>", showlegend=True))
    keep = [f for f in all_ids if val.get(f) is not None and not pd.isna(val.get(f))]
    if view == "County":
        nf.place_labels(fig, keep, [val[f] for f in keep], cents, cfg["label_fn"])
    else:                                                  # region name over its value, like the district labels in the NASS map
        fig.add_trace(go.Scattergeo(
            lon=[cents[f][0] for f in keep], lat=[cents[f][1] for f in keep], mode="text",
            text=[f"{f.replace(' Ontario', '').upper()}<br>{cfg['label_fn'](val[f])}" for f in keep],
            textfont=dict(color="black", size=14, family="Arial Black"), showlegend=False, hoverinfo="skip"))


def combined_map(vdf: pd.DataFrame, metric: str, change: str, title: str, sub: str, kinds: list, base: go.Figure | None = None,
                 ny_geo: dict | None = None, view: str = "County", zrange: tuple | None = None) -> go.Figure:
    """Ontario drawn onto the New York map (base), so both share one frame, one colour scale and one colour bar. With no base,
    Ontario alone."""
    fig = base if base is not None else go.Figure()
    _add_ontario(fig, vdf, metric, change, kinds, view, zrange, showscale=base is None)
    frame = _frame(ny_geo if base is not None else None)
    fig.update_geos(projection_type="mercator", lonaxis_range=frame["lon"], lataxis_range=frame["lat"], visible=False,
                    bgcolor=nf.DARK, landcolor=nf.LAND, showframe=False, showlakes=False)
    if base is not None:                                   # park the colour bar over Pennsylvania, clear of both maps
        fig.update_traces(colorbar=dict(x=0.31, y=0.02, len=0.24), selector=dict(type="choropleth"))
    fig.update_layout(**nf._base_layout(f"{title}<br><sup>{sub}</sup>", 18, 64), height=map_height(frame=frame), dragmode=False,
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.005, yanchor="top",
                                  bgcolor="rgba(255,255,255,0.9)", bordercolor=nf.BORDER, borderwidth=1,
                                  font=dict(size=13, color=nf.TEXT), itemsizing="constant"))
    fig.update_layout(margin=dict(l=0, r=0, t=64, b=120))
    return fig


def history_chart(crop: str, sel_kt: float, since: int = 1990) -> go.Figure:
    """Ontario production by year (StatCan), with the latest year at the estimate you picked and a ten-year average."""
    h = on.history(crop, since)
    latest = on.latest_statcan_year(crop)
    y = [sel_kt if int(yr) == latest else kt for yr, kt in zip(h["year"], h["kt"])]
    avg10 = h[h["year"] < latest]["kt"].tail(10).mean()
    colors = ["#0693e3" if int(yr) == latest else "#9bb0a2" for yr in h["year"]]
    fig = go.Figure(go.Bar(x=h["year"], y=y, marker_color=colors, marker_line_width=0,
                           hovertemplate="%{x}: %{y:,.0f} kt<extra></extra>"))
    fig.add_hline(y=avg10, line_dash="dash", line_color="#d97706", line_width=1.5,
                  annotation_text=f"Last 10-year average {avg10:,.0f} kt", annotation_position="top left",
                  annotation_font=dict(color="#d97706", size=10))
    fig.update_layout(height=340, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE, font=dict(color=nf.TEXT, family="Arial"),
                      margin=dict(l=60, r=20, t=50, b=40), showlegend=False,
                      title=dict(text=f"Ontario production since {since} (kt): {latest} at the selected estimate",
                                 font=dict(size=14, color=nf.ACCENT)),
                      xaxis=dict(tickfont=dict(color=nf.MUTED), gridcolor=nf.BORDER),
                      yaxis=dict(tickformat=",.0f", gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)))
    return fig


def tour_chart(crop: str) -> go.Figure:
    """August yield-tour forecast against StatCan's yield, year by year."""
    t = on.tour_table(crop)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["year"], y=t["tour"], name="Yield tour (August)", marker_color="#f59e0b", marker_line_width=0,
                         hovertemplate="Tour: %{y:.1f} bu/ac<extra></extra>"))
    fig.add_trace(go.Bar(x=t["year"], y=t["statcan"], name="StatCan (latest year = model)", marker_color="#5e7164",
                         marker_line_width=0, hovertemplate="StatCan: %{y:.1f} bu/ac<extra></extra>"))
    fig.update_layout(barmode="group", height=360, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE,
                      font=dict(color=nf.TEXT, family="Arial"), margin=dict(l=60, r=20, t=50, b=80),
                      title=dict(text="Yield tour vs StatCan yield (bu/ac)", font=dict(size=14, color=nf.ACCENT)),
                      legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center", yanchor="top", font=dict(size=11)),
                      xaxis=dict(tickvals=list(t["year"]), gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)),
                      yaxis=dict(gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED), rangemode="tozero"))
    return fig


# ── satellite crop condition (NOAA STAR) ──────────────────────────────────────
def vh_chart(vh: pd.DataFrame, col: str, year: int) -> go.Figure:
    """One index by week for `year` and the two before it, over the min-max band and average of all earlier years."""
    import ontario_vh as ovh
    m = ovh.matrix(vh, col)
    weeks = [w for w in m.index if 14 <= w <= 44]
    hist = m[[y for y in m.columns if y < year]].loc[weeks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=hist.max(axis=1), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=weeks, y=hist.min(axis=1), mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(94,113,100,0.18)", name=f"Range {hist.columns.min()}-{hist.columns.max()}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=weeks, y=hist.mean(axis=1), mode="lines", name="Average", line=dict(color="#32373c", width=2),
                             hovertemplate="Average: %{y:.3f}<extra></extra>" if col == "SMN" else "Average: %{y:.1f}<extra></extra>"))
    for yr, color, width, dash in ((year - 2, "#9ca3af", 1.6, "dash"), (year - 1, "#f59e0b", 2.0, "solid"), (year, "#0693e3", 3.2, "solid")):
        if yr in m.columns:
            s = m[yr].loc[weeks].where(lambda v: v.index <= vh[vh["year"] == yr]["week"].max())
            fig.add_trace(go.Scatter(x=weeks, y=s, mode="lines", name=str(yr), line=dict(color=color, width=width, dash=dash),
                                     hovertemplate=f"{yr}: " + ("%{y:.3f}" if col == "SMN" else "%{y:.1f}") + "<extra></extra>"))
    ticks = [14, 18, 23, 27, 31, 36, 40, 44]
    fig.update_layout(height=400, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE, font=dict(color=nf.TEXT, family="Arial"),
                      margin=dict(l=60, r=20, t=50, b=90), hovermode="x unified",
                      title=dict(text=f"Ontario cropland, weekly {ovh.LABEL[col]}", font=dict(size=14, color=nf.ACCENT)),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top", font=dict(size=11)),
                      xaxis=dict(tickvals=ticks, ticktext=[ovh.week_start(year, w).strftime("%b %d") for w in ticks],
                                 gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)),
                      yaxis=dict(gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)))
    return fig


def vh_scatter(m: dict, crop: str, year: int) -> go.Figure:
    """Yield above or below trend against the crop-window NDVI, one dot a year, with this year's reading marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["x"], y=m["anomaly"], mode="markers+text", text=[str(y) if y >= 2010 or abs(a) > 8 else "" for y, a in
                                                                                  zip(m["years"], m["anomaly"])],
                             textposition="top center", textfont=dict(size=9, color=nf.MUTED), marker=dict(size=8, color="#5e7164"),
                             hovertemplate="%{customdata}: NDVI %{x:.3f}, %{y:+.1f} bu/ac vs trend<extra></extra>",
                             customdata=m["years"], name="Years"))
    xs = [min(m["x"]), max(m["x"])]
    b = np_polyfit(m["x"], m["anomaly"])
    fig.add_trace(go.Scatter(x=xs, y=[b[0] * x + b[1] for x in xs], mode="lines", line=dict(color="#d97706", dash="dash"), name="Fit"))
    if m["x_now"] is not None:
        fig.add_vline(x=m["x_now"], line_color="#0693e3", line_width=2.5, annotation_text=f"{year}", annotation_font=dict(color="#0693e3"))
    fig.update_layout(height=360, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE, font=dict(color=nf.TEXT, family="Arial"),
                      margin=dict(l=60, r=20, t=50, b=50), showlegend=False,
                      title=dict(text=f"{crop} yield vs trend against the crop-window NDVI", font=dict(size=14, color=nf.ACCENT)),
                      xaxis=dict(title="Mean NDVI over the crop window", gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)),
                      yaxis=dict(title="Yield minus trend (bu/ac)", gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED), zeroline=True))
    return fig


def np_polyfit(x, y):
    import numpy as np
    return np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)


# ── US soybean shipments to Canada ────────────────────────────────────────────
def trade_chart(latest: int) -> go.Figure:
    """Monthly Ontario-gateway shipments for the last three crop years over the average of the five before them."""
    import ontario_trade as ot
    pv, avg, hist = ot.monthly_pivot(latest)
    months = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=[avg.get(i) for i in range(12)], mode="lines", name=f"Median {hist[0]}/{str(hist[0] + 1)[2:]}"
                             f" to {hist[-1]}/{str(hist[-1] + 1)[2:]}", line=dict(color="#32373c", width=2, dash="dash"),
                             hovertemplate="Median: %{y:.1f} kt<extra></extra>"))
    styles = {latest - 2: ("#9ca3af", None), latest - 1: ("#f59e0b", None), latest: ("#0693e3", None)}
    for cy in pv.columns:
        color, _ = styles[cy]
        fig.add_trace(go.Bar(x=months, y=[pv[cy].get(i) for i in range(12)], name=f"{cy}/{str(cy + 1)[2:]}", marker_color=color,
                             marker_line_width=0, hovertemplate=f"{cy}/{str(cy + 1)[2:]}: " + "%{y:.1f} kt<extra></extra>"))
    fig.update_layout(barmode="group", height=380, paper_bgcolor=nf.DARK, plot_bgcolor=nf.SURFACE, font=dict(color=nf.TEXT, family="Arial"),
                      margin=dict(l=60, r=20, t=50, b=80),
                      title=dict(text="US soybeans shipped to Canada through the Ontario gateways, by month (kt)",
                                 font=dict(size=14, color=nf.ACCENT)),
                      legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center", yanchor="top", font=dict(size=11)),
                      xaxis=dict(gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED)),
                      yaxis=dict(gridcolor=nf.BORDER, tickfont=dict(color=nf.MUTED), rangemode="tozero"))
    return fig
