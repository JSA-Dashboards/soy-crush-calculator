"""Plotly figures for the Ontario sub-tab (no Streamlit import)."""
import math

import pandas as pd
import plotly.graph_objects as go

import ny_asd_figs as nf
import ontario as on

VIEW = {"lon": [-83.9, -73.9], "lat": [41.6, 46.5]}      # southern Ontario, with the New York side for context
LINE = dict(color="#8fa6bb", width=1.1)


def _ratio() -> float:
    def merc(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return (merc(VIEW["lat"][1]) - merc(VIEW["lat"][0])) / math.radians(VIEW["lon"][1] - VIEW["lon"][0])


def map_height(width: int = 960) -> int:
    return int(width * _ratio()) + 64 + 120


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


def _kt(v):
    if pd.isna(v) or v <= 0:
        return ""
    return f"{v / 1000:.1f}M" if v >= 1000 else (f"{v:.0f}K" if v >= 10 else (f"{v:.1f}K" if v >= 1 else "<1K"))


_LABEL = {"kt": _kt, "ac": nf.fmt_acres, "yd": nf.fmt_yield}


def county_map(df: pd.DataFrame, metric: str, title: str, sub: str, kinds: list, ny_geo: dict | None) -> go.Figure:
    """Ontario census divisions coloured by one metric, with demand nodes and the Du Mond plants."""
    geo = on.load_geo()
    counties = geo["counties"]
    all_ids = [f["id"] for f in counties["features"]]
    val = dict(zip(df["fips"], df[metric]))
    hover = {r["fips"]: (f"<b>{r['County']}</b><br>Production: {r['Production (kt)']:,.0f} kt ({r['Production (M bu)']:.2f}M bu)"
                         f"<br>Harvested: {r['Harvested acres']:,.0f} ac · {r['Yield (bu/ac)']:.1f} bu/ac"
                         f"<br>{r['Share']:.1%} of the province") for _, r in df.iterrows()}
    fig = go.Figure()
    if ny_geo:
        ny_ids = [f["id"] for f in ny_geo["counties"]["features"]]
        fig.add_trace(go.Choropleth(geojson=ny_geo["counties"], featureidkey="id", locations=ny_ids, z=[0] * len(ny_ids),
                                    colorscale=[[0, "#eceeed"], [1, "#eceeed"]], showscale=False,
                                    marker=dict(line=dict(color="#d3d8d5", width=0.8)), hoverinfo="skip"))
    fig.add_trace(go.Choropleth(geojson=counties, featureidkey="id", locations=all_ids, z=[0] * len(all_ids),
                                colorscale=[[0, nf.PANEL], [1, nf.PANEL]], showscale=False, marker=dict(line=LINE),
                                hoverinfo="skip"))
    z = [val.get(f) for f in all_ids]
    nums = [v for v in z if v is not None and not pd.isna(v)]
    fig.add_trace(go.Choropleth(
        geojson=counties, featureidkey="id", locations=all_ids, z=z, zmin=0, zmax=max(nums or [1]), colorscale=nf.SAGE_SEQ,
        marker=dict(line=LINE), text=[hover.get(f, "") for f in all_ids], hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text=metric.replace(" (", "<br>("), font=dict(color=nf.TEXT, size=13)), thickness=22,
                      tickfont=dict(color=nf.TEXT, size=12), len=0.34, x=0.01, xanchor="left", y=0.02, yanchor="bottom",
                      bgcolor="rgba(255,255,255,0.85)")))
    keep = [f for f in all_ids if val.get(f) is not None and not pd.isna(val.get(f))]
    key = "kt" if metric.endswith("(kt)") else ("ac" if "acres" in metric else "yd")
    nf.place_labels(fig, keep, [val[f] for f in keep], geo["centroids"], _LABEL[key])
    for kind in kinds:
        meta = on.NODE_KINDS[kind]
        if kind == "ny":
            pts = [dict(name=f"Du Mond Grain - {f['name']}", town=f["name"], lat=f["lat"], lon=f["lon"], cap="", conf="")
                   for f in nf.FACILITIES]
        else:
            pts = [n for n in on.NODES if n["kind"] == kind]
        if not pts:
            continue
        text = [f"<b>{p['name']}</b><br>{p['town']}" + (f"<br>{nf._wrap(p['cap'])}" if p["cap"] else "")
                + (f"<br><i>Confidence in this entry: {p['conf']}</i>" if p["conf"] else "") for p in pts]
        fig.add_trace(go.Scattergeo(
            lon=[p["lon"] for p in pts], lat=[p["lat"] for p in pts], mode="markers", name=meta["label"],
            marker=dict(symbol=meta["symbol"], size=meta["size"], color=meta["color"],
                        line=dict(color="#7c2d12" if kind == "ny" else "white", width=2 if kind == "ny" else 1.4)),
            text=text, hovertemplate="%{text}<extra></extra>", showlegend=True))
    fig.update_geos(projection_type="mercator", lonaxis_range=VIEW["lon"], lataxis_range=VIEW["lat"], visible=False,
                    bgcolor=nf.DARK, landcolor=nf.LAND, showframe=False, showlakes=False)
    fig.update_layout(**nf._base_layout(f"{title}<br><sup>{sub}</sup>", 18, 64), height=map_height(), dragmode=False,
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
