"""Plotly figure builders for the New York NASS Production view (no Streamlit import).

Port of the NASS tab figures in rma-production-map, drawn from the static NY geometry in
ny_asd_geo.json (county outlines, dissolved ASD district polygons, label centroids).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ny_ho

GEO_FILE = Path(__file__).with_name("ny_asd_geo.json")

# Same tokens as the ASD dashboard
DARK, PANEL, SURFACE = "#f6f8f7", "#ffffff", "#eef3f0"
BORDER, TEXT, MUTED = "#d7e2dc", "#32373c", "#5f7267"
ACCENT, LAND = "#5e7164", "#e1e9e4"
SAGE_SEQ = [[0, "#f0f6f3"], [0.2, "#cfe6da"], [0.55, "#98d0b6"], [1, "#58a784"]]
SAGE_DIV = [[0, "#b22222"], [0.25, "#e08080"], [0.5, "#f5f5f5"], [0.75, "#7dbfa7"], [1, "#1a6645"]]

ASD_PALETTE = {"10": "#0693e3", "20": "#f59e0b", "30": "#ef4444", "40": "#a78bfa", "50": "#16a34a",
               "60": "#fb923c", "70": "#38bdf8", "80": "#e879f9", "90": "#6366f1", "91": "#14b8a6"}


def load_geo() -> dict:
    return json.loads(GEO_FILE.read_text(encoding="utf-8"))


# ── label formatters ──────────────────────────────────────────────────────────
def fmt_bu(v):
    return "" if pd.isna(v) or v == 0 else f"{v / 1_000_000:.1f}"


def fmt_chg(v):
    return "" if pd.isna(v) or abs(v) < 0.05 else f"{v:+.1f}%"


def fmt_acres(v):
    return "" if pd.isna(v) or v < 500 else f"{v / 1_000:,.0f}K"


def fmt_yield(v):
    return "" if pd.isna(v) or v == 0 else f"{v:.0f}"


def fmt_head(v):
    if pd.isna(v) or v == 0:
        return ""
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e4:
        return f"{v / 1e3:.0f}K"
    return f"{v / 1e3:.1f}K" if v >= 1e3 else f"{v:.0f}"


def fmt_count(v):
    return "" if pd.isna(v) or v == 0 else f"{v:,.0f}"


def auto_bu(mx: float) -> tuple:
    if mx >= 500e6:
        return 1e9, "B bu"
    if mx >= 1e6:
        return 1e6, "M bu"
    if mx >= 1e3:
        return 1e3, "K bu"
    return 1, "bu"


def auto_ac(mx: float) -> tuple:
    if mx >= 500e3:
        return 1e6, "M ac"
    if mx >= 1e3:
        return 1e3, "K ac"
    return 1, "ac"


def view_cfg(metric: str, change_view: str) -> dict:
    if change_view != "Current Year":
        return {"cscale": SAGE_DIV, "diverging": True, "clabel": "Change (%)", "hover_fmt": ":+.1f",
                "hover_sfx": "%", "label_unit": "% chg", "label_fn": fmt_chg,
                "rank_unit": "%", "rank_div": 1, "rank_fmt": "+.1f"}
    bu = {"cscale": SAGE_SEQ, "diverging": False, "hover_fmt": ":,.0f", "hover_sfx": " bu", "label_unit": "M bu",
          "label_fn": fmt_bu, "rank_unit": "M bu", "rank_div": 1_000_000, "rank_fmt": ",.2f"}
    ac = {"cscale": SAGE_SEQ, "diverging": False, "hover_fmt": ":,.0f", "hover_sfx": " ac", "label_unit": "K ac",
          "label_fn": fmt_acres, "rank_unit": "K ac", "rank_div": 1_000, "rank_fmt": ",.1f"}
    head = {"cscale": SAGE_SEQ, "diverging": False, "hover_fmt": ":,.0f", "hover_fn": fmt_count}
    return {
        "Inventory (head)": {**head, "clabel": "Inventory<br>(head)", "hover_sfx": " head", "label_unit": "head",
                             "label_fn": fmt_head, "rank_unit": "K head", "rank_div": 1_000, "rank_fmt": ",.1f"},
        "Farms with Inventory": {**head, "clabel": "Farms with<br>inventory", "hover_sfx": " farms", "label_unit": "farms",
                                 "label_fn": fmt_count, "rank_unit": "farms", "rank_div": 1, "rank_fmt": ",.0f"},
        "Head per Farm": {**head, "clabel": "Head per<br>farm", "hover_sfx": " head/farm", "label_unit": "head/farm",
                          "label_fn": fmt_yield, "rank_unit": "head/farm", "rank_div": 1, "rank_fmt": ",.0f"},
        "Total Supply (bu)": {**bu, "clabel": "Total Supply<br>(bu)"},
        "Production (bu)": {**bu, "clabel": "Production<br>(bu)"},
        "Planted Acres": {**ac, "clabel": "Planted<br>Acres"},
        "Harvested Acres": {**ac, "cscale": "BuGn", "clabel": "Harvested<br>Acres"},
        "Prevent Plant Acres": {**ac, "clabel": "Prevent<br>Plant Acres"},
        "Yield (bu/ac)": {"cscale": SAGE_DIV, "diverging": False, "clabel": "Yield<br>(bu/ac)", "hover_fmt": ":.1f",
                          "hover_sfx": " bu/ac", "label_unit": "bu/ac", "label_fn": fmt_yield,
                          "rank_unit": "bu/ac", "rank_div": 1, "rank_fmt": ".1f"},
        "% Harvested": {"cscale": SAGE_DIV, "diverging": False, "clabel": "% Harvested", "hover_fmt": ":.1f",
                        "hover_sfx": "%", "label_unit": "%", "label_fn": fmt_yield,
                        "rank_unit": "%", "rank_div": 1, "rank_fmt": ".1f"},
    }[metric]


def _base_layout(title: str, size: int = 15, top: int = 50) -> dict:
    return dict(paper_bgcolor=DARK, plot_bgcolor=SURFACE, font=dict(color=TEXT, family="Arial"),
                title=dict(text=title, font=dict(size=size, color=ACCENT)), margin=dict(l=0, r=0, t=top, b=0))


# Du Mond Grain plants, drawn as a star on every map. Union Springs is the 5083 White Road address;
# York is the hamlet of York (Livingston County), as no street address is published for that site.
FACILITIES = (
    {"name": "Union Springs", "lat": 42.8639, "lon": -76.6093, "lab_lon": -76.55, "lab_lat": 43.95},
    {"name": "York", "lat": 42.8673, "lon": -77.8867, "lab_lon": -78.0, "lab_lat": 43.95},
)
_STAR_COLOR, _STAR_EDGE = "#fbbf24", "#7c2d12"
# District labels default to the district centroid; these two sit beside the stars, so they are placed by hand.
LABEL_ANCHORS = {"Western": (-78.55, 43.08), "Central": (-75.5, 43.0)}


def _repel(lon: float, lat: float, r: float):
    """Move a label out to r degrees from any facility star so the star never covers it."""
    k = math.cos(math.radians(43.0))
    for fac in FACILITIES:
        dx, dy = (lon - fac["lon"]) * k, lat - fac["lat"]
        d = math.hypot(dx, dy)
        if d < r:
            if d < 1e-6:
                dx, dy, d = 0.0, 1.0, 1.0
            lon, lat = fac["lon"] + dx * r / d / k, fac["lat"] + dy * r / d
    return lon, lat


def add_facilities(fig, legend: str = "tl") -> None:
    """Du Mond Grain facilities: a large star, with its name on a leader line out over Lake Ontario so it
    never covers map data. Always drawn on top of the map."""
    lx, ly = [], []
    for f in FACILITIES:
        lx += [f["lon"], f["lab_lon"], None]
        ly += [f["lat"], f["lab_lat"] - 0.1, None]
    fig.add_trace(go.Scattergeo(lon=lx, lat=ly, mode="lines", line=dict(color=_STAR_EDGE, width=2),
                                showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(
        lon=[f["lab_lon"] for f in FACILITIES], lat=[f["lab_lat"] for f in FACILITIES], mode="text",
        text=[f["name"] for f in FACILITIES], textposition="middle center",
        textfont=dict(size=15, color=_STAR_EDGE, family="Arial Black"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(
        lon=[f["lon"] for f in FACILITIES], lat=[f["lat"] for f in FACILITIES], mode="markers",
        text=[f["name"] for f in FACILITIES],
        marker=dict(symbol="star", size=30, color=_STAR_COLOR, line=dict(color=_STAR_EDGE, width=2.5)),
        name="Du Mond Grain facility", showlegend=True,
        hovertemplate="<b>Du Mond Grain</b><br>%{text}<extra></extra>"))
    if legend == "bottom":
        fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.005, yanchor="top",
                                      bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1,
                                      font=dict(size=13, color=TEXT), itemsizing="constant"),
                          margin=dict(l=0, r=0, t=56, b=110))
    else:
        fig.update_layout(legend=dict(x=0.01, y=0.985, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.9)",
                                      bordercolor=BORDER, borderwidth=1, font=dict(size=13, color=TEXT)))


def geo_view(geo: dict, pad_lon: float = 0.2, pad_lat: float = 0.12) -> dict:
    """Explicit map frame around New York: lon/lat ranges and the height/width ratio of that frame in a
    Mercator map. Used for both the axes and the figure height, so the map fills its frame."""
    lats, lons = [], []

    def walk(o):
        if isinstance(o, list):
            if o and isinstance(o[0], (int, float)):
                lons.append(o[0])
                lats.append(o[1])
            else:
                for i in o:
                    walk(i)
    for f in geo["counties"]["features"]:
        walk(f["geometry"]["coordinates"])
    lon_r = [min(lons) - pad_lon, max(lons) + pad_lon]
    lat_r = [min(lats) - pad_lat, max(lats) + pad_lat]

    def merc(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    ratio = (merc(lat_r[1]) - merc(lat_r[0])) / math.radians(lon_r[1] - lon_r[0])
    return {"lon": lon_r, "lat": lat_r, "ratio": ratio}


def map_height(geo: dict, width: int = 960, top: int = 60) -> int:
    return int(width * geo_view(geo)["ratio"]) + top


def frame_map(fig, geo: dict) -> None:
    v = geo_view(geo)
    fig.update_geos(projection_type="mercator", lonaxis_range=v["lon"], lataxis_range=v["lat"],
                    visible=False, bgcolor=DARK, landcolor=LAND, showframe=False, showlakes=False)


def place_labels(fig, fips_list, values, centroids, label_fn):
    areas = [centroids[f][2] for f in fips_list if f in centroids]
    if areas:
        avg_area = float(np.mean(areas))
        size = int(np.clip(13 + np.log(max(avg_area, 0.01) / 0.05) * 2.0, 13, 19))
    else:
        avg_area, size = 0.1, 13
    cands = []
    for fips, val in zip(fips_list, values):
        label = label_fn(val)
        if label and fips in centroids:
            cx, cy, area = centroids[fips]
            cands.append((area, cx, cy, label))
    cands.sort(reverse=True)
    min_sep = float(np.clip(0.15 + avg_area * 0.8, 0.20, 0.45)) * 1.15
    placed, lons, lats, texts = [], [], [], []
    for area, cx, cy, label in cands:
        if not any((cx - px) ** 2 + (cy - py) ** 2 < min_sep ** 2 for px, py in placed):
            placed.append((cx, cy))
            cx, cy = _repel(cx, cy, 0.2)
            lons.append(cx)
            lats.append(cy)
            texts.append(label)
    if lons:
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, text=texts, mode="text",
                                    textfont=dict(color="#4b5563", size=size, family="Arial Black"),
                                    showlegend=False, hoverinfo="skip"))


def _z_range(cfg, z_vals):
    if cfg["diverging"]:
        m = max(max((abs(v) for v in z_vals if not pd.isna(v)), default=1.0), 1.0)
        return -m, m
    pos = [v for v in z_vals if v and v > 0]
    return (min(pos) if pos else 0), (max(pos) if pos else 1)


# ── competition layer ─────────────────────────────────────────────────────────
def _wrap(text: str, width: int = 52) -> str:
    import textwrap
    return "<br>".join(textwrap.wrap(text, width))


def competition_points(geo: dict, cats=None) -> tuple:
    """(plottable competitors, number off the map frame). cats = category keys to include (None = all)."""
    v = geo_view(geo)
    keep, off = [], 0
    for c in ny_ho.COMPETITORS:
        if c.get("lat") is None or (cats is not None and c["cat"] not in cats):
            continue
        if not (v["lon"][0] <= c["lon"] <= v["lon"][1] and v["lat"][0] <= c["lat"] <= v["lat"][1]):
            off += 1
            continue
        keep.append(c)
    return keep, off


def add_competition(fig, geo: dict, cats=None) -> int:
    """One legend-toggleable trace per competitor category. Returns the number drawn."""
    keep, _ = competition_points(geo, cats)
    n = 0
    for key, meta in ny_ho.CATEGORIES.items():
        pts = [c for c in keep if c["cat"] == key]
        if not pts:
            continue
        hover = []
        for c in pts:
            st = {"planned": " (planned)", "unconfirmed": " (unconfirmed)"}.get(c.get("status", ""), "")
            prec = {"address": "street address", "town": "town centre", "county": "county centre"}.get(c["prec"], "")
            hover.append(f"<b>{c['name']}</b>{st}<br>{meta['label']}<br>{c['town']}"
                         f"{', ' + c['county'] + ' Co.' if c['county'] and 'as listed' not in c['county'] else ''}"
                         f" <i>({prec})</i><br>{_wrap(c['note'])}")
        faded = [0.55 if c.get("status") else 1.0 for c in pts]
        is_open = meta["symbol"].endswith("-open")
        fig.add_trace(go.Scattergeo(
            lon=[c["lon"] for c in pts], lat=[c["lat"] for c in pts], mode="markers", name=meta["label"],
            marker=dict(symbol=meta["symbol"], size=meta["size"], opacity=faded,
                        color=meta["color"], line=dict(color=meta["color"] if is_open else "white", width=2.2 if is_open else 1.4)),
            text=hover, hovertemplate="%{text}<extra></extra>", showlegend=True))
        n += len(pts)
    return n


# ── maps ──────────────────────────────────────────────────────────────────────
def county_fig(df: pd.DataFrame, geo: dict, crop: str, year: int, metric: str, change_view: str, competition=None,
               src: str = "NASS", zrange: tuple | None = None):
    """df: [County, fips, Value, is_estimated]. Estimated counties get an 'Est' label and hover suffix."""
    if df is None or df.empty:
        return None
    cfg = view_cfg(metric, change_view)
    view_label = metric if change_view == "Current Year" else f"{change_view} — {metric}"
    counties = geo["counties"]
    all_fips = [f["id"] for f in counties["features"]]
    val = dict(zip(df["fips"], df["Value"]))
    name = dict(zip(df["fips"], df["County"]))
    est = dict(zip(df["fips"], df["is_estimated"]))
    z, hover = [], []
    for f in all_fips:
        v = val.get(f)
        z.append(v if v is not None and not pd.isna(v) else None)
        lbl = name.get(f) or next((x["properties"]["NAME"] for x in counties["features"] if x["id"] == f), f)
        sfx = " (Est)" if est.get(f) else ""
        hover.append(f"{lbl}{sfx}: {cfg.get('hover_fn', cfg['label_fn'])(v)}{cfg['hover_sfx']}" if v else f"{lbl}: No data")
    z_min, z_max = zrange or _z_range(cfg, [v for v in z if v is not None])
    line = dict(color="#8fa6bb", width=1.1)
    fig = go.Figure()
    fig.add_trace(go.Choropleth(geojson=counties, featureidkey="id", locations=all_fips, z=[0] * len(all_fips),
                                colorscale=[[0, PANEL], [1, PANEL]], showscale=False,
                                marker=dict(line=line), hoverinfo="skip"))
    fig.add_trace(go.Choropleth(
        geojson=counties, featureidkey="id", locations=all_fips, z=z, colorscale=cfg["cscale"],
        zmin=z_min, zmax=z_max, marker=dict(line=line), text=hover, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text=cfg["clabel"], font=dict(color=TEXT, size=13)), tickfont=dict(color=TEXT, size=12),
                      thickness=22, len=0.34, x=0.01, xanchor="left", y=0.02, yanchor="bottom", bgcolor="rgba(255,255,255,0.85)")))
    cents = geo["centroids"]
    est_fips = [f for f in all_fips if est.get(f) and f in cents]
    if est_fips:
        pts = [_repel(cents[f][0], cents[f][1], 0.2) for f in est_fips]
        fig.add_trace(go.Scattergeo(lon=[p[0] for p in pts], lat=[p[1] for p in pts],
                                    mode="text", text=["Est"] * len(est_fips),
                                    textfont=dict(color="#374151", size=11, family="Arial Black"),
                                    showlegend=False, hoverinfo="skip"))
    rep = [f for f in all_fips if val.get(f) is not None and not pd.isna(val.get(f)) and not est.get(f)]
    place_labels(fig, rep, [val[f] for f in rep], cents, cfg["label_fn"])
    if competition:
        add_competition(fig, geo, competition)
    add_facilities(fig, "bottom" if competition else "tl")
    sub = f"Map labels in {cfg['label_unit']}" + ("  ·  Est = county value estimated" if est_fips else "")
    frame_map(fig, geo)
    fig.update_layout(**_base_layout(f"{src} {year} {crop} — {view_label} | New York Counties<br><sup>{sub}</sup>",
                                     18, 64),
                      height=map_height(geo) + (90 if competition else 0))
    if competition:
        fig.update_layout(margin=dict(l=0, r=0, t=64, b=110))
    return fig


def _exteriors(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [p[0] for p in polys]


def district_fig(view_df: pd.DataFrame, raw_df: pd.DataFrame, geo: dict, crop: str, year: int,
                 metric: str, change_view: str, est_districts: set, competition=None, src: str = "NASS",
                 zrange: tuple | None = None):
    """Counties coloured by ASD district value, bold district boundaries, name + value labels."""
    if view_df is None or view_df.empty:
        return None
    cfg = view_cfg(metric, change_view)
    abs_cfg = view_cfg(metric, "Current Year")
    view_label = metric if change_view == "Current Year" else f"{change_view} — {metric}"
    dist_val = dict(zip(view_df["District"], view_df["Value"]))
    raw_map = dict(zip(raw_df["District"], raw_df["Value"])) if raw_df is not None and not raw_df.empty else {}
    dfeats = geo["districts"]["features"]
    districts = [f["properties"]["District"] for f in dfeats]
    z = [dist_val.get(d, 0) for d in districts]

    def hover(d, zv):
        rv = raw_map.get(d)
        s = abs_cfg["label_fn"](rv) if rv is not None else ""
        if change_view != "Current Year" and zv:
            return f"<b>{d}</b><br>{s}<br>{'+' if zv >= 0 else ''}{zv:.1f}%"
        return f"<b>{d}</b><br>{s}"

    pos = [v for v in z if v > 0]
    if cfg["diverging"]:
        m = max(max((abs(v) for v in z), default=1.0), 1.0)
        z_min, z_max = -m, m
    else:
        z_min, z_max = 0, (max(pos) if pos else 1)
    if zrange:
        z_min, z_max = zrange
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geo["districts"], featureidkey="properties.District", locations=districts, z=z,
        colorscale=cfg["cscale"], zmin=z_min, zmax=z_max, marker=dict(line=dict(color=BORDER, width=0.3)),
        colorbar=dict(title=dict(text=cfg["clabel"], font=dict(color=TEXT, size=13)), tickfont=dict(color=TEXT, size=12),
                      thickness=22, len=0.34, x=0.01, xanchor="left", y=0.02, yanchor="bottom", bgcolor="rgba(255,255,255,0.85)"),
        text=[hover(d, v) for d, v in zip(districts, z)], hovertemplate="%{text}<extra></extra>"))
    counties = geo["counties"]
    cfips = [f["id"] for f in counties["features"]]
    fig.add_trace(go.Choropleth(
        geojson=counties, featureidkey="id", locations=cfips, z=[0] * len(cfips),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
        marker=dict(line=dict(color="rgba(90,90,90,0.6)", width=0.8)), hoverinfo="skip"))
    xs, ys, lons, lats, texts = [], [], [], [], []
    for f in dfeats:
        for ring in _exteriors(f["geometry"]):
            xs += [p[0] for p in ring] + [None]
            ys += [p[1] for p in ring] + [None]
        d = f["properties"]["District"]
        dv, rv = dist_val.get(d), raw_map.get(d)
        rv_s = abs_cfg["label_fn"](rv) if rv is not None else ""
        flag = "<br>Est" if d in est_districts else ""
        if change_view != "Current Year" and dv is not None:
            texts.append(f"{d.upper()}{flag}<br>{'+' if dv >= 0 else ''}{dv:.1f}%")
        else:
            texts.append(f"{d.upper()}{flag}<br>{rv_s}")
        lon, lat = LABEL_ANCHORS.get(d) or _repel(f["properties"]["centroid_lon"], f["properties"]["centroid_lat"], 0.42)
        lons.append(lon)
        lats.append(lat)
    fig.add_trace(go.Scattergeo(lon=xs, lat=ys, mode="lines", line=dict(color="white", width=3.2),
                                showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="text", text=texts,
                                textfont=dict(color="black", size=14, family="Arial Black"),
                                showlegend=False, hoverinfo="skip"))
    if competition:
        add_competition(fig, geo, competition)
    add_facilities(fig, "bottom" if competition else "tl")
    frame_map(fig, geo)
    fig.update_layout(**{**_base_layout(f"{src} {year} {crop} — {view_label} | New York AG Districts", 18, 56),
                         "height": map_height(geo) + (90 if competition else 0), "geo": dict(showlakes=False)})
    if competition:
        fig.update_layout(margin=dict(l=0, r=0, t=56, b=110))
    return fig


# ── high-oleic density map ────────────────────────────────────────────────────
HO_VIEWS = ["Est. HO share of soy acres", "Est. HO acres", "Conventional pool (bu)", "Your zones"]
_OR_RD = [[0, "#fff7ec"], [0.25, "#fdd49e"], [0.5, "#fc8d59"], [0.75, "#d7301f"], [1, "#7f0000"]]


def _k_label(v):
    if v is None or pd.isna(v) or v <= 0:
        return ""
    return f"{v / 1e6:.1f}M" if v >= 1e6 else (f"{v / 1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")


def ho_fig(tbl: pd.DataFrame, geo: dict, view: str, year: int, state_share: float, comp_cats=None,
           show_zones: bool = True):
    """County map of the modeled high-oleic density (or the conventional pool / your zones), with zone
    outlines, the competition layer and the Du Mond stars."""
    counties = geo["counties"]
    all_fips = [f["id"] for f in counties["features"]]
    t = tbl.set_index("fips")
    line = dict(color="#8fa6bb", width=1.1)
    fig = go.Figure()
    fig.add_trace(go.Choropleth(geojson=counties, featureidkey="id", locations=all_fips, z=[0] * len(all_fips),
                                colorscale=[[0, PANEL], [1, PANEL]], showscale=False, marker=dict(line=line),
                                hoverinfo="skip"))

    def hover(f):
        r = t.loc[f]
        base = f"<b>{r['County']}</b><br>Your zone: {ny_ho.ZONE_LABEL[r['Zone']]}"
        if r["Soy acres"] and not pd.isna(r["HO share"]):
            return (f"{base}<br>Soy acres: {r['Soy acres']:,.0f}<br>Dairy cows (Jan 2023): "
                    f"{'n/a' if pd.isna(r['Dairy cows']) else format(r['Dairy cows'], ',.0f')}"
                    f"<br>Est. HO share: {r['HO share']:.0%} ({r['HO acres']:,.0f} ac)"
                    f"<br>Conventional: {r['Conv acres']:,.0f} ac / {r['Conv bu'] / 1e3:,.0f}K bu")
        return f"{base}<br>No soybean acres reported"

    texts = [hover(f) for f in all_fips]
    if view == "Your zones":
        order = {"other": 0, "open": 1, "partial": 2, "blocked": 3}
        z = [order[t.loc[f, "Zone"]] for f in all_fips]
        cscale = [[0, ny_ho.ZONE_COLOR["other"]], [0.34, ny_ho.ZONE_COLOR["open"]],
                  [0.67, ny_ho.ZONE_COLOR["partial"]], [1, ny_ho.ZONE_COLOR["blocked"]]]
        fig.add_trace(go.Choropleth(geojson=counties, featureidkey="id", locations=all_fips, z=z, zmin=0, zmax=3,
                                    colorscale=cscale, showscale=False, marker=dict(line=line), text=texts,
                                    hovertemplate="%{text}<extra></extra>"))
        for key in ("blocked", "partial", "open", "other"):
            fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="markers", name=ny_ho.ZONE_LABEL[key],
                                        marker=dict(symbol="square", size=14, color=ny_ho.ZONE_COLOR[key]),
                                        showlegend=True))
        title, sub, label_vals, label_fn = "Your zone calls", "From the 2026-10-22 notes (section six)", None, None
    else:
        col, scale, cb, fmt = {
            "Est. HO share of soy acres": ("HO share", _OR_RD, "Est. HO<br>share (%)", None),
            "Est. HO acres": ("HO acres", _OR_RD, "Est. HO<br>acres", None),
            "Conventional pool (bu)": ("Conv bu", SAGE_SEQ, "Conventional<br>bushels", None),
        }[view]
        vals = {f: t.loc[f, col] for f in all_fips if t.loc[f, "Soy acres"] and not pd.isna(t.loc[f, col])}
        z = [(vals[f] * 100 if view.startswith("Est. HO share") else vals[f]) if f in vals else None for f in all_fips]
        zmax = 100 if view.startswith("Est. HO share") else max([v for v in z if v is not None] or [1])
        fig.add_trace(go.Choropleth(
            geojson=counties, featureidkey="id", locations=all_fips, z=z, zmin=0, zmax=zmax, colorscale=scale,
            marker=dict(line=line), text=texts, hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title=dict(text=cb, font=dict(color=TEXT, size=13)), tickfont=dict(color=TEXT, size=12),
                          thickness=22, len=0.34, x=0.01, xanchor="left", y=0.02, yanchor="bottom",
                          bgcolor="rgba(255,255,255,0.85)")))
        label_fn = (lambda v: f"{v * 100:.0f}%") if view.startswith("Est. HO share") else _k_label
        label_vals = vals
        title = view
        sub = {"Est. HO share of soy acres": f"Modeled, scaled to a {state_share:.0%} state share",
               "Est. HO acres": f"Modeled, scaled to a {state_share:.0%} state share",
               "Conventional pool (bu)": "Soybean bushels not under high-oleic programs (modeled)"}[view]
    cents = geo["centroids"]
    if label_vals:
        keys = [f for f in all_fips if f in label_vals]
        place_labels(fig, keys, [label_vals[f] for f in keys], cents, label_fn)
    if show_zones and view != "Your zones":
        for key, width in (("blocked", 3.6), ("partial", 3.2), ("open", 3.6)):
            locs = [f for f in all_fips if t.loc[f, "Zone"] == key]
            fig.add_trace(go.Choropleth(
                geojson=counties, featureidkey="id", locations=locs, z=[0] * len(locs),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
                marker=dict(line=dict(color=ny_ho.ZONE_COLOR[key], width=width)), hoverinfo="skip"))
            fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="lines", name=f"Outline: {ny_ho.ZONE_LABEL[key]}",
                                        line=dict(color=ny_ho.ZONE_COLOR[key], width=4), showlegend=True))
    if comp_cats:
        add_competition(fig, geo, comp_cats)
    add_facilities(fig, "bottom")
    frame_map(fig, geo)
    fig.update_layout(**_base_layout(f"{title} — {year} soybeans | New York counties<br><sup>{sub}</sup>", 18, 64),
                      height=map_height(geo) + 120)
    fig.update_layout(margin=dict(l=0, r=0, t=64, b=120))
    return fig


# ── freight lane maps ─────────────────────────────────────────────────────────
def _lane_finish(fig, geo: dict, title: str, sub: str, comp_cats) -> go.Figure:
    if comp_cats:
        add_competition(fig, geo, comp_cats)
    add_facilities(fig, "bottom")
    frame_map(fig, geo)
    fig.update_layout(**_base_layout(f"{title}<br><sup>{sub}</sup>", 18, 64), height=map_height(geo) + 120,
                      dragmode=False)
    fig.update_layout(margin=dict(l=0, r=0, t=64, b=120))
    return fig


def territory_fig(geo: dict, win: pd.DataFrame, colors: dict, title: str, sub: str, comp_cats=None) -> go.Figure:
    """Counties coloured by the buyer with the best net-back. win: [fips, key, Buyer, cost_c, hover]."""
    counties = geo["counties"]
    all_fips = [f["id"] for f in counties["features"]]
    line = dict(color="#8fa6bb", width=1.1)
    fig = go.Figure()
    fig.add_trace(go.Choropleth(geojson=counties, featureidkey="id", locations=all_fips, z=[0] * len(all_fips),
                                colorscale=[[0, PANEL], [1, PANEL]], showscale=False, marker=dict(line=line),
                                hoverinfo="skip"))
    for key in win["key"].unique():
        g = win[win["key"] == key]
        col = colors.get(key, MUTED)
        fig.add_trace(go.Choropleth(
            geojson=counties, featureidkey="id", locations=g["fips"], z=[0] * len(g), colorscale=[[0, col], [1, col]],
            showscale=False, marker=dict(line=line, opacity=0.78), text=g["hover"],
            hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="markers", name=f"{g['Buyer'].iloc[0]} ({len(g)})",
                                    marker=dict(symbol="square", size=14, color=col), showlegend=True))
    keep = win[win["cost_c"].notna()]
    place_labels(fig, list(keep["fips"]), list(keep["cost_c"]), geo["centroids"],
                 lambda v: f"{v:.0f}¢" if v < 1000 else f"${v / 100:.1f}")
    return _lane_finish(fig, geo, title, sub, comp_cats)


def lane_value_fig(geo: dict, df: pd.DataFrame, title: str, sub: str, clabel: str, cscale=None,
                   diverging: bool = False, comp_cats=None) -> go.Figure:
    """Counties coloured by one number in cents per bushel. df: [fips, Value, hover]."""
    counties = geo["counties"]
    all_fips = [f["id"] for f in counties["features"]]
    line = dict(color="#8fa6bb", width=1.1)
    val = dict(zip(df["fips"], df["Value"]))
    hov = dict(zip(df["fips"], df["hover"]))
    z = [val.get(f) for f in all_fips]
    nums = [v for v in z if v is not None and not pd.isna(v)]
    zmax = max([abs(v) for v in nums] or [1.0])
    zmin, zhi = (-zmax, zmax) if diverging else (min(nums + [0.0]), zmax)
    fig = go.Figure()
    fig.add_trace(go.Choropleth(geojson=counties, featureidkey="id", locations=all_fips, z=[0] * len(all_fips),
                                colorscale=[[0, PANEL], [1, PANEL]], showscale=False, marker=dict(line=line),
                                hoverinfo="skip"))
    fig.add_trace(go.Choropleth(
        geojson=counties, featureidkey="id", locations=all_fips, z=z, zmin=zmin, zmax=zhi,
        colorscale=cscale or _OR_RD, marker=dict(line=line), text=[hov.get(f, "") for f in all_fips],
        hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text=clabel, font=dict(color=TEXT, size=13)), tickfont=dict(color=TEXT, size=12),
                      thickness=22, len=0.34, x=0.01, xanchor="left", y=0.02, yanchor="bottom",
                      bgcolor="rgba(255,255,255,0.85)")))
    keep = [f for f in all_fips if val.get(f) is not None and not pd.isna(val.get(f))]
    place_labels(fig, keep, [val[f] for f in keep], geo["centroids"],
                 lambda v: f"{v:+.0f}¢" if diverging else (f"{v:.0f}¢" if abs(v) < 1000 else f"${v / 100:.1f}"))
    return _lane_finish(fig, geo, title, sub, comp_cats)


# ── charts ────────────────────────────────────────────────────────────────────
def history_bar(series: dict, years: list, title: str, y_label: str = "", fmt: str = ".2f") -> go.Figure:
    colors = ["#0693e3", "#f59e0b", "#ef4444", "#a78bfa", "#16a34a", "#fb923c", "#e879f9"]
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=years, y=vals, name=name, marker_color=colors[i % len(colors)],
                             marker_line_width=0, text=[f"{v:{fmt}}" for v in vals], textposition="outside",
                             textfont=dict(color=TEXT, size=9), cliponaxis=False))
    layout = _base_layout(title)
    layout.update(height=320, barmode="group",
                  legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(color=TEXT, size=10)),
                  xaxis=dict(tickvals=years, ticktext=[str(y) for y in years], gridcolor=BORDER,
                             tickfont=dict(color=MUTED)),
                  yaxis=dict(title=y_label, gridcolor=BORDER, tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
                  margin=dict(l=60, r=20, t=50, b=30))
    fig.update_layout(**layout)
    return fig


def _rank_layout(fig, title, n, cfg, view_label, l_margin=10):
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=SURFACE, font=dict(color=TEXT, family="Arial"),
        title=dict(text=title, font=dict(size=14, color=ACCENT)),
        height=max(380, n * 22 + 80), margin=dict(l=l_margin, r=90, t=50, b=20), bargap=0.18,
        xaxis=dict(title=f"{view_label} ({cfg['rank_unit']})", gridcolor=BORDER, tickfont=dict(color=MUTED),
                   title_font=dict(color=MUTED), zeroline=True, zerolinecolor=MUTED),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=9), automargin=True))


def ranking_chart(df: pd.DataFrame, crop: str, year: int, metric: str, change_view: str, by_district: bool = False,
                  src: str = "NASS"):
    """Horizontal county ranking; optionally grouped and coloured by ASD district."""
    cfg = view_cfg(metric, change_view)
    view_label = metric if change_view == "Current Year" else f"{change_view} — {metric}"
    d = df.dropna(subset=["Value"]).copy()
    if d.empty:
        return go.Figure()
    fmt = cfg["rank_fmt"]
    raw_avg = d["Value"].mean()
    avg_disp = raw_avg / cfg["rank_div"]
    if by_district:
        d = d.sort_values(["DistrictCode", "Value"]).reset_index(drop=True)
        x = d["Value"] / cfg["rank_div"]
        fig = go.Figure(go.Bar(
            x=x, y=d["County"], orientation="h", marker_color=d["DistrictCode"].map(ASD_PALETTE).fillna(MUTED).tolist(),
            marker_line_width=0, text=[f"{v:{fmt}}" for v in x], textposition="outside",
            textfont=dict(color=TEXT, size=8), cliponaxis=False, customdata=d["District"].fillna("—").tolist(),
            hovertemplate=f"%{{y}} — %{{customdata}}: %{{x:{fmt}}} {cfg['rank_unit']}<extra></extra>"))
        cum = 0
        for code in sorted(d["DistrictCode"].dropna().unique()):
            grp = d[d["DistrictCode"] == code]
            if cum > 0:
                fig.add_hline(y=cum - 0.5, line_color=BORDER, line_width=1.5, line_dash="dot")
            fig.add_annotation(x=0, y=cum + len(grp) / 2 - 0.5, xref="paper", yref="y",
                               text=f"<b>{grp.iloc[0]['District'] or code}</b>",
                               font=dict(color=ASD_PALETTE.get(code, MUTED), size=9), showarrow=False,
                               xanchor="right", xshift=-72)
            cum += len(grp)
        title = f"New York County Rankings by AG District — {crop} {view_label} ({src} {year})"
        margin = 165
    else:
        d = d.sort_values("Value", ascending=True)
        x = d["Value"] / cfg["rank_div"]
        fig = go.Figure(go.Bar(
            x=x, y=d["County"], orientation="h",
            marker_color=[ACCENT if v >= raw_avg else "#e05252" for v in d["Value"]], marker_line_width=0,
            text=[f"{v:{fmt}}" for v in x], textposition="outside", textfont=dict(color=TEXT, size=8),
            cliponaxis=False, hovertemplate=f"%{{y}}: %{{x:{fmt}}} {cfg['rank_unit']}<extra></extra>"))
        title = f"New York County Rankings — {crop} {view_label} ({src} {year})"
        margin = 10
    fig.add_vline(x=avg_disp, line_color="#d97706", line_width=1.5, line_dash="dash",
                  annotation_text=f"  Avg: {avg_disp:{fmt}} {cfg['rank_unit']}", annotation_position="top left",
                  annotation_font=dict(color="#d97706", size=10))
    _rank_layout(fig, title, len(d), cfg, view_label, margin)
    return fig


def _row_bg_text(rank: int, n: int):
    if n == 0:
        return SURFACE, TEXT
    p = rank / max(n - 1, 1)
    if rank == 0:
        return "#fecaca", "#991b1b"
    if rank == 1:
        return "#fee2e2", "#b91c1c"
    if rank == n - 1:
        return "#bbf7d0", "#166534"
    if rank == n - 2:
        return "#dcfce7", "#15803d"
    if p < 0.35:
        return "#fef2f2", TEXT
    if p > 0.65:
        return "#f0fdf4", TEXT
    return SURFACE, TEXT


def heatmap_table(rows: dict, years: list, title: str, unit: str = "M bu", divisor: float = 1e6,
                  is_ratio: bool = False, total: dict | None = None, total_label: str = "NY Total",
                  fmt: str = ".1f", top_row: dict | None = None, top_row_label: str = "",
                  cell_status: dict | None = None, top_row_cell_status: dict | None = None,
                  first_col: str = "Row", ly_label: str = "% vs LY") -> go.Figure:
    """Per-row heat map (top 2 green, bottom 2 red) with % vs LY, olympic average, % of average, min/max
    and share of the total. rows: {label: {year: raw_value_or_None}}."""
    _D, _P, _S, _BR, _T, _M, _A, _G = ("#f5f7fa", "#edf1f7", "#e2e8f2", "#c8d5e3", "#1a2332", "#64748b",
                                      "#0693e3", "#d97706")

    def disp(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v / divisor

    def num(v):
        return "—" if v is None else f"{v:{fmt}}"

    def pct(v):
        return "—" if v is None else f"{'▲' if v >= 0 else '▼'} {abs(v):.1f}%"

    def olympic(vals):
        c = [v for v in vals if v is not None]
        if len(c) >= 4:
            c = sorted(c)[1:-1]
        return sum(c) / len(c) if c else None

    def row_colors(vals):
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        n = len(valid)
        rank_of = {idx: r for r, (idx, _) in enumerate(sorted(valid, key=lambda x: x[1]))}
        bgs, fgs = [], []
        for i in range(len(vals)):
            if i not in rank_of:
                bgs.append(_S)
                fgs.append(_M)
            else:
                bg, fg = _row_bg_text(rank_of[i], n)
                bgs.append(bg)
                fgs.append(fg)
        return bgs, fgs

    labels, data, bg, fg, pinned = [], [], [], [], []
    if top_row and top_row_label:
        labels.append(f"▶  {top_row_label}")
        data.append([disp(top_row.get(y)) for y in years])
        bg.append(["#dbeafe"] * len(years))
        fg.append([_G] * len(years))
        pinned.append(True)
    for lbl in rows:
        vals = [disp(rows[lbl].get(y)) for y in years]
        b, f = row_colors(vals)
        labels.append(lbl)
        data.append(vals)
        bg.append(b)
        fg.append(f)
        pinned.append(False)
    if total:
        vals = [disp(total.get(y)) for y in years]
        b, f = row_colors(vals)
        labels.append(total_label)
        data.append(vals)
        bg.append(b)
        fg.append(f)
        pinned.append(True)
    n_rows = len(labels)
    latest, prev = years[-1], (years[-2] if len(years) >= 2 else None)
    s_ly, s_avg, s_pavg, s_min, s_max, s_share = [], [], [], [], [], []
    bg_ly, fg_ly, bg_pavg, fg_pavg, bg_sh, fg_sh = [], [], [], [], [], []
    for lbl, vals in zip(labels, data):
        clean = [v for v in vals if v is not None]
        cur = vals[years.index(latest)]
        pv = vals[years.index(prev)] if prev is not None else None
        if cur and pv and pv != 0:
            p = (cur - pv) / abs(pv) * 100
            s_ly.append(pct(p))
            bg_ly.append("#dcfce7" if p >= 0 else "#fee2e2")
            fg_ly.append("#166534" if p >= 0 else "#991b1b")
        else:
            s_ly.append("—")
            bg_ly.append(_S)
            fg_ly.append(_M)
        avg = olympic(clean)
        s_avg.append(num(avg))
        if cur and avg and avg != 0:
            pa = (cur - avg) / abs(avg) * 100
            s_pavg.append(pct(pa))
            bg_pavg.append("#dcfce7" if pa >= 0 else "#fee2e2")
            fg_pavg.append("#166534" if pa >= 0 else "#991b1b")
        else:
            s_pavg.append("—")
            bg_pavg.append(_S)
            fg_pavg.append(_M)
        s_min.append(num(min(clean)) if clean else "—")
        s_max.append(num(max(clean)) if clean else "—")
        tot = disp(total.get(latest)) if total else None
        if not is_ratio and tot and tot > 0 and cur:
            s_share.append(f"{cur / tot * 100:.1f}%")
            bg_sh.append(_S)
            fg_sh.append(_G)
        else:
            s_share.append("—")
            bg_sh.append(_S)
            fg_sh.append(_M)

    status = dict(cell_status or {})
    if top_row_cell_status and top_row_label:
        status[f"▶  {top_row_label}"] = top_row_cell_status

    def cell(r, c):
        s = num(data[r][c])
        return f"{s} E" if status.get(labels[r], {}).get(years[c], False) else s

    hdr = [first_col] + [str(y) for y in years] + [ly_label, f"Olympic Avg\n({unit})", "% of Avg", "Min", "Max",
                                                    "% of NY"]
    vals = [labels] + [[cell(r, c) for r in range(n_rows)] for c in range(len(years))] + [s_ly, s_avg, s_pavg, s_min, s_max, s_share]
    fills = ([[(_BR if pinned[i] else _P) for i in range(n_rows)]]
             + [[bg[r][c] for r in range(n_rows)] for c in range(len(years))]
             + [bg_ly, [_S] * n_rows, bg_pavg, [_S] * n_rows, [_S] * n_rows, bg_sh])
    fonts = ([[(_G if pinned[i] else _T) for i in range(n_rows)]]
             + [[fg[r][c] for r in range(n_rows)] for c in range(len(years))]
             + [fg_ly, [_T] * n_rows, fg_pavg, [_T] * n_rows, [_A] * n_rows, fg_sh])
    fig = go.Figure(go.Table(
        columnwidth=[130] + [60] * len(years) + [70, 90, 70, 55, 55, 65],
        header=dict(values=hdr, fill_color=[_P] * len(hdr), font=dict(color=[_M] + [_T] * len(years) + [_M] * 6,
                                                                       size=11, family="Arial"),
                    align="center", height=28, line_color=_BR),
        cells=dict(values=vals, fill_color=fills, font=dict(color=fonts, size=11, family="Arial"),
                   align=["left"] + ["right"] * (len(years) + 6), height=24, line_color=_BR)))
    fig.update_layout(paper_bgcolor=_D, margin=dict(l=0, r=0, t=30, b=0), height=max(150, n_rows * 26 + 70),
                      title=dict(text=title, font=dict(color=_A, size=13)))
    return fig
