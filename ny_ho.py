"""New York high-oleic (HO) context: competition registry and a county density model (no Streamlit import).

Source: Kolten's 2026-10-22 research notes (roasters, feed mills, licensed dealers, dairy density, zones).
County-level HO trait data does not exist publicly, so the density is MODELED, not observed: a propensity
score from proximity to HO roasting / feed demand plus dairy density, scaled so the soybean-acre-weighted
state HO share equals the share you set (60% default). Seed-company placement data (under NDA) would replace it.
"""
import math
import re

import numpy as np
import pandas as pd

STATE_HO_SHARE = 0.60          # Kolten, 2026-10-22: closer to 60% than the 33% NYCSGA figure from early 2025
SHARE_CAP = 0.95

CATEGORIES = {
    "commercial": {"label": "Commercial roaster", "color": "#dc2626", "symbol": "diamond", "size": 17},
    "onfarm":     {"label": "On-farm roaster", "color": "#ea580c", "symbol": "circle", "size": 15},
    "feedmill":   {"label": "Feed mill", "color": "#2563eb", "symbol": "square", "size": 15},
    "equipment":  {"label": "Roaster equipment", "color": "#6b7280", "symbol": "triangle-up", "size": 15},
    "crush":      {"label": "Crusher / export terminal", "color": "#7c3aed", "symbol": "pentagon", "size": 18},
    "elevator":   {"label": "Competing elevator", "color": "#92400e", "symbol": "hexagon", "size": 14},
    "dealer":     {"label": "Licensed dealer (soybeans)", "color": "#0d9488", "symbol": "circle-open", "size": 13},
}

# pull = assumed weight (0-3) of an HO demand node in the density model; 0 = not an HO demand node.
# prec: address | town | county (how exactly the marker is placed). status: active | planned | unconfirmed.
COMPETITORS = [
    # Commercial roasters
    dict(name="Edgewood Grain (Craig Phelps)", cat="commercial", town="Groveland", county="Livingston", lat=42.699, lon=-77.7914,
         prec="town", pull=3.0, note="~650,000 bu this year to ~25 dairies + 1 feed mill. Doubled contracted acres for 2026 at $1 over the board."),
    dict(name="Cohocton Valley Mills (Edgewood)", cat="commercial", town="Cohocton", county="Steuben", lat=42.5023, lon=-77.5072,
         prec="town", pull=0.3, note="Edgewood's organic-grain intake point."),
    dict(name="Linwood Commodities (Edgewood)", cat="commercial", town="Linwood (Town of York)", county="Livingston", lat=42.8968, lon=-77.9475,
         prec="town", pull=0.5, note="Edgewood's brokerage arm; another western Finger Lakes intake point."),
    dict(name="Weiler's Grain Roasting", cat="commercial", town="Seneca Falls", county="Seneca", lat=42.9107, lon=-76.7968,
         prec="town", pull=1.5, note="Custom roasting; also active in the on-farm roaster equipment space."),
    # On-farm roasters
    dict(name="Half Full Dairy", cat="onfarm", town="Warners (Town of Van Buren)", county="Onondaga", lat=43.0853, lon=-76.3291,
         prec="town", pull=2.0, note="3,500 cows; also runs western NY dairies; high-oleic since 2018."),
    dict(name="Elmer Richards & Sons", cat="onfarm", town="Skaneateles", county="Onondaga", lat=42.947, lon=-76.4291,
         prec="town", pull=1.5, note="2,700 cows; 100% high-oleic since ~2019."),
    dict(name="Dennis Underwood", cat="onfarm", town="central NY (site not given)", county="", lat=None, lon=None,
         prec="none", pull=0.0, note="1,700 cows; grows 650 acres and buys from 3 neighbors at a premium noted as 'one-seventy' (confirm $/bu). Not on the map."),
    # Feed mills
    dict(name="CNY Feeds (Karl Bitz)", cat="feedmill", town="Jordan", county="Onondaga", lat=43.0653, lon=-76.473,
         prec="town", pull=3.0, note="Largest independent dairy mill in NY: >125,000 cows/day, >90 farms, 100+ ingredients."),
    dict(name="CNY Feeds - new Avon mill", cat="feedmill", town="Avon (William J. Bruckel Dr)", county="Livingston", lat=42.8666, lon=-77.7121,
         prec="address", pull=1.2, status="planned", note="Breaking ground; opens summer 2027. Dead centre in the western soybean belt."),
    # Equipment channel
    dict(name="A.N. Martin Systems (SoyRoaster)", cat="equipment", town="Clyde", county="Wayne", lat=43.0842, lon=-76.8694,
         prec="town", pull=0.5, note="On-farm electric roasters, 3,000-20,000 lb/day. Not a bean buyer, but every unit sold is a farm that needs raw HO delivered."),
    # Crushers / export
    dict(name="York Milling & Grain", cat="crush", town="Livingston County (site not published)", county="Livingston", lat=42.7248, lon=-77.7796,
         prec="county", pull=1.0, status="unconfirmed",
         note="Extrusion/expeller (not solvent). $225,000 Empire State Development grant; target completion Jan 2026; sources western NY growers, sells meal and oil to nearby feed mills. Wildcard: pulls beans from the Livingston-Genesee-Wyoming core."),
    dict(name="Bunge Hamilton (Ontario, Canada)", cat="crush", town="Hamilton, ON (515 Victoria Ave N)", county="", lat=43.2702, lon=-79.8495,
         prec="address", pull=0.0, note="~3,000 tonnes/day soy or canola; one of 3 oilseed crushers in eastern Canada. Canada's 25% surtax on US soybeans (as of 2026-09-08) is the hurdle. Peace Bridge / Queenston-Lewiston."),
    dict(name="DeLong - Port of Oswego Grain Export Center", cat="crush", town="Oswego", county="Oswego", lat=43.4647, lon=-76.5102,
         prec="address", pull=0.0, note="Leased July 2024: 750,000 bu storage, ship loading, rail, USDA lab. Vessels 8,000-22,000 mt or containers to Asia. HO program open; #1 US containerized ag exporter."),
    dict(name="Perdue Agribusiness - Bainbridge, PA", cat="crush", town="Bainbridge, PA", county="", lat=40.0909, lon=-76.6675,
         prec="town", pull=0.0, note="~20 million bu/yr crush; the closest Perdue plant, ~3 hrs from the southern tier. Off the NY map (south)."),
    dict(name="Perdue Agribusiness - Lancaster Co., PA", cat="crush", town="Lancaster, PA", county="", lat=40.038, lon=-76.3057,
         prec="town", pull=0.0, note="17.5 million bu/yr; said its draw area includes New York when it opened. Off the NY map (south)."),
    dict(name="Perdue entity - Livonia", cat="crush", town="Livonia", county="Livingston", lat=42.8215, lon=-77.6686,
         prec="town", pull=0.0, status="unconfirmed", note="Perdue entity registered in Livonia; suggests receiving activity, no crush operation confirmed."),
    # DeLong origination
    dict(name="DeLong - Geneva elevator", cat="elevator", town="Geneva", county="Ontario", lat=42.869, lon=-76.9786, prec="town", pull=0.0,
         note="DeLong-owned elevator."),
    dict(name="DeLong - Manchester elevator", cat="elevator", town="Manchester", county="Ontario", lat=42.9694, lon=-77.2297, prec="town", pull=0.0,
         note="DeLong-owned elevator."),
    dict(name="DeLong - Clifton Springs elevator", cat="elevator", town="Clifton Springs", county="Ontario", lat=42.9617, lon=-77.14, prec="town", pull=0.0,
         note="DeLong-owned elevator."),
    dict(name="DeLong - Sangerfield elevator", cat="elevator", town="Sangerfield", county="Oneida", lat=42.914, lon=-75.3791, prec="town", pull=0.0,
         note="DeLong-owned elevator."),
    dict(name="DeLong - leased space, York", cat="elevator", town="York", county="Livingston", lat=42.8712, lon=-77.8853, prec="town", pull=0.0,
         note="DeLong leases space in York, the same hamlet as the Du Mond York elevator."),
    dict(name="DeLong - leased space, Phelps", cat="elevator", town="Phelps", county="Ontario", lat=42.9576, lon=-77.0575, prec="town", pull=0.0,
         note="DeLong leases space in Phelps."),
    # Licensed farm product dealers handling soybeans (NYS dealer directory)
    dict(name="Weber Farms", cat="dealer", town="Ontario", county="Wayne", lat=43.2209, lon=-77.283, prec="town", pull=0.0, note="Corn, soybeans, wheat."),
    dict(name="Sheppard Grain", cat="dealer", town="Phelps", county="Ontario", lat=42.9576, lon=-77.0575, prec="town", pull=0.0, note="Organic and conventional soybeans."),
    dict(name="Howlett Holdings", cat="dealer", town="Avon", county="Monroe (as listed)", lat=42.912, lon=-77.7455, prec="town", pull=0.0,
         note="Full grain line. Directory lists Monroe County; Avon village is in Livingston County."),
    dict(name="CNY Feeds (dealer licence)", cat="dealer", town="Elbridge", county="Onondaga", lat=43.0345, lon=-76.448, prec="town", pull=0.0, note="Dealer licence, Elbridge."),
    dict(name="Louis Gale & Son", cat="dealer", town="Waterville", county="Oneida", lat=42.9312, lon=-75.3799, prec="town", pull=0.0, note="Licensed soybean dealer."),
    dict(name="Lutz Feed", cat="dealer", town="Oneonta", county="Otsego", lat=42.4535, lon=-75.063, prec="town", pull=0.0, note="Licensed soybean dealer."),
    dict(name="Brown's Feed (Route 20)", cat="dealer", town="Route 20 (town not given)", county="Herkimer", lat=43.4205, lon=-74.9622, prec="county", pull=0.0,
         note="Licensed soybean dealer; marker at the county centre."),
    dict(name="Clifford C Hay / Cold Springs Farm Organics", cat="dealer", town="Sharon Springs", county="Schoharie", lat=42.7959, lon=-74.6171, prec="town", pull=0.0,
         note="Raw and roasted soybeans."),
    dict(name="Andy's Produce", cat="dealer", town="Queensbury", county="Warren", lat=43.3619, lon=-73.6652, prec="town", pull=0.0, note="Licensed soybean dealer."),
    dict(name="B Giambrone", cat="dealer", town="Fulton County (town not given)", county="Fulton", lat=43.1138, lon=-74.4169, prec="county", pull=0.0,
         note="Licensed soybean dealer; marker at the county centre."),
    dict(name="Bunge USA Grain (roasted soybeans licence)", cat="dealer", town="not given", county="", lat=None, lon=None, prec="none", pull=0.0,
         note="Lists roasted soybeans on its dealer licence. Not on the map."),
]

# New York's running corn-ethanol plant, checked 2026-10-27 (EIA, RFA, local news). Drawn on the NASS maps and on the Ontario + New
# York map, and shares its marker with Ontario's ethanol plants. The idle Fulton plant (Volney, Oswego Co.) is left off at Kolten's
# request; see the reference-ny-ethanol-plants memory for its history. conf = how well the entry is sourced.
ETHANOL_STYLE = {"label": "Ethanol plant (corn)", "color": "#ea580c", "symbol": "diamond", "size": 17}
ETHANOL_PLANTS = [
    dict(name="Western New York Energy - Medina", town="Medina (Orleans Co., NY)", lat=43.2112, lon=-78.3661, conf="high",
         cap="~62 million gal/yr of fuel ethanol (EIA, Jan 1 2025), roughly 22 million bu of corn a year at 2.8 gal/bu "
             "(our arithmetic); runs about 30% over its 50 million gal design. The RFA (Oct 2025) calls it New York's "
             "only large-scale plant. Also makes high-purity alcohol after a $38 million expansion.",
         note="The in-state outlet for New York corn."),
]

# Which markers belong to which crop (Kolten, 2026-10-27): beans see crushers, roasters and elevators; corn sees elevators, feed mills
# and ethanol plants. Roaster-equipment makers and licensed dealers are in neither list, and the Du Mond stars always draw.
# comp_label names the competition layer alone; label adds the ethanol plants, which have their own switch on the NASS tab.
CROP_MARKERS = {
    "Soybeans": {"cats": ["commercial", "onfarm", "crush", "elevator"], "ethanol": False,
                 "comp_label": "crushers, roasters, elevators", "label": "crushers, roasters, elevators"},
    "Corn": {"cats": ["feedmill", "elevator"], "ethanol": True,
             "comp_label": "elevators, feed mills", "label": "elevators, feed mills, ethanol plants"},
}


def markers_for(crop_label: str) -> dict:
    """Competition categories (in legend order), whether the ethanol plants show, and the labels, for one crop."""
    m = CROP_MARKERS.get(crop_label, {"cats": [], "ethanol": False, "comp_label": "none for this crop", "label": "none for this crop"})
    return {**m, "cats": [k for k in CATEGORIES if k in m["cats"]]}

# Milk cows by county, January 2023 (head), as given in the notes. Counties not listed are treated as 0
# in the score (e.g. Oswego, Otsego, Yates, Warren were not in the list); Suffolk and Westchester withheld.
DAIRY_COWS = {
    "Cayuga": 43000, "Wyoming": 47000, "St. Lawrence": 35000, "Livingston": 29000, "Genesee": 28000, "Jefferson": 28000,
    "Lewis": 27000, "Ontario": 27000, "Onondaga": 25500, "Washington": 23000, "Steuben": 22500, "Madison": 18500,
    "Oneida": 16500, "Clinton": 16500, "Franklin": 15500, "Chautauqua": 18900, "Cortland": 12200, "Montgomery": 12000,
    "Chenango": 11300, "Erie": 11700, "Herkimer": 10400, "Tompkins": 10300, "Saratoga": 9500, "Allegany": 8300,
    "Wayne": 7400, "Seneca": 7500, "Schuyler": 6900, "Delaware": 6900, "Columbia": 6800, "Schoharie": 5500,
    "Tioga": 5700, "Niagara": 9000, "Broome": 4800, "Orleans": 3600, "Rensselaer": 3800, "Orange": 3400,
    "Dutchess": 1700, "Albany": 1800, "Monroe": 1800, "Essex": 1400, "Fulton": 1400, "Sullivan": 1100,
    "Greene": 200, "Schenectady": 300, "Ulster": 400,
}

# Zones as called in the notes (section six).
ZONES = {
    "blocked": ["Cayuga", "Seneca", "Ontario", "Wayne", "Livingston", "Genesee", "Orleans", "Niagara", "Monroe", "Onondaga"],
    "partial": ["Wyoming", "Steuben"],   # "northern Wyoming and Steuben"
    "open":    ["Jefferson", "Lewis", "St. Lawrence", "Franklin", "Clinton", "Broome", "Tioga", "Cortland", "Tompkins"],
}
ZONE_LABEL = {"blocked": "Blocked (HO core)", "partial": "Partly blocked", "open": "Conventional pool", "other": "Not assessed"}
ZONE_COLOR = {"blocked": "#ef4444", "partial": "#f59e0b", "open": "#16a34a", "other": "#d1d5db"}


def norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def fips_lookup(geo: dict) -> dict:
    return {norm(f["properties"]["NAME"]): f["id"] for f in geo["counties"]["features"]}


def zone_of(name: str) -> str:
    n = norm(name)
    for z, names in ZONES.items():
        if n in {norm(x) for x in names}:
            return z
    return "other"


def haversine_mi(lon1, lat1, lon2, lat2) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


PENDING = ("planned", "unconfirmed")     # sites that are not running yet, or not confirmed to exist


def node_weight(c: dict, include_pending: bool = False, feed_scale: float = 1.0) -> float:
    """Pull a node actually adds to the density model: its assumed weight, scaled for feed mills, and zero for a planned or
    unconfirmed site unless include_pending (a mill that opens in 2027 should not pull on 2025 acres)."""
    if c.get("lat") is None or (c.get("status") in PENDING and not include_pending):
        return 0.0
    return float(c.get("pull", 0) or 0) * (feed_scale if c.get("cat") == "feedmill" else 1.0)


def roaster_pull(centroids: dict, radius_mi: float, include_pending: bool = False, feed_scale: float = 1.0) -> dict:
    """{fips: index}: sum over HO demand nodes of pull x exp(-miles / radius) from each county centroid."""
    nodes = [(c, node_weight(c, include_pending, feed_scale)) for c in COMPETITORS]
    nodes = [(c, w) for c, w in nodes if w > 0]
    out = {}
    for fips, (lon, lat, _area) in centroids.items():
        out[fips] = sum(w * math.exp(-haversine_mi(lon, lat, n["lon"], n["lat"]) / radius_mi) for n, w in nodes)
    return out


def calibrate(acres: dict, score: dict, state_share: float, cap: float = SHARE_CAP) -> dict:
    """County HO share = min(cap, k x score), k solved so the acre-weighted state share = state_share."""
    keys = [k for k in acres if acres[k] > 0]
    total = sum(acres[k] for k in keys)
    if not keys or total <= 0:
        return {}
    target = state_share * total

    def hit(k):
        return sum(acres[f] * min(cap, k * score.get(f, 0.0)) for f in keys)
    top = max(score.get(f, 0.0) for f in keys)
    if top <= 0:
        return {f: 0.0 for f in keys}
    lo, hi = 0.0, 1.0
    while hit(hi) < target and hi < 1e6:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if hit(mid) < target:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    return {f: min(cap, k * score.get(f, 0.0)) for f in keys}


# How much each of your zone calls multiplies a county's score when "weight your zone calls" is 1.
ZONE_PRIOR = {"blocked": 1.0, "partial": 0.85, "open": 0.25, "other": 0.6}


def ho_table(geo: dict, acres_by_fips: dict, bu_by_fips: dict, state_share: float = STATE_HO_SHARE,
             radius_mi: float = 35.0, dairy_weight: float = 0.3, zone_weight: float = 0.5,
             include_pending: bool = False, feed_scale: float = 1.0) -> pd.DataFrame:
    """One row per county: soybean base, dairy, roaster pull, modeled HO share, HO / conventional acres and bushels.
    score = (1 - dairy_weight) x roaster pull + dairy_weight x dairy density, then scaled by your zone call:
    x [(1 - zone_weight) + zone_weight x ZONE_PRIOR[zone]]. Planned / unconfirmed sites add nothing unless include_pending;
    feed_scale multiplies the feed mills' weights (0 = leave them out)."""
    look = fips_lookup(geo)
    names = {f["id"]: f["properties"]["NAME"] for f in geo["counties"]["features"]}
    cows = {look[norm(n)]: v for n, v in DAIRY_COWS.items() if norm(n) in look}
    pull = roaster_pull(geo["centroids"], radius_mi, include_pending, feed_scale)
    pmax = max(pull.values()) or 1.0
    cmax = max(cows.values()) or 1.0
    score = {f: ((1 - dairy_weight) * pull.get(f, 0.0) / pmax + dairy_weight * cows.get(f, 0.0) / cmax)
             * ((1 - zone_weight) + zone_weight * ZONE_PRIOR[zone_of(names[f])]) for f in names}
    share = calibrate(acres_by_fips, score, state_share)
    rows = []
    for f, n in names.items():
        a = float(acres_by_fips.get(f, 0.0) or 0.0)
        bu = float(bu_by_fips.get(f, 0.0) or 0.0)
        s = share.get(f)
        rows.append({
            "fips": f, "County": n, "Zone": zone_of(n), "Soy acres": a, "Soy bu": bu,
            "Dairy cows": cows.get(f), "Roaster pull": pull.get(f, 0.0), "Score": score[f],
            "HO share": s, "HO acres": a * s if s is not None else None,
            "Conv acres": a * (1 - s) if s is not None else None,
            "HO bu": bu * s if s is not None else None,
            "Conv bu": bu * (1 - s) if s is not None else None,
        })
    return pd.DataFrame(rows)
