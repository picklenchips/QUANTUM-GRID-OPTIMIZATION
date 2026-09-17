"""
data.climrr_overlay

Climate-hazard overlay for a grid network, from Argonne's **ClimRR** (Climate Risk & Resilience
Portal). Supports the climate-impact-visualization angle in docs/00-VISION.md and the "Climate-data-
overlay tools" entry in docs/07-TOOLING-UPDATES.md.

ClimRR serves ~12 km grid cells over CONUS via an ArcGIS REST server:
  portal   : https://www.anl.gov/ccrds/ClimRR
  services : https://disgeoportal.egs.anl.gov/arcgis/rest/services/Hosted/
  raw data : https://anl.app.box.com/s/hmkkgkrkzxxocfe9kpgrzk2gfc4gizp8  (bulk download, per 04-DATA-SOURCES.md)

Each seasonal Fire Weather Index service (e.g. `Summer_MidCentury_FWI_Average/MapServer/0`) carries
all three periods in one layer:
  Crossmodel          grid-cell id (string)
  {season}_Hist_Mean  historical seasonal-mean FWI
  {season}_Midc_Mean  mid-century (RCP8.5)
  {season}_Endc_Mean  end-century (RCP8.5)
  {season}_Dmid_hist / {season}_Dend_hist   absolute change vs historical
  {season}_Pmid_hist / {season}_Pend_hist   percent change vs historical

NOTE (2026-08): the ClimRR ArcGIS `query`/`identify` operations were returning HTTP 500 during
development -- 07-TOOLING-UPDATES.md already flags "check anl.gov/ccrds/ClimRR directly before
citing specifics". `fetch_climrr_cells()` targets the standard ArcGIS query API (correct when the
server is healthy); the notebook falls back to the bundled fixture
`data/fixtures/climrr/fwi_summer_demo.geojson` (synthetic, schema-faithful -- see make_demo_fixture).
"""
import os, sys, json, math, urllib.parse, urllib.request

_pdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pdir not in sys.path:
    sys.path.append(_pdir)

CLIMRR_REST = "https://disgeoportal.egs.anl.gov/arcgis/rest/services/Hosted"
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "climrr")

# season -> ArcGIS service name (each has Hist/Midc/Endc in one layer); value field prefix = season
CLIMRR_LAYERS = {
    "winter": "Winter_MidCentury_FWI_Average",
    "spring": "Spring_MidCentury_FWI_Average",
    "summer": "Summer_MidCentury_FWI_Average",
    "autumn": "Autumn_MidCentury_FWI_Average",
}
CLIMRR_PERIODS = {"historical": "Hist_Mean", "midcentury": "Midc_Mean", "endcentury": "Endc_Mean"}


def fetch_climrr_cells(bbox: tuple[float, float, float, float], season: str = "summer",
                       timeout: int = 60) -> dict:
    """ bbox = (west, south, east, north) in degrees (WGS84). Returns a GeoJSON FeatureCollection of
    ClimRR grid cells intersecting the bbox, with the FWI fields for `season`. """
    service = CLIMRR_LAYERS[season]
    w, s, e, n = bbox
    fields = ["Crossmodel"] + [f"{season}_{suf}" for suf in CLIMRR_PERIODS.values()]
    params = {
        "geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "where": "1=1", "outFields": ",".join(fields),
        "returnGeometry": "true", "f": "geojson",
    }
    url = f"{CLIMRR_REST}/{service}/MapServer/0/query?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:            # nosec - fixed anl.gov https host
        body = r.read().decode()
    data = json.loads(body)   # raises if the server returned an HTML 500 page
    if "features" not in data:
        raise RuntimeError(f"ClimRR query returned no features: {str(data)[:200]}")
    return data


def _poly_centroid(geom: dict) -> tuple[float, float]:
    """ rough centroid of a GeoJSON Polygon / MultiPolygon (mean of the outer ring vertices). """
    if geom["type"] == "Polygon":
        ring = geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        ring = geom["coordinates"][0][0]
    else:
        return tuple(geom["coordinates"][:2])
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def bus_hazard_scores(net, cells_geojson: dict, season: str = "summer",
                      period: str = "midcentury", assume_latlon: bool = True) -> "list[dict]":
    """ nearest-cell-centroid join of a pandapower net's buses to ClimRR cells.
    Returns [{bus, lon, lat, crossmodel, fwi, fwi_historical, delta}] -- one row per bus that has geodata.
    period: 'historical' | 'midcentury' | 'endcentury'.
    assume_latlon: this repo's converters store bus geodata as [lat, lon]; True (default) swaps to lon/lat. """
    field = f"{season}_{CLIMRR_PERIODS[period]}"
    hist_field = f"{season}_{CLIMRR_PERIODS['historical']}"
    cells = []
    for ft in cells_geojson["features"]:
        cx, cy = _poly_centroid(ft["geometry"])
        cells.append((cx, cy, ft["properties"]))
    if not cells:
        raise ValueError("no ClimRR cells provided")

    rows = []
    for b in net.bus.index:
        geo = net.bus.at[b, "geo"] if "geo" in net.bus.columns else None
        lon = lat = None
        if isinstance(geo, str) and geo:
            try:
                lon, lat = json.loads(geo)["coordinates"]
            except (ValueError, KeyError):
                pass
        elif isinstance(geo, (list, tuple)) and len(geo) == 2:
            lon, lat = geo
        if lon is None:
            continue
        if assume_latlon:
            lon, lat = lat, lon
        cx, cy, props = min(cells, key=lambda c: (c[0] - lon) ** 2 + (c[1] - lat) ** 2)
        fwi = props.get(field)
        hist = props.get(hist_field)
        rows.append({"bus": int(b), "lon": lon, "lat": lat,
                     "crossmodel": props.get("Crossmodel"),
                     "fwi": fwi, "fwi_historical": hist,
                     "delta": (fwi - hist) if (fwi is not None and hist is not None) else None})
    return rows


def make_demo_fixture(path: str | None = None, season: str = "summer",
                      bounds: tuple[float, float, float, float] = (-124.5, 32.5, -114.0, 42.1),
                      step: float = 0.5) -> str:
    """ write a SYNTHETIC but schema-faithful ClimRR FWI GeoJSON grid. Default covers California
    (~0.5 deg cells). For offline notebook/demo use only -- NOT real ClimRR data. FWI follows a
    plausible coast(low, marine)->interior(high, dry) + south(higher) gradient, +15-35% mid-century. """
    w, s, e, n = bounds
    nx = int(round((e - w) / step))
    ny = int(round((n - s) / step))
    feats = []
    for j in range(ny):
        for i in range(nx):
            x0, y0 = w + i * step, s + j * step
            inland = i / max(nx - 1, 1)                 # 0 coast .. 1 interior
            southness = 1.0 - j / max(ny - 1, 1)        # 0 north .. 1 south
            hist = 6.0 + 22.0 * inland + 6.0 * southness
            mid = hist * (1.16 + 0.14 * inland)
            end = hist * (1.30 + 0.22 * inland)
            feats.append({
                "type": "Feature",
                "properties": {
                    "Crossmodel": f"C{j:03d}{i:03d}",
                    f"{season}_Hist_Mean": round(hist, 2),
                    f"{season}_Midc_Mean": round(mid, 2),
                    f"{season}_Endc_Mean": round(end, 2),
                },
                "geometry": {"type": "Polygon", "coordinates": [[
                    [x0, y0], [x0 + step, y0], [x0 + step, y0 + step], [x0, y0 + step], [x0, y0]]]},
            })
    gj = {"type": "FeatureCollection", "_synthetic": True, "features": feats}
    path = path or os.path.join(FIXTURE_DIR, f"fwi_{season}_demo.geojson")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gj, f)
    return path


if __name__ == "__main__":
    fx = make_demo_fixture()
    print("wrote synthetic fixture:", fx)
    bbox = (-122.6, 37.6, -121.5, 38.7)
    try:
        cells = fetch_climrr_cells(bbox, "summer")
        print(f"LIVE ClimRR: {len(cells['features'])} cells")
    except Exception as e:
        print(f"live ClimRR unavailable ({type(e).__name__}: {e}); using fixture")
        with open(fx) as f:
            cells = json.load(f)
    print(f"{len(cells['features'])} cells; sample props:", cells["features"][0]["properties"])
