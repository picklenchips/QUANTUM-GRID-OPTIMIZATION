"""
data.openinframap_hook

Python-side companion to the web map (web/opengridmap_source.html), which already renders
OpenInfraMap's vector tiles. This module gives the *backend* a way to pull the same OSM power
infrastructure as GeoJSON, and to export a converted pandapower network as GeoJSON so both can be
drawn on one Leaflet/MapLibre map.

Sources (see docs/07-TOOLING-UPDATES.md "OpenInfraMap / OpenGridMap / osm2pgsql"):
  OpenInfraMap tiles : https://openinframap.org/tiles/{z}/{x}/{y}.pbf   (vector, pbf, z2-17)
  OpenInfraMap code  : https://github.com/openinframap   (BSD-3, actively maintained -- last touch Aug 2026)
  Overpass API       : https://overpass-api.de/api/interpreter   (OSM query endpoint)

Why Overpass and not the raw OSM `/map` endpoint that data/osm_to_pp.py uses: Overpass lets us ask
for exactly `power=substation|plant|generator|line|cable` in one request with no bbox-splitting
recursion, and returns geometry inline (`out geom`) so lines already carry their coordinate lists.
"""
import os, sys, json, time
import requests

_pdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pdir not in sys.path:
    sys.path.append(_pdir)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = ("https://overpass-api.de/api/interpreter",
                    "https://overpass.osm.ch/api/interpreter",
                    "https://overpass.private.coffee/api/interpreter")
# Overpass 406s the default python-requests UA; send a real one
_UA = "qpgrid/0.1 (grid-optimization research; https://github.com/picklenchips)"
OPENINFRAMAP_VECTOR_TILES = "https://openinframap.org/tiles/{z}/{x}/{y}.pbf"
# vector-tile source-layers, mirrored from web/openinframap_tile.json
OPENINFRAMAP_POWER_LAYERS = ("power_line", "power_substation", "power_substation_point",
                             "power_plant", "power_plant_point", "power_transformer",
                             "power_generator", "power_switch", "power_tower")
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "openinframap")

_DEFAULT_POWER = ("substation", "plant", "generator", "line", "cable", "minor_line")


def overpass_power_query(bbox: tuple[float, float, float, float], power_values=_DEFAULT_POWER,
                         timeout: int = 60, retries: int = 2) -> dict:
    """ bbox = (south, west, north, east) in degrees (Overpass order!).
    Returns the raw Overpass JSON ({'elements': [...]}) with inline geometry. """
    s, w, n, e = bbox
    clauses = "".join(f'node["power"="{v}"]({s},{w},{n},{e});'
                      f'way["power"="{v}"]({s},{w},{n},{e});' for v in power_values)
    q = f"[out:json][timeout:{timeout}];({clauses});out geom;"
    last = None
    for attempt in range(retries + 1):
        endpoint = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        try:
            r = requests.post(endpoint, data={"data": q},
                              headers={"User-Agent": _UA, "Accept": "application/json"},
                              timeout=timeout + 10)
        except requests.RequestException as ex:
            last = f"{type(ex).__name__}"; time.sleep(2 * (attempt + 1)); continue
        if r.status_code == 200:
            return r.json()
        last = f"HTTP {r.status_code} from {endpoint}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Overpass query failed after {retries + 1} tries: {last}")


def osm_to_geojson(overpass_json: dict) -> dict:
    """ Overpass JSON -> GeoJSON FeatureCollection. Nodes -> Point, ways(with geometry) -> LineString
    (or Polygon if closed and tagged as an area like a substation outline). """
    feats = []
    for el in overpass_json.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node" and "lat" in el:
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and el.get("geometry"):
            coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
            closed = len(coords) > 3 and coords[0] == coords[-1]
            is_area = closed and (tags.get("power") in ("substation", "plant", "generator")
                                  or tags.get("area") == "yes")
            geom = {"type": "Polygon", "coordinates": [coords]} if is_area \
                else {"type": "LineString", "coordinates": coords}
        else:
            continue
        feats.append({"type": "Feature", "id": f"{el['type']}/{el['id']}",
                      "properties": {"power": tags.get("power"), "voltage": tags.get("voltage"),
                                     "name": tags.get("name"), **tags},
                      "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}


def fetch_power_geojson(bbox: tuple[float, float, float, float], cache_path: str | None = None,
                        **kw) -> dict:
    """ Overpass -> GeoJSON, with optional on-disk cache (bbox is (south, west, north, east)). """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    gj = osm_to_geojson(overpass_power_query(bbox, **kw))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(gj, f)
    return gj


def pp_net_to_geojson(net, assume_latlon: bool = True) -> dict:
    """ pandapower network -> GeoJSON FeatureCollection of buses (Point) + lines (LineString).
    Reads geodata from net.bus['geo'] (pandapower >=3.0 stores it as a GeoJSON-geometry string).

    assume_latlon: this repo's converters (transnet_to_pp, gridsfm_to_pp) pass geodata as (lat, lon),
    so the stored coordinates are [lat, lon] -- GeoJSON wants [lon, lat]. True (default) swaps them.
    """
    feats = []
    bus_xy = {}
    for b in net.bus.index:
        geo = net.bus.at[b, "geo"] if "geo" in net.bus.columns else None
        xy = None
        if isinstance(geo, str) and geo:
            try:
                xy = list(json.loads(geo)["coordinates"])
            except (ValueError, KeyError):
                xy = None
        elif isinstance(geo, (list, tuple)) and len(geo) == 2:
            xy = list(geo)
        if xy is None:
            continue
        if assume_latlon:
            xy = [xy[1], xy[0]]
        bus_xy[b] = xy
        feats.append({"type": "Feature", "properties": {"kind": "bus", "id": int(b),
                      "name": str(net.bus.at[b, "name"]), "vn_kv": float(net.bus.at[b, "vn_kv"])},
                      "geometry": {"type": "Point", "coordinates": xy}})
    for li in net.line.index:
        f, t = int(net.line.at[li, "from_bus"]), int(net.line.at[li, "to_bus"])
        if f in bus_xy and t in bus_xy:
            feats.append({"type": "Feature", "properties": {"kind": "line", "id": int(li),
                          "name": str(net.line.at[li, "name"]), "from_bus": f, "to_bus": t},
                          "geometry": {"type": "LineString", "coordinates": [bus_xy[f], bus_xy[t]]}})
    return {"type": "FeatureCollection", "features": feats}


def _voltage_kv(tags: dict, default_kv: float = 20.0) -> float:
    """ OSM 'voltage' tag -> kV (it is documented to be in volts; may be a ';'-separated list --
    take the max). Falls back to default_kv. Standalone local copy of osm_to_pp._parse_voltage_kv
    so this module stays importable when run as a script. """
    v = tags.get("voltage")
    if not v:
        return default_kv
    try:
        v = max(float(x) for x in str(v).split(";") if x.strip()) if ";" in str(v) else float(v)
        return v / 1000.0
    except (ValueError, TypeError):
        return default_kv


def geojson_power_to_pp(gj: dict, tol: float = 1e-3):
    """ Overpass-derived power GeoJSON (from osm_to_geojson / fetch_power_geojson) -> pandapower net.

    Unlike data/osm_to_pp.network_from_OSM, which matches a line's OSM node-id path against known bus
    node ids, Overpass `out geom` output carries geometry only (no node ids). So endpoints are matched
    to buses by *coincident coordinates* within `tol` degrees -- the only linkage available here.
    A line whose endpoint matches no bus is dropped (documented simplification).

    Buses come from Point features (and Polygon centroids) tagged power in {substation, plant,
    generator}; lines from LineString features tagged power in {line, cable, minor_line}.
    Returns a pandapowerNet with net.bus['geo'] set (lon/lat GeoJSON strings) so pp_net_to_geojson
    round-trips it for the map.
    """
    import json as _json
    import pandapower as pp

    net = pp.create_empty_network()
    bus_pts = []                       # (lon, lat, bus_idx)

    def _centroid(coords):
        pts = coords[0] if coords and isinstance(coords[0][0], (list, tuple)) else coords
        return [sum(c[0] for c in pts) / len(pts), sum(c[1] for c in pts) / len(pts)]

    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        geom = ft.get("geometry") or {}
        power = props.get("power")
        if power not in ("substation", "plant", "generator"):
            continue
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"][:2]
        elif geom.get("type") in ("Polygon", "MultiPolygon"):
            lon, lat = _centroid(geom["coordinates"])
        else:
            continue
        default_kv = 110.0 if power == "plant" else 20.0
        name = props.get("name") or f"{power} {ft.get('id', len(bus_pts))}"
        idx = pp.create_bus(net, vn_kv=_voltage_kv(props, default_kv), name=str(name), type="b")
        net.bus.at[idx, "geo"] = _json.dumps({"type": "Point", "coordinates": [lat, lon]})
        if power in ("plant", "generator"):
            pp.create_gen(net, idx, p_mw=float(props.get("_pmw", 50.0) or 50.0), vm_pu=1.0,
                          name=f"gen@{name}")
        bus_pts.append((lon, lat, idx))

    def _nearest(lon, lat):
        best, bd = None, tol
        for blon, blat, bidx in bus_pts:
            d = ((blon - lon) ** 2 + (blat - lat) ** 2) ** 0.5
            if d <= bd:
                best, bd = bidx, d
        return best

    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        geom = ft.get("geometry") or {}
        if props.get("power") not in ("line", "cable", "minor_line"):
            continue
        if geom.get("type") != "LineString" or len(geom["coordinates"]) < 2:
            continue
        (alon, alat), (blon, blat) = geom["coordinates"][0][:2], geom["coordinates"][-1][:2]
        fb, tb = _nearest(alon, alat), _nearest(blon, blat)
        if fb is None or tb is None or fb == tb:
            continue
        pp.create_line(net, from_bus=fb, to_bus=tb, length_km=max(
            0.05, 111.0 * (((alon - blon) ** 2 + (alat - blat) ** 2) ** 0.5)),
            std_type="NAYY 4x150 SE", name=str(props.get("name") or f"line {ft.get('id', '')}"))
    return net


if __name__ == "__main__":
    # Shiloh wind farm area, CA -- (south, west, north, east)
    bbox = (38.0780, -121.9452, 38.2452, -121.7295)
    cache = os.path.join(FIXTURE_DIR, "shiloh_power.geojson")
    gj = fetch_power_geojson(bbox, cache_path=cache)
    kinds = {}
    for ft in gj["features"]:
        kinds[ft["properties"].get("power")] = kinds.get(ft["properties"].get("power"), 0) + 1
    print(f"{len(gj['features'])} features:", kinds)
    print("vector tiles:", OPENINFRAMAP_VECTOR_TILES)
