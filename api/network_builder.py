"""
api.network_builder -- a drawn map selection -> pandapower net -> GeoJSON for the map.

Reuses data/openinframap_hook.py unchanged for the Overpass fetch + cache and the pandapower
round-trip; adds only the frontend<->Overpass bbox-order remap (easy to get backwards, so it
lives in exactly one named place).
"""
from __future__ import annotations
import hashlib
import os

from data.openinframap_hook import fetch_power_geojson, geojson_power_to_pp, pp_net_to_geojson
from .config import CACHE_DIR, MAX_BBOX_DEG2


def frontend_box_to_overpass_bbox(box) -> tuple[float, float, float, float]:
    """frontend draws [west, south, east, north]; Overpass wants (south, west, north, east)."""
    w, s, e, n = box
    return (s, w, n, e)


def _validate(box) -> tuple[float, float, float, float]:
    w, s, e, n = (float(x) for x in box)
    if not (w < e and s < n and abs(w) <= 180 and abs(e) <= 180 and abs(s) <= 90 and abs(n) <= 90):
        raise ValueError(f"invalid bbox {box!r}")
    if (e - w) * (n - s) > MAX_BBOX_DEG2:
        raise ValueError(f"selection too large ({(e-w)*(n-s):.2f} deg^2 > {MAX_BBOX_DEG2}); "
                         "zoom in before loading full detail")
    return (w, s, e, n)


def build_network_from_box(box) -> tuple[object, dict]:
    """box = [w, s, e, n]. Returns (pandapowerNet, geojson). Raises ValueError on a bad/huge box."""
    w, s, e, n = _validate(box)
    over = frontend_box_to_overpass_bbox((w, s, e, n))
    key = hashlib.sha1(",".join(f"{x:.5f}" for x in over).encode()).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, f"{key}.geojson")
    gj = fetch_power_geojson(over, cache_path=cache_path)
    net = geojson_power_to_pp(gj)
    return net, pp_net_to_geojson(net)
