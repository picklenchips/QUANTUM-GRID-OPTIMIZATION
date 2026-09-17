# web/ — static grid maps

Builds on [OpenGridMap] and [OpenInfraMap] for easy access to the world's power-grid
infrastructure, and layers this project's own pipeline output on top.

**No Mapbox token.** Both pages use MapLibre GL JS with OpenInfraMap vector tiles + Esri
"no-key" raster basemaps. The old Mapbox-token version of `opengridmap_source.html` is gone.

| File | What |
|---|---|
| `opengridmap_source.html` | Full map: OpenInfraMap power lines (voltage-tiered), substations, plants, voltage labels; panels for layer filters / basemap / bounding-box / **Optimize** / **Topology**; QPGrid pipeline overlays (converted `pandapower` network, Overpass-hook GeoJSON, ClimRR choropleth). Terra Draw select/lasso, Cytoscape topology graph, live network-from-selection + optimizer runs when the `api/` backend is up. Embedded by the mobile app (`QPGrid/app/tabs-layout/map.jsx`). |
| `js/` | The page's script, split into classic-global modules: `map-init` (map build), `api-client` (backend calls + event bus), `selection-tools` (Terra Draw, click select, bbox, bus drag), `topology-panel` (Cytoscape), `results-panel` (optimizer picker + Plotly). |
| `tooling_dashboard.html` | Tooling-integration status matrix (`docs/08-TOOLING-INTEGRATION.md`) + the same map overlays. |
| `data/` | GeoJSON + `tooling_status.json` written by `notebooks/tooling_integration.ipynb` (regenerate by running it). |
| `*_preview.png` | Rendered screenshots. |
| `openinframap_tile.json` | OpenInfraMap vector-tile layer manifest (source-layer names used by both pages). |

Run locally — two servers:

```
python -m http.server 8777                 # repo root; open http://localhost:8777/web/opengridmap_source.html
uvicorn api.main:app --reload --port 8000   # the optimizer + network-from-selection backend
```

The page works with only the static server (backend badge shows "offline"); the Optimize and
live-selection features need the `api/` server. Endpoint contracts: [`../docs/09-WEB-OVERHAUL.md`](../docs/09-WEB-OVERHAUL.md).

TODO:
1. Wire `web/data/` refresh into CI or a small fetch script.
2. Regenerate `demo_network.geojson` with `from_bus`/`to_bus` line properties so its topology graph shows edges.

---

*Related: [[../CLAUDE]] · [[../docs/09-WEB-OVERHAUL]] · [[../docs/08-TOOLING-INTEGRATION]] · [[../docs/01-ARCHITECTURE]]*
