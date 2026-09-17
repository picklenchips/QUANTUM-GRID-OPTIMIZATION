# Architecture

> Condensed from original planning doc's "Outline" section. Intended 3-layer system: data pipeline → optimization core → map/app front end. Not all layers built — see [versions/V0_SUMMARY.md](versions/V0_SUMMARY.md) for what exists in code today.

## Layer 1 — grid data → `ElectricalGrid` graph

Translate real-world grid data into Python graph structure (`pandapower` network), run optimizations on it.

**Node** = power prosumer: plant / distributed / consumer, generation type (solar, wind, nuclear, battery storage...), supply/demand scheduling info. Advanced: model transformers, individual consumers.

**Edge** = transmission line, encoded by voltage.

**Microgrid** = subset of nodes, grouped by voltage and/or geography; stored as node-index groups within larger network.

Data sources → pandapower, by format:
| Format | Status | Reference |
|---|---|---|
| transnet CSV (nodes/edges) | Implemented | [transnet-models](https://github.com/OpenGridMap/transnet-models) |
| PSS/E | Planned | [format docs](https://docs.andes.app/en/latest/getting_started/formats/psse.html) |
| Raw OSM (real-time circuit inference) | Planned | [transnet/app](https://github.com/OpenGridMap/transnet/tree/master/app) |

Prior art worth building on rather than re-deriving: [pandapower topology / graph search](https://github.com/e2nIEE/pandapower/blob/master/pandapower/topology/graph_searches.py), [pandapower↔networkx conversion](https://github.com/e2nIEE/pandapower/blob/master/pandapower/topology/create_graph.py), [RL powernet.py](https://github.com/MarvinLer/pypownet), OpenGridMap's [transnet load estimator](https://github.com/OpenGridMap/transnet/blob/master/app/LoadEstimator.py#L141). Graph viz: PyVis.

Backend integration: local FastAPI service (`api/`) for the web layer; D-Wave QUBO for quantum layer ([SampleSet docs](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html#dimod.SampleSet)).

## Layer 2 — interactive grid map (web) + optimizer bridge

Built. Spec + endpoint contracts: [09-WEB-OVERHAUL.md](09-WEB-OVERHAUL.md).

- **Map**: `web/opengridmap_source.html` — MapLibre GL, OpenInfraMap vector tiles, no Mapbox token. Inline `<script>` split into `web/js/{map-init,api-client,selection-tools,topology-panel,results-panel}.js` (classic globals, no bundler).
- **Backend**: `api/` — FastAPI, run `uvicorn api.main:app --reload --port 8000` alongside `python -m http.server`. In-memory network/job registries (no database — a local single-user tool; PostGIS could return if cross-session persistence is ever needed). The page degrades to static behaviour when the backend is absent.
- **Source**: OSM `power` data via Overpass, through `data/openinframap_hook.py` (`fetch_power_geojson` + `geojson_power_to_pp`). No osm2pgsql / PostGIS ingest.
- **Interaction**: Terra Draw rectangle/lasso + click/shift-click feature selection; a drawn box → `POST /network/from-selection` builds a pandapower net; Cytoscape.js topology panel synced two-way with the map selection; bus drag-to-reposition.
- **Bridge to Python**: server round-trip. `POST /optimize` + an SSE stream run any `algos/` optimizer on the selected network; results colour the map (partition) and the topology graph. Optimizer forms are generated from `GET /optimizers` introspection — a new `@register` optimizer needs no frontend change.

## Layer 3 — mobile/web app

Wraps map + optimization tools + educational content into one accessible surface (Expo/React Native — see `QPGrid/`).

- Map viewer/explorer with same tooling as web JS version.
- Predictive analytics + optimization tools surfaced directly (not just visualization).
- Learning content: grid literacy + quantum computing literacy + explanation of each implemented optimization problem.

Carries "communicate & inform, demonstrate" goal from [00-VISION.md](00-VISION.md) — highest-leverage surface for making quantum grid optimization legible to non-experts, including grid operators evaluating approach.

---

*Related: [[00-VISION]] · [[02-ALGORITHMS]] · [[04-DATA-SOURCES]] · [[09-WEB-OVERHAUL]] · [[versions/V0_SUMMARY]]*
