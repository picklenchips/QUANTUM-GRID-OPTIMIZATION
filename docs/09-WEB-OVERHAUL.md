# Web visualization overhaul — implementation spec

> Executable spec for the plan approved 2026-09-05, **built 2026-09-08**. Wires the MapLibre map ([../web/opengridmap_source.html](../web/opengridmap_source.html)) to the optimizer stack ([../algos/](../algos/)) and the OpenInfraMap hook ([../data/openinframap_hook.py](../data/openinframap_hook.py)). Supersedes Layer 2 of [01-ARCHITECTURE.md](01-ARCHITECTURE.md) (PostgreSQL+PostGIS vision, never built). Current impl state: [versions/V0_SUMMARY.md](versions/V0_SUMMARY.md).

## As-built notes (differ from / add to the plan)

- **Terra Draw adapter class is `terraDrawMaplibreGlAdapter.TerraDrawMapLibreGLAdapter`** (capital `L`, capital `ML`) — not `TerraDrawMaplibreGLAdapter`. terra-draw `1.33.0`, adapter `1.4.1`, cytoscape `3.34.3`, plotly `2.35.2`.
- **Event names are prefixed `qpg:`** (`QPGrid.emit`/`QPGrid.on` add it) — the bare name `selectionchange` collides with the browser's native `document` event and fires the handler with an undefined detail.
- **turf was dropped** — point-in-polygon for the lasso is a ~10-line ray cast in `selection-tools.js`. One less CDN dependency.
- `web/data/demo_network.geojson` predates the `from_bus`/`to_bus` line-property addition, so the topology graph shows nodes-only for the built-in demo. A network built from a map selection has the endpoints and graphs correctly. Regenerating the demo file is a `web/README.md` TODO.
- An OpenInfraMap selection is dominated by `power=generator` points (a wind farm = hundreds of turbines), so `QUBOProblem.from_microgrid` on it is large-n: `brute_force`/`qaoa`/`vqe`/`numpy_min_eigen` refuse, `steepest_descent`/`simulated_annealing`/`admm`/`gurobi`/`highs` handle it. The topology panel hides degree-0 buses (keeps the count in its note).
- Backend deps live in `environment.yml`'s `pip:` block: `fastapi`, `uvicorn[standard]`.

## Why

`web/opengridmap_source.html` renders grid layers + a bbox tool, but the bbox tool is decorative — `wireControls()` validates a rectangle and calls `fitBounds`, sends it nowhere. No path from the map to the 14 optimizers in `algos/`, no network-from-selection, no topology view, no component select/drag. Goal: an additive, framework-free layer that degrades to today's behavior when no backend runs (keeps the `QPGrid/` WebView embed unaffected — it can't reach `localhost:8000`).

## Constraints found during research (do not rediscover these)

| # | Constraint | Consequence for impl |
|---|---|---|
| 1 | `pp_net_to_geojson` line features carry `{kind,id,name}` only — no endpoint bus IDs | Add `from_bus`/`to_bus` additively; topology graph edges need them |
| 2 | `algos/optimizers.py` + `problems.py` use `from __future__ import annotations` | Introspection must use `typing.get_type_hints(cls.solve)`, not `Parameter.annotation` (else every type is a literal string) |
| 3 | `Optimizer.solve()` is sync, CPU-bound, no progress callback exposed (QAOA/VQE have an internal `cb()` only) | SSE reports job **status** + post-hoc `Result.trace` replay, not live per-iteration progress. Do not overpromise |
| 4 | `QUBOProblem.from_microgrid`: `x_i` = **array position** in `net.bus.index`, not raw bus id | Result→map projection: `raw_bus_id = list(net.bus.index)[i]`; `pp_net_to_geojson` keys features by raw `int(b)` |
| 5 | `osm_to_pp.network_from_OSM` needs OSM node IDs on line paths; Overpass `out geom` output (from `osm_to_geojson`) has coordinates only, no node IDs | Add a new small coordinate-matching converter `geojson_power_to_pp`; do not force reuse of `osm_to_pp.py` |
| 6 | MapLibre `setData()` on a source added **after** `map.on('load')` is unreliable under software-GL (see `opengridmap_source.html:126-127` comment) | Sources that will change must exist at construction. Bus-drag (needs live `setData()`) is lowest-priority; fall back to a "commit move" full rebuild if it doesn't repaint |
| 7 | `openinframap_hook` bbox order is `(south, west, north, east)`; frontend box is `[w,s,e,n]` | One named remap helper `frontend_box_to_overpass_bbox(box)`, unit-checked by hand. Never inline the swap |
| 8 | Terra Draw selects/edits features **it drew**, not arbitrary existing MapLibre features | Terra Draw replaces the draw-a-box/lasso tool only. Click/shift-click select of existing buses/lines is separate, via `map.setFeatureState({selected:true})` + a `feature-state` paint expression |
| 9 | `QUBOProblem` subclasses `Result` (quirk, harmless); `from_microgrid` sets `.meta` post-init | Leave alone; do not "fix" |

## Backend — `api/`

Local-only FastAPI. Run alongside the static server:

```
uvicorn api.main:app --reload --port 8000        # terminal 1
python -m http.server                            # terminal 2 (repo root), open /web/opengridmap_source.html
```

Add to [../environment.yml](../environment.yml) `pip:` block: `fastapi`, `uvicorn[standard]`.

```
api/
  __init__.py
  main.py              # FastAPI(), CORSMiddleware(allow_origins=["*"]), mounts routers, GET /health
  config.py            # CACHE_DIR, MAX_BBOX_DEG2 guardrail, CORS origins
  schemas.py           # pydantic models (SelectionRequest, OptimizeRequest, ...)
  optimizers_api.py    # describe() over Optimizer.registry()
  network_builder.py   # frontend_box_to_overpass_bbox + fetch_power_geojson + geojson_power_to_pp
  jobs.py              # NETWORKS: dict[str, pandapowerNet], JOBS: dict[str, dict]; run_in_threadpool dispatch
  routers/
    network.py         # POST /network/from-selection
    optimize.py        # POST /optimize, GET /optimize/{job_id}/stream
```

**Threading**: never call `opt.solve(...)` inside an `async def` handler — 2-30 s block on the event loop. Dispatch via `starlette.concurrency.run_in_threadpool` from a background task.

**CORS `*`**: single-user local research tool, different ports for static vs API. Fine because local-only; not a hardening target.

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","optimizers":<int>}` |
| GET | `/optimizers?kind=qubo` | — | `{"optimizers":[<describe()>...]}` |
| POST | `/network/from-selection` | `{"bbox":[w,s,e,n]}` | `{"network_id","geojson","n_bus","n_line"}` |
| POST | `/optimize` | `OptimizeRequest` (below) | `{"job_id"}` |
| GET | `/optimize/{job_id}/stream` | — | `text/event-stream` |

`OptimizeRequest`:

```json
{
  "network_id": "<uuid from /network/from-selection>",
  "problem_kind": "qubo",
  "problem_params": {"lambd": 1.0},
  "optimizer": "simulated_annealing",
  "solver_params": {"sweeps": 3000, "restarts": 4}
}
```

- `problem_kind: "qubo"` → `QUBOProblem.from_microgrid(net, **problem_params)`
- `problem_kind: "uc"` → `UnitCommitmentProblem.from_pp(net, demand=problem_params["demand"])`

### `describe(cls)` — optimizer introspection

```python
import inspect, typing

def describe(cls) -> dict:
    inst = cls()
    hints = typing.get_type_hints(cls.solve)          # resolves the __future__ annotation strings
    params = []
    for name, p in inspect.signature(cls.solve).parameters.items():
        if name in ("self", "problem", "kw") or p.kind is p.VAR_KEYWORD:
            continue
        params.append(_param_schema(name, hints.get(name), p.default))
    return {"name": cls.name, "kinds": list(cls.kinds), "available": inst.available,
            "docstring": inspect.getdoc(cls) or "", "params": params}
```

`_param_schema`: `int`→`"integer"`, `float`→`"number"`, `str`→`"string"`, `bool`→`"boolean"`; `T | None` → base type + `"nullable": true`; `default` from `p.default` unless `inspect._empty`. Unannotated params (`GurobiOptimizer.solve`'s `problem`) already filtered; other unannotated → `"type": "string"` fallback.

This is the zero-frontend-change mechanism: a new `@register` class in `algos/optimizers.py` appears in the picker with a generated form, no JS edit. Verify by adding a throwaway dummy optimizer, confirming it renders, removing it.

### `geojson_power_to_pp(gj, tol=1e-4)` — new, in [../data/openinframap_hook.py](../data/openinframap_hook.py)

Overpass-derived GeoJSON → `pandapowerNet`. Cannot use `network_from_OSM` (constraint 5). Rules:

- **Bus** per `Point` feature with `power in {substation, plant, generator}`. `vn_kv` parsed from the `voltage` tag (reuse `osm_to_pp`'s voltage parser if importable, else a local `_parse_voltage_kv`; `"220000;110000"` → take max, `kV`).
- **Line** per `LineString` feature whose first and last coordinate each fall within `tol` degrees of a bus Point. `from_bus`/`to_bus` = nearest bus. Lines with an unmatched endpoint are dropped (documented simplification, same style as `osm_to_pp.py` docstrings).
- Return `pp_net_to_geojson(net)` for the response — same schema the map's `net-bus`/`net-line` layers already read.

### SSE stream events (`GET /optimize/{job_id}/stream`)

```
event: status\ndata: {"status":"running"}\n\n
: heartbeat                                        (comment line, ~10 s cadence while running)
event: done\ndata: {<result payload>}\n\n
event: error\ndata: {"type":"ValueError","message":"..."}\n\n
```

Result payload (all numpy → `.tolist()`):

```json
{
  "method": "simulated_annealing", "objective": -12.4, "feasible": true,
  "runtime_s": 0.31, "n_iter": 8000, "bound": null, "gap": null,
  "trace": [ ... ], "meta": { ... },
  "bus_assignment": {"<raw_bus_id>": 0, ...},
  "geojson": { ... pp_net_to_geojson(net), bus features gain "partition":<int>,
               line features gain "cross_partition":<bool> ... }
}
```

`bus_assignment`: `{str(list(net.bus.index)[i]): int(res.x[i]) for i in range(len(res.x))}` (constraint 4). `cross_partition` = endpoints in different groups.

For `problem_kind: "uc"`: `res.x` is `(G, T)` dispatch. Payload swaps `bus_assignment`/`geojson` for `dispatch: [[...]]`, `gens: [{pmin,pmax,cost}...]`, `demand: [...]` — client renders a Plotly stacked-area chart. No GeoJSON augmentation (dispatch isn't spatial).

## Frontend — `web/js/` module split

Split the inline `<script>` in `opengridmap_source.html` into classic-global scripts (no bundler), shared `window.QPGrid` namespace, cross-module comms via `CustomEvent` on `window`.

| File | Responsibility |
|---|---|
| `web/js/map-init.js` | style/sources/layers (today's `init()` minus controls). Exposes `QPGrid.map`. Adds `feature-state` `selected` paint expr + `["match",["get","partition"],...]` paint expr (default gray) to `net-bus`/`net-line` at construction |
| `web/js/api-client.js` | `fetch`/`EventSource` wrappers, `/health` gate. All calls no-op cleanly when backend absent |
| `web/js/selection-tools.js` | Terra Draw rectangle/lasso (replaces shift-drag box, `opengridmap_source.html:292-307`); click/shift-click feature-state select; `moveend` in-view counter via `map.queryRenderedFeatures()`; bus-drag (last) |
| `web/js/topology-panel.js` | Cytoscape.js graph from `net` GeoJSON, `cose` layout, two-way sync with map selection (`suppressEcho` flag guards ping-pong) |
| `web/js/results-panel.js` | optimizer picker from `/optimizers`, generic param form, `EventSource` on stream, Plotly charts (shared `PAL`), run-history in `localStorage` |

CDN additions (classic globals, pin versions), loaded before the `js/` scripts:

```html
<script src="https://unpkg.com/terra-draw/dist/terra-draw.umd.js"></script>
<script src="https://unpkg.com/terra-draw-maplibre-gl-adapter/dist/terra-draw-maplibre-gl-adapter.umd.js"></script>
<script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/@turf/turf/turf.min.js"></script>
```

`PAL` (match [../algos/benchmark.py](../algos/benchmark.py) exactly, for cross-surface color consistency):

```js
const PAL = ["#DC267F","#648FFF","#FE6100","#785EF0","#FFB000","#009E73","#3DDBD9","#808080"];
```

### CustomEvents

| Event | Fired by | Consumed by |
|---|---|---|
| `selectionchange` `{ids}` | selection-tools | topology-panel, results-panel |
| `topologyselectionchange` `{ids}` | topology-panel | selection-tools (→ feature-state + `fitBounds`) |
| `busmoved` `{busId, lngLat}` | selection-tools | results-panel (marks network dirty) |
| `resultready` `{payload}` | results-panel | map-init (`setData` on `net`), topology-panel (node `partition` data) |

### Selection mechanics

- **Terra Draw**: `TerraDrawRectangleMode` + `TerraDrawFreehandMode`. On `finish`, hand the drawn polygon to `turf.booleanPointInPolygon` / `booleanIntersects` against cached `net` GeoJSON (or `queryRenderedFeatures`) → selected-id set → `selectionchange`.
- **Click select**: extend the existing `map.on('click', id, ...)` handlers (`opengridmap_source.html:219`) to toggle `map.setFeatureState({source, id}, {selected: true})`. Shift-click = add to set.
- **Mode exclusion**: one `QPGrid.mode` string (`"select" | "draw-rect" | "draw-lasso" | "drag-bus"`), toolbar-toggled; only the active mode's handlers armed. `"drag-bus"` calls `terraDraw.setMode("static")` first.
- **OpenInfraMap load**: `moveend` → `queryRenderedFeatures` counter (zero network calls — verify in devtools). A single "Load full detail" button (enabled only with a box + backend up; client cooldown + `MAX_BBOX_DEG2` server guardrail) → `POST /network/from-selection`.

### Topology panel

- Elements once per `net` payload: nodes `{id:"bus-<id>", label, vn_kv}`, edges `{id:"line-<id>", source:"bus-<from_bus>", target:"bus-<to_bus>"}` (needs constraint-1 fix).
- Shows electrical adjacency independent of geography — geographically-overlapping-but-electrically-unrelated substations separate in the graph.
- Two-way sync guarded by shared `let suppressEcho = false` set/cleared around each cross-update.
- On `resultready`: `cy.nodes().forEach(n => n.data('partition', payload.bus_assignment[n.id().replace('bus-','')]))`, stylesheet `background-color` from a `PAL`-indexed mapping.

## Phasing (all built 2026-09-08)

| Phase | Scope | Verified |
|---|---|---|
| 0 | `api/main.py`, `config.py`, env deps, backend badge in HTML | ✅ `uvicorn` runs, `/health` → `{"status":"ok","optimizers":14}`, page identical backend up/down |
| 1 | `optimizers_api.py`, `jobs.py`, `routers/optimize.py`, `api-client.js`, `results-panel.js`, Plotly CDN — against network `"demo"` (`create_minimal_example`) | ✅ picker + generated form + SSE + convergence chart; dummy `@register` optimizer surfaced with `wobble`/`tries`/`loud` params then reverted |
| 2 | `map-init.js`, `selection-tools.js` (Terra Draw + click select + lasso), `network_builder.py`, `geojson_power_to_pp`, `routers/network.py`; inline script split | ✅ Shiloh bbox `[-121.9452,38.0780,-121.7295,38.2452]` → 527 buses / 7 lines, optimizer run → pink/blue partition colours on the map (headless-Chrome screenshot) |
| 3 | `topology-panel.js`, Cytoscape CDN, `from_bus`/`to_bus` schema add, panel in HTML | ✅ graph builds from the selection (527 buses, 7 connected), selection events wired both directions with `suppressEcho` guard, result colours nodes |
| 4 | `moveend`/`idle` in-view counter, "load full detail" button (cooldown + `MAX_BBOX_DEG2` guardrail), `frontend_box_to_overpass_bbox` | ✅ counter updates on pan with no new requests; oversized bbox → HTTP 422 with a clear message |
| 5 | run-history/provenance (`localStorage`), bus-drag, doc updates | ✅ history persists + click-to-re-render without re-solving; bus-drag mutates in-memory GeoJSON + `setData` on the construction-time `net` source |

## Verification

- Per phase: run both servers, exercise in a real browser; headless-Chrome screenshot for visual phases (2, 3) — matches this repo's no-test-suite convention (manual + notebook/screenshot).
- Phase 1: add/remove a throwaway `@register` optimizer, confirm UI reflects it with no JS edit.
- Phase 2 / 4: devtools Network tab alongside the functional check.
- No new automated tests for this surface.

## Doc updates (phase 5)

| File | Change |
|---|---|
| [01-ARCHITECTURE.md](01-ARCHITECTURE.md) | Replace Layer 2 PostgreSQL+PostGIS vision (lines 26-33) with the `api/` architecture (in-memory `NETWORKS`/`JOBS`, no DB). Trim the Layer 1 "PostgreSQL for web layer" clause (line 24) |
| [../web/README.md](../web/README.md) | Add `api/` + `web/js/*.js` to the file table; replace TODO 1 with a pointer here; document running both servers |
| [../CLAUDE.md](../CLAUDE.md) | Repo-map row for `api/`; note `web/js/` split + `environment.yml` additions. Fix the stale "known issues" line — `pp_to_microgrid.py` compiles cleanly as of V0.2.0 |

---

*Related: [[01-ARCHITECTURE]] · [[02-ALGORITHMS]] · [[08-TOOLING-INTEGRATION]] · [[versions/V0_SUMMARY]] · [[CLAUDE]] · [[README]]*
