# Tooling integration — plan + status

> 2026-08-27. Turns [07-TOOLING-UPDATES.md](07-TOOLING-UPDATES.md)'s survey into code. One thin Python
> hook per external tool/dataset called out there, following the repo's existing `data/*_to_pp.py`
> convention (module docstring w/ source URL + verified format notes, `HAVE_<dep>` import guards for
> optional deps, `if __name__ == '__main__'` demo, no hardcoded tokens). Proof they run:
> [notebooks/tooling_integration.ipynb](../notebooks/tooling_integration.ipynb) +
> [web/tooling_dashboard.html](../web/tooling_dashboard.html).

## Decision — what got adopted, and why

| 07 survey entry | Verdict | Rationale |
|---|---|---|
| **Microsoft GridSFM** | **Adopt** (already wired, `data/gridsfm_to_pp.py`) | continental scale, AC-OPF-solvable, same open-data lineage as transnet; HIFLD replacement |
| **OPFData** (DeepMind) | **Adopt** — `data/opfdata_to_pp.py` | public GCS bucket, no auth, `grid`+`solution` JSON is trivial to parse; gives a *reference AC-OPF solution* per instance — exactly the ACOPF baseline [06-ML-SOTA.md](06-ML-SOTA.md) says to measure against |
| **PGLearn** | **Adopt** — `data/pglearn_to_pp.py` | HF-hosted, MIT-ish; `case.json` carries a complete per-unit admittance-primitive grid spec → pandapower; primal+dual HDF5 is the only open dataset with full dual solutions (KKT multipliers) |
| **gridfm-datakit** | **Adopt (thin wrapper)** — `data/gridfm_datakit_hook.py` | Apache-2.0, on PyPI, actively developed; it *generates* PF/OPF scenarios from a base case (N-k perturbations) rather than shipping a fixed dump — complements the static datasets |
| **PyPSA v1.0** | **Adopt (bridge)** — `pypsa_bridge.py` | native two-stage stochastic optimization (scenario trees, CVaR) — a capability pandapower has no equivalent for; bridge converts a pandapower `net` ↔ PyPSA `Network` |
| **OpenInfraMap / osm2pgsql** | **Adopt (hook)** — `data/openinframap_hook.py` | already the web map's tile source; hook adds a Python-side Overpass query → GeoJSON so a converted network and live OSM infra can be shown on the same map |
| **HIFLD Open** (dead) | **Repoint** — `data/hifld_archive.py` | live ArcGIS endpoints gone Aug 2025; hook points at the Data Rescue Project / DataLumos archive + source.coop parquet mirror |
| **ClimRR** (Argonne) | **Adopt (overlay)** — `data/climrr_overlay.py` | ArcGIS REST FeatureServer, no auth; fetches seasonal Fire Weather Index (and other hazard) grid cells, spatial-joins to a network's bus geodata → per-bus hazard score for the climate-overlay angle in [00-VISION.md](00-VISION.md) |
| pandapower 3.x | **Track** (done in V0.1.0) | converters already migrated (`bus_geodata` table → `net.bus['geo']` GeoJSON string); env unpinned, resolves 3.5.x |
| D-Wave Ocean 9.x | **Pin in pip** | conda-forge stops at 6.0.1; 9.x is PyPI-only. `environment.yml` `pip:` section pins `>=9,<10` |
| Classiq SDK | **Defer** | python-3.13 ceiling respected in `environment.yml`; still commented out — no grid-specific Classiq code yet, revisit with [05-QUANTUM-UPDATES.md](05-QUANTUM-UPDATES.md) item |
| "QuantumGridOS" | **Reject** | fails verification (no repo/stars/papers) — 07 already flagged it |

## Modules

All live in `data/` except the solver-side bridge (`pypsa_bridge.py`, repo root, next to `pp_to_microgrid.py`).

| Module | Public entry points | External dep (guarded) | Offline fixture |
|---|---|---|---|
| `data/gridsfm_to_pp.py` | `download_gridsfm_model`, `GridSFMOut`, `gridsfm_to_pp` | `huggingface_hub` | `data/fixtures/gridsfm/rhode_island_model_16h.json` |
| `data/opfdata_to_pp.py` | `download_opfdata_group`, `load_opfdata_example`, `opfdata_to_pp`, `opfdata_solution_df` | none (stdlib `urllib`) | `data/fixtures/opfdata/*.json` |
| `data/pglearn_to_pp.py` | `download_pglearn_case`, `pglearn_case_to_pp`, `read_pglearn_h5` | `huggingface_hub`, `h5py` | `data/fixtures/pglearn/case14_ieee.json` |
| `data/gridfm_datakit_hook.py` | `have_gridfm_datakit`, `generate_pf_scenarios`, `load_scenarios_df` | `gridfm-datakit` (pip) | — (generator, not a dataset) |
| `data/openinframap_hook.py` | `overpass_power_query`, `osm_to_geojson`, `pp_net_to_geojson`, `OPENINFRAMAP_VECTOR_TILES` | none (`requests`) | `data/fixtures/openinframap/*.geojson` (cached demo bbox) |
| `data/hifld_archive.py` | `HIFLD_ARCHIVE_MIRRORS`, `fetch_hifld_transmission_lines`, `hifld_lines_to_geojson` | `pyarrow` (parquet path) | — |
| `data/climrr_overlay.py` | `CLIMRR_LAYERS`, `fetch_climrr_cells`, `bus_hazard_scores` | none (`requests`) | `data/fixtures/climrr/fwi_demo.geojson` |
| `pypsa_bridge.py` | `pp_net_to_pypsa`, `stochastic_microgrid_scenarios` | `pypsa>=1.0` | uses `pp_to_microgrid.create_minimal_example` |

## Verification

`notebooks/tooling_integration.ipynb` runs one section per module against the offline fixtures (so it
executes with no network), and each section has an opt-in `LIVE = True` switch that re-runs the same
call against the real remote source. `jupyter nbconvert --execute` passes end-to-end;
`web/tooling_dashboard.html` renders the status matrix + a Leaflet map (OSM + OpenInfraMap overlay
tiles, no Mapbox token) with a converted network's GeoJSON overlaid.

## Not done / follow-ups

- PGLearn full `train` primal/dual tensors are 300 MB–700 MB per case — the hook streams a single
  split and the notebook only pulls `test`/`meta`; no bulk-download helper.
- `gridfm_datakit_hook` runs the library's own PF solver; it does **not** yet feed results back into
  `pp_to_microgrid`'s QUBO path.
- ClimRR spatial join is nearest-cell centroid, not true polygon-in-polygon (the grid is 12 km, buses
  are points — good enough for a hazard score, not for area statistics).
- `pypsa_bridge` covers buses/lines/loads/generators/storage; **not** transformers (PyPSA models them
  as lines with a tap) or switches.

---

*Related: [[07-TOOLING-UPDATES]] · [[04-DATA-SOURCES]] · [[06-ML-SOTA]] · [[versions/V0.1.0_TODOS]] · [[01-ARCHITECTURE]]*
