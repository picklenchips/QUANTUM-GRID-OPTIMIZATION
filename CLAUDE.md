# /physics/qpgrid — agent guide

> Read root context first: [Documents/CLAUDE.md](../../CLAUDE.md) + [agent-context/MD_README.md](../../agent-context/MD_README.md) — vault-wide conventions, required before creating/editing any `.md` file here. Then [physics/CLAUDE.md](../CLAUDE.md) for the physics workspace map. This file covers qpgrid specifics only.

Quantum + classical optimization for electrical grid problems: microgrid formation, optimal power flow, unit commitment. Started as Womanium Quantum+AI 2024 hackathon project (archived rubric/submission: [womanium_README.md](womanium_README.md)) — now an independent passion project, no hackathon constraints or deadline. Vision + research survey: [docs/](docs/). Current implementation state: [docs/versions/V0_SUMMARY.md](docs/versions/V0_SUMMARY.md).

---

## Repo map

| Path | What |
|---|---|
| `pp_to_microgrid.py` | Core pipeline: pandapower network construction, microgrid partitioning, QUBO formulation. Parses + runs end-to-end (minimal example + real transnet-california, lambd=1 and lambd=0.5 paths both verified). |
| `algos/` | Optimizer layer, classical + quantum. `problems.py`: `QUBOProblem` (`.from_microgrid(net)` wraps `pp_to_microgrid`), `MILPProblem`, `UnitCommitmentProblem`. `optimizers.py`: one `Optimizer` family — brute_force / steepest_descent / simulated_annealing / monte_carlo / tabu / tree_decomposition / gurobi / highs / lagrangian / admm / **qaoa / vqe / numpy_min_eigen** (Qiskit, gate-model, on the same QUBO the D-Wave path anneals). `benchmark.py`: run-all-and-compare + Plotly. `dqi_marimo.py` = quantum DQI (marimo). See `docs/02-ALGORITHMS.md`. |
| `pypsa_bridge.py` | pandapower ↔ PyPSA v1.0 bridge + `stochastic_microgrid_scenarios()` (two-stage stochastic / CVaR demand-uncertainty sizing). See `docs/08-TOOLING-INTEGRATION.md`. |
| `api/` | Local FastAPI backend bridging `web/` to `algos/` + `data/openinframap_hook.py`. `optimizers_api.py` introspects the `Optimizer` registry (→ generated frontend forms); `network_builder.py` turns a drawn bbox into a pandapower net; `routers/optimize.py` runs any optimizer over SSE. Run `uvicorn api.main:app --reload --port 8000`. Spec: `docs/09-WEB-OVERHAUL.md`. |
| `quantum_util.py` | General quantum-physics utilities (Bloch sphere sim, gates, pulse sequences) — confirmed AMO coursework carryover, not grid-specific. Only load-bearing use is `plot_op` in `qubo_formulation.ipynb`. |
| `creation.py`, `pp_learn.py`, `util.py` | Supporting scripts. `optimization.py` is now a thin re-export of `algos/`. `pp_learn.py` is dead (imports undefined `aux`; documented, not fixed). |
| `data/` | Grid data + format converters. Local-file → pandapower: `transnet_to_pp.py`, `psse_to_pp.py`, `osm_to_pp.py`. Remote-dataset hooks (docs/08): `gridsfm_to_pp.py`, `opfdata_to_pp.py`, `pglearn_to_pp.py`, `gridfm_datakit_hook.py`, `openinframap_hook.py`, `hifld_archive.py`, `climrr_overlay.py`. `fixtures/` = small offline test data. Raw datasets in `NYISO_data/`, `ppnets/`, `psse/`, `transnet/`. |
| `notebooks/` | Exploratory work + `tooling_integration.ipynb` (runs every data/ hook against `fixtures/`, offline; nbconvert-clean) + `classical_optimizers.ipynb` (benchmarks `algos/` on microgrid QUBO + unit commitment). `qubo_formulation.ipynb` = quantum QUBO coursework. |
| `docs/` | Condensed project docs + post-mid-2024 research updates (05/06/07) + `08-TOOLING-INTEGRATION.md` (07's survey → code) + `versions/` for dated state summaries |
| `QPGrid/` | React Native (Expo Router) mobile app scaffold — map + learning tabs |
| `web/` | Grid maps, **MapLibre + OpenInfraMap tiles, no Mapbox token**: `opengridmap_source.html` (full map — voltage-tiered lines, substations/plants, filter/basemap/bbox/**Optimize**/**Topology** panels, pipeline overlays; script split into `web/js/*.js` classic-global modules; Terra Draw + Cytoscape + Plotly; live network-from-selection + optimizer runs when `api/` is up) and `tooling_dashboard.html` (tooling-status matrix + converted network). Both read `web/data/*` written by `notebooks/tooling_integration.ipynb`. `*_preview.png` = screenshots. See `web/README.md` + `docs/09-WEB-OVERHAUL.md`. |
| `environment.yml` | Conda env `qpgrid` — Python 3.12; `pandapower`/`pypsa`/`dwave-ocean-sdk`/`gridfm-datakit` via the `pip:` section (conda-forge lags). Create with `--solver=libmamba`. |
| `plots/`, `nx_pyvis_tests/`, `third_party/`, `external/` | Misc experiments / scratch |

---

## Before any task

1. Read this file + [docs/versions/V0_SUMMARY.md](docs/versions/V0_SUMMARY.md) for current implementation state.
2. Touching optimization/QUBO logic → read [docs/02-ALGORITHMS.md](docs/02-ALGORITHMS.md) + [docs/03-LITERATURE.md](docs/03-LITERATURE.md) first; don't reinvent a formulation already surveyed there.
3. Touching data ingestion → check `data/*_to_pp.py` for the existing converter pattern before writing a new one.
4. Touching `QPGrid/` app → read `QPGrid/README.md` (Expo setup); it's a separate npm project, not part of the conda env.
5. No conda env active yet → `conda env create -f environment.yml && conda activate qpgrid` before running any Python here.

---

## Known issues / repo debt

Full inventory + priority order: [docs/versions/V0_SUMMARY.md](docs/versions/V0_SUMMARY.md). Highlights:

- `pp_to_microgrid.py` parses and runs cleanly (fixed in V0.2.0; the old `SyntaxError` at line 301 is gone). `python -c "import ast; ast.parse(open('pp_to_microgrid.py').read())"` passes.
- **Hardcoded Mapbox secret tokens, duplicated across 3 files**: `pp_to_microgrid.py:32-34`, `data/psse_to_pp.py:296`, `data/transnet_to_pp.py:225` (same `sk.`-prefixed `fullAccess` secret in all three; `pp_to_microgrid.py` also has a second `noWriting` secret). Treat as compromised — this repo has been public on GitHub. Rotate in the Mapbox dashboard, move to env vars. (The 4th token — a `pk.` under account `nirosjt` in `web/opengridmap_source.html` — is **gone**: that file was rewritten to MapLibre + no-token Esri/OpenInfraMap tiles.)
- `.github/workflows/webpack.yml` is GitHub's unmodified default Node/Webpack template — no root `package.json`, no `webpack.config.js` anywhere in the repo, so it doesn't actually build or test anything. CI is cosmetic only.
- Root `README.md` was Womanium hackathon boilerplate; now superseded — see [README.md](README.md) for the current version and `womanium_README.md` for the archived original.

---

## What NOT to do

- Don't commit new large binary datasets at repo root — `data/` already carries a 28MB CSV; keep additions there and prefer `.gitignore` + a fetch script over checked-in raw dumps where feasible.
- Don't hardcode API tokens/keys — see the Mapbox issue above; don't repeat it.
- Don't assume `quantum_util.py` is grid-specific — confirm actual call sites before extending it as if it were.

---

## Plotting

- **Plotly for plotting.** Charts: `plotly.graph_objects` / `plotly.subplots.make_subplots`. Grid networks: `pandapower.plotting.plotly` (`simple_plotly`, `pf_res_plotly`, `vlevel_plotly`). No matplotlib in repo code (it stays in the env only because pandapower depends on it).

## Python environment

- Conda only: `conda env create -f environment.yml --solver=libmamba`, env name `qpgrid`. Never pip install globally.
- Python 3.12 — highest that satisfies both Classiq (≤3.13) and gridfm-datakit (<3.13). `pandapower` / `pypsa` / `dwave-ocean-sdk` / `gridfm-datakit` are in the `pip:` section (conda-forge lags).
- `QPGrid/` (the app) is a separate Node/Expo project — `npm install` inside `QPGrid/`, unrelated to the conda env.

---

## Commit conventions

Follow root vault style: `<type>: <description>` + `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Types: `feature`, `fix`, `docs`, `chore`.

---

---

*Related: [[physics/CLAUDE]] · [[CLAUDE]] · [[agent-context/MD_README]] · [[SKILLS]] · [[docs/00-VISION]] · [[docs/versions/V0_SUMMARY]] · [[AGENTS]] · [[README]]*
