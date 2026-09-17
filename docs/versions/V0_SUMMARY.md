# V0 summary — state of the repo

> 2026-08-07. First full read-through since reviving this as an independent project. Ground truth from static analysis + import testing of every Python file, every notebook's cells/outputs, the app, and the web prototype — not from the docs, which describe intent. Vision/vs/reality gap, file-by-file status, security findings, and where to start.

## TL;DR

- **One pipeline actually works end-to-end**: transnet CSV → pandapower network. Proof: `data/ppnets/transnet-california-n.db` exists on disk.
- **The core file is currently broken**: `pp_to_microgrid.py` — where the QUBO microgrid-partitioning math lives — has a `SyntaxError` and doesn't parse. Fixing that reveals further logic bugs in the top-level orchestration function.
- **OSM and PSS/E converters both stop at the same wall**: raw-format parsing works (tested against real data), conversion into a pandapower network is an explicit `NotImplementedError` stub in both.
- **The QUBO math is designed**, not just imagined: `microgrids.ipynb` derives modularity + self-reliance objectives from two cited papers, and `pp_to_microgrid.py` implements them (`modularity_matrix`, `self_reliance_matrix`, `microgrid_objective`, `to_QUBO`, `simulate_anneal`). It just doesn't run right now.
- **`qubo_formulation.ipynb` is generic QWorld/Womanium coursework** (knapsack, TSP, graph coloring, Max-Cut, D-Wave BQM walkthrough) — good reusable QUBO tooling, but not grid-specific, and its last saved run has a `dimod` import failure plus a dangling variable reference.
- **The app is a themed scaffold**, not a working product: one real feature (embeds the web map prototype via `WebView`), rest is static copy, dead nav routes, and unused template leftovers from a generic Expo starter.
- **The web map prototype is the most finished piece of frontend work** here: real Mapbox GL JS app against OpenInfraMap's live tile server, voltage-tiered styling, click popups, filter panel. Not wired to the Python backend — the button that would do it is commented out.
- **CI is cosmetic**: unmodified GitHub default template, no root `package.json` or `webpack.config.js` for it to build.
- **Security**: a Mapbox secret token is committed in 3 files; rotate it. Detail below.

## Vision vs. reality, by architecture layer

Cross-reference: [../01-ARCHITECTURE.md](../01-ARCHITECTURE.md), [../02-ALGORITHMS.md](../02-ALGORITHMS.md).

| Layer | Planned | Actual |
|---|---|---|
| Data → pandapower (transnet) | Implemented | **Done.** `data/transnet_to_pp.py` fully converts; artifact on disk proves it ran |
| Data → pandapower (PSS/E) | Planned | Parser works (tested on real 240-bus WECC data); `raw_to_pp()` conversion step is a stub |
| Data → pandapower (OSM) | Planned | Download/inspection works; `network_from_OSM()` conversion is a stub — notebook explains the team deliberately punted on this in favor of transnet |
| Microgrid formation (QUBO) | Implemented | Math designed + implemented (`pp_to_microgrid.py`), **currently broken** (syntax error + logic bugs) |
| Optimal power flow (AC/DC) | Core open research question | Not started — `optimal_power_flow.ipynb` is one unfinished markdown cell |
| Unit commitment, pricing, siting | Open problems, surveyed | Not started — literature-only, see [../03-LITERATURE.md](../03-LITERATURE.md) |
| Interactive web map | PostGIS + OSM, selection → Python bridge | Map UI real and polished (Mapbox GL, OpenInfraMap tiles); PostGIS/selection/bridge-to-Python never built — "Query" button is commented out in the HTML |
| Mobile app | Map + optimization tools + learning content | Navigation shell + static learning copy + embedded web map via WebView. No optimization tools, no live data, no backend calls anywhere in the app |

## File-by-file status

Legend: ✅ works · ⚠️ partially works / has bugs · 🚧 explicit stub (`NotImplementedError` or empty) · 💀 dead/superseded code

| File | Status | Note |
|---|---|---|
| `data/transnet_to_pp.py` | ✅ | Only fully-working conversion pipeline; produced `data/ppnets/transnet-california-n.db` |
| `data/osm_to_pp.py` | ⚠️ | `get_OSM_data()` + `print_osm_info()` work (live API, tested); `network_from_OSM()` is 🚧 |
| `data/psse_to_pp.py` | ⚠️ | `.raw`/`.dyr` parsers work (tested against real WECC 240-bus data); `raw_to_pp()` is 🚧 |
| `util.py` | ✅ | Clean general-helper library (`uFormat`, `timeIt`, stats). Hardcodes personal absolute paths (not secrets) — non-portable, not a security issue |
| `pp_to_microgrid.py` | ⚠️ **broken** | `SyntaxError` line 301 (`NetGraph.only_keep_buses`). Beyond that: `cut_to_nbus()` indexes a `set` (line 240, `TypeError`); module-level `pp.from_sqlite(...)` (line 82) runs on import with a missing path separator; `microgrid_optimization()` calls a non-existent function `cut_net_to_nbusses` and misuses a `Queue`/`dict` API; hardcoded personal path in `__main__`. Core math (`modularity_matrix`, `self_reliance_matrix`, `to_QUBO`, `simulate_anneal`) reads as complete once the file parses. `PartitionStorage.__init__` is 🚧 (`raise NotImplementedError`) |
| `quantum_util.py` | ⚠️ | Syntax-valid, fails to import (`ImportError: PLOTDIR` — not defined in `util.py`). Generic AMO/quantum-coursework utility file (Bloch spheres, pulse sequences), not grid-specific. Only load-bearing use anywhere in the repo: `plot_op` in `qubo_formulation.ipynb` |
| `pp_learn.py` | 💀 | `NameError` on first function def (`aux` never imported); unqualified numpy names throughout. Reads as abandoned internals-notes, not used elsewhere |
| `creation.py` | 💀 | Explicitly superseded first-prototype (own docstring says so); `Grid.propogateVoltage()` unfinished stub |
| `optimization.py` | 🚧 | 4-line placeholder docstring, no code |
| `QPGrid/app/tabs-layout/map.jsx` | ✅ | Real feature: embeds `web/opengridmap_source.html` via `WebView` |
| `QPGrid/app/_layout.jsx`, `index.jsx`, tab layout | ✅ | Functional navigation shell/boilerplate |
| `QPGrid/app/tabs-layout/home.jsx` | ⚠️ | Nav buttons target routes (`Dashboard`, `LiveData`, `About`) that don't exist in the app |
| `QPGrid/app/tabs-layout/info.jsx` | ⚠️ | Static content works; one image uses a filesystem path as a `uri` (won't resolve in RN) |
| `QPGrid/components/Loader.jsx`, `constants/*` | 💀 | Unused template leftovers (generic Expo "Aora" tutorial scaffold), not imported anywhere |
| `web/opengridmap_source.html` | ⚠️ | Real, functional Mapbox GL app against OpenInfraMap tiles. Selection→backend bridge (the "Query" button) is commented out |
| `web/openinframap_tile.json` | 💀 | Orphaned — never referenced by any other file; the HTML hardcodes its own layer list instead |
| `.github/workflows/webpack.yml` | 💀 | Unmodified GitHub default template; nothing in the repo matches what it tries to build |

Notebooks:

| Notebook | Status | Note |
|---|---|---|
| `OSM_power.ipynb` | ✅ (exploratory) | Real executed OSM pulls (2 bboxes). Markdown explicitly documents the pivot away from OSM inference toward transnet's pre-built data |
| `data_pipeline.ipynb` | ✅ (exploratory) | Broader dev version of the same exploration; confirms PSS/E raw parser works on real data. Top markdown cell's TODO framing is stale — OSM/transnet both have working code lower in the same notebook |
| `microgrids.ipynb` | ✅ (math only) | No code — derives the modularity + self-reliance QUBO objective from [arXiv:2112.08300](https://arxiv.org/abs/2112.08300) and [arXiv:2403.17495](https://arxiv.org/pdf/2403.17495). This is the paper trail for `pp_to_microgrid.py`'s objective functions |
| `optimal_power_flow.ipynb` | 💀 | One markdown cell, cuts off mid-sentence. Least-developed file in the repo |
| `qubo_formulation.ipynb` | ⚠️ | Generic QWorld/Womanium QUBO coursework (brute-force solver, knapsack, TSP, graph coloring, Max-Cut, D-Wave BQM/Ising walkthrough) — not grid-specific. Last saved run: `dimod` `ModuleNotFoundError` partway through, and the final cell references `bqm_1`, only ever defined in the cell that errored. **Correction**: filesystem shows a Mar 2025 mtime, but `git diff HEAD` confirms content is byte-identical to the Aug 2024 commit — treat this as 2024-era work, not "most recent activity" |
| `pp_learn.ipynb` (both copies — `notebooks/` and `nx_pyvis_tests/`) | Not read (19MB / 7.7MB) | Two different-sized copies exist in two locations — check which is canonical before editing either |

## Security — action needed

Hardcoded Mapbox tokens, confirmed via exact-string grep across the whole repo:

| Token | Type | Location(s) |
|---|---|---|
| `fullAccess` (`sk.…`) | **Secret** | `pp_to_microgrid.py:32-34`, `data/psse_to_pp.py:296`, `data/transnet_to_pp.py:225` — same string, duplicated 3×, activated via `ppplot.set_mapbox_token(fullAccess)` at import time |
| `noWriting` (`sk.…`) | **Secret** | `pp_to_microgrid.py` only |
| `publicToken` (`pk.…`, account `benkroul`) | Public | `pp_to_microgrid.py` only |
| unnamed `pk.…` token, account **`nirosjt`** (not `benkroul`) | Public | `web/opengridmap_source.html:177` — different account than the others; confirm this is intentional (possibly copied from a tutorial) before treating it as this project's own |

This repo has been public on GitHub. Rotate both `sk.` secrets in the Mapbox dashboard, move all four to environment variables, and don't reintroduce the pattern in new code (see `CLAUDE.md` → What NOT to do). No other credential types found (no AWS keys, D-Wave Leap tokens, DB passwords) in a full repo-wide sweep.

## Notable finds beyond the original plan

- `data/psse/WECC_240bus_2018summer_2021_IEEE-NASPI_OSL/` — a real, NREL-published 240-bus WECC dynamic model (`.raw` + `.dyr`), already sitting in the repo. Matches the "NREL test grids up to 240-bus" entry in [../04-DATA-SOURCES.md](../04-DATA-SOURCES.md) — no need to go source it.
- `data/NYISO_data/` — nine real NYISO market/operations files from 2024-07-12 (constraints, flows, real-time load, fuel mix). Not referenced by any script yet — a data pull that never got wired into a pipeline. Wasn't in the original planning doc's data-source list at all.
- `data/Electric__Power_Transmission_Lines.csv` (27MB) — another standalone dataset, also not yet referenced by any script.

## Priority-ordered punch list

1. **Fix `pp_to_microgrid.py`'s syntax error** (line 301, `NetGraph.only_keep_buses`) — nothing else in that file is reachable until it parses.
2. Fix the module-level import-time side effect (`pp.from_sqlite(cwd+'data/...')`, missing path separator, runs unconditionally on import) — should be behind `if __name__ == "__main__":` regardless.
3. Fix `cut_to_nbus()`'s `set` indexing (line 240) and `microgrid_optimization()`'s `Queue`/`dict` misuse + reference to the nonexistent `cut_net_to_nbusses`.
4. Rotate the Mapbox secret tokens; move all 4 tokens to env vars.
5. Fix `quantum_util.py`'s `PLOTDIR` import (add it to `util.py`, or stop importing it if unused beyond `plot_op`).
6. Decide: is `qubo_formulation.ipynb`'s generic QUBO tooling (`qubo_solver`, `tsp_qubo`, the D-Wave BQM walkthrough) worth promoting into a proper module now that it'll get reused, or left as coursework reference?
7. Either implement `raw_to_pp()` / `network_from_OSM()`, or delete them and document PSS/E + OSM as "parsing only" until there's a reason to finish the conversion.
8. QPGrid app: fix or remove the dead nav buttons in `home.jsx`, fix the `info.jsx` image path, decide whether to keep the unused `react-native-maps` dependency and template leftovers (`Loader.jsx`, `constants/`) or strip them.
9. CI: replace `webpack.yml` with something that actually exercises this repo (e.g. `conda env create` + `pytest` for Python, `expo` build check for `QPGrid/`), or remove it.

None of the above has been touched in this pass — this doc is a state snapshot, not a changelog. Start here next session.

---

*Related: [[00-VISION]] · [[01-ARCHITECTURE]] · [[02-ALGORITHMS]] · [[CLAUDE]] · [[README]]*
