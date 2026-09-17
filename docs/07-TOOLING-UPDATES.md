# Data & tooling updates (Jun 2024 – Aug 2026)

> Deep-research pass extending [04-DATA-SOURCES.md](04-DATA-SOURCES.md). Every entry fetched/verified against changelog/release page; approximate/unverified figures flagged inline.
>
> **This survey is now wired into code** — one Python hook per adopted tool, adopt/reject rationale, and verification: [08-TOOLING-INTEGRATION.md](08-TOOLING-INTEGRATION.md).

## Act on this now

**HIFLD Open shut down August 2025.** Any pipeline pointing at live `hifld-geoplatform.hub.arcgis.com` URLs (already catalogued in 04-DATA-SOURCES.md) is dead. Repoint to the [Data Rescue Project archive](https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/) or [SeerAI/source.coop mirror](https://source.coop/seerai/hifld) (Parquet), or replace with Microsoft GridSFM below.

## New/replacement datasets

| Dataset | Date | Scale | Note |
|---|---|---|---|
| [**Microsoft GridSFM_US_power_grid**](https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid) ([blog](https://www.microsoft.com/en-us/research/blog/building-realistic-electric-transmission-grid-dataset-at-scale-a-pipeline-from-open-dataset/)) | May 2026 | 48 US states + 6 interconnections, up to full Eastern Interconnection (**~21,697 buses**, 36 states) | **Standout find** — built same open-data way as transnet/HIFLD (OSM+EIA+Census) but continental scale, AC-OPF-solvable out of the box. Strong candidate to replace/supplement transnet-models |
| PGLearn ([arXiv:2505.22825](https://arxiv.org/abs/2505.22825), [huggingface.co/PGLearn](https://huggingface.co/PGLearn), code at [AI4OPT/PGLearn.jl](https://github.com/AI4OPT/PGLearn.jl)) | May 2025 | N-1 contingency set: ~3.6M feasible samples, up to >10,000-bus | ML-training-oriented (primal+dual solutions), not a direct pandapower/PyPSA network — needs conversion step |
| OPFData (Google DeepMind, [arXiv:2406.07234](https://arxiv.org/abs/2406.07234), code [AI4OPT/OPFData](https://github.com/AI4OPT/OPFData)) | Jun 2024 (window edge) | 13,000 AC-OPF instances, 8 grids, 118–6,515 buses | Topological perturbations incl. N-1 outages |
| gridfm-datakit-v1 ([arXiv:2512.14658](https://arxiv.org/abs/2512.14658), [github.com/gridfm/gridfm-datakit](https://github.com/gridfm/gridfm-datakit), Apache 2.0) | Dec 2025 | Stochastic PF to 30,000 buses, OPF to 10,000 | Actively developed (134 stars/27 forks, open PRs); designed to fix gaps in OPFData/PGLearn |
| Gridfinder | — | — | No update found since mid-2024; appears static |
| New synthetic test grids beyond Texas 7000-bus/NREL 240-bus | — | — | None found — PGLearn/gridfm-datakit above are the closest "bigger test case" equivalent |

## pandapower

Verified via [CHANGELOG.rst](https://github.com/e2nIEE/pandapower/blob/develop/CHANGELOG.rst).

- **v3.0.0** (2025-03-06) — **breaking**: internal "blueprint" network-structure redesign, hybrid AC/DC power flow (VSC elements, DC buses/lines), CGMES v3.0 support, `setup.py`→`pyproject.toml`, dropped Python 3.8. Check before any environment upgrade — this project's `pp_to_microgrid.py` does direct DataFrame manipulation on `net` internals, exactly what the redesign touches.
- **v3.3.0** (2025-12-15) — Julia backend via `juliacall`, parallel contingency analysis (multiprocessing), per-phase SVC voltage control.
- **v3.5.0** (2026-07-06, latest) — redispatch optimizer (cost/power-based), OpenDSS converter, HELMpy solver integration, DC power flow elements.
- `pandapower.topology`: no QUBO/graph-search-specific feature since mid-2024 (custom switch/transformer edge weights landed 2.14.5, Mar 2024 — just pre-window). Otherwise stability/bugfix stream only.

## PyPSA — the biggest tooling change in this survey

**PyPSA v1.0** ([release notes](https://docs.pypsa.org/latest/release-notes/)) — first stable release, ~Oct 14 2025, 200+ contributors, ~5,000 commits. New: native two-stage **stochastic programming** (scenario trees, risk-neutral + CVaR risk-averse), opt-in Components-class API, Xarray-based backend for custom constraints, backward-compat guaranteed until v2.0.

Pandapower has no equivalent uncertainty-aware optimization mode. If this project ever extends into stochastic/robust grid optimization (uncertain demand, renewable variability), **PyPSA v1.0 is now a real alternative worth evaluating**, not just the "if pandapower doesn't fit" fallback [04-DATA-SOURCES.md](04-DATA-SOURCES.md) already lists it as.

## OpenInfraMap / OpenGridMap / osm2pgsql

- **OpenInfraMap actively maintained** — [main repo](https://github.com/openinframap) (TypeScript, BSD-3, 590 stars) last updated **Aug 10, 2026**; companion repos (`josm-power-network-tools`, `black-marble` nighttime-lights layer, `imposm3-docker`) all updated within months. Good news for this project's plan to fork/build on it ([01-ARCHITECTURE.md](01-ARCHITECTURE.md)).
- **osm2pgsql v2.1.0** (Apr 2025) — connection-pooler compatibility, prepared statements. **v2.2.0** (Sep 2025) — new "flex" building-block features for custom import schemas.
- OpenGridMap (transnet-models source) — no new activity found.

## Quantum-for-grid open-source toolkits

- **Nothing new found beyond Classiq's existing electric-grid-optimization example.** "QuantumGridOS" (quantumgridos.com) surfaced in search but **fails verification** — no visible stars/forks/commit history, no papers. Reads as a marketing site, not a real maintained project. **Do not adopt as a dependency without independently confirming the repo exists.**
- PyQUBO (pre-existing, not new) remains the likely-relevant general QUBO-construction library.
- Classiq + Wolfram Research joined **CERN's Open Quantum Institute** (Apr 2025, see [05-QUANTUM-UPDATES.md](05-QUANTUM-UPDATES.md)) specifically for grid Unit Commitment — a research collaboration announcement, no code release found yet.

## D-Wave Ocean SDK / Classiq SDK (developer tooling specifics)

Verified via [dwave-ocean-sdk releases](https://github.com/dwavesystems/dwave-ocean-sdk/releases).

- Ocean **v9.0.0** (2024-09-10) — removed `dwavebinarycsp`, new SAPI v3 solver representation. **If any code here ever depended on `dwavebinarycsp`, it breaks on 9.x** — check before upgrading.
- v9.1.0 (2024-11-06); v9.2.0/9.3.0 (Dec 2024/Jan 2025) — Python 3.14 support, new `dwave-optimization` array ops (Transpose, IsIn, Roll, Cos, Sin, SoftMax).
- dimod gained problem generators migrated from `dwave-networkx` (`vertex_coloring`, `matching`, `traveling_salesperson`); dropped Python 3.9 support.
- Classiq SDK ~v1.14.0 per search (**not directly verified against a changelog — treat as approximate**); platform-level "Version 1.0" announced Feb 2026 (see [05-QUANTUM-UPDATES.md](05-QUANTUM-UPDATES.md)).
- No breaking changes found that block this project's current D-Wave usage — just pin to a 9.x release given the `dwavebinarycsp` removal.

## Climate-data-overlay tools

- ClimRR (Argonne) confirmed still active, described as "newly enhanced" (customized reports combining demographic + hazard layers) and won a Climate Registry Climate Leadership Award — **exact enhancement date not independently verified**, check [anl.gov/ccrds/ClimRR](https://www.anl.gov/ccrds/ClimRR) directly before citing specifics.
- No other new climate-overlay tool found relevant to grid-state visualization.

## Most actionable

1. **Fix any HIFLD-Open-pointing code now** — it's dead, not degraded.
2. **Evaluate Microsoft GridSFM** as transnet-models replacement/supplement — same open-data philosophy, continental scale, solvable out of the box.
3. **Read pandapower 3.0.0's breaking changes** before any `environment.yml` bump — direct relevance to `pp_to_microgrid.py`'s internals access.
4. **Consider PyPSA v1.0** if/when this project wants uncertainty-aware optimization — capability gap pandapower doesn't fill.
5. **Pin Ocean SDK to 9.x deliberately**, not accidentally — note the `dwavebinarycsp` removal.

---

*Related: [[08-TOOLING-INTEGRATION]] · [[04-DATA-SOURCES]] · [[01-ARCHITECTURE]] · [[05-QUANTUM-UPDATES]] · [[versions/V0_SUMMARY]]*
