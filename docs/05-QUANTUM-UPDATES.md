# Quantum updates (Jun 2024 – Aug 2026)

> Deep-research pass, three threads: grid-specific quantum optimization papers, general QAOA/annealing/quantum-inspired advances, hardware/platform landscape. Extends [02-ALGORITHMS.md](02-ALGORITHMS.md) + [03-LITERATURE.md](03-LITERATURE.md) (compiled ~mid-2024) — read those first. Every citation fetched/verified against primary source; unconfirmed items flagged inline.

## Bottom line up front

No verified, unrebutted quantum-advantage claim exists for combinatorial optimization (QUBO/Ising) as of Aug 2026 — same conclusion as mid-2024, richer/more skeptical literature behind it now. Now goes further than "no verified win yet": [**Proving the Limits of Quantum Power Flow**](https://arxiv.org/abs/2607.19263) (Khanpour & Talkington, Jul 2026, Lean-4-verified proof) shows DC power flow's susceptance-Laplacian solve is provably *not* quantum-accelerable at any readout level — transmission-topology separators force the pseudo condition number κ up polynomially (Ω(n) treewidth-bounded, Ω(n²) for long corridors), and query+tomography lower bounds mean Θ̃(n/ε) fresh QLS solves are required regardless of algorithm, beating any quantum speedup with classical nearly-linear Laplacian solvers (Õ(m log 1/ε)). Authors state this **obstruction persists through AC power flow, OPF, and unit commitment**, not just DCPF. This resolves the "is DC simpler, or same AC/DC gap in different formulation" open question flagged below re: 2311.02436 — answer: same gap, proven not just suspected, and it's a *topology* property (readout + conditioning), not a loss/unitarity property. Pattern across all three threads: bold claim → credible classical rebuttal within weeks-to-months → vendor pushback, repeatedly. D-Wave's flagship "advantage" result (*Science*, Mar 2025) is a physics-*simulation* claim, not optimization, disputed within days. On a real energy problem, D-Wave hybrid still loses to Gurobi/CPLEX ([2409.05542](https://arxiv.org/abs/2409.05542)). Most recent, most rigorous grid-specific benchmark ([2607.15543](https://arxiv.org/abs/2607.15543), Jul 2026): qubit-efficient hybrid UC method *not outperforming uniform random sampling*.

Recommendation: keep D-Wave primary (tooling maturity, not proven speedup); upgrade `SimulatedAnnealingSampler` from no-hardware fallback to actively-benchmarked competitor — GPU quantum-inspired solver (Simulated Bifurcation-style) beats plain CPU annealing, fairer baseline than current setup.

---

## 1. Grid-specific quantum optimization

### ACOPF / DCOPF / OPF

| Paper | Date | Note |
|---|---|---|
| [Power flow/OPF via quantum + digital annealers: scalability analysis](https://arxiv.org/abs/2505.15978) (TU Delft) | May 2025 | "AQOPF" QUBO, **4-bus to 1354-bus** on D-Wave Advantage + Fujitsu DA v3 — largest quantum OPF test found. Scalability study, not perf win. |
| [Quantum Hardware-in-the-Loop OPF, Renewable-Integrated](https://arxiv.org/abs/2505.13356) (same group) | May 2025 | Real-time digital sim + actual quantum/quantum-inspired HW, IEEE 9-bus + solar/wind. |
| [CVQLS-Augmented Interior Point Method](https://arxiv.org/abs/2412.14095) (Amani & Kargarian, LSU) | Dec 2024, IEEE TSG | Quantum linear solver in OPF IPM — successor lineage to 2311.02436. |
| [Nested-Loop Trajectory-Informed VQS-IPM](https://arxiv.org/abs/2607.03361) (same authors) | Jul 2026 | Own follow-up: trajectory info cuts variational updates up to 95%, real-HW validated. |
| [Hybrid Quantum NN for ACOPF](https://arxiv.org/abs/2410.20275) | Oct 2024 | PQC + classical DL + physics-informed module — learning-based, not equation-solving. |
| [Physics-Informed Hybrid Dispatching](https://arxiv.org/abs/2601.18482) | Jan 2026 | Embeds power-flow/storage physics, noise-adaptive. |

**Verdict**: no clean speedup/accuracy win over classical IPM since mid-2024 — see Critical findings.

### Unit commitment

| Paper | Date | Note |
|---|---|---|
| [Survey: QC for Unit Commitment](https://arxiv.org/abs/2601.01777) | Jan 2026 (rev. Aug 2026) | Read first for UC deep-dive. |
| [Hybrid Quantum-Classical UC](https://arxiv.org/abs/2505.00145) (incl. Roetteler) | Apr 2025 | VQA + Benders heuristic, 3–26 units, IonQ Forte. |
| [Structure-Informed QAOA for Large-Scale UC](https://arxiv.org/abs/2503.20509) | Mar 2025 | Topology-based decomposition for limited qubit budgets. |
| [Qubit-Efficient QA for Stochastic UC](https://arxiv.org/abs/2502.15917) (Imperial) | Feb 2025 (rev. Jun 2026) | Aug-Lagrangian + quantum ADMM, **118-bus** on D-Wave QPU — largest UC demo found. |
| [Quantum RL Two-Stage UC](https://arxiv.org/abs/2410.21240) | Oct 2024 | Deep RL + quantum search, different angle than ADMM. |

Industry: [Classiq + Wolfram join CERN Open Quantum Institute](https://www.classiq.io/insights/classiq-and-wolfram-join-cerns-open-quantum-institute-to-develop-quantum-optimization-for-smart-power-grids) (Apr 2025) — targets UC.

### Microgrid formation / community detection

Direct lineage from papers already in [03-LITERATURE.md](03-LITERATURE.md) (Nikmehr/Zhang/Bragin, 2112.08300):

- [**Reforming Quantum Microgrid Formation**](https://arxiv.org/abs/2406.05916) (same Stony Brook group) — Jun 2024. Compact lossless QUBO, fewer qubits. **Read before touching `pp_to_microgrid.py`'s formulation** — direct upgrade path.
- [QA-Infused Microgrid Formation: Restoration/Resilience](https://ieeexplore.ieee.org/document/10526448/) — IEEE TPWRS Jan 2025. Direct sequel, disaster restoration; D-Wave CQM vs. Gurobi.
- [REGRID-QAOA: Islanding](https://arxiv.org/abs/2606.15083) | Jun 2026 | IEEE 9–57 bus, matches classical quality w/ fewer quantum resources.
- [Less Greedy Quantum Coalition Structure Gen](https://arxiv.org/abs/2408.04366) (LMU + **E.ON in-house quantum team**) | Aug 2024 | Honest: **"not yet" competitive** vs classical QBSolv on real D-Wave HW.
- [QA-Based Power Grid Partitioning](https://arxiv.org/abs/2408.04097) (Jülich) | Aug 2024 | D-Wave embedding caps feasibility **under ~200 buses**.
- [Spanning Trees for Power Distribution Grids](https://arxiv.org/abs/2511.00582) (Jülich) | Nov 2025 | QAOA for radial-topology reconfig.
- [Behavior-aware energy mgmt in microgrids](https://www.nature.com/articles/s41598-025-06199-z) | Jul 2025 | QA + NSGA-III, P2P demand-side w/ behavioral modeling.

### Dynamic pricing / P2P trading — thin category

Most 2024–2026 pricing/trading work classical (blockchain/game-theory), not quantum:
- [Grid Cost Allocation in P2P Markets: Classical vs Quantum](https://arxiv.org/abs/2501.05253) (LMU/E.ON) | Jan 2025 | 57-node IEEE — **classical branch-and-cut decisively wins**. Also a Critical finding.
- [Quantum Learning for Distribution-Network/Energy-Community Coordination](https://arxiv.org/abs/2506.11730) | Jun 2025 | Self-reported gains, **not independently benchmarked**.
- [Fermionic QAOA for Demand Portfolio](https://arxiv.org/abs/2505.02282) | May 2025 | Adjacent: procurement/portfolio risk, not P2P trading.

### Infrastructure siting

- [Joint gen-transmission expansion, hybrid quantum-classical](https://www.sciencedirect.com/science/article/pii/S0142061525006635) | Sep 2025 | WSQA inside Benders decomposition.
- [Quantum Opt for EV Charging Station Placement](https://arxiv.org/abs/2410.16231) | Oct 2024 | Grover-style amplitude amplification.
- [QC for EVs — Grid Resilience/Disaster Relief](https://arxiv.org/abs/2511.00736) | Nov 2025 | Opportunities framing, no full algo demo yet.
- **Real pilot**: [Iberdrola + Multiverse Computing — battery siting](https://multiversecomputing.com/resources/iberdrola-and-multiverse-computing-announce-pilot-project-success-to-optimize-battery) | Jul 2024 | 10-month, real **Guipúzcoa distribution grid** (Basque Country) via i-DE. Strongest siting pilot found.

### Real-world pilots / utility partnerships — genuine acceleration

| Partnership | Date | Detail |
|---|---|---|
| Iberdrola + Multiverse | Jul 2024 | Battery siting, real Spanish grid (above) |
| [Pasqal + EDF + GENCI](https://thequantuminsider.com/2025/01/21/pasqal-genci-and-edf-use-neutral-atom-quantum-computing-to-advance-ev-smart-charging-and-energy-forecasting/) | Jan 2025 | EV charging forecast, GENCI's **Ruby, 100+ qubit neutral-atom** |
| [IBM + E.ON](https://thequantuminsider.com/2024/12/02/powering-the-future-ibm-e-on-engineer-quantum-solutions-to-navigate-energy-challenges/) | Dec 2024 | Qiskit dynamic circuits, 27-qubit, weather-conditional pricing; "quantum utility" by 2029 target |
| [Infleqtion ARPA-E ENCODE](https://infleqtion.com/infleqtion-secures-62m-arpa-e-award-to-advance-quantum-powered-energy-grid-optimization/) | Mar 2025/Feb 2026 | $6.2M, first ARPA-E quantum award. Argonne, NREL, EPRI, **ComEd/Exelon**. 1,600-qubit neutral-atom |
| [IonQ + Oak Ridge (DOE GRID-Q)](https://www.ionq.com/news/ionq-partners-with-oak-ridge-national-laboratory-demonstrating-quantum-power) | Jul 2025 | Hybrid UC, **26 gens × 24 periods**, IonQ Forte Enterprise (36 qubits). Explicit "not yet advantage." 100–200 qubits projected needed |
| [Eaton + Infleqtion + Penn State — AFRL contract](https://www.eaton.com/us/en-us/company/news-insights/news-releases/2026/eaton-wins-contract-to-apply-quantum-computing.html) | Aug 2026 | $7M/24mo, grid-security (contingency/cascading-outage), not pure opt |

E.ON now dual-tracked (IBM HW partnership + in-house quantum team publishing skeptical benchmarks w/ LMU). EDF moved exploratory→real >100-qubit runs. No new Eni work found.

### Survey/review papers — read for fast context

- [**Quantum computing for smart grid**](https://www.nature.com/articles/s44287-026-00295-6) — *Nat Rev Elec Eng*, May 2026. **New anchor survey**, likely supersedes parts of mid-2024 framing. Key line: *"real near-term contest is not quantum vs classical but hybrid pipelines"* — benchmark vs. optimized classical, not weak baselines.
- [Opportunities for QC within net-zero power system opt](https://www.cell.com/joule/fulltext/S2542-4351(24)00155-7) — *Joule*, Jun 2024 (window start, Oxford).
- [PNNL-37598: Review of QC Technologies in Power System Opt](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-37598.pdf) — DOE/PNNL, Mar 2025.
- [QML early opportunities for energy industry: scoping review](https://www.frontiersin.org/journals/quantum-science-and-technology/articles/10.3389/frqst.2025.1653104/full) — Oct 2025.
- [Quantum Tech + Edge Devices in Electrical Grids](https://arxiv.org/abs/2603.06783) (RWTH/Jülich) — Mar 2026.
- [QC for Smart Grid Digital Twins](https://arxiv.org/abs/2508.18654) — Aug 2025.
- [QC for Energy Management: Practitioner's Guide](https://arxiv.org/abs/2411.11901) — Nov 2024, IET book chapter 2025.

### Critical / skeptical findings — read before trusting any grid-quantum claim

- [**Proving the Limits of Quantum Power Flow**](https://arxiv.org/abs/2607.19263) — Jul 2026, formally verified (Lean 4). Strongest/most rigorous finding in this doc: no encoding trick escapes it (see bottom-line note above) — full-vector readout cost + topology-forced ill-conditioning, not an algorithm-specific flaw. Explicitly claims to extend to AC-OPF and UC.
- [**Benchmarking Hybrid Quantum-Classical Algos for Power Grid Opt**](https://arxiv.org/abs/2607.15543) — Jul 2026, most recent + damning. AC-OPF-UC, 5–13 gens: *"qubit-efficient hybrid method does not outperform uniform sampling."* 25+ gens needed to maybe show advantage — exceeds current sim capacity.
- [Grid Cost Allocation in P2P Markets](https://arxiv.org/abs/2501.05253) — *"classical branch-and-cut outperforms all solvers... binary least-squares problems may not be suitable for near-term quantum utility."*
- [QC and Future of Power-System Opt: Promise, Hype, Road Ahead](https://gridforesight.ca/quantum-computing-and-the-future-of-power-system-optimisation-promise-hype-and-the-road-ahead/) — Jan 2026: *"advantages not demonstrated... verdict inconclusive."* Recommends classical HPC investment now, not waiting on quantum.
- Grid partitioning paper (above) — D-Wave caps feasibility under ~200 buses.

**Unverified, flagged**: MDPI *Sustainability* paper "QC as Catalyst for Microgrid Management" — 403 on fetch, don't cite until confirmed.

---

## 2. QAOA / annealing / quantum-inspired classical advances

### QAOA improvements — has "QA beats QAOA" held up?

Original finding (2301.00520) reaffirmed in journal version (*npj QI*, ~Mar 2024, LANL): **"QA outperforms QAOA on all instances"** for that short-depth/heavy-hex comparison — stands unchallenged on its own terms. New: other QAOA-*family* protocols claim to move past it — now themselves contested:

| Method | Date | Claim | Status |
|---|---|---|---|
| [LR-QAOA](https://www.nature.com/articles/s41534-025-01082-1) (Jülich) | 2024/2025 | Scaling advantage vs SA/Tabu/branch-and-bound; real HW to **109 qubits** (IonQ, Quantinuum, IBM) | Strongest QAOA-family result; doesn't re-test vs QA directly |
| [Bias-Field DCQO](https://arxiv.org/abs/2409.04477) (Kipu) | Sep 2024 | Beats QAOA/QA/SA/Tabu, 156-qubit HUBO on IBM; numerics to 433 | **Directly challenges "QA beats QAOA"** — see rebuttal |
| [Recent quantum runtime (dis)advantages](https://arxiv.org/abs/2510.06337) | Oct 2025 | Neither QA nor BF-DCQO claims survive rigorous runtime accounting | **Rebuts both rows above** |
| [Multi-angle QAOA](https://arxiv.org/abs/2312.00200) | *Sci Rep*, Aug 2024 | 4× shallower circuits, equal quality | Not optimal for total QPU time |
| [Warm-start QAOA via Max-Cut reduction](https://arxiv.org/abs/2504.06253) | Apr 2025 | Extends SDP warm-start to general QUBO | **No universal winner — problem-dependent** |
| [Optimisation-Free RQAOA](https://arxiv.org/abs/2507.10908) | Jul 2025, PRR | Robust to param transfer, cuts resources | Only paint-shop problem demo'd |
| [ADAPT-QAOA (AMA-QAOA+)](https://arxiv.org/abs/2412.19621) | Dec 2024 | +5.3–5.4% approx ratio, 75–85% fewer CNOTs | Max independent set only |
| CV-QAOA | Apr/Jun 2026 | [Complex-valued](https://arxiv.org/abs/2604.25950), [photonic demo](https://arxiv.org/abs/2606.10432) | Search-confirmed only, early-stage |

### Quantum-inspired classical solvers — toughest honest baseline

| Solver | Date | Result |
|---|---|---|
| [Toshiba 3rd-gen SB (edge-of-chaos)](https://news.toshiba.com/press-releases/press-release-details/2026/Toshibas-Breakthrough-Algorithm-Harnesses-Edge-of-Chaos-to-Dramatically-Boost-Performance-of-its-QuantumInspired-Computer/default.aspx) | Apr 2026, PRA | ~100× faster, near-100% success on **dense 2,000-spin** Ising — covers this project's target scale |
| [GPU-SBM rebuts QA scaling claim](https://arxiv.org/abs/2505.22514) | May 2025, PRA | Matches/beats QA advantage claim (PRL 134,160601/2401.07184) on same QUBO class — shows QA study undersized |
| [TII+NVIDIA 500K-qubit annealing sim](https://www.tii.ae/news/tii-demonstrates-large-scale-quantum-annealing-simulations-reaching-500000-qubits-nvidia) | Mar 2026 | GPU tensor-network emulator, 500K vars, beats every MQLib solver at scale |
| [Fujitsu DA vs D-Wave hybrid vs QIS3 — independent](https://arxiv.org/abs/2507.22117) | Jul 2025 | >2,000 MQLib graphs to 53K vars — **DA competitive w/ D-Wave hybrid** |
| [NTT/Tohoku Coherent Ising Machine](https://www.businesswire.com/news/home/20250709644790/en/NTT-Research-and-Tohoku-University-Collaborate-to-Accelerate-Development-of-Quantum-Enhanced-Coherent-Ising-Machines) | Jul 2025 | Roadmap to 100M spins; early-stage |
| NVIDIA cuOpt (v26.04) | Apr 2026 | GPU MIP/QP/LP, not QUBO-native but usable on MILP-formulated UC/scheduling — bake-off candidate vs `SimulatedAnnealingSampler` |
| [Tensor-network QUBO solvers](https://arxiv.org/abs/2409.01699) | *Quantum*, Jul 2025 | Good for sparse QUBOs, not yet leader for dense grid-scale |

### Benchmark/survey papers

- [**QOBLIB**](https://arxiv.org/abs/2504.03832) ("Intractable Decathlon") | Apr 2025 | 10 hard problem classes, ~100–100K vars. Closest emerging community standard — adopt methodology before internal quantum-vs-classical claims.
- [QA benchmarking study](https://arxiv.org/abs/2504.06201) | *npj QI*, May 2025 | ~6,561× speedup claim on dense Hamiltonian instances — **extraordinary, methodology unverifiable from abstract; don't cite number without full paper.**
- [Benchmarking QC algos for combinatorial opt](https://www.nature.com/articles/s41534-024-00825-w) | *npj QI*, Jun 2024 | MFB-CIM sub-exponential scaling vs near-exponential for discrete-adiabatic/Grover.
- [**QA vs CPLEX/Gurobi, real energy problem**](https://arxiv.org/abs/2409.05542) | Sep 2024 | D-Wave hybrid **did not outperform classical** on real-world energy opt. Direct reality check.

---

## 3. Hardware / platform landscape

### D-Wave

- **Advantage2 GA May 20, 2025** — [press release](https://www.dwavequantum.com/company/newsroom/press-release/d-wave-announces-general-availability-of-advantage2-quantum-computer-its-most-advanced-and-performant-system/). 4,400+ qubits, **Zephyr topology, 20-way connectivity** (up from Pegasus 15-way). Vendor: +40% energy scale, −75% noise, ~2× coherence, same 12.5kW draw.
- **Independent (LANL) 3-gen benchmark**: [Chimera/Pegasus/Zephyr on minor-embedded opt](https://iopscience.iop.org/article/10.1088/2058-9565/adb029) | Feb 2025 | Zephyr wins on approx ratio, chain-break freq, shorter embedding chains (4-6 vs 14 qubits).
- **No Advantage3.** [Jun 2026 gate-model roadmap](https://www.businesswire.com/news/home/20260601444734/en/D-Wave-Charts-a-New-Course-to-Fault-Tolerant-Quantum-Computing-with-Gate-Model-Roadmap) is long-horizon (fault tolerance ~2030+), not near-term annealing upgrade. Plan around Advantage2 for next few years.
- **Acquired Quantum Circuits Inc., $550M** (Jan 2026) — dual-platform strategy, gate-model roadmap to 100 logical qubits by 2032.
- **Simulation "advantage" disputed** — [*Science*, Mar/Apr 2025](https://www.science.org/doi/10.1126/science.ado6285): spin-glass sim claim, "millions of years" classical equivalent. **Disputed within days**: Tindall (Flatiron) classical belief-propagation beats it on some instances; [Mauron & Carleo (EPFL)](https://arxiv.org/abs/2503.08247) match/exceed to 128 spins, polynomial resources. Simulation claim, not optimization — don't cite as QUBO evidence.
- **CQM scale-up**: variable limits grown substantially, figures inconsistent across sources (500K vs 5M) — check `docs.dwavequantum.com` at implementation time. Irrelevant either way at this project's 100s-1000s scale.
- **Leap free tier persists** + new **LaunchPad program** (Jan 2025, 3-month free trial, fuller access). AWS Marketplace pay-as-you-go still works.
- **Company status mixed**: Nasdaq move (Jul 2026); Q2 2026 rev $3.1M flat, QCaaS +50% YoY, EBITDA loss +85% to $37.1M; ~$9B mkt cap vs ~$12M trailing rev (speculative); IDC MarketScape Leader (Jun/Jul 2026). Real growth + recognition, but small/unprofitable + big unproven gate-model bet — long-term platform risk.
- **E.ON grid-partitioning case study exists**, unverifiable in detail (no date/scale/numbers confirmable).

### IBM Quantum

- **Nighthawk (120q) + Loon (EC testbed) unveiled Nov 2025** — [IBM newsroom](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance). Roadmap: 7,500 2-qubit gates by 2026→15,000 by 2028. **Heron r2 (156q) is production HW today** — Nighthawk not deployed yet.
- **Cross-platform LR-QAOA benchmark** ran IBM Brisbane/Kyoto/Osaka/Fez to 109-156 qubits — real capability gain, but **Quantinuum ranked best** among tested QPUs on fully-connected graphs (grid QUBOs typically not sparse).
- [HSQC hybrid on Heron r3](https://arxiv.org/html/2603.13607) | 2026 | 156-var HUBO: matched ground state 14/20 instances, but classical parallel tempering/GPU still matched/beat overall. Authors: "no single solver dominates" — not an advantage claim.
- [QOBLIB](https://thequantuminsider.com/2025/04/10/a-decathalon-of-difficulty-working-group-benchmarks-the-limits-of-quantum-optimization/) (IBM+Zuse+Kipu) | Apr 2025 | Even IBM's Gambetta: quantum "not expected to solve efficiently in general."
- [HSBC+IBM bond trading](https://www.hsbc.com/news-and-views/news/media-releases/2025/hsbc-demonstrates-worlds-first-known-quantum-enabled-algorithmic-trading-with-ibm) | Sep 2025 | Real production-scale, 34% improvement — **classification task**, not combinatorial opt, not transferable to grid QUBO.
- Jul/Aug 2026 "advantage era" papers — sampling/dynamics, excluded (not optimization).

### Classiq

- **Classiq 1.0, Feb 2026** — [release](https://www.globenewswire.com/news-release/2026/02/11/3236142/0/en/Classiq-Releases-Quantum-Software-Engineering-Platform-Version-1-0.html). AI-assisted circuit gen, correct-by-construction, >10× compile speed claimed.
- **Much better capitalized**: $110M Series C (May 2025, largest quantum-software round ever), extended w/ AMD/Qualcomm/**IonQ**/Mirae — >$200M total. AWS Marketplace, NVIDIA CUDA-Q integration, NL-to-circuit agents.
- **Electric Grid Optimization example still tutorial-scale**: [docs.classiq.io](https://docs.classiq.io/latest/explore/applications/optimization/electric_grid_optimization/electric_grid_optimization/) — 3 sources × 4 consumers (12 binary vars), QAOA underperforms classical exact solver (3.4 vs 2.5 cost). No newer/larger grid demo found.

### IonQ / Quantinuum (trapped-ion)

- [**IonQ + Oak Ridge — direct grid UC demo**](https://www.ionq.com/news/ionq-partners-with-oak-ridge-national-laboratory-demonstrating-quantum-power) | Jul 2025 | Most directly on-topic find in whole survey (see pilots above).
- [IonQ/ORNL QITE demo](https://www.ionq.com/blog/ionq-and-oak-ridge-national-laboratory-demonstrate-a-novel-scalable-and) | Nov 2024 | 32-qubit, MaxCut — **vendor blog, own preprint, not independently verified**.
- **Quantinuum Helios, Nov 2025**, peer-reviewed *Nature* — 98 fully-connected qubits, 99.921% 2-qubit fidelity, 94 error-detected/48 fully error-corrected logical qubits. All-to-all connectivity good for dense grid QUBOs, still well below 1000s-var scale.
- **Independent benchmark ranks Quantinuum best** among gate-model QPUs for QAOA on full-connectivity (same LR-QAOA study; D-Wave excluded, gate-model-only). Strongest evidence a gate-model vendor pulling ahead for opt-relevant circuits specifically — still not beating classical solvers.
- Well-capitalized: ~$10B pre-money raise, IPO Jun 2026 >$14B valuation. No grid-specific case study found.

### Cloud access (AWS Braket, Azure Quantum)

- D-Wave not natively on Braket since Nov 2022 (predates window). Braket QPUs: AQT, IonQ, IQM, QuEra, Rigetti — no native annealing.
- Azure Quantum: **no D-Wave provider** (confirmed live, Apr 2026 docs), drop date unconfirmed. Current: IonQ, Pasqal, Quantinuum, Rigetti. Toshiba SQBM+ available as classical alternative in same UI.
- **Neither hyperscaler offers native D-Wave today** — unchanged from mid-2024. Leap direct/AWS Marketplace remains only path.

### PennyLane / QML

- Repositioning as general framework beyond QML (`qml`→`qp` import, Mar 2026); v0.43/Catalyst v0.13 added dynamic wire allocation, resource estimation, "quantum optimization with qjit."
- **Distinction**: tooling improvement (better QAOA-writing alternative to Qiskit/Classiq), not evidence QML *techniques* help QUBO. No evidence QML useful for combinatorial opt since mid-2024 — every substantive benchmark here uses QAOA/annealing/counterdiabatic, not QML. **Verdict unchanged: low priority.**

### Advantage claims on optimization — none survive

D-Wave *Science* sim claim (disputed), Munoz-Bauza & Lidar QA scaling claim on QUBO (PRL 134,160601/2401.07184 — rebutted by GPU-SBM), IBM 2026 "advantage era" (sampling/dynamics, excluded), unverified Pasqal materials claim (pending peer review). Google's Willow "Quantum Echoes" — molecular/NMR sim, also excluded.

---

## Recommendation (synthesis)

1. Keep D-Wave/Ocean SDK primary — no rival modality has verified unrebutted optimization win at this scale; Advantage2/Zephyr is real (independently-confirmed) embedding-overhead improvement over Pegasus-gen assumptions in this project's original notes.
2. **Read [2406.05916](https://arxiv.org/abs/2406.05916)** before touching `pp_to_microgrid.py`'s QUBO formulation — same lineage as cited paper, compact/lossless, fewer qubits.
3. Before trusting any "beats classical" number (this project's own or a paper's) — benchmark vs GPU quantum-inspired solver (SB-style) first. That's the real bar now, not CPU `SimulatedAnnealingSampler`.
4. Read **IonQ+ORNL UC** and **new anchor survey** (*Nat Rev Elec Eng*, May 2026) in full — most on-topic finds.
5. D-Wave financial risk (small rev, big unproven gate-model bet, speculative valuation) — real long-horizon consideration, periodic check-in not a reason to switch now.

---

*Related: [[02-ALGORITHMS]] · [[03-LITERATURE]] · [[06-ML-SOTA]] · [[07-TOOLING-UPDATES]] · [[versions/V0_SUMMARY]]*
