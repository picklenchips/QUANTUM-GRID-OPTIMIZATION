# Literature review

> Condensed from original planning doc's "Papers/Brainstorming" section. Organized by topic, not stream-of-consciousness order; one-line takeaway per paper. Full math writeups were being ported to Overleaf (link lost — Womanium-era, not recovered); redo in-repo if needed rather than chasing old link. **Stops at ~mid-2024** — for everything since, see [05-QUANTUM-UPDATES.md](05-QUANTUM-UPDATES.md) (quantum) and [06-ML-SOTA.md](06-ML-SOTA.md) (classical ML baseline).

## Grid background — terms

- **Load shedding**: preventing generator overload via deliberate supply interruption (rolling blackouts).
- **On-line power flow (OLPF)**: steady-state power-system solution from real-time substation/feeder/bus data — current, voltage, phase, real/reactive flow, losses. Hard part: bidirectional flow from distributed renewables; needs stochastic engine at microgrid level even though DMS level is deterministic.
- **Short-circuit analysis (SCA)**: fault-current distribution for hypothetical faults. Microgrids introduce unbalanced multi-phase feeders; "phase domain analysis" via modified nodal admittance matrix — standard analytical approach. Related: FLISR (fault location, isolation, service restoration) — restore loads first, then microgrids.
- **Volt-VAR Optimization (VVO)**: control capacitor banks / tap positions to hold voltage + power factor in range while minimizing losses. Classically binary/discrete MIP; microgrids make it mixed-integer (discrete + continuous vars), coordinating control loops across timescales (sub-second microgrid controllers to minute-scale regulators). Should fail safe.
- **Unit commitment**: which generators run when, to meet demand at lowest cost/impact.
- **Load balancing**: real-time supply/demand matching.

Source survey: Argonne, [Interconnection, Integration, and Interactive Impact Analysis of Microgrids and Distribution Systems (2017)](https://anl.app.box.com/s/26igx0s67ixclkckhp2v2lulhpn307km) — argues current grid-management techniques don't scale to microgrid/renewable future.

## Classical numerical methods

- Newton-Raphson: baseline zero-finder underlying most classical power-flow solvers.
- [Mathematical Programming formulations for ACOPF](https://arxiv.org/pdf/2007.05334) (2020) — reference formulation; worth reproducing in-repo as ACOPF ground truth.
- [Numerical Performance of Different Formulations for ACOPF](https://arxiv.org/pdf/2107.07700) (2021), Sadat & Kim — interior-point linear solver; box-constrained variant improves on it.

## Optimal power flow — classical/ML

- [Synergizing Machine Learning with ACOPF](https://arxiv.org/html/2406.10428v1) (Jun 2024) — deep NN surrogates for ACOPF. 7× faster than interior-point solvers; physics-informed "OPF-DNN" (Lagrangian-constrained) needs less data, hits 104–10000× speedup vs. other ML models (next-best 63×) and 0.27% avg prediction error. Best at predicting voltage, worst at power factor. Flags field as early-stage — room to contribute.

## Grid optimization — quantum

- [Quantum Optimization for the Future Energy Grid: Summary and Quantum Utility Prospects](https://arxiv.org/pdf/2403.17495) (2024) — **anchor paper**. Surveys graph problems with exponential quantum speedup potential across grid stack:
  - Supply scheduling / dynamic demand response.
  - Microgrid formation — "self-sufficient energy community detection" (§2, QUBO) and "prosumer coalition formation" (§4, QAOA). Implemented in this repo.
  - New infrastructure placement (generation + battery siting, transmission line optimization).
  - Dynamic price incentivization for renewables — discount scheduling problem, §3, solved via D-Wave annealing.
  - P2P energy trading as game-theoretic graph problem (§5), quantum simulated annealing.
- [Optimal Power Flow Solutions via Noise-Resilient Quantum-Inspired Interior-Point Methods](https://arxiv.org/pdf/2311.02436) (Nov 2023) — three QIPM variants for **DC**OPF on NISQ hardware; DCOPF reformulated as linearly-constrained quadratic optimization. Uses HHL inside Newton descent, bottlenecked by linear-system resolution. Preliminary quadratic speedup for NT-QIPM, not CNT-QIPM. Open question: is DC simpler (no phase), or same AC/DC-gap problem in different formulation?
- [Quantum Computing for Power Flow Algorithms: Testing on Real Quantum Computers](https://arxiv.org/pdf/2204.14028) (2022) — linearized **AC**OPF solved via HHL, run on real NISQ hardware for 3-bus and 5-bus toy networks. Good "smallest possible demo" reference.
- [Quantum computing in power systems](https://ieeexplore.ieee.org/abstract/document/9831167) (IEEE, 2022) — broad application survey:
  - Quantum amplitude estimation (QAE) for stochastic grid reliability analysis, quadratic speedup over Monte Carlo ([2022 example](https://www.osti.gov/servlets/purl/1889646)).
  - Distributed unit commitment via quantum ADMM: decompose into QUBO subproblems, recombine via QAOA (Nikmehr, Zhang, Bragin, [IEEE TPWRS 2022](https://ieeexplore.ieee.org/document/9677977)); follow-up ([Feng et al., IEEE TPWRS 2023](https://ieeexplore.ieee.org/document/9793720)) replaces it with quantum surrogate Lagrangian relaxation, claims better convergence via contraction operator.
  - Quantum-secured distributed grid control via quantum-encrypted keys / photonic links ([2022](https://www.osti.gov/servlets/purl/1889649)).
- [Quantum-Enhanced Grid of the Future: A Primer](https://ieeexplore.ieee.org/abstract/document/9226502) (2020) — field-framing citation: grid modernization (DERs, renewables, electrification) has outrun classical computational tools; motivates quantum as infrastructure, not novelty.
- [QuEnergy](https://quantumconsortium.org/quenergy22/) / [QuEnergy Resilience](https://quantumconsortium.org/quenergy23/) (Quantum Economic Development Consortium) — unit commitment + load balancing framing; resilience track covers quantum-sensor anomaly detection and quantum-network cybersecurity for grid.
- Historical context (skip unless writing related-work section): [History of Optimal Power Flow and Formulations](https://www.ferc.gov/sites/default/files/2020-05/acopf-1-history-formulation-testing.pdf) (FERC, 2012).

## Quantum algorithms — general technique notes

- **QUBO ≡ Ising ground state**: any quadratic optimization problem maps to finding ground state of Ising Hamiltonian. Unifying trick behind nearly every paper above.
- [Optimization and energy management of a hybrid marine energy system via quantum computing](https://www.sciencedirect.com/science/article/pii/S0360544222010349) (2022) — quantum artificial bee colony (QABC) for scheduling, multi-objective quantum particle swarm optimization (QPSO). Marine-specific but metaheuristics generalize.
- [Quantum firefly swarms for multimodal dynamic optimization](https://www.sciencedirect.com/science/article/pii/S0957417418305153) (2023) — splits optimization into parallel "sub-swarms," combines via firefly-style attraction. Candidate technique for distributed/microgrid-parallel optimization.
- [Quantum computing for real-time building HVAC control](https://www.sciencedirect.com/science/article/pii/S0306261922018785) (2023) — QUBO formulation for nonlinear control problem (heat transfer dynamics), classically stuck with local PI controllers. Structurally similar to VVO above; possible formulation template.

## Quantum simulation (speculative offshoot)

- Long-duration battery storage material discovery (e.g. [liquid metal batteries](https://ambri.com/), ceramic insulation improvements) via quantum simulation. Unlikely near-term fit for this repo — noted for completeness, not pursued.

## ML for optimization (speculative ideas, not literature-backed)

- RL agent on grid simulation environment (actions = supply management, states = demand).
- Gradient boosting / decision-tree embeddings for supply scheduling.
- [BUTTER](https://github.com/NREL/BUTTER-Empirical-Deep-Learning-Experimental-Framework) (NREL) for ensemble-parallel model search.

---

*Related: [[02-ALGORITHMS]] · [[00-VISION]] · [[04-DATA-SOURCES]] · [[05-QUANTUM-UPDATES]] · [[06-ML-SOTA]] · [[versions/V0_SUMMARY]]*
