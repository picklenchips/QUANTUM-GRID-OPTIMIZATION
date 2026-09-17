# Algorithms — problem survey

> Condensed from original planning doc's "Grid Optimization Algos" + "Optimization Tools" sections. Each problem class: combine physics + math + CS to formulate and speed up with quantum/ML. Goal — implement basics, push one formulation past published state of the art. Paper detail + full citations: [03-LITERATURE.md](03-LITERATURE.md). Post-mid-2024 literature: [05-QUANTUM-UPDATES.md](05-QUANTUM-UPDATES.md) (quantum) + [06-ML-SOTA.md](06-ML-SOTA.md) (classical ML baseline).

## Problem classes

| Problem | Formulation | Status |
|---|---|---|
| **Microgrid formation** (self-sufficient energy communities, prosumer coalitions) | QUBO (community detection) / QAOA (coalition formation) | **Implemented, classical + quantum** — `pp_to_microgrid.py` builds the QUBO; `algos/` solves it with 14 optimizers (`QUBOProblem.from_microgrid`), including Qiskit `qaoa` / `vqe` / `numpy_min_eigen` — the same QUBO the D-Wave path anneals |
| **Dynamic pricing** (renewable incentivization, P2P energy trading) | QUBO / graph problem, quantum simulated annealing | Open — room to add pricing on top of microgrid formation |
| **Optimal power flow — AC** (ACOPF) | Nonlinear; ML surrogates, HHL-based linear solves | Open — core research question, see [00-VISION.md](00-VISION.md) |
| **Optimal power flow — DC** (DCOPF) | Linearly-constrained quadratic; simpler than AC (no phase) | Open — quantum interior-point methods (QIPM) are the lead |
| **Unit commitment** (supply scheduling) | MILP; Lagrangian relaxation / ADMM decomposition; QUBO subproblems + QAOA combination | **Classical implemented** — `algos.UnitCommitmentProblem` (MILP + per-generator DP); solvers: `lagrangian`, `admm`, `gurobi`, `highs`. Quantum path open |
| **Demand modeling** (variable/distributed supply) | RNN forecasting; nonlinear least squares for real-time synthesis | Open |
| **New infrastructure siting** | Plant/battery placement under budget + demand constraints; transmission line placement/capacity | Open — harder if substation placement also a variable |
| **Quantum networking** | Cybersecurity / secure grid control comms | Adjacent — not core optimization, but real differentiator (quantum-encrypted grid control) |
| **ML for data-center load** | VQL kernel in RL value function | Adjacent / stretch idea |

## Classical optimizers — `algos/`

`algos/` is the in-repo classical solver layer. Problems (`algos/problems.py`): `QUBOProblem`
(binary quadratic — `.from_microgrid(net, lambd)` wraps `pp_to_microgrid.py`'s objective),
`MILPProblem`, `UnitCommitmentProblem` (thermal UC as a MILP + a per-generator T-period DP for
decomposition). One `Optimizer` class family (`algos/optimizers.py`), dispatched by name:

| Optimizer | Applies to | Notes |
|---|---|---|
| `brute_force` | QUBO (n ≤ 22) | exact reference |
| `steepest_descent` | QUBO | multi-start 1-flip local search — fast, strong baseline |
| `simulated_annealing` | QUBO | numpy Metropolis w/ geometric schedule; `backend="dwave"` uses `dwave-samplers` |
| `monte_carlo` | QUBO | parallel tempering (geometric temperature ladder + replica swaps) |
| `tabu` | QUBO | `dwave-samplers` TabuSampler |
| `tree_decomposition` | QUBO | exact if the QUBO graph treewidth is low (`dwave-samplers`) |
| `gurobi` | QUBO / MILP / UC | MIQP / MILP — ships a size-limited trial license (no key needed for ≤2000 vars) |
| `highs` | QUBO / MILP / UC | open-source, via `scipy.optimize.milp` → HiGHS; QUBO McCormick-linearised for small n |
| `lagrangian` | UC | relax the demand-balance coupling → per-generator DP subproblems, subgradient on λ; gives a dual bound (≈0.6% gap on the example vs the MILP optimum) |
| `admm` | QUBO / UC | QUBO: box-relaxed `x` / binary `z` consensus split; UC: consensus across generators |
| `qaoa` | QUBO | **gate-model quantum** (Qiskit `QAOAAnsatz` on the Ising Hamiltonian) — the textbook algorithm this repo's D-Wave path is the *annealing* alternative to. Simulated exactly (statevector) while optimizing; slowest here since its circuit depth scales with QUBO density, not just qubit count |
| `vqe` | QUBO | Qiskit `VQE` + a shallow `RealAmplitudes` ansatz on the same Ising Hamiltonian — no QAOA-specific structure, much faster in simulation; a useful "generic quantum circuit" contrast to `qaoa` |
| `numpy_min_eigen` | QUBO (n ≤ 14) | exact ground state via Qiskit `NumPyMinimumEigensolver` — the quantum-stack reference `qaoa`/`vqe` are graded against (builds a 2^n statevector, so capped below `brute_force`) |

`from algos import benchmark; benchmark(problem)` runs every applicable optimizer and returns a
DataFrame (objective, runtime, optimality gap); `plot_benchmark(df)` → Plotly. Demo notebook:
[`../notebooks/classical_optimizers.ipynb`](../notebooks/classical_optimizers.ipynb).

**Qiskit implementation note** (2026-08): a correct-but-naive first pass handed the un-transpiled
`QAOAAnsatz`/ansatz straight to Qiskit's V2 `StatevectorEstimator` — it silently re-transpiles/
re-synthesises the circuit on *every* objective evaluation, ~100-1000x slower (a real run took
2.5 hours on 11 qubits). Fix: transpile once via `generate_preset_pass_manager`, reuse the ISA
circuit for every evaluation (ms, not minutes) — see `_run_vqe_like` in `algos/optimizers.py`.
Separately, `dimod`'s `BQM.to_ising()` does **not** guarantee `i<j` key order in its returned `J`
dict; code that only reads the upper triangle silently drops ~2/3 of the interaction terms —
`QUBOProblem.to_ising()` now normalises key order. Both were caught by comparing every quantum
result against `brute_force` on small instances before trusting the larger ones.

## Classical tools (external)

| Tool | Use |
|---|---|
| [MIPLearn](https://anl-ceeesa.github.io/MIPLearn/0.4/) (Argonne) | Mixed-integer LP + ML |
| [UnitCommitment.jl](https://github.com/ANL-CEEESA/UnitCommitment.jl) (Argonne) | Security-constrained unit commitment (SCUC), day-ahead market clearing |
| MATLAB/Simulink | Via OpenGridMap's [transnet matlab](https://github.com/OpenGridMap/transnet/tree/master/matlab) + [CIM2Simulink](https://github.com/OpenGridMap/CIM2Simulink) conversion |

## Quantum tools

| Tool | Use | Note |
|---|---|---|
| **D-Wave** (Ocean SDK) | Quantum annealing (QA) for QUBO — BQM, CQM | Current backend. Best-in-market tooling; QA empirically beats QAOA today even though QAOA is theoretically favored ([2023](https://arxiv.org/pdf/2301.00520)) |
| **IBM Qiskit** | QAOA | Not yet integrated — needs its own formulation work |
| **Classiq** | QAOA w/ Pyomo, circuit design/visualization | Best fit for designing quantum optimization circuit from scratch, or explaining one visually |
| **PennyLane** (QML) | — | Unclear if QML helps this problem class; low priority |

Any quadratic optimization problem reduces to finding the ground state of an Ising Hamiltonian — throughline connecting QUBO formulations across every problem class above.

---

*Related: [[00-VISION]] · [[03-LITERATURE]] · [[01-ARCHITECTURE]] · [[05-QUANTUM-UPDATES]] · [[06-ML-SOTA]] · [[versions/V0_SUMMARY]]*
