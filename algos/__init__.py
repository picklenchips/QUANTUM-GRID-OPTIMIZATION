"""
algos -- classical optimizers for the grid problems in this repo.

    from algos import QUBOProblem, UnitCommitmentProblem, Optimizer, optimize, benchmark

    q = QUBOProblem.from_microgrid(net, lambd=1)       # microgrid partitioning (pp_to_microgrid.py)
    Optimizer.list(q)                                  # which optimizers apply
    optimize(q, "simulated_annealing", sweeps=3000)    # run one

    uc = UnitCommitmentProblem.example()               # thermal unit commitment
    optimize(uc, "lagrangian")                         # dual decomposition
    benchmark(uc)                                      # run all applicable, compare

Problem kinds: 'qubo' (binary quadratic), 'milp' (mixed-integer linear/quadratic),
'uc' (unit commitment -- structured MILP with a per-generator decomposition).

Optimizers: brute_force, random_search, steepest_descent, simulated_annealing, monte_carlo,
tabu, tree_decomposition, gurobi, highs, lagrangian, admm.  See docs/02-ALGORITHMS.md.
"""
from .problems import (Result, QUBOProblem, MILPProblem, LinCon, Generator,
                       UnitCommitmentProblem, brute_force)
from .optimizers import Optimizer, optimize, register
from .benchmark import benchmark, plot_benchmark

__all__ = [
    "Result", "QUBOProblem", "MILPProblem", "LinCon", "Generator", "UnitCommitmentProblem",
    "brute_force", "Optimizer", "optimize", "register", "benchmark", "plot_benchmark",
]
