"""
Grid optimization algorithms.

Moved into the `algos/` package:

    from algos import QUBOProblem, UnitCommitmentProblem, Optimizer, optimize, benchmark

    q = QUBOProblem.from_microgrid(net, lambd=1)      # microgrid partitioning
    optimize(q, "simulated_annealing")               # or "gurobi" / "tabu" / "admm" / ...
    benchmark(q)                                     # run every applicable optimizer, compare

    uc = UnitCommitmentProblem.example()
    optimize(uc, "lagrangian")                       # dual decomposition
    optimize(uc, "gurobi")                           # exact MILP

Classical optimizers: brute_force, random_search, steepest_descent, simulated_annealing,
monte_carlo (parallel tempering), tabu, tree_decomposition (exact), gurobi (MIQP/MILP),
highs (scipy.optimize.milp), lagrangian, admm.  See docs/02-ALGORITHMS.md + algos/.

The QUBO formulation of microgrid partitioning still lives in pp_to_microgrid.py; algos/ wraps it.
"""
from algos import (QUBOProblem, MILPProblem, UnitCommitmentProblem, Generator,
                   Optimizer, optimize, benchmark, Result)  # noqa: F401
