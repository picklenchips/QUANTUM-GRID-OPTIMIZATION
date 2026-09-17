"""
algos.optimizers -- classical optimizers for the problems in algos.problems.

    from algos import Optimizer, QUBOProblem
    p   = QUBOProblem.from_microgrid(net, lambd=1)
    res = Optimizer.get("simulated_annealing").solve(p, sweeps=2000)
    # or
    from algos import optimize
    res = optimize(p, "gurobi")

Every solver is an `Optimizer` subclass registered by name; `.solve(problem, **kw) -> Result`.
`Optimizer.list(problem)` gives the names that apply to a problem. Optional backends (Gurobi,
dwave-samplers) degrade gracefully -- `.available` is False and `.solve` raises a clear error.

Coverage: brute force, simulated annealing (numpy + dwave), Monte Carlo / parallel tempering,
tabu search, steepest descent, tree decomposition (exact), Gurobi (MIQP/MILP), HiGHS (scipy.milp),
Lagrangian relaxation (unit commitment), ADMM (QUBO consensus + UC generator split),
QAOA + SamplingVQE (Qiskit, gate-model quantum -- the same QUBO the D-Wave path anneals).
See docs/02-ALGORITHMS.md.
"""
from __future__ import annotations
import time, importlib.util
import numpy as np

from .problems import QUBOProblem, MILPProblem, UnitCommitmentProblem, LinCon, Result, brute_force

_REGISTRY: dict[str, type["Optimizer"]] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# --------------------------------------------------------------------------- base

class Optimizer:
    name: str = "optimizer"
    kinds: tuple[str, ...] = ()          # problem.kind values this optimizer handles
    available: bool = True               # False if an optional backend is missing

    def supports(self, problem) -> bool:
        return getattr(problem, "kind", None) in self.kinds

    def solve(self, problem, **kw) -> Result:               # pragma: no cover - abstract
        raise NotImplementedError

    # -- registry helpers --
    @staticmethod
    def get(name: str) -> "Optimizer":
        if name not in _REGISTRY:
            raise KeyError(f"unknown optimizer {name!r}; have {sorted(_REGISTRY)}")
        return _REGISTRY[name]()

    @staticmethod
    def list(problem=None) -> list[str]:
        out = []
        for name, cls in sorted(_REGISTRY.items()):
            inst = cls()
            if problem is not None and not inst.supports(problem):
                continue
            out.append(name + ("" if inst.available else " (backend missing)"))
        return out

    @staticmethod
    def registry() -> dict[str, type["Optimizer"]]:
        return dict(_REGISTRY)


def optimize(problem, method: str = "auto", **kw) -> Result:
    """ convenience: run one named optimizer (or the first applicable one for method='auto'). """
    if method == "auto":
        for name in ("gurobi", "tree_decomposition", "tabu", "simulated_annealing", "highs", "lagrangian"):
            opt = Optimizer.get(name)
            if opt.supports(problem) and opt.available:
                method = name
                break
        else:
            raise RuntimeError(f"no applicable optimizer for {problem!r}")
    return Optimizer.get(method).solve(problem, **kw)


# --------------------------------------------------------------------------- QUBO: exact / local

@register
class BruteForce(Optimizer):
    name, kinds = "brute_force", ("qubo",)

    def solve(self, problem: QUBOProblem, max_n: int = 22, **kw) -> Result:
        return brute_force(problem, max_n=max_n)


@register
class RandomSearch(Optimizer):
    name, kinds = "random_search", ("qubo",)

    def solve(self, problem: QUBOProblem, samples: int = 20000, seed: int = 0, **kw) -> Result:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        best_x, best_e, trace = None, np.inf, []
        for k in range(samples):
            x = rng.integers(0, 2, problem.n).astype(float)
            e = problem.evaluate(x)
            if e < best_e:
                best_e, best_x = e, x
            if k % max(1, samples // 100) == 0:
                trace.append(best_e)
        return Result(best_x, best_e, self.name, True, time.perf_counter() - t0, samples, trace=trace)


@register
class SteepestDescent(Optimizer):
    """ greedy 1-flip local search from `restarts` random starts (a fast, strong baseline). """
    name, kinds = "steepest_descent", ("qubo",)

    def solve(self, problem: QUBOProblem, restarts: int = 20, seed: int = 0, **kw) -> Result:
        rng = np.random.default_rng(seed)
        Q = problem.Q
        t0 = time.perf_counter()
        best_x, best_e, iters = None, np.inf, 0
        for _ in range(restarts):
            x = rng.integers(0, 2, problem.n).astype(float)
            qx = Q @ x
            improved = True
            while improved:
                improved = False
                d = 1 - 2 * x                                  # flip direction per bit
                dE = 2 * d * qx + np.diag(Q)
                k = int(np.argmin(dE))
                iters += 1
                if dE[k] < -1e-12:
                    x[k] = 1 - x[k]
                    qx += d[k] * Q[:, k]
                    improved = True
            e = problem.evaluate(x)
            if e < best_e:
                best_e, best_x = e, x
        return Result(best_x, best_e, self.name, True, time.perf_counter() - t0, iters)


# --------------------------------------------------------------------------- QUBO: annealing / MC

def _auto_temps(Q: np.ndarray) -> tuple[float, float]:
    scale = np.abs(Q).sum(axis=1)
    hi = float(np.max(scale)) or 1.0
    lo = float(np.median(scale[scale > 0])) if np.any(scale > 0) else 1.0
    return max(hi, 1.0), max(lo * 1e-2, 1e-3)


@register
class SimulatedAnnealing(Optimizer):
    name, kinds = "simulated_annealing", ("qubo",)

    def solve(self, problem: QUBOProblem, sweeps: int = 2000, restarts: int = 4,
              T0: float | None = None, T1: float | None = None, seed: int = 0,
              backend: str = "numpy", **kw) -> Result:
        if backend == "dwave":
            return _dwave_sampler("SimulatedAnnealingSampler", problem, self.name,
                                  num_reads=restarts, num_sweeps=sweeps, seed=seed)
        rng = np.random.default_rng(seed)
        Q, n = problem.Q, problem.n
        diag = np.diag(Q).copy()
        a, b = _auto_temps(Q)
        T0 = a if T0 is None else T0
        T1 = b if T1 is None else T1
        schedule = T0 * (T1 / T0) ** (np.arange(sweeps) / max(sweeps - 1, 1))
        best_x, best_e, trace, tot = None, np.inf, [], 0
        t0 = time.perf_counter()
        for _ in range(restarts):
            x = rng.integers(0, 2, n).astype(float)
            qx = Q @ x
            e = float(x @ qx + problem.offset)
            for s in range(sweeps):
                Temp = schedule[s]
                order = rng.permutation(n)
                for k in order:
                    d = 1.0 - 2.0 * x[k]
                    dE = 2.0 * d * qx[k] + diag[k]
                    if dE <= 0 or rng.random() < np.exp(-dE / Temp):
                        x[k] = 1.0 - x[k]
                        qx += d * Q[:, k]
                        e += dE
                    tot += 1
                if e < best_e:
                    best_e, best_x = e, x.copy()
                if s % max(1, sweeps // 100) == 0:
                    trace.append(best_e)
        return Result(best_x, best_e, self.name, True, time.perf_counter() - t0, tot,
                      trace=trace, meta={"T0": T0, "T1": T1, "restarts": restarts})


@register
class MonteCarlo(Optimizer):
    """ parallel-tempering Metropolis: `replicas` chains at geometric temperatures with swap moves. """
    name, kinds = "monte_carlo", ("qubo",)

    def solve(self, problem: QUBOProblem, sweeps: int = 3000, replicas: int = 8,
              Tmin: float | None = None, Tmax: float | None = None, seed: int = 0, **kw) -> Result:
        rng = np.random.default_rng(seed)
        Q, n = problem.Q, problem.n
        diag = np.diag(Q).copy()
        a, b = _auto_temps(Q)
        Tmax = a if Tmax is None else Tmax
        Tmin = b if Tmin is None else Tmin
        temps = Tmin * (Tmax / Tmin) ** (np.arange(replicas) / max(replicas - 1, 1))
        X = rng.integers(0, 2, (replicas, n)).astype(float)
        QX = X @ Q.T
        E = np.einsum("ri,ri->r", X, QX) + problem.offset
        best_x, best_e, trace, acc, tot = X[0].copy(), np.inf, [], 0, 0
        t0 = time.perf_counter()
        for s in range(sweeps):
            for r in range(replicas):
                for k in rng.permutation(n):
                    d = 1.0 - 2.0 * X[r, k]
                    dE = 2.0 * d * QX[r, k] + diag[k]
                    if dE <= 0 or rng.random() < np.exp(-dE / temps[r]):
                        X[r, k] = 1.0 - X[r, k]
                        QX[r] += d * Q[:, k]
                        E[r] += dE
                        acc += 1
                    tot += 1
            for r in range(replicas - 1):                       # adjacent-temperature swaps
                delta = (1.0 / temps[r] - 1.0 / temps[r + 1]) * (E[r] - E[r + 1])
                if delta > 0 or rng.random() < np.exp(delta):
                    X[[r, r + 1]] = X[[r + 1, r]]
                    QX[[r, r + 1]] = QX[[r + 1, r]]
                    E[[r, r + 1]] = E[[r + 1, r]]
            i = int(np.argmin(E))
            if E[i] < best_e:
                best_e, best_x = float(E[i]), X[i].copy()
            if s % max(1, sweeps // 100) == 0:
                trace.append(best_e)
        return Result(best_x, best_e, self.name, True, time.perf_counter() - t0, tot,
                      trace=trace, meta={"acceptance": acc / max(tot, 1), "replicas": replicas})


# --------------------------------------------------------------------------- QUBO: dwave-samplers

def _dwave_sampler(cls_name: str, problem: QUBOProblem, method: str, **params) -> Result:
    if not _have("dwave.samplers"):
        raise ImportError("pip install dwave-samplers (bundled in dwave-ocean-sdk) for " + method)
    import dwave.samplers as ds
    sampler = getattr(ds, cls_name)()
    bqm = problem.to_bqm()
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, **{k: v for k, v in params.items() if v is not None})
    dt = time.perf_counter() - t0
    best = ss.first
    x = np.array([best.sample[i] for i in range(problem.n)], dtype=float)
    return Result(x, problem.evaluate(x), method, True, dt, params.get("num_reads", 1) or 1,
                  meta={"sampler": cls_name})


@register
class TabuSearch(Optimizer):
    name, kinds = "tabu", ("qubo",)
    available = _have("dwave.samplers")

    def solve(self, problem: QUBOProblem, num_reads: int = 10, tenure: int | None = None, **kw) -> Result:
        return _dwave_sampler("TabuSampler", problem, self.name, num_reads=num_reads, tenure=tenure)


@register
class TreeDecomposition(Optimizer):
    """ exact via dynamic programming on a tree decomposition -- only if the QUBO graph treewidth is low. """
    name, kinds = "tree_decomposition", ("qubo",)
    available = _have("dwave.samplers")

    def solve(self, problem: QUBOProblem, **kw) -> Result:
        r = _dwave_sampler("TreeDecompositionSolver", problem, self.name)
        r.bound = r.objective                    # exact
        return r


# --------------------------------------------------------------------------- Gurobi (MIQP / MILP)

@register
class GurobiOptimizer(Optimizer):
    name, kinds = "gurobi", ("qubo", "milp", "uc")
    available = _have("gurobipy")

    def solve(self, problem, time_limit: float = 30.0, mip_gap: float = 1e-4, verbose: bool = False, **kw) -> Result:
        if not self.available:
            raise ImportError("pip install gurobipy (ships a size-limited trial license) for the Gurobi backend")
        import gurobipy as gp
        from gurobipy import GRB
        if isinstance(problem, UnitCommitmentProblem):
            milp = problem.build_milp()
            res = self._milp(milp, gp, GRB, time_limit, mip_gap, verbose)
            res.method = self.name
            npv = problem.G * problem.T
            res.x = res.x[:npv].reshape(problem.G, problem.T)      # dispatch straight from the MILP
            res.objective = problem.evaluate(res.x)
            res.feasible = problem.is_feasible(res.x)
            return res
        if isinstance(problem, MILPProblem):
            return self._milp(problem, gp, GRB, time_limit, mip_gap, verbose)
        if isinstance(problem, QUBOProblem):
            return self._qubo(problem, gp, GRB, time_limit, mip_gap, verbose)
        raise TypeError(problem)

    def _qubo(self, p: QUBOProblem, gp, GRB, tl, gap, verbose):
        m = gp.Model("qubo"); m.Params.OutputFlag = int(verbose)
        m.Params.TimeLimit = tl; m.Params.MIPGap = gap
        x = m.addMVar(p.n, vtype=GRB.BINARY)
        m.setObjective(x @ p.Q @ x + p.offset, GRB.MINIMIZE)
        t0 = time.perf_counter(); m.optimize(); dt = time.perf_counter() - t0
        xv = np.round(x.X).astype(float)
        return Result(xv, p.evaluate(xv), self.name, True, dt, m.NodeCount,
                      bound=m.ObjBound if m.SolCount else None,
                      meta={"status": m.Status, "mip_gap": getattr(m, "MIPGap", None)})

    def _milp(self, p: MILPProblem, gp, GRB, tl, gap, verbose):
        m = gp.Model("milp"); m.Params.OutputFlag = int(verbose)
        m.Params.TimeLimit = tl; m.Params.MIPGap = gap
        vtype = {"C": GRB.CONTINUOUS, "I": GRB.INTEGER, "B": GRB.BINARY}
        x = [m.addVar(lb=p.lb[i], ub=p.ub[i], vtype=vtype[p.kinds[i]]) for i in range(p.n)]
        obj = gp.quicksum(p.c[i] * x[i] for i in range(p.n)) + p.const
        if p.Qobj is not None:
            obj += gp.quicksum(p.Qobj[i, j] * x[i] * x[j] for i in range(p.n) for j in range(p.n) if p.Qobj[i, j])
        m.setObjective(obj, GRB.MINIMIZE)
        for con in p.cons:
            expr = gp.quicksum(con.A[i] * x[i] for i in np.nonzero(con.A)[0])
            m.addConstr(expr <= con.b if con.sense == "<=" else
                        (expr >= con.b if con.sense == ">=" else expr == con.b))
        t0 = time.perf_counter(); m.optimize(); dt = time.perf_counter() - t0
        xv = np.array([v.X for v in x])
        return Result(xv, p.evaluate(xv), self.name, p.is_feasible(xv), dt, m.NodeCount,
                      bound=m.ObjBound if m.SolCount else None, meta={"status": m.Status})

    @staticmethod
    def _uc_commitment(uc: UnitCommitmentProblem, x_milp: np.ndarray) -> np.ndarray:
        npv = uc.G * uc.T
        return np.round(x_milp[npv:2 * npv]).reshape(uc.G, uc.T)


# --------------------------------------------------------------------------- HiGHS via scipy.optimize.milp

@register
class HiGHS(Optimizer):
    """ open-source MILP (scipy.optimize.milp -> HiGHS). QUBO is linearised (McCormick) for small n. """
    name, kinds = "highs", ("qubo", "milp", "uc")

    def solve(self, problem, qubo_linearize_max: int = 60, time_limit: float = 30.0, **kw) -> Result:
        from scipy.optimize import milp, LinearConstraint, Bounds
        if isinstance(problem, UnitCommitmentProblem):
            milp_p = problem.build_milp()
            res = self._run(milp_p, milp, LinearConstraint, Bounds, time_limit)
            npv = problem.G * problem.T
            res.x = res.x[:npv].reshape(problem.G, problem.T)      # dispatch straight from the MILP
            res.objective = problem.evaluate(res.x)
            res.feasible = problem.is_feasible(res.x)
            res.method = self.name
            return res
        if isinstance(problem, MILPProblem):
            return self._run(problem, milp, LinearConstraint, Bounds, time_limit)
        if isinstance(problem, QUBOProblem):
            if problem.n > qubo_linearize_max:
                raise ValueError(f"HiGHS QUBO linearisation refused for n={problem.n} > {qubo_linearize_max}")
            return self._run(_linearize_qubo(problem), milp, LinearConstraint, Bounds, time_limit, base=problem)
        raise TypeError(problem)

    def _run(self, p: MILPProblem, milp, LinearConstraint, Bounds, time_limit, base=None):
        cons = []
        for con in p.cons:
            lo = -np.inf if con.sense == "<=" else con.b
            hi = np.inf if con.sense == ">=" else con.b
            cons.append(LinearConstraint(con.A, lo, hi))
        integrality = np.array([0 if k == "C" else 1 for k in p.kinds])
        t0 = time.perf_counter()
        r = milp(p.c, constraints=cons, bounds=Bounds(p.lb, p.ub), integrality=integrality,
                 options={"time_limit": time_limit})
        dt = time.perf_counter() - t0
        x = r.x if r.x is not None else np.zeros(p.n)
        prob = base or p
        xx = x[:prob.n] if base is not None else x
        return Result(np.asarray(xx, float), prob.evaluate(xx), self.name, bool(r.success), dt, 0,
                      bound=getattr(r, "mip_dual_bound", None), meta={"message": r.message})


def _linearize_qubo(p: QUBOProblem) -> MILPProblem:
    """ y_ij = x_i x_j for i<j via McCormick (x binary): y<=x_i, y<=x_j, y>=x_i+x_j-1, y>=0. """
    n = p.n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if p.Q[i, j] + p.Q[j, i]]
    pidx = {pr: n + k for k, pr in enumerate(pairs)}
    N = n + len(pairs)
    c = np.zeros(N)
    for i in range(n):
        c[i] = p.Q[i, i]
    for (i, j), k in pidx.items():
        c[k] = p.Q[i, j] + p.Q[j, i]
    cons: list[LinCon] = []
    for (i, j), k in pidx.items():
        A1 = np.zeros(N); A1[k] = 1; A1[i] = -1; cons.append(LinCon(A1, "<=", 0))
        A2 = np.zeros(N); A2[k] = 1; A2[j] = -1; cons.append(LinCon(A2, "<=", 0))
        A3 = np.zeros(N); A3[k] = -1; A3[i] = 1; A3[j] = 1; cons.append(LinCon(A3, "<=", 1))
    lb, ub = np.zeros(N), np.ones(N)
    kinds = ["B"] * n + ["C"] * len(pairs)
    return MILPProblem(c, cons, lb, ub, kinds, const=p.offset, name=p.name + "_linearized")


# --------------------------------------------------------------------------- Lagrangian relaxation (UC)

@register
class LagrangianRelaxation(Optimizer):
    """ relax the per-period demand balance with multipliers lambda_t; the problem decouples into
    one single-unit T-period DP per generator. Subgradient ascent on lambda gives a lower bound;
    the primal comes from economic dispatch on the (repaired) commitment. """
    name, kinds = "lagrangian", ("uc",)

    def solve(self, uc: UnitCommitmentProblem, iters: int = 200, step0: float = 2.0,
              seed: int = 0, **kw) -> Result:
        T, G = uc.T, uc.G
        lam = np.zeros(T)
        best_lb, best_primal, best_p, trace = -np.inf, np.inf, None, []
        t0 = time.perf_counter()
        for it in range(iters):
            P = np.zeros((G, T)); sub_val = 0.0
            for g in range(G):
                p_g, v = uc.gen_subproblem(g, lam)
                P[g] = p_g; sub_val += v
            lb = sub_val + float(lam @ uc.demand)          # L(lambda)
            best_lb = max(best_lb, lb)
            g_sub = uc.demand - P.sum(axis=0)              # subgradient of the dual
            # primal repair: commitment from subproblems, then economic dispatch
            u = (P > 1e-6).astype(int)
            u = self._repair(uc, u)
            p_feas = uc.dispatch_from_commitment(u)
            if uc.is_feasible(p_feas):
                pc = uc.evaluate(p_feas)
                if pc < best_primal:
                    best_primal, best_p = pc, p_feas
            step = step0 / (1 + 0.05 * it)                  # diminishing step
            lam += step * g_sub / (np.linalg.norm(g_sub) + 1e-9)
            lam = np.clip(lam, 0, None)
            trace.append(best_primal if np.isfinite(best_primal) else lb)
        if best_p is None:                                 # fall back: commit everything
            best_p = uc.dispatch_from_commitment(np.ones((G, T)))
            best_primal = uc.evaluate(best_p)
        return Result(best_p, best_primal, self.name, uc.is_feasible(best_p),
                      time.perf_counter() - t0, iters, bound=best_lb, trace=trace,
                      meta={"dual_bound": best_lb})

    @staticmethod
    def _repair(uc: UnitCommitmentProblem, u: np.ndarray) -> np.ndarray:
        """ ensure each period can meet demand: switch on cheapest available units until capacity suffices. """
        u = u.copy()
        for t in range(uc.T):
            cap = sum(uc.gens[g].pmax for g in range(uc.G) if u[g, t])
            order = sorted(range(uc.G), key=lambda g: uc.gens[g].cost)
            for g in order:
                if cap >= uc.demand[t]:
                    break
                if not u[g, t]:
                    u[g, t] = 1
                    cap += uc.gens[g].pmax
        return u


# --------------------------------------------------------------------------- ADMM

@register
class ADMM(Optimizer):
    """ alternating direction method of multipliers.

    QUBO:  min x^T Q x  s.t. x in {0,1}   ->  split  x (box-relaxed) = z (binary).
           x-update solves (2Q + rho I) x = rho(z - u) then clips to [0,1];  z-update rounds.
    UC:    consensus over generators sharing the demand-balance -- each unit does its own dispatch,
           z-update projects the aggregate onto demand.
    """
    name, kinds = "admm", ("qubo", "uc")

    def solve(self, problem, rho: float = 1.0, iters: int = 300, seed: int = 0, **kw) -> Result:
        if isinstance(problem, QUBOProblem):
            return self._qubo(problem, rho, iters, seed)
        if isinstance(problem, UnitCommitmentProblem):
            return self._uc(problem, rho, iters)
        raise TypeError(problem)

    def _qubo(self, p: QUBOProblem, rho, iters, seed):
        rng = np.random.default_rng(seed)
        n = p.n
        # rho must make (2Q + rho I) positive definite for the box-QP x-update to be well posed
        lam_min = float(np.linalg.eigvalsh(p.Q).min())
        rho = max(rho, -2.0 * lam_min + 1.0)
        Ainv = np.linalg.inv(2 * p.Q + rho * np.eye(n))
        z = rng.integers(0, 2, n).astype(float)
        u = np.zeros(n)
        best_x, best_e, trace = z.copy(), p.evaluate(z), []
        t0 = time.perf_counter()
        for it in range(iters):
            x = np.clip(Ainv @ (rho * (z - u)), 0, 1)      # box-relaxed x-update
            z = (x + u > 0.5).astype(float)                # binary z-update (proximal of the {0,1} indicator)
            u += x - z
            # also try the sign-rounded relaxed point (helps escape z stalls)
            for cand in (z, (x > 0.5).astype(float)):
                e = p.evaluate(cand)
                if e < best_e:
                    best_e, best_x = e, cand.copy()
            trace.append(best_e)
        return Result(best_x, best_e, self.name, True, time.perf_counter() - t0, iters,
                      trace=trace, meta={"rho": rho})

    def _uc(self, uc: UnitCommitmentProblem, rho, iters):
        G, T = uc.G, uc.T
        Z = np.zeros((G, T)); W = np.zeros((G, T))
        # feasible warm start: proportional to pmax
        share = np.array([g.pmax for g in uc.gens]); share = share / share.sum()
        P = np.outer(share, uc.demand)
        best_p, best_e, trace = None, np.inf, []
        t0 = time.perf_counter()
        for it in range(iters):
            # x-update: each unit minimises cost_g p + (rho/2)||p - (Z - W)||^2 over [pmin,pmax] (u fixed on)
            for g, gen in enumerate(uc.gens):
                target = Z[g] - W[g]
                P[g] = np.clip(target - gen.cost / rho, gen.pmin, gen.pmax)
            # z-update: project the aggregate onto sum_g z_{.,t} = demand_t
            PW = P + W
            adj = (uc.demand - PW.sum(axis=0)) / G
            Z = PW + adj
            W += P - Z
            u = (P > 1e-6).astype(int)
            p_feas = uc.dispatch_from_commitment(np.ones((G, T)) if not u.any() else u)
            if uc.is_feasible(p_feas):
                e = uc.evaluate(p_feas)
                if e < best_e:
                    best_e, best_p = e, p_feas
            trace.append(best_e if np.isfinite(best_e) else np.nan)
        if best_p is None:
            best_p = uc.dispatch_from_commitment(np.ones((G, T)))
            best_e = uc.evaluate(best_p)
        return Result(best_p, best_e, self.name, uc.is_feasible(best_p),
                      time.perf_counter() - t0, iters, trace=trace, meta={"rho": rho})


# --------------------------------------------------------------------------- Qiskit (gate model)

_HAVE_QISKIT = _have("qiskit") and _have("qiskit_algorithms")


def _ising_hamiltonian(p: QUBOProblem):
    """ QUBOProblem -> (SparsePauliOp, constant) via x=(s+1)/2 (dimod's convention, see to_ising()). """
    from qiskit.quantum_info import SparsePauliOp
    h, J, c = p.to_ising()
    n = p.n
    terms = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            s = ["I"] * n; s[n - 1 - i] = "Z"           # Qiskit is little-endian: qubit 0 = rightmost char
            terms.append(("".join(s), float(h[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                s = ["I"] * n; s[n - 1 - i] = "Z"; s[n - 1 - j] = "Z"
                terms.append(("".join(s), float(J[i, j])))
    if not terms:
        terms = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(terms), float(c)


def _classical_optimizer(name: str, maxiter: int):
    import qiskit_algorithms.optimizers as O
    return {"COBYLA": lambda: O.COBYLA(maxiter=maxiter),
            "SPSA":   lambda: O.SPSA(maxiter=maxiter),
            "NELDER_MEAD": lambda: O.NELDER_MEAD(maxiter=maxiter),
            "L_BFGS_B": lambda: O.L_BFGS_B(maxiter=maxiter)}.get(name, lambda: O.COBYLA(maxiter=maxiter))()


def _run_vqe_like(problem: QUBOProblem, ansatz, method: str, maxiter: int, optimizer: str,
                  backend: str, shots: int | None, seed: int) -> Result:
    """ shared engine for QAOA and VQE: minimise <ansatz(theta)|H|ansatz(theta)> with an exact
    (statevector) Estimator -- fast and noise-free, the standard way to prototype these algorithms
    classically -- then sample the optimised circuit once to read out the best bitstring.

    IMPORTANT perf note: qiskit's V2 Estimator silently re-transpiles/re-synthesises an
    un-transpiled parameterised circuit (e.g. QAOAAnsatz's PauliEvolutionGate layers) on *every*
    call if you hand it the raw ansatz -- ~100x-1000x slower. Transpiling once up front (below) and
    reusing that ISA circuit for every objective evaluation is what makes this fast (ms, not minutes). """
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_algorithms import VQE

    H, offset = _ising_hamiltonian(problem)
    if not np.any(np.abs(H.coeffs)):                      # degenerate all-zero objective: nothing to optimize
        x0 = np.zeros(problem.n)
        return Result(x0, problem.evaluate(x0), method, True, 0.0, 0, bound=problem.evaluate(x0),
                      meta={"qubits": problem.n, "degenerate": True})
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["cx", "rz", "sx", "x"])
    isa_ansatz = pm.run(ansatz)
    isa_H = H.apply_layout(isa_ansatz.layout)

    trace: list[float] = []
    best = [np.inf]

    def cb(eval_count, params, mean, meta):
        best[0] = min(best[0], float(mean) + offset)
        trace.append(best[0])

    vqe = VQE(StatevectorEstimator(), isa_ansatz,
             _classical_optimizer(optimizer, maxiter), callback=cb)
    t0 = time.perf_counter()
    eig = vqe.compute_minimum_eigenvalue(isa_H)
    dual_bound = float(eig.eigenvalue.real) + offset

    # one-shot readout: bind the ORIGINAL (untranspiled, logical-qubit-ordered) ansatz and sample it
    bound = ansatz.assign_parameters(eig.optimal_point)
    bound.measure_all()
    if backend == "aer" and _have("qiskit_aer"):
        from qiskit_aer.primitives import SamplerV2 as _Sampler
    else:
        _Sampler = StatevectorSampler
    counts = _Sampler(seed=seed).run([bound], shots=shots or 2048).result()[0].data.meas.get_counts()
    dt = time.perf_counter() - t0

    best_x, best_e = None, np.inf
    for bitstr, _cnt in counts.items():
        b = np.array([int(c) for c in bitstr[::-1]], dtype=float)[:problem.n]
        # Qiskit's Z-eigenvalue convention (bit 0 -> spin +1) is the opposite sign of dimod's
        # to_ising() (x=1 -> spin +1) -- rather than trust one hand-derived sign, just score both
        # the measured bitstring and its global complement and keep whichever is the true QUBO min.
        for x in (b, 1.0 - b):
            e = problem.evaluate(x)
            if e < best_e:
                best_e, best_x = e, x
    return Result(best_x, best_e, method, problem.is_feasible(best_x), dt, len(trace),
                  bound=dual_bound, trace=trace,
                  meta={"qubits": problem.n, "optimizer": optimizer, "backend": backend,
                        "distinct_samples": len(counts)})


@register
class QAOA(Optimizer):
    """ Quantum Approximate Optimization Algorithm (Qiskit) on the microgrid / QUBO Ising Hamiltonian
    -- the same problem the D-Wave path anneals, here a p-layer (`reps`) parameterised circuit whose
    angles a classical optimizer tunes to minimise <H>. Simulated exactly (statevector) during
    optimization; the final answer is read out by sampling the optimised circuit once.

    SLOWEST of the three Qiskit optimizers here, and not just from qubit count: QAOAAnsatz's cost
    layer has one multi-qubit rotation *per nonzero Ising term*, so runtime scales with how dense
    the QUBO is, not n alone (a 20-qubit sparse QUBO can be faster than an 11-qubit dense one).
    `vqe`'s RealAmplitudes ansatz is shallow regardless of Hamiltonian density -- prefer it for
    anything denser than a handful of qubits; reach for `qaoa` when the *algorithm itself* is the
    point of comparison. """
    name, kinds = "qaoa", ("qubo",)
    available = _HAVE_QISKIT
    max_qubits = 20

    def solve(self, problem: QUBOProblem, reps: int = 1, maxiter: int = 60,
              optimizer: str = "COBYLA", backend: str = "statevector", shots: int | None = None,
              seed: int = 0, **kw) -> Result:
        if not self.available:
            raise ImportError("pip install qiskit qiskit-algorithms for QAOA")
        if problem.n > self.max_qubits:
            raise ValueError(f"QAOA refuses n={problem.n} qubits > {self.max_qubits} "
                             f"(statevector sim); reduce the network or raise .max_qubits")
        from qiskit.circuit.library import QAOAAnsatz
        H, _ = _ising_hamiltonian(problem)
        ansatz = QAOAAnsatz(H, reps=reps)
        r = _run_vqe_like(problem, ansatz, self.name, maxiter, optimizer, backend, shots, seed)
        r.meta["reps"] = reps
        return r


@register
class QiskitVQE(Optimizer):
    """ VQE with a hardware-efficient RealAmplitudes ansatz (Qiskit) on the same Ising Hamiltonian --
    no QAOA-specific structure, just a generic trainable circuit; a useful contrast to `qaoa`. """
    name, kinds = "vqe", ("qubo",)
    available = _HAVE_QISKIT
    max_qubits = 20

    def solve(self, problem: QUBOProblem, reps: int = 2, maxiter: int = 150,
              optimizer: str = "COBYLA", backend: str = "statevector", shots: int | None = None,
              seed: int = 0, **kw) -> Result:
        if not self.available:
            raise ImportError("pip install qiskit qiskit-algorithms for VQE")
        if problem.n > self.max_qubits:
            raise ValueError(f"VQE refuses n={problem.n} qubits > {self.max_qubits}")
        from qiskit.circuit.library import RealAmplitudes
        ansatz = RealAmplitudes(problem.n, reps=reps)
        r = _run_vqe_like(problem, ansatz, self.name, maxiter, optimizer, backend, shots, seed)
        r.meta.update({"reps": reps, "ansatz": "RealAmplitudes"})
        return r


@register
class QiskitExact(Optimizer):
    """ exact ground state via Qiskit's NumPyMinimumEigensolver on the Ising Hamiltonian -- the
    reference `qaoa` / `vqe` are graded against. Builds a 2^n-dim statevector, so this is capped
    well below `brute_force`'s reach (which just enumerates bitstrings, no linear algebra). """
    name, kinds = "numpy_min_eigen", ("qubo",)
    available = _HAVE_QISKIT
    max_qubits = 14

    def solve(self, problem: QUBOProblem, **kw) -> Result:
        if not self.available:
            raise ImportError("pip install qiskit qiskit-algorithms for numpy_min_eigen")
        if problem.n > self.max_qubits:
            raise ValueError(f"numpy_min_eigen refuses n={problem.n} > {self.max_qubits} (dense "
                             f"eigensolve); use 'brute_force' for larger exact QUBO reference")
        from qiskit.quantum_info import Statevector
        from qiskit_algorithms import NumPyMinimumEigensolver
        H, offset = _ising_hamiltonian(problem)
        if not np.any(np.abs(H.coeffs)):
            x0 = np.zeros(problem.n)
            return Result(x0, problem.evaluate(x0), self.name, True, 0.0, 0,
                          bound=problem.evaluate(x0), meta={"qubits": problem.n, "degenerate": True})
        t0 = time.perf_counter()
        eig = NumPyMinimumEigensolver().compute_minimum_eigenvalue(H)
        dt = time.perf_counter() - t0
        sv = Statevector(eig.eigenstate.to_matrix() if hasattr(eig.eigenstate, "to_matrix")
                         else np.asarray(eig.eigenstate))
        probs = sv.probabilities_dict()
        best_x, best_e = None, np.inf
        for bitstr, prob in probs.items():                # ground state may be (near-)degenerate
            if prob < 1e-9:
                continue
            b = np.array([int(c) for c in bitstr[::-1]], dtype=float)[:problem.n]
            for x in (b, 1.0 - b):                         # see _run_vqe_like: qiskit/dimod spin-sign mismatch
                e = problem.evaluate(x)
                if e < best_e:
                    best_e, best_x = e, x
        return Result(best_x, best_e, self.name, problem.is_feasible(best_x), dt, 0, bound=best_e,
                      meta={"qubits": problem.n})


# --------------------------------------------------------------------------- module-level demo

def _demo():
    print("registered optimizers:", sorted(_REGISTRY))

    q = QUBOProblem.random(n=14, seed=1)
    print("\nQUBO n=14 -- exact:", brute_force(q))
    for name in ("steepest_descent", "simulated_annealing", "monte_carlo", "tabu", "admm",
                 "tree_decomposition", "gurobi", "highs"):
        opt = Optimizer.get(name)
        if not opt.available:
            print(f"  {name:20s} backend missing"); continue
        try:
            print(f"  {name:20s} {opt.solve(q)}")
        except Exception as e:
            print(f"  {name:20s} FAILED {type(e).__name__}: {e}")

    uc = UnitCommitmentProblem.example(T=12)
    print(f"\n{uc}")
    for name in ("lagrangian", "admm", "gurobi", "highs"):
        opt = Optimizer.get(name)
        if not opt.available:
            print(f"  {name:20s} backend missing"); continue
        try:
            print(f"  {name:20s} {opt.solve(uc)}")
        except Exception as e:
            print(f"  {name:20s} FAILED {type(e).__name__}: {e}")

    # gate-model quantum (Qiskit) -- separate, smaller instance: qaoa's cost circuit scales with
    # Hamiltonian density (not just qubit count), so keep this one modest
    q2 = QUBOProblem.random(n=8, seed=1)
    print(f"\n{q2} -- exact:", brute_force(q2))
    for name in ("numpy_min_eigen", "vqe", "qaoa"):
        opt = Optimizer.get(name)
        if not opt.available:
            print(f"  {name:20s} backend missing"); continue
        try:
            print(f"  {name:20s} {opt.solve(q2)}")
        except Exception as e:
            print(f"  {name:20s} FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    _demo()
