"""
algos.problems -- optimization-problem definitions the classical solvers in algos.optimizers act on.

Two base shapes:

  QUBOProblem   min_x  x^T Q x + offset,   x in {0,1}^n
                (Q symmetric; the microgrid-partitioning objective from pp_to_microgrid.py lives here)

  MILPProblem   min  c^T x (+ x^T Qobj x),  s.t. linear (in)equalities,  x mixed binary/integer/continuous
                UnitCommitmentProblem is a structured MILPProblem that also exposes the per-generator
                decomposition Lagrangian relaxation / ADMM need.

Every problem can `.evaluate(x)` (objective, no feasibility check) and `.is_feasible(x)`.
See docs/02-ALGORITHMS.md.
"""
from __future__ import annotations
import os, sys, time, itertools
from dataclasses import dataclass, field
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)


# --------------------------------------------------------------------------- result

@dataclass
class Result:
    """ what every Optimizer.solve() returns. """
    x: np.ndarray                      # best point found (0/1 for QUBO, mixed for MILP)
    objective: float                  # problem.evaluate(x)
    method: str = ""
    feasible: bool = True
    runtime_s: float = 0.0
    n_iter: int = 0
    bound: float | None = None        # dual / relaxation bound when the method produces one
    trace: list[float] = field(default_factory=list)   # best-objective-so-far per iteration
    meta: dict = field(default_factory=dict)

    @property
    def gap(self) -> float | None:
        if self.bound is None or self.objective == 0:
            return None
        return abs(self.objective - self.bound) / max(abs(self.objective), 1e-9)

    def __str__(self) -> str:
        g = f", gap {self.gap:.1%}" if self.gap is not None else ""
        return (f"<{self.method}: obj={self.objective:.6g}{g}, "
                f"feasible={self.feasible}, {self.runtime_s*1e3:.1f} ms, {self.n_iter} it>")


# --------------------------------------------------------------------------- QUBO

class QUBOProblem(Result):
    """ min x^T Q x + offset over x in {0,1}^n.  Q is stored symmetric. """
    kind = "qubo"
    
    def __init__(self, Q, offset: float = 0.0, labels=None, name: str = "qubo"):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square")
        self.Q = 0.5 * (Q + Q.T)                     # symmetrise (energy is unchanged)
        self.offset = float(offset)
        self.n = Q.shape[0]
        self.labels = list(labels) if labels is not None else list(range(self.n))
        self.name = name

    # ---- construction helpers ----
    @classmethod
    def from_qubo_dict(cls, q: dict[tuple[int, int], float], offset: float = 0.0, **kw) -> "QUBOProblem":
        n = 1 + max(max(i, j) for i, j in q)
        Q = np.zeros((n, n))
        for (i, j), w in q.items():
            Q[i, j] += w
        return cls(Q, offset, **kw)

    @classmethod
    def from_microgrid(cls, net, lambd: float = 1.0, name: str | None = None) -> "QUBOProblem":
        """ the microgrid-partitioning QUBO: minimise pp_to_microgrid.microgrid_objective(net, lambd).
        x_i in {0,1} is the partition group of bus i (array position in net.bus.index). """
        from pp_to_microgrid import microgrid_objective, to_QUBO
        f = microgrid_objective(net, lambd)
        q, off = to_QUBO(f)
        p = cls.from_qubo_dict(q, off, name=name or f"microgrid(lambd={lambd})")
        p.meta = {"net_buses": int(len(net.bus)), "lambd": lambd}
        return p

    @classmethod
    def random(cls, n: int = 20, density: float = 0.4, seed: int = 0, name: str = "random_qubo") -> "QUBOProblem":
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((n, n))
        mask = rng.random((n, n)) < density
        Q = np.triu(Q * mask)
        return cls(Q, 0.0, name=name)

    # ---- evaluation ----
    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def is_feasible(self, x) -> bool:                # unconstrained
        x = np.asarray(x)
        return bool(np.all((x == 0) | (x == 1)))

    def field(self, x) -> np.ndarray:
        """ local field h_k s.t. flipping bit k by delta d changes the energy by 2*d*(Qx)_k + Q_kk. """
        return self.Q @ np.asarray(x, dtype=float)

    def random_solution(self, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.integers(0, 2, self.n).astype(float)

    # ---- exports ----
    def to_bqm(self):
        import dimod
        return dimod.BinaryQuadraticModel.from_qubo(self.qubo_dict(), self.offset)

    def qubo_dict(self) -> dict[tuple[int, int], float]:
        q: dict[tuple[int, int], float] = {}
        for i in range(self.n):
            if self.Q[i, i]:
                q[(i, i)] = self.Q[i, i]
            for j in range(i + 1, self.n):
                w = self.Q[i, j] + self.Q[j, i]
                if w:
                    q[(i, j)] = w
        return q

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """ spin form via x = (s+1)/2 :  E(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + c
        (dimod's own BQM.to_ising() is the canonical path; this is for the numpy solvers). """
        import dimod
        h, J, c = dimod.BinaryQuadraticModel.from_qubo(self.qubo_dict(), self.offset).to_ising()
        hv = np.array([h.get(i, 0.0) for i in range(self.n)])
        Jm = np.zeros((self.n, self.n))
        for (i, j), w in J.items():
            # dimod does NOT guarantee i<j key order in the returned dict -- normalise, or terms
            # silently vanish for any caller that only reads the upper triangle (as _ising_hamiltonian
            # in algos/optimizers.py does; verified this was dropping ~2/3 of interaction terms).
            a, b = (i, j) if i < j else (j, i)
            Jm[a, b] += w
        return hv, Jm, float(c)

    def __repr__(self):
        return f"QUBOProblem({self.name!r}, n={self.n})"


# --------------------------------------------------------------------------- MILP

@dataclass
class LinCon:
    """ one linear constraint  A @ x  <sense>  b  ,  sense in {'<=','>=','=='} """
    A: np.ndarray
    sense: str
    b: float


class MILPProblem:
    """ min c^T x (+ x^T Qobj x if given), s.t. the LinCons, x_i in [lb_i, ub_i], kinds in {'C','I','B'}. """
    kind = "milp"

    def __init__(self, c, cons: list[LinCon] | None = None, lb=None, ub=None, kinds=None,
                 Qobj=None, const: float = 0.0, name: str = "milp"):
        self.c = np.asarray(c, dtype=float)
        self.n = len(self.c)
        self.cons = list(cons or [])
        self.lb = np.zeros(self.n) if lb is None else np.asarray(lb, dtype=float)
        self.ub = np.full(self.n, np.inf) if ub is None else np.asarray(ub, dtype=float)
        self.kinds = list(kinds) if kinds is not None else ["C"] * self.n
        self.Qobj = None if Qobj is None else 0.5 * (np.asarray(Qobj, float) + np.asarray(Qobj, float).T)
        self.const = float(const)
        self.name = name

    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        v = float(self.c @ x + self.const)
        if self.Qobj is not None:
            v += float(x @ self.Qobj @ x)
        return v

    def is_feasible(self, x, tol: float = 1e-6) -> bool:
        x = np.asarray(x, dtype=float)
        if np.any(x < self.lb - tol) or np.any(x > self.ub + tol):
            return False
        for k, kind in enumerate(self.kinds):
            if kind in ("I", "B") and abs(x[k] - round(x[k])) > tol:
                return False
        for con in self.cons:
            lhs = float(con.A @ x)
            if con.sense == "<=" and lhs > con.b + tol: return False
            if con.sense == ">=" and lhs < con.b - tol: return False
            if con.sense == "==" and abs(lhs - con.b) > tol: return False
        return True

    def __repr__(self):
        return f"MILPProblem({self.name!r}, n={self.n}, {len(self.cons)} cons)"


# --------------------------------------------------------------------------- unit commitment

@dataclass
class Generator:
    pmin: float
    pmax: float
    cost: float             # $/MWh marginal (linear)
    startup: float = 0.0    # $ per start
    min_up: int = 1
    min_down: int = 1
    u0: int = 0             # committed at t = -1 ?


class UnitCommitmentProblem:
    """ classic thermal unit commitment.

        min  sum_{g,t} [ cost_g * p_{g,t} + startup_g * v_{g,t} ]
        s.t. sum_g p_{g,t} = demand_t                         (COUPLING -- relaxed by Lagrangian / split by ADMM)
             pmin_g u_{g,t} <= p_{g,t} <= pmax_g u_{g,t}
             min up / min down times on u_{g,t}
             v_{g,t} >= u_{g,t} - u_{g,t-1}                    (startup indicator)

    Exposes `.build_milp()` for the monolithic solvers and `.gen_subproblem(g, lam)` (a T-period
    single-unit DP) for Lagrangian relaxation / ADMM.
    """
    kind = "uc"

    def __init__(self, gens: list[Generator], demand: list[float], name: str = "uc"):
        self.gens = list(gens)
        self.demand = np.asarray(demand, dtype=float)
        self.G, self.T = len(self.gens), len(self.demand)
        self.name = name

    @classmethod
    def example(cls, T: int = 24, seed: int = 0) -> "UnitCommitmentProblem":
        rng = np.random.default_rng(seed)
        gens = [
            Generator(pmin=100, pmax=450, cost=18.0, startup=900, min_up=4, min_down=4, u0=1),
            Generator(pmin=50,  pmax=300, cost=22.0, startup=500, min_up=3, min_down=3, u0=1),
            Generator(pmin=30,  pmax=180, cost=28.0, startup=250, min_up=2, min_down=2, u0=0),
            Generator(pmin=20,  pmax=120, cost=35.0, startup=120, min_up=1, min_down=1, u0=0),
            Generator(pmin=10,  pmax=80,  cost=45.0, startup=60,  min_up=1, min_down=1, u0=0),
        ]
        base = 300 + 260 * np.sin(np.linspace(0, np.pi, T)) ** 2      # a daily hump
        demand = base + rng.normal(0, 15, T)
        return cls(gens, list(np.clip(demand, 150, None)), name=f"uc_example(T={T})")

    @classmethod
    def from_pp(cls, net, demand, name: str | None = None) -> "UnitCommitmentProblem":
        """ pull generators from a pandapower net (uses max_p_mw / min_p_mw; cost from poly_cost if present). """
        import pandas as pd
        gens = []
        cost_lookup = {}
        if len(getattr(net, "poly_cost", [])):
            for _, r in net.poly_cost.iterrows():
                if r["et"] == "gen":
                    cost_lookup[int(r["element"])] = float(r.get("cp1_eur_per_mw", 20.0))
        for gi, g in net.gen.iterrows():
            pmax = float(g.get("max_p_mw", g["p_mw"]) or g["p_mw"] or 100.0)
            pmin = float(g.get("min_p_mw", 0.0) or 0.0)
            gens.append(Generator(pmin=max(pmin, 0.0), pmax=max(pmax, pmin + 1.0),
                                  cost=cost_lookup.get(int(gi), 20.0 + 5.0 * (gi % 4)),
                                  startup=100.0 * (1 + gi % 3), min_up=2, min_down=2))
        return cls(gens, list(np.asarray(demand, float)), name=name or "uc_from_pp")

    # ---- objective on a full schedule ----
    def evaluate(self, p: np.ndarray) -> float:
        """ p: (G, T) dispatch. u is inferred (u=1 where p>0); startup counted vs u0. """
        p = np.asarray(p, dtype=float).reshape(self.G, self.T)
        u = (p > 1e-6).astype(int)
        cost = 0.0
        for g, gen in enumerate(self.gens):
            cost += gen.cost * p[g].sum()
            prev = gen.u0
            for t in range(self.T):
                cost += gen.startup * max(u[g, t] - prev, 0)
                prev = u[g, t]
        return float(cost)

    def is_feasible(self, p: np.ndarray, tol: float = 1e-3) -> bool:
        p = np.asarray(p, dtype=float).reshape(self.G, self.T)
        if np.any(np.abs(p.sum(axis=0) - self.demand) > tol * np.maximum(self.demand, 1)):
            return False
        for g, gen in enumerate(self.gens):
            on = p[g] > 1e-6
            if np.any(p[g, on] < gen.pmin - tol) or np.any(p[g] > gen.pmax + tol) or np.any(p[g] < -tol):
                return False
        return True

    # ---- per-generator subproblem: T-period single-unit DP given price signal `lam` (len T) ----
    def gen_subproblem(self, g: int, lam: np.ndarray) -> tuple[np.ndarray, float]:
        """ min over this unit's (u, p):  sum_t [ (cost_g - lam_t) * p_{g,t} + startup_g * start_t ]
        with pmin/pmax and min-up/min-down. Returns (p_g of length T, subproblem optimal value).
        Solved exactly by DP over (state = periods spent in current on/off run, capped). """
        gen = self.gens[g]
        lam = np.asarray(lam, dtype=float)
        # when ON at t, optimal p is pmax if (cost - lam_t) < 0 else pmin (linear in p)
        pon = np.where(gen.cost - lam < 0, gen.pmax, gen.pmin)
        run_cost_on = (gen.cost - lam) * pon                  # per-period cost if ON
        cap = max(gen.min_up, gen.min_down) + 1
        INF = 1e18
        # dp[(on, r)] = best cost to reach period t with current status `on` and `r` periods spent
        # in it (capped). Assume the unit has been in state u0 long enough to switch freely (r = cap).
        dp = {(gen.u0, cap): 0.0}
        hist: list[dict] = []
        for t in range(self.T):
            ndp: dict[tuple[int, int], float] = {}
            back: dict[tuple[int, int], tuple] = {}
            for (on, r), cost in dp.items():
                for nxt in (0, 1):
                    if nxt == on:
                        can, nr = True, min(r + 1, cap)
                    else:
                        can = (on == 1 and r >= gen.min_up) or (on == 0 and r >= gen.min_down)
                        nr = 1
                    if not can:
                        continue
                    add = run_cost_on[t] if nxt == 1 else 0.0
                    add += gen.startup if (nxt == 1 and on == 0) else 0.0
                    key = (nxt, nr)
                    if cost + add < ndp.get(key, INF):
                        ndp[key] = cost + add
                        back[key] = (on, r)
            dp = ndp
            hist.append(back)
        # recover
        end = min(dp, key=dp.get)
        val = dp[end]
        states = [end]
        for t in range(self.T - 1, 0, -1):
            states.append(hist[t][states[-1]])
        states = states[::-1]
        p_g = np.array([pon[t] if states[t][0] == 1 else 0.0 for t in range(self.T)])
        return p_g, float(val)

    def build_milp(self) -> MILPProblem:
        """ monolithic MILP. Variables: p_{g,t} (C), u_{g,t} (B), v_{g,t} (B, startup). """
        G, T = self.G, self.T
        npv = G * T
        idx_p = lambda g, t: g * T + t
        idx_u = lambda g, t: npv + g * T + t
        idx_v = lambda g, t: 2 * npv + g * T + t
        n = 3 * npv
        c = np.zeros(n)
        lb, ub, kinds = np.zeros(n), np.zeros(n), ["C"] * n
        for g, gen in enumerate(self.gens):
            for t in range(T):
                c[idx_p(g, t)] = gen.cost
                c[idx_v(g, t)] = gen.startup
                ub[idx_p(g, t)] = gen.pmax
                ub[idx_u(g, t)] = 1; kinds[idx_u(g, t)] = "B"
                ub[idx_v(g, t)] = 1; kinds[idx_v(g, t)] = "B"
        cons: list[LinCon] = []
        for t in range(T):                                           # demand balance (coupling)
            A = np.zeros(n)
            for g in range(G):
                A[idx_p(g, t)] = 1.0
            cons.append(LinCon(A, "==", float(self.demand[t])))
        for g, gen in enumerate(self.gens):
            for t in range(T):
                a1 = np.zeros(n); a1[idx_p(g, t)] = 1.0; a1[idx_u(g, t)] = -gen.pmax
                cons.append(LinCon(a1, "<=", 0.0))                   # p <= pmax u
                a2 = np.zeros(n); a2[idx_p(g, t)] = -1.0; a2[idx_u(g, t)] = gen.pmin
                cons.append(LinCon(a2, "<=", 0.0))                   # p >= pmin u
                a3 = np.zeros(n); a3[idx_v(g, t)] = -1.0; a3[idx_u(g, t)] = 1.0
                a3[idx_u(g, t - 1)] = -1.0 if t > 0 else 0.0
                cons.append(LinCon(a3, "<=", (gen.u0 if t == 0 else 0.0)))   # v >= u_t - u_{t-1}
        m = MILPProblem(c, cons, lb, ub, kinds, name=self.name + "_milp")
        m.meta = {"G": G, "T": T, "idx_p": idx_p}
        return m

    def dispatch_from_commitment(self, u: np.ndarray) -> np.ndarray:
        """ per-period economic dispatch (merit order) for a commitment u (G,T) -> p (G,T).
        Repairs the commitment column when needed: turns on cheap units if capacity is short,
        turns off the most expensive units if the min-generation floor exceeds demand. """
        u = (np.asarray(u).reshape(self.G, self.T) > 0.5).astype(int)
        p = np.zeros((self.G, self.T))
        cheap = sorted(range(self.G), key=lambda g: self.gens[g].cost)
        for t in range(self.T):
            D = self.demand[t]
            on = [g for g in cheap if u[g, t]]
            # capacity short -> commit more cheap units
            for g in cheap:
                if sum(self.gens[h].pmax for h in on) >= D:
                    break
                if g not in on:
                    on.append(g)
            on.sort(key=lambda g: self.gens[g].cost)
            # min-gen floor above demand -> drop the most expensive units (keep >=1, keep capacity)
            while len(on) > 1 and sum(self.gens[g].pmin for g in on) > D \
                    and sum(self.gens[g].pmax for g in on[:-1]) >= D:
                on.pop()
            alloc = {g: self.gens[g].pmin for g in on}
            need = D - sum(alloc.values())                 # >= 0 once the loop above has run (if feasible)
            for g in on:                                   # merit-order fill upward
                take = np.clip(need, 0.0, self.gens[g].pmax - alloc[g])
                alloc[g] += take
                need -= take
            for g, v in alloc.items():
                p[g, t] = v
        return p

    def __repr__(self):
        return f"UnitCommitmentProblem({self.name!r}, G={self.G}, T={self.T})"


# --------------------------------------------------------------------------- brute force (shared)

def brute_force(problem: QUBOProblem, max_n: int = 22) -> Result:
    """ exact minimiser by enumeration -- QUBO only, small n. """
    if problem.n > max_n:
        raise ValueError(f"brute force refuses n={problem.n} > {max_n}")
    t0 = time.perf_counter()
    best_x, best_e = None, np.inf
    for bits in itertools.product((0, 1), repeat=problem.n):
        x = np.array(bits, dtype=float)
        e = problem.evaluate(x)
        if e < best_e:
            best_e, best_x = e, x
    return Result(best_x, best_e, "brute_force", True, time.perf_counter() - t0, 2 ** problem.n,
                  bound=best_e)
