"""
api.jobs -- in-memory network + optimization-job registries and the solve dispatch.

Process-local dicts: fine for a single-user local tool, gone on restart. Every Optimizer.solve()
is synchronous CPU-bound code, so the route handler hands it to run_in_threadpool and this module
just builds the problem, runs the solver, and serialises the Result (numpy -> lists).
"""
from __future__ import annotations
import time
import uuid

import numpy as np

from algos import Optimizer, QUBOProblem, UnitCommitmentProblem

# network_id -> pandapowerNet
NETWORKS: dict[str, object] = {}
# job_id -> {status, request, result, error, t_start, t_end}
JOBS: dict[str, dict] = {}


def _demo_network():
    from pp_to_microgrid import create_minimal_example
    return create_minimal_example(nbusses=4)


def register_network(net, network_id: str | None = None) -> str:
    nid = network_id or uuid.uuid4().hex
    NETWORKS[nid] = net
    return nid


def get_network(network_id: str):
    if network_id == "demo" and "demo" not in NETWORKS:
        NETWORKS["demo"] = _demo_network()
    if network_id not in NETWORKS:
        raise KeyError(f"unknown network_id {network_id!r}")
    return NETWORKS[network_id]


# --------------------------------------------------------------------------- problem construction

def build_problem(net, kind: str, params: dict):
    if kind == "qubo":
        return QUBOProblem.from_microgrid(net, lambd=float(params.get("lambd", 1.0)))
    if kind == "uc":
        demand = params.get("demand")
        if not demand:
            raise ValueError("problem_kind 'uc' needs problem_params.demand (list of per-period MW)")
        return UnitCommitmentProblem.from_pp(net, demand=list(map(float, demand)))
    raise ValueError(f"unknown problem_kind {kind!r}")


def _clean_params(params: dict) -> dict:
    """drop nulls/blank strings so optimizer defaults win; keep real values."""
    return {k: v for k, v in (params or {}).items() if v is not None and v != ""}


# --------------------------------------------------------------------------- result serialisation

def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _augment_geojson(net, assignment: dict[int, int]) -> dict:
    from data.openinframap_hook import pp_net_to_geojson
    gj = pp_net_to_geojson(net)
    for ft in gj["features"]:
        p = ft["properties"]
        if p.get("kind") == "bus":
            p["partition"] = assignment.get(int(p["id"]))
        elif p.get("kind") == "line":
            a, b = assignment.get(p.get("from_bus")), assignment.get(p.get("to_bus"))
            p["cross_partition"] = (a is not None and b is not None and a != b)
    return gj


def serialize_result(res, problem, net, kind: str) -> dict:
    out = {
        "method": res.method,
        "objective": _jsonable(res.objective),
        "feasible": bool(res.feasible),
        "runtime_s": float(res.runtime_s),
        "n_iter": int(res.n_iter),
        "bound": _jsonable(res.bound),
        "gap": _jsonable(res.gap),
        "trace": _jsonable(list(res.trace or [])),
        "meta": _jsonable({k: v for k, v in (res.meta or {}).items() if k != "idx_p"}),
    }
    if kind == "qubo":
        x = np.asarray(res.x).ravel().astype(int)
        bus_ids = list(net.bus.index)
        assignment = {int(bus_ids[i]): int(x[i]) for i in range(min(len(x), len(bus_ids)))}
        out["bus_assignment"] = {str(k): v for k, v in assignment.items()}
        out["geojson"] = _augment_geojson(net, assignment)
    elif kind == "uc":
        P = np.asarray(res.x).reshape(problem.G, problem.T)
        out["dispatch"] = P.tolist()
        out["demand"] = problem.demand.tolist()
        out["gens"] = [{"pmin": g.pmin, "pmax": g.pmax, "cost": g.cost} for g in problem.gens]
    return out


# --------------------------------------------------------------------------- run one job (blocking)

def run_job(job_id: str) -> None:
    """blocking; call via run_in_threadpool from the route's background task."""
    job = JOBS[job_id]
    req = job["request"]
    job["status"] = "running"
    job["t_start"] = time.time()
    try:
        net = get_network(req["network_id"])
        problem = build_problem(net, req["problem_kind"], req["problem_params"])
        opt = Optimizer.get(req["optimizer"])
        if not opt.supports(problem):
            raise ValueError(f"optimizer {req['optimizer']!r} does not apply to {req['problem_kind']!r}")
        if not opt.available:
            raise RuntimeError(f"optimizer {req['optimizer']!r}: backend not installed")
        res = opt.solve(problem, **_clean_params(req["solver_params"]))
        job["result"] = serialize_result(res, problem, net, req["problem_kind"])
        job["status"] = "done"
    except Exception as e:
        job["error"] = {"type": type(e).__name__, "message": str(e)}
        job["status"] = "error"
    finally:
        job["t_end"] = time.time()


def create_job(request: dict) -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "request": request, "result": None, "error": None}
    return job_id
