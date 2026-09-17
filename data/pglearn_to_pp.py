"""
data.pglearn_to_pp

Loader + pandapower converter for the **PGLearn** OPF learning toolkit datasets:
  paper : https://arxiv.org/abs/2505.22825   (arXiv:2505.22825, May 2025)
  code  : https://github.com/AI4OPT/PGLearn.jl
  data  : https://huggingface.co/PGLearn   (collections: PGLearn-Small / -Medium / -Large)

See docs/07-TOOLING-UPDATES.md and docs/08-TOOLING-INTEGRATION.md. PGLearn is the only open OPF
dataset that ships *complete primal AND dual* solutions (KKT multipliers) for three formulations
(ACOPF, DCOPF, SOCOPF) -- useful for dual-warm-starting and for sanity-checking a quantum/QUBO
solution's optimality, not just its feasibility.

REPO LAYOUT (verified 2026-08-27 against the HF API for PGLearn/PGLearn-Small):
  {case}/case.json.gz                     bundled grid spec + one reference solution per formulation
  {case}/config.toml                      generation config (sampler sigma/bounds, solver attrs)
  {case}/{split}/input.h5.gz              sampled inputs (pd, qd, br_status, ...)   split: train|test|infeasible
  {case}/{split}/{ACOPF|DCOPF|SOCOPF}/primal.h5.gz   primal variables  (pg, qg, vm, va, ...)
  {case}/{split}/{ACOPF|DCOPF|SOCOPF}/dual.h5.gz     dual variables    (lam_kirchhoff, mu_sm, ...)
  {case}/{split}/{ACOPF|DCOPF|SOCOPF}/meta.h5.gz     per-instance solve metadata
  cases in PGLearn-Small: 14_ieee, 30_ieee, 57_ieee, 89_pegase, 118_ieee, 300_ieee

case.json TOP-LEVEL KEYS (verified against 14_ieee/case.json):
  data     -> internal per-unit admittance-primitive grid spec (used by pglearn_case_to_pp below)
  config   -> {pglib_case, floating_point_type, OPF{...}, sampler{...}}
  ACOPF/DCOPF/SOCOPF -> {primal{}, dual{}, meta{}}   one reference (base-case) solution each

case.json['data'] fields consumed here (all per-unit on base_mva, 1-indexed bus refs):
  N, E, G, L, base_mva, ref_bus
  vmin[N], vmax[N], vnom[N], gs[N], bs[N], bus_gens[N] (lists), bus_loads[N] (lists)
  bus_fr[E], bus_to[E], g[E], b[E] (series admittance), bff/btt[E] (pi-model diag, for line charging),
  smax[E] (MVA), branch_status[E]
  pgmin/pgmax/qgmin/qgmax[G], c0/c1/c2[G] (cost), gen_status[G]
  pd[L], qd[L]

HDF5 solution files: flat datasets keyed by variable name, first axis = instance index
  (e.g. PGLearn-Small/14_ieee/test has 189,052 instances). read_pglearn_h5() below reads any of them.
"""
import os, sys, json, gzip, math
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux

_pdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pdir not in sys.path:
    sys.path.append(_pdir)

try:
    from huggingface_hub import hf_hub_download
    HAVE_HF_HUB = True
except ImportError:
    HAVE_HF_HUB = False
try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

PGLEARN_SMALL_REPO = "PGLearn/PGLearn-Small"
PGLEARN_CASES_SMALL = ("14_ieee", "30_ieee", "57_ieee", "89_pegase", "118_ieee", "300_ieee")
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pglearn")


### DOWNLOAD ###

def download_pglearn_case(case: str = "14_ieee", repo_id: str = PGLEARN_SMALL_REPO,
                          local_dir: str | None = None) -> str:
    """ download {case}/case.json.gz from HuggingFace, decompress, return the local .json path. """
    if not HAVE_HF_HUB:
        raise ImportError("pip install huggingface_hub to fetch PGLearn data")
    gz = hf_hub_download(repo_id=repo_id, filename=f"{case}/case.json.gz",
                         repo_type="dataset", local_dir=local_dir)
    out = gz[:-3]
    with gzip.open(gz, "rt") as fi, open(out, "w") as fo:
        fo.write(fi.read())
    return out


def download_pglearn_h5(case: str, split: str, kind: str, formulation: str = "ACOPF",
                        repo_id: str = PGLEARN_SMALL_REPO, local_dir: str | None = None) -> str:
    """ download one solution file. kind: 'input'|'primal'|'dual'|'meta'. split: 'train'|'test'|'infeasible'.
    NB train primal/dual are 300 MB - 700 MB per case; prefer split='test', kind='meta' for a quick look. """
    if not HAVE_HF_HUB:
        raise ImportError("pip install huggingface_hub to fetch PGLearn data")
    fn = f"{case}/{split}/input.h5.gz" if kind == "input" else f"{case}/{split}/{formulation}/{kind}.h5.gz"
    gz = hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset", local_dir=local_dir)
    out = gz[:-3]
    if not os.path.exists(out):
        with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
            fo.write(fi.read())
    return out


### case.json ###

def load_pglearn_case(path: str) -> dict:
    """ read a (decompressed) case.json. If given a .gz, transparently decompress. """
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def pglearn_case_to_pp(case: dict, nominal_kv: float = 1.0) -> aux.pandapowerNet:
    """ convert case.json['data'] into a pandapower network.

    Series admittance y = g + jb (per-unit on base_mva) -> z = 1/y -> r_ohm/x_ohm via
    z_base = nominal_kv**2 / base_mva. Line charging from the pi-model diagonal: b_c/2 = bff - b.
    bus/gen/load refs in the file are 1-indexed; pp bus indices here are 0-indexed (ref - 1).
    """
    d = case["data"]
    N, E, G, L = int(d["N"]), int(d["E"]), int(d["G"]), int(d["L"])
    baseMVA = float(d["base_mva"])
    z_base = nominal_kv ** 2 / baseMVA
    y_base = 1.0 / z_base
    ref_bus = int(d["ref_bus"]) - 1

    net = pp.create_empty_network(sn_mva=baseMVA)
    for i in range(N):
        pp.create_bus(net, index=i, vn_kv=nominal_kv, name=f"bus_{i+1}",
                      min_vm_pu=float(d["vmin"][i]), max_vm_pu=float(d["vmax"][i]), type="b")
        gs, bs = float(d["gs"][i]), float(d["bs"][i])
        if gs or bs:
            pp.create_shunt(net, bus=i, q_mvar=-bs * baseMVA, p_mw=gs * baseMVA, name=f"shunt_bus{i+1}")

    # generators: bus_gens[bus] -> [gen ids] (1-indexed both ways)
    gen_bus = {}
    for bi, gids in enumerate(d["bus_gens"]):
        for gid in gids:
            gen_bus[int(gid) - 1] = bi
    for gi in range(G):
        if not d["gen_status"][gi]:
            continue
        b = gen_bus.get(gi, gi)
        pmax, pmin = float(d["pgmax"][gi]) * baseMVA, float(d["pgmin"][gi]) * baseMVA
        if b == ref_bus and len(net.ext_grid) == 0:
            pp.create_ext_grid(net, bus=b, vm_pu=1.0, name=f"gen{gi+1}_slack",
                               max_p_mw=pmax, min_p_mw=pmin)
        else:
            pp.create_gen(net, bus=b, p_mw=0.5 * (pmax + pmin), vm_pu=1.0, name=f"gen_{gi+1}",
                          max_p_mw=pmax, min_p_mw=pmin,
                          max_q_mvar=float(d["qgmax"][gi]) * baseMVA,
                          min_q_mvar=float(d["qgmin"][gi]) * baseMVA, controllable=True)
    if len(net.ext_grid) == 0:
        pp.create_ext_grid(net, bus=ref_bus, vm_pu=1.0, name="fallback_slack")

    # loads: bus_loads[bus] -> [load ids]
    load_bus = {}
    for bi, lids in enumerate(d["bus_loads"]):
        for lid in lids:
            load_bus[int(lid) - 1] = bi
    for li in range(L):
        pp.create_load(net, bus=load_bus.get(li, li), p_mw=float(d["pd"][li]) * baseMVA,
                       q_mvar=float(d["qd"][li]) * baseMVA, name=f"load_{li+1}")

    # branches
    for e in range(E):
        if not d["branch_status"][e]:
            continue
        fr, to = int(d["bus_fr"][e]) - 1, int(d["bus_to"][e]) - 1
        g_s, b_s = float(d["g"][e]), float(d["b"][e])
        y = complex(g_s, b_s)
        z = 1.0 / y if abs(y) else complex(1e-6, 1e-6)
        b_charge = 2.0 * (float(d["bff"][e]) - b_s)                  # b_c (pu), both sides
        c_nf = max(b_charge * y_base / (2 * math.pi * 60) * 1e9, 0.0)
        smax = float(d["smax"][e]) * baseMVA
        max_i_ka = max(smax / (math.sqrt(3) * nominal_kv), 1e-3) if smax > 0 else 1.0
        pp.create_line_from_parameters(net, from_bus=fr, to_bus=to, length_km=1.0,
                                       r_ohm_per_km=z.real * z_base, x_ohm_per_km=z.imag * z_base,
                                       c_nf_per_km=c_nf, max_i_ka=max_i_ka, name=f"branch_{e+1}")
    return net


def pglearn_reference_solution(case: dict, formulation: str = "ACOPF") -> dict:
    """ the single base-case reference solution bundled in case.json (primal + dual + meta). """
    return case[formulation]


### HDF5 sampled dataset ###

def read_pglearn_h5(path: str, rows: slice | None = None) -> dict[str, np.ndarray]:
    """ read a PGLearn input/primal/dual/meta HDF5 into {dataset_name: ndarray}.
    rows: optional slice on the instance axis (axis 0) to avoid loading all ~190k rows. """
    if not HAVE_H5PY:
        raise ImportError("pip install h5py to read PGLearn HDF5 solution files")
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[rows] if rows is not None else obj[()]
        f.visititems(visit)
        out["_attrs"] = dict(f.attrs)
    return out


def pglearn_meta_df(path: str, rows: slice | None = None) -> pd.DataFrame:
    """ meta.h5 -> DataFrame (one row per solved instance: objective, statuses, solve_time, seed). """
    d = read_pglearn_h5(path, rows)
    d.pop("_attrs", None)
    cols = {}
    for k, v in d.items():
        v = np.asarray(v)
        cols[k] = [x.decode() if isinstance(x, bytes) else x for x in v] if v.dtype.kind in "SO" else v
    return pd.DataFrame(cols)


if __name__ == "__main__":
    fx = os.path.join(FIXTURE_DIR, "case14_ieee.json")
    path = fx if os.path.exists(fx) else download_pglearn_case("14_ieee")
    case = load_pglearn_case(path)
    print("pglib_case:", case["config"]["pglib_case"])
    net = pglearn_case_to_pp(case)
    print(net)
    pp.runpp(net)
    print("runpp converged; total load =", float(net.res_load.p_mw.sum()), "MW")
    ref = pglearn_reference_solution(case, "ACOPF")
    print("reference ACOPF primal keys:", list(ref["primal"])[:10])
    print("reference ACOPF dual keys  :", list(ref["dual"])[:10])
    print("reference ACOPF objective  :", ref["meta"].get("primal_objective_value"))
