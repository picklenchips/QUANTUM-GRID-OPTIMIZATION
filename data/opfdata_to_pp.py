"""
data.opfdata_to_pp

Loader + pandapower converter for Google DeepMind's **OPFData / GridOpt** dataset:
  paper : https://arxiv.org/abs/2406.07234   (arXiv:2406.07234, Jun 2024)
  code  : https://github.com/AI4OPT/OPFData
  data  : gs://gridopt-dataset/  (public, no auth) -- also reachable over plain HTTPS at
          https://storage.googleapis.com/gridopt-dataset/

See docs/07-TOOLING-UPDATES.md (13,000-instance AC-OPF set, 8 grids, 118-6,515 buses, N-1 outages)
and docs/08-TOOLING-INTEGRATION.md for why this is wired in: every instance ships a solved
*reference AC-OPF solution*, which is the ACOPF baseline docs/06-ML-SOTA.md says to measure against.

BUCKET LAYOUT (verified 2026-08-27 against a real listing + a downloaded case14 group):
  dataset_release_1/{case}_{group}.tar.gz            load-perturbation variant  (300k examples/case)
  dataset_release_1_nminusone/{case}_{group}.tar.gz  topology-perturbation variant (N-1: line/xfmr/gen drop)
    where {case} is a pglib-opf name: pglib_opf_case14_ieee, pglib_opf_case118_ieee,
    pglib_opf_case500_goc, pglib_opf_case2000_goc, pglib_opf_case10000_goc, ...
    and {group} is an integer 0..N. Each tarball unpacks to
      gridopt-dataset-tmp/dataset_release_1/{case}/group_{group}/example_{k}.json

PER-INSTANCE JSON SCHEMA (verified against example_10368.json, pglib_opf_case14_ieee):
  grid.context                       -> [[[baseMVA]]]              (nested; baseMVA = 100.0)
  grid.nodes.bus       [ base_kv, bus_type, vmin, vmax ]           bus_type 1=PQ 2=PV 3=ref 4=isolated
  grid.nodes.generator [ mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_squared, cost_linear, cost_offset ]
  grid.nodes.load      [ pd, qd ]                                  (per-unit on baseMVA)
  grid.nodes.shunt     [ bs, gs ]                                  (per-unit; note bs before gs)
  grid.edges.ac_line     senders[], receivers[] (0-indexed bus),
                         features [ angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c ]
  grid.edges.transformer features [ angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to ]
  grid.edges.generator_link  senders (gen idx)  -> receivers (bus idx)
  grid.edges.load_link       senders (load idx) -> receivers (bus idx)
  grid.edges.shunt_link      senders (shunt idx)-> receivers (bus idx)
  solution.nodes.bus         [ va (rad), vm (pu) ]
  solution.nodes.generator   [ pg, qg ]        (per-unit on baseMVA)
  solution.edges.ac_line.features / .transformer.features  [ pt, qt, pf, qf ]
  metadata.objective         solved AC-OPF objective ($)

All electrical quantities are per-unit on baseMVA. OPFData normalises every bus to base_kv = 1.0, so
the converter picks a nominal vn_kv (default 1.0 kV) and de-normalises impedances with the standard
z_base = vn_kv**2 / baseMVA identity -- self-consistent, and pp.runpp() converges on it.
"""
import os, sys, json, math, io, tarfile, urllib.request
from glob import glob
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux

# import modules from the enclosing directory (same pattern as the other data/*_to_pp.py)
_pdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pdir not in sys.path:
    sys.path.append(_pdir)

OPFDATA_HTTP_ROOT = "https://storage.googleapis.com/gridopt-dataset"
OPFDATA_VARIANTS = {
    "load": "dataset_release_1",            # load perturbations only
    "nminusone": "dataset_release_1_nminusone",  # + topology (N-1) perturbations
}
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "opfdata")


### DOWNLOAD ###

def opfdata_group_url(case: str, group: int = 0, variant: str = "load") -> str:
    """ full HTTPS URL of one OPFData group tarball.
    case: a pglib-opf name, e.g. 'pglib_opf_case14_ieee' (the 'pglib_opf_' prefix is optional here). """
    if not case.startswith("pglib_opf_"):
        case = "pglib_opf_" + case
    sub = OPFDATA_VARIANTS[variant]
    return f"{OPFDATA_HTTP_ROOT}/{sub}/{case}_{group}.tar.gz"


def download_opfdata_group(case: str, group: int = 0, variant: str = "load",
                           dest_dir: str = "data/opfdata", limit: int | None = None) -> list[str]:
    """ download + extract one group tarball, return the list of extracted example_*.json paths.
    limit: stop after extracting this many JSONs (the tarballs hold ~15k files / ~27 MB each). """
    url = opfdata_group_url(case, group, variant)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"GET {url}")
    with urllib.request.urlopen(url, timeout=120) as r:      # nosec - fixed google bucket, https
        raw = r.read()
    out = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            member.name = os.path.basename(member.name)      # flatten the gridopt-dataset-tmp/... prefix
            tf.extract(member, dest_dir)                      # nosec - names flattened to basename above
            out.append(os.path.join(dest_dir, member.name))
            if limit and len(out) >= limit:
                break
    print(f"  extracted {len(out)} example JSONs -> {dest_dir}")
    return out


def get_opfdata_paths(pathname: str = FIXTURE_DIR) -> list[str]:
    """ mirror of get_transnet_paths / get_gridsfm_paths: list example JSONs already on disk. """
    return sorted(glob(os.path.join(pathname, "*.json")))


### PARSE ###

def load_opfdata_example(path: str) -> dict:
    """ read one example_*.json. Returns the raw nested dict (keys: grid, solution, metadata). """
    with open(path) as f:
        return json.load(f)


def _baseMVA(grid: dict) -> float:
    ctx = grid.get("context", [[[100.0]]])
    try:
        return float(ctx[0][0][0])
    except (TypeError, IndexError):
        return 100.0


### CONVERT ###

def opfdata_to_pp(example: dict, nominal_kv: float = 1.0) -> aux.pandapowerNet:
    """ convert one OPFData instance dict (from load_opfdata_example) into a pandapower network.

    - one pp bus per grid.nodes.bus (0-indexed, kept as the pp index)
    - bus_type 3 -> ext_grid (slack); bus_type 2 or a gen present -> pp.create_gen; else load bus
    - ac_line + transformer both become pp lines on a `nominal_kv` base (OPFData normalises base_kv=1.0,
      so there's no real transformer ratio to preserve); de-normalised via z_base = kv**2 / baseMVA
    - shunts via pp.create_shunt (q_mvar sign: pandapower + = absorbing, so q = -bs * baseMVA)
    """
    grid = example["grid"]
    baseMVA = _baseMVA(grid)
    z_base = nominal_kv ** 2 / baseMVA
    y_base = 1.0 / z_base

    net = pp.create_empty_network(sn_mva=baseMVA)
    buses = grid["nodes"]["bus"]
    bus_types = []
    for i, (base_kv, bus_type, vmin, vmax) in enumerate(buses):
        bus_types.append(int(bus_type))
        pp.create_bus(net, index=i, vn_kv=nominal_kv, name=f"bus_{i}",
                      min_vm_pu=float(vmin), max_vm_pu=float(vmax), type="b")

    # generators -- generator_link maps gen index -> bus index
    gl = grid["edges"]["generator_link"]
    gen_bus = {int(s): int(r) for s, r in zip(gl["senders"], gl["receivers"])}
    for gi, g in enumerate(grid["nodes"]["generator"]):
        mbase, pg, pmin, pmax, qg, qmin, qmax, vg, c2, c1, c0 = g
        b = gen_bus.get(gi, gi)
        if bus_types[b] == 3:
            pp.create_ext_grid(net, bus=b, vm_pu=float(vg) or 1.0, name=f"gen_{gi}_slack",
                               max_p_mw=float(pmax) * baseMVA, min_p_mw=float(pmin) * baseMVA)
        else:
            pp.create_gen(net, bus=b, p_mw=float(pg) * baseMVA, vm_pu=float(vg) or 1.0,
                          name=f"gen_{gi}", max_p_mw=float(pmax) * baseMVA, min_p_mw=float(pmin) * baseMVA,
                          max_q_mvar=float(qmax) * baseMVA, min_q_mvar=float(qmin) * baseMVA,
                          controllable=True)
    # a ref bus with no generator still needs a slack
    if len(net.ext_grid) == 0:
        ref = next((i for i, t in enumerate(bus_types) if t == 3), 0)
        pp.create_ext_grid(net, bus=ref, vm_pu=1.0, name="fallback_slack")

    # loads
    ll = grid["edges"]["load_link"]
    load_bus = {int(s): int(r) for s, r in zip(ll["senders"], ll["receivers"])}
    for li, (pd_, qd_) in enumerate(grid["nodes"]["load"]):
        pp.create_load(net, bus=load_bus.get(li, li), p_mw=float(pd_) * baseMVA, q_mvar=float(qd_) * baseMVA,
                       name=f"load_{li}")

    # shunts
    sl = grid["edges"].get("shunt_link", {"senders": [], "receivers": []})
    shunt_bus = {int(s): int(r) for s, r in zip(sl["senders"], sl["receivers"])}
    for si, (bs, gs) in enumerate(grid["nodes"].get("shunt", [])):
        pp.create_shunt(net, bus=shunt_bus.get(si, si), q_mvar=-float(bs) * baseMVA,
                        p_mw=float(gs) * baseMVA, name=f"shunt_{si}")

    # ac lines: features [angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c]
    acl = grid["edges"]["ac_line"]
    for k, (s, r, feat) in enumerate(zip(acl["senders"], acl["receivers"], acl["features"])):
        _, _, b_fr, b_to, br_r, br_x, rate_a, _, _ = feat
        _add_branch_as_line(net, int(s), int(r), br_r, br_x, b_fr + b_to, rate_a,
                            z_base, y_base, nominal_kv, baseMVA, name=f"ac_line_{k}")

    # transformers: features [angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
    tr = grid["edges"].get("transformer", {"senders": [], "receivers": [], "features": []})
    for k, (s, r, feat) in enumerate(zip(tr["senders"], tr["receivers"], tr["features"])):
        _, _, br_r, br_x, rate_a, _, _, tap, shift, b_fr, b_to = feat
        _add_branch_as_line(net, int(s), int(r), br_r, br_x, b_fr + b_to, rate_a,
                            z_base, y_base, nominal_kv, baseMVA, name=f"transformer_{k}")
    return net


def _add_branch_as_line(net, fr, to, br_r, br_x, b_total_pu, rate_a,
                        z_base, y_base, vn_kv, baseMVA, name, length_km=1.0):
    r_ohm = float(br_r) * z_base
    x_ohm = float(br_x) * z_base
    b_siemens = float(b_total_pu) * y_base
    c_nf = max(b_siemens / (2 * math.pi * 60) * 1e9, 0.0)      # 60 Hz US grid
    rate_mva = float(rate_a) * baseMVA
    max_i_ka = max(rate_mva / (math.sqrt(3) * vn_kv), 1e-3) if rate_mva > 0 else 1.0
    pp.create_line_from_parameters(net, from_bus=fr, to_bus=to, length_km=length_km,
                                   r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=c_nf,
                                   max_i_ka=max_i_ka, name=name)


def opfdata_apply_solution(net: aux.pandapowerNet, example: dict, baseMVA: float | None = None) -> None:
    """ set generator/slack setpoints on `net` from the shipped reference AC-OPF solution, so a
    subsequent pp.runpp(net) reproduces the reference voltages (a faithfulness check on the
    converter -- without this, runpp just solves PF at nominal setpoints, a different operating point). """
    if baseMVA is None:
        baseMVA = _baseMVA(example["grid"])
    sol = example["solution"]["nodes"]
    gl = example["grid"]["edges"]["generator_link"]
    gen_bus = {int(s): int(r) for s, r in zip(gl["senders"], gl["receivers"])}
    for gi, (pg, qg) in enumerate(sol["generator"]):
        b = gen_bus.get(gi, gi)
        vm = sol["bus"][b][1]
        m = net.gen["bus"] == b
        if m.any():
            net.gen.loc[m, "p_mw"] = float(pg) * baseMVA
            net.gen.loc[m, "vm_pu"] = float(vm)
        m = net.ext_grid["bus"] == b
        if m.any():
            net.ext_grid.loc[m, "vm_pu"] = float(vm)


def opfdata_solution_df(example: dict) -> dict[str, pd.DataFrame]:
    """ pull the shipped reference AC-OPF solution into tidy DataFrames:
        {'bus': DataFrame[va_rad, vm_pu], 'gen': DataFrame[pg_pu, qg_pu], 'objective': float} """
    sol = example["solution"]["nodes"]
    return {
        "bus": pd.DataFrame(sol["bus"], columns=["va_rad", "vm_pu"]),
        "gen": pd.DataFrame(sol["generator"], columns=["pg_pu", "qg_pu"]),
        "objective": float(example.get("metadata", {}).get("objective", float("nan"))),
    }


if __name__ == "__main__":
    paths = get_opfdata_paths()
    if not paths:
        print("no local fixtures; downloading a small slice of pglib_opf_case14_ieee ...")
        paths = download_opfdata_group("case14_ieee", group=0, limit=3)
    ex = load_opfdata_example(paths[0])
    print("loaded", os.path.basename(paths[0]))
    net = opfdata_to_pp(ex)
    print(net)
    pp.runpp(net)
    print("runpp converged; slack P =", float(net.res_ext_grid.p_mw.iloc[0]), "MW")
    sol = opfdata_solution_df(ex)
    print("reference AC-OPF objective: $%.2f" % sol["objective"])
    print(sol["bus"].head())
