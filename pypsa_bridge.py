"""
pypsa_bridge

pandapower <-> PyPSA v1.0 bridge, plus a stochastic-optimization entry point.

Why (see docs/07-TOOLING-UPDATES.md "PyPSA -- the biggest tooling change in this survey" +
docs/08-TOOLING-INTEGRATION.md): PyPSA v1.0 (stable, ~Oct 2025) added *native two-stage stochastic
programming* -- scenario trees, risk-neutral and CVaR risk-averse formulations. pandapower has no
equivalent. This project's microgrid-formation QUBO (pp_to_microgrid.py) assumes a single
deterministic demand snapshot; the moment uncertain demand / renewable variability enters the
picture, PyPSA is the tool.

Bridge scope: buses, lines (pp trafos are mapped to PyPSA lines -- PyPSA's Transformer needs a type
library or explicit tap model, out of scope here), loads, generators (pp ext_grid -> a cheap slack
generator), storage. Switches and detailed transformer tap models are not carried over.

PyPSA v1.0 stochastic API used below (verified against docs.pypsa.org/v1.0.0 user-guide):
    n.set_scenarios({"low": 0.4, "high": 0.6})     # names -> probabilities (uniform if omitted)
    n.loads_t.p_set.loc[:, ("low", load_name)] = ...   # component _t frames gain an outer 'scenario' level
    n.set_risk_preference(alpha=0.9, omega=0.5)     # CVaR: tail = worst (1-alpha); omega 0=neutral..1=max averse
    n.optimize()
    n.statistics.energy_balance().unstack(level="scenario")
"""
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux

try:
    import pypsa
    HAVE_PYPSA = True
    _PYPSA_VERSION = getattr(pypsa, "__version__", "0")
except ImportError:
    HAVE_PYPSA = False
    _PYPSA_VERSION = None


def _require_pypsa():
    if not HAVE_PYPSA:
        raise ImportError("pip install 'pypsa>=1.0' (see environment.yml) to use pypsa_bridge")
    if int(_PYPSA_VERSION.split(".")[0]) < 1:
        raise RuntimeError(f"pypsa_bridge needs PyPSA >= 1.0 for the stochastic API; found {_PYPSA_VERSION}")


def pp_net_to_pypsa(net: aux.pandapowerNet, default_gen_cost: float = 50.0,
                    slack_cost: float = 1000.0) -> "pypsa.Network":
    """ convert a pandapower network into a single-snapshot PyPSA Network.

    - r/x for lines are total ohms = per-km * length_km (PyPSA wants absolute ohms)
    - pp ext_grid -> an expensive ('slack_cost') extendable generator, so the LP always has a balancing
      resource but prefers the real generators
    - real pp gens -> extendable generators at 'default_gen_cost' unless net.poly_cost says otherwise
    """
    _require_pypsa()
    n = pypsa.Network()
    n.set_snapshots([0])

    for b in net.bus.index:
        n.add("Bus", f"b{b}", v_nom=float(net.bus.at[b, "vn_kv"]))

    for li in net.line.index:
        ln = net.line.loc[li]
        length = float(ln.get("length_km", 1.0)) or 1.0
        n.add("Line", f"l{li}", bus0=f"b{int(ln['from_bus'])}", bus1=f"b{int(ln['to_bus'])}",
              r=float(ln["r_ohm_per_km"]) * length, x=float(ln["x_ohm_per_km"]) * length,
              s_nom=float(ln.get("max_i_ka", 1.0)) * np.sqrt(3) * float(net.bus.at[int(ln['from_bus']), "vn_kv"]))

    for ti in net.trafo.index:                     # trafo -> line (documented simplification)
        tr = net.trafo.loc[ti]
        z_base = float(tr["vn_lv_kv"]) ** 2 / float(tr["sn_mva"])
        r = float(tr["vkr_percent"]) / 100.0 * z_base
        x = np.sqrt(max((float(tr["vk_percent"]) / 100.0) ** 2 - (float(tr["vkr_percent"]) / 100.0) ** 2, 0.0)) * z_base
        n.add("Line", f"t{ti}", bus0=f"b{int(tr['hv_bus'])}", bus1=f"b{int(tr['lv_bus'])}",
              r=max(r, 1e-4), x=max(x, 1e-3), s_nom=float(tr["sn_mva"]))

    for ld in net.load.index:
        lo = net.load.loc[ld]
        n.add("Load", f"load{ld}", bus=f"b{int(lo['bus'])}", p_set=_num(lo["p_mw"], 0.0))

    for g in net.gen.index:
        ge = net.gen.loc[g]
        pmax = _num(ge.get("max_p_mw"), _num(ge["p_mw"], 1.0)) or 1.0
        n.add("Generator", f"gen{g}", bus=f"b{int(ge['bus'])}", p_nom=max(pmax, 1.0),
              p_nom_extendable=True, marginal_cost=default_gen_cost, capital_cost=20.0)
    for s in net.sgen.index:
        sg = net.sgen.loc[s]
        n.add("Generator", f"sgen{s}", bus=f"b{int(sg['bus'])}", p_nom=max(_num(sg["p_mw"], 1.0), 1.0),
              p_nom_extendable=True, marginal_cost=default_gen_cost * 0.5, capital_cost=15.0)
    for eg in net.ext_grid.index:
        # import from the wider grid: an always-available but expensive balancing resource
        n.add("Generator", f"slack{eg}", bus=f"b{int(net.ext_grid.at[eg, 'bus'])}", p_nom=1e4,
              p_nom_extendable=False, marginal_cost=slack_cost)

    for st in net.storage.index:
        so = net.storage.loc[st]
        pn = max(abs(_num(so["p_mw"], 1.0)), 1.0)
        n.add("StorageUnit", f"stor{st}", bus=f"b{int(so['bus'])}", p_nom=pn, p_nom_extendable=True,
              max_hours=max(_num(so.get("max_e_mwh"), pn) / pn, 1.0),
              marginal_cost=1.0, capital_cost=8.0)
    return n


def _num(v, default: float) -> float:
    try:
        v = float(v)
        return default if (v != v) else v      # NaN check
    except (TypeError, ValueError):
        return default


def stochastic_microgrid_scenarios(net_or_n, load_scale: dict[str, float] | None = None,
                                   probabilities: dict[str, float] | None = None,
                                   risk_preference: tuple[float, float] | None = None,
                                   solver_name: str = "highs") -> dict:
    """ run a two-stage stochastic capacity problem: first-stage generator/storage sizing that is
    robust across demand scenarios (second stage = per-scenario operation).

    net_or_n     : a pandapower net (converted here) or an already-built PyPSA Network
    load_scale   : {scenario_name: multiplier on every load's p_set}. default {'low':0.8,'exp':1.0,'high':1.3}
    probabilities: {scenario_name: p}. default uniform.
    risk_preference: (alpha, omega) for CVaR; None -> risk-neutral.

    returns {'objective', 'generator_capacity' (Series), 'energy_balance' (DataFrame by scenario)}
    """
    _require_pypsa()
    n = net_or_n if hasattr(net_or_n, "set_scenarios") or isinstance(net_or_n, pypsa.Network) \
        else pp_net_to_pypsa(net_or_n)

    load_scale = load_scale or {"low": 0.8, "expected": 1.0, "high": 1.3}
    # base load per load-name, captured BEFORE the scenario index is added
    base_p = {name: float(p) for name, p in n.loads["p_set"].items()}

    n.set_scenarios(probabilities or {k: 1.0 / len(load_scale) for k in load_scale})

    # after set_scenarios the component frames gain an outer 'scenario' index level
    for idx in n.loads.index:
        scen, name = (idx[0], idx[-1]) if isinstance(idx, tuple) else (None, idx)
        if scen in load_scale:
            n.loads.loc[idx, "p_set"] = base_p.get(name, next(iter(base_p.values()))) * load_scale[scen]

    if risk_preference is not None:
        n.set_risk_preference(alpha=risk_preference[0], omega=risk_preference[1])

    res = n.optimize(solver_name=solver_name, include_objective_constant=True)
    status, cond = res if isinstance(res, tuple) else (res, "")
    out = {"status": str(status), "condition": str(cond)}
    try:
        out["objective"] = float(n.objective)
    except Exception:
        out["objective"] = float("nan")
    try:
        out["generator_capacity"] = n.generators["p_nom_opt"].groupby(level=-1).first()
    except Exception:
        out["generator_capacity"] = n.generators.get("p_nom_opt")
    try:
        out["energy_balance"] = n.statistics.energy_balance().unstack(level="scenario")
    except Exception as e:
        out["energy_balance"] = f"unavailable: {e}"
    return out


if __name__ == "__main__":
    try:
        from pp_to_microgrid import create_minimal_example
        net = create_minimal_example(nbusses=3)
    except Exception:
        net = pp.create_empty_network()
        b0 = pp.create_bus(net, vn_kv=20.0); b1 = pp.create_bus(net, vn_kv=20.0)
        pp.create_ext_grid(net, b0); pp.create_line(net, b0, b1, 1.0, "NAYY 4x150 SE")
        pp.create_load(net, b1, p_mw=5.0); pp.create_gen(net, b0, p_mw=10.0, max_p_mw=20.0)

    n = pp_net_to_pypsa(net)
    print("PyPSA", _PYPSA_VERSION, "->", n)
    res = stochastic_microgrid_scenarios(net, risk_preference=(0.9, 0.5))
    print("status:", res["status"], res["condition"])
    print("stochastic objective:", res["objective"])
    print("first-stage generator capacity:\n", res["generator_capacity"])
