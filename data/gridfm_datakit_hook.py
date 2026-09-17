"""
data.gridfm_datakit_hook

Thin wrapper around **gridfm-datakit** -- a PF/OPF *data-generator* library (it perturbs a base case
and solves the results), not a fixed dataset dump.
  paper : https://arxiv.org/abs/2512.14658   (arXiv:2512.14658, Dec 2025)
  code  : https://github.com/gridfm/gridfm-datakit   (Apache-2.0, on PyPI: `pip install gridfm-datakit`)
  docs  : https://gridfm.github.io/gridfm-datakit/

See docs/07-TOOLING-UPDATES.md ("actively developed ... designed to fix gaps in OPFData/PGLearn",
stochastic PF to 30,000 buses, OPF to 10,000) and docs/08-TOOLING-INTEGRATION.md.

The library is config-driven (YAML). Output goes to `{data_dir}/{network_name}/raw/*.parquet`:
  bus_data.parquet, gen_data.parquet, branch_data.parquet, y_bus_data.parquet, runtime_data.parquet

This hook:
  * `have_gridfm_datakit()`         -- is the optional dep importable
  * `default_config(case, ...)`     -- a minimal config dict (few scenarios, no perturbation) for a smoke run
  * `generate_pf_scenarios(...)`    -- write a YAML + invoke the generator (Python API if present, else the CLI)
  * `load_scenarios_df(data_dir, network_name)` -- read the parquet outputs back into DataFrames
"""
import os, sys, json, shutil, subprocess, importlib
import pandas as pd

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def have_gridfm_datakit() -> bool:
    """ True only if gridfm_datakit actually imports (it has a fragile julia/powsybl dependency
    chain -- e.g. gridfm-datakit 1.0.5 needs juliapkg<0.1.26, see environment.yml). """
    if importlib.util.find_spec("gridfm_datakit") is None:
        return False
    try:
        importlib.import_module("gridfm_datakit")
        return True
    except Exception:
        return False


def default_config(case: str = "case14_ieee", scenarios: int = 8, data_dir: str = "data/gridfm",
                   mode: str = "pf", seed: int = 0) -> dict:
    """ smallest sensible config: a handful of load scenarios, no topology/gen/admittance perturbation.
    `case` is a PGLib name understood by gridfm-datakit's `network.source: pglib`.
    Schema mirrors scripts/config/default.yaml from the gridfm-datakit repo (every `settings` key the
    NestedNamespace loader expects is present -- a partial dict raises AttributeError deep in _setup). """
    return {
        "network": {"name": case, "source": "pglib", "network_dir": "scripts/grids"},
        "load": {"generator": "agg_load_profile", "agg_profile": "default", "scenarios": int(scenarios),
                 "sigma": 0.05, "change_reactive_power": True, "global_range": 0.4,
                 "max_scaling_factor": 4.0, "step_size": 0.1, "start_scaling_factor": 1.0},
        "topology_perturbation": {"type": "none", "k": 1, "n_topology_variants": 1, "elements": ["branch"]},
        "generation_perturbation": {"type": "none", "sigma": 1.0},
        "admittance_perturbation": {"type": "none", "sigma": 0.2},
        "settings": {"num_processes": 1, "data_dir": data_dir, "large_chunk_size": 1000,
                     "overwrite": True, "mode": mode, "include_dc_res": False,
                     "enable_solver_logs": False, "pf_fast": True, "dcpf_fast": True,
                     "opf_formulation": "polar", "max_iter": 200, "seed": int(seed)},
    }


def _write_yaml(config: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if HAVE_YAML:
            yaml.safe_dump(config, f, sort_keys=False)
        else:
            json.dump(config, f, indent=2)   # gridfm-datakit's loader accepts JSON-compatible YAML
    return path


def generate_pf_scenarios(config: dict | None = None, case: str = "case14_ieee",
                          workdir: str = "data/gridfm", **cfg_kw) -> dict:
    """ run one generation job. Returns {'data_dir', 'network_name', 'outputs': [parquet paths], 'log'}.

    Tries the Python API first (entry point name varies across releases -- several are attempted),
    then falls back to the documented CLI `gridfm_datakit generate <config.yaml>`.
    """
    if not have_gridfm_datakit():
        raise ImportError("pip install gridfm-datakit (see environment.yml pip: section)")
    config = config or default_config(case=case, data_dir=workdir, **cfg_kw)
    yaml_path = _write_yaml(config, os.path.join(workdir, f"{config['network']['name']}_config.yaml"))

    log = ""
    ran = False
    api_paths = {}
    for modname, fnname in [("gridfm_datakit.generate", "generate_power_flow_data"),
                            ("gridfm_datakit", "generate_power_flow_data"),
                            ("gridfm_datakit", "run"),
                            ("gridfm_datakit.pipeline", "run")]:
        try:
            fn = getattr(importlib.import_module(modname), fnname)
        except (ImportError, AttributeError):
            continue
        try:
            out = fn(config)
        except TypeError:
            out = fn(yaml_path)
        if isinstance(out, dict):
            api_paths = out
        ran = True
        log = f"python API: {modname}.{fnname}"
        break

    if not ran:
        exe = shutil.which("gridfm_datakit") or shutil.which("gridfm-datakit")
        cmd = [exe, "generate", yaml_path] if exe else [sys.executable, "-m", "gridfm_datakit", "generate", yaml_path]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        log = f"CLI: {' '.join(cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        if p.returncode != 0:
            raise RuntimeError(f"gridfm-datakit generation failed:\n{log}")

    # prefer the artifact paths the API returned; else scan the conventional raw/ dir
    raw = os.path.join(config["settings"]["data_dir"], config["network"]["name"], "raw")
    outputs = [p for p in api_paths.values() if isinstance(p, str) and p.endswith(".parquet") and os.path.exists(p)]
    if not outputs:
        outputs = [os.path.join(raw, f) for f in ("bus_data.parquet", "gen_data.parquet",
                   "branch_data.parquet", "y_bus_data.parquet", "runtime_data.parquet")
                   if os.path.exists(os.path.join(raw, f))]
    return {"data_dir": config["settings"]["data_dir"], "network_name": config["network"]["name"],
            "outputs": outputs, "artifacts": api_paths, "log": log}


def load_scenarios_df(data_dir: str = "data/gridfm", network_name: str = "case14_ieee") -> dict[str, pd.DataFrame]:
    """ read {data_dir}/{network_name}/raw/*.parquet -> {'bus': df, 'gen': df, 'branch': df, ...}. """
    raw = os.path.join(data_dir, network_name, "raw")
    out = {}
    for key, fn in [("bus", "bus_data.parquet"), ("gen", "gen_data.parquet"),
                    ("branch", "branch_data.parquet"), ("y_bus", "y_bus_data.parquet"),
                    ("runtime", "runtime_data.parquet")]:
        p = os.path.join(raw, fn)
        if os.path.exists(p):
            out[key] = pd.read_parquet(p)
    return out


if __name__ == "__main__":
    print("gridfm-datakit importable:", have_gridfm_datakit())
    cfg = default_config("case14_ieee", scenarios=4)
    print(json.dumps(cfg, indent=2))
    if have_gridfm_datakit():
        res = generate_pf_scenarios(cfg)
        print("outputs:", res["outputs"])
        dfs = load_scenarios_df(res["data_dir"], res["network_name"])
        for k, v in dfs.items():
            print(k, v.shape)
    else:
        print("(dep not installed -- config generation still works for the CLI path)")
