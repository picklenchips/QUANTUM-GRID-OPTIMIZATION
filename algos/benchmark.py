"""
algos.benchmark -- run every applicable optimizer on a problem, compare objective / runtime / gap.

    from algos import QUBOProblem, benchmark, plot_benchmark
    df = benchmark(QUBOProblem.random(n=16))
    plot_benchmark(df).show()          # Plotly (see repo CLAUDE.md)
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from .optimizers import Optimizer, _REGISTRY


def benchmark(problem, names: list[str] | None = None, reference: str | None = "auto",
              raise_on_error: bool = False, **solve_kw) -> pd.DataFrame:
    """ solve `problem` with each optimizer in `names` (default: all that apply).
    Adds a `rel_to_best` column (objective / best objective found). If `reference` is a solver name
    (or 'auto' -> brute_force / tree_decomposition / gurobi when available) its objective is the
    baseline for a true optimality `gap`. Returns a tidy DataFrame, sorted by objective. """
    names = names or [n for n, c in _REGISTRY.items() if c().supports(problem) and c().available]
    rows, traces = [], {}
    for name in names:
        opt = Optimizer.get(name)
        if not opt.supports(problem):
            continue
        try:
            res = opt.solve(problem, **solve_kw.get(name, {}))
        except Exception as e:
            if raise_on_error:
                raise
            rows.append({"optimizer": name, "objective": np.nan, "feasible": False,
                         "runtime_ms": np.nan, "n_iter": 0, "bound": np.nan, "error": type(e).__name__})
            continue
        traces[name] = res.trace
        rows.append({"optimizer": name, "objective": res.objective, "feasible": res.feasible,
                     "runtime_ms": res.runtime_s * 1e3, "n_iter": res.n_iter,
                     "bound": res.bound if res.bound is not None else np.nan, "error": ""})
    df = pd.DataFrame(rows)
    if df["objective"].notna().any():
        best = df["objective"].min()
        df["rel_to_best"] = df["objective"] / best if best else np.nan
        ref_obj = _reference_objective(problem, df, reference)
        if ref_obj is not None:
            df["opt_gap"] = (df["objective"] - ref_obj) / max(abs(ref_obj), 1e-9)
    df = df.sort_values("objective", na_position="last").reset_index(drop=True)
    df.attrs["traces"] = traces
    df.attrs["problem"] = repr(problem)
    return df


def _reference_objective(problem, df, reference):
    if reference is None:
        return None
    if reference != "auto":
        row = df.loc[df.optimizer == reference, "objective"]
        return float(row.iloc[0]) if len(row) and np.isfinite(row.iloc[0]) else None
    for cand in ("brute_force", "tree_decomposition", "gurobi"):
        row = df.loc[df.optimizer == cand, "objective"]
        if len(row) and np.isfinite(row.iloc[0]):
            return float(row.iloc[0])
    return None


def plot_benchmark(df: pd.DataFrame):
    """ Plotly: (left) objective vs runtime scatter, (right) best-objective-so-far convergence traces. """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PAL = ["#DC267F", "#648FFF", "#FE6100", "#785EF0", "#FFB000", "#009E73", "#3DDBD9", "#808080"]
    traces = df.attrs.get("traces", {})
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=("objective vs runtime", "convergence (best objective so far)"))
    ok = df[df.objective.notna()]
    fig.add_trace(go.Scatter(x=ok.runtime_ms, y=ok.objective, mode="markers+text",
                             text=ok.optimizer, textposition="top center",
                             marker=dict(size=10, color=PAL[1]), showlegend=False), row=1, col=1)
    for i, (name, tr) in enumerate(traces.items()):
        if tr:
            fig.add_trace(go.Scatter(y=tr, mode="lines", name=name,
                                     line_color=PAL[i % len(PAL)]), row=1, col=2)
    fig.update_xaxes(title_text="runtime (ms)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="iteration checkpoint", row=1, col=2)
    fig.update_yaxes(title_text="objective", row=1, col=1)
    fig.update_layout(height=380, title=df.attrs.get("problem", "benchmark"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


if __name__ == "__main__":
    from .problems import QUBOProblem, UnitCommitmentProblem
    print(benchmark(QUBOProblem.random(n=16, seed=2)).to_string(index=False))
    print()
    print(benchmark(UnitCommitmentProblem.example(T=12)).to_string(index=False))
