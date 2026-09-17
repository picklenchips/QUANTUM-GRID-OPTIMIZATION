import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell
def __():
    import os
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    ROOT = os.path.abspath(
        os.path.join(
            os.getcwd(),
            ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
        )
    )
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    return ROOT, sys


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import pandapower as pp
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    pio.renderers.default = "notebook_connected"
    PAL = [
        "#DC267F",
        "#648FFF",
        "#FE6100",
        "#785EF0",
        "#FFB000",
        "#009E73",
        "#3DDBD9",
        "#808080",
    ]

    from algos import (
        QUBOProblem,
        UnitCommitmentProblem,
        Optimizer,
        optimize,
        benchmark,
        plot_benchmark,
    )

    return (
        np,
        pd,
        pp,
        go,
        pio,
        make_subplots,
        PAL,
        QUBOProblem,
        UnitCommitmentProblem,
        Optimizer,
        optimize,
        benchmark,
        plot_benchmark,
    )


@app.cell
def __(Optimizer):
    import marimo as mo

    optimizers_list = sorted(Optimizer.registry())
    return mo.md(
        f"""
    # Classical optimizers — `algos/`

    One `Optimizer` class family over three problem shapes (`algos/problems.py`):

    - **`QUBOProblem`** — binary quadratic. `.from_microgrid(net, lambd)` wraps the microgrid-partitioning
      objective in `pp_to_microgrid.py`.
    - **`MILPProblem`** — mixed-integer linear/quadratic.
    - **`UnitCommitmentProblem`** — thermal unit commitment as a MILP + a per-generator DP for decomposition.

    Optimizers: `brute_force`, `steepest_descent`, `simulated_annealing`, `monte_carlo` (parallel
    tempering), `tabu`, `tree_decomposition` (exact), `gurobi` (MIQP/MILP), `highs` (scipy→HiGHS),
    `lagrangian` (dual decomposition), `admm`. See [`docs/02-ALGORITHMS.md`](../docs/02-ALGORITHMS.md).

    Available optimizers: {optimizers_list}
    """
    )


@app.cell
def __(mo):
    return mo.md(
        """
    ## 1 · Microgrid partitioning as a QUBO

    `pp_to_microgrid.microgrid_objective(net, lambd)` → symmetric objective; `to_QUBO` → binary
    quadratic. `QUBOProblem.from_microgrid` packages both. `x_i ∈ {0,1}` is the partition group of bus `i`.
    """
    )


@app.cell
def __(QUBOProblem, Optimizer, benchmark, mo):
    # a real converted network (GridSFM Rhode Island); fall back to the minimal example
    try:
        from data.gridsfm_to_pp import GridSFMOut, gridsfm_to_pp

        g = GridSFMOut("rhode_island", "16h", "data/fixtures/gridsfm/rhode_island_model_16h.json")
        g.read_data()
        net = gridsfm_to_pp(g.bus, g.gen, g.branch, g.load, g.shunt, g.dcline, g.baseMVA)
        net_name = "GridSFM Rhode Island"
    except Exception:
        from pp_to_microgrid import create_minimal_example

        net = create_minimal_example(nbusses=4)
        net_name = "minimal example"

    q = QUBOProblem.from_microgrid(net, lambd=0.5)
    print(net_name, "->", q)
    print("applies:", Optimizer.list(q))

    df_q = benchmark(q)
    return mo.ui.table(
        df_q[["optimizer", "objective", "feasible", "runtime_ms", "opt_gap"]]
    ), net, net_name, q, df_q


@app.cell
def __(plot_benchmark, df_q):
    return plot_benchmark(df_q).show()


@app.cell
def __(df_q, q, optimize, PAL, go, net_name, net):
    # the winning partition, drawn on the network
    best = df_q.iloc[0]
    res = optimize(q, best.optimizer)
    part = res.x.astype(int)
    print(
        f"{best.optimizer}: {part.sum()} buses in group 1, {len(part)-part.sum()} in group 0  (obj {res.objective:.4g})"
    )

    fig = None
    if "geo" in net.bus.columns and net.bus.geo.notna().any():
        import json as _json

        xy = net.bus.geo.map(
            lambda s: _json.loads(s)["coordinates"]
            if isinstance(s, str)
            else (None, None)
        )
        lat = xy.map(lambda c: c[0])
        lon = xy.map(lambda c: c[1])
        fig = go.Figure()
        for _, r in net.line.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[lon[r.from_bus], lon[r.to_bus]],
                    y=[lat[r.from_bus], lat[r.to_bus]],
                    mode="lines",
                    line=dict(color="#bbb", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        for grp, col in [(0, PAL[1]), (1, PAL[0])]:
            m = part == grp
            fig.add_trace(
                go.Scatter(
                    x=lon[m],
                    y=lat[m],
                    mode="markers",
                    name=f"group {grp}",
                    marker=dict(size=11, color=col),
                )
            )
        fig.update_layout(
            title=f"microgrid partition — {best.optimizer} ({net_name})",
            xaxis_title="lon",
            yaxis_title="lat",
            height=380,
        )

    return fig if fig is not None else None


@app.cell
def __(mo):
    return mo.md(
        """
    ## 2 · Unit commitment

    `min Σ (marginal·p + startup·v)` s.t. per-period demand balance (the **coupling** constraint),
    `pmin·u ≤ p ≤ pmax·u`, and min-up/min-down times. `gurobi` / `highs` solve the monolithic MILP;
    `lagrangian` relaxes the demand balance into one single-unit DP per generator; `admm` splits across
    generators.
    """
    )


@app.cell
def __(UnitCommitmentProblem, benchmark, mo):
    uc = UnitCommitmentProblem.example(T=24, seed=1)
    print(
        uc,
        "— peak demand",
        uc.demand.max().round(1),
        "MW,  total capacity",
        sum(g.pmax for g in uc.gens),
        "MW",
    )

    df_uc = benchmark(uc)
    return mo.ui.table(
        df_uc[["optimizer", "objective", "feasible", "runtime_ms", "bound", "opt_gap"]]
    ), uc, df_uc


@app.cell
def __(plot_benchmark, df_uc):
    return plot_benchmark(df_uc).show()


@app.cell
def __(np, df_uc, uc, optimize, PAL, go, make_subplots):
    # dispatch stack for the best schedule + Lagrangian dual-bound convergence
    best_uc = df_uc.iloc[0]
    res_uc = optimize(uc, best_uc.optimizer)
    P = res_uc.x.reshape(uc.G, uc.T)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(
            f"dispatch stack — {best_uc.optimizer} (${res_uc.objective:,.0f})",
            "Lagrangian: primal cost vs dual bound",
        ),
    )
    cum = np.zeros(uc.T)
    for gi in range(uc.G):
        fig.add_trace(
            go.Scatter(
                x=list(range(uc.T)),
                y=cum + P[gi],
                fill="tonexty" if gi else "tozeroy",
                mode="lines",
                name=f"gen {gi} (${uc.gens[gi].cost}/MWh)",
                line=dict(width=0.5, color=PAL[gi % len(PAL)]),
            ),
            row=1,
            col=1,
        )
        cum = cum + P[gi]
    fig.add_trace(
        go.Scatter(
            x=list(range(uc.T)),
            y=uc.demand,
            mode="lines+markers",
            name="demand",
            line=dict(color="black", dash="dash"),
        ),
        row=1,
        col=1,
    )

    lag = optimize(uc, "lagrangian", iters=150)
    fig.add_trace(
        go.Scatter(y=lag.trace, mode="lines", name="best primal", line_color=PAL[0]),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=lag.bound,
        line_dash="dot",
        line_color=PAL[1],
        row=1,
        col=2,
        annotation_text=f"dual bound {lag.bound:,.0f}",
    )
    fig.update_xaxes(title_text="hour", row=1, col=1)
    fig.update_xaxes(title_text="iteration", row=1, col=2)
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))

    print(
        f"lagrangian duality gap: {lag.gap:.2%}   (primal ${lag.objective:,.0f}, dual bound ${lag.bound:,.0f})"
    )

    return fig


@app.cell
def __(mo):
    return mo.md(
        """
    ## 3 · Which optimizer when

    - **exact, small**: `brute_force` (QUBO n≤22), `tree_decomposition` (low treewidth), `gurobi` / `highs` (MILP)
    - **fast heuristic**: `steepest_descent` (surprisingly strong), then `simulated_annealing` / `tabu`
    - **rugged landscape**: `monte_carlo` (parallel tempering)
    - **structured / decomposable**: `lagrangian` (gives a bound + gap), `admm` (consensus split, warm-startable)
    - **quantum-inspired baseline to beat**: `dwave-samplers` path (`simulated_annealing(backend="dwave")`, `tabu`)
    """
    )


@app.cell
def __(pd, df_q, df_uc, mo):
    summary = pd.concat(
        [df_q.assign(problem="microgrid QUBO"), df_uc.assign(problem="unit commitment")],
        ignore_index=True,
    )[["problem", "optimizer", "objective", "feasible", "runtime_ms", "opt_gap"]]

    return mo.ui.table(
        summary.round({"objective": 4, "runtime_ms": 1, "opt_gap": 4})
    )


if __name__ == "__main__":
    app.run()
