# AGENTS.md — qpgrid

Generic entry point (any agent/tool, not just Claude Code). Load context in this order before doing anything else:

1. [`/Users/benkroul/Documents/CLAUDE.md`](../../CLAUDE.md) — vault root: workspace map, global tool conventions.
2. [`/Users/benkroul/Documents/agent-context/MD_README.md`](../../agent-context/MD_README.md) — `.md` file conventions (naming, linking, structure). Required reading before creating or editing any `.md` file in this repo.
3. [`../CLAUDE.md`](../CLAUDE.md) — physics workspace map.
4. [`./CLAUDE.md`](CLAUDE.md) — this repo's own agent guide: repo map, current implementation state, known issues, environment setup, what not to do.

Then proceed with the task. Human-facing project vision lives in [README.md](README.md); condensed research/planning docs live in [docs/](docs/).

**Plotly for plotting** (charts: `plotly.graph_objects`; grid networks: `pandapower.plotting.plotly`).

---

*Related: [[CLAUDE]] · [[physics/CLAUDE]] · [[agent-context/MD_README]]*
