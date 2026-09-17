# SKILLS.md — qpgrid project-scoped skills

Project-scoped index. Full vault-wide plugin/skill reference: [Documents/SKILLS.md](../../SKILLS.md) — read that first for how each skill activates.

---

## Enabled here (`.claude/settings.json`)

| Plugin | Why it's on for qpgrid |
|---|---|
| `caveman` | Docs in this repo (`docs/`, `CLAUDE.md`, this file) are written and maintained in caveman-lite — see [agent-context/MD_README.md §2](../../agent-context/MD_README.md). `/caveman-compress <file>` for a deliberate condensation pass. |
| `frontend-design` | `QPGrid/` is a React Native (Expo) app — auto-triggers on frontend work there. |
| `ralph-wiggum` | Installed, **disabled** by default (matches vault root). Flip on in `.claude/settings.json` when starting a real iteration loop — see below. |

---

## Available on request

- **`ralph-pipeline`** (personal skill, `~/.claude/skills/ralph-pipeline/`) — Planner → Coder → Verifier → Finalizer loop on top of ralph-wiggum. Good fit for grinding through the algorithm backlog in [docs/02-ALGORITHMS.md](docs/02-ALGORITHMS.md) unattended. Not wired up yet — to set up:
  1. Copy `~/.claude/skills/ralph-pipeline/CLAUDE.md` → `docs/pipeline/CLAUDE.md`.
  2. Fill in placeholders (`PROJECT` = qpgrid, `PROJECT_CONTEXT.md` → [docs/versions/V0_SUMMARY.md](docs/versions/V0_SUMMARY.md), guardrails from this repo's `CLAUDE.md` "What NOT to do").
  3. Point `state.md` references at `docs/pipeline/state.md` (project-local, not the global template's).
  4. Flip `ralph-wiggum@claude-code-plugins` to `true` in `.claude/settings.json` if you want the voice trigger (`"ralph loop"` phonetic variants) — it re-reads `.claude/hooks/ralph-loop-trigger.sh`, which isn't installed here yet; copy + adapt from `~/Documents/.claude/hooks/ralph-loop-trigger.sh` if wiring the voice trigger, pointing it at `docs/pipeline/state.md`.
- **`dataviz`** — for any grid/microgrid visualization work (plots, dashboards) beyond raw matplotlib. Load before writing chart code.
- **`vercel:*` family** — if `web/` prototype or `QPGrid/` ever gets a real deploy target.
- **`security-review`** — run before any commit touching auth/tokens, given the existing hardcoded-Mapbox-token issue (see `CLAUDE.md` known issues).

---

*Related: [[SKILLS]] · [[CLAUDE]] · [[agent-context/MD_README]]*
