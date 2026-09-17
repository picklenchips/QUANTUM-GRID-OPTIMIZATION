---
name: agent-pipeline
description: Use when you want to autonomously plan, build, and verify a feature through a multi-phase loop with minimal human intervention. Invokes make-plan → do → verify cycle driven by ralph-loop.
---

# Agent pipeline

## Overview

A self-driving development loop: Claude plans work, executes it with TDD, verifies the result, then loops back for the next task — autonomously, until done.

**Core principle:** ralph-loop drives the outer iteration. The pipeline prompt is re-injected each cycle. Claude reads `skills/agent-pipeline/state/state.md` to determine which phase to run next.

---

## Roles

| Role | Activated when | Uses |
|---|---|---|
| **Planner** | `state.md` absent or phase = `plan` | `claude-mem:make-plan` |
| **Coder** | phase = `code`, next task uncomplete | `superpowers:test-driven-development` |
| **Verifier** | phase = `verify`, task just coded | `cavecrew-reviewer` diff review |
| **Finalizer** | all tasks complete + verified | outputs `PIPELINE_COMPLETE` |

---

## Invocation

```bash
/ralph-loop "$(cat skills/agent-pipeline/CLAUDE.md)

---
TASK: <describe what to build>
SUCCESS: all tasks in skills/agent-pipeline/state/state.md marked verified=true
Output PIPELINE_COMPLETE when done." \
  --completion-promise "PIPELINE_COMPLETE" \
  --max-iterations 30
```

Set `--max-iterations` based on task size. Rule of thumb: 3–5 iterations per task (plan + code + verify + fix if needed).

---

## State file

Runtime state lives in `skills/agent-pipeline/state/state.md`. Format defined in @/Users/benkroul/Documents/agent-context/. Claude reads and updates this file every iteration to track phase and task completion.

Do not edit `skills/agent-pipeline/state/state.md` during a running loop.

---

## When to use

- Well-scoped feature with clear acceptance criteria
- Work that benefits from iteration (tests failing → fix → re-run)
- Greenfield implementation where you can walk away
- Any task where `npm test` or `npm run typecheck` gives automatic signal

**Not suitable for:**
- Design decisions requiring human judgment
- DB migrations (require explicit Ben approval per supabase/CLAUDE.md)
- Changes to `lib/prompts/advisor.ts` (system prompt — iterate carefully)
- Any v0.1+ feature not yet scoped

---

## Safety

Always include `--max-iterations`. The loop has no natural exit except the completion promise. If the task is impossible or blocked, the loop will burn iterations — cap it.

The Verifier role blocks the Coder from advancing when it finds a critical issue. The Coder must fix the issue before the Verifier approves and the next task begins.

---

*See [[SKILLS]] · [[skills/agent-pipeline/CLAUDE]] · [[skills/agent-pipeline/SKILL]]*
