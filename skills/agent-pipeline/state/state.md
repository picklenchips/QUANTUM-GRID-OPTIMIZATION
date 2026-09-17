# Pipeline state qpgrid

Runtime state file at `skills/agent-pipeline/state/state.md`. Created by Planner, updated every iteration.

```markdown
# Pipeline state
phase: TBD                  # plan | code | verify | complete | blocked
current_task: TBD              # id of active task

tasks:
  - id: 1
    description: "..."
    status: verified         # pending | in-progress | coded | verified | blocked
    notes: ""                # Verifier rejection reason, or block explanation
  - id: 2
    description: "..."
    status: in-progress
    notes: ""

## Summary
<!-- Written by Finalizer when phase = complete -->

## Blocked reason
<!-- Written when phase = blocked -->
```

**Status transitions:**

```
pending → in-progress → coded → verified
                              ↘ pending   (Verifier rejected; notes has fix instruction)
in-progress → blocked         (Coder cannot proceed)
```

**Phase transitions:**

```
plan → code (Planner complete)
code → verify (Coder committed task)
verify → code (Verifier approved; more tasks remain)
verify → complete (Verifier approved; all tasks done)
verify → code (Verifier rejected; task reset to pending)
* → blocked (any phase; requires human intervention)
complete → (Finalizer outputs PIPELINE_COMPLETE)
```

---
*See [[SKILLS]] · [[skills/agent-pipeline/CLAUDE]] · [[skills/agent-pipeline/SKILL]]*