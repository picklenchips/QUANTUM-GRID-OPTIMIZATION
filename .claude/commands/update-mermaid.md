Run `npm run diagrams` to regenerate all SVG files from the `.mmd` source files in `docs/diagrams/`.

After running:
1. Report which SVGs regenerated successfully and flag any parse errors
2. If a parse error occurred, read the offending `.mmd` file and identify the syntax issue

Workflow when editing diagrams:
- Edit the `.mmd` file in `docs/diagrams/`
- Run `/update-mermaid` to regenerate its `.svg`
- The `.md` files that reference the SVG update automatically (they embed by path)
- Commit `.mmd` + `.svg` together

Source map (which .md files reference which diagrams):
- `docs/technical/V0.0-BEN-DIAGRAMS.md` — all 9 BEN-DIAGRAMS diagrams
- `docs/technical/V0.0-TECH-BRIEF.md` — system-overview, agent-data-connection, product-surface-v0, onboarding-flow, data-flow, internal-architecture, data-model-erd-full
- `docs/SCHEMA_CONTEXT.md` — advisor-request-flow-v0
- `docs/DATABASE_PLAN.md` — data-model-erd-v0
- `PRD.md` — roadmap
- `docs/BUILD_WORKFLOW.md` — sprint-sequence
- `CLAUDE.md` — repo-structure