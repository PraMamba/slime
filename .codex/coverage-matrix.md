# Coverage matrix

This matrix maps every source workflow asset currently present under the source
workflow directory to its Codex target or record-only destination.

| Source asset | Target | Disposition | Lane |
| --- | --- | --- | --- |
| `.claude/agents/algorithm-expert.md` | `.codex/agents/algorithm-expert.toml` | Convert to custom agent | worker-1 |
| `.claude/agents/code-verifier.md` | `.codex/agents/code-verifier.toml` | Convert to custom agent | worker-1 |
| `.claude/agents/megatron-expert.md` | `.codex/agents/megatron-expert.toml` | Convert to custom agent | worker-1 |
| `.claude/agents/planner.md` | `.codex/agents/planner.toml` | Convert to custom agent | worker-1 |
| `.claude/agents/rollout-expert.md` | `.codex/agents/rollout-expert.toml` | Convert to custom agent | worker-1 |
| `.claude/agents/simple-code-reviewer.md` | `.codex/agents/simple-code-reviewer.toml` | Convert to custom agent | worker-1 |
| `.claude/agents/weight-sync-expert.md` | `.codex/agents/weight-sync-expert.toml` | Convert to custom agent | worker-1 |
| `.claude/commands/create-pr.md` | `.codex/skills/create-pr/SKILL.md` | Convert to workflow skill | worker-3 |
| `.claude/commands/gen-commit-msg.md` | `.codex/skills/gen-commit-msg/SKILL.md` | Convert to workflow skill | worker-3 |
| `.claude/commands/review-pr.md` | `.codex/skills/review-pr/SKILL.md` | Convert to workflow skill | worker-3 |
| `.claude/data/review-pr-change-types.md` | `.codex/workflows/review-pr/change-types.md` | Workflow support data | worker-3 |
| `.claude/data/review-pr-templates.md` | `.codex/workflows/review-pr/templates.md` | Workflow support data | worker-3 |
| `.claude/hooks/check-expert-update.sh` | `.codex/migration-notes.md` | Record-only mapping; do not activate as a hook | worker-4 |
| `.claude/rules/api-config.md` | `.codex/rules/api-config.md` | Preserve as Codex rule | worker-4 |
| `.claude/rules/code-style.md` | `.codex/rules/code-style.md` | Preserve as Codex rule | worker-4 |
| `.claude/rules/distributed.md` | `.codex/rules/distributed.md` | Preserve as Codex rule | worker-4 |
| `.claude/rules/testing.md` | `.codex/rules/testing.md` | Preserve as Codex rule | worker-4 |
| `.claude/skills/add-dynamic-filter/SKILL.md` | `.codex/skills/add-dynamic-filter/SKILL.md` | Convert to skill folder | worker-2 |
| `.claude/skills/add-eval-dataset-config/SKILL.md` | `.codex/skills/add-eval-dataset-config/SKILL.md` | Convert to skill folder | worker-2 |
| `.claude/skills/add-reward-function/SKILL.md` | `.codex/skills/add-reward-function/SKILL.md` | Convert to skill folder | worker-2 |
| `.claude/skills/add-rollout-function/SKILL.md` | `.codex/skills/add-rollout-function/SKILL.md` | Convert to skill folder | worker-2 |
| `.claude/skills/add-tests-and-ci/SKILL.md` | `.codex/skills/add-tests-and-ci/SKILL.md` | Convert to skill folder | worker-2 |

## Record-only notes

- The planning artifact referenced a local settings file, but no
  `.claude/settings.local.json` is present in the current source tree. See
  `migration-notes.md` for the discrepancy note.
