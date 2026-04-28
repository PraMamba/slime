# Migration Notes

This file records compatibility decisions made while porting the prior Claude-oriented workflow layer to Codex-compatible project assets.

## Active conversions

| Area | Decision |
| --- | --- |
| Agents | Converted each Markdown expert into project custom-agent TOML with `name`, `description`, and `developer_instructions`. Claude-only `tools` and `model` fields were not copied as active Codex schema. |
| Skills | Preserved the five original implementation skills as Codex skill folders with `SKILL.md` files and source headers. |
| Commands | Reframed command documents as workflow skills: `create-pr`, `gen-commit-msg`, and `review-pr`. |
| Review data | Moved review change-type/template data to `.codex/workflows/review-pr/` and referenced it from the `review-pr` skill. |
| Rules | Preserved project rules under `.codex/rules/` and added a Codex-facing rule index. |

## Record-only assets

| Source | Treatment | Reason |
| --- | --- | --- |
| `.claude/hooks/check-expert-update.sh` | Documented below only; not installed or activated. | The requested first version intentionally excludes active hook behavior. |
| `.claude/settings.local.json` | Summarized below only; not migrated as active permissions/config. | The file is a Claude-local allowlist and should not silently grant Codex permissions. |

## Preserved hook knowledge, not active hook behavior

The old reminder script mapped edited paths to expert agents. The mapping is preserved here for humans and future tooling:

| Changed path pattern | Suggested Codex agent |
| --- | --- |
| `slime/backends/megatron_utils/loss*`, `actor*`, `model*`, `cp_utils*`, `data*`, `checkpoint*`, `initialize*` | `megatron-expert` |
| `slime/backends/megatron_utils/update_weight/`, `slime/backends/megatron_utils/megatron_to_hf/`, `slime_plugins/mbridge/` | `weight-sync-expert` |
| `slime/ray/rollout*`, `slime/rollout/`, `slime/backends/sglang_utils/` | `rollout-expert` |
| `slime/utils/ppo_utils*`, `slime/backends/megatron_utils/loss*`, `slime/rollout/rm_hub/` | `algorithm-expert` |

If automatic reminders are desired later, design a new Codex-native mechanism instead of copying the Claude hook directly.

## Settings summary

The old local settings allowed selected shell commands around Git, worktrees, fetching, and GitHub repository inspection. This migration records that permission intent but does not add an active permission override. Codex sessions should continue to follow the current sandbox/approval policy supplied by the runtime.

## Fidelity and usability tradeoffs

- Codex usability takes priority over preserving slash-command-only invocation text.
- Source paths are retained in converted assets as audit headers, coverage rows, or migration notes.
- Destructive/remote Git safety gates from the PR workflow were preserved in the `create-pr` skill.
