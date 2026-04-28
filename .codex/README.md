# Codex Workflow Layer for slime

This directory contains the Codex-compatible workflow layer migrated from the prior Claude-oriented project guidance. It is organized for Codex discoverability: custom agents live in `agents/`, skills live in `skills/`, review workflow data lives in `workflows/`, and reusable project rules live in `rules/`.

## Custom agents

Use these agents when the task benefits from focused slime expertise:

| Agent | Use for |
| --- | --- |
| `algorithm-expert` | PPO, reward functions, advantage estimation, loss math, and algorithm correctness. |
| `code-verifier` | Validation plans, format/test checks, and completion evidence. |
| `megatron-expert` | Megatron backend, parallelism, losses, model/data/checkpoint paths. |
| `planner` | Multi-step implementation planning before edits. |
| `rollout-expert` | SGLang rollout, async generation, rollout manager, and rollout plugins. |
| `simple-code-reviewer` | Lightweight review for small changes. |
| `weight-sync-expert` | Megatron-to-HF conversion, NCCL weight update, and mbridge sync paths. |

Example prompt: “Use the `rollout-expert` agent to inspect this rollout change, then have `code-verifier` propose verification.”

## Skills

Configured skill folders are listed in `config.toml` under `skills.config`.

### Domain implementation skills

- `add-dynamic-filter`
- `add-eval-dataset-config`
- `add-reward-function`
- `add-rollout-function`
- `add-tests-and-ci`

Example prompt: “Use the `add-reward-function` skill to add a new reward function and include tests.”

### Workflow skills

- `create-pr` — PR preparation with explicit safety gates for rebase/reset/push/PR creation.
- `gen-commit-msg` — commit message drafting adapted to the project’s Lore commit protocol.
- `review-pr` — PR review workflow using local change-type and review-template data in `workflows/review-pr/`.

Example prompt: “Use `$review-pr` on PR 123 and keep it quick.”

## Rules

See `rules/README.md` for the rule index. In short:

- consult `api-config.md` before changing arguments or config surfaces;
- consult `code-style.md` for general project style;
- consult `distributed.md` for Ray, NCCL, Megatron, SGLang, and colocate paths;
- consult `testing.md` before adding or changing tests/CI.

## Omitted active behavior

The migrated layer does not enable automatic hooks or local permission allowlists. Those were recorded in `migration-notes.md` for auditability and can be revisited later if Codex hook or permission behavior is intentionally adopted for this repo.

## Coverage and audit

- `coverage-matrix.md` maps each source asset to its Codex target or record-only rationale.
- `migration-notes.md` documents adaptation decisions, intentionally omitted active features, and the path-to-expert reminder knowledge preserved from the old hook.
