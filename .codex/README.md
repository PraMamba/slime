# Codex workflow assets for slime

This directory provides project-local Codex assets for slime's expert agents,
implementation skills, PR workflows, and scoped rules.

## Layout

- `agents/` — custom specialist agents for planning, verification, Megatron,
  rollout/SGLang, weight sync, and algorithm work.
- `skills/` — reusable task workflows for adding reward functions, rollout
  functions, eval dataset config, dynamic filters, tests/CI wiring, and PR
  operations.
- `rules/` — scoped project rules for API/config work, code style,
  distributed systems, and testing.
- `workflows/review-pr/` — supporting review-pr data files.

## Using the custom agents

Name the specialist you want in the prompt so Codex can route work to the right
agent.

| Agent | When to use it |
| --- | --- |
| `algorithm-expert` | RL loss math, PPO/GRPO/GSPO, reward shaping, KL/clipping, and advantage estimation |
| `code-verifier` | Post-change formatting, linting, tests, and verification loops |
| `megatron-expert` | Megatron model setup, loss code, context parallelism, checkpoints, and data iterators |
| `planner` | Multi-file implementation plans or architecture-sensitive changes |
| `rollout-expert` | RolloutManager, SGLang engines, async generation, rewards, and eval rollout behavior |
| `simple-code-reviewer` | Lightweight quality review after a focused change |
| `weight-sync-expert` | Megatron-to-HF conversion, NCCL broadcasts, model converters, and mbridge adapters |

Example prompts:

- `Use the rollout-expert agent to inspect changes under slime/backends/sglang_utils/.`
- `Ask the weight-sync-expert agent to review converter changes before merging.`

## Using the skills

These skills are intended to be discoverable through this repository's Codex
workflow layer. Invoke them directly when you want the corresponding workflow.

### Build/change skills

- `$add-dynamic-filter` — add rollout buffer filtering or sample-group hooks.
- `$add-eval-dataset-config` — add or validate eval dataset config behavior.
- `$add-reward-function` — add a custom reward model or reward shaping logic.
- `$add-rollout-function` — add a custom rollout generation function.
- `$add-tests-and-ci` — extend tests, CI registration, or workflow coverage.

### Workflow skills

- `$create-pr [--draft] [--base <branch>]` — rebase, squash, preview, and open
  a PR with explicit safety gates.
- `$gen-commit-msg [--amend] [--scope <scope>]` — generate a commit message from
  staged changes.
- `$review-pr [<number>] [--quick]` — perform dynamic PR review using the
  review-pr workflow data.

If you are not using the `$name` workflow surface, ask Codex in natural
language to use the named workflow skill.

## Rules to consult

| Rule | Scope |
| --- | --- |
| `rules/api-config.md` | CLI arguments, eval config, dataclasses, and SGLang config files |
| `rules/code-style.md` | Logging, imports, naming, performance, typing, and tensor conventions |
| `rules/distributed.md` | Megatron, SGLang, Ray, NCCL process groups, and colocate/offload flows |
| `rules/testing.md` | Pytest, plugin contract tests, E2E scripts, and CI templates |

Ask Codex to consult the relevant rule before editing the corresponding area,
or reference `rules/README.md` for a consumer-oriented index.

## PR workflow safety expectations

The PR-oriented workflows keep the original safety posture:

- verify the current branch and working tree before rewriting history,
- preview generated commit/PR text before pushing,
- preserve explicit confirmation before rebase, reset, force-push, or PR
  creation,
- keep review workflows read-oriented and non-posting by default.

## Migration boundaries

- Project-local hooks are documented but not activated here.
- Local permission/settings files are documented only when present in the
  source tree.
- Active assets should stay Codex-native: no live imports from legacy command
  assets, and no slash-command-only usage guidance.

