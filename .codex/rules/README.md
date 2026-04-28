# Codex Rules Index

This directory preserves slime-specific engineering rules for Codex agents and skills. Treat these as consultable project guidance, not as an automatic hook system.

## Rule map

| Rule | Path scope | Primary consumers |
| --- | --- | --- |
| `api-config.md` | `slime/utils/arguments.py`, `slime/utils/eval_config.py`, SGLang argument/config modules | `planner`, `algorithm-expert`, `rollout-expert`, add-* skills |
| `code-style.md` | General Python/project style | all custom agents and all skills |
| `distributed.md` | Megatron/SGLang backends and Ray orchestration | `megatron-expert`, `weight-sync-expert`, `rollout-expert`, `code-verifier` |
| `testing.md` | tests, `*_test.py`, `test_*.py` | `code-verifier`, `simple-code-reviewer`, `add-tests-and-ci`, PR workflow skills |

## How to use

- Before editing scoped files, ask the relevant custom agent or skill to consult this index.
- For reviews, combine `code-style.md` with any domain-specific rule that matches the touched paths.
- For test work, use `testing.md` to decide whether a change needs GPU E2E scripts, pytest unit/contract tests, or CI-template updates.
