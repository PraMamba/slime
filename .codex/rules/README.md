# Rule index

Use this index to decide which project rule to consult before editing a given
area.

| Rule | Path scope | Typical consumers |
| --- | --- | --- |
| `api-config.md` | `slime/utils/arguments.py`, `slime/utils/eval_config.py`, `slime/backends/sglang_utils/sglang_config.py`, `slime/backends/sglang_utils/arguments.py` | `algorithm-expert`, `rollout-expert`, `planner`, `$add-eval-dataset-config` |
| `code-style.md` | General project code style across the repository | All agents and all skills |
| `distributed.md` | `slime/backends/megatron_utils/**`, `slime/backends/sglang_utils/**`, `slime/ray/**` | `megatron-expert`, `rollout-expert`, `weight-sync-expert`, `$review-pr` |
| `testing.md` | `tests/**`, `*_test.py`, `test_*.py` | `code-verifier`, `simple-code-reviewer`, `$add-tests-and-ci`, `$create-pr`, `$review-pr` |

## Practical guidance

- Consult `code-style.md` for any non-trivial code change.
- Consult `distributed.md` before touching Ray, NCCL, colocate mode, or weight
  synchronization flows.
- Consult `api-config.md` before adding arguments, validation, eval config, or
  dataclass-backed config surfaces.
- Consult `testing.md` before changing tests, CI templates, or plugin-facing
  contracts.
