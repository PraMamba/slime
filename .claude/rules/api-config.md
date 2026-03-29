---
paths:
  - slime/utils/arguments.py
  - slime/utils/eval_config.py
  - slime/backends/sglang_utils/sglang_config.py
  - slime/backends/sglang_utils/arguments.py
---

# API & Config Rules

## Argument System

slime uses `argparse.Namespace` as the primary configuration mechanism. All arguments
are defined in `slime/utils/arguments.py`.

### Adding Arguments

New arguments must be added inside `get_slime_extra_args_provider()` -> `add_slime_arguments()`,
in the appropriate category closure:

| Category | Closure | Prefix |
| --- | --- | --- |
| Cluster | `add_cluster_arguments()` | `--actor-*`, `--critic-*`, `--rollout-*` |
| Training | `add_train_arguments()` | `--custom-*-path`, `--freeze-*`, `--qkv-format` |
| Rollout | `add_rollout_arguments()` | `--rollout-*`, `--hf-checkpoint` |
| Data | `add_data_arguments()` | `--prompt-data`, `--rm-type`, `--n-samples-*` |
| Algorithm | `add_algo_arguments()` | `--advantage-estimator`, `--eps-clip`, `--kl-*` |
| Eval | `add_eval_arguments()` | `--eval-*` |
| Debug | `add_debug_arguments()` | `--use-wandb`, `--use-tensorboard` |

### Override Megatron Defaults

Use `reset_arg()` to override Megatron's default values:

```python
reset_arg(parser, "micro_batch_size", default=1)
```

### Validation

- Add validation logic in `slime_validate_args()` at the bottom of `arguments.py`
- SGLang-specific validation in `validate_args()` of `slime/backends/sglang_utils/arguments.py`
- Raise `ValueError` with clear message for invalid combinations
- Use assertions for internal invariants only
- Consider cross-argument dependencies (e.g., `colocate` implies `offload_train`)

### Backward Compatibility

- **Adding fields**: Add with default value that maintains existing behavior
- **Removing fields**: Deprecate first, add warning
- **Changing defaults**: Document in PR description, consider impact on all users

## Dataclass Conventions

Used for data models (NOT for CLI configuration):

```python
@dataclass
class XxxConfig:
    """One-line description."""
    required_field: str
    optional_field: int = 32
```

Key dataclasses:

- `Sample` (`slime/utils/types.py`) -- core data unit with `to_dict()` / `from_dict()`
- `EvalDatasetConfig` (`slime/utils/eval_config.py`) -- per-dataset eval settings
- `SglangConfig` / `ModelConfig` / `ServerGroupConfig` (`slime/backends/sglang_utils/sglang_config.py`)
- `RolloutFnTrainOutput` / `RolloutFnEvalOutput` (`slime/rollout/base_types.py`)

### Sample Dataclass Rules

When modifying `Sample`:

- Update both `to_dict()` and `from_dict()` serialization
- Note that `from_dict()` uses `setattr` for unknown keys -- maintain field discipline
- Plugin contract tests (`tests/plugin_contracts/`) may need updates

## SGLang Config (YAML)

`SglangConfig` is loaded from `--sglang-config` YAML file:

- Supports multi-model deployment (actor, ref, reward models)
- Per-model and per-server-group overrides
- PD disaggregation, encoder-only engines
- Validated via `validate_args()` in `slime/backends/sglang_utils/arguments.py` (imported as `sglang_validate_args` in the main arguments module)

## Eval Config

`EvalDatasetConfig` supports:

- Per-dataset `rm_type`, sampling params, custom generate function
- YAML-based via `--eval-config` with OmegaConf
- CLI-based via `--eval-prompt-data` key-value pairs
- Built via `build_eval_dataset_configs()`
