# PR Review: Change Type Detection Reference

Referenced by: `.claude/commands/review-pr.md`

---

## CRITICAL Level (Must use Opus)

| Change Type | File Path Pattern | Code Pattern |
| --- | --- | --- |
| **MEGATRON_LOSS** | `slime/backends/megatron_utils/loss.py` | `policy_loss_function`, `value_loss_function`, `compute_advantages_and_returns` |
| **MEGATRON_ACTOR** | `slime/backends/megatron_utils/actor.py` | `MegatronTrainRayActor`, `train_actor`, `train_critic` |
| **MEGATRON_CP** | `slime/backends/megatron_utils/cp_utils.py` | `slice_log_prob_with_cp`, `all_gather_with_cp` |
| **WEIGHT_SYNC** | `slime/backends/megatron_utils/update_weight/` | `UpdateWeightFromDistributed`, `update_weights`, NCCL broadcast |
| **MEGATRON_TO_HF** | `slime/backends/megatron_utils/megatron_to_hf/` | `hf_weight_iterator`, `convert_to_hf` |
| **ROLLOUT_MANAGER** | `slime/ray/rollout.py` | `RolloutManager`, `ServerGroup`, engine lifecycle |

## HIGH Level (Recommend Opus)

| Change Type | File Path Pattern | Code Pattern |
| --- | --- | --- |
| **PPO_UTILS** | `slime/utils/ppo_utils.py` | `compute_advantages`, KL divergence, advantage normalization |
| **SGLANG_ENGINE** | `slime/backends/sglang_utils/sglang_engine.py` | `SGLangEngine`, weight update, memory management |
| **SGLANG_CONFIG** | `slime/backends/sglang_utils/sglang_config.py` | `SglangConfig`, `ModelConfig`, `ServerGroupConfig` |
| **ASYNC_ROLLOUT** | `slime/rollout/sglang_rollout.py` | `generate_rollout_async`, `GenerateState`, async generation |
| **DISTRIBUTED** | `slime/ray/placement_group.py`, `slime/ray/actor_group.py` | Ray placement groups, NCCL, process groups |
| **ARGUMENTS** | `slime/utils/arguments.py` | `parse_args`, `validate_args`, argument definitions |
| **SAMPLE_TYPE** | `slime/utils/types.py` | `Sample`, `RolloutBatch`, `to_dict`, `from_dict` |
| **TRAIN_LOOP** | `train.py`, `train_async.py` | Main training loop, orchestration |

## MEDIUM Level (Use Sonnet)

| Change Type | File Path Pattern | Code Pattern |
| --- | --- | --- |
| **REWARD_FUNCTION** | `slime/rollout/rm_hub/` | `async_rm`, `batched_async_rm`, reward dispatch |
| **DATA_SOURCE** | `slime/rollout/data_source.py` | `DataSource`, `RolloutDataSource`, buffer |
| **EVAL_CONFIG** | `slime/utils/eval_config.py` | `EvalDatasetConfig`, `build_eval_dataset_configs` |
| **MEGATRON_DATA** | `slime/backends/megatron_utils/data.py` | `DataIterator`, `get_batch`, micro-batch |
| **MEGATRON_MODEL** | `slime/backends/megatron_utils/model.py` | `setup_model_and_optimizer`, `forward_only`, `train` |
| **FILTER_HUB** | `slime/rollout/filter_hub/` | Dynamic sampling filters |
| **HEALTH_MONITOR** | `slime/utils/health_monitor.py` | `RolloutHealthMonitor`, engine health checks |
| **HTTP_UTILS** | `slime/utils/http_utils.py` | Async HTTP client, retry logic |
| **PLUGIN_MODEL** | `slime_plugins/models/` | Custom model providers |
| **PLUGIN_MBRIDGE** | `slime_plugins/mbridge/` | Weight mapping bridges |
| **CHECKPOINT** | `slime/backends/megatron_utils/checkpoint.py` | Checkpoint load/save |
| **OFFLOAD** | - | `torch_memory_saver`, `sleep`, `wake_up`, `offload`, `onload` |

## LOW Level (Use Haiku)

| Change Type | File Path Pattern | Code Pattern |
| --- | --- | --- |
| **TESTS** | `tests/` | Test files |
| **DOCS** | `docs/`, `*.md` | Documentation |
| **CONFIG_ONLY** | `*.yaml`, `*.json`, `*.toml` | Configuration files |
| **SCRIPTS** | `scripts/` | Shell scripts, model configs |
| **EXAMPLES** | `examples/` | Example implementations |
| **TOOLS** | `tools/` | Conversion scripts |

---

## Framework-Specific Risk Identification

### Megatron Backend Risks (When MEGATRON_* types detected)

- **CP offset error**: Off-by-one in context parallelism sequence slicing causes silent gradient corruption
- **Loss computation NaN**: Missing log-prob clamping, division by zero in advantage normalization
- **Pipeline schedule error**: Micro-batch ordering with PP, dynamic batch size interaction
- **Weight backup/restore error**: TensorBackuper snapshot order, `_switch_model()` logic
- **QKV format mismatch**: `thd` vs `bshd` code paths diverge in loss computation

### Weight Sync Risks (When WEIGHT_SYNC or MEGATRON_TO_HF detected)

- **Converter key mismatch**: Megatron parameter names don't match HF expected names
- **TP gather dimension error**: Wrong dimension in `all_gather_param()`
- **NCCL broadcast hang**: Lock ordering, process group mismatch
- **Expert weight corruption**: EP all-gather missing for MoE models

### Rollout Risks (When ROLLOUT_* or ASYNC_* detected)

- **Async deadlock**: Semaphore count mismatch, pending task cleanup in abort()
- **Race condition**: GenerateState singleton mutation from concurrent tasks
- **Memory leak**: HTTP client not properly closed, pending task references
- **Data alignment**: Reward/sample ordering mismatch after async completion

### Argument Risks (When ARGUMENTS detected)

- **Default change breaks existing**: New default changes behavior for current users
- **Missing validation**: Cross-argument dependency not enforced
- **Namespace mutation**: args modified in multiple places post-parsing

---

## Risk Linkage Rules

| Detected Change | Auto-Linked Review |
| --- | --- |
| MEGATRON_LOSS changes | CP interaction + advantage estimation check |
| WEIGHT_SYNC changes | MEGATRON_TO_HF converter + NCCL check |
| ASYNC_ROLLOUT changes | Semaphore + abort + GenerateState check |
| ARGUMENTS changes | validate_args + backward compatibility check |
| SAMPLE_TYPE changes | Plugin contract + serialization check |
| SGLANG_CONFIG changes | Multi-model + PD disaggregation check |
| TRAIN_LOOP changes | Weight update + offload lifecycle check |

---

## Core Framework Paths (Must Use Opus)

**Megatron Backend Core**:
- `slime/backends/megatron_utils/loss.py`
- `slime/backends/megatron_utils/actor.py`
- `slime/backends/megatron_utils/cp_utils.py`

**Weight Synchronization Core**:
- `slime/backends/megatron_utils/update_weight/`
- `slime/backends/megatron_utils/megatron_to_hf/`

**Rollout Core**:
- `slime/ray/rollout.py`
- `slime/rollout/sglang_rollout.py`

**Algorithm Core**:
- `slime/utils/ppo_utils.py`
