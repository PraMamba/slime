# PR Review: Change Type Detection Reference

Source: `.claude/data/review-pr-change-types.md`
Referenced by: `.codex/skills/review-pr/SKILL.md`

---

## CRITICAL Level

Use the deepest available review path for these changes (for example a high-reasoning reviewer or the most capable review subagent allowed in the current runtime).

| Change Type | File Path Pattern | Code Pattern |
| --- | --- | --- |
| **MEGATRON_LOSS** | `slime/backends/megatron_utils/loss.py` | `policy_loss_function`, `value_loss_function`, `compute_advantages_and_returns` |
| **MEGATRON_ACTOR** | `slime/backends/megatron_utils/actor.py` | `MegatronTrainRayActor`, `train_actor`, `train_critic` |
| **MEGATRON_CP** | `slime/backends/megatron_utils/cp_utils.py` | `slice_log_prob_with_cp`, `all_gather_with_cp` |
| **WEIGHT_SYNC** | `slime/backends/megatron_utils/update_weight/` | `UpdateWeightFromDistributed`, `update_weights`, NCCL broadcast |
| **MEGATRON_TO_HF** | `slime/backends/megatron_utils/megatron_to_hf/` | `hf_weight_iterator`, `convert_to_hf` |
| **ROLLOUT_MANAGER** | `slime/ray/rollout.py` | `RolloutManager`, `ServerGroup`, engine lifecycle |

## HIGH Level

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

## MEDIUM Level

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

## LOW Level

| Change Type | File Path Pattern | Code Pattern |
| --- | --- | --- |
| **TESTS** | `tests/` | Test files |
| **DOCS** | `docs/`, `*.md` | Documentation |
| **CONFIG_ONLY** | `*.yaml`, `*.json`, `*.toml` | Configuration files |
| **SCRIPTS** | `scripts/` | Shell scripts, model configs |
| **EXAMPLES** | `examples/` | Example implementations |
| **TOOLS** | `tools/` | Conversion scripts |

---

## Framework-specific risk identification

### Megatron backend risks

- CP offset error: off-by-one in context parallel sequence slicing causes silent gradient corruption
- loss computation NaN: missing log-prob clamping or division-by-zero protection
- pipeline schedule error: micro-batch ordering interacts badly with dynamic batch sizing
- weight backup/restore error: tensor snapshot order or `_switch_model()` logic drift
- QKV format mismatch: `thd` vs `bshd` paths diverge in loss handling

### Weight sync risks

- converter key mismatch: Megatron parameter names do not match HF expectations
- TP gather dimension error: wrong dimension in `all_gather_param()`
- NCCL broadcast hang: lock ordering or process-group mismatch
- expert weight corruption: EP all-gather missing for MoE models

### Rollout risks

- async deadlock: semaphore mismatch or incomplete pending-task cleanup in `abort()`
- race condition: shared `GenerateState` mutation under concurrency
- memory leak: HTTP client not closed or pending task references retained
- data alignment: reward/sample ordering mismatch after async completion

### Argument risks

- default change breaks existing users
- missing validation for cross-argument dependencies
- namespace mutation after parsing

---

## Risk linkage rules

| Detected Change | Auto-linked Review |
| --- | --- |
| MEGATRON_LOSS changes | CP interaction + advantage estimation check |
| WEIGHT_SYNC changes | converter + NCCL correctness check |
| ASYNC_ROLLOUT changes | semaphore + abort + GenerateState check |
| ARGUMENTS changes | backward-compatibility + validation check |
| SAMPLE_TYPE changes | plugin contract + serialization check |
| SGLANG_CONFIG changes | multi-model + PD disaggregation check |
| TRAIN_LOOP changes | weight update + offload lifecycle check |

## Core framework paths requiring deepest review

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
