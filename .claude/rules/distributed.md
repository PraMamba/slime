---
paths:
  - slime/backends/megatron_utils/**
  - slime/backends/sglang_utils/**
  - slime/ray/**
---

# Distributed Code Rules

## Ray Actor Management

- **Ray placement groups**: Use PACK strategy, sorted by node IP + GPU ID
  - Actor GPUs: `--actor-num-nodes` * `--actor-num-gpus-per-node`
  - Critic GPUs: `--critic-num-nodes` * `--critic-num-gpus-per-node`
  - Rollout GPUs: `--rollout-num-gpus`
- **RolloutManager** is a single Ray actor -- central coordinator, potential bottleneck
- **RayTrainGroup** broadcasts operations to all workers via `ray.get()` on lists
- Use `ray.get()` for synchronous collection; avoid long blocking in hot paths
- `CUDA_VISIBLE_DEVICES` is managed per-actor via environment variables in `slime/ray/train_actor.py`

## NCCL Process Groups

- **Weight sync groups**: Created per PP stage between training rank 0 and SGLang engines
- **Lock mechanism**: Prevents concurrent NCCL broadcasts (deadlock prevention)
- **Reloadable process groups**: `slime/utils/reloadable_process_group.py` monkey-patches `torch.distributed`
  for colocate mode offload/onload cycles
- Never create global process groups in module-level code
- Always pass `process_group` explicitly when group matters

## Megatron Parallelism

| Dimension | Purpose | Config Flag |
| --- | --- | --- |
| TP | Tensor Parallel (shard layers) | `--tensor-model-parallel-size` |
| PP | Pipeline Parallel (split stages) | `--pipeline-model-parallel-size` |
| DP | Data Parallel (replicate) | Computed from world_size / (TP * PP * CP * EP) |
| CP | Context Parallel (split sequences) | `--context-parallel-size` |
| EP | Expert Parallel (MoE experts) | `--expert-model-parallel-size` |
| VP | Virtual Pipeline (multiple chunks) | `--num-layers-per-virtual-pipeline-stage` |

## Offload/Onload (Colocate Mode)

When `--colocate` is enabled:

- Training uses `torch_memory_saver` for GPU-CPU migration: `sleep()` pauses, `wake_up()` resumes
- SGLang uses `release_memory_occupation` / `resume_memory_occupation`
- Coordinated lifecycle: train offloads -> rollout onloads -> generate -> rollout offloads -> train onloads
- NCCL process groups must be recreated after offload cycles (reloadable groups)

**Ordering is critical**: Incorrect offload/onload sequencing causes CUDA OOM or NCCL hangs.

## Communication Patterns

- **All-reduce**: Must be called by all ranks in the group
- **All-gather**: Used for TP weight gathering before HF conversion
- **Broadcast**: Training rank 0 -> SGLang engines for weight updates
- **Barrier**: Avoid unless necessary (debugging only)
- Set `NCCL_ASYNC_ERROR_HANDLING` for deadlock debugging

## Common Pitfalls

| Issue | Cause | Fix |
| --- | --- | --- |
| Hang | Mismatched collective calls across ranks | Ensure all ranks call same op |
| Wrong results | Incorrect reduction op | Check ReduceOp (SUM vs MEAN) |
| OOM | Uncoordinated offload/onload | Verify lifecycle ordering |
| Weight corruption | TP gather dimension error | Check `all_gather_param()` logic |
| CP gradient error | Off-by-one in sequence slicing | Review `cp_utils.py` offsets |

## Debugging

- Set `TORCH_DISTRIBUTED_DEBUG=DETAIL` for verbose logging
- Set `NCCL_DEBUG=INFO` for NCCL-level issues
- Check Ray dashboard for actor status and resource allocation
- Use `ray.get()` timeouts to detect hung actors
- Check `torch_memory_saver` state for colocate issues
