# PR Review: Task Templates Reference

Source: `.claude/data/review-pr-templates.md`
Referenced by: `.codex/skills/review-pr/SKILL.md`

---

## Framework-specific review task templates

### Megatron loss tasks

**Task: Loss Computation Correctness Review**

```text
Checklist:
- policy loss clipping logic (eps_clip / eps_clip_high asymmetry)
- advantage computation and normalization per estimator type
- log-prob clamping and numerical stability
- context parallelism offset calculations
- QKV format (`thd` / `bshd`) path divergence
- dynamic batch size interaction with loss normalization
- TIS / OPSM / ICE-POP feature interaction
```

**Task: Context Parallelism Correctness**

```text
Checklist:
- slice_log_prob_with_cp() offset calculations
- all_gather_with_cp() dimension ordering
- get_logits_and_tokens_offset_with_cp() per-rank offsets
- zigzag ring-attention vs all-gather CP code paths
- CP interaction with PP micro-batch scheduling
```

### Weight sync tasks

**Task: Megatron-to-HF Converter Correctness**

```text
Checklist:
- parameter name mapping completeness
- QKV split/merge transformation correctness
- TP shard gathering dimension
- expert weight reshaping for MoE models
- buffer / non-parameter handling
```

**Task: NCCL Weight Broadcasting Correctness**

```text
Checklist:
- process-group creation per PP stage
- lock mechanism for concurrent broadcast prevention
- weight iteration ordering consistency
- expert weight EP all-gather before broadcast
- colocate-mode tensor sharing correctness
```

### Rollout tasks

**Task: Async Rollout Correctness**

```text
Checklist:
- asyncio.Semaphore concurrency limits
- task creation and cleanup lifecycle
- abort() mechanism for pending-task cancellation
- GenerateState singleton thread safety
- HTTP client lifecycle and connection pooling
- partial rollout buffer interaction
```

**Task: RolloutManager Coordination**

```text
Checklist:
- engine lifecycle management (create, health check, restart)
- weight update orchestration across engine groups
- data source lifecycle and epoch management
- evaluation execution correctness
- offload/onload coordination in colocate mode
- router lifecycle management
```

### Training actor tasks

**Task: MegatronTrainRayActor Lifecycle**

```text
Checklist:
- TensorBackuper snapshot ordering and correctness
- _switch_model() weight swap logic
- sleep()/wake_up() offload/onload coordination
- actor/critic role initialization
- routing replay fill and application
```

---

## General review task templates

### Logic and boundary conditions

```text
Applicable: any non-doc/config changes
Checklist:
- conditional logic errors (if/else inversion, boundary omission)
- loop errors (off-by-one, infinite loops, early exit)
- missing None / empty-list handling
- type mismatch or implicit conversion
- exception handling (swallowing, wrong type)
- return-value errors (wrong type, missing return)
```

### Concurrency and async

```text
Applicable: ASYNC_ROLLOUT or DISTRIBUTED detected
Checklist:
- race conditions in shared state
- deadlock risks (lock ordering, nested locks)
- missing await in async code
- blocking calls in async functions
- resource leaks (HTTP clients, GPU memory)
- asyncio.Semaphore lifecycle
- Ray actor-state thread safety
```

### Tensor shape and data type

```text
Applicable: MEGATRON_LOSS, PPO_UTILS, or MEGATRON_CP detected
Checklist:
- tensor shape mismatch (dimension or broadcast errors)
- batch-dimension handling
- sequence length and padding handling
- dtype mismatch (fp16/fp32/bf16 mixing)
- device-placement errors (CPU/GPU mixed operations)
- gradient-related issues (missing detach, no_grad)
- view/reshape contiguity requirements
```

### Numerical stability

```text
Applicable: MEGATRON_LOSS or PPO_UTILS detected
Checklist:
- log(0), division by zero, exp overflow
- softmax stability
- loss-function numerical issues
- gradient vanishing/exploding risks
- mixed-precision scaling issues
```

### Argument compatibility

```text
Applicable: ARGUMENTS detected
Checklist:
- new argument default maintains backward compatibility
- validate_args() updated for new constraints
- cross-argument dependencies enforced
- namespace mutation tracked
- help text clear and accurate
- reset_arg() usage for Megatron overrides
```

### Sample data contract

```text
Applicable: SAMPLE_TYPE detected
Checklist:
- to_dict()/from_dict() serialization completeness
- new fields have proper defaults
- plugin contract tests updated
- downstream consumers handle new fields
```

### Reward function correctness

```text
Applicable: REWARD_FUNCTION detected
Checklist:
- async rm() signature matches contract
- deterministic computation (same input -> same output)
- numerical range reasonableness
- edge-case handling (empty input, malformed answers)
- batched_async_rm() compatibility if applicable
```

### SGLang config validation

```text
Applicable: SGLANG_CONFIG detected
Checklist:
- multi-model config consistency
- PD disaggregation settings valid
- per-server-group override correctness
- sglang_validate_args() updated
```

### Performance regression risk

```text
Applicable: any non-doc changes
Checklist:
- unnecessary GPU-CPU sync (.item(), .tolist(), print(tensor))
- memory-allocation changes with OOM risk
- communication-volume increase
- new strict=False zip usage (justify it)
- unnecessary tensor copies
```

### Plugin interface changes

```text
Applicable: plugin-facing interface changes
Checklist:
- generate_rollout() signature stability
- RolloutFnTrainOutput / RolloutFnEvalOutput contract
- custom function signatures (--custom-*-path)
- load_function() compatibility
- plugin contract tests pass
```

### Documentation format check

```text
Applicable: DOCS detected
Checklist:
- Markdown format correctness
- internal link validity
- code-example correctness
```

### Test coverage check

```text
Applicable: TESTS detected
Checklist:
- test cases cover main paths
- NUM_GPUS declared for E2E tests
- prepare()/execute() pattern followed
- plugin contract tests updated if interfaces changed
```

### Import and dependencies

```text
Applicable: any Python file changes
Checklist:
- no wildcard imports
- correct import grouping (stdlib, third-party, slime)
- heavy optional deps moved inside functions when appropriate
- circular-import risks
```

### Logging and metrics

```text
Applicable: runtime / training changes
Checklist:
- new logging is actionable, not noisy
- metrics names are stable and descriptive
- failure paths log enough context
- distributed logs avoid excessive duplication
```
