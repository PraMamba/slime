# PR Review: Task Templates Reference

Referenced by: `.claude/commands/review-pr.md`

---

## Framework-Specific Review Task Templates

### Megatron Loss Tasks [Opus]

**Task: Loss Computation Correctness Review**

```
Checklist:
- Policy loss clipping logic (eps_clip / eps_clip_high asymmetry)
- Advantage computation and normalization per estimator type
- Log-prob clamping and numerical stability
- Context parallelism offset calculations
- QKV format (thd/bshd) code path divergence
- Dynamic batch size interaction with loss normalization
- TIS / OPSM / ICE-POP feature interaction
```

**Task: Context Parallelism Correctness**

```
Checklist:
- slice_log_prob_with_cp() offset calculations
- all_gather_with_cp() dimension ordering
- get_logits_and_tokens_offset_with_cp() per-rank offsets
- Zigzag ring-attention vs allgather CP code paths
- CP interaction with PP micro-batch scheduling
```

### Weight Sync Tasks [Opus]

**Task: Megatron-to-HF Converter Correctness**

```
Checklist:
- Parameter name mapping completeness
- QKV split/merge transformation correctness
- TP shard gathering dimension
- Expert weight reshaping for MoE models
- Buffer/non-parameter handling
```

**Task: NCCL Weight Broadcasting Correctness**

```
Checklist:
- Process group creation per PP stage
- Lock mechanism for concurrent broadcast prevention
- Weight iteration ordering consistency
- Expert weight EP all-gather before broadcast
- Colocate mode tensor sharing correctness
```

### Rollout Tasks [Opus]

**Task: Async Rollout Correctness**

```
Checklist:
- asyncio.Semaphore concurrency limits
- Task creation and cleanup lifecycle
- abort() mechanism for pending task cancellation
- GenerateState singleton thread safety
- HTTP client lifecycle and connection pooling
- Partial rollout buffer interaction
```

**Task: RolloutManager Coordination**

```
Checklist:
- Engine lifecycle management (create, health check, restart)
- Weight update orchestration across engine groups
- Data source lifecycle and epoch management
- Evaluation execution correctness
- Offload/onload coordination in colocate mode
- Router lifecycle management
```

### Training Actor Tasks [Opus]

**Task: MegatronTrainRayActor Lifecycle**

```
Checklist:
- TensorBackuper snapshot ordering and correctness
- _switch_model() weight swap logic
- sleep()/wake_up() offload/onload coordination
- Actor/critic role initialization
- Routing replay fill and application
```

---

## General Review Task Templates

### Logic and Boundary Conditions [Opus]

```
Applicable: Any non-doc/config changes
Checklist:
- Conditional logic errors (if/else inversion, boundary condition omission)
- Loop errors (off-by-one, infinite loops, early exit)
- Missing None/empty list handling
- Type mismatch or implicit conversion
- Exception handling (swallowing, wrong type)
- Return value errors (wrong type, missing return)
```

### Concurrency and Async [Opus]

```
Applicable: ASYNC_ROLLOUT or DISTRIBUTED type detected
Checklist:
- Race conditions in shared state
- Deadlock risks (lock ordering, nested locks)
- Missing await in async code
- Blocking calls in async functions
- Resource leaks (HTTP clients, GPU memory)
- asyncio.Semaphore lifecycle
- Ray actor state thread safety
```

### Tensor Shape and Data Type [Opus]

```
Applicable: MEGATRON_LOSS, PPO_UTILS, MEGATRON_CP detected
Checklist:
- Tensor shape mismatch (dimension errors, broadcast errors)
- Batch dimension handling (missing batch dim, wrong order)
- Sequence length and padding handling
- dtype mismatch (fp16/fp32/bf16 mixing)
- Device placement errors (CPU/GPU mixed operations)
- Gradient-related issues (missing detach, no_grad)
- view/reshape contiguity requirements
```

### Numerical Stability [Sonnet]

```
Applicable: MEGATRON_LOSS, PPO_UTILS detected
Checklist:
- log(0), division by zero, exp overflow
- Softmax stability
- Loss function numerical issues
- Gradient vanishing/exploding risks
- Mixed precision scaling issues
```

### Argument Compatibility [Sonnet]

```
Applicable: ARGUMENTS type detected
Checklist:
- New argument default maintains backward compatibility
- validate_args() updated for new constraints
- Cross-argument dependencies enforced
- Namespace mutation tracked
- Help text clear and accurate
- reset_arg() usage for Megatron overrides
```

### Sample Data Contract [Sonnet]

```
Applicable: SAMPLE_TYPE type detected
Checklist:
- to_dict()/from_dict() serialization completeness
- New fields have proper defaults
- Plugin contract tests updated
- Downstream consumers handle new fields
```

### Reward Function Correctness [Sonnet]

```
Applicable: REWARD_FUNCTION type detected
Checklist:
- async rm() signature matches contract
- Deterministic computation (same input -> same output)
- Numerical range reasonableness
- Edge case handling (empty input, malformed answers)
- batched_async_rm() compatibility if applicable
```

### SGLang Config Validation [Sonnet]

```
Applicable: SGLANG_CONFIG type detected
Checklist:
- Multi-model config consistency
- PD disaggregation settings valid
- Per-server-group override correctness
- sglang_validate_args() updated
```

### Performance Regression Risk [Sonnet]

```
Applicable: Any non-doc changes
Checklist:
- Unnecessary GPU-CPU sync (.item(), .tolist(), print(tensor))
- Memory allocation changes (potential OOM)
- Communication volume increase
- New strict=False zip usage (question justification)
- Unnecessary tensor copies
```

### Plugin Interface Changes [Sonnet]

```
Applicable: Changes to plugin-facing interfaces
Checklist:
- generate_rollout() signature stability
- RolloutFnTrainOutput/RolloutFnEvalOutput contract
- Custom function signatures (--custom-*-path)
- load_function() compatibility
- Plugin contract tests pass
```

### Documentation Format Check [Haiku]

```
Applicable: DOCS type detected
Checklist:
- Markdown format correctness
- Internal link validity
- Code example correctness
```

### Test Coverage Check [Haiku]

```
Applicable: TESTS type detected
Checklist:
- Test cases cover main paths
- NUM_GPUS declared for E2E tests
- prepare()/execute() pattern followed
- Plugin contract tests updated if interfaces changed
```

### Import and Dependencies [Haiku]

```
Applicable: Any Python file changes
Checklist:
- No wildcard imports (from x import *)
- Correct import grouping (stdlib, third-party, slime)
- Heavy optional deps inside functions
- Circular import risks
```

### Logging and Metrics [Haiku]

```
Applicable: logging, wandb, tensorboard changes
Checklist:
- Use logging.getLogger(__name__) not print
- W&B/TensorBoard metrics via logging_utils.log()
- Reasonable log levels (no DEBUG on hot paths)
- No sensitive info logged
```
