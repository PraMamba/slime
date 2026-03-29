# Code Style Rules

Rules beyond pre-commit (Black, isort, Ruff, autoflake).

## Formatting

- **Black** with `line_length = 119`
- **isort** with `profile = "black"`, first-party: `slime`, `slime_plugins`
- **Ruff** rules: E, F, B, UP (E402 and E501 currently ignored)

## Logging

- Use `logging.getLogger(__name__)` at module level, NOT `print`
- f-string interpolation in log messages is acceptable (project convention)
- External loggers (httpx, httpcore, megatron) should be silenced at module level
- Log levels:
  - DEBUG: Detailed tracing (avoid in hot paths)
  - INFO: Milestones (training start, rollout complete, checkpoint saved)
  - WARNING: Recoverable issues
  - ERROR: Failures requiring attention
- W&B and TensorBoard metrics via `slime.utils.logging_utils.log()`, not logger

## Design Patterns

- **Prefer composition over inheritance**: Avoid deep class hierarchies
- **Plugin via `load_function()`**: Use `slime.utils.misc.load_function("module.path.func")` for
  dynamic dispatch of custom functions -- do NOT hardcode custom implementations
- **Singleton via `SingletonMeta`**: Use `slime.utils.misc.SingletonMeta` for process-wide singletons
- **Async bridge**: Use `slime.utils.async_utils.run(coro)` to bridge sync code to async event loop

## Performance Patterns

- **Avoid GPU-CPU sync**: `.item()`, `.tolist()`, `print(tensor)` cause sync
- **Prefer batch operations**: Avoid Python loops over tensor elements
- **In-place ops**: Use when safe, but careful with autograd (`.add_()` vs `+`)
- **Semaphore for concurrency**: Use `asyncio.Semaphore` to control concurrent HTTP requests

## Naming Conventions

| Type | Pattern | Example |
| --- | --- | --- |
| Ray actor class | `XxxRayActor` | `MegatronTrainRayActor` |
| Actor group | `RayXxxGroup` | `RayTrainGroup` |
| Config dataclass | `XxxConfig` | `SglangConfig`, `EvalDatasetConfig` |
| Reward function | `async rm(args, sample)` | Custom RM signature |
| Rollout function | `generate_rollout(args, ...)` | Custom rollout signature |
| Weight converter | `hf_weight_iterator(model, args)` | Megatron-to-HF pattern |
| CLI arguments | `--kebab-case` | `--advantage-estimator`, `--rm-type` |

## Tensor Conventions

- Shape convention: `[1, T, V]` for logits, `[1, T, 1]` for value head
- QKV format: `"thd"` (concatenated variable-length) or `"bshd"` (padded batch)
- Use explicit dtype/device over implicit conversion
- `strict=False` in `zip()` is existing convention but should be questioned for new code

## Import Style

- Group: stdlib, third-party, slime first-party (isort handles order)
- Avoid `from x import *`
- Relative imports within subpackages (e.g., `from .cp_utils import ...`)
- Absolute imports for cross-package (e.g., `from slime.utils.types import Sample`)
- Heavy optional deps inside functions (e.g., `torch`, `megatron` in utility modules)

## Type Annotations

- Use Python 3.10+ syntax: `int | None`, `list[dict]`, not `Optional[int]`, `List[Dict]`
- `from __future__ import annotations` is used inconsistently -- follow the file's existing pattern
- `argparse.Namespace` for configuration, `@dataclass` for data models
- No pydantic or mypy enforcement
