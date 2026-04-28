<!-- Source: `.claude/rules/testing.md` -->

---
paths:
  - tests/**
  - '*_test.py'
  - test_*.py
---

# Testing Rules

## Test Types

slime has two distinct test categories:

### 1. End-to-End Tests (GPU Required)

Located in `tests/test_*.py`. These are **standalone scripts**, NOT standard pytest tests.

Pattern:

```python
NUM_GPUS = 4  # Used by CI for GPU allocation

def prepare():
    """Download models, datasets, setup environment."""
    ...

def execute():
    """Run training with specific configuration."""
    ...

if __name__ == "__main__":
    prepare()
    execute()
```

- Run with: `python tests/test_*.py`
- Require real models and GPU access
- CI uses `gpu_lock_exec.py` for GPU resource management
- Declare `NUM_GPUS` at module level for CI detection

### 2. Unit / Contract Tests (No GPU)

Located in `tests/plugin_contracts/`, `tests/utils/`, `tests/test_chunked_gae.py`.

Standard pytest with proper fixtures, markers, parametrize:

```python
@pytest.mark.parametrize("B,T", [(1, 10), (2, 20)])
def test_chunked_gae(B, T):
    ...
```

## Pytest Markers

| Marker | When to Use |
| --- | --- |
| `@pytest.mark.unit` | Unit tests |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.system` | System-level tests |
| `@pytest.mark.skipduringci` | Skip in CI |
| `@pytest.mark.pleasefixme` | Known broken tests |
| `@pytest.mark.parametrize(...)` | Parameterized tests |

Markers must be registered in `pyproject.toml` (`--strict-markers` enforced).

## Plugin Contract Tests

Location: `tests/plugin_contracts/`

Test that plugin interfaces conform to expected contracts:

- `test_plugin_rollout_contracts.py` -- rollout function I/O contract
- `test_plugin_runtime_hook_contracts.py` -- before-train-step hook contract
- `test_plugin_generate_contracts.py` -- custom generate function contract

These tests stub heavy dependencies (Ray, SGLang, transformers) using `sys.modules` injection
in `tests/plugin_contracts/_shared.py`.

**Run on ALL PRs** without CI labels.

## CI Configuration

CI workflows are generated from Jinja2 templates:

- **Template**: `.github/workflows/pr-test.yml.j2`
- **Generated**: `.github/workflows/pr-test.yml`
- **IMPORTANT**: Edit the `.j2` template, NOT the `.yml` file directly

### CI Triggers

| Label | Test Scope |
| --- | --- |
| `run-ci-short` | Short E2E tests |
| `run-ci-sglang-config` | SGLang config tests |
| `run-ci-megatron` | Megatron backend tests |
| `run-ci-precision` | Precision/FP8 tests |
| `run-ci-ckpt` | Checkpoint tests |
| `run-ci-image` | Image/VLM tests |
| `run-ci-changed` | Auto-detect changed test files |
| (no label) | Plugin contracts only |

### Test Environment

- Docker container: `slimerl/slime:latest`
- GPU access via self-hosted runners
- Install: `pip install -e . --no-deps`
- GPU locking: `tests/ci/gpu_lock_exec.py`

## Test Structure (Pytest)

```python
def test_<what>_<condition>_<expected>():
    """Test that <what> does <expected> when <condition>."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

## Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Use descriptive assertion messages
- `assert` statements in production code are acceptable (project convention, no `-O` flag)

## Fixtures

- Prefer `tmp_path` over manual temp directories
- Use `monkeypatch` for environment variables
- Scope expensive fixtures appropriately (`session` > `module` > `function`)
- Use `sys.modules` injection for stubbing heavy dependencies in contract tests
