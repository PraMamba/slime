# CLAUDE.md - slime

## WHAT: Project Overview

slime is a distributed RL post-training framework for LLMs, connecting Megatron-LM
training with SGLang inference via Ray orchestration.

**Tech Stack**: Python 3.10+ | PyTorch | Megatron-LM | SGLang | Ray

**Core Directories**:

- `slime/` - Core package
  - `backends/megatron_utils/` - Megatron training backend (loss, model, checkpoint, CP)
    - `megatron_to_hf/` - Per-model Megatron-to-HuggingFace weight converters
    - `update_weight/` - Weight sync from Megatron to SGLang (NCCL broadcast, tensor)
    - `kernels/` - Custom CUDA kernels (FP8, INT4 QAT)
  - `backends/sglang_utils/` - SGLang inference engine management
  - `ray/` - Ray-based distributed orchestration (RolloutManager, RayTrainGroup, placement groups)
  - `rollout/` - Rollout generation, reward models, data sources, filters
    - `rm_hub/` - Built-in reward functions (deepscaler, math_utils, math_dapo_utils, f1, gpqa, ifbench)
    - `filter_hub/` - Dynamic sampling filters
  - `utils/` - Arguments, logging, PPO math, async bridge, types, metrics
- `slime_plugins/` - Model-specific plugins
  - `mbridge/` - mbridge weight mapping adapters
  - `models/` - Custom Megatron model architectures (GLM4, GLM5, GPT-OSS, Qwen3.5, Qwen3-Next)
  - `rollout_buffer/` - Rollout buffer implementations
- `scripts/` - Training launch scripts and model config shell scripts
- `tools/` - Weight conversion scripts (HF↔Megatron, FP8, INT4)
- `examples/` - Custom rollout, reward, multi-agent, search-R1, VLM, async training
- `tests/` - E2E GPU tests, plugin contract tests, unit tests
- `docs/` - Sphinx documentation (English + Chinese)

## WHY: Purpose

- Enable efficient RL training for LLMs at scale (GLM-4.5 through GLM-5, Qwen3, DeepSeek, LLaMA)
- Async rollout (SGLang) + distributed training (Megatron) for high throughput
- Extensible via `--custom-*-path` function loading -- custom rollout, reward, loss, generate, filters

## HOW: Core Commands

```bash
# Recommended: use Docker
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash

# Install slime in editable mode (inside Docker or prepared env)
pip install -e . --no-deps

# Pre-commit hooks
apt install pre-commit -y
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always

# Run plugin contract tests (CPU-only, no GPU required)
python -m pytest tests/plugin_contracts/ -v

# Run unit tests
python -m pytest tests/utils/ -v

# Run E2E GPU tests (require multi-GPU)
python tests/test_qwen2.5_0.5B_short.py

# Weight conversion (HF -> Megatron torch_dist)
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
  --hf-checkpoint /path/to/hf --save /path/to/output

# Launch training (typical usage via script)
bash scripts/run-qwen3-4B.sh
```

## Boundaries

### Constraints

- Designed for NVIDIA GPU clusters; containerized execution assumed (Docker recommended)
- Multi-GPU is the expected deployment mode; single-GPU is not a tested path
- Megatron-LM and SGLang must be available in the environment (not pip-installable)
- Weights must be converted to Megatron `torch_dist` format before training
- Integration tests require multi-GPU hardware; explain skips when unavailable
- CI workflows are generated from `.j2` templates -- edit templates, NOT the YAML directly

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Use `load_function()` from `slime.utils.misc` for custom behavior injection
- Add validation in `slime_validate_args()` when adding new arguments
- Run plugin contract tests (`tests/plugin_contracts/`) when modifying extension interfaces
- Use `logging.getLogger(__name__)` for logging, not `print`

### Ask First

- Modifying the argument system in `slime/utils/arguments.py` (central config, 1760 lines)
- Changing loss computation in `slime/backends/megatron_utils/loss.py`
- Modifying the RolloutManager in `slime/ray/rollout.py`
- Changing weight sync pipeline in `slime/backends/megatron_utils/update_weight/`
- Adding new dependencies
- Deleting or renaming public APIs or CLI arguments
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`)

### Never Do

- Hardcode secrets, paths, or endpoints
- Skip pre-commit hooks
- Use wildcard imports (`from x import *`)
- Propose large-scale code refactoring (see CONTRIBUTING.md)
- Propose major modifications to Megatron
- Add files larger than 1000KB (pre-commit rejects them)
- Add a bare `dist.barrier()` without specifying the correct process group
- Remove the zero-loss safeguard (`loss + 0 * logits.sum()`) in `loss.py` -- it prevents NCCL deadlock

## Progressive Disclosure: Detailed Guides

| Task | Reference |
| --- | --- |
| Quick Start | `docs/en/get_started/quick_start.md` |
| Parameter Reference | `docs/en/get_started/usage.md` |
| Customization (18 hooks) | `docs/en/get_started/customization.md` |
| Add Custom Reward | `slime/rollout/rm_hub/deepscaler.py`, `/add-reward-function` skill |
| Add Custom Rollout | `examples/multi_agent/`, `/add-rollout-function` skill |
| Add Model Support | `slime/backends/megatron_utils/megatron_to_hf/`, `scripts/models/` |
| Algorithm Details | `slime/backends/megatron_utils/loss.py`, `slime/utils/ppo_utils.py` |
| SGLang Config | `docs/en/advanced/sglang-config.md` |
| Low-Precision Training | `docs/en/advanced/low-precision.md` |
| Debugging | `docs/en/developer_guide/debug.md` |
| CI System | `docs/en/developer_guide/ci.md` |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations
- **CI labels**: `run-ci-short`, `run-ci-sglang-config`, `run-ci-megatron`, `run-ci-precision`,
  `run-ci-ckpt`, `run-ci-plugin-contracts`, `run-ci-image`, `run-ci-changed`
  (plugin contracts run automatically on all PRs without labels)
- **CI generation**: Edit `.github/workflows/pr-test.yml.j2`, then run
  `python .github/workflows/generate_github_workflows.py`

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent | Purpose | Activation Trigger |
| --- | --- | --- |
| `planner` | Implementation planning | Before multi-file changes, new features, or architectural decisions |
| `simple-code-reviewer` | Quick code quality checks | After code changes, before committing |
| `code-verifier` | Formatting/linting/tests | After code changes, before committing |
| `megatron-expert` | Megatron backend (loss, CP, model, checkpoint) | Megatron backend code changes or questions |
| `rollout-expert` | RolloutManager, SGLang, async generation, rewards | Rollout/SGLang code changes or questions |
| `algorithm-expert` | RL algorithms (GRPO/PPO/GSPO/REINFORCE++) | Algorithm or loss computation questions |
| `weight-sync-expert` | Megatron-to-HF conversion, NCCL weight broadcast | Weight sync or model converter changes |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and
   implementation planning
2. **Code Formatting & Linting** (After coding): Use `code-verifier` to automatically
   run formatting, linting, and tests, catching syntax errors and style issues quickly
3. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for quick code
   quality checks, focusing on logic issues and code smells

### Skills (Guided Development Workflows)

Skills provide step-by-step guides for common development tasks:

- `/add-reward-function` - Reward function creation guide
- `/add-rollout-function` - Rollout function implementation guide
- `/add-eval-dataset-config` - Evaluation dataset configuration guide
- `/add-dynamic-filter` - Dynamic sampling filter guide
- `/add-tests-and-ci` - Test development and CI wiring guide

### Commands (User-invoked Actions)

Commands perform specific actions when invoked:

- `/create-pr` - Rebase, squash commits, and create/update PR with intelligent messages
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/review-pr` - Intelligent PR code review with dynamic agent allocation

### Rules (Code Quality Standards)

Project-wide standards enforced across all code changes:

- `code-style.md` - Logging, naming, performance, import, and formatting conventions
- `api-config.md` - Argument system patterns, dataclass conventions, validation rules
- `distributed.md` - Ray actors, NCCL groups, Megatron parallelism, offload/onload
- `testing.md` - E2E test patterns, pytest conventions, CI label triggers

## Code Intelligence & Navigation

When navigating and understanding code:

1. **ALWAYS prefer LSP tools over text search for code relationships**:
   - Use `goToDefinition` to jump to symbol definitions
   - Use `findReferences` to find all usages across the codebase
   - Use `goToImplementation` for interfaces/abstract methods
   - Use `workspaceSymbol` to search symbols across entire project

2. **Use Grep/Glob/Read ONLY for**:
   - Text/pattern searches in comments or strings
   - Searching configuration files (JSON, YAML, shell scripts)
   - Exploratory "fuzzy" searches when unsure what you're looking for
   - Finding files by name patterns

3. **Workflow**:
   - First: Use LSP to understand code structure and relationships
   - Second: Use text tools only when LSP cannot help (non-code content)
   - NEVER read entire large files to find references; use LSP instead

Remember: LSP provides semantic understanding (types, inheritance, references), while grep only provides text matching.
