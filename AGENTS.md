# AGENTS.md - slime Codex Guide

## WHAT: Project Overview

slime is a distributed RL post-training framework for LLMs. It connects
Megatron-LM training with SGLang inference through Ray orchestration, with
extensible hooks for rollout generation, reward computation, model conversion,
weight synchronization, and evaluation.

**Tech Stack**: Python 3.10+ | PyTorch | Megatron-LM | SGLang | Ray | Docker/NVIDIA GPUs

**Codex Scope**: This file is the project-level instruction surface for Codex
agents operating in this repository. It governs this directory and all child
paths unless a deeper `AGENTS.md` overrides it.

**Core Directories**:

- `slime/` - Core package
  - `backends/megatron_utils/` - Megatron training backend: loss, model,
    checkpointing, context parallelism, data, and initialization
    - `megatron_to_hf/` - Per-model Megatron-to-HuggingFace converters
    - `update_weight/` - Weight sync from Megatron to SGLang via NCCL broadcast
    - `kernels/` - Custom CUDA kernels, including FP8 and INT4 QAT paths
  - `backends/sglang_utils/` - SGLang engine configuration and lifecycle
  - `ray/` - Ray orchestration: `RolloutManager`, `RayTrainGroup`, placement groups
  - `rollout/` - Rollout generation, reward models, data sources, filters
    - `rm_hub/` - Built-in reward functions such as deepscaler, math utilities,
      F1, GPQA, and IFBench
    - `filter_hub/` - Dynamic sampling filters
  - `utils/` - Arguments, logging, PPO math, async bridge, types, metrics
- `slime_plugins/` - Model-specific and integration plugins
  - `mbridge/` - mbridge weight mapping adapters
  - `models/` - Custom Megatron model architectures such as GLM, GPT-OSS,
    Qwen, and Qwen-Next variants
  - `rollout_buffer/` - Rollout buffer implementations
- `scripts/` - Training launch scripts and model config shell scripts
- `tools/` - Weight conversion scripts: HF↔Megatron, FP8, INT4
- `examples/` - Custom rollout, reward, multi-agent, Search-R1, VLM, async training
- `tests/` - E2E GPU scripts, plugin contract tests, unit tests
- `docs/` - Sphinx documentation in English and Chinese
- `.codex/` - Codex-native agents, skills, workflow data, and rules

## WHY: Purpose

- Enable efficient RL training for large language models at scale.
- Combine asynchronous SGLang rollout with distributed Megatron training.
- Keep custom behavior extensible through `--custom-*-path` function loading.
- Preserve slime-specific expert guidance in Codex-native assets under `.codex/`.

## HOW: Core Commands

```bash
# Recommended: use Docker
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash

# Install slime in editable mode, inside Docker or a prepared environment
pip install -e . --no-deps

# Pre-commit hooks
apt install pre-commit -y
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always

# Plugin contract tests: CPU-only, no GPU required
python -m pytest tests/plugin_contracts/ -v

# Unit tests
python -m pytest tests/utils/ -v

# E2E GPU tests: require multi-GPU hardware
python tests/test_qwen2.5_0.5B_short.py

# Weight conversion: HuggingFace -> Megatron torch_dist
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
  --hf-checkpoint /path/to/hf --save /path/to/output

# Launch training, typical script path
bash scripts/run-qwen3-4B.sh
```

## Boundaries

### Constraints

- Designed for NVIDIA GPU clusters; Docker/container execution is the expected path.
- Multi-GPU is the normal deployment mode; single-GPU paths are not the primary test target.
- Megatron-LM and SGLang must already be present in the environment.
- Weights must be converted to Megatron `torch_dist` format before training.
- GPU/distributed tests require appropriate hardware; document skips when unavailable.
- CI workflows are generated from `.j2` templates; edit templates, not generated YAML.
- Codex hooks are not activated by this repository-level migration. Treat hook-like
  behavior as record-only guidance unless a separate implementation plan explicitly enables it.

### Always Do

- Read relevant files before modifying code.
- Prefer small, reviewable diffs and reuse existing patterns.
- Run `pre-commit run --all-files` before committing when the environment supports it.
- Follow conventions in the same module before introducing new abstractions.
- Use `load_function()` from `slime.utils.misc` for custom behavior injection.
- Add validation in `slime_validate_args()` when adding or changing CLI arguments.
- Run plugin contract tests when modifying extension interfaces.
- Use `logging.getLogger(__name__)` for logging; do not use `print` for runtime logs.
- Consult `.codex/rules/README.md` and the scoped rule files before changing matching areas.
- Verify claims with concrete command output before reporting completion.

### Ask or Escalate First

Ask the user, or clearly surface the risk before proceeding, when the next step is
irreversible, destructive, expensive, or materially changes project behavior:

- Modifying the central argument system in `slime/utils/arguments.py`.
- Changing loss computation in `slime/backends/megatron_utils/loss.py`.
- Modifying `RolloutManager` in `slime/ray/rollout.py`.
- Changing the weight sync pipeline under `slime/backends/megatron_utils/update_weight/`.
- Adding new dependencies.
- Deleting or renaming public APIs or CLI arguments.
- Running GPU/distributed tests. First check GPU availability:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`.
- Force-pushing, rebasing shared history, or creating/updating remote PRs.

### Never Do

- Hardcode secrets, credentials, user-specific paths, or private endpoints.
- Skip pre-commit or relevant tests when they are available and applicable.
- Use wildcard imports such as `from x import *`.
- Propose broad refactors when a focused fix is sufficient.
- Propose major Megatron rewrites without a dedicated plan.
- Add files larger than 1000KB; pre-commit rejects them.
- Add a bare `dist.barrier()` without the correct process group.
- Remove the zero-loss safeguard `loss + 0 * logits.sum()` in `loss.py`; it prevents NCCL deadlock.

## Progressive Disclosure: Detailed Guides

| Task | Reference |
| --- | --- |
| Quick Start | `docs/en/get_started/quick_start.md` |
| Parameter Reference | `docs/en/get_started/usage.md` |
| Customization | `docs/en/get_started/customization.md` |
| Add Custom Reward | `slime/rollout/rm_hub/deepscaler.py`, `.codex/skills/add-reward-function/SKILL.md` |
| Add Custom Rollout | `examples/multi_agent/`, `.codex/skills/add-rollout-function/SKILL.md` |
| Add Dynamic Filter | `.codex/skills/add-dynamic-filter/SKILL.md` |
| Add Eval Dataset Config | `.codex/skills/add-eval-dataset-config/SKILL.md` |
| Add Tests / CI | `.codex/skills/add-tests-and-ci/SKILL.md`, `.codex/rules/testing.md` |
| Add Model Support | `slime/backends/megatron_utils/megatron_to_hf/`, `scripts/models/` |
| Algorithm Details | `slime/backends/megatron_utils/loss.py`, `slime/utils/ppo_utils.py` |
| SGLang Config | `docs/en/advanced/sglang-config.md`, `.codex/rules/api-config.md` |
| Low-Precision Training | `docs/en/advanced/low-precision.md` |
| Debugging | `docs/en/developer_guide/debug.md` |
| CI System | `docs/en/developer_guide/ci.md` |
| Codex Asset Map | `.codex/README.md` |

## Git Workflow

- Keep commits focused and explain why the change was made.
- Prefer Conventional Commit-style subjects when no stronger repository instruction applies.
- In Codex/OMX sessions that require Lore trailers, follow the active Lore commit protocol.
- Squash WIP commits before opening a PR unless preserving review history is explicitly desired.
- Before PR creation, run available formatting/tests, document coverage, and note hardware limitations.
- CI labels include `run-ci-short`, `run-ci-sglang-config`, `run-ci-megatron`,
  `run-ci-precision`, `run-ci-ckpt`, `run-ci-plugin-contracts`, `run-ci-image`,
  and `run-ci-changed`. Plugin contracts run on all PRs without labels.
- Edit `.github/workflows/pr-test.yml.j2`, then regenerate with:
  `python .github/workflows/generate_github_workflows.py`.

## Codex Configuration

Codex-native workflow assets live under `.codex/`:

- `.codex/agents/` - project custom agents in TOML format.
- `.codex/skills/` - workflow skill folders containing `SKILL.md`.
- `.codex/rules/` - scoped rules for code style, API/config, distributed systems, and testing.
- `.codex/workflows/review-pr/` - supporting data for PR review workflows.
- `.codex/config.toml` - project-local skill and agent settings.
- `.codex/README.md` - human-facing usage map.

Do not treat legacy Claude hook/settings behavior as active Codex behavior. If automatic
reminders or permission changes are desired, create a separate Codex-native plan first.

### Custom Agents

Use project custom agents by naming them in the prompt or delegating matching work to
their role. Agent definitions are in `.codex/agents/*.toml` and use Codex fields such
as `name`, `description`, and `developer_instructions`.

| Agent | Purpose | Activation Trigger |
| --- | --- | --- |
| `planner` | Implementation planning | Before multi-file changes, new features, or architecture-sensitive decisions |
| `simple-code-reviewer` | Lightweight quality review | After focused code changes, before committing |
| `code-verifier` | Formatting, linting, tests, verification evidence | After code changes and before claiming completion |
| `megatron-expert` | Megatron backend: loss, CP, model, data, checkpoint | Megatron backend changes or questions |
| `rollout-expert` | RolloutManager, SGLang, async generation, rewards/eval | Rollout or SGLang changes/questions |
| `algorithm-expert` | RL algorithms: GRPO, PPO, GSPO, REINFORCE++, loss math | Algorithm, advantage, KL, clipping, or reward questions |
| `weight-sync-expert` | Megatron↔HF conversion and NCCL weight broadcast | Weight sync, converter, or mbridge changes |

**Stage-by-stage guidance**:

1. **Planning**: use `planner` for architecture and implementation sequencing.
2. **Domain work**: use `megatron-expert`, `rollout-expert`, `algorithm-expert`, or
   `weight-sync-expert` for specialized code paths.
3. **Verification**: use `code-verifier` for formatting, linting, test planning, and evidence.
4. **Quality pass**: use `simple-code-reviewer` for focused post-change review.

### Skills

Skills provide guided Codex workflows. Prefer `$skill-name` invocation when available,
or ask Codex in natural language to use the named skill.

| Skill | Use for |
| --- | --- |
| `$add-reward-function` | Add a custom reward function or reward model integration |
| `$add-rollout-function` | Add custom rollout generation behavior |
| `$add-eval-dataset-config` | Add or modify evaluation dataset configuration |
| `$add-dynamic-filter` | Add dynamic sampling filters |
| `$add-tests-and-ci` | Add tests and CI wiring |
| `$create-pr` | Rebase/squash/preview/create PR with explicit safety gates |
| `$gen-commit-msg` | Generate commit messages from staged changes |
| `$review-pr` | Review PRs using risk-based workflow data |

### Rules

Project standards are stored in `.codex/rules/`:

- `code-style.md` - logging, naming, performance, imports, typing, formatting.
- `api-config.md` - argument system, dataclasses, validation, SGLang/eval config.
- `distributed.md` - Ray actors, NCCL groups, Megatron parallelism, offload/onload.
- `testing.md` - E2E scripts, pytest conventions, plugin contracts, CI labels.

Consult `rules/README.md` for path scopes and recommended agent/skill consumers.

### Workflow Data

The PR review workflow uses:

- `.codex/workflows/review-pr/change-types.md`
- `.codex/workflows/review-pr/templates.md`

Review workflows are read-oriented and must not post comments or mutate GitHub state
unless the user explicitly asks.

## Code Intelligence & Navigation

When navigating code, prefer semantic tools over raw text search where possible.

1. **Prefer LSP/code-intelligence tools for code relationships**:
   - Go to definitions for symbols and classes.
   - Find references for call sites and usage.
   - Search workspace symbols for project-wide navigation.
   - Inspect diagnostics for type/build issues when supported.

2. **Use grep/find/read for**:
   - Comments, strings, docs, shell scripts, YAML/JSON/TOML, and generated configs.
   - Fuzzy exploration when the symbol name is unknown.
   - File discovery by path/name patterns.

3. **Workflow**:
   - First: understand the touched code paths and existing patterns.
   - Second: inspect scoped rules and relevant tests.
   - Third: edit narrowly, then run targeted verification.
   - Do not read huge files just to find references; use semantic or targeted search.

## Verification Before Completion

Before claiming completion, confirm the following as applicable:

- working tree status is understood;
- modified files are limited to the requested scope;
- formatting/linting checks have run or are explicitly skipped with reason;
- targeted tests have run or are explicitly skipped with reason;
- generated Codex assets, if changed, pass TOML/path/coverage/reference checks;
- no active hook or permission behavior was introduced accidentally;
- final response includes changed files, verification evidence, and remaining risks.
