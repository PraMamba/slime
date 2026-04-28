---
name: add-eval-dataset-config
description: Guide for adding and validating evaluation dataset configuration in slime. Use when you need to configure eval datasets via --eval-config or --eval-prompt-data, add per-dataset overrides, or customize evaluation rollout behavior.
---

# Add Eval Dataset Config

Source: `.claude/skills/add-eval-dataset-config/SKILL.md`

Relevant rules to consult when available:
- `.codex/rules/api-config.md`
- `.codex/rules/testing.md`

Use this skill in Codex when configuring evaluation datasets in slime with explicit dataset-level overrides and predictable runtime behavior.

## When to Use

Use this skill when:

- You need to add evaluation datasets for periodic eval
- You need to migrate from `--eval-prompt-data` to structured `--eval-config`
- You need per-dataset eval overrides such as sampling params, keys, `rm_type`, or metadata

## Step-by-Step Guide

### Step 1: Choose Config Entry Method

Supported inputs:

- Structured config file: `--eval-config <yaml>`
- Legacy CLI pairs: `--eval-prompt-data <name1> <path1> <name2> <path2> ...`

If `--eval-interval` is set, eval datasets must be configured.

### Step 2: Build YAML with Required Fields

Each dataset needs at least:

- `name`
- `path`

Example:

```yaml
eval:
  defaults:
    n_samples_per_eval_prompt: 1
    temperature: 0.7
    top_p: 1.0
  datasets:
    - name: aime
      path: /path/to/aime.jsonl
      rm_type: math
      input_key: prompt
      label_key: answer
      metadata_overrides:
        split: test
```

### Step 3: Understand Override Priority

`slime/utils/eval_config.py` resolves fields in this order:

1. Dataset-level values in `eval.datasets[*]`
2. `eval.defaults`
3. CLI arg fallback such as `eval_*` or `rollout_*`

Common overridable fields include:

- Runtime: `n_samples_per_eval_prompt`, `temperature`, `top_p`, `top_k`, `max_response_len`
- Sample keys: `input_key`, `label_key`, `tool_key`, `metadata_key`
- Extra: `rm_type`, `custom_generate_function_path`, `metadata_overrides`

### Step 4: Wire Eval Function if Needed

By default, eval uses `--eval-function-path` (which defaults to the rollout function path).
Use a separate eval function when inference or eval behavior must differ from training rollout.

### Step 5: Validate Parsing and Runtime

- Start with config parsing sanity by running a short launch command.
- Confirm dataset entries are loaded into `args.eval_datasets`.
- Verify output keys match eval logging and metrics expectations.

## Common Mistakes

- Missing `name` in dataset entries
- Odd-length `--eval-prompt-data` pairs
- Setting `--eval-interval` without any eval dataset
- Mixing reward dict outputs without `eval_reward_key` configuration

## Reference Locations

- Eval config model: `slime/utils/eval_config.py`
- Eval config resolution: `slime/utils/arguments.py`
- Eval rollout path: `slime/rollout/sglang_rollout.py`
- Customization docs: `docs/en/get_started/customization.md`
