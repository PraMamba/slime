---
name: add-dynamic-filter
description: Guide for adding dynamic/filter hooks in slime rollout pipeline. Use when you need sample-group selection during rollout, buffer filtering before training, or per-sample masking/processing hooks.
---

# Add Dynamic Filter

Source: `.claude/skills/add-dynamic-filter/SKILL.md`

Relevant rules to consult when available:
- `.codex/rules/distributed.md`
- `.codex/rules/testing.md`

Use this skill in Codex when implementing or reviewing filtering hooks in rollout and buffer stages while preserving sample-group contracts.

## When to Use

Use this skill when:

- You need to filter sample groups during dynamic sampling
- You need to customize buffer extraction strategy
- You need to mask/remove some rollout samples before training
- You need to process all generated samples for logging or analysis

## Step-by-Step Guide

### Step 1: Pick the Correct Hook

- Dynamic sampling filter: `--dynamic-sampling-filter-path`
- Buffer filter: `--buffer-filter-path`
- Per-sample rollout filter: `--rollout-sample-filter-path`
- All-samples post process: `--rollout-all-samples-process-path`

### Step 2: Implement the Function Signature

Dynamic sampling filter (called in `slime/rollout/sglang_rollout.py`):

```python
def filter_function(args, samples, **kwargs):
    # return DynamicFilterOutput or bool
```

Preferred return type:

```python
from slime.rollout.filter_hub.base_types import DynamicFilterOutput

return DynamicFilterOutput(keep=True, reason=None)
```

Buffer filter (called in `slime/rollout/data_source.py`):

```python
def buffer_filter(args, rollout_id, buffer, num_samples):
    return selected_groups
```

Rollout sample filter:

```python
def rollout_sample_filter(args, samples):
    # modify sample.remove_sample in-place where needed
```

All-samples process:

```python
def process_all_samples(args, all_samples, data_source):
    ...
```

### Step 3: Preserve Group Structure

- Keep `list[list[Sample]]` structure intact where required.
- Do not flatten sample groups unless the downstream path expects flattened samples.
- For sample removal, prefer `sample.remove_sample = True` over deleting objects.

### Step 4: Wire and Validate

Example wiring:

```bash
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--buffer-filter-path <module>.buffer_filter
--rollout-sample-filter-path <module>.rollout_sample_filter
--rollout-all-samples-process-path <module>.process_all_samples
```

### Step 5: Measure Side Effects

- Check that final sample count remains aligned with `rollout_batch_size` expectations.
- Verify drop reasons are surfaced in rollout metrics when a dynamic filter is used.
- Validate that training still receives valid loss masks/rewards after filtering.

## Common Mistakes

- Returning the wrong container type for buffer filters
- Dropping samples by deletion instead of mask flagging
- Losing sample-group alignment in group-RM setups
- Adding expensive logic in hot filtering paths

## Reference Locations

- Dynamic filter types: `slime/rollout/filter_hub/base_types.py`
- Dynamic filter example: `slime/rollout/filter_hub/dynamic_sampling_filters.py`
- Rollout generation hook points: `slime/rollout/sglang_rollout.py`
- Buffer filter hook point: `slime/rollout/data_source.py`
