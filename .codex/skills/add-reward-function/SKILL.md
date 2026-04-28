---
name: add-reward-function
description: Guide for adding a custom reward function in slime and wiring it through --custom-rm-path (and optional reward post-processing). Use when you need new reward logic, remote/service reward integration, or task-specific reward shaping.
---

# Add Reward Function

Source: `.claude/skills/add-reward-function/SKILL.md`

Relevant rules to consult when available:
- `.codex/rules/distributed.md`
- `.codex/rules/testing.md`

Use this skill in Codex when implementing custom reward logic and connecting it to slime rollout and training safely.

## When to Use

Use this skill when:

- You need to add new reward computation logic
- You need to integrate an external reward service
- You need to customize reward normalization or post-processing

## Step-by-Step Guide

### Step 1: Choose Reward Mode

Pick one of these:

- Single-sample mode (`--group-rm` disabled): custom function gets one `Sample`
- Group or batch mode (`--group-rm` enabled): custom function gets `list[Sample]`

`slime.rollout.rm_hub.__init__.py` calls your function via `--custom-rm-path`.

### Step 2: Create Reward Module

Create `slime/rollout/rm_hub/<your_rm>.py`.

Supported signatures:

```python
async def custom_rm(args, sample):
    return float_reward_or_reward_dict
```

```python
async def custom_rm(args, samples):
    return list_of_rewards
```

If using group mode, return one reward per sample in input order.

### Step 3: Keep Reward Type Consistent

- Return scalar numeric rewards unless your pipeline explicitly uses keyed rewards.
- If using reward dicts, ensure downstream `reward_key` or `eval_reward_key` is configured.
- Keep exceptions explicit for invalid metadata instead of silently returning zeros.

### Step 4: Optional Reward Post-Processing

To customize normalization or shaping before advantage computation, add:

```python
def post_process_rewards(args, samples):
    # return (raw_rewards, processed_rewards)
    ...
```

Wire with:

```bash
--custom-reward-post-process-path <module>.post_process_rewards
```

This hook is consumed in `slime/ray/rollout.py`.

### Step 5: Wire and Validate

Use:

```bash
--custom-rm-path slime.rollout.rm_hub.<your_rm>.custom_rm
```

## Common Mistakes

- Returning the wrong output shape in group mode
- Mixing scalar rewards and reward dicts without `reward_key` config
- Doing blocking network calls without async handling
- Forgetting to validate reward behavior on truncated or failed samples

## Reference Locations

- Reward dispatch: `slime/rollout/rm_hub/__init__.py`
- Reward post-process hook: `slime/ray/rollout.py`
- Customization docs: `docs/en/get_started/customization.md`
