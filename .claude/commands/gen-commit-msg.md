---
name: gen-commit-msg
description: Generate intelligent commit messages based on staged changes. Invoke with /gen-commit-msg.
---

# Generate Commit Message

Generate a well-formatted commit message based on staged changes.

## Usage

```
/gen-commit-msg [--amend] [--scope <scope>]
```

## Workflow

### Step 1: Analyze Changes

```bash
git diff --cached --name-only
git diff --cached
git log --oneline -5
```

### Step 2: Categorize Changes

| Type | When to Use |
| --- | --- |
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without feature/fix |
| `test` | Adding or fixing tests |
| `chore` | Build, deps, config changes |
| `perf` | Performance improvement |

### Step 3: Determine Scope

Infer from changed files:

- `slime/backends/megatron_utils/` → `megatron`
- `slime/backends/sglang_utils/` → `sglang`
- `slime/rollout/` → `rollout`
- `slime/ray/` → `ray`
- `slime/utils/` → `utils`
- `slime_plugins/` → `plugins`
- `scripts/` → `scripts`
- `tests/` → `tests`
- `docs/` → `docs`
- `examples/` → `examples`
- Multiple areas → omit scope or use broader term

### Step 4: Generate Message

**Format:**

```
<type>(<scope>): <subject>

<body>

[Optional:]
Key changes:
- change 1
- change 2

Refs: #123, #456
```

**Rules:**

- Subject: imperative mood, ~50-72 chars, no period
- Body: explain "why" not "what", wrap at 72 chars
- Key changes: bullet list for complex commits

### Step 5: Confirm and Commit

Show preview, confirm, then:

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

## Examples

**Single file fix:**

```
fix(rollout): handle empty completion in deepscaler reward

Return 0 reward instead of raising exception when
completion string is empty after think tag extraction.
```

**Multi-file feature:**

```
feat(megatron): add OPD on-policy distillation support

Enable on-policy distillation with teacher model KL penalty
during RL training.

Key changes:
- Add OPD loss computation in policy_loss_function
- Add teacher log-prob rollout in on_policy_distillation.py
- Add --opd-coef argument
```

**Docs only:**

```
docs: update algorithm comparison table

Add GSPO and REINFORCE++ to the algorithm family documentation
with configuration examples.
```
