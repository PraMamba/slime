---
name: gen-commit-msg
description: Generate a commit message from staged changes, with repo-specific formatting overrides when the workspace requires them.
---

# Generate Commit Message

Source: `.claude/commands/gen-commit-msg.md`

Use this workflow when the user asks for a commit message for currently staged changes.
Primary invocation path in Codex/OMX: mention `$gen-commit-msg` or ask Codex to use the `gen-commit-msg` skill.

Relevant rules to consult when available:
- `.codex/rules/code-style.md`
- `.codex/rules/testing.md`
- `.codex/rules/README.md`

## Usage

```text
$gen-commit-msg [--amend] [--scope <scope>]
```

## Workflow

### Step 1: Inspect staged changes

```bash
git diff --cached --name-only
git diff --cached
git log --oneline -5
```

Use the staged diff as the primary source of truth.
Recent commit history is only for style/context calibration.

### Step 2: Categorize the change

Choose the best-fitting type:

| Type | When to use |
| --- | --- |
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without feature/fix |
| `test` | Adding or fixing tests |
| `chore` | Build, dependency, or config changes |
| `perf` | Performance improvement |

### Step 3: Determine scope

Infer scope from changed files when it adds clarity:
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
- multiple areas → omit scope or use a broader scope

A user-provided `--scope` overrides inferred scope.

### Step 4: Draft the commit message

Default source-format template:

```text
<type>(<scope>): <subject>

<body>

[Optional:]
Key changes:
- change 1
- change 2

Refs: #123, #456
```

Rules from the source workflow:
- subject in imperative mood
- subject roughly 50-72 characters
- no trailing period in the subject
- body explains why, not just what
- wrap body text near 72 columns
- use a short `Key changes:` list for complex diffs

### Step 5: Reconcile with repo commit conventions

If the current workspace has an explicit commit-message policy, follow it.
Example: if `AGENTS.md` requires Lore commit messages, keep the same change analysis but emit the repository-required structure instead of forcing the conventional-commit template.

Practical rule:
- no repo override found → use the default source-format template above
- repo override found → adapt the draft into that required format while preserving the underlying categorization, scope reasoning, and rationale

### Step 6: Preview and optionally commit

Show the generated message before execution.
Only run a commit command after the user has explicitly asked to create the commit.

```bash
git commit -m "$(cat <<'COMMIT_MSG'
<message>
COMMIT_MSG
)"
```

If `--amend` is requested, use the repository-safe amend workflow instead of a fresh commit.

## Examples

### Single-file fix

```text
fix(rollout): handle empty completion in deepscaler reward

Return 0 reward instead of raising an exception when the
completion string is empty after think-tag extraction.
```

### Multi-file feature

```text
feat(megatron): add OPD on-policy distillation support

Enable on-policy distillation with a teacher-model KL penalty
during RL training.

Key changes:
- add OPD loss computation in policy_loss_function
- add teacher log-prob rollout in on_policy_distillation.py
- add --opd-coef argument
```

### Documentation change

```text
docs: update algorithm comparison table

Add GSPO and REINFORCE++ to the algorithm-family
documentation with configuration examples.
```
