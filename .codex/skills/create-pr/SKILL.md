---
name: create-pr
description: Rebase from the latest base branch, squash branch commits, and create a GitHub pull request with a reviewed title, commit message, and description.
---

# Create Pull Request

Source: `.claude/commands/create-pr.md`

Use this workflow when the user asks to prepare a pull request for the current branch.
Primary invocation path in Codex/OMX: mention `$create-pr` or ask Codex to use the `create-pr` skill.

## Scope

This skill walks through a guarded PR-preparation flow:
- verify the branch and working tree are safe,
- check whether a PR already exists,
- rebase onto the chosen base branch,
- squash branch commits,
- synthesize a commit message / PR title / PR body from the diff,
- show a full preview,
- only then push and open the PR.

Relevant rules to consult when available:
- `.codex/rules/testing.md`
- `.codex/rules/code-style.md`
- `.codex/rules/README.md`

## Usage

```text
$create-pr [--draft] [--base <branch>]
```

Default base branch: `main` unless the user or repo conventions say otherwise.

## Workflow

### Step 1: Verify prerequisites

Run:

```bash
git branch --show-current
git status --short
gh --version
```

Confirm all of the following before continuing:
- current branch is not `main` or `master`
- working tree is clean
- GitHub CLI is installed and authenticated

If any check fails, stop and explain the blocker.

### Step 2: Check for an existing PR

```bash
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

If a PR already exists, show it and ask whether the user wants to update that PR instead of creating a new one.

### Step 3: Fetch and rebase

Default command sequence:

```bash
git fetch origin <base>
git rebase origin/<base>
```

On conflict:
- stop immediately,
- explain the conflicting files,
- recommend either resolving conflicts or aborting the rebase,
- do not continue to squash/push until the rebase is clean.

### Step 4: Count and squash branch commits

```bash
git rev-list --count origin/<base>..HEAD
git reset --soft origin/<base>
```

Before running `git reset --soft`, create a backup branch name and show it to the user.
Example: `backup/<current-branch>-before-squash-<date>`.

### Step 5: Analyze the staged diff

```bash
git diff --cached --name-only
git diff --cached
```

Classify the change as one of:
- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`
- `perf`

Infer scope from changed files when it is genuinely helpful:
- `slime/backends/` → `backend`
- `slime/rollout/` → `rollout`
- `slime/ray/` → `ray`
- `slime/utils/` → `utils`
- `slime_plugins/` → `plugins`
- `scripts/` → `scripts`
- `tests/` → `tests`
- `docs/` → `docs`
- `examples/` → `examples`
- multiple areas → omit scope or use a broader scope

Commit-message guidance:
- if the repo expects conventional commits, use `<type>(<scope>): <subject>`
- if repo instructions override that format (for example Lore commit rules in `AGENTS.md`), preserve the same change analysis but emit the repo-required format instead

### Step 6: Draft the PR title and description

PR title guidance:
- imperative mood
- under ~70 characters
- aligned with the staged diff

PR description template:

```markdown
## Description

[2-4 sentences explaining what changed and why]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test coverage improvement

## Changes

- `path/to/file`: summary
- `path/to/file`: summary

## Testing

- [ ] Pre-commit passes
- [ ] Plugin contract tests pass
- [ ] E2E tests pass (if applicable)
```

Follow any repository-specific PR template or contribution rules if they exist.

### Step 7: Preview, confirm, push, and create the PR

Show the user a full preview of:
- the squashed commit message,
- PR title,
- PR body,
- push target,
- base branch.

Only after explicit confirmation, run:

```bash
git push -f -u origin $(git branch --show-current)

gh pr create \
  --base <base> \
  --title "<title>" \
  --body "$(cat <<'PR_BODY'
<description>
PR_BODY
)"
```

Add `--draft` when requested.

## Safety gates

Do not skip these:
- confirm the tree is clean
- confirm the branch is not `main`/`master`
- create/show a backup branch name before squash
- warn clearly before any force push
- show the full generated preview before creating the PR
- stop if `gh pr view` already reports an open PR and the user has not chosen to update it

## Maintenance notes

When updating this skill, keep these source behaviors intact:
- scope inference by path
- explicit rebase/squash/push safety checks
- PR description structure
