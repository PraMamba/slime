---
name: create-pr
description: Rebase from the latest `origin/main`, squash the commits from it, and then create a PR on github with intelligent commit messages based on staged changes. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub with an
intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

## Workflow

### Step 1: Verify Prerequisites

```bash
git branch --show-current
git status --short
gh --version
```

- Confirm not on main/master
- Confirm no uncommitted changes
- Confirm `gh` CLI available

### Step 2: Check for Existing PR

```bash
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

### Step 3: Fetch and Rebase

```bash
git fetch origin main
git rebase origin/main
```

On conflict: abort rebase, inform user.

### Step 4: Squash Commits

```bash
git rev-list --count origin/main..HEAD
git reset --soft origin/main
```

### Step 5: Analyze Changes and Generate Commit Message

```bash
git diff --cached --name-only
git diff --cached
```

**Categorize** (`feat`/`fix`/`docs`/`refactor`/`test`/`chore`/`perf`).

**Determine scope** from changed files:

- `slime/backends/` → `backend`
- `slime/rollout/` → `rollout`
- `slime/ray/` → `ray`
- `slime/utils/` → `utils`
- `slime_plugins/` → `plugins`
- `scripts/` → `scripts`
- `tests/` → `tests`
- `docs/` → `docs`
- `examples/` → `examples`
- Multiple areas → omit scope or use broader term

**Format**: `<type>(<scope>): <subject>`

### Step 6: Generate PR Title and Description

**PR Title**: Under 70 chars, imperative mood.

**PR Description** following CONTRIBUTING.md conventions:

```markdown
## Description

[2-4 sentences explaining what and why]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test coverage improvement

## Changes

- `file1.py`: description
- `file2.py`: description

## Testing

- [ ] Pre-commit passes
- [ ] Plugin contract tests pass
- [ ] E2E tests pass (if applicable)
```

### Step 7: Push and Create PR

Show preview, confirm with user, then:

```bash
git push -f -u origin $(git branch --show-current)

gh pr create \
  --base main \
  --title "<title>" \
  --body "$(cat <<'EOF'
<description>
EOF
)"
```

## Safety Checks

- Confirm no uncommitted changes
- Confirm not on main/master
- Backup branch before squash
- Warn about force push
- Show full preview before creating

---

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

## How to Update

### Adding New Scopes
Update "Determine scope" section with new file path mappings.

### Changing PR Template
Update "PR Description" format section.

================================================================================
-->
