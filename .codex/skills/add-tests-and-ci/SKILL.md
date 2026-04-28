---
name: add-tests-and-ci
description: Guide for adding or updating slime tests and CI wiring. Use when tasks require new test cases, CI registration, test matrix updates, or workflow template changes.
---

# Add Tests and CI

Source: `.claude/skills/add-tests-and-ci/SKILL.md`

Relevant rules to consult when available:
- `.codex/rules/testing.md`
- `.codex/rules/code-style.md`

Use this skill in Codex when adding reliable tests and integrating them with the slime CI flow.

## When to Use

Use this skill when:

- You need to add tests for new behavior
- You need to fix or update existing tests in `tests/`
- You need to update CI workflow behavior
- You need to explain or run targeted checks before a PR

## Step-by-Step Guide

### Step 1: Pick the Right Test Pattern

- Follow existing naming: `tests/test_<feature>.py`
- Start from the nearest existing test file for your model or path
- Keep test scope small and behavior focused

### Step 2: Keep CI Compatibility

When creating CI-discoverable tests, ensure top-level constants and conventions match repository patterns, including `NUM_GPUS = <N>` where expected.

### Step 3: Run Local Validation

- Run the exact existing test files you changed, if any.
- Run repository-wide checks only when they are already part of the task or workflow.
- Avoid documenting placeholder test commands that do not exist in the current tree.

### Step 4: Update Workflow Template Correctly

For CI workflow changes:

1. Edit `.github/workflows/pr-test.yml.j2`
2. Regenerate workflows:

```bash
python .github/workflows/generate_github_workflows.py
```

3. Commit both the template and generated workflow files

### Step 5: Provide Verifiable PR Notes

Include:

- Which tests were added or changed
- Exact commands executed
- GPU assumptions for each test path
- Why this coverage protects against regression

## Common Mistakes

- Editing only the generated workflow file
- Adding tests without following existing constants or conventions
- Making tests too large or non-deterministic
- Skipping local validation and relying only on remote CI

## Reference Locations

- Pytest config: `pyproject.toml`
- Tests: `tests/`
- CI template: `.github/workflows/pr-test.yml.j2`
- CI guide: `docs/en/developer_guide/ci.md`
