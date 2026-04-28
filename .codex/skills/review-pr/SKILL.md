---
name: review-pr
description: Review a pull request with risk-based task generation, using Codex-native workflow guidance and local review data.
---

# Review Pull Request

Source: `.claude/commands/review-pr.md`

Use this workflow when the user asks for a PR review of the current branch or a specific GitHub PR.
Primary invocation path in Codex/OMX: mention `$review-pr` or ask Codex to use the `review-pr` skill.

Local workflow data used by this skill:
- `.codex/workflows/review-pr/change-types.md`
- `.codex/workflows/review-pr/templates.md`

Relevant rules to consult when available:
- `.codex/rules/testing.md`
- `.codex/rules/code-style.md`
- `.codex/rules/distributed.md`
- `.codex/rules/README.md`

## Arguments

```text
$review-pr [<pr-number>] [--quick]
```

- no arguments: review the PR associated with the current branch
- `<pr-number>`: review that specific PR
- `--quick`: stop after Phase 1 analysis and return the analysis report only

## Hard constraints

- use `gh` for GitHub interaction rather than generic web scraping
- do not post comments to the PR automatically
- do not use build status or run build/typecheck as part of this review workflow unless the user separately asks for that
- every reported issue must include file path and line number when available

## Workflow overview

```text
Phase 1: Deep PR analysis
Phase 2: Dynamic review-task planning
Phase 3: Execute review tasks
Phase 4: Confidence scoring and final summary
```

## Review depth routing

Translate source model tiers into Codex-native review depth:

| Risk | Recommended review path |
| --- | --- |
| CRITICAL / HIGH | highest-judgment review lane available (for example `code-reviewer`, `security-reviewer`, `performance-reviewer`, or equivalent high-effort reviewer) |
| MEDIUM | standard review lane (for example `quality-reviewer`, `critic`, or equivalent) |
| LOW | lightweight review lane or direct review |
| `--quick` | keep all phases lightweight and stop after analysis |

If the current runtime explicitly allows parallel review agents, independent review tasks may run in parallel. Otherwise, execute them sequentially.

---

## Phase 1: Deep PR analysis

### 1.0 PR status check

Run the appropriate `gh pr view` query.
Stop early if:
- the PR does not exist
- the PR is closed

If the PR is a draft, note it but continue.
If the PR appears bot-generated and the user did not ask for bot-review validation, note that and consider stopping.

### 1.1 Get PR summary

Collect:
- title
- description
- base/head branches
- modified files
- high-level diff summary

### 1.2 Detect change types

Analyze changed files against `.codex/workflows/review-pr/change-types.md`.
Identify:
- detected change types
- highest risk level
- affected files
- linked risks

### 1.3 Identify framework-specific risks

Use the same review data file to expand risk areas, especially for:
- Megatron backend changes
- weight sync / converter changes
- rollout / async changes
- argument / compatibility changes

### 1.4 Emit the analysis report

```text
CHANGE_ANALYSIS_REPORT:
- detected_types: [MEGATRON_LOSS, ASYNC_ROLLOUT, ...]
- risk_level: CRITICAL | HIGH | MEDIUM | LOW
- affected_files: [file1.py, file2.py, ...]
- identified_risks: [risk1, risk2, ...]
```

If `--quick` is active, return this report and stop.

---

## Phase 2: Dynamic review-task planning

### Planning principles

1. Generate tasks by risk area.
2. Merge strongly related changes into one task when it improves coherence.
3. Give CRITICAL/HIGH areas the deepest review.
4. Ensure at least one review task even for simple changes.

### Template selection

Use `.codex/workflows/review-pr/templates.md` to select task checklists.
Create a task list with:
- task name
- reason for inclusion
- checklist
- focus files
- recommended review depth

### Planning output format

```text
GENERATED_REVIEW_TASKS:
1. [deepest-review] Task Name
   - Reason: XXX change type detected
   - Checklist: [...]
   - Focus files: [...]
```

---

## Phase 3: Execute review tasks

### Execution rules

- execute each task independently
- keep findings traceable to concrete files / lines
- separate confirmed issues from suspicions
- avoid duplicate findings across tasks

### Reviewer output format

```text
REVIEW_RESULT:
task_name: "Task Name"
review_depth: deepest-review | standard-review | lightweight-review
findings:
  - issue: "Issue description"
    severity: CRITICAL | HIGH | MEDIUM | LOW
    file: "path/to/file.py"
    line: 123
    reason: "Why this is an issue"
    suggestion: "Fix suggestion"
```

If no issue is found for a task, say so explicitly.

---

## Phase 4: Confidence scoring and summary

### Confidence scoring (0-100)

| Score | Meaning |
| --- | --- |
| **0** | False positive or clearly pre-existing issue |
| **25** | Plausible issue, not verified |
| **50** | Real but minor or rare |
| **75** | Very likely real and important |
| **100** | Confirmed real and likely to recur |

### Final summary format

```markdown
# PR Review Summary

## PR Overview
- **Title**: PR title
- **Detected Change Types**: [...]
- **Risk Level**: CRITICAL | HIGH | MEDIUM | LOW
- **Generated Review Tasks**: N

## Executed Review Tasks
| # | Review Depth | Task Name | Reason |
|---|--------------|-----------|--------|

## Findings

### CRITICAL Severity (Confidence >= 75)
#### Issue 1: [Title]
- **File**: `path/to/file.py:123`
- **Confidence**: 85
- **Description**: ...
- **Fix Suggestion**: ...

### HIGH Severity (Confidence >= 50)
...

## Review Statistics
- Total issues: X (CRITICAL: X, HIGH: X, MEDIUM: X, LOW: X)
- Filtered false positives: X
```

The Phase 4 summary is the final deliverable.
Return the complete report rather than a partial outline.
