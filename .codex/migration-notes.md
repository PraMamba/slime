# Migration notes

## Decisions captured in this lane

| Area | Decision | Reason |
| --- | --- | --- |
| Custom agents | Move expert personas into `.codex/agents/*.toml` with Codex-native required fields | Codex discovers custom agents from TOML, not Markdown frontmatter |
| Skills | Keep each skill as a dedicated folder with `SKILL.md` | Matches project-local skill loading via `skills.config[*].path` |
| Command workflows | Rewrite PR and commit helpers as skills instead of slash-command-only docs | Keeps workflow invocation discoverable in Codex |
| Rules | Preserve project rules as Markdown under `.codex/rules/` with a rule index | Keeps domain guidance available without activating hooks |
| Hooks | Do not activate hook scripts in project config | The requirement is record-only preservation, not live automation |
| Local settings | Do not create active permission/config overrides | Source-local permissions are environment-specific and out of scope for the migration |

## Record-only omissions

### Hook mapping retained as documentation only

The former hook script mapped code paths to expert reminders. The knowledge is
preserved here but intentionally not activated as a Codex hook.

| Path pattern family | Expert to consult |
| --- | --- |
| `slime/backends/megatron_utils/loss*`, `actor*`, `model*`, `cp_utils*`, `data*`, `checkpoint*`, `initialize*` | `megatron-expert` |
| `slime/backends/megatron_utils/update_weight/`, `slime/backends/megatron_utils/megatron_to_hf/`, `slime_plugins/mbridge/` | `weight-sync-expert` |
| `slime/ray/rollout*`, `slime/rollout/`, `slime/backends/sglang_utils/` | `rollout-expert` |
| `slime/utils/ppo_utils*`, `slime/backends/megatron_utils/loss*`, `slime/rollout/rm_hub/` | `algorithm-expert` |

### Missing local settings file

The consensus plan mentioned a source-local settings file, but the current tree
for this migration contains no `.claude/settings.local.json`. No active Codex
settings were created as a substitute. If such a file is reintroduced later,
handle it as a documentation-only migration note unless a separate plan approves
active Codex configuration changes.

## Codex compatibility notes

- Project-scoped configuration lives in `.codex/config.toml`.
- Skill entries are declared via `[[skills.config]]` and point to folders that
  contain `SKILL.md`.
- Active assets should not depend on `@.claude/...` imports or legacy
  slash-command-only instructions.
- The rules remain documentation assets; this lane does not enable
  `features.codex_hooks` or inline hook configuration.

## Validation expectations after merge

Once the agent and skill lanes are merged, the integrated `.codex` tree should
satisfy all of the following:

1. every custom agent TOML parses and includes `name`, `description`, and
   `developer_instructions`;
2. every `skills.config[*].path` resolves to a folder containing `SKILL.md`;
3. no active asset contains `@.claude/` imports;
4. README guidance stays Codex-native and names the intended trigger paths.

See `verification.md` for exact commands.
