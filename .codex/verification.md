# Verification runbook

## Lane-local checks

Run these checks in this branch to verify the support-file lane itself:

```bash
python - <<'PY'
from pathlib import Path
for path in [
    Path('.codex/config.toml'),
    Path('.codex/README.md'),
    Path('.codex/coverage-matrix.md'),
    Path('.codex/migration-notes.md'),
    Path('.codex/verification.md'),
    Path('.codex/rules/README.md'),
    Path('.codex/rules/api-config.md'),
    Path('.codex/rules/code-style.md'),
    Path('.codex/rules/distributed.md'),
    Path('.codex/rules/testing.md'),
]:
    assert path.is_file(), path
print('support files present')
PY
```

```bash
python - <<'PY'
from pathlib import Path
import tomllib
cfg = tomllib.loads(Path('.codex/config.toml').read_text())
assert cfg['agents']['max_threads'] == 6
assert cfg['agents']['max_depth'] == 1
paths = [item['path'] for item in cfg['skills']['config']]
assert len(paths) == 8
assert len(set(paths)) == len(paths)
print('config structure ok')
PY
```

```bash
python - <<'PY'
from pathlib import Path
sources = sorted(str(p) for p in Path('.claude').glob('**/*') if p.is_file())
matrix = Path('.codex/coverage-matrix.md').read_text()
missing = [s for s in sources if s not in matrix]
assert not missing, missing
print('coverage matrix lists all source files')
PY
```

## Full integration checks

Run these after merging the agent and skill lanes:

```bash
python - <<'PY'
from pathlib import Path
import tomllib
for path in [Path('.codex/config.toml'), *Path('.codex/agents').glob('*.toml')]:
    with path.open('rb') as fh:
        data = tomllib.load(fh)
    if path.parent.name == 'agents':
        for key in ('name', 'description', 'developer_instructions'):
            assert data.get(key), f'{path}: missing {key}'
print('toml ok')
PY
```

```bash
python - <<'PY'
from pathlib import Path
import tomllib
cfg_path = Path('.codex/config.toml')
cfg = tomllib.loads(cfg_path.read_text())
for item in cfg.get('skills', {}).get('config', []):
    skill_dir = cfg_path.parent / item['path']
    assert skill_dir.is_dir(), skill_dir
    assert (skill_dir / 'SKILL.md').is_file(), skill_dir / 'SKILL.md'
print('skills.config paths ok')
PY
```

```bash
python - <<'PY'
from pathlib import Path
violations = []
for path in Path('.codex').glob('**/*'):
    if not path.is_file() or path.suffix not in {'.md', '.toml'}:
        continue
    text = path.read_text(errors='ignore')
    if path.name not in {'coverage-matrix.md', 'migration-notes.md'} and '@.claude/' in text:
        violations.append(str(path))
assert not violations, violations
print('no active legacy imports')
PY
```
