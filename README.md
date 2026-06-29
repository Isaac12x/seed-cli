# seed-cli

![seed-cli](https://github.com/user-attachments/assets/5661d43b-816f-40d3-b47e-23f85a0eae34)

**seed** is a Terraform-inspired, spec-driven filesystem orchestration tool.

It captures directory trees, plans changes, applies them safely, syncs drift,
scaffolds projects from reusable templates, and can also execute
manifest-driven repository or service maintenance.

Think **Terraform for directory trees**, plus **template scaffolding** and
**workspace maintenance**.

[![ci-cd](https://github.com/Isaac12x/seed-cli/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Isaac12x/seed-cli/actions/workflows/ci-cd.yml)

## Highlights

- Multiple spec inputs: `.tree`, `.seed`, YAML, JSON, DOT, image OCR, and stdin
- Deterministic planning with exportable plans: `seed plan spec.seed --out plan.json`
- Safe execution of immutable plans: `seed apply plan.json`
- Drift workflows: `diff`, `sync`, `match`, snapshots, and spec history
- Template variables in paths and content: `<varname>/` and `{{var}}`
- Project-local template registration under `.seed/templates/`
- Reusable template registry with versions, locking, content sources, and built-in templates
- Copier-style template config support: questions, defaults, answers files, excludes, skip-if-exists, and gated tasks
- Manifest-driven repository/service/system maintenance with `seed maintain`
- GBrain integration: `seed export gbrain`, `seed amend`, `gbrain-sync` goal, plus brain-aware hooks and watch
- Structure locking, watch mode, state locks, hooks, Graphviz export, and shell completion

## Install

```bash
pip install seed-cli
pip install "seed-cli[image]"   # OCR/image parsing
pip install "seed-cli[ui]"      # rich terminal output
```

Python `>=3.10` is required.

## Quick Start

Capture an existing directory, preview a plan, then apply it:

```bash
seed capture --out project.tree
seed plan project.tree --out plan.json
seed apply plan.json
```

For direct spec execution:

```bash
seed register project.tree
seed apply project.tree
seed diff project.tree
```

For repository or service automation:

```bash
seed maintain maintenance.yml
seed maintain maintenance.yml --execute
```

## Agent-First Filesystem State

For autonomous coding agents, `filesystem.tree` can be the repository structure
contract:

```bash
seed init
seed import . --out filesystem.tree
seed plan filesystem.tree --json
seed apply filesystem.tree
seed check filesystem.tree --json
```

`seed plan --json` and `seed check --json` provide machine-readable drift
reports. `seed apply` does not delete extras unless pruning is explicit with
`--prune`; `seed sync filesystem.tree --prune` is the converge-and-prune form.

Framework maintainers can inspect the recommended integration contract with:

```bash
seed agents --json
seed agents --framework codex --json
seed agents --framework codex --format proposal
seed mcp
```

See [docs/agents.md](docs/agents.md) for OpenHands, Codex, Claude Code, Aider,
Cline, Roo Code, Continue, and Goose adoption guidance.

## Commands

`seed --help` groups commands by workflow so related tasks are easier to scan.

| Group | Commands | Description |
| --- | --- | --- |
| Plan & Apply | `init`, `plan`, `check`, `diff`, `apply`, `sync`, `match` | Preview, compare, and apply filesystem changes |
| Templates | `register`, `create`, `templates` (`template`) | Register, instantiate, and manage reusable specs |
| State & History | `import`, `capture`, `revert`, `specs`, `lock` | Capture state, recover snapshots, inspect history, and enforce structure versions |
| Maintenance | `validate`, `repair`, `doctor`, `maintain`, `hooks` | Lint specs, run repository/service maintenance, and install hooks |
| GBrain | `export gbrain`, `amend`, `specs watch --gbrain`, `hooks install --gbrain` | Compile a brain pack from a `.seed` spec, reconcile drift, and wire up brain hooks |
| Export & Utilities | `graph`, `export`, `agents`, `mcp`, `utils` | Export trees/plans/DOT output, agent metadata, MCP tools, and helper tools |

## Core Workflow

### Immutable Plans

```bash
seed plan dir_structure.tree --out plan.json
seed apply plan.json
```

`seed apply` also accepts a spec directly:

```bash
seed apply dir_structure.tree
```

When you apply a spec with template placeholders such as `<name>/` or
`features/<name>.ts`, `seed apply`
first runs `seed register` semantics: it writes the supporting files under
`.seed/templates/` and `.seed/templates/project/`, then removes any stale
literal placeholder paths like `features/<name>/` left behind by older runs.

### Drift Detection

```bash
seed diff dir_structure.tree
seed sync dir_structure.tree --dangerous
seed match dir_structure.tree --dangerous
```

`sync` deletes extras not in the spec. `match` also creates missing paths while
respecting directories marked with `...`.

### Partial Plans

```bash
seed plan dir_structure.tree --target scripts/
seed plan dir_structure.tree --targets "services/*"
seed plan dir_structure.tree --target-mode exact
```

## Spec Syntax

Use `.tree` for simple filesystem specs, or `.seed` when you want the same tree-shaped format plus richer inline metadata such as kinds, tags, and URLs.

### Basic Example

```text
@include base.tree

scripts/
├── build.py        @generated
├── notes.txt       @manual
├── cache/          ...
└── docs/           ?
```

### Markers

- `@include file.tree`: include another spec
- `@generated`: generated file
- `@manual`: manually maintained file
- `?`: optional file or directory
- `...`: allow extras inside a directory
- `<varname>`: template placeholder in a path segment or filename
- `{{var}}`: variable interpolation in file contents

`.seed` also supports inline metadata markers:

- `!kind`: semantic kind marker such as `!service`, `!doc`, or `!template`
- `+tag`: repeatable tags such as `+remote +shared`
- `-> URL`: attach a metadata URL to a node; on directory nodes this can be
  used as a template content source

Example:

```text
vendor/
└── api/ !service +remote -> https://github.com/acme/repo.git
```

Structured YAML and JSON specs can also carry metadata with either a
`metadata` object or top-level `kind`, `tags`, and `url` fields.

### Variable Usage

```bash
seed plan spec.tree --vars project_name=myapp
seed apply spec.tree --vars project_name=myapp
seed create spec.tree project_name=myapp
```

## Templates

### Template Directories

Define repeating structures with template variables:

```text
files/
├── <version_id>/
│   ├── data.json
│   └── meta/
└── ...
```

Placeholders can also appear in path-per-line specs or filenames, and templates
can include more than one placeholder:

```text
features/<domain>/<name>/route.ts
features/<name>.ts
```

Create instances:

```bash
seed create releases.tree version_id=v3
seed create component.tree domain=billing name=invoices
seed create releases.tree version_id=v3 --dry-run
```

### Project-Local Templates

Use `seed register` to mirror any `.tree` or `.seed` spec into the project-level
`.seed/templates/` directory. When the spec contains placeholders anywhere in a
path, it also extracts the outermost placeholder subtree into
`.seed/templates/project/`. `seed apply <spec>` runs the same registration step
automatically before execution.

```bash
seed register releases.tree
seed apply releases.tree
```

That lets you create from either a path-based project template:

```bash
seed create --template .seed/templates/releases.tree version_id=v3
```

or a registered project template name:

```bash
seed create --project version_id version_id=v3
```

### Template Registry

Manage reusable templates stored under `~/.seed/templates/` by default, or
under `$SEED_HOME/templates/` when `SEED_HOME` is set.

```bash
seed templates list
seed templates add ./template.tree --name my-template
seed templates add ./template.seed --name service-template
seed templates show my-template
seed templates use my-template target-folder
seed templates versions my-template --add ./updated.tree --name v2
seed templates lock my-template
seed templates update my-template
seed templates remove my-template
```

`seed templates list` also shows project-local templates discovered from
`.seed/templates/project/`, before global registry templates, using paths
relative to the project. Subtemplates are scoped to the parent directory where
they are meant to be used, so a template registered under `features/.seed/` is
only visible from `features/` or its children. `seed templates use <name>
<folder>` resolves a visible project template first, then falls back to the
global registry. The singular alias `seed template use ...` is also accepted.

Built-in templates include `fastapi`, `python-package`, and `node-typescript`.

### Template Content Sources

Templates can point at a local directory, a GitHub tree URL, or a git
repository URL so seed can fetch real file contents alongside the structure
spec. Repository sources are cloned without their `.git` metadata.

```bash
seed templates add ./fastapi --name fastapi \
  --content-url https://github.com/tiangolo/full-stack-fastapi-template/tree/master/backend/app

seed templates add ./service.seed --name service \
  --content-url https://github.com/acme/service-skeleton.git

seed templates update fastapi
seed templates update --all
seed templates update fastapi --content-url /path/to/local/files
```

Templates that include a `source.json` file with `{"content_url": "..."}` are
fetched automatically when installed. `.seed` directory nodes can also declare
their own content sources inline:

```text
vendor/
└── api/ !service +remote -> https://github.com/acme/api-client.git
```

### Copier-Style Scaffolding

`seed templates use` supports template config files named `copier.yml`,
`copier.yaml`, `.seed-template.yml`, or `.seed-template.yaml`.

Supported workflow features include:

- promptable questions and defaults
- `--data-file` for JSON/YAML answers
- `--defaults` and `--non-interactive`
- `--answers-file` or `_answers_file`
- `_exclude` and `_skip_if_exists`
- `_tasks`, which only execute with `--unsafe`
- `--overwrite` for existing files

Example:

```bash
seed templates use python-package \
  --base ./myapp \
  --data-file answers.yml \
  --defaults \
  --answers-file .seed/answers.yml \
  --overwrite
```

If a template defines `_tasks`, they are shown but skipped unless you opt into
execution:

```bash
seed templates use python-package --unsafe
```

## Repository & System Maintenance

`seed maintain` orchestrates repository, service, system, and project upkeep
from YAML or JSON manifests.

Built-in maintenance goals include:

- repositories: `ensure_path`, `git_fetch`, `git_status`, `git_pull_ff_only`
- services: `ensure_paths`, `compose_pull`, `compose_up`, `launchctl_restart`
- custom actions with `tool`, `args`, `cwd`, `env`, or shell commands

You can point `seed maintain` at a manifest file or a directory containing
`maintenance.yml`, `project.yml`, or `service.yml`.

### Workspace Manifest

```yaml
targets:
  - name: seed-cli
    kind: repository
    path: ./repos/seed-cli
    goals:
      - ensure_path
      - git_fetch
      - git_status

  - name: notes-api
    kind: service
    path: ./systems/services/notes-api
    config_dir: ./systems/services/notes-api/config
    data_dir: ./local/services/notes-api
    compose_file: compose.yml
    deploy_engine: docker-compose
    launch_agent: user/com.example.notes-api
    goals:
      - ensure_paths
      - compose_pull
      - compose_up
      - launchctl_restart
```

### `project.yml`

```yaml
name: product-x
type: project
path: ~/work/projects/active/product-x
maintenance:
  goals:
    - git_fetch
    - git_status
repos:
  - name: web-app
    path: repos/web-app
  - name: api
    path: repos/api
```

### `service.yml`

```yaml
name: notes-api
type: service
path: ~/systems/services/notes-api
config_dir: ~/systems/services/notes-api/config
data_dir: ~/local/services/notes-api
compose_file: compose.yml
deploy_engine: docker-compose
launch_agent: user/com.example.notes-api
maintenance:
  goals:
    - ensure_paths
    - compose_pull
    - compose_up
    - launchctl_restart
  actions:
    - tool: python
      args: ["scripts/rebuild_index.py"]
      cwd: "{{path}}"
```

Run the planner first, then execute:

```bash
seed maintain ./workspace
seed maintain ./workspace --execute
```

## GBrain Integration

seed-cli compiles `.seed` specs into [gbrain](https://github.com/garrytan/gbrain)
schema packs, keeps them in sync as the spec evolves, and folds drift back into
the spec so the brain stays authoritative.

### Export a pack

```bash
seed export gbrain brain.seed
seed export gbrain brain.seed --name my-brain --install --activate repo
seed export gbrain brain.seed --dry-run --json
```

Default output is `./.gbrain/pack/`. Versions are deterministic: `0.0.<patch>`
derived from a hash of the compiled spec (build metadata is stripped because
gbrain rejects it). Bundled at `resources/gbrain/kindmap.yml`; override with
`--kindmap`.

Key flags:

- `--name` pack name (default derived from spec)
- `--extends` base pack to inherit (default `gbrain-base`)
- `--kindmap` override the kindmap
- `--version-from {hash,spec,<literal>}` version source (default `hash`)
- `--install` copy pack to `~/.gbrain/schema-packs/<name>/`
- `--activate {repo,home,both,none}` write `gbrain.yml` (repo) and/or `gbrain schema use` (home)
- `--gbrain-min-version` override the `gbrain_min_version` field
- `--skip-validate` skip `gbrain schema validate`
- `--migrate {off,prompt,auto}` emit `migration_from` + `mapping_rules` from prefix diff against the prior spec; `auto` also submits `gbrain jobs submit unify-types`
- `--migrate-from` prior spec path (default: latest `.seed/specs/`)
- `--dry-run`, `--json`

### Reconcile drift back into the spec

`seed amend` folds filesystem drift (and optionally gbrain's active pack
typing) back into the `.seed` spec so the spec stays the source of truth.

```bash
seed amend brain.seed --from-fs --policy adopt --reexport
seed amend brain.seed --from-gbrain --policy quarantine --quarantine-dir _inbox/
seed amend brain.seed --dry-run --json
```

- `--from-fs` capture live filesystem (default on)
- `--from-gbrain` also read `gbrain schema show --json`; already-typed prefixes are not double-created
- `--policy {adopt,ignore,quarantine}` how to resolve new paths
- `--quarantine-dir` prefix for quarantined entries (default `_inbox/`)
- `--reexport` re-run `seed export gbrain` after amending
- Ignored entries persist at `.seed/gbrain/ignore.yml`

### Maintenance goal

The `gbrain-sync` goal in a maintenance manifest exports → optionally migrates
types → optionally syncs the brain.

```yaml
targets:
  - name: my-brain
    kind: project
    path: ./brains/my-brain
    metadata:
      gbrain:
        spec: brain.seed
        name: my-brain-pack
        extends: gbrain-base
        activate: repo          # repo|home|both|none
        migrate: prompt         # off|prompt|auto
        run_sync: true          # emit `gbrain sync`
        # optional: kindmap, version_from
    goals:
      - gbrain-sync
```

### Brain hooks and watch

```bash
seed hooks install --gbrain --spec brain.seed --name my-brain --activate repo
seed specs watch --gbrain --spec brain.seed --name my-brain --activate repo
```

`hooks install --gbrain` writes `hooks/post_apply_gbrain.sh` and a
`.git/hooks/pre-push` stale-pack guard. `specs watch --gbrain` re-exports the
pack each time `.seed/specs/` advances.

### Upstream PR draft

See `docs/gbrain/upstream-detect-pr.md` for the planned `gbrain schema detect`
seed-spec provider (file lookup order, kindmap-mirror strategy, tests, risks).

## Snapshots, Spec History, and Locks

Snapshots are created automatically before `apply`, `sync`, and `match`:

```bash
seed revert --list
seed revert
seed revert abc123 --dry-run
```

Applied structures are also captured as versioned specs:

```bash
seed specs list
seed specs show
seed specs diff v1 v3
seed specs watch
```

`seed specs watch` polls the workspace and writes a new `.seed/specs/vN.tree`
whenever the filesystem structure changes, so manual file creation after an
initial `apply` advances the internal reference automatically.

Lock a filesystem structure and watch it for drift:

```bash
seed lock set spec.tree
seed lock list
seed lock status
seed lock watch
seed lock upgrade v2 --dry-run
seed lock downgrade v1 --dangerous
```

## Export, Hooks, and Utilities

Export current state or a plan:

```bash
seed export tree --out structure.tree
seed export json --out structure.json
seed export dot --out structure.dot
seed export gbrain brain.seed --install --activate repo
seed plan spec.tree --dot > plan.dot
```

Install git hooks:

```bash
seed hooks install
seed hooks install --hook pre-push
seed hooks install --gbrain --spec brain.seed --name my-brain
```

Utilities:

```bash
seed utils extract-tree screenshot.png --out spec.tree
seed utils state-lock
seed utils state-lock --force-unlock
```

## Shell Autocomplete

Click provides shell completion for Bash, Zsh, and Fish. Generate the completion
script from the installed `seed` entry point:

```bash
# zsh
eval "$(_SEED_COMPLETE=zsh_source seed)"

# bash
eval "$(_SEED_COMPLETE=bash_source seed)"

# fish
_SEED_COMPLETE=fish_source seed | source
```

Then reload your shell and use tab completion:

```bash
seed <TAB>
seed templates <TAB>
seed lock <TAB>
```

## Safety Model

seed is designed to be safe by default:

- destructive workflows require explicit dangerous flags
- execution state is lock-protected with heartbeat renewal
- plans are validated before writes and deletes
- template tasks require explicit `--unsafe`
- answers files and execution targets are constrained to the base directory
- git maintenance refuses `git pull --ff-only` on dirty worktrees

## Philosophy

seed-cli is:

- Declarative
- Deterministic
- Auditable
- Safe by default

## License

Modified MIT. See `LICENSE.md`.
