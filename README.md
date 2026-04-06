# seed-cli

![seed-cli](https://github.com/user-attachments/assets/5661d43b-816f-40d3-b47e-23f85a0eae34)

**seed** is a Terraform-inspired, spec-driven filesystem orchestration tool.

It captures directory trees, plans changes, applies them safely, syncs drift,
scaffolds projects from reusable templates, and can also execute
manifest-driven repository or service maintenance.

Think **Terraform for directory trees**, plus **template scaffolding** and
**workspace maintenance**.

## Highlights

- Multiple spec inputs: `.tree`, YAML, JSON, DOT, image OCR, and stdin
- Deterministic planning with exportable plans: `seed plan spec.tree --out plan.json`
- Safe execution of immutable plans: `seed apply plan.json`
- Drift workflows: `diff`, `sync`, `match`, snapshots, and spec history
- Template variables in paths and content: `<varname>/` and `{{var}}`
- Project-local template registration under `.seed/templates/`
- Reusable template registry with versions, locking, content sources, and built-in templates
- Copier-style template config support: questions, defaults, answers files, excludes, skip-if-exists, and gated tasks
- Manifest-driven repository/service/system maintenance with `seed maintain`
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
seed apply project.tree
seed diff project.tree
```

For repository or service automation:

```bash
seed maintain maintenance.yml
seed maintain maintenance.yml --execute
```

## Commands

| Command | Description |
| --- | --- |
| `plan` | Parse a spec and generate an execution plan |
| `apply` | Apply a spec or a saved plan |
| `sync` | Apply a spec and delete extras |
| `diff` | Compare a spec with the filesystem |
| `match` | Modify the filesystem to match a spec, respecting `...` |
| `maintain` | Build or execute repository/service/system maintenance plans |
| `create` | Instantiate template directory structures |
| `revert` | Revert to a previous snapshot |
| `doctor` | Lint a spec and optionally auto-fix issues |
| `capture` | Capture filesystem state as a spec |
| `export` | Export a tree, JSON spec, plan, or DOT graph |
| `lock` | Manage structure locks, versions, and watch mode |
| `hooks` | Install git hooks |
| `specs` | View captured spec history |
| `templates` | Manage reusable templates |
| `utils` | `extract-tree` and `state-lock` helpers |

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
- `<varname>/`: template directory placeholder
- `{{var}}`: variable interpolation in file contents

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

Create instances:

```bash
seed create releases.tree version_id=v3
seed create releases.tree version_id=v3 --dry-run
```

### Project-Local Templates

When a parsed spec contains template subtrees, seed mirrors the spec into the
project-level `.seed/templates/` directory and also extracts nested template
subtrees into `.seed/templates/project/`.

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
seed templates show my-template
seed templates use my-template
seed templates versions my-template --add ./updated.tree --name v2
seed templates lock my-template
seed templates update my-template
seed templates remove my-template
```

Built-in templates include `fastapi`, `python-package`, and `node-typescript`.

### Template Content Sources

Templates can point at a local directory or a GitHub tree URL so seed can fetch
real file contents alongside the structure spec.

```bash
seed templates add ./fastapi --name fastapi \
  --content-url https://github.com/tiangolo/full-stack-fastapi-template/tree/master/backend/app

seed templates update fastapi
seed templates update --all
seed templates update fastapi --content-url /path/to/local/files
```

Templates that include a `source.json` file with `{"content_url": "..."}` are
fetched automatically when installed.

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
```

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
seed plan spec.tree --dot > plan.dot
```

Install git hooks:

```bash
seed hooks install
seed hooks install --hook pre-push
```

Utilities:

```bash
seed utils extract-tree screenshot.png --out spec.tree
seed utils state-lock
seed utils state-lock --force-unlock
```

## Shell Autocomplete

Enable completion with `argcomplete`:

```bash
# zsh / bash
eval "$(register-python-argcomplete seed)"

# fish
register-python-argcomplete --shell fish seed | source
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
