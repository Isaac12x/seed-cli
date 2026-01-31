# seed-cli

![seed-cli](https://github.com/user-attachments/assets/5661d43b-816f-40d3-b47e-23f85a0eae34)


**seed** is a Terraform-inspired, spec-driven filesystem orchestration tool.

It allows you to declaratively describe directory structures, generate plans,
diff against reality, safely apply changes, detect drift, lock state, and
collaborate using immutable execution plans.

Think **Terraform for directory trees**.

---

## Features

- Tree / YAML / JSON / stdin / image-based specs
- Deterministic planning (`seed plan`)
- Immutable plans (`seed plan --out plan.json`)
- Safe execution (`seed apply plan.json`)
- Sync with deletion (`seed sync --dangerous`)
- Match filesystem to spec (`seed match --dangerous`)
- Template directories (`<varname>/` syntax)
- Extras-allowed markers (`...` in spec)
- Optional items with prompting (`?` marker)
- Snapshot & revert (`seed revert`)
- Automatic spec history (`seed specs`)
- Structure versioning (`seed lock set/upgrade/downgrade`)
- Watch mode for continuous enforcement
- State locking + heartbeat renewal
- Partial plans (`--target scripts/`)
- Spec inheritance (`@include`)
- Variables (`{{project_name}}`)
- Template content from GitHub (`source.json`)
- Plugins
- Checksums & drift detection
- CI & pre-commit hooks
- Graphviz execution graphs (`--dot`)
- Shell tab completion

---

## Install

```bash
pip install seed-cli
pip install "seed-cli[image]"   # OCR support
pip install "seed-cli[ui]"      # Rich terminal output
```

---

## Shell Autocomplete

Enable tab completion for your shell:

```bash
# Add to ~/.zshrc or ~/.bashrc
eval "$(register-python-argcomplete seed)"
```

For fish shell:
```fish
register-python-argcomplete --shell fish seed | source
```

Then reload your shell (`source ~/.zshrc`) and use tab completion:
```bash
seed <TAB>          # shows available commands
seed diff --<TAB>   # shows available options
```

---

## Core Workflow

```bash
seed plan dir_structure.tree --out plan.json
seed apply plan.json
```

---

## Commands

| Command  | Description                                      |
| -------- | ------------------------------------------------ |
| plan     | Generate execution plan                          |
| apply    | Apply spec or plan                               |
| sync     | Apply + delete extras                            |
| match    | Modify filesystem to match spec (respects `...`) |
| diff     | Compare filesystem vs spec                       |
| create   | Create new instance of template structure        |
| revert   | Revert to previous snapshot (undo)               |
| specs    | View captured spec version history               |
| capture  | Capture filesystem to spec                       |
| doctor   | Lint & repair specs                              |
| export   | Export filesystem state or plan                  |
| lock     | Manage structure locks and versioning            |
| utils    | Utility functions (extract-tree, state-lock)     |

---

## Example Spec

```text
@include base.tree

scripts/
├── build.py    @generated
├── notes.txt   @manual
└── ...                        # extra files allowed here
```

---

## Optional Items

Mark files or directories as optional with `?`. The user will be prompted whether to create them:

```text
project/
├── src/
├── tests/           ?         # optional - prompt before creating
├── docs/            ?         # optional
└── config.json
```

Control prompting behavior:
```bash
# Prompt for each optional item (default)
seed apply spec.tree

# Create all optional items without prompting
seed apply spec.tree --yes

# Skip all optional items without prompting
seed apply spec.tree --skip-optional
```

---

## Template Directories

Define repeating structures with template variables:

```text
files/
├── <version_id>/              # matches any directory name
│   ├── data.json
│   └── meta/
└── ...
```

Create new instances:
```bash
seed create spec.tree version_id=v3
```

---

## Template Content Sources

Templates can point to real file contents via `source.json` or the `--content-url` flag. When a content source is set, `seed` fetches actual files (from a local directory or a GitHub tree URL) and stores them alongside the spec.

```bash
# Add a template with content from GitHub
seed templates add ./fastapi --name fastapi \
  --content-url https://github.com/tiangolo/full-stack-fastapi-template/tree/master/backend/app

# Re-fetch content from the stored source
seed templates update fastapi

# Update all templates with content sources
seed templates update --all

# Change where content is fetched from
seed templates update fastapi --content-url https://github.com/other/repo/tree/main/src
```

Templates that include a `source.json` file (containing `{"content_url": "..."}`) will automatically fetch content when installed. The built-in `fastapi`, `python-package`, and `node-typescript` templates use this feature.

---

## Match Command

The `match` command modifies the filesystem to match your spec, creating missing items and deleting extras:

```bash
# Preview changes
seed match spec.tree --dry-run

# Apply changes (creates and deletes)
seed match spec.tree --dangerous
```

Use `...` in your spec to mark directories where extra files are allowed:
```text
src/
├── index.ts
├── lib/
│   └── ...        # extras allowed in lib/
└── config.json    # no extras allowed at src/ level
```

---

## Snapshots & Revert

Snapshots are automatically created before apply/match/sync operations:

```bash
# List available snapshots
seed revert --list

# Revert to latest snapshot
seed revert

# Revert to specific snapshot
seed revert abc123

# Preview what would be reverted
seed revert --dry-run
```

---

## Spec History

After every `apply`, `sync`, or `match` operation, seed automatically captures the resulting filesystem structure as a versioned spec in `.seed/specs/`:

```
.seed/specs/
├── v1.tree      # first captured state
├── v2.tree      # after second operation
├── v3.tree      # etc.
└── current.tree # symlink to latest
```

View and manage spec history:
```bash
# List all captured versions
seed specs list

# Show latest spec
seed specs show

# Show specific version
seed specs show v2

# Compare two versions
seed specs diff v1 v3
```

This provides an audit trail of how your project structure evolved over time.

---

## Structure Locking & Versioning

Lock your filesystem to a specific structure spec with versioning support:

```bash
# Set active structure (creates version)
seed lock set spec.tree

# List versions
seed lock list

# Watch for changes (continuous enforcement)
seed lock watch

# Upgrade/downgrade versions
seed lock upgrade
seed lock downgrade

# Check current status
seed lock status
```

---

## Diff Options

```bash
# Basic diff
seed diff spec.tree

# Ignore specific patterns
seed diff spec.tree --ignore "*.log" --ignore "node_modules/**"

# Hide extras inside directories defined in spec
seed diff spec.tree --no-sublevels
```

---

## State & Locking

State is stored in `.seed/state.json`.
Execution locks are stored in `.seed/lock.json`.
Structure versions are stored in `.seed/structures/`.

Locks:

- TTL-based
- Auto-renewed during apply
- Force-unlock available

---

## Partial Plans

```bash
seed plan spec.tree --target scripts/
```

---

## Graphviz

```bash
seed plan spec.tree --dot > plan.dot
dot -Tpng plan.dot -o plan.png
```

---

## Plugins

seed-cli is extensible. Create plugins for directory modifications, transformations, or custom behaviors.

Local plugins live in:

```text
.seed/plugins/*.py
```

---

## Philosophy

seed-cli is:

- Declarative
- Deterministic
- Auditable
- Safe by default

## License

Modified MIT file.
Read the `LICENSE.md` file in this project.
