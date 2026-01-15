# seed-cli

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
- State locking + heartbeat renewal
- Partial plans (`--target scripts/`)
- Spec inheritance (`@include`)
- Variables (`{{project_name}}`)
- Plugins
- Checksums & drift detection
- CI & pre-commit hooks
- Graphviz execution graphs (`--dot`)

---

## Install

```bash
pip install seed-cli
pip install "seed-cli[image]"   # OCR support
```

---

## Core Workflow

```bash
seed plan dir_structure.tree --out plan.json
seed apply plan.json
```

---

## Commands

| Command | Description                     |
| ------- | ------------------------------- |
| plan    | Generate execution plan         |
| apply   | Apply spec or plan              |
| sync    | Apply + delete extras           |
| diff    | Compare FS vs spec              |
| capture | Capture FS to spec              |
| doctor  | Lint & repair specs             |
| export  | Export filesystem state or plan |
| lock    | Manage state locks              |
| hooks   | Install git hooks               |

---

## Example Spec

```text
@include base.tree

scripts/
├── build.py (@generated)
└── notes.txt (@manual)
```

---

## State & Locking

State is stored in `.seed/state.json`.
Locks are stored in `.seed/lock.json`.

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

seed-cli is extensible by default. You can create your own plugins and use them to do any dir modifications, transformations, or anything you can conceive and is not captured in the previous statement.

Note: this is Work In Progress.

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
