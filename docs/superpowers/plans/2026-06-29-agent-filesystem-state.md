# Agent Filesystem State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make seed-cli expose a deterministic, agent-first filesystem state interface that coding agents can safely call and recommend.

**Architecture:** Reuse the existing parse, plan, diff, apply, sync, doctor, capture, and graphviz modules. Add thin CLI adapters for PRD command names, machine-readable summaries, PRD exit codes where new agent-focused surfaces require them, and an agent-framework manifest that documents adoption-ready workflows.

**Tech Stack:** Python 3.10+, argparse CLI, pytest, existing seed_cli modules only.

## Global Constraints

- Do not remove any current functionality in pros of this PRD.
- The default format remains the existing tree DSL.
- Every command supports `--json` where implemented in this plan.
- Exit codes for new agent-facing surfaces: `0` success/already matches, `1` fatal error, `2` drift detected, `3` validation failed, `4` unsafe operation refused.
- Deletion is never implicit; use `--prune` or existing `--dangerous` flags explicitly.
- Core should remain stable and dependency-light.

---

### Task 1: Agent JSON and PRD Command Tests

**Files:**
- Modify: `tests/test_cli.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: existing `run()` helper in `tests/test_cli.py`
- Produces: failing tests for `plan --json`, `check`, `init`, `import`, `validate`, `repair`, `graph`, `apply --prune`, `sync --prune`, and `agents --json`

- [ ] **Step 1: Write failing CLI tests**

Add tests that assert:

```python
def test_cli_plan_json_reports_drift_with_exit_code_2(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text("src/\n└── app.py\n", encoding="utf-8")
    code, out, err = run(["plan", "filesystem.tree", "--json"], tmp_path)
    payload = json.loads(out)
    assert code == 2
    assert payload["status"] == "drift"
    assert payload["create"] == ["src", "src/app.py"]
    assert payload["delete"] == []
    assert payload["errors"] == []
```

```python
def test_cli_check_json_uses_drift_exit_code(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text("src/\n", encoding="utf-8")
    code, out, err = run(["check", "filesystem.tree", "--json"], tmp_path)
    payload = json.loads(out)
    assert code == 2
    assert payload["status"] == "drift"
    assert payload["missing"] == ["src"]
```

```python
def test_cli_init_writes_default_filesystem_tree(tmp_path):
    code, out, err = run(["init"], tmp_path)
    assert code == 0
    assert (tmp_path / "filesystem.tree").read_text(encoding="utf-8") == ".\n"
```

```python
def test_cli_import_alias_writes_tree_spec(tmp_path):
    (tmp_path / "src").mkdir()
    code, out, err = run(["import", ".", "--out", "filesystem.tree"], tmp_path)
    assert code == 0
    assert "src/" in (tmp_path / "filesystem.tree").read_text(encoding="utf-8")
```

```python
def test_cli_validate_and_repair_aliases_doctor(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text("a.txt\na.txt\n", encoding="utf-8")
    validate_code, validate_out, _ = run(["validate", "filesystem.tree", "--json"], tmp_path)
    repair_code, repair_out, _ = run(["repair", "filesystem.tree", "--json"], tmp_path)
    assert validate_code == 3
    assert json.loads(validate_out)["valid"] is False
    assert repair_code == 3
    assert json.loads(repair_out)["repaired"] is True
```

```python
def test_cli_graph_outputs_mermaid(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text("src/\n└── app.py\n", encoding="utf-8")
    code, out, err = run(["graph", "filesystem.tree", "--format", "mermaid"], tmp_path)
    assert code == 0
    assert "graph TD" in out
    assert "src/app.py" in out
```

```python
def test_cli_apply_prune_deletes_extras_explicitly(tmp_path):
    (tmp_path / "tmp").mkdir()
    spec = tmp_path / "filesystem.tree"
    spec.write_text(".\n", encoding="utf-8")
    code, out, err = run(["apply", "filesystem.tree", "--prune"], tmp_path)
    assert code == 0
    assert not (tmp_path / "tmp").exists()
```

```python
def test_cli_sync_prune_is_safe_deletion_spelling(tmp_path):
    (tmp_path / "tmp").mkdir()
    spec = tmp_path / "filesystem.tree"
    spec.write_text(".\n", encoding="utf-8")
    code, out, err = run(["sync", "filesystem.tree", "--prune"], tmp_path)
    assert code == 0
    assert not (tmp_path / "tmp").exists()
```

- [ ] **Step 2: Write failing agent manifest tests**

Create `tests/test_agent.py` with tests for:

```python
from seed_cli.agent import agent_manifest, agent_manifest_markdown

def test_agent_manifest_names_target_frameworks():
    manifest = agent_manifest()
    assert manifest["tool"] == "seed"
    assert "filesystem_state_management" in manifest["capabilities"]
    for framework in ["OpenHands", "Codex", "Claude Code", "Aider", "Cline", "Roo Code", "Continue", "Goose"]:
        assert framework in manifest["frameworks"]

def test_agent_manifest_markdown_includes_install_and_json_workflow():
    markdown = agent_manifest_markdown()
    assert "pip install seed-cli" in markdown
    assert "seed plan filesystem.tree --json" in markdown
    assert "seed apply filesystem.tree" in markdown
```

- [ ] **Step 3: Run tests and verify RED**

Run: `uv run pytest -o addopts='' tests/test_cli.py::test_cli_plan_json_reports_drift_with_exit_code_2 tests/test_agent.py -q`

Expected: fail because `--json` plan and `seed_cli.agent` do not exist.

### Task 2: Core PRD CLI Surfaces

**Files:**
- Modify: `src/seed_cli/cli.py`

**Interfaces:**
- Produces: `_plan_payload(plan) -> dict`, `_diff_payload(res) -> dict`, `init`, `import`, `check`, `validate`, `repair`, `graph`, `--prune`, and core `--json` handling.

- [ ] **Step 1: Implement JSON helpers**

Add helpers near CLI utilities:

```python
def _plan_status(plan):
    return "match" if not any(step.op != "skip" for step in plan.steps) and plan.delete_skipped == 0 else "drift"

def _plan_json(plan):
    return {
        "status": _plan_status(plan),
        "create": [s.path for s in plan.steps if s.op in ("mkdir", "create")],
        "update": [s.path for s in plan.steps if s.op == "update"],
        "delete": [s.path for s in plan.steps if s.op == "delete"],
        "skipped_delete": [s.path for s in plan.steps if s.op == "skip" and s.reason == "extra"],
        "rename": [],
        "errors": [],
        "summary": plan.to_json()["summary"],
        "steps": plan.to_json()["steps"],
    }
```

- [ ] **Step 2: Add parser entries**

Add subcommands and options:

```text
seed init [--base .] [--force] [--json]
seed import [path] [--base .] [--out filesystem.tree] [--json]
seed check SPEC [--base .] [--json]
seed validate SPEC [--base .] [--fix] [--json]
seed repair SPEC [--base .] [--json]
seed graph SPEC [--base .] [--format dot|mermaid|ascii] [--json]
```

Add `--json` to `plan`, `apply`, `sync`, `diff`, and `doctor`. Add `--prune` to `apply` and `sync` as the PRD spelling for explicit deletion.

- [ ] **Step 3: Implement command handlers**

Reuse existing helpers:

```text
init: write filesystem.tree with ".\n"; refuse overwrite unless --force.
import: capture nodes from path/base and write tree text to --out or stdout.
check: parse spec, run diff, return 0 clean or 2 drift.
validate: run doctor; return 0 valid or 3 validation failed.
repair: run doctor(..., fix=True), write repaired tree back to the spec if issues were fixable, return 3 if issues were found.
graph: render nodes as DOT, Mermaid, or ASCII tree text.
apply --prune: pass allow_delete=True and dangerous=True.
sync --prune: allow sync without --dangerous when --prune is present.
```

- [ ] **Step 4: Run tests and verify GREEN for Task 2**

Run: `uv run pytest -o addopts='' tests/test_cli.py::test_cli_plan_json_reports_drift_with_exit_code_2 tests/test_cli.py::test_cli_check_json_uses_drift_exit_code tests/test_cli.py::test_cli_init_writes_default_filesystem_tree tests/test_cli.py::test_cli_import_alias_writes_tree_spec tests/test_cli.py::test_cli_validate_and_repair_aliases_doctor tests/test_cli.py::test_cli_graph_outputs_mermaid tests/test_cli.py::test_cli_apply_prune_deletes_extras_explicitly tests/test_cli.py::test_cli_sync_prune_is_safe_deletion_spelling -q`

Expected: all selected CLI tests pass.

### Task 3: Agent Framework Manifest

**Files:**
- Create: `src/seed_cli/agent.py`
- Modify: `src/seed_cli/cli.py`
- Create: `docs/agents.md`

**Interfaces:**
- Produces: `agent_manifest() -> dict`
- Produces: `agent_manifest_markdown() -> str`
- Produces CLI: `seed agents [--json] [--format json|markdown]`

- [ ] **Step 1: Implement manifest module**

The manifest must include:

```python
frameworks = ["OpenHands", "Codex", "Claude Code", "Aider", "Cline", "Roo Code", "Continue", "Goose"]
capabilities = ["filesystem_state_management", "deterministic_plan_apply", "drift_detection", "safe_prune", "json_cli"]
recommended_commands = [
    "seed init",
    "seed import . --out filesystem.tree",
    "seed plan filesystem.tree --json",
    "seed apply filesystem.tree",
    "seed check filesystem.tree --json",
    "seed sync filesystem.tree --prune",
]
```

- [ ] **Step 2: Add CLI command**

Add `agents` to the parser and `_run()` with Markdown default and JSON when `--json` or `--format json` is used.

- [ ] **Step 3: Add docs**

Create `docs/agents.md` describing why frameworks should recommend seed, the safe command contract, exit codes, and short integration snippets for the named frameworks.

- [ ] **Step 4: Run tests and verify GREEN for Task 3**

Run: `uv run pytest -o addopts='' tests/test_agent.py tests/test_cli.py::test_cli_agents_json_manifest -q`

Expected: all selected tests pass.

### Task 4: Documentation and Release Notes

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create/Modify: `handoff.md`

**Interfaces:**
- Produces docs for PRD command aliases and agent adoption.

- [ ] **Step 1: README**

Add concise sections covering:

```text
Agent-first filesystem state management
PRD command aliases: init/import/check/validate/repair/graph
Machine JSON: seed plan filesystem.tree --json, seed check filesystem.tree --json
Safe pruning: seed apply --prune, seed sync --prune
Agent manifest: seed agents --json
```

- [ ] **Step 2: CHANGELOG**

Under `[Unreleased]`, add:

```markdown
### Added
- Added PRD-aligned agent-first CLI surfaces...
- Added `seed agents` framework integration manifest...
```

- [ ] **Step 3: handoff**

Append a short handoff with changed files, verification commands, and known risks.

- [ ] **Step 4: Final verification**

Run targeted tests, then run the broad suite if practical:

```bash
uv run pytest -o addopts='' tests/test_cli.py tests/test_agent.py tests/test_capture.py tests/test_doctor.py tests/test_graphviz.py -q
uv run pytest -q
```

Expected: targeted tests pass. Full suite may fail only for pre-existing coverage threshold if run as a subset; full run should satisfy the project gate.

### Task 5: Commit and PR

**Files:**
- Git metadata only

**Interfaces:**
- Produces commit, push, PR, and PR feedback check.

- [ ] **Step 1: Inspect diff**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

- [ ] **Step 2: Commit**

Run:

```bash
git add docs/superpowers/plans/2026-06-29-agent-filesystem-state.md src/seed_cli/cli.py src/seed_cli/agent.py tests/test_cli.py tests/test_agent.py docs/agents.md README.md CHANGELOG.md handoff.md
git commit -m "feat: add agent-first filesystem state surfaces"
```

- [ ] **Step 3: Push and PR**

Run:

```bash
git push -u origin feature/codex/prd-agent-filesystem-state
gh pr create --title "Add agent-first filesystem state surfaces" --body-file <generated-pr-body>
```

- [ ] **Step 4: Monitor PR once**

Run:

```bash
gh pr view --comments
```

Respond or patch if there are actionable comments available immediately.
