# Tree-to-Seed Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a round-trippable `seed utils convert INPUT.tree [OUTPUT.seed]` command that emits compact brace-grouped `.seed` specs.

**Architecture:** A focused `seed_cli.conversion` module owns brace expansion, trie-based compact rendering, and file conversion. The existing parser calls brace expansion before its normal hierarchy logic, while both argparse and Click command definitions dispatch to the same conversion entry point.

**Tech Stack:** Python 3.10+, `pathlib`, dataclasses, Click, argparse, pytest.

---

## File Structure

- Create `src/seed_cli/conversion.py`: brace expansion, semantic node identity,
  path trie, compact rendering, and file conversion.
- Modify `src/seed_cli/parsers.py`: expand brace expressions before normal tree
  parsing.
- Modify `src/seed_cli/cli.py`: register and dispatch `utils convert` in both
  command frontends.
- Create `tests/test_conversion.py`: focused unit and round-trip tests.
- Modify `tests/test_parsers.py`: parser-level brace syntax tests.
- Modify `tests/test_cli.py`: command behavior and help integration tests.
- Modify `README.md`, `docs/index.html`, and `CHANGELOG.md`: document the new
  command and syntax.

### Task 1: Parse Compact Brace Expressions

**Files:**
- Create: `src/seed_cli/conversion.py`
- Modify: `src/seed_cli/parsers.py`
- Modify: `tests/test_parsers.py`

- [ ] **Step 1: Write failing parser tests**

Add tests equivalent to:

```python
def test_parse_seed_brace_group_with_shared_extension():
    nodes = parse_tree_text("memories/{global,facts,episodes}.jsonl")
    assert {n.relpath.as_posix() for n in nodes} == {
        "memories/global.jsonl",
        "memories/facts.jsonl",
        "memories/episodes.jsonl",
    }


def test_parse_seed_brace_group_with_mixed_files_and_directories():
    nodes = parse_tree_text(
        "services/<service-id>/{service.json,knowledge/,prompts/,tools.json}"
    )
    assert {(n.relpath.as_posix(), n.is_dir) for n in nodes} == {
        ("services/<service-id>/service.json", False),
        ("services/<service-id>/knowledge", True),
        ("services/<service-id>/prompts", True),
        ("services/<service-id>/tools.json", False),
    }


def test_parse_seed_multiple_brace_groups_as_cartesian_product():
    nodes = parse_tree_text("{people,teams}/{active,archived}.json")
    assert len(nodes) == 4
```

Also cover whitespace around alternatives, inherited inline metadata, literal
unmatched braces, and brace pairs without commas.

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```bash
pytest tests/test_parsers.py -q --no-cov
```

Expected: new assertions fail because grouped paths are currently parsed as
literal filenames.

- [ ] **Step 3: Implement brace expansion**

Create a public helper with this contract:

```python
def expand_brace_paths(text: str) -> list[str]:
    """Expand comma groups while preserving literal unmatched braces."""
```

Use a balanced-brace scan to find the first `{...}` containing a top-level
comma, strip whitespace around alternatives, and recursively expand the
prefix/alternative/suffix combination. Return `[text]` when no expandable
group exists.

In `parse_tree_text`, expand the cleaned node name and process each expanded
name against a copy of the same parent stack. Each expanded item must inherit
the original comment, annotation, optional flag, and metadata.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run:

```bash
pytest tests/test_parsers.py -q --no-cov
```

Expected: all parser tests pass.

- [ ] **Step 5: Commit parser support**

```bash
git add src/seed_cli/conversion.py src/seed_cli/parsers.py tests/test_parsers.py
git commit -m "feat: parse compact seed brace groups"
```

### Task 2: Render and Convert Compact Specs

**Files:**
- Modify: `src/seed_cli/conversion.py`
- Create: `tests/test_conversion.py`

- [ ] **Step 1: Write failing conversion tests**

Cover:

```python
def test_render_compact_seed_groups_shared_extensions():
    nodes = [
        Node(Path("memories/global.jsonl"), False),
        Node(Path("memories/facts.jsonl"), False),
        Node(Path("memories/episodes.jsonl"), False),
    ]
    assert (
        render_compact_seed(nodes)
        == "memories/{episodes,facts,global}.jsonl\n"
    )


def test_render_compact_seed_collapses_chains_and_mixed_leaves():
    nodes = [
        Node(Path("services/<service-id>/service.json"), False),
        Node(Path("services/<service-id>/knowledge"), True),
        Node(Path("services/<service-id>/prompts"), True),
        Node(Path("services/<service-id>/tools.json"), False),
    ]
    assert render_compact_seed(nodes) == (
        "services/<service-id>/{knowledge/,prompts/,service.json,tools.json}\n"
    )


def test_convert_tree_to_seed_defaults_to_same_stem(tmp_path):
    source = tmp_path / "nested" / "brain.tree"
    source.parent.mkdir()
    source.write_text("memories/global.jsonl\nmemories/facts.jsonl\n")
    output = convert_tree_to_seed(source)
    assert output == source.with_suffix(".seed")
    assert output.read_text() == "memories/{facts,global}.jsonl\n"
```

Add tests for explicit output paths, parent creation, invalid suffixes, missing
input, directories, metadata grouping, incompatible metadata staying separate,
deterministic ordering, deduplication, and semantic round trips through
`parse_spec`.

- [ ] **Step 2: Run conversion tests and verify RED**

Run:

```bash
pytest tests/test_conversion.py -q --no-cov
```

Expected: collection or assertion failure because rendering and conversion do
not exist.

- [ ] **Step 3: Implement semantic trie rendering**

Define an internal trie node that stores children and an optional terminal
`Node`. Insert every non-root path segment. Render recursively:

```python
def render_compact_seed(nodes: Iterable[Node]) -> str:
    """Return deterministic compact .seed path lines with a final newline."""


def convert_tree_to_seed(
    input_path: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Parse a .tree file and write a compact, round-trippable .seed file."""
```

Rendering rules:

1. Collapse a child chain while each level has one child and no terminal
   markers that would be lost.
2. Partition direct terminal children by `format_node_suffix(node)`.
3. Group partitions of at least two siblings.
4. Factor a shared file extension when every grouped entry is a file with the
   same non-empty suffix.
5. Factor `/` when every grouped entry is a directory; otherwise keep `/`
   inside each directory alternative.
6. Sort output paths and alternatives lexicographically.
7. Render incompatible metadata or comments as separate lines.

Validate `.tree` input and `.seed` output suffixes before parsing. Render fully
before creating output parents and writing, so parse/render errors do not
truncate an existing destination.

- [ ] **Step 4: Run conversion and parser tests**

Run:

```bash
pytest tests/test_conversion.py tests/test_parsers.py -q --no-cov
```

Expected: all tests pass.

- [ ] **Step 5: Commit conversion**

```bash
git add src/seed_cli/conversion.py tests/test_conversion.py
git commit -m "feat: convert tree specs to compact seed files"
```

### Task 3: Add `seed utils convert`

**Files:**
- Modify: `src/seed_cli/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add integration tests:

```python
def test_cli_utils_convert_with_default_output(tmp_path):
    (tmp_path / "brain.tree").write_text(
        "memories/global.jsonl\nmemories/facts.jsonl\n"
    )
    code, out, _ = run(["utils", "convert", "brain.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "brain.seed").exists()
    assert "brain.seed" in out


def test_cli_utils_convert_with_explicit_output(tmp_path):
    (tmp_path / "input.tree").write_text("a.txt\nb.txt\n")
    code, _, _ = run(
        ["utils", "convert", "input.tree", "generated/output.seed"],
        tmp_path,
    )
    assert code == 0
    assert (tmp_path / "generated/output.seed").exists()


def test_cli_utils_convert_help(tmp_path):
    code, out, _ = run(["utils", "convert", "--help"], tmp_path)
    assert code == 0
    assert "INPUT" in out
    assert "OUTPUT" in out
    assert "same stem" in out
```

Also assert a missing input returns non-zero with a concise error.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
pytest tests/test_cli.py -q --no-cov
```

Expected: Click reports that `convert` is not a known utility command.

- [ ] **Step 3: Wire argparse, Click, grouped help, and dispatch**

Add `"convert"` under a new `Specs` utility help section. Define:

```python
@utils_group.command(
    "convert",
    help="Convert a .tree spec to a compact declarative .seed spec.",
)
@click.argument("input_path", metavar="INPUT")
@click.argument("output_path", required=False, metavar="[OUTPUT]")
@click.pass_context
def utils_convert_command(ctx, input_path, output_path):
    return _dispatch(
        ctx,
        "utils",
        util_action="convert",
        input=input_path,
        output=output_path,
        base=".",
    )
```

Mirror the arguments in `build_parser()`. In `_run`, call
`convert_tree_to_seed(args.input, args.output)`, print
`Converted <input> to <output>`, and catch `FileNotFoundError`/`ValueError`
with return code `1`.

- [ ] **Step 4: Run CLI and conversion tests**

Run:

```bash
pytest tests/test_cli.py tests/test_conversion.py -q --no-cov
```

Expected: all tests pass.

- [ ] **Step 5: Commit CLI support**

```bash
git add src/seed_cli/cli.py tests/test_cli.py
git commit -m "feat: add seed utils convert command"
```

### Task 4: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/index.html`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Document command and syntax**

Add command examples:

```bash
seed utils convert project.tree
seed utils convert path/to/project.tree output/project.seed
seed utils convert --help
```

Explain that `.seed` brace groups such as
`memories/{global,facts}.jsonl` expand declaratively and are accepted by all
spec-consuming commands.

- [ ] **Step 2: Add changelog entries**

Under `[Unreleased]`, record the conversion command, compact brace syntax,
round-trip parsing, and associated tests using Keep a Changelog headings.

- [ ] **Step 3: Run formatting and focused verification**

Run:

```bash
black --check src/seed_cli/conversion.py src/seed_cli/parsers.py src/seed_cli/cli.py tests/test_conversion.py tests/test_parsers.py tests/test_cli.py
git diff --check
pytest tests/test_conversion.py tests/test_parsers.py tests/test_cli.py -q --no-cov
```

Expected: formatting, whitespace, and focused tests all pass.

- [ ] **Step 4: Run the full suite**

Run:

```bash
pytest
```

Expected: full suite passes and repository coverage remains at or above 85%.

- [ ] **Step 5: Manually verify command output**

Create a temporary `.tree` fixture, run both default and explicit output forms,
run `seed utils convert --help`, then run `seed plan` on the generated `.seed`.
Expected: conversion succeeds, help documents arguments, and the generated spec
plans the same semantic paths.

- [ ] **Step 6: Commit documentation and final adjustments**

```bash
git add README.md docs/index.html CHANGELOG.md
git commit -m "docs: document compact seed conversion"
```

- [ ] **Step 7: Push the branch**

```bash
git push -u origin feature/v1.0.13
```

Expected: the branch and all implementation commits are present on `origin`.
