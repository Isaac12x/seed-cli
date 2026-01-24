"""Tests for seed_cli.match module.

Comprehensive tests for match functionality including extras, templates, and filtering.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from seed_cli.match import (
    _extract_extras_allowed,
    _extract_templates,
    _expand_templates,
    _filter_templates_from_extras,
    _is_under_extras_allowed,
    _filter_plan_for_extras,
    _filter_nodes_for_planning,
    match,
    match_plan,
    create_from_template,
)
from seed_cli.parsers import Node
from seed_cli.planning import PlanResult, PlanStep


# -----------------------------------------------------------------------------
# Tests for _extract_extras_allowed
# -----------------------------------------------------------------------------

def test_extract_extras_allowed_empty_nodes():
    """Should return empty set for empty node list."""
    result = _extract_extras_allowed([])
    assert result == set()


def test_extract_extras_allowed_no_markers():
    """Should return empty set when no ... markers."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/file.py"), False),
    ]
    result = _extract_extras_allowed(nodes)
    assert result == set()


def test_extract_extras_allowed_root_level():
    """Should add empty string for root level ... marker."""
    nodes = [
        Node(Path("..."), False, annotation="extras"),
    ]
    result = _extract_extras_allowed(nodes)
    assert "" in result


def test_extract_extras_allowed_nested_marker():
    """Should extract parent directory from nested ... marker."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/..."), False),
    ]
    result = _extract_extras_allowed(nodes)
    assert "src" in result


def test_extract_extras_allowed_multiple_markers():
    """Should extract all directories with ... markers."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/..."), False),
        Node(Path("lib"), True),
        Node(Path("lib/..."), False),
        Node(Path("tests"), True),
    ]
    result = _extract_extras_allowed(nodes)
    assert "src" in result
    assert "lib" in result
    assert "tests" not in result


def test_extract_extras_allowed_annotation_style():
    """Should recognize extras annotation."""
    nodes = [
        Node(Path("config/..."), False, annotation="extras"),
    ]
    result = _extract_extras_allowed(nodes)
    assert "config" in result


def test_extract_extras_allowed_deep_nesting():
    """Should handle deeply nested ... markers."""
    nodes = [
        Node(Path("a/b/c/..."), False),
    ]
    result = _extract_extras_allowed(nodes)
    assert "a/b/c" in result


# -----------------------------------------------------------------------------
# Tests for _extract_templates
# -----------------------------------------------------------------------------

def test_extract_templates_empty_nodes():
    """Should return empty dict for empty nodes."""
    result = _extract_templates([])
    assert result == {}


def test_extract_templates_no_templates():
    """Should return empty dict when no template markers."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/file.py"), False),
    ]
    result = _extract_templates(nodes)
    assert result == {}


def test_extract_templates_single_template():
    """Should extract template directory with annotation."""
    nodes = [
        Node(Path("files/<version_id>"), True, annotation="template:version_id"),
        Node(Path("files/<version_id>/data.json"), False),
    ]
    result = _extract_templates(nodes)
    assert "files/<version_id>" in result
    assert len(result["files/<version_id>"]) == 1


def test_extract_templates_multiple_children():
    """Should collect all children of template directory."""
    nodes = [
        Node(Path("files/<id>"), True, annotation="template:id"),
        Node(Path("files/<id>/file1.txt"), False),
        Node(Path("files/<id>/file2.txt"), False),
        Node(Path("files/<id>/subdir"), True),
    ]
    result = _extract_templates(nodes)
    assert "files/<id>" in result
    assert len(result["files/<id>"]) == 3


def test_extract_templates_root_level_template():
    """Should handle template at root level."""
    nodes = [
        Node(Path("<project>"), True, annotation="template:project"),
        Node(Path("<project>/src"), True),
    ]
    result = _extract_templates(nodes)
    assert "<project>" in result


# -----------------------------------------------------------------------------
# Tests for _expand_templates
# -----------------------------------------------------------------------------

def test_expand_templates_empty_templates(tmp_path):
    """Should return original nodes when no templates."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/file.py"), False),
    ]
    result = _expand_templates(nodes, {}, tmp_path)
    assert len(result) == 2


def test_expand_templates_creates_concrete_nodes(tmp_path):
    """Should expand template to concrete directories."""
    # Create actual directories on filesystem
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "v1").mkdir()
    (tmp_path / "files" / "v2").mkdir()

    nodes = [
        Node(Path("files"), True),
        Node(Path("files/<id>"), True, annotation="template:id"),
        Node(Path("files/<id>/data.json"), False),
    ]
    templates = {"files/<id>": [Node(Path("files/<id>/data.json"), False)]}

    result = _expand_templates(nodes, templates, tmp_path)

    # Should have original files/ + expanded v1/, v1/data.json, v2/, v2/data.json
    paths = [n.relpath.as_posix() for n in result]
    assert "files" in paths
    assert "files/v1" in paths
    assert "files/v2" in paths
    assert "files/v1/data.json" in paths
    assert "files/v2/data.json" in paths


def test_expand_templates_skips_template_markers(tmp_path):
    """Should not include template marker nodes in result."""
    (tmp_path / "files").mkdir()

    nodes = [
        Node(Path("files"), True),
        Node(Path("files/<id>"), True, annotation="template:id"),
    ]
    templates = {"files/<id>": []}

    result = _expand_templates(nodes, templates, tmp_path)
    paths = [n.relpath.as_posix() for n in result]
    assert "files/<id>" not in paths


def test_expand_templates_parent_not_exists(tmp_path):
    """Should handle when parent directory doesn't exist yet."""
    nodes = [
        Node(Path("nonexistent/<id>"), True, annotation="template:id"),
    ]
    templates = {"nonexistent/<id>": []}

    result = _expand_templates(nodes, templates, tmp_path)
    # Should return nodes without expansion since parent doesn't exist
    assert len(result) == 0  # Template marker filtered, no expansion possible


# -----------------------------------------------------------------------------
# Tests for _filter_templates_from_extras
# -----------------------------------------------------------------------------

def test_filter_templates_from_extras_empty():
    """Should return original extras when no templates."""
    extras = {"src"}
    result = _filter_templates_from_extras(extras, {})
    assert result == {"src"}


def test_filter_templates_from_extras_adds_parents():
    """Should add template parent directories to extras."""
    extras = set()
    templates = {"files/<id>": []}

    result = _filter_templates_from_extras(extras, templates)
    assert "files" in result


def test_filter_templates_from_extras_root_template():
    """Should add empty string for root-level templates."""
    extras = set()
    templates = {"<project>": []}

    result = _filter_templates_from_extras(extras, templates)
    assert "" in result


def test_filter_templates_from_extras_preserves_existing():
    """Should preserve existing extras."""
    extras = {"config", "src"}
    templates = {"files/<id>": []}

    result = _filter_templates_from_extras(extras, templates)
    assert "config" in result
    assert "src" in result
    assert "files" in result


# -----------------------------------------------------------------------------
# Tests for _is_under_extras_allowed
# -----------------------------------------------------------------------------

def test_is_under_extras_allowed_root():
    """Should recognize paths under root extras."""
    extras = {""}
    assert _is_under_extras_allowed("file.txt", extras) is True
    # When root allows extras, all paths are allowed (any path has root as parent)
    assert _is_under_extras_allowed("subdir/file.txt", extras) is True


def test_is_under_extras_allowed_nested():
    """Should recognize paths under nested extras directory."""
    extras = {"src"}
    assert _is_under_extras_allowed("src/file.py", extras) is True
    assert _is_under_extras_allowed("src/sub/file.py", extras) is True
    assert _is_under_extras_allowed("lib/file.py", extras) is False


def test_is_under_extras_allowed_exact_match():
    """Should handle exact directory match."""
    extras = {"config"}
    assert _is_under_extras_allowed("config/settings.json", extras) is True


def test_is_under_extras_allowed_no_match():
    """Should return False when not under any extras."""
    extras = {"src", "lib"}
    assert _is_under_extras_allowed("tests/test.py", extras) is False


def test_is_under_extras_allowed_deep_nesting():
    """Should handle deeply nested extras directories."""
    extras = {"a/b/c"}
    assert _is_under_extras_allowed("a/b/c/file.txt", extras) is True
    assert _is_under_extras_allowed("a/b/d/file.txt", extras) is False


# -----------------------------------------------------------------------------
# Tests for _filter_plan_for_extras
# -----------------------------------------------------------------------------

def test_filter_plan_for_extras_removes_deletes():
    """Should filter out delete operations in extras-allowed dirs."""
    plan = PlanResult(
        steps=[
            PlanStep("delete", "src/extra.py", "extra"),
            PlanStep("create", "src/new.py", "missing"),
        ],
        add=1, change=0, delete=1, delete_skipped=0,
    )
    extras = {"src"}

    result = _filter_plan_for_extras(plan, extras)

    ops = [(s.op, s.path) for s in result.steps]
    assert ("delete", "src/extra.py") not in ops
    assert ("create", "src/new.py") in ops
    assert result.delete == 0
    assert result.delete_skipped == 1


def test_filter_plan_for_extras_keeps_non_delete():
    """Should keep create/update operations even in extras dirs."""
    plan = PlanResult(
        steps=[
            PlanStep("create", "src/file.py", "missing"),
            PlanStep("update", "src/other.py", "drift"),
            PlanStep("mkdir", "src/new", "missing"),
        ],
        add=2, change=1, delete=0, delete_skipped=0,
    )
    extras = {"src"}

    result = _filter_plan_for_extras(plan, extras)
    assert len(result.steps) == 3


def test_filter_plan_for_extras_keeps_deletes_outside():
    """Should keep delete operations outside extras directories."""
    plan = PlanResult(
        steps=[
            PlanStep("delete", "tests/old.py", "extra"),
        ],
        add=0, change=0, delete=1, delete_skipped=0,
    )
    extras = {"src"}

    result = _filter_plan_for_extras(plan, extras)
    assert len(result.steps) == 1
    assert result.delete == 1


def test_filter_plan_for_extras_empty_extras():
    """Should keep all operations when no extras."""
    plan = PlanResult(
        steps=[
            PlanStep("delete", "extra.txt", "extra"),
        ],
        add=0, change=0, delete=1, delete_skipped=0,
    )

    result = _filter_plan_for_extras(plan, set())
    assert len(result.steps) == 1
    assert result.delete == 1


# -----------------------------------------------------------------------------
# Tests for _filter_nodes_for_planning
# -----------------------------------------------------------------------------

def test_filter_nodes_for_planning_removes_marker_nodes():
    """Should remove ... marker nodes."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/..."), False),
        Node(Path("src/file.py"), False),
    ]
    result = _filter_nodes_for_planning(nodes)
    paths = [n.relpath.as_posix() for n in result]
    assert "src/..." not in paths
    assert "src" in paths
    assert "src/file.py" in paths


def test_filter_nodes_for_planning_removes_template_markers():
    """Should remove template annotation nodes."""
    nodes = [
        Node(Path("files"), True),
        Node(Path("files/<id>"), True, annotation="template:id"),
        Node(Path("files/<id>/data.json"), False),
    ]
    result = _filter_nodes_for_planning(nodes)
    paths = [n.relpath.as_posix() for n in result]
    assert "files/<id>" not in paths
    assert "files/<id>/data.json" not in paths
    assert "files" in paths


def test_filter_nodes_for_planning_keeps_regular_nodes():
    """Should keep regular nodes without markers."""
    nodes = [
        Node(Path("src"), True),
        Node(Path("src/main.py"), False),
        Node(Path("README.md"), False),
    ]
    result = _filter_nodes_for_planning(nodes)
    assert len(result) == 3


def test_filter_nodes_for_planning_removes_paths_with_template_vars():
    """Should remove paths containing unexpanded template variables."""
    nodes = [
        Node(Path("files/<version>/config.json"), False),
        Node(Path("files/v1/config.json"), False),
    ]
    result = _filter_nodes_for_planning(nodes)
    paths = [n.relpath.as_posix() for n in result]
    assert "files/<version>/config.json" not in paths
    assert "files/v1/config.json" in paths


# -----------------------------------------------------------------------------
# Tests for match (integration)
# -----------------------------------------------------------------------------

def test_match_requires_dangerous_flag(tmp_path):
    """Should raise error without dangerous flag."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("file.txt")

    with pytest.raises(RuntimeError, match="--dangerous"):
        match(str(spec_path), tmp_path, dangerous=False, dry_run=False)


def test_match_allows_dry_run_without_dangerous(tmp_path):
    """Should allow dry_run without dangerous flag."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("file.txt")

    # Should not raise
    result = match(str(spec_path), tmp_path, dangerous=False, dry_run=True)
    assert "created" in result or "dry_run" in str(result)


def test_match_creates_missing_files(tmp_path):
    """Should create files specified in spec."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("newfile.txt")

    result = match(str(spec_path), tmp_path, dangerous=True, lock=False)

    assert (tmp_path / "newfile.txt").exists()
    assert result.get("created", 0) >= 1


def test_match_respects_extras_marker(tmp_path):
    """Should not delete extras in directories with ... marker."""
    # Create extra file
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "extra.py").write_text("extra")

    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""src/
├── main.py
└── ...""")

    result = match(str(spec_path), tmp_path, dangerous=True, lock=False)

    # Extra file should NOT be deleted
    assert (tmp_path / "src" / "extra.py").exists()


def test_match_deletes_extras_without_marker(tmp_path):
    """Should delete extras when no ... marker present."""
    # Create extra file
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "extra.py").write_text("extra")

    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""src/
└── main.py""")

    result = match(str(spec_path), tmp_path, dangerous=True, lock=False)

    # Extra file should be deleted
    assert not (tmp_path / "src" / "extra.py").exists()


def test_match_dry_run_no_changes(tmp_path):
    """Dry run should not modify filesystem."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("newfile.txt")

    result = match(str(spec_path), tmp_path, dangerous=False, dry_run=True, lock=False)

    assert not (tmp_path / "newfile.txt").exists()


def test_match_with_variables(tmp_path):
    """Should expand template variables in spec."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("{{name}}.txt")

    result = match(
        str(spec_path),
        tmp_path,
        dangerous=True,
        vars={"name": "generated"},
        lock=False,
    )

    assert (tmp_path / "generated.txt").exists()


# -----------------------------------------------------------------------------
# Tests for match_plan
# -----------------------------------------------------------------------------

def test_match_plan_returns_plan_result(tmp_path):
    """Should return PlanResult object."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("file.txt")

    result = match_plan(str(spec_path), tmp_path)

    assert hasattr(result, "steps")
    assert hasattr(result, "add")
    assert hasattr(result, "delete")


def test_match_plan_includes_extras_filtering(tmp_path):
    """Should filter deletes in extras-allowed directories."""
    # Create extra file
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "extra.py").write_text("extra")

    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""src/
├── main.py
└── ...""")

    result = match_plan(str(spec_path), tmp_path)

    # Should not have delete for extra.py
    delete_ops = [s for s in result.steps if s.op == "delete" and s.path == "src/extra.py"]
    assert len(delete_ops) == 0


def test_match_plan_with_targets(tmp_path):
    """Should respect target filtering."""
    spec_path = tmp_path / "spec.tree"
    # Parser collapses single-root, so use paths relative to collapsed root
    spec_path.write_text("""project/
├── src/
│   ├── a.py
│   └── b.py
└── lib/
    └── c.py""")

    # Target 'src' because parser collapses 'project/' to root
    result = match_plan(str(spec_path), tmp_path, targets=["src"])

    paths = [s.path for s in result.steps]
    # Should only include src paths
    assert any("src" in p for p in paths)
    # Should not include lib paths
    assert not any("lib" in p for p in paths)


# -----------------------------------------------------------------------------
# Tests for create_from_template
# -----------------------------------------------------------------------------

def test_create_from_template_creates_directory(tmp_path):
    """Should create template instance directory."""
    spec_path = tmp_path / "spec.tree"
    # Parser collapses single-root specs, so template at "." level creates at base
    spec_path.write_text("""project/
├── files/
│   └── <version_id>/
│       └── data.json
└── README.md""")

    result = create_from_template(
        str(spec_path),
        tmp_path,
        {"version_id": "v1"},
    )

    # Template creates at project/files/v1 level
    assert result["created"] >= 2
    assert any("v1" in p for p in result["paths"])


def test_create_from_template_dry_run(tmp_path):
    """Dry run should not create files."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""files/
└── <version_id>/
    └── data.json""")

    (tmp_path / "files").mkdir()

    result = create_from_template(
        str(spec_path),
        tmp_path,
        {"version_id": "v1"},
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert not (tmp_path / "files" / "v1").exists()


def test_create_from_template_no_templates_raises(tmp_path):
    """Should raise when spec has no template patterns."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""src/
└── main.py""")

    with pytest.raises(ValueError, match="No template patterns"):
        create_from_template(str(spec_path), tmp_path, {"name": "value"})


def test_create_from_template_returns_paths(tmp_path):
    """Should return list of created paths."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""project/
├── files/
│   └── <id>/
│       ├── a.txt
│       └── b.txt
└── keep""")

    result = create_from_template(
        str(spec_path),
        tmp_path,
        {"id": "test"},
    )

    # Should have created paths containing 'test'
    assert any("test" in p for p in result["paths"])
    assert any("a.txt" in p for p in result["paths"])
    assert any("b.txt" in p for p in result["paths"])


def test_create_from_template_root_level_template(tmp_path):
    """Should handle template at root level."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""workspace/
├── <project>/
│   ├── src/
│   └── README.md
└── keep""")

    result = create_from_template(
        str(spec_path),
        tmp_path,
        {"project": "myapp"},
    )

    # Should have created project structure
    assert result["created"] >= 1
    assert any("myapp" in p for p in result["paths"])


def test_create_from_template_nested_directories(tmp_path):
    """Should create nested directory structures."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""workspace/
├── modules/
│   └── <module>/
│       ├── src/
│       │   └── main.py
│       └── tests/
│           └── test_main.py
└── keep""")

    result = create_from_template(
        str(spec_path),
        tmp_path,
        {"module": "auth"},
    )

    # Should have created nested structure
    assert result["created"] >= 3
    assert any("auth" in p for p in result["paths"])
    assert any("main.py" in p for p in result["paths"])


def test_create_from_template_skips_extras_markers(tmp_path):
    """Should skip ... markers when creating template instance."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""workspace/
├── data/
│   └── <id>/
│       ├── config.json
│       └── ...
└── keep""")

    result = create_from_template(
        str(spec_path),
        tmp_path,
        {"id": "set1"},
    )

    # Should not create a file named "..."
    assert not any("..." in p for p in result["paths"])
    # Should have created config.json
    assert any("config.json" in p for p in result["paths"])


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------

def test_match_handles_empty_spec(tmp_path):
    """Should handle empty spec file."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("")

    result = match(str(spec_path), tmp_path, dangerous=True, lock=False)
    # Should not raise


def test_match_plan_handles_nonexistent_base(tmp_path):
    """Should handle when base directory content varies."""
    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("file.txt")

    # Base exists but is empty
    result = match_plan(str(spec_path), tmp_path)
    assert result.add >= 1


def test_extras_at_multiple_levels(tmp_path):
    """Should handle ... markers at multiple directory levels."""
    # Create extra files at different levels
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "components").mkdir()
    (tmp_path / "src" / "extra1.py").write_text("e1")
    (tmp_path / "src" / "components" / "extra2.jsx").write_text("e2")

    spec_path = tmp_path / "spec.tree"
    spec_path.write_text("""src/
├── main.py
├── ...
└── components/
    ├── Button.jsx
    └── ...""")

    result = match(str(spec_path), tmp_path, dangerous=True, lock=False)

    # Both extra files should be preserved
    assert (tmp_path / "src" / "extra1.py").exists()
    assert (tmp_path / "src" / "components" / "extra2.jsx").exists()
