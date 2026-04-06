from pathlib import Path

from seed_cli.parsers import parse_spec
from seed_cli.project_templates import (
    complete_registered_project_template_names,
    complete_project_template_paths,
    has_template_subtree,
    resolve_registered_project_template,
    resolve_project_template_path,
)


def test_parse_spec_registers_template_tree_in_project_seed(tmp_path):
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_file = spec_dir / "component.tree"
    spec_file.write_text(
        "features/\n"
        "├── <name>/\n"
        "│   └── api/\n"
        "│       └── route.ts\n"
    )

    parse_spec(str(spec_file), base=tmp_path)

    registered = tmp_path / ".seed" / "templates" / "specs" / "component.tree"
    assert registered.exists()
    assert registered.read_text() == spec_file.read_text()


def test_parse_spec_registers_subtree_template_at_template_parent(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )

    parse_spec(str(spec_file), base=tmp_path)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree"
    assert registered.exists()
    assert registered.read_text() == (
        ".\n"
        "└── <name>/\n"
        "    └── api/\n"
        "        └── route.ts\n"
    )


def test_parse_json_registers_subtree_template_at_template_parent(tmp_path):
    spec_file = tmp_path / "component.json"
    spec_file.write_text(
        "{\n"
        '  "entries": [\n'
        '    {"path": "features/", "type": "dir"},\n'
        '    {"path": "features/<name>/", "type": "dir", "annotation": "template:name"},\n'
        '    {"path": "features/<name>/api/", "type": "dir"},\n'
        '    {"path": "features/<name>/api/route.ts", "type": "file"}\n'
        "  ]\n"
        "}\n"
    )

    parse_spec(str(spec_file), base=tmp_path)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree"
    assert registered.exists()


def test_parse_spec_skips_registration_without_template_children(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text("features/\n└── <name>/\n")

    _, nodes = parse_spec(str(spec_file), base=tmp_path)

    assert has_template_subtree(nodes) is False
    assert not (tmp_path / ".seed" / "templates" / "component.tree").exists()


def test_resolve_project_template_path_uses_top_level_seed(tmp_path):
    project_root = tmp_path / "repo"
    nested = project_root / "packages" / "app"
    nested.mkdir(parents=True)
    (project_root / ".git").mkdir()
    template_file = project_root / ".seed" / "templates" / "spec.tree"
    template_file.parent.mkdir(parents=True)
    template_file.write_text("files/\n└── <name>/\n    └── item.txt\n")

    resolved = resolve_project_template_path(".seed/templates/spec.tree", nested)

    assert resolved == template_file


def test_complete_project_template_paths_uses_top_level_seed(tmp_path):
    project_root = tmp_path / "repo"
    nested = project_root / "packages" / "app"
    nested.mkdir(parents=True)
    (project_root / ".git").mkdir()
    template_dir = project_root / ".seed" / "templates" / "nested"
    template_dir.mkdir(parents=True)
    (template_dir / "spec.tree").write_text("files/\n└── <name>/\n    └── item.txt\n")

    suggestions = complete_project_template_paths(".seed/", nested)

    assert ".seed/templates/" in suggestions
    assert ".seed/templates/nested/spec.tree" in suggestions


def test_resolve_registered_project_template_uses_nearest_local_seed(tmp_path):
    project_root = tmp_path / "repo"
    nested = project_root / "features" / "api"
    nested.mkdir(parents=True)
    (project_root / ".git").mkdir()
    root_template = project_root / ".seed" / "templates" / "project" / "root.tree"
    root_template.parent.mkdir(parents=True)
    root_template.write_text("<root>/\n<root>/item.txt\n")
    local_template = project_root / "features" / ".seed" / "templates" / "project" / "name.tree"
    local_template.parent.mkdir(parents=True)
    local_template.write_text("<name>/\n<name>/item.txt\n")

    resolved = resolve_registered_project_template("name", nested)

    assert resolved == local_template


def test_complete_registered_project_template_names_lists_visible_names(tmp_path):
    project_root = tmp_path / "repo"
    nested = project_root / "features" / "api"
    nested.mkdir(parents=True)
    local_template = project_root / "features" / ".seed" / "templates" / "project" / "name.tree"
    local_template.parent.mkdir(parents=True)
    local_template.write_text("<name>/\n<name>/item.txt\n")

    suggestions = complete_registered_project_template_names("", nested)

    assert "name" in suggestions
