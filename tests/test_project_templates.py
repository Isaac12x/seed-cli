from pathlib import Path

from seed_cli.parsers import parse_spec
from seed_cli.project_templates import (
    complete_registered_project_template_names,
    complete_project_template_paths,
    has_template_subtree,
    register_spec_project_templates,
    resolve_registered_project_template,
    resolve_project_template_path,
)


def test_register_spec_project_templates_mirrors_template_tree_in_project_seed(tmp_path):
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    spec_file = spec_dir / "component.tree"
    spec_file.write_text(
        "features/\n"
        "├── <name>/\n"
        "│   └── api/\n"
        "│       └── route.ts\n"
    )

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / ".seed" / "templates" / "specs" / "component.tree"
    assert registered.exists()
    assert registered.read_text() == spec_file.read_text()
    assert result.mirrored_spec == registered


def test_register_spec_project_templates_extracts_subtree_template_at_template_parent(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree"
    assert registered.exists()
    assert registered.read_text() == (
        ".\n"
        "└── <name>/\n"
        "    └── api/\n"
        "        └── route.ts\n"
    )
    assert registered in result.project_templates


def test_register_spec_project_templates_infers_path_line_placeholder_template(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text("features/<name>/api/route.ts\n")

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree"
    assert registered.exists()
    assert registered.read_text() == (
        ".\n"
        "└── <name>/\n"
        "    └── api/\n"
        "        └── route.ts\n"
    )
    assert registered in result.project_templates


def test_register_spec_project_templates_does_not_infer_placeholder_filename_template(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text("features/<name>.ts\n")

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree"
    assert not registered.exists()
    assert result.project_templates == []


def test_register_spec_project_templates_registers_nested_placeholder_templates(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text(
        ".\n"
        "└── features/\n"
        "    └── <domain>/\n"
        "        └── <name>/\n"
        "            └── route.ts\n"
    )

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "domain.tree"
    assert registered.exists()
    assert registered.read_text() == (
        ".\n"
        "└── <domain>/\n"
        "    └── <name>/\n"
        "        └── route.ts\n"
    )
    assert registered in result.project_templates
    nested_registered = tmp_path / ".seed" / "templates" / "project" / "name.tree"
    assert not nested_registered.exists()
    assert nested_registered not in result.project_templates
    assert all(path.exists() for path in result.project_templates)
    assert not (tmp_path / "features" / "<domain>").exists()
    assert not (tmp_path / "features" / "<domain>" / ".seed").exists()


def test_register_spec_project_templates_strips_id_suffix_from_template_storage_name(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text(
        ".\n"
        "└── features/\n"
        "    └── <person_id>/\n"
        "        └── profile.json\n"
    )

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "person.tree"
    old_name = tmp_path / "features" / ".seed" / "templates" / "project" / "person_id.tree"
    assert registered.exists()
    assert not old_name.exists()
    assert registered.read_text() == ".\n└── <person_id>/\n    └── profile.json\n"
    assert registered in result.project_templates
    assert resolve_registered_project_template("person_id", tmp_path / "features") == registered


def test_register_spec_project_templates_preserves_seed_extension_for_subtrees(tmp_path):
    spec_file = tmp_path / "component.seed"
    spec_file.write_text(
        ".\n"
        "└── vendor/\n"
        "    └── <name>/ !service +remote -> https://github.com/acme/repo.git\n"
        "        └── README.md\n"
    )

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "vendor" / ".seed" / "templates" / "project" / "name.seed"
    assert registered.exists()
    assert "!service +remote -> https://github.com/acme/repo.git" in registered.read_text()
    assert registered in result.project_templates


def test_register_filestructure_seed_scopes_duplicate_run_subtemplates_to_parent_paths(tmp_path):
    spec_file = tmp_path / "FILESTRUCTURE.seed"
    spec_file.write_text(
        ".\n"
        "├── project/\n"
        "│   └── RUN_ID/ (some files)\n"
        "│       └── project.json\n"
        "└── agent/\n"
        "    └── RUN_ID/ (some files)\n"
        "        └── agent.json\n"
    )

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    project_template = tmp_path / "project" / ".seed" / "templates" / "project" / "run.seed"
    agent_template = tmp_path / "agent" / ".seed" / "templates" / "project" / "run.seed"
    root_project_template = tmp_path / ".seed" / "templates" / "project" / "project_run.seed"
    root_agent_template = tmp_path / ".seed" / "templates" / "project" / "agent_run.seed"
    assert project_template.exists()
    assert project_template.read_text() == ".\n└── RUN_ID/ (some files)\n    └── project.json\n"
    assert agent_template.exists()
    assert agent_template.read_text() == ".\n└── RUN_ID/ (some files)\n    └── agent.json\n"
    assert not root_project_template.exists()
    assert not root_agent_template.exists()
    assert project_template in result.project_templates
    assert agent_template in result.project_templates
    assert resolve_registered_project_template("RUN_ID", tmp_path / "project") == project_template
    assert resolve_registered_project_template("RUN_ID", tmp_path / "agent") == agent_template


def test_register_spec_project_templates_handles_json_specs(tmp_path):
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

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree"
    assert registered.exists()


def test_register_spec_project_templates_registers_directory_only_placeholder_template(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text("features/\n└── <name>/\n")

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    registered = tmp_path / ".seed" / "templates" / "project" / "name.tree"
    assert has_template_subtree(nodes) is True
    assert (tmp_path / ".seed" / "templates" / "component.tree").exists()
    assert result.mirrored_spec == (tmp_path / ".seed" / "templates" / "component.tree")
    assert registered.exists()
    assert registered.read_text() == ".\n└── <name>/\n"
    assert result.project_templates == [registered]


def test_register_spec_project_templates_skips_non_tree_spec_without_template_children(tmp_path):
    spec_file = tmp_path / "component.json"
    spec_file.write_text('{"entries":[{"path":"features/","type":"dir"}]}')

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    assert result.changed is False
    assert not (tmp_path / ".seed" / "templates" / "component.json").exists()


def test_register_spec_project_templates_deletes_materialized_placeholder_dir(tmp_path):
    spec_file = tmp_path / "component.tree"
    spec_file.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )
    stale_dir = tmp_path / "features" / "<name>" / "api"
    stale_dir.mkdir(parents=True)
    (stale_dir / "route.ts").write_text("legacy")

    _, nodes = parse_spec(str(spec_file), base=tmp_path)
    result = register_spec_project_templates(spec_file, nodes, tmp_path, cleanup_materialized=True)

    assert not (tmp_path / "features" / "<name>").exists()
    assert (tmp_path / "features" / "<name>") in result.deleted_paths


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


def test_resolve_project_template_path_maps_directory_name_to_same_name_tree(tmp_path):
    project_root = tmp_path / "repo"
    nested = project_root / "packages" / "app"
    nested.mkdir(parents=True)
    (project_root / ".git").mkdir()
    template_file = project_root / ".seed" / "templates" / "project" / "project.tree"
    template_file.parent.mkdir(parents=True)
    template_file.write_text("<name>/\n<name>/item.txt\n")

    resolved = resolve_project_template_path("project", nested)

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
