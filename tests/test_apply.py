from pathlib import Path
import json
from unittest.mock import patch
from seed_cli.apply import apply
from seed_cli.planning import PlanResult, PlanStep


def test_apply_from_spec(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("a/file.txt")
    res = apply(str(spec), tmp_path, dry_run=False)
    assert (tmp_path / "a/file.txt").exists()
    assert res["created"] >= 1


def test_apply_from_plan_json(tmp_path):
    plan = {
        "summary": {"add": 1, "change": 0, "delete": 0, "delete_skipped": 0},
        "steps": [
            {"op": "create", "path": "x.txt", "reason": "missing", "annotation": None, "depends_on": None, "note": None}
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    res = apply(str(plan_path), tmp_path)
    assert (tmp_path / "x.txt").exists()
    assert res["created"] == 1


def test_apply_dry_run(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("x.txt")
    res = apply(str(spec), tmp_path, dry_run=True)
    assert not (tmp_path / "x.txt").exists()
    assert res["created"] == 1


def test_apply_ignores_extras_when_deletions_disabled(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("tracked/")
    (tmp_path / "tracked").mkdir()
    (tmp_path / "extra.txt").write_text("x")

    res = apply(str(spec), tmp_path, dry_run=True)

    assert res["skipped"] == 0


def test_apply_does_not_scan_entire_tree_when_deletions_disabled(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("tracked/")
    (tmp_path / "tracked").mkdir()
    (tmp_path / "tracked" / "nested").mkdir()
    (tmp_path / "tracked" / "nested" / "extra.txt").write_text("x")

    with patch("pathlib.Path.rglob", side_effect=AssertionError("rglob should not be called")):
        res = apply(str(spec), tmp_path, dry_run=False)

    assert res["created"] == 0
    assert res["skipped"] == 0


def test_apply_handles_unicode_guides_and_at_prefixed_dirs(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "├── tracking/\n"
        "│   └── status/\n"
        "│\n"
        "├── @userfiles/\n"
        "│   └── incoming/\n"
        "└── archive/\n"
    )

    res = apply(str(spec), tmp_path, dry_run=False)

    assert (tmp_path / "tracking" / "status").is_dir()
    assert (tmp_path / "@userfiles" / "incoming").is_dir()
    assert (tmp_path / "archive").is_dir()
    assert not (tmp_path / "│").exists()
    assert res["created"] >= 4


def test_apply_regression_root_entries_before_archive_do_not_flatten(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "├── work/\n"
        "│   └── projects/\n"
        "│       └── active/\n"
        "├── personal/\n"
        "│   └── archive/\n"
        "├── tracking/\n"
        "│   └── status/\n"
        "│\n"
        "├── @userfiles/\n"
        "│   └── incoming/\n"
        "├── systems/\n"
        "│   └── services/\n"
        "├── archive/\n"
        "│   └── work/\n"
        "└── scratch/\n"
        "    └── temp/\n"
    )

    apply(str(spec), tmp_path, dry_run=False)

    assert (tmp_path / "work" / "projects" / "active").is_dir()
    assert (tmp_path / "personal" / "archive").is_dir()
    assert (tmp_path / "tracking" / "status").is_dir()
    assert (tmp_path / "@userfiles" / "incoming").is_dir()
    assert (tmp_path / "systems" / "services").is_dir()
    assert (tmp_path / "archive" / "work").is_dir()
    assert (tmp_path / "scratch" / "temp").is_dir()

    assert not (tmp_path / "│").exists()
    assert not (tmp_path / "active").exists()
    assert not (tmp_path / "incoming").exists()
    assert not (tmp_path / "services").exists()


def test_apply_registers_nested_project_template_without_materializing_template_subtree(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )

    res = apply(str(spec), tmp_path, dry_run=False)

    assert (tmp_path / "features").is_dir()
    assert not (tmp_path / "features" / "<name>").exists()
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree").exists()
    assert (tmp_path / ".seed" / "templates" / "spec.tree").exists()
    assert res["created"] == 1


def test_apply_deletes_stale_materialized_template_subtree_before_execution(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )
    stale_dir = tmp_path / "features" / "<name>" / "api"
    stale_dir.mkdir(parents=True)
    (stale_dir / "route.ts").write_text("legacy")

    res = apply(str(spec), tmp_path, dry_run=False)

    assert not (tmp_path / "features" / "<name>").exists()
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree").exists()
    assert res["deleted"] == 1
