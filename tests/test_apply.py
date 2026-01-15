from pathlib import Path
import json
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
