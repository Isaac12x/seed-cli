import json
from pathlib import Path

import pytest

from seed_cli.apply import apply
from seed_cli.executor import execute_plan
from seed_cli.planning import PlanResult, PlanStep
from seed_cli.security import normalize_relpath, safe_target_path


def test_execute_plan_rejects_parent_escape(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    plan = PlanResult(
        steps=[PlanStep(op="create", path="../escaped.txt", reason="test")],
        add=1,
        change=0,
        delete=0,
        delete_skipped=0,
    )

    with pytest.raises(ValueError, match="Parent traversal is not allowed"):
        execute_plan(plan, base)

    assert not (tmp_path / "escaped.txt").exists()


def test_execute_plan_rejects_absolute_path(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    plan = PlanResult(
        steps=[PlanStep(op="create", path="/tmp/escaped.txt", reason="test")],
        add=1,
        change=0,
        delete=0,
        delete_skipped=0,
    )

    with pytest.raises(ValueError, match="Absolute path is not allowed"):
        execute_plan(plan, base)


def test_apply_rejects_unsafe_plan_json(tmp_path):
    base = tmp_path / "base"
    base.mkdir()

    plan = {
        "summary": {"add": 1, "change": 0, "delete": 0, "delete_skipped": 0},
        "steps": [
            {
                "op": "create",
                "path": "../escape.txt",
                "reason": "test",
                "annotation": None,
                "depends_on": None,
                "note": None,
                "optional": False,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="Parent traversal is not allowed"):
        apply(str(plan_path), base, dangerous=False)


def test_normalize_relpath_and_safe_target(tmp_path):
    assert normalize_relpath("./a//b.txt") == "a/b.txt"

    with pytest.raises(ValueError, match="Absolute path"):
        normalize_relpath("/tmp/x")
    with pytest.raises(ValueError, match="Parent traversal"):
        normalize_relpath("../x")
    with pytest.raises(ValueError, match="NUL"):
        normalize_relpath("a\x00b")

    base = (tmp_path / "base")
    base.mkdir()
    target = safe_target_path(base, "nested/file.txt")
    assert target == (base / "nested/file.txt").resolve()


def test_safe_target_path_rejects_symlink_escape(tmp_path):
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()

    link = base / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="escapes base directory"):
        safe_target_path(base, "link/file.txt")
