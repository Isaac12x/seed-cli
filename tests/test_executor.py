from pathlib import Path
from seed_cli.executor import execute_plan
from seed_cli.planning import PlanResult, PlanStep
from seed_cli.checksums import load_checksums


def test_executor_create_and_mkdir(tmp_path):
    plan = PlanResult(
        steps=[
            PlanStep("mkdir", "a", "missing"),
            PlanStep("create", "a/file.txt", "missing"),
        ],
        add=2, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path)
    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "a/file.txt").exists()
    assert res["created"] == 2

    checks = load_checksums(tmp_path)
    assert "a/file.txt" in checks


def test_executor_update(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("old")

    plan = PlanResult(
        steps=[PlanStep("update", "x.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=True)
    assert res["updated"] == 1
    assert load_checksums(tmp_path)["x.txt"]["sha256"]


def test_executor_skip(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("skip", "x.txt", "manual")],
        add=0, change=0, delete=0, delete_skipped=1,
    )
    res = execute_plan(plan, tmp_path)
    assert res["skipped"] == 1


def test_executor_delete_requires_dangerous(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")

    plan = PlanResult(
        steps=[PlanStep("delete", "x.txt", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )

    try:
        execute_plan(plan, tmp_path)
        assert False, "delete should have failed"
    except RuntimeError:
        pass


def test_executor_delete(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")

    plan = PlanResult(
        steps=[PlanStep("delete", "x.txt", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, dangerous=True)
    assert not f.exists()
    assert res["deleted"] == 1


def test_executor_gitkeep(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("mkdir", "d", "missing")],
        add=1, change=0, delete=0, delete_skipped=0,
    )

    execute_plan(plan, tmp_path, gitkeep=True)
    assert (tmp_path / "d/.gitkeep").exists()


def test_executor_dry_run(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("create", "x.txt", "missing")],
        add=1, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, dry_run=True)
    assert not (tmp_path / "x.txt").exists()
    assert res["created"] == 1
