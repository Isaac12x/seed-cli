from pathlib import Path
from seed_cli.planning import plan
from seed_cli.parsers import Node
from seed_cli.checksums import save_checksums, sha256


def test_plan_adds_missing_files_and_dirs(tmp_path):
    nodes = [
        Node(Path("a"), True),
        Node(Path("a/file.txt"), False),
        Node(Path("b/file2.txt"), False),
    ]
    res = plan(nodes, tmp_path, allow_delete=False)
    ops = {(s.op, s.path) for s in res.steps}
    assert ("mkdir", "a") in ops
    assert ("create", "a/file.txt") in ops
    assert ("create", "b/file2.txt") in ops
    assert res.add >= 3


def test_plan_orders_parents_before_children(tmp_path):
    nodes = [Node(Path("a/file.txt"), False)]
    res = plan(nodes, tmp_path)
    paths = [s.path for s in res.steps]
    assert "a" in paths
    assert paths.index("a") < paths.index("a/file.txt")


def test_plan_skips_delete_when_not_allowed(tmp_path):
    (tmp_path / "extra.txt").write_text("x")
    res = plan([], tmp_path, allow_delete=False)
    assert any(s.op == "skip" and s.path == "extra.txt" and s.reason == "extra" for s in res.steps)
    assert res.delete == 0
    assert res.delete_skipped == 1


def test_plan_emits_delete_when_allowed(tmp_path):
    (tmp_path / "extra.txt").write_text("x")
    res = plan([], tmp_path, allow_delete=True)
    assert any(s.op == "delete" and s.path == "extra.txt" for s in res.steps)
    assert res.delete == 1


def test_plan_checksum_drift_update(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("old")
    save_checksums(tmp_path, {"x.txt": {"sha256": sha256(f), "annotation": None}})

    f.write_text("new")

    nodes = [Node(Path("x.txt"), False, annotation="generated")]
    res = plan(nodes, tmp_path)
    assert any(s.op == "update" and s.path == "x.txt" and s.reason == "checksum_drift" for s in res.steps)


def test_plan_checksum_drift_manual_skips(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("old")
    save_checksums(tmp_path, {"x.txt": {"sha256": sha256(f), "annotation": "manual"}})
    f.write_text("new")

    nodes = [Node(Path("x.txt"), False, annotation="manual")]
    res = plan(nodes, tmp_path)
    step = next(s for s in res.steps if s.path == "x.txt")
    assert step.op == "skip"
    assert step.reason == "checksum_drift"
    assert step.note and "manual" in step.note


def test_plan_targets_scope(tmp_path):
    nodes = [
        Node(Path("a/file.txt"), False),
        Node(Path("b/file.txt"), False),
    ]
    res = plan(nodes, tmp_path, targets=["a"])
    paths = {s.path for s in res.steps}
    assert "a/file.txt" in paths
    assert "b/file.txt" not in paths


def test_plan_targets_closure_adds_parent(tmp_path):
    nodes = [Node(Path("a/b/c.txt"), False)]
    res = plan(nodes, tmp_path, targets=["a/b/c.txt"], target_mode="exact")
    paths = [s.path for s in res.steps]
    assert "a" in paths
    assert "a/b" in paths
    assert paths.index("a") < paths.index("a/b") < paths.index("a/b/c.txt")
