from pathlib import Path
from seed_cli.diff import diff
from seed_cli.parsers import Node
from seed_cli.checksums import save_checksums, sha256


def test_diff_missing(tmp_path):
    nodes = [Node(Path("a.txt"), False)]
    res = diff(nodes, tmp_path)
    assert res.missing == ["a.txt"]
    assert res.extra == []
    assert res.type_mismatch == []
    assert res.drift == []


def test_diff_extra(tmp_path):
    (tmp_path / "x.txt").write_text("x")
    res = diff([], tmp_path)
    assert res.extra == ["x.txt"]
    assert res.missing == []


def test_diff_type_mismatch(tmp_path):
    # fs has dir, spec wants file
    (tmp_path / "p").mkdir()
    nodes = [Node(Path("p"), False)]
    res = diff(nodes, tmp_path)
    assert res.type_mismatch == ["p"]
    assert res.drift == []


def test_diff_drift_only_with_recorded_checksum(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("one")
    # no checksum recorded => no drift
    nodes = [Node(Path("a.txt"), False)]
    res = diff(nodes, tmp_path)
    assert res.drift == []

    # record checksum then change => drift
    save_checksums(tmp_path, {"a.txt": {"sha256": sha256(f), "annotation": None}})
    f.write_text("two")
    res2 = diff(nodes, tmp_path)
    assert res2.drift == ["a.txt"]


def test_diff_ignores_seed_and_git(tmp_path):
    (tmp_path / ".seed").mkdir()
    (tmp_path / ".seed" / "state.json").write_text("{}")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    res = diff([], tmp_path)
    assert res.extra == []
