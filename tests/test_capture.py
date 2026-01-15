from pathlib import Path
from seed_cli.capture import capture_nodes, to_tree_text, to_json


def test_capture_basic(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a/file.txt").write_text("x")

    nodes = capture_nodes(tmp_path)
    paths = {n.relpath.as_posix() for n in nodes}
    assert "a" in paths
    assert "a/file.txt" in paths


def test_capture_ignores_seed(tmp_path):
    (tmp_path / ".seed").mkdir()
    (tmp_path / ".seed/state.json").write_text("{}")

    nodes = capture_nodes(tmp_path)
    assert not nodes


def test_to_tree_text(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a/file.txt").write_text("x")
    nodes = capture_nodes(tmp_path)
    text = to_tree_text(nodes)
    assert "a/" in text
    assert "a/file.txt" in text


def test_to_json(tmp_path):
    (tmp_path / "a").mkdir()
    nodes = capture_nodes(tmp_path)
    j = to_json(nodes)
    assert "entries" in j
    assert "a/" in j
