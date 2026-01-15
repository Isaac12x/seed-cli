import pytest
from pathlib import Path
from seed_cli.includes import resolve_includes, IncludeError


def test_simple_include(tmp_path):
    a = tmp_path / "a.tree"
    b = tmp_path / "b.tree"
    b.write_text("x.txt")
    a.write_text("@include b.tree\ny.txt")

    out = resolve_includes(a.read_text(), a)
    assert "x.txt" in out
    assert "y.txt" in out


def test_nested_include(tmp_path):
    a = tmp_path / "a.tree"
    b = tmp_path / "b.tree"
    c = tmp_path / "c.tree"
    c.write_text("c.txt")
    b.write_text("@include c.tree\nb.txt")
    a.write_text("@include b.tree\na.txt")

    out = resolve_includes(a.read_text(), a)
    assert "c.txt" in out
    assert "b.txt" in out
    assert "a.txt" in out


def test_include_cycle(tmp_path):
    a = tmp_path / "a.tree"
    b = tmp_path / "b.tree"
    a.write_text("@include b.tree")
    b.write_text("@include a.tree")

    with pytest.raises(IncludeError):
        resolve_includes(a.read_text(), a)


def test_missing_include(tmp_path):
    a = tmp_path / "a.tree"
    a.write_text("@include missing.tree")

    with pytest.raises(IncludeError):
        resolve_includes(a.read_text(), a)
