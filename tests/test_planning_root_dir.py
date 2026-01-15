from pathlib import Path
from seed_cli.planning import plan as build_plan
from seed_cli.parsers import Node

def test_plan_strips_root_if_base_is_root(tmp_path: Path):
    # Simulate being inside offlineimap dir
    base = tmp_path / "offlineimap"
    base.mkdir()

    spec_nodes = [
        Node(Path("offlineimap"), is_dir=True),
        Node(Path("offlineimap/hooks"), is_dir=True),
        Node(Path("offlineimap/hooks/postsync.py"), is_dir=False),
    ]

    res = build_plan(spec_nodes, base)

    # Should NOT contain steps for "offlineimap" prefix at all
    for s in res.steps:
        assert not s.path.startswith("offlineimap")



def test_plan_creates_missing_root_first(tmp_path: Path):
    base = tmp_path  # user outside root
    spec_nodes = [
        Node(Path("offlineimap"), is_dir=True),
        Node(Path("offlineimap/hooks"), is_dir=True),
    ]

    res = build_plan(spec_nodes, base)

    # first mkdir should be the root (mkdir ordering exists in your rank())
    mkdirs = [s for s in res.steps if s.op == "mkdir"]
    assert mkdirs, "expected at least one mkdir"
    assert mkdirs[0].path == "offlineimap"


def test_plan_inside_root_with_tilde(tmp_path, monkeypatch):
    home = tmp_path
    base = home / ".config" / "offlineimap"
    base.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))

    spec_nodes = [
        Node(Path("~/.config/offlineimap"), is_dir=True),
        Node(Path("~/.config/offlineimap/data"), is_dir=True),
    ]

    res = build_plan(spec_nodes, base)

    assert not any(s.path.startswith("~") for s in res.steps)
    assert not any("offlineimap" in s.path for s in res.steps)


def test_nested_paths_preserved_when_root_collapses(tmp_path):
    base = tmp_path / "offlineimap"
    base.mkdir()

    spec_nodes = [
        Node(Path("offlineimap"), is_dir=True),
        Node(Path("offlineimap/hooks"), is_dir=True),
        Node(Path("offlineimap/hooks/postsync.py"), is_dir=False),
    ]

    plan = build_plan(spec_nodes, base)

    paths = {s.path for s in plan.steps}
    assert "hooks/postsync.py" in paths
    assert "postsync.py" not in paths
