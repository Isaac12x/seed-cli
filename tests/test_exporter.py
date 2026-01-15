from pathlib import Path
from seed_cli.exporter import export_tree, export_json_spec, export_plan, export_dot
from seed_cli.parsers import Node
from seed_cli.planning import PlanStep, PlanResult


def test_export_tree(tmp_path):
    nodes = [Node(Path("a"), True), Node(Path("a/file.txt"), False)]
    out = tmp_path / "spec.tree"
    export_tree(nodes, out)
    assert "a/" in out.read_text()


def test_export_json_spec(tmp_path):
    nodes = [Node(Path("a"), True)]
    out = tmp_path / "spec.json"
    export_json_spec(nodes, out)
    assert "entries" in out.read_text()


def test_export_plan_and_dot(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("mkdir", "a", "missing")],
        add=1,
        change=0,
        delete=0,
        delete_skipped=0,
    )
    pj = tmp_path / "plan.json"
    pd = tmp_path / "plan.dot"
    export_plan(plan, pj)
    export_dot(plan, pd)
    assert "mkdir" in pj.read_text()
    assert "digraph" in pd.read_text()
