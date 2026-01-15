from seed_cli.graphviz import plan_to_dot
from seed_cli.planning import PlanResult, PlanStep


def test_plan_to_dot_basic():
    plan = PlanResult(
        steps=[
            PlanStep("mkdir", "a", "missing"),
            PlanStep("create", "a/file.txt", "missing", depends_on=["a"]),
        ],
        add=2,
        change=0,
        delete=0,
        delete_skipped=0,
    )

    dot = plan_to_dot(plan)
    assert "digraph seed_plan" in dot
    assert "a_file_txt" in dot
    assert '"a" -> "a_file_txt"' in dot


def test_plan_to_dot_annotations():
    plan = PlanResult(
        steps=[
            PlanStep("update", "x.txt", "checksum_drift", annotation="manual"),
        ],
        add=0,
        change=1,
        delete=0,
        delete_skipped=0,
    )

    dot = plan_to_dot(plan)
    assert "@manual" in dot
    assert "checksum_drift" in dot
