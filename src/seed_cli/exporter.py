

"""seed_cli.exporter

Export specs, plans, and state into external artifacts.
"""

from pathlib import Path
from typing import List
import json

from .parsers import Node
from .planning import PlanResult
from .graphviz import plan_to_dot
from .capture import to_tree_text


def export_tree(nodes: List[Node], out: Path) -> None:
    out.write_text(to_tree_text(nodes), encoding="utf-8")


def export_json_spec(nodes: List[Node], out: Path) -> None:
    entries = []
    for n in nodes:
        entries.append({
            "path": n.relpath.as_posix() + ("/" if n.is_dir else ""),
            "type": "dir" if n.is_dir else "file",
            "annotation": n.annotation,
            "comment": n.comment,
        })
    out.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")


def export_plan(plan: PlanResult, out: Path) -> None:
    out.write_text(json.dumps(plan.to_json(), indent=2), encoding="utf-8")


def export_dot(plan: PlanResult, out: Path) -> None:
    out.write_text(plan_to_dot(plan), encoding="utf-8")
