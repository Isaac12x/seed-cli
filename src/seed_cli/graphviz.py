

"""seed_cli.graphviz

Graphviz (.dot) export for execution plans.

This produces a directed acyclic graph:
- Nodes are paths (directories/files)
- Edges represent dependencies (depends_on)
- Node labels include operation and annotations

Intended for:
  seed plan --dot > plan.dot
  dot -Tpng plan.dot -o plan.png
"""

from typing import List
from pathlib import Path

from .planning import PlanResult, PlanStep


def _node_id(path: str) -> str:
    # Graphviz-safe identifier
    return path.replace("/", "_").replace(".", "_") or "root"


def plan_to_dot(plan: PlanResult) -> str:
    """Convert a PlanResult to Graphviz DOT format."""
    lines: List[str] = []
    lines.append("digraph seed_plan {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, fontname=Helvetica];")

    # Emit nodes
    for step in plan.steps:
        nid = _node_id(step.path)
        label = f"{step.op}\n{step.path}"
        if step.annotation:
            label += f"\n@{step.annotation}"
        if step.reason:
            label += f"\n({step.reason})"
        lines.append(f'  "{nid}" [label="{label}"];')

    # Emit edges
    for step in plan.steps:
        if not step.depends_on:
            continue
        nid = _node_id(step.path)
        for dep in step.depends_on:
            did = _node_id(dep)
            lines.append(f'  "{did}" -> "{nid}";')

    lines.append("}")
    return "\n".join(lines)
