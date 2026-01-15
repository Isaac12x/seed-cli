

"""seed_cli.doctor

Spec linter and auto-repair tool.

Detects:
- duplicate paths
- parent-as-file conflicts
- file/dir collisions
- invalid annotations

Optionally fixes:
- removes duplicates
- normalizes directory paths
- reports unfixable issues

This operates purely on spec Nodes (no filesystem mutation).
"""

from pathlib import Path
from typing import List, Dict, Tuple
from .parsers import Node

VALID_ANNOTATIONS = {"manual", "generated"}


def doctor(nodes: List[Node], base: Path, fix: bool = False) -> List[str]:
    issues: List[str] = []

    seen: Dict[str, Node] = {}
    fixed_nodes: List[Node] = []

    for n in nodes:
        rel = n.relpath.as_posix().rstrip("/")

        # duplicate path
        if rel in seen:
            issues.append(f"duplicate: {rel}")
            if fix:
                continue
        else:
            seen[rel] = n

        # invalid annotation
        if n.annotation and n.annotation not in VALID_ANNOTATIONS:
            issues.append(f"invalid annotation @{n.annotation} on {rel}")
            if fix:
                n = Node(n.relpath, n.is_dir, n.comment, None)

        fixed_nodes.append(n)

    # parent-as-file conflict
    paths = {n.relpath.as_posix(): n for n in fixed_nodes}
    for p, n in paths.items():
        parent = Path(p).parent
        if parent.as_posix() == ".":
            continue
        parent_str = parent.as_posix()
        if parent_str in paths and not paths[parent_str].is_dir:
            issues.append(f"parent is file: {parent_str} blocks {p}")

    if fix:
        nodes[:] = fixed_nodes

    return issues
