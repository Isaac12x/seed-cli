

"""seed_cli.capture

Capture the current filesystem into a spec representation.

Supports:
- tree text output
- JSON structured output

Used for:
- bootstrapping specs from existing projects
- drift inspection snapshots
"""

from pathlib import Path
from typing import List, Dict, Iterable
import fnmatch
import json

from .parsers import Node

DEFAULT_IGNORE = [
    ".git/**",
    ".seed/**",
    "**/.DS_Store",
    "**/__pycache__/**",
]


def _ignored(rel: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def capture_nodes(base: Path, ignore: List[str] | None = None) -> List[Node]:
    """Capture filesystem under base into Node list."""
    ignore_patterns = (ignore or []) + DEFAULT_IGNORE
    nodes: List[Node] = []

    if not base.exists():
        return nodes

    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base).as_posix()
        if not rel:
            continue
        # Check if path or any parent matches ignore pattern
        parts = rel.split("/")
        ignored = False
        for i in range(len(parts)):
            check_path = "/".join(parts[:i+1])
            if _ignored(check_path, ignore_patterns) or _ignored(check_path + "/", ignore_patterns):
                ignored = True
                break
        if ignored:
            continue
        nodes.append(Node(Path(rel), is_dir=p.is_dir()))

    return nodes


def to_tree_text(nodes: List[Node]) -> str:
    """Render nodes to simple tree-text format."""
    lines: List[str] = []
    for n in sorted(nodes, key=lambda n: n.relpath.as_posix()):
        suffix = "/" if n.is_dir else ""
        lines.append(f"{n.relpath.as_posix()}{suffix}")
    return "\n".join(lines)


def to_json(nodes: List[Node]) -> str:
    """Render nodes to structured JSON spec."""
    entries: List[Dict[str, str]] = []
    for n in nodes:
        entries.append({
            "path": n.relpath.as_posix() + ("/" if n.is_dir else ""),
            "type": "dir" if n.is_dir else "file",
        })
    return json.dumps({"entries": entries}, indent=2)
