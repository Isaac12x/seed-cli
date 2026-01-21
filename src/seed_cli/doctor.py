

"""seed_cli.doctor

Spec linter and auto-repair tool.

Detects:
- duplicate paths
- parent-as-file conflicts
- file/dir collisions
- invalid annotations
- potential directories without trailing /

Optionally fixes:
- removes duplicates
- normalizes directory paths
- converts potential dirs to dirs (when --fix)
- reports unfixable issues

This operates purely on spec Nodes (no filesystem mutation).
"""

from pathlib import Path
from typing import List, Dict, Tuple, Set
from .parsers import Node

VALID_ANNOTATIONS = {"manual", "generated", "extras", "template"}

# Common file extensions - items without these might be directories
COMMON_EXTENSIONS = {
    ".txt", ".md", ".json", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".css", ".js", ".ts", ".jsx", ".tsx", ".py", ".rb", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".proto", ".toml", ".ini", ".cfg", ".conf",
    ".env", ".gitignore", ".dockerignore", ".editorconfig",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".lock", ".log", ".tree", ".dot",
}


def _looks_like_directory(name: str) -> bool:
    """Check if a name looks like it might be a directory (no extension)."""
    # Skip special markers
    if name in ("...", "…") or name.startswith("<"):
        return False

    path = Path(name)
    suffix = path.suffix.lower()

    # Has a recognized extension - likely a file
    if suffix in COMMON_EXTENSIONS:
        return False

    # No extension at all - likely a directory
    if not suffix:
        return True

    # Has some extension but it's not common - could be either
    # Be conservative and don't flag these
    return False


def doctor(nodes: List[Node], base: Path, fix: bool = False) -> List[str]:
    issues: List[str] = []

    seen: Dict[str, Node] = {}
    fixed_nodes: List[Node] = []

    # Track which paths have children (definitely directories)
    has_children: Set[str] = set()
    for n in nodes:
        parent = Path(n.relpath).parent.as_posix()
        if parent != ".":
            has_children.add(parent)

    for n in nodes:
        rel = n.relpath.as_posix().rstrip("/")

        # duplicate path
        if rel in seen:
            issues.append(f"duplicate: {rel}")
            if fix:
                continue
        else:
            seen[rel] = n

        # invalid annotation (but allow template annotations)
        if n.annotation and not n.annotation.startswith("template:"):
            if n.annotation not in VALID_ANNOTATIONS:
                issues.append(f"invalid annotation @{n.annotation} on {rel}")
                if fix:
                    n = Node(n.relpath, n.is_dir, n.comment, None)

        # potential directory without trailing /
        if not n.is_dir and _looks_like_directory(Path(rel).name):
            # Check if this path has children in the spec (definitely a dir)
            if rel in has_children:
                issues.append(f"potential directory: {rel} (has children, should end with /)")
                if fix:
                    n = Node(n.relpath, is_dir=True, comment=n.comment, annotation=n.annotation)
            else:
                # Just warn, don't auto-fix since we're not sure
                issues.append(f"potential directory: {rel} (no extension, consider adding /)")

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
