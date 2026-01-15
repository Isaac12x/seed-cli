

"""Parsing layer for seed-cli.

Supports:
- Tree text specs (ASCII tree or simple path-per-line)
- YAML / JSON structured specs
- stdin support ("-")
- Comments and annotations
- Variable templating ({{var}})
- Includes (@include)

Outputs Node(relpath, is_dir, comment, annotation).
"""

import os
import json
import yaml
import re
import inspect

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from .schema import validate_document
from .templating import apply_vars
from .includes import resolve_includes


@dataclass
class Node:
    relpath: Path
    is_dir: bool
    comment: Optional[str] = None
    annotation: Optional[str] = None  # generated | manual | None


# _COMMENT_RE matches comments in parentheses (e.g., (note here)), and ignores # style comments in a line.
_COMMENT_RE = re.compile(r"\(([^)]+)\)|//(.*)$|#(.*)$")
_ANNOT_RE = re.compile(r"@([a-zA-Z_][\w-]*)")
TREE_LINE = re.compile(r"""
^(?P<prefix>[\s│]*)(?P<branch>├──|└──)?\s*(?P<name>.+?)\s*$
""", re.VERBOSE)

def _tree_depth(prefix: str) -> int:
    """
    Each indentation level in `tree` output is typically 4 chars: '│   ' or '    '
    We'll treat any group of 4 columns as one depth.
    """
    # Normalize tabs just in case
    prefix = prefix.replace("\t", "    ")
    return len(prefix) // 4

def _extract_comment_and_annotation(text: str) -> tuple[str, Optional[str], Optional[str]]:
    comment = None
    annotation = None

    ann = _ANNOT_RE.search(text)
    if ann:
        annotation = ann.group(1)
        text = _ANNOT_RE.sub("", text)

    com = _COMMENT_RE.search(text)
    if com:
        comment = com.group(1).strip()
        text = _COMMENT_RE.sub("", text)

    return text.strip(), comment, annotation

def _make_node(*, rel: str, is_dir: bool, comment: str | None = None, annotation: str | None = None):
    """
    Construct Node with proper Path for relpath.
    """
    return Node(relpath=Path(rel), is_dir=is_dir, comment=comment, annotation=annotation)

def read_input(path_or_dash: str) -> str:
    """Read text input from file or stdin.
    
    For image files, use parse_spec() instead.
    """
    if path_or_dash == "-":
        return os.read(0, 10_000_000).decode("utf-8")
    path = Path(path_or_dash)
    # Check if it's an image file
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        raise ValueError(
            f"Image file detected: {path_or_dash}. "
            "Use parse_spec() or parse_image() instead of read_input()."
        )
    return path.read_text(encoding="utf-8")


def parse_spec(
    spec_path: str,
    vars: Optional[Dict[str, str]] = None,
    base: Optional[Path] = None,
    mode: str = "loose",
) -> Tuple[Optional[Path], List[Node]]:
    """Parse a spec file (text or image) into nodes.
    
    Handles both text files (.tree, .yaml, .json) and image files (.png, .jpg, .jpeg).
    For text files, reads and parses the content.
    For image files, uses OCR to extract text then parses it.
    
    Args:
        spec_path: Path to spec file or image
        vars: Optional template variables
        base: Optional base directory
        mode: Parse mode ("loose" or "strict")
    
    Returns:
        tuple: (spec_path, nodes)
    """
    from .image import parse_image
    
    path = Path(spec_path)
    
    # Handle image files
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return parse_image(path, vars=vars, mode=mode)
    
    # Handle text files
    text = read_input(spec_path)
    return parse_any(spec_path, text, vars=vars, base=base, mode=mode)


def parse_tree_text(text: str, *args, **kwargs) -> List["Node"]:
    """
    Parse `tree`-like text into Nodes with correct hierarchical paths.
    """
    nodes: List["Node"] = []

    # stack[depth] = path at that depth
    stack: List[Path] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue

        # Skip the very first root line 
        if line.endswith("/") and ("├──" not in line and "└──" not in line):
            # Create explicit root node as "."
            nodes.append(_make_node(rel=".", is_dir=True))
            stack = [Path(".")]
            continue

        # Skip empty lines, sometimes those are added by the tree command 
        if not line.strip() or line.strip() == "|":
            continue

        m = TREE_LINE.match(line)
        if not m:
            continue

        prefix = m.group("prefix") or ""
        name = (m.group("name") or "").strip()

        # Extract comment and annotation before processing name
        name, comment, annotation = _extract_comment_and_annotation(name)

        is_dir = name.endswith("/")
        if is_dir:
            name = name[:-1]

        depth = _tree_depth(prefix)

        # Ensure stack has parent for this depth
        if not stack:
            stack = [Path(".")]

        # stack length should be depth+1 (root at 0)
        # If we move up, truncate
        if depth + 1 <= len(stack):
            stack = stack[: depth + 1]

        parent = stack[-1] if stack else Path(".")
        path = (parent / name).as_posix()

        nodes.append(_make_node(rel=path, is_dir=is_dir, comment=comment, annotation=annotation))

        if is_dir:
            # push this dir as current at next depth
            # Ensure stack is long enough
            while len(stack) <= depth + 1:
                stack.append(Path("."))
            stack[depth + 1] = Path(path)

    return nodes

def parse_structured(doc: dict) -> Tuple[Optional[Path], List[Node]]:
    validate_document(doc)

    root = Path(doc.get("root", "."))
    nodes: List[Node] = []

    for entry in doc.get("entries", []):
        path = entry["path"].rstrip("/")
        is_dir = entry.get("type") == "dir" or entry["path"].endswith("/")
        comment = entry.get("comment")
        annotation = entry.get("annotation")
        nodes.append(Node(Path(path), is_dir=is_dir, comment=comment, annotation=annotation))

    return root, nodes

def parse_any(
    path_or_dash: str,
    text: str,
    mode: str = "loose",
    vars: Optional[Dict[str, str]] = None,
    base: Optional[Path] = None,
) -> Tuple[Optional[Path], List[Node]]:
    base = base or Path(".")

    if path_or_dash != "-":
        text = resolve_includes(text, Path(path_or_dash))

    if vars:
        text = apply_vars(text, vars)

    stripped = text.lstrip()

    # JSON
    if stripped.startswith("{"):
        return parse_structured(json.loads(text))

    # YAML
    try:
        doc = yaml.safe_load(text)
        if isinstance(doc, dict) and "entries" in doc:
            return parse_structured(doc)
    except Exception:
        pass

    nodes = parse_tree_text(text, mode=mode)
    return None, nodes
