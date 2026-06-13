

"""Parsing layer for seed-cli.

Supports:
- Tree text specs (ASCII tree or simple path-per-line)
- YAML / JSON structured specs
- stdin support ("-")
- Comments and annotations
- Variable templating ({{var}})
- Includes (@include)

Outputs Node(relpath, is_dir, comment, annotation, metadata).
"""

import os
import json
import yaml
import re
import inspect

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

from .logging import get_logger
from .schema import validate_document
from .templating import apply_vars
from .includes import resolve_includes
from .conversion import expand_brace_paths

log = get_logger("parsers")


@dataclass
class Node:
    relpath: Path
    is_dir: bool
    comment: Optional[str] = None
    annotation: Optional[str] = None  # generated | manual | None
    optional: bool = False  # marked with ? - prompt user before creating
    metadata: Dict[str, Any] = field(default_factory=dict)


# _COMMENT_RE matches comments in parentheses (e.g., (note here)), and ignores # style comments in a line.
_COMMENT_RE = re.compile(r"\(([^)]+)\)|//(.*)$|#(.*)$")
_ANNOT_RE = re.compile(r"(?:\s+\(@([a-zA-Z_][\w-]*)\)|\s+@([a-zA-Z_][\w-]*)\b)")
_OPTIONAL_RE = re.compile(r"\?(?:\s|$)")
_KIND_RE = re.compile(r"(?:^|\s)!(?P<kind>[a-zA-Z_][\w-]*)\b")
_TAG_RE = re.compile(r"(?:^|\s)\+(?P<tag>[a-zA-Z_][\w-]*)\b")
_URL_RE = re.compile(
    r"\s+->\s+(?P<url>(?:git\+)?https?://[^\s)]+|ssh://[^\s)]+|git://[^\s)]+|git@[^\s)]+)"
)
TREE_LINE = re.compile(r"""
^(?P<prefix>[\s│|]*)(?P<branch>├──|└──)?\s*(?P<name>.+?)\s*$
""", re.VERBOSE)
_GUIDE_ONLY_RE = re.compile(r"^[\s│|]+$")

def _tree_depth(prefix: str) -> int:
    """
    Each indentation level in `tree` output is typically 4 chars: '│   ' or '    '
    We'll treat any group of 4 columns as one depth.
    """
    # Normalize tabs just in case
    prefix = prefix.replace("\t", "    ")
    return len(prefix) // 4

def _compact_metadata(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize metadata and drop empty values."""
    normalized = dict(metadata or {})

    tags = normalized.get("tags")
    if isinstance(tags, list):
        deduped: list[str] = []
        for tag in tags:
            tag_text = str(tag).strip()
            if tag_text and tag_text not in deduped:
                deduped.append(tag_text)
        if deduped:
            normalized["tags"] = deduped
        else:
            normalized.pop("tags", None)

    for key in ("kind", "url"):
        if key not in normalized:
            continue
        value = str(normalized.get(key, "")).strip()
        if value:
            normalized[key] = value
        else:
            normalized.pop(key, None)

    return normalized


def _extract_inline_metadata(text: str) -> tuple[str, Dict[str, Any]]:
    """Extract .seed-style inline metadata tokens."""
    metadata: Dict[str, Any] = {}

    url_match = _URL_RE.search(text)
    if url_match:
        metadata["url"] = url_match.group("url").rstrip(",.;")
        text = _URL_RE.sub("", text, count=1)

    kinds = _KIND_RE.findall(text)
    if kinds:
        metadata["kind"] = kinds[-1]
        text = _KIND_RE.sub(" ", text)

    tags = _TAG_RE.findall(text)
    if tags:
        metadata["tags"] = tags
        text = _TAG_RE.sub(" ", text)

    return " ".join(text.split()), _compact_metadata(metadata)


def _extract_comment_and_annotation(text: str) -> tuple[str, Optional[str], Optional[str], bool, Dict[str, Any]]:
    """Extract comment, annotation, optional marker, and inline metadata from text.

    Returns: (cleaned_text, comment, annotation, is_optional, metadata)
    """
    comment = None
    annotation = None
    is_optional = False
    metadata: Dict[str, Any] = {}

    # Check for optional marker (?)
    if _OPTIONAL_RE.search(text):
        is_optional = True
        text = _OPTIONAL_RE.sub("", text)

    text, metadata = _extract_inline_metadata(text)

    ann = _ANNOT_RE.search(text)
    if ann:
        annotation = ann.group(1) or ann.group(2)
        text = _ANNOT_RE.sub("", text)

    com = _COMMENT_RE.search(text)
    if com:
        # Check which group matched: (1) parenthetical, (2) //, (3) #
        comment = (com.group(1) or com.group(2) or com.group(3) or "").strip()
        text = _COMMENT_RE.sub("", text)

    return text.strip(), comment, annotation, is_optional, metadata


def _make_node(
    *,
    rel: str,
    is_dir: bool,
    comment: str | None = None,
    annotation: str | None = None,
    optional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Construct Node with proper Path for relpath.
    """
    return Node(
        relpath=Path(rel),
        is_dir=is_dir,
        comment=comment,
        annotation=annotation,
        optional=optional,
        metadata=_compact_metadata(metadata),
    )


def format_node_suffix(node: Node) -> str:
    """Render inline suffix tokens for a node."""
    parts: list[str] = []

    if node.optional:
        parts.append("?")
    if node.annotation and not node.annotation.startswith("template:"):
        parts.append(f"@{node.annotation}")

    metadata = _compact_metadata(node.metadata)

    kind = metadata.get("kind")
    if kind:
        parts.append(f"!{kind}")

    for tag in metadata.get("tags", []):
        parts.append(f"+{tag}")

    url = metadata.get("url")
    if url:
        parts.append(f"-> {url}")

    if node.comment:
        parts.append(f"({node.comment})")

    return f" {' '.join(parts)}" if parts else ""


def render_node_text(node: Node, *, basename: bool = False) -> str:
    """Render a node to tree-like text."""
    target = node.relpath.name if basename else node.relpath.as_posix()
    if node.is_dir:
        target += "/"
    return f"{target}{format_node_suffix(node)}"

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
    """Parse a spec file (text, image, or graphviz) into nodes.
    
    Handles:
    - Text files (.tree, .seed, .yaml, .json)
    - Image files (.png, .jpg, .jpeg) - uses OCR
    - Graphviz files (.dot) - parses DOT format
    
    For text files, reads and parses the content.
    For image files, uses OCR to extract text then parses it.
    For DOT files, parses the graph structure into nodes.
    
    Args:
        spec_path: Path to spec file, image, or DOT file
        vars: Optional template variables
        base: Optional base directory
        mode: Parse mode ("loose" or "strict")
    
    Returns:
        tuple: (spec_path, nodes)
    """
    from .image import parse_image
    from .graphviz import dot_to_nodes
    
    path = Path(spec_path)
    
    # Handle image files
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        try:
            return parse_image(path, vars=vars, mode=mode)
        except Exception as e:
            raise RuntimeError(f"Failed to parse image spec '{spec_path}': {e}") from e
    
    # Handle DOT files
    if path.suffix.lower() == ".dot":
        text = read_input(spec_path)
        # Apply vars if provided (though DOT files typically don't use vars)
        if vars:
            from .templating import apply_vars
            text = apply_vars(text, vars)
        nodes = dot_to_nodes(text)
        return path, nodes
    
    # Handle text files
    text = read_input(spec_path)
    return parse_any(spec_path, text, vars=vars, base=base, mode=mode)


_TEMPLATE_VAR_RE = re.compile(r"^<([a-zA-Z_][a-zA-Z0-9_]*)>$")


def parse_tree_text(text: str, *args, **kwargs) -> List["Node"]:
    """
    Parse `tree`-like text into Nodes with correct hierarchical paths.

    Special syntax:
    - `...` as a child entry marks the parent directory as allowing extra files.
      This creates a marker node with annotation="extras".
    - `<varname>/` marks a template directory that can match multiple actual directories.
      This creates a marker node with annotation="template:<varname>".
      Children of template dirs inherit the template path.
    """
    nodes: List["Node"] = []

    # stack[depth] = path at that depth
    stack: List[Path] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not line:
            continue

        # Skip the very first root line
        if stripped in (".", "./"):
            nodes.append(_make_node(rel=".", is_dir=True))
            stack = [Path(".")]
            continue
        if stripped.endswith("/") and ("├──" not in stripped and "└──" not in stripped):
            # Create explicit root node as "."
            nodes.append(_make_node(rel=".", is_dir=True))
            stack = [Path(".")]
            continue

        # Skip empty guide-only lines, sometimes those are added by tree output.
        if not stripped or _GUIDE_ONLY_RE.fullmatch(stripped):
            continue

        m = TREE_LINE.match(line)
        if not m:
            continue

        prefix = m.group("prefix") or ""
        name = (m.group("name") or "").strip()

        # Extract comment, annotation, optional marker, and metadata before processing name
        name, comment, annotation, is_optional, metadata = _extract_comment_and_annotation(name)

        stack_before_expansion = list(stack)
        for expanded_name in expand_brace_paths(name):
            stack = list(stack_before_expansion)

            # Handle "..." marker for allowing extras in parent directory
            if expanded_name == "..." or expanded_name == "…":
                depth = _tree_depth(prefix)
                if not stack:
                    stack = [Path(".")]
                if depth + 1 <= len(stack):
                    stack = stack[: depth + 1]
                parent = stack[-1] if stack else Path(".")
                # Create marker node: path is parent/..., annotation is "extras"
                marker_path = (parent / "...").as_posix()
                nodes.append(_make_node(
                    rel=marker_path,
                    is_dir=False,
                    comment=comment,
                    annotation="extras",
                    metadata=metadata,
                ))
                continue

            is_dir = expanded_name.endswith("/")
            if is_dir:
                expanded_name = expanded_name[:-1]

            depth = _tree_depth(prefix)

            # Ensure stack has parent for this depth
            if not stack:
                stack = [Path(".")]

            # stack length should be depth+1 (root at 0)
            # If we move up, truncate
            if depth + 1 <= len(stack):
                stack = stack[: depth + 1]

            parent = stack[-1] if stack else Path(".")

            # Check if this is a template variable directory like <version_id>
            template_match = _TEMPLATE_VAR_RE.match(expanded_name)
            if template_match and is_dir:
                var_name = template_match.group(1)
                # Keep the <varname> in the path for matching logic
                path = (parent / expanded_name).as_posix()
                nodes.append(_make_node(
                    rel=path,
                    is_dir=True,
                    comment=comment,
                    annotation=f"template:{var_name}",
                    optional=is_optional,
                    metadata=metadata,
                ))
                # Push to stack so children can reference this template path
                while len(stack) <= depth + 1:
                    stack.append(Path("."))
                stack[depth + 1] = Path(path)
                continue

            path = (parent / expanded_name).as_posix()

            nodes.append(_make_node(
                rel=path,
                is_dir=is_dir,
                comment=comment,
                annotation=annotation,
                optional=is_optional,
                metadata=metadata,
            ))

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
        optional = entry.get("optional", False)
        metadata = entry.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("entry metadata must be an object")

        merged_metadata: Dict[str, Any] = dict(metadata or {})
        if "kind" in entry:
            merged_metadata["kind"] = entry.get("kind")
        if "url" in entry:
            merged_metadata["url"] = entry.get("url")
        if "tags" in entry:
            merged_metadata["tags"] = entry.get("tags")

        nodes.append(Node(
            Path(path),
            is_dir=is_dir,
            comment=comment,
            annotation=annotation,
            optional=optional,
            metadata=_compact_metadata(merged_metadata),
        ))

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
