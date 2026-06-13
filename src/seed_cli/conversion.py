"""Conversion helpers for compact declarative Seed specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .parsers import Node


def _matching_brace(text: str, start: int) -> int | None:
    """Return the closing brace for ``start``, accounting for nested groups."""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_alternatives(text: str) -> list[str]:
    """Split top-level comma alternatives while preserving nested groups."""
    alternatives: list[str] = []
    start = 0
    depth = 0

    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            alternatives.append(text[start:index].strip())
            start = index + 1

    if alternatives:
        alternatives.append(text[start:].strip())
    return alternatives


def expand_brace_paths(text: str) -> list[str]:
    """Expand comma-separated brace groups in a path expression.

    Unmatched braces and groups without a top-level comma remain literal.
    Multiple groups are expanded recursively as a Cartesian product.
    """
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue

        closing = _matching_brace(text, index)
        if closing is None:
            return [text]

        alternatives = _split_alternatives(text[index + 1 : closing])
        if alternatives:
            prefix = text[:index]
            suffix = text[closing + 1 :]
            expanded: list[str] = []
            for alternative in alternatives:
                expanded.extend(expand_brace_paths(prefix + alternative + suffix))
            return expanded

        index = closing + 1

    return [text]


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    terminals: list[Any] = field(default_factory=list)


def _node_identity(node: "Node") -> tuple[Any, ...]:
    return (
        node.relpath.as_posix(),
        node.is_dir,
        node.comment,
        node.annotation,
        node.optional,
        json.dumps(node.metadata, sort_keys=True),
    )


def _build_trie(nodes: Iterable["Node"]) -> _TrieNode:
    root = _TrieNode()
    seen: set[tuple[Any, ...]] = set()

    for node in nodes:
        identity = _node_identity(node)
        if identity in seen:
            continue
        seen.add(identity)

        current = root
        for part in node.relpath.parts:
            current = current.children.setdefault(part, _TrieNode())
        current.terminals.append(node)

    return root


def _has_meaningful_directory_marker(node: "Node") -> bool:
    from .parsers import format_node_suffix

    return bool(format_node_suffix(node)) or bool(
        node.annotation and node.annotation.startswith("template:")
    )


def _renderable_nodes(trie: _TrieNode) -> list["Node"]:
    nodes: list["Node"] = []

    def visit(current: _TrieNode) -> None:
        for node in current.terminals:
            if node.relpath.as_posix() == ".":
                continue
            if (
                not node.is_dir
                or not current.children
                or _has_meaningful_directory_marker(node)
            ):
                nodes.append(node)

        for child in current.children.values():
            visit(child)

    visit(trie)
    return nodes


def _group_key(node: "Node") -> tuple[str, str | None]:
    from .parsers import format_node_suffix

    template_annotation = (
        node.annotation
        if node.annotation and node.annotation.startswith("template:")
        else None
    )
    return format_node_suffix(node), template_annotation


def _group_expression(nodes: list["Node"]) -> str | None:
    names = [node.relpath.name for node in nodes]
    if len(names) < 2 or any(
        any(char in name for char in "{},") for name in names
    ):
        return None

    if all(node.is_dir for node in nodes):
        return "{" + ",".join(sorted(names)) + "}/"

    if all(not node.is_dir for node in nodes):
        extensions = {Path(name).suffix for name in names}
        if len(extensions) == 1:
            extension = extensions.pop()
            if extension:
                stems = sorted(name[: -len(extension)] for name in names)
                if all(stems):
                    return "{" + ",".join(stems) + "}" + extension

    alternatives = sorted(
        node.relpath.name + ("/" if node.is_dir else "") for node in nodes
    )
    return "{" + ",".join(alternatives) + "}"


def _join_parent(parent: Path, child: str) -> str:
    parent_text = parent.as_posix()
    if parent_text in {"", "."}:
        return child
    return f"{parent_text}/{child}"


def _render_sibling_group(
    parent: Path,
    suffix: str,
    siblings: list["Node"],
) -> list[str]:
    from .parsers import format_node_suffix

    expression = _group_expression(siblings)
    if expression is not None:
        return [f"{_join_parent(parent, expression)}{suffix}"]

    return [
        (
            node.relpath.as_posix()
            + ("/" if node.is_dir else "")
            + format_node_suffix(node)
        )
        for node in siblings
    ]


def _partition_siblings(siblings: list["Node"]) -> list[list["Node"]]:
    buckets: dict[tuple[str, str], list["Node"]] = {}
    for node in siblings:
        key = (
            ("directory", "")
            if node.is_dir
            else ("file", Path(node.relpath.name).suffix)
        )
        buckets.setdefault(key, []).append(node)

    grouped = [bucket for bucket in buckets.values() if len(bucket) >= 2]
    leftovers = [
        node for bucket in buckets.values() if len(bucket) < 2 for node in bucket
    ]
    if leftovers:
        grouped.append(leftovers)
    return grouped


def _render_best_sibling_layout(
    parent: Path,
    suffix: str,
    siblings: list["Node"],
) -> list[str]:
    direct = _render_sibling_group(parent, suffix, siblings)
    partitions = _partition_siblings(siblings)
    if len(partitions) <= 1:
        return direct

    partitioned = [
        line
        for partition in partitions
        for line in _render_sibling_group(parent, suffix, partition)
    ]
    direct_size = sum(len(line) + 1 for line in direct)
    partitioned_size = sum(len(line) + 1 for line in partitioned)
    return partitioned if partitioned_size < direct_size else direct


def render_compact_seed(nodes: Iterable["Node"]) -> str:
    """Return deterministic compact .seed path lines."""
    candidates = _renderable_nodes(_build_trie(nodes))
    grouped: dict[tuple[Path, tuple[str, str | None]], list["Node"]] = {}

    for node in candidates:
        key = (node.relpath.parent, _group_key(node))
        grouped.setdefault(key, []).append(node)

    lines: list[str] = []
    for (parent, (suffix, _template_annotation)), siblings in grouped.items():
        lines.extend(_render_best_sibling_layout(parent, suffix, siblings))

    if not lines:
        return ""
    return "\n".join(sorted(lines)) + "\n"


def convert_tree_to_seed(
    input_path: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Convert a .tree file into a compact declarative .seed file."""
    from .parsers import parse_spec

    source = Path(input_path)
    if source.suffix.lower() != ".tree":
        raise ValueError(f"Tree input path must use the .tree suffix: {source}")
    if not source.exists():
        raise FileNotFoundError(f"Tree input file not found: {source}")
    if not source.is_file():
        raise ValueError(f"Tree input path must be a file: {source}")

    destination = (
        Path(output_path) if output_path is not None else source.with_suffix(".seed")
    )
    if destination.suffix.lower() != ".seed":
        raise ValueError(
            f"Seed output path must use the .seed suffix: {destination}"
        )

    _, nodes = parse_spec(str(source), base=source.parent)
    rendered = render_compact_seed(nodes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination
