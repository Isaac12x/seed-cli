"""Normalize loose file-tree text into connected tree output."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable


BRANCH_RE = re.compile(
    r"^(?P<prefix>[ \t│|]*)(?P<branch>├──|└──|\+--|`--|\\--|\|--)\s*(?P<label>.*?)\s*$"
)
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
PATH_SUFFIX_RE = re.compile(
    r"^(?P<path>.+?)(?P<suffix>\s+(?:\?|@[A-Za-z_]|![A-Za-z_]|\+[A-Za-z_]|->|\(|#|//).*)$"
)
MARKDOWN_TREE_INFOS = {"", "text", "txt", "plain", "tree", "seed", "filetree"}


@dataclass
class TreeNode:
    """Small display tree used by the fixer."""

    label: str
    children: list["TreeNode"] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedLine:
    label: str
    depth: int


def fix_tree_text(text: str, *, markdown: bool = False) -> str:
    """Return text with file-tree blocks rendered using tree connectors."""
    if markdown:
        return fix_markdown_filetrees(text)

    return _preserve_terminal_newline(text, _fix_tree_blocks(text))


def fix_markdown_filetrees(text: str) -> str:
    """Fix fenced Markdown file-tree blocks, or the whole document if it is a tree."""
    lines = text.splitlines()
    out: list[str] = []
    transformed_fence = False
    i = 0

    while i < len(lines):
        fence_match = FENCE_RE.match(lines[i])
        if not fence_match:
            out.append(lines[i])
            i += 1
            continue

        opener = lines[i]
        fence = fence_match.group("fence")
        info_parts = fence_match.group("info").strip().split(maxsplit=1)
        info = info_parts[0].lower() if info_parts else ""
        block: list[str] = []
        i += 1

        while i < len(lines):
            close_match = FENCE_RE.match(lines[i])
            if close_match and close_match.group("fence").startswith(
                fence[0] * len(fence)
            ):
                break
            block.append(lines[i])
            i += 1

        should_fix = info in MARKDOWN_TREE_INFOS and _looks_like_tree_block(block)
        out.append(opener)
        if should_fix:
            out.extend(_fix_tree_blocks("\n".join(block)).splitlines())
            transformed_fence = True
        else:
            out.extend(block)

        if i < len(lines):
            out.append(lines[i])
            i += 1

    if transformed_fence:
        return _preserve_terminal_newline(text, "\n".join(out))

    if _looks_like_standalone_tree(lines):
        return _preserve_terminal_newline(text, _fix_tree_blocks(text))

    return text


def _preserve_terminal_newline(original: str, updated: str) -> str:
    if original.endswith("\n") and not updated.endswith("\n"):
        return f"{updated}\n"
    return updated


def _fix_tree_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    block: list[str] = []

    def flush_block() -> None:
        if not block:
            return
        if _looks_like_tree_block(block):
            out.extend(_fix_block(block).splitlines())
        else:
            out.extend(block)
        block.clear()

    for line in lines:
        if line.strip():
            block.append(line)
            continue
        flush_block()
        out.append(line)

    flush_block()
    return "\n".join(out)


def _looks_like_tree_block(lines: Iterable[str]) -> bool:
    meaningful = [line for line in lines if line.strip()]
    if len(meaningful) < 2:
        return False

    if any(_branch_match(line) for line in meaningful):
        return True

    treeish = 0
    for line in meaningful:
        stripped = line.strip()
        if stripped in (".", "./"):
            treeish += 1
        elif "/" in stripped:
            treeish += 1
        elif _looks_like_filename(stripped):
            treeish += 1

    return treeish >= 2


def _looks_like_standalone_tree(lines: Iterable[str]) -> bool:
    meaningful = [line for line in lines if line.strip()]
    if not meaningful:
        return False
    for line in meaningful:
        stripped = line.strip()
        if stripped.startswith(("#", "- ", "* ", ">")):
            return False
    return _looks_like_tree_block(meaningful)


def _looks_like_filename(text: str) -> bool:
    first = text.split(maxsplit=1)[0]
    if first.startswith(("#", "-", "*", ">")):
        return False
    return "." in first and "/" not in first


def _fix_block(lines: list[str]) -> str:
    meaningful = [line.rstrip() for line in lines if line.strip()]
    if _is_path_list(meaningful):
        return _render_path_list(meaningful)
    return _render_parsed_lines(_parse_indented_or_connected(meaningful))


def _is_path_list(lines: list[str]) -> bool:
    if any(_branch_match(line) for line in lines):
        return False
    if any(_indent_columns(line) for line in lines):
        return False
    return True


def _render_path_list(lines: list[str]) -> str:
    root = TreeNode("./")
    by_path: dict[str, TreeNode] = {"": root}

    for line in lines:
        path_text, suffix = _split_path_suffix(line.strip())
        if path_text in (".", "./"):
            root.label = path_text
            continue

        is_dir = path_text.endswith("/")
        parts = [part for part in path_text.strip("/").split("/") if part]
        if not parts:
            continue

        current_path = ""
        parent = root
        for index, part in enumerate(parts):
            current_path = f"{current_path}/{part}" if current_path else part
            last = index == len(parts) - 1
            label = part
            if not last or is_dir:
                label = f"{label}/"
            if last:
                label = f"{label}{suffix}"

            node = by_path.get(current_path)
            if node is None:
                node = TreeNode(label)
                by_path[current_path] = node
                parent.children.append(node)
            elif last and suffix:
                node.label = label
            parent = node

    if len(root.children) == 1 and _is_rendered_dir(root.children[0].label):
        only = root.children[0]
        return "\n".join(_render_tree(only, root=True))

    return "\n".join(_render_tree(root, root=True))


def _split_path_suffix(line: str) -> tuple[str, str]:
    match = PATH_SUFFIX_RE.match(line)
    if not match:
        return line, ""
    return match.group("path"), match.group("suffix")


def _parse_indented_or_connected(lines: list[str]) -> list[ParsedLine]:
    explicit_root = _has_explicit_root(lines)
    indent_unit = _indent_unit(lines)
    parsed: list[ParsedLine] = []

    for line in lines:
        branch_match = _branch_match(line)
        if branch_match:
            prefix = branch_match.group("prefix")
            depth = _prefix_depth(prefix)
            if explicit_root:
                depth += 1
            parsed.append(
                ParsedLine(
                    label=branch_match.group("label").strip(),
                    depth=depth,
                )
            )
            continue

        label = line.strip()
        depth = _indent_columns(line) // indent_unit if indent_unit else 0
        parsed.append(ParsedLine(label=label, depth=depth))

    return parsed


def _render_parsed_lines(lines: list[ParsedLine]) -> str:
    if not lines:
        return ""

    root = TreeNode("./")
    stack: list[TreeNode] = [root]
    first = lines[0]

    if len(lines) > 1 and first.depth == 0 and lines[1].depth > first.depth:
        root = TreeNode(first.label)
        stack = [root]
        iterable = lines[1:]
    else:
        iterable = lines

    for line in iterable:
        depth = max(line.depth, 0)
        if depth + 1 < len(stack):
            stack = stack[: depth + 1]
        while len(stack) <= depth:
            stack.append(stack[-1])

        parent = stack[depth] if depth < len(stack) else root
        node = TreeNode(line.label)
        parent.children.append(node)

        if len(stack) <= depth + 1:
            stack.append(node)
        else:
            stack[depth + 1] = node

    return "\n".join(_render_tree(root, root=True))


def _render_tree(
    node: TreeNode,
    *,
    root: bool = False,
    prefix: str = "",
    last: bool = True,
) -> list[str]:
    lines: list[str] = []
    if root:
        lines.append(node.label)
        child_prefix = ""
    else:
        connector = "└── " if last else "├── "
        lines.append(f"{prefix}{connector}{node.label}")
        child_prefix = f"{prefix}{'    ' if last else '│   '}"

    for index, child in enumerate(node.children):
        lines.extend(
            _render_tree(
                child,
                prefix=child_prefix,
                last=index == len(node.children) - 1,
            )
        )
    return lines


def _is_rendered_dir(label: str) -> bool:
    head = label.split(maxsplit=1)[0]
    return head.endswith("/")


def _has_explicit_root(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    first = lines[0]
    if _branch_match(first):
        return False
    first_indent = _indent_columns(first)
    second = lines[1]
    return bool(_branch_match(second)) or _indent_columns(second) > first_indent


def _branch_match(line: str):
    return BRANCH_RE.match(line.rstrip())


def _prefix_depth(prefix: str) -> int:
    normalized = prefix.expandtabs(4).replace("│", "|")
    return len(normalized) // 4


def _indent_unit(lines: list[str]) -> int:
    values = [
        _indent_columns(line)
        for line in lines
        if not _branch_match(line) and _indent_columns(line) > 0
    ]
    if not values:
        return 4
    return math.gcd(*values) or min(values)


def _indent_columns(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))
