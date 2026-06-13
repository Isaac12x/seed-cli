"""Conversion helpers for compact declarative Seed specifications."""

from __future__ import annotations


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
