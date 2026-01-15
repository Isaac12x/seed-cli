

"""seed_cli.includes

Spec include resolver.

Supports:
- @include relative/path.tree
- @include ./relative.yaml
- nested includes
- cycle detection
- preserves line order

Includes are resolved at *text level* before parsing.
"""

from pathlib import Path
from typing import Set
import re

_INCLUDE_RE = re.compile(r"^\s*@include\s+(.+?)\s*$")


class IncludeError(RuntimeError):
    pass


def resolve_includes(text: str, base_path: Path, _seen: Set[Path] | None = None) -> str:
    """Resolve @include directives recursively.

    Parameters:
    - text: raw spec text
    - base_path: path of the current spec file
    """
    if _seen is None:
        _seen = set()

    lines = []
    for raw in text.splitlines():
        m = _INCLUDE_RE.match(raw)
        if not m:
            lines.append(raw)
            continue

        inc = m.group(1).strip().strip('"').strip("'")
        inc_path = (base_path.parent / inc).resolve()

        if inc_path in _seen:
            raise IncludeError(f"Include cycle detected: {inc_path}")

        if not inc_path.exists():
            raise IncludeError(f"Included file not found: {inc_path}")

        _seen.add(inc_path)
        inc_text = inc_path.read_text(encoding="utf-8")
        resolved = resolve_includes(inc_text, inc_path, _seen)
        lines.append(resolved)

    return "\n".join(lines)
