"""Forward-migration helpers (PRD §7.1 / M3).

When the exporter sees a prior spec version (under ``.seed/specs/``) that is
distinct from the current spec, it emits a *successor* pack that declares
``migration_from`` + a list of ``mapping_rules`` derived from the diff. These
rules tell gbrain how to re-type already-imported pages when the user runs
``gbrain jobs submit unify-types``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..parsers import Node, parse_any
from ..spec_history import get_spec_version, list_spec_versions
from .compiler import CompiledPack, compile_pack
from .kindmap import KindEntry


def _index_prefixes(pack: CompiledPack) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pt in pack.manifest.get("page_types", []) or []:
        name = pt.get("name")
        for prefix in pt.get("path_prefixes", []) or []:
            out[prefix] = name
    return out


def compute_mapping_rules(
    prev_nodes: List[Node],
    new_nodes: List[Node],
    *,
    name: str,
    kindmap: Optional[Dict[str, KindEntry]] = None,
) -> List[Dict[str, str]]:
    """Diff prev/new page-type prefixes and emit retype rules.

    Rules cover three drift shapes:
      * type renamed at the same prefix     -> retype old -> new under that path
      * prefix removed entirely             -> retype old -> note under that path
      * brand-new prefix                    -> no rule needed (new type covers it)

    The unconditional ``*unknown*`` -> ``note`` catch-all is added by the
    compiler regardless.
    """
    prev = compile_pack(prev_nodes, name=name, version="0.0.0", kindmap=kindmap)
    new = compile_pack(new_nodes, name=name, version="0.0.0", kindmap=kindmap)
    prev_map = _index_prefixes(prev)
    new_map = _index_prefixes(new)

    rules: List[Dict[str, str]] = []
    for prefix, new_type in new_map.items():
        old_type = prev_map.get(prefix)
        if old_type and old_type != new_type:
            rules.append({
                "kind": "retype",
                "from_type": old_type,
                "to_type": new_type,
                "path_filter": f"{prefix}%",
            })
    for prefix, old_type in prev_map.items():
        if prefix not in new_map:
            rules.append({
                "kind": "retype",
                "from_type": old_type,
                "to_type": "note",
                "path_filter": f"{prefix}%",
            })
    return rules


def derive_predecessor_version(prev_version_number: int) -> str:
    """Map a seed spec version (an int) to a semver predecessor string."""
    return f"0.0.{max(0, prev_version_number - 1)}.x" if False else f"0.{prev_version_number}.x"


def find_predecessor_spec(
    base: Path,
    current_text: str,
) -> Optional[Tuple[int, str]]:
    """Return ``(version_number, spec_text)`` for the most recent prior version,
    or ``None`` when there's no captured history that differs from ``current_text``.
    """
    versions = list_spec_versions(base)
    if not versions:
        return None
    normalised_current = _normalise(current_text)
    for version, _path, _ts in reversed(versions):
        prev_text = get_spec_version(base, version)
        if prev_text is None:
            continue
        if _normalise(prev_text) == normalised_current:
            continue
        return version, prev_text
    return None


def _normalise(text: str) -> str:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_spec_text(text: str) -> List[Node]:
    _, nodes = parse_any("-", text)
    return nodes
