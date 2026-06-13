"""Serialise / validate gbrain pack manifest dicts.

`dump_manifest` produces deterministic YAML so re-running the exporter on an
unchanged spec yields a byte-identical pack file (PRD AC2).

`lint_manifest` is an internal fallback for environments without a `gbrain`
binary; the external `gbrain schema validate` is preferred when available.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Tuple

import yaml


_TOP_ORDER: Tuple[str, ...] = (
    "api_version",
    "name",
    "version",
    "gbrain_min_version",
    "extends",
    "description",
    "page_types",
    "link_types",
    "takes_kinds",
    "borrow_from",
    "frontmatter_links",
    "enrichable_types",
    "filing_rules",
    "mapping_rules",
    "migration_from",
)

_PAGE_TYPE_ORDER: Tuple[str, ...] = (
    "name",
    "primitive",
    "path_prefixes",
    "aliases",
    "extractable",
    "expert_routing",
)


def _ordered(value: Any, order: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return value
    out: Dict[str, Any] = {}
    for key in order:
        if key in value:
            out[key] = value[key]
    for key, val in sorted(value.items()):
        if key not in out:
            out[key] = val
    return out


def canonicalise(manifest: Dict[str, Any]) -> Dict[str, Any]:
    out = _ordered(dict(manifest), _TOP_ORDER)
    page_types = out.get("page_types") or []
    out["page_types"] = [_ordered(pt, _PAGE_TYPE_ORDER) for pt in page_types]
    return out


class _IndentedSafeDumper(yaml.SafeDumper):
    """SafeDumper with indented list items (matches gbrain-base's YAML shape)."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


def dump_manifest(manifest: Dict[str, Any]) -> str:
    """Return deterministic YAML for the manifest."""
    canonical = canonicalise(manifest)
    return yaml.dump(
        canonical,
        Dumper=_IndentedSafeDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=4096,
        indent=2,
    )


def manifest_hash(manifest: Dict[str, Any]) -> str:
    """Stable hash of the canonical manifest text."""
    return hashlib.sha256(dump_manifest(manifest).encode("utf-8")).hexdigest()


_VALID_PRIMITIVES = {"entity", "concept", "media", "temporal", "annotation"}


def lint_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems with the manifest."""
    problems: List[str] = []
    if manifest.get("api_version") != "gbrain-schema-pack-v1":
        problems.append("api_version must be 'gbrain-schema-pack-v1'")
    if not manifest.get("name"):
        problems.append("name is required")
    if not manifest.get("version"):
        problems.append("version is required")

    seen_names: set[str] = set()
    seen_prefixes: dict[str, str] = {}
    for index, page in enumerate(manifest.get("page_types") or []):
        prefix = f"page_types[{index}]"
        if not isinstance(page, dict):
            problems.append(f"{prefix} must be an object")
            continue
        name = page.get("name")
        if not name:
            problems.append(f"{prefix} missing name")
            continue
        if name in seen_names:
            problems.append(f"{prefix} duplicate name '{name}'")
        seen_names.add(name)
        primitive = page.get("primitive")
        if primitive not in _VALID_PRIMITIVES:
            problems.append(
                f"{prefix} '{name}' has invalid primitive '{primitive}'"
            )
        prefixes = page.get("path_prefixes") or []
        if not isinstance(prefixes, list):
            problems.append(f"{prefix} '{name}' path_prefixes must be a list")
            continue
        for pfx in prefixes:
            if not isinstance(pfx, str) or not pfx.endswith("/"):
                problems.append(
                    f"{prefix} '{name}' prefix '{pfx}' must end with '/'"
                )
            owner = seen_prefixes.get(pfx)
            if owner and owner != name:
                problems.append(
                    f"{prefix} prefix '{pfx}' claimed by both '{owner}' and '{name}'"
                )
            seen_prefixes[pfx] = name
    return problems
