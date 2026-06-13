"""Compile parsed seed nodes into a gbrain-schema-pack-v1 manifest dict.

The compiler is intentionally tolerant (PRD §4):

  - unknown `!kind` markers degrade to a derived ``concept`` page-type;
  - every pack carries an unconditional ``*unknown*`` -> ``note`` mapping rule
    so a page gbrain encounters that the spec never anticipated is still typed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..parsers import Node
from .kindmap import KindEntry, resolve, default_kindmap


GBRAIN_API_VERSION = "gbrain-schema-pack-v1"
DEFAULT_EXTENDS = "gbrain-base"
DEFAULT_MIN_VERSION = "0.41.22"
DEFAULT_TAKES = ["fact", "take", "bet", "hunch"]
ENTITY_DIR_HINTS = {"people", "person", "contacts", "companies", "company", "orgs", "organizations"}


@dataclass
class _PageType:
    name: str
    primitive: str
    path_prefixes: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    extractable: bool = False
    expert_routing: bool = False

    def add_prefix(self, prefix: str) -> None:
        if prefix and prefix not in self.path_prefixes:
            self.path_prefixes.append(prefix)

    def merge_aliases(self, aliases: Iterable[str]) -> None:
        for alias in aliases:
            if alias and alias != self.name and alias not in self.aliases:
                self.aliases.append(alias)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "primitive": self.primitive,
            "path_prefixes": sorted(self.path_prefixes),
            "aliases": sorted(self.aliases),
            "extractable": self.extractable,
            "expert_routing": self.expert_routing,
        }


@dataclass
class CompiledPack:
    manifest: Dict[str, Any]
    diagnostics: List[str] = field(default_factory=list)


def _prefix_for_typed_dir(rel: PurePosixPath) -> str:
    parent = rel.parent
    if str(parent) in (".", ""):
        return rel.as_posix().rstrip("/") + "/"
    return parent.as_posix().rstrip("/") + "/"


def _singularize(name: str) -> str:
    name = name.strip("/")
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("ses") and len(name) > 3:
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def _entity_primitive_for_dir(parent_name: str) -> str:
    return "entity" if parent_name.lower() in ENTITY_DIR_HINTS else "concept"


def _add_pages(
    pages: Dict[str, _PageType],
    *,
    name: str,
    primitive: str,
    prefix: str,
    aliases: Iterable[str],
    extractable: bool,
    expert_routing: bool,
) -> None:
    existing = pages.get(name)
    if existing is None:
        existing = _PageType(name=name, primitive=primitive)
        pages[name] = existing
    else:
        if primitive == "entity":
            existing.primitive = "entity"
    existing.add_prefix(prefix)
    existing.merge_aliases(aliases)
    existing.extractable = existing.extractable or extractable
    existing.expert_routing = existing.expert_routing or expert_routing


def compile_pack(
    nodes: List[Node],
    *,
    name: str,
    version: str,
    kindmap: Optional[Dict[str, KindEntry]] = None,
    extends: Optional[str] = DEFAULT_EXTENDS,
    description: Optional[str] = None,
    gbrain_min_version: str = DEFAULT_MIN_VERSION,
    migration_from: Optional[Dict[str, Any]] = None,
    mapping_rules: Optional[List[Dict[str, Any]]] = None,
) -> CompiledPack:
    """Compile parsed seed nodes into a gbrain pack manifest dict."""
    if kindmap is None:
        kindmap = default_kindmap()

    pages: Dict[str, _PageType] = {}
    diagnostics: List[str] = []

    for node in nodes:
        if not node.is_dir:
            continue
        rel = PurePosixPath(node.relpath.as_posix())
        if rel.as_posix() in (".", ""):
            continue
        kind = (node.metadata or {}).get("kind")
        tags = (node.metadata or {}).get("tags") or []
        is_placeholder = bool(node.annotation and node.annotation.startswith("template:"))

        if kind:
            entry = resolve(kindmap, kind, tags=tags)
            if kind not in kindmap:
                diagnostics.append(
                    f"unknown !kind '{kind}' at {rel} -> derived '{entry.type}' (concept)"
                )
            prefix = _prefix_for_typed_dir(rel)
            _add_pages(
                pages,
                name=entry.type,
                primitive=entry.primitive,
                prefix=prefix,
                aliases=entry.aliases,
                extractable=entry.extractable,
                expert_routing=entry.expert_routing,
            )
            continue

        if is_placeholder:
            parent = rel.parent
            if str(parent) in (".", ""):
                continue
            parent_name = parent.name
            primitive = _entity_primitive_for_dir(parent_name)
            derived = _singularize(parent_name).lower() or parent_name
            entry = KindEntry(
                type=derived,
                primitive=primitive,
                expert_routing=(primitive == "entity"),
            )
            _add_pages(
                pages,
                name=entry.type,
                primitive=entry.primitive,
                prefix=parent.as_posix().rstrip("/") + "/",
                aliases=[],
                extractable=False,
                expert_routing=entry.expert_routing,
            )

    page_types = [pages[k].to_dict() for k in sorted(pages)]

    rules: List[Dict[str, Any]] = []
    if mapping_rules:
        rules.extend(mapping_rules)
    if not any(_is_catch_all(rule) for rule in rules):
        rules.append({
            "kind": "retype",
            "from_type": "*unknown*",
            "to_type": "note",
            "subtype_field": "legacy_type",
            "subtype": "*original_type*",
        })

    manifest: Dict[str, Any] = {
        "api_version": GBRAIN_API_VERSION,
        "name": name,
        "version": version,
        "gbrain_min_version": gbrain_min_version,
        "extends": extends,
        "description": description or "Generated from .seed by seed-cli. Do not hand-edit; edit the .seed spec.",
        "page_types": page_types,
        "link_types": [],
        "takes_kinds": list(DEFAULT_TAKES),
        "borrow_from": [],
        "frontmatter_links": [],
        "enrichable_types": [],
        "filing_rules": [],
        "mapping_rules": rules,
    }
    if migration_from:
        manifest["migration_from"] = dict(migration_from)
    return CompiledPack(manifest=manifest, diagnostics=diagnostics)


def _is_catch_all(rule: Dict[str, Any]) -> bool:
    return (
        rule.get("kind") == "retype"
        and rule.get("from_type") == "*unknown*"
    )
