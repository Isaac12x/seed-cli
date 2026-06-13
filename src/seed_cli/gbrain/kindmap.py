"""Loose kindmap loader.

`!kind` markers in a `.seed` spec map to gbrain page-type defaults via a
kindmap. Ship a bundled default, allow per-repo override at
`.seed/gbrain/kindmap.yml`, allow explicit `--kindmap PATH`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


ENTITY_PRIMITIVE = "entity"
DEFAULT_PRIMITIVE = "concept"
VALID_PRIMITIVES = {"entity", "concept", "media", "temporal", "annotation"}


@dataclass
class KindEntry:
    """Resolved type defaults for a single seed !kind."""

    type: str
    primitive: str = DEFAULT_PRIMITIVE
    extractable: bool = False
    expert_routing: bool = False
    aliases: list[str] = field(default_factory=list)

    def merged_with(self, override: "KindEntry | dict[str, Any]") -> "KindEntry":
        data = asdict(self)
        if isinstance(override, KindEntry):
            override_data = asdict(override)
        else:
            override_data = dict(override)
        merged_aliases = list(data["aliases"])
        for alias in override_data.get("aliases") or []:
            if alias not in merged_aliases:
                merged_aliases.append(alias)
        data.update({k: v for k, v in override_data.items() if v is not None})
        data["aliases"] = merged_aliases
        return KindEntry(**data)


def _coerce_entry(key: str, raw: Any) -> KindEntry:
    if raw is None:
        return KindEntry(type=key)
    if not isinstance(raw, dict):
        raise ValueError(f"kindmap entry '{key}' must be an object")
    type_name = str(raw.get("type") or key)
    primitive = str(raw.get("primitive") or DEFAULT_PRIMITIVE)
    if primitive not in VALID_PRIMITIVES:
        raise ValueError(
            f"kindmap entry '{key}' has invalid primitive '{primitive}'; "
            f"expected one of {sorted(VALID_PRIMITIVES)}"
        )
    aliases_raw = raw.get("aliases") or []
    if not isinstance(aliases_raw, list):
        raise ValueError(f"kindmap entry '{key}' aliases must be a list")
    aliases = [str(a) for a in aliases_raw]
    expert_routing = bool(raw.get("expert_routing", primitive == ENTITY_PRIMITIVE))
    extractable = bool(raw.get("extractable", False))
    return KindEntry(
        type=type_name,
        primitive=primitive,
        extractable=extractable,
        expert_routing=expert_routing,
        aliases=aliases,
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"kindmap at {path} must be a YAML mapping")
    return doc


def _bundled_kindmap_text() -> str:
    pkg = resources.files("seed_cli") / "resources" / "gbrain" / "kindmap.yml"
    return pkg.read_text(encoding="utf-8")


def _parse_doc(doc: Dict[str, Any]) -> Dict[str, KindEntry]:
    out: Dict[str, KindEntry] = {}
    for key, raw in doc.items():
        out[str(key)] = _coerce_entry(str(key), raw)
    return out


_DEFAULT_CACHE: Optional[Dict[str, KindEntry]] = None


def default_kindmap() -> Dict[str, KindEntry]:
    """Return a fresh copy of the bundled default kindmap."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = _parse_doc(yaml.safe_load(_bundled_kindmap_text()))
    return {key: KindEntry(**asdict(entry)) for key, entry in _DEFAULT_CACHE.items()}


# Convenience alias for callers that want the dict at import time.
DEFAULT_KINDMAP = default_kindmap()


def load_kindmap(
    *,
    base: Optional[Path] = None,
    extra_path: Optional[Path] = None,
) -> Dict[str, KindEntry]:
    """Load the effective kindmap.

    Precedence (later overrides earlier):
        1. bundled default
        2. ``<base>/.seed/gbrain/kindmap.yml``
        3. ``extra_path`` (typically ``--kindmap PATH``)
    """
    merged = default_kindmap()
    if base is not None:
        repo_path = Path(base) / ".seed" / "gbrain" / "kindmap.yml"
        if repo_path.exists():
            merged = _apply_override(merged, _parse_doc(_load_yaml(repo_path)))
    if extra_path is not None:
        merged = _apply_override(merged, _parse_doc(_load_yaml(Path(extra_path))))
    return merged


def _apply_override(
    base_map: Dict[str, KindEntry],
    override: Dict[str, KindEntry],
) -> Dict[str, KindEntry]:
    out = dict(base_map)
    for key, entry in override.items():
        if key in out:
            out[key] = out[key].merged_with(entry)
        else:
            out[key] = entry
    return out


def resolve(
    kindmap: Dict[str, KindEntry],
    kind: str,
    *,
    tags: Iterable[str] = (),
) -> KindEntry:
    """Resolve a !kind token, applying graceful defaults for unknown kinds."""
    kind = (kind or "").strip()
    if kind and kind in kindmap:
        entry = KindEntry(**asdict(kindmap[kind]))
    else:
        derived_type = _slugify(kind) if kind else "note"
        entry = KindEntry(type=derived_type)
    for tag in tags or []:
        token = str(tag).strip().lower()
        if not token:
            continue
        if token in {"active", "remote", "draft", "wip", "legacy"}:
            continue
        if token in {"extractable", "extract"}:
            entry.extractable = True
            continue
        if token in {"expert", "expert_routing"}:
            entry.expert_routing = True
            continue
        if token not in entry.aliases:
            entry.aliases.append(token)
    return entry


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in value).strip("-").lower() or "note"


def dump_kindmap(kindmap: Dict[str, KindEntry]) -> str:
    """Serialise the resolved kindmap for the audit lock-file."""
    plain: Dict[str, Dict[str, Any]] = {}
    for key in sorted(kindmap):
        entry = kindmap[key]
        plain[key] = {
            "type": entry.type,
            "primitive": entry.primitive,
            "extractable": entry.extractable,
            "expert_routing": entry.expert_routing,
            "aliases": list(entry.aliases),
        }
    return yaml.safe_dump(plain, sort_keys=False)
