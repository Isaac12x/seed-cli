"""Reverse-drift reconciliation (`seed amend`).

GBrain can mutate its own schema or relocate pages; ``seed amend`` folds those
changes back into the ``.seed`` spec so the spec remains the single source of
truth (PRD §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from ..capture import DEFAULT_IGNORE, capture_nodes
from ..parsers import Node, parse_spec, render_node_text
from . import gbrain_cli


@dataclass
class AmendChange:
    path: str
    is_dir: bool
    policy: str  # adopt | ignore | quarantine
    kind: Optional[str] = None
    source: str = "fs"  # fs | gbrain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "is_dir": self.is_dir,
            "policy": self.policy,
            "kind": self.kind,
            "source": self.source,
        }


@dataclass
class AmendResult:
    spec_path: Path
    base: Path
    changes: List[AmendChange] = field(default_factory=list)
    ignored_prefixes: List[str] = field(default_factory=list)
    quarantine_dir: str = "_inbox/"
    spec_rewritten: bool = False
    ignore_file_path: Optional[Path] = None
    reexport_summary: Optional[Dict[str, Any]] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "spec": str(self.spec_path),
            "changes": [c.to_dict() for c in self.changes],
            "ignored_prefixes": list(self.ignored_prefixes),
            "quarantine_dir": self.quarantine_dir,
            "spec_rewritten": self.spec_rewritten,
            "reexport": self.reexport_summary,
        }


def _spec_path_set(nodes: List[Node]) -> Set[str]:
    return {n.relpath.as_posix() for n in nodes if str(n.relpath) not in (".", "")}


def _node_prefixes(nodes: List[Node]) -> Set[str]:
    """Return prefixes (top-level dirs) the spec already covers."""
    prefixes: Set[str] = set()
    for node in nodes:
        if not node.is_dir:
            continue
        rel = node.relpath.as_posix()
        if rel in (".", ""):
            continue
        prefixes.add(rel.rstrip("/").split("/", 1)[0] + "/")
    return prefixes


def _load_ignore_file(base: Path) -> Tuple[Path, List[str]]:
    path = base / ".seed" / "gbrain" / "ignore.yml"
    if not path.exists():
        return path, []
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return path, []
    entries = doc.get("ignore") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        return path, []
    return path, [str(x) for x in entries]


def _write_ignore_file(path: Path, entries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"ignore": sorted(set(entries))}
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _gbrain_prefixes_in_active_pack() -> Dict[str, str]:
    """Return ``{prefix: type_name}`` from gbrain's active pack JSON."""
    doc = gbrain_cli.schema_show_json()
    if not doc:
        return {}
    out: Dict[str, str] = {}
    for pt in doc.get("page_types") or []:
        name = pt.get("name")
        for prefix in pt.get("path_prefixes") or []:
            out[str(prefix)] = str(name)
    return out


def _is_ignored(path: str, ignore_prefixes: List[str]) -> bool:
    return any(path == pref or path.startswith(pref) for pref in ignore_prefixes)


def detect_drift(
    spec_path: Path,
    base: Path,
    *,
    from_fs: bool,
    from_gbrain: bool,
    spec_ignore: List[str],
) -> Tuple[List[Tuple[str, bool]], Dict[str, str]]:
    """Return ``(new_fs_paths, gbrain_prefix_to_type)``.

    ``new_fs_paths`` is a list of ``(relpath, is_dir)`` present on disk but
    absent from the spec (and not in the spec_ignore list).
    """
    _, spec_nodes = parse_spec(str(spec_path), base=base)
    known = _spec_path_set(spec_nodes)
    new_fs: List[Tuple[str, bool]] = []

    if from_fs:
        ignore = list(spec_ignore) + [
            "brain.seed",
            ".gbrain/**",
            "gbrain.yml",
            "hooks/**",
            "hooks",
            "*.seed",
            "*.tree",
        ]
        fs_nodes = capture_nodes(base, ignore=ignore)
        for node in fs_nodes:
            rel = node.relpath.as_posix()
            if rel in (".", ""):
                continue
            if rel in known:
                continue
            if any(parent in known for parent in _path_ancestors(rel)):
                # already implicitly covered by a known directory? still report so user can decide
                pass
            new_fs.append((rel, node.is_dir))

    gbrain_prefixes: Dict[str, str] = {}
    if from_gbrain:
        gbrain_prefixes = _gbrain_prefixes_in_active_pack()
    return new_fs, gbrain_prefixes


def _path_ancestors(path: str) -> List[str]:
    parts = path.split("/")
    return ["/".join(parts[: i + 1]) for i in range(len(parts) - 1)]


def _classify(
    drift: List[Tuple[str, bool]],
    *,
    spec_prefixes: Set[str],
    gbrain_prefixes: Dict[str, str],
    policy: str,
    quarantine_dir: str,
) -> List[AmendChange]:
    changes: List[AmendChange] = []
    for rel, is_dir in drift:
        top = rel.split("/", 1)[0] + "/"
        kind: Optional[str] = None
        for prefix, type_name in gbrain_prefixes.items():
            if rel.startswith(prefix) or top == prefix:
                kind = type_name
                break
        # Already-covered prefixes don't need new spec entries: keep them out
        # but still allow adopt-policy if user asks for them explicitly.
        if top in spec_prefixes and policy != "quarantine":
            continue
        changes.append(
            AmendChange(
                path=rel,
                is_dir=is_dir,
                policy=policy,
                kind=kind,
                source="gbrain" if kind and policy != "ignore" else "fs",
            )
        )
    return changes


def _render_spec_with_changes(
    spec_path: Path,
    spec_nodes: List[Node],
    changes: List[AmendChange],
    quarantine_dir: str,
) -> str:
    """Render the spec as tree text, augmented with adopt / quarantine entries."""
    nodes = list(spec_nodes)
    seen_paths = _spec_path_set(nodes)

    for change in changes:
        if change.policy == "ignore":
            continue
        if change.policy == "quarantine":
            target_dir = quarantine_dir.rstrip("/") + "/"
            target_path = target_dir + change.path.replace("/", "__")
            if target_path in seen_paths:
                continue
            metadata: Dict[str, Any] = {}
            if change.kind:
                metadata["kind"] = change.kind
            nodes.append(
                Node(relpath=Path(target_path), is_dir=change.is_dir, metadata=metadata)
            )
            seen_paths.add(target_path)
            continue
        if change.policy == "adopt":
            if change.path in seen_paths:
                continue
            metadata = {"kind": change.kind} if change.kind else {}
            nodes.append(
                Node(relpath=Path(change.path), is_dir=change.is_dir, metadata=metadata)
            )
            seen_paths.add(change.path)

    nodes_sorted = sorted(nodes, key=lambda n: n.relpath.as_posix())
    lines = [render_node_text(n) for n in nodes_sorted if str(n.relpath) not in (".", "")]
    return "\n".join(lines) + "\n"


def amend(
    *,
    spec: str,
    base: Path,
    policy: str = "adopt",
    from_fs: bool = True,
    from_gbrain: bool = False,
    quarantine_dir: str = "_inbox/",
    reexport: bool = False,
    dry_run: bool = False,
) -> AmendResult:
    spec_path = Path(spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"spec '{spec}' does not exist")
    if policy not in {"adopt", "ignore", "quarantine"}:
        raise ValueError("policy must be one of adopt | ignore | quarantine")

    ignore_path, ignore_entries = _load_ignore_file(base)
    drift, gbrain_prefixes = detect_drift(
        spec_path,
        base,
        from_fs=from_fs,
        from_gbrain=from_gbrain,
        spec_ignore=ignore_entries,
    )
    drift = [(p, d) for p, d in drift if not _is_ignored(p, ignore_entries)]

    _, spec_nodes = parse_spec(str(spec_path), base=base)
    changes = _classify(
        drift,
        spec_prefixes=_node_prefixes(spec_nodes),
        gbrain_prefixes=gbrain_prefixes,
        policy=policy,
        quarantine_dir=quarantine_dir,
    )

    result = AmendResult(
        spec_path=spec_path,
        base=base,
        changes=changes,
        ignored_prefixes=list(ignore_entries),
        quarantine_dir=quarantine_dir,
        ignore_file_path=ignore_path,
    )

    if dry_run:
        return result

    if policy == "ignore":
        new_entries = list(ignore_entries) + [c.path for c in changes]
        _write_ignore_file(ignore_path, new_entries)
        return result

    if changes:
        text = _render_spec_with_changes(spec_path, spec_nodes, changes, quarantine_dir)
        spec_path.write_text(text, encoding="utf-8")
        result.spec_rewritten = True

    if reexport:
        from .exporter import export_gbrain

        export = export_gbrain(
            spec=str(spec_path),
            base=base,
            skip_validate=True,
        )
        result.reexport_summary = export.to_summary()

    return result
