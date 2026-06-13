"""End-to-end orchestration for ``seed export gbrain``."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__ as SEED_CLI_VERSION
from ..parsers import parse_spec, Node, parse_any
from ..spec_history import SPECS_DIR, list_spec_versions, get_spec_version
from . import gbrain_cli
from .activator import activate, ActivationResult
from .compiler import compile_pack, CompiledPack, DEFAULT_EXTENDS, DEFAULT_MIN_VERSION
from .kindmap import KindEntry, dump_kindmap, load_kindmap
from .manifest import dump_manifest, lint_manifest, manifest_hash
from .migration import compute_mapping_rules, derive_predecessor_version


SHORT_HASH_LEN = 8


@dataclass
class GbrainExportResult:
    pack_path: Path
    source_json_path: Path
    kindmap_lock_path: Path
    manifest: Dict[str, Any]
    spec_hash: str
    diagnostics: List[str] = field(default_factory=list)
    lint_problems: List[str] = field(default_factory=list)
    validate_ok: Optional[bool] = None
    activation: Optional[ActivationResult] = None
    migration_submitted: Optional[bool] = None
    migration_message: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "name": self.manifest.get("name"),
            "version": self.manifest.get("version"),
            "pack": str(self.pack_path),
            "spec_hash": self.spec_hash,
            "page_types": [p.get("name") for p in self.manifest.get("page_types", [])],
            "diagnostics": list(self.diagnostics),
            "lint_problems": list(self.lint_problems),
            "validate_ok": self.validate_ok,
            "activation": _activation_summary(self.activation),
            "migration_from": self.manifest.get("migration_from"),
            "mapping_rules": self.manifest.get("mapping_rules"),
            "migration_submitted": self.migration_submitted,
            "migration_message": self.migration_message,
        }


def _activation_summary(act: Optional[ActivationResult]) -> Optional[Dict[str, Any]]:
    if act is None:
        return None
    return {
        "repo_yaml": str(act.repo_yaml) if act.repo_yaml else None,
        "home_pack": str(act.home_pack) if act.home_pack else None,
        "schema_use_ok": act.schema_use_ok,
        "notes": list(act.notes),
    }


def _normalize_spec_text(text: str) -> str:
    """Normalise for hashing: strip blank lines, trailing whitespace, comments."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
    return "\n".join(out)


def _spec_hash(text: str) -> str:
    return hashlib.sha256(_normalize_spec_text(text).encode("utf-8")).hexdigest()


def _short_hash(value: str) -> str:
    return value[:SHORT_HASH_LEN]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return slug or "seed-pack"


def derive_pack_name(spec_path: Path, base: Path) -> str:
    """Pick a sensible default pack name from the spec/base dir."""
    stem = spec_path.stem
    candidate = stem if stem and stem != "brain" else base.name
    if not candidate or candidate == ".":
        candidate = "seed-pack"
    return _slugify(candidate) + "-pack" if not candidate.endswith("-pack") else _slugify(candidate)


def resolve_version(spec_text: str, mode: str) -> str:
    """Resolve ``--version-from`` modes (``hash``, ``spec``, or literal).

    Default ``hash`` mode encodes the spec hash into the patch field so the
    version is plain ``M.m.p`` semver (gbrain rejects build metadata).
    """
    if not mode or mode == "hash":
        h = _spec_hash(spec_text)
        patch = int(h[:6], 16) % 9999  # 0-9998, stable per spec
        return f"0.0.{patch}"
    if mode == "spec":
        match = re.search(r"v(\d+)", spec_text)
        if match:
            major = int(match.group(1))
            return f"{major}.0.0"
        return "0.1.0"
    return mode


def _resolve_migration(
    *,
    base: Path,
    spec_text: str,
    migrate: str,
    migrate_from: Optional[str],
    new_nodes: List["Node"],
    pack_name: str,
    kindmap: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return ``{migration_from, mapping_rules}`` when a predecessor exists."""
    if migrate == "off" and not migrate_from:
        return None

    prev_text: Optional[str] = None
    prev_version_label: Optional[str] = None
    if migrate_from:
        explicit = Path(migrate_from)
        if explicit.exists():
            prev_text = explicit.read_text(encoding="utf-8")
            prev_version_label = explicit.stem
        else:
            return None
    else:
        from .migration import find_predecessor_spec

        predecessor = find_predecessor_spec(base, spec_text)
        if not predecessor:
            return None
        prev_version_int, prev_text = predecessor
        prev_version_label = f"0.{prev_version_int}.x"

    if not prev_text or not prev_version_label:
        return None

    from .migration import parse_spec_text, compute_mapping_rules

    try:
        prev_nodes = parse_spec_text(prev_text)
    except Exception:
        return None
    rules = compute_mapping_rules(
        prev_nodes,
        new_nodes,
        name=pack_name,
        kindmap=kindmap,
    )
    return {
        "migration_from": {"pack": pack_name, "version": prev_version_label},
        "mapping_rules": rules,
    }


def _write_source_json(
    path: Path,
    *,
    spec_path: Path,
    spec_hash: str,
    seed_cli_version: str,
) -> None:
    payload = {
        "spec_path": str(spec_path),
        "spec_hash": spec_hash,
        "seed_cli_version": seed_cli_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _maybe_validate(name: str, *, install_target: Optional[Path]) -> Optional[bool]:
    """Validate via the real ``gbrain`` CLI when available."""
    if not gbrain_cli.is_available():
        return None
    if install_target is None:
        return None  # Can't validate a pack gbrain hasn't seen on disk.
    result = gbrain_cli.schema_validate(name)
    return result.ok


def export_gbrain(
    *,
    spec: str,
    base: Path,
    out: Optional[Path] = None,
    name: Optional[str] = None,
    extends: Optional[str] = DEFAULT_EXTENDS,
    kindmap_path: Optional[Path] = None,
    version_from: str = "hash",
    install: bool = False,
    activate_mode: str = "none",
    vars: Optional[Dict[str, str]] = None,
    gbrain_min_version: str = DEFAULT_MIN_VERSION,
    dry_run: bool = False,
    home_schema_packs: Optional[Path] = None,
    skip_validate: bool = False,
    migrate: str = "off",
    migrate_from: Optional[str] = None,
) -> GbrainExportResult:
    """Compile ``spec`` and (optionally) install + activate the resulting pack."""
    spec_path = Path(spec)
    out_dir = Path(out) if out else (base / ".gbrain" / "pack")
    pack_yaml = out_dir / "pack.yaml"
    kindmap_lock = out_dir / "kindmap.lock.yml"
    source_json = out_dir / "source.json"

    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    if not spec_text:
        raise FileNotFoundError(f"spec '{spec}' is missing or empty")
    spec_hash = _spec_hash(spec_text)

    _, nodes = parse_spec(str(spec_path), vars=vars, base=base)

    kindmap = load_kindmap(base=base, extra_path=kindmap_path)
    pack_name = name or derive_pack_name(spec_path, base)
    pack_version = resolve_version(spec_text, version_from)

    migration_info = _resolve_migration(
        base=base,
        spec_text=spec_text,
        migrate=migrate,
        migrate_from=migrate_from,
        new_nodes=nodes,
        pack_name=pack_name,
        kindmap=kindmap,
    )

    compiled: CompiledPack = compile_pack(
        nodes,
        name=pack_name,
        version=pack_version,
        kindmap=kindmap,
        extends=extends,
        gbrain_min_version=gbrain_min_version,
        migration_from=migration_info["migration_from"] if migration_info else None,
        mapping_rules=migration_info["mapping_rules"] if migration_info else None,
    )

    yaml_text = dump_manifest(compiled.manifest)
    lint_problems = lint_manifest(compiled.manifest)

    validate_ok: Optional[bool] = None
    activation: Optional[ActivationResult] = None

    if dry_run:
        return GbrainExportResult(
            pack_path=pack_yaml,
            source_json_path=source_json,
            kindmap_lock_path=kindmap_lock,
            manifest=compiled.manifest,
            spec_hash=spec_hash,
            diagnostics=list(compiled.diagnostics),
            lint_problems=lint_problems,
            validate_ok=None,
            activation=None,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pack_yaml.write_text(yaml_text, encoding="utf-8")
    kindmap_lock.write_text(dump_kindmap(kindmap), encoding="utf-8")
    _write_source_json(
        source_json,
        spec_path=spec_path,
        spec_hash=spec_hash,
        seed_cli_version=SEED_CLI_VERSION,
    )

    install_target: Optional[Path] = None
    if install or activate_mode in {"home", "both"}:
        from .activator import install_to_home

        install_target = install_to_home(
            out_dir, name=pack_name, home_root=home_schema_packs
        )

    if not skip_validate and install_target is not None:
        validate_ok = _maybe_validate(pack_name, install_target=install_target)

    if activate_mode != "none":
        activation = activate(
            pack_dir=out_dir,
            repo_dir=base,
            name=pack_name,
            mode=activate_mode,
            home_root=home_schema_packs,
        )

    migration_submitted: Optional[bool] = None
    migration_message: Optional[str] = None
    if migration_info and migrate == "auto" and install_target is not None:
        if gbrain_cli.is_available():
            result_call = gbrain_cli.submit_unify_types(pack_name, dry_run=False)
            migration_submitted = result_call.ok
            migration_message = (
                result_call.stderr.strip() or result_call.stdout.strip()
            )
        else:
            migration_message = "gbrain binary unavailable; submit unify-types manually"
    elif migration_info and migrate == "prompt":
        migration_message = (
            f"successor pack detected; run `gbrain jobs submit unify-types "
            f"--allow-protected --params '{{\"target_pack\":\"{pack_name}\"}}'`"
        )

    return GbrainExportResult(
        pack_path=pack_yaml,
        source_json_path=source_json,
        kindmap_lock_path=kindmap_lock,
        manifest=compiled.manifest,
        spec_hash=spec_hash,
        diagnostics=list(compiled.diagnostics),
        lint_problems=lint_problems,
        validate_ok=validate_ok,
        activation=activation,
        migration_submitted=migration_submitted,
        migration_message=migration_message,
    )
