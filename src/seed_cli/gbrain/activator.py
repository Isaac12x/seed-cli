"""Install + activate generated gbrain packs.

Two activation tiers (PRD §3 resolution chain):
  - tier 5: write/update ``gbrain.yml`` (repo-checked, portable, highest among user-controllable tiers)
  - tier 6: shell ``gbrain schema use <name>`` (machine-local, overridable)

The pack files always live at ``~/.gbrain/schema-packs/<name>/`` so they can be
loaded by name from any tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import gbrain_cli


HOME_SCHEMA_PACKS = Path.home() / ".gbrain" / "schema-packs"


@dataclass
class ActivationResult:
    repo_yaml: Optional[Path] = None
    home_pack: Optional[Path] = None
    schema_use_ok: Optional[bool] = None
    notes: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def install_to_home(pack_dir: Path, *, name: str, home_root: Optional[Path] = None) -> Path:
    """Copy ``pack.yaml`` from ``pack_dir`` to ``~/.gbrain/schema-packs/<name>/pack.yaml``."""
    root = home_root or HOME_SCHEMA_PACKS
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "pack.yaml"
    target.write_text((pack_dir / "pack.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return target


def write_repo_activation(repo_dir: Path, *, name: str) -> Path:
    """Write/update ``gbrain.yml`` in ``repo_dir`` with ``schema: { pack: <name> }``."""
    path = repo_dir / "gbrain.yml"
    doc: Dict = {}
    if path.exists():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(existing, dict):
                doc = existing
        except yaml.YAMLError:
            doc = {}
    schema_block = doc.get("schema") if isinstance(doc.get("schema"), dict) else {}
    schema_block["pack"] = name
    doc["schema"] = schema_block
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def activate(
    *,
    pack_dir: Path,
    repo_dir: Path,
    name: str,
    mode: str,
    home_root: Optional[Path] = None,
) -> ActivationResult:
    """Apply the activation mode (``repo``, ``home``, ``both``, ``none``)."""
    result = ActivationResult()
    if mode not in {"repo", "home", "both", "none"}:
        raise ValueError(f"invalid activation mode '{mode}'")
    if mode == "none":
        result.notes.append("activation skipped (mode=none)")
        return result

    if mode in {"home", "both"}:
        target = install_to_home(pack_dir, name=name, home_root=home_root)
        result.home_pack = target
        if gbrain_cli.is_available():
            use = gbrain_cli.schema_use(name)
            result.schema_use_ok = use.ok
            if not use.ok:
                result.notes.append(f"gbrain schema use {name} failed: {use.stderr.strip()}")
        else:
            result.notes.append("gbrain binary unavailable; skipping `schema use`")

    if mode in {"repo", "both"}:
        yaml_path = write_repo_activation(repo_dir, name=name)
        result.repo_yaml = yaml_path

    return result
