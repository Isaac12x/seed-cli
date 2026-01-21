"""seed_cli.structure_lock

Structure locking - lock filesystem to a versioned spec.

Features:
- Version management: store specs as .seed/structures/v1.tree, v2.tree, etc.
- Watch mode: continuously enforce structure
- Upgrade/downgrade between versions

Usage:
    seed lock set my-spec.tree          # Lock to spec (creates version)
    seed lock set my-spec.tree -v v2    # Lock with explicit version
    seed lock list                      # List available versions
    seed lock status                    # Show current lock
    seed lock upgrade v2                # Switch to v2
    seed lock downgrade v1              # Switch back to v1
    seed lock watch                     # Watch and enforce continuously
"""

import json
import shutil
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .parsers import parse_spec
from .match import match, match_plan
from .logging import get_logger

log = get_logger("structure_lock")

STRUCTURES_DIR = ".seed/structures"
LOCK_FILE = ".seed/structure-lock.json"


@dataclass
class StructureLockInfo:
    """Current structure lock state."""
    active_version: Optional[str]
    spec_path: Optional[str]
    locked_at: Optional[float]
    versions: List[str]


def _ensure_structures_dir(base: Path) -> Path:
    """Ensure .seed/structures directory exists."""
    structures_dir = base / STRUCTURES_DIR
    structures_dir.mkdir(parents=True, exist_ok=True)
    return structures_dir


def _get_lock_file(base: Path) -> Path:
    """Get path to structure lock file."""
    return base / LOCK_FILE


def _load_lock_info(base: Path) -> Dict:
    """Load structure lock info from file."""
    lock_file = _get_lock_file(base)
    if lock_file.exists():
        return json.loads(lock_file.read_text())
    return {}


def _save_lock_info(base: Path, info: Dict) -> None:
    """Save structure lock info to file."""
    lock_file = _get_lock_file(base)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(json.dumps(info, indent=2))


def list_versions(base: Path) -> List[Tuple[str, Path]]:
    """List available structure versions.

    Returns list of (version_name, path) tuples sorted by version number.
    """
    structures_dir = base / STRUCTURES_DIR
    if not structures_dir.exists():
        return []

    versions = []
    for f in structures_dir.iterdir():
        if f.is_file() and f.suffix == ".tree":
            # Extract version from filename (v1.tree -> v1)
            version = f.stem
            versions.append((version, f))

    # Sort by version number (v1, v2, v10, etc.)
    def version_key(item):
        v = item[0]
        match = re.match(r'v(\d+)', v)
        if match:
            return int(match.group(1))
        return 0

    return sorted(versions, key=version_key)


def get_next_version(base: Path) -> str:
    """Get the next available version number."""
    versions = list_versions(base)
    if not versions:
        return "v1"

    # Find highest version number
    max_num = 0
    for v, _ in versions:
        match = re.match(r'v(\d+)', v)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return f"v{max_num + 1}"


def get_status(base: Path) -> StructureLockInfo:
    """Get current structure lock status."""
    info = _load_lock_info(base)
    versions = [v for v, _ in list_versions(base)]

    return StructureLockInfo(
        active_version=info.get("active_version"),
        spec_path=info.get("spec_path"),
        locked_at=info.get("locked_at"),
        versions=versions,
    )


def set_lock(
    base: Path,
    spec_path: str,
    version: Optional[str] = None,
) -> Tuple[str, Path]:
    """Set a structure lock from a spec file.

    Copies the spec to .seed/structures/<version>.tree and sets it as active.

    Args:
        base: Base directory
        spec_path: Path to spec file
        version: Version name (auto-increments if not provided)

    Returns:
        (version, stored_path) tuple
    """
    base = base.resolve()
    spec = Path(spec_path).resolve()

    if not spec.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    # Determine version
    if version is None:
        version = get_next_version(base)

    # Ensure version starts with 'v'
    if not version.startswith("v"):
        version = f"v{version}"

    # Copy spec to structures directory
    structures_dir = _ensure_structures_dir(base)
    stored_path = structures_dir / f"{version}.tree"
    shutil.copy2(spec, stored_path)

    # Update lock info
    info = {
        "active_version": version,
        "spec_path": stored_path.relative_to(base).as_posix(),
        "locked_at": time.time(),
    }
    _save_lock_info(base, info)

    return version, stored_path


def switch_version(
    base: Path,
    version: str,
    *,
    dangerous: bool = False,
    dry_run: bool = False,
) -> Dict:
    """Switch to a different structure version.

    Args:
        base: Base directory
        version: Version to switch to
        dangerous: Required to apply changes
        dry_run: Preview changes without applying

    Returns:
        Result dict with created/updated/deleted/skipped counts
    """
    base = base.resolve()

    # Normalize version name
    if not version.startswith("v"):
        version = f"v{version}"

    # Find version
    structures_dir = base / STRUCTURES_DIR
    spec_path = structures_dir / f"{version}.tree"

    if not spec_path.exists():
        raise FileNotFoundError(f"Version not found: {version}")

    if dry_run:
        # Return plan without executing
        plan = match_plan(str(spec_path), base)
        return {
            "plan": plan,
            "version": version,
            "dry_run": True,
        }

    if not dangerous:
        raise RuntimeError("switch_version requires dangerous=True")

    # Apply the new structure
    result = match(
        str(spec_path),
        base,
        dangerous=True,
        dry_run=False,
    )

    # Update lock info
    info = {
        "active_version": version,
        "spec_path": spec_path.relative_to(base).as_posix(),
        "locked_at": time.time(),
    }
    _save_lock_info(base, info)

    return result


def watch(
    base: Path,
    interval: float = 1.0,
    callback=None,
) -> None:
    """Watch filesystem and enforce structure continuously.

    Args:
        base: Base directory
        interval: Check interval in seconds
        callback: Optional callback for events (type, message)

    Raises:
        RuntimeError: If no structure lock is active
    """
    base = base.resolve()
    status = get_status(base)

    if not status.active_version:
        raise RuntimeError("No structure lock is active. Use 'seed lock set' first.")

    spec_path = base / status.spec_path
    if not spec_path.exists():
        raise RuntimeError(f"Locked spec not found: {status.spec_path}")

    def notify(msg_type: str, message: str):
        if callback:
            callback(msg_type, message)
        else:
            print(f"[{msg_type}] {message}")

    notify("info", f"Watching structure: {status.active_version}")
    notify("info", f"Spec: {status.spec_path}")
    notify("info", f"Checking every {interval}s (Ctrl+C to stop)")

    try:
        while True:
            try:
                # Get plan to see if any changes needed
                plan = match_plan(str(spec_path), base)

                if plan.steps:
                    notify("drift", f"Structure drift detected: {len(plan.steps)} changes needed")

                    # Apply fixes automatically
                    result = match(
                        str(spec_path),
                        base,
                        dangerous=True,
                        dry_run=False,
                    )

                    created = result.get("created", 0)
                    deleted = result.get("deleted", 0)
                    notify("fixed", f"Applied: {created} created, {deleted} deleted")

            except Exception as e:
                notify("error", f"Watch error: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        notify("info", "Watch stopped")
