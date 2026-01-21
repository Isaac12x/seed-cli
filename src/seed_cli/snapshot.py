"""seed_cli.snapshot

Snapshot and restore functionality for undo/revert operations.

Snapshots capture the filesystem state before changes, allowing rollback
if something goes wrong.

Storage: .seed/snapshots/<id>/
  - manifest.json: metadata and file list
  - files/: backed up file contents
"""

import json
import shutil
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

from .planning import PlanResult, PlanStep, DEFAULT_IGNORE
from .logging import get_logger

log = get_logger("snapshot")

SNAPSHOTS_DIR = ".seed/snapshots"
MAX_SNAPSHOTS = 10  # Keep last N snapshots


@dataclass
class SnapshotManifest:
    """Snapshot metadata."""
    id: str
    created_at: float
    operation: str  # apply, match, sync, etc.
    spec_path: Optional[str]
    base_path: str
    files: List[Dict]  # [{path, type, backed_up}]
    directories: List[str]
    plan_summary: Optional[Dict]


def _generate_snapshot_id() -> str:
    """Generate a unique snapshot ID."""
    timestamp = int(time.time() * 1000)
    return f"snap_{timestamp}"


def _get_snapshots_dir(base: Path) -> Path:
    """Get the snapshots directory."""
    return base / SNAPSHOTS_DIR


def _cleanup_old_snapshots(base: Path, keep: int = MAX_SNAPSHOTS) -> None:
    """Remove old snapshots, keeping only the most recent ones."""
    snapshots_dir = _get_snapshots_dir(base)
    if not snapshots_dir.exists():
        return

    snapshots = list_snapshots(base)
    if len(snapshots) <= keep:
        return

    # Sort by creation time (newest first)
    snapshots.sort(key=lambda s: s.created_at, reverse=True)

    # Remove old ones
    for snap in snapshots[keep:]:
        snap_dir = snapshots_dir / snap.id
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
            log.debug(f"Removed old snapshot: {snap.id}")


def create_snapshot(
    base: Path,
    plan: PlanResult,
    operation: str,
    spec_path: Optional[str] = None,
) -> str:
    """Create a snapshot before applying changes.

    Backs up files that will be modified or deleted.

    Args:
        base: Base directory
        plan: The plan about to be executed
        operation: Operation name (apply, match, sync)
        spec_path: Path to spec file (for reference)

    Returns:
        Snapshot ID
    """
    base = base.resolve()
    snapshot_id = _generate_snapshot_id()

    snapshots_dir = _get_snapshots_dir(base)
    snapshot_dir = snapshots_dir / snapshot_id
    files_dir = snapshot_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    files_info: List[Dict] = []
    directories: List[str] = []

    # Determine what needs to be backed up
    paths_to_backup: Set[str] = set()

    for step in plan.steps:
        if step.op in ("update", "delete"):
            paths_to_backup.add(step.path)
        elif step.op == "create":
            # For creates, back up if file already exists (shouldn't, but safety)
            target = base / step.path
            if target.exists():
                paths_to_backup.add(step.path)

    # Back up files
    for rel_path in sorted(paths_to_backup):
        target = base / rel_path
        if not target.exists():
            continue

        if target.is_dir():
            directories.append(rel_path)
            files_info.append({
                "path": rel_path,
                "type": "dir",
                "backed_up": False,
            })
        else:
            # Back up file content
            backup_path = files_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_path)

            files_info.append({
                "path": rel_path,
                "type": "file",
                "backed_up": True,
            })

    # Also record directories that will be deleted
    for step in plan.steps:
        if step.op == "delete":
            target = base / step.path
            if target.exists() and target.is_dir():
                # Back up entire directory tree
                for item in target.rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(base).as_posix()
                        backup_path = files_dir / rel
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, backup_path)
                        files_info.append({
                            "path": rel,
                            "type": "file",
                            "backed_up": True,
                        })

    # Create manifest
    manifest = SnapshotManifest(
        id=snapshot_id,
        created_at=time.time(),
        operation=operation,
        spec_path=spec_path,
        base_path=base.as_posix(),
        files=files_info,
        directories=directories,
        plan_summary={
            "add": plan.add,
            "change": plan.change,
            "delete": plan.delete,
        } if plan else None,
    )

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))

    # Cleanup old snapshots
    _cleanup_old_snapshots(base)

    log.debug(f"Created snapshot: {snapshot_id}")
    return snapshot_id


def list_snapshots(base: Path) -> List[SnapshotManifest]:
    """List available snapshots.

    Returns list of SnapshotManifest objects, newest first.
    """
    base = base.resolve()
    snapshots_dir = _get_snapshots_dir(base)

    if not snapshots_dir.exists():
        return []

    snapshots = []
    for snap_dir in snapshots_dir.iterdir():
        if not snap_dir.is_dir():
            continue

        manifest_path = snap_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            data = json.loads(manifest_path.read_text())
            snapshots.append(SnapshotManifest(**data))
        except Exception as e:
            log.warning(f"Failed to load snapshot {snap_dir.name}: {e}")

    # Sort newest first
    snapshots.sort(key=lambda s: s.created_at, reverse=True)
    return snapshots


def get_snapshot(base: Path, snapshot_id: str) -> Optional[SnapshotManifest]:
    """Get a specific snapshot by ID."""
    base = base.resolve()
    snapshot_dir = _get_snapshots_dir(base) / snapshot_id
    manifest_path = snapshot_dir / "manifest.json"

    if not manifest_path.exists():
        return None

    data = json.loads(manifest_path.read_text())
    return SnapshotManifest(**data)


def revert_snapshot(
    base: Path,
    snapshot_id: str,
    *,
    dry_run: bool = False,
) -> Dict:
    """Revert to a previous snapshot.

    Args:
        base: Base directory
        snapshot_id: ID of snapshot to revert to
        dry_run: Preview without making changes

    Returns:
        dict with restored/deleted counts
    """
    base = base.resolve()
    snapshot_dir = _get_snapshots_dir(base) / snapshot_id
    files_dir = snapshot_dir / "files"
    manifest_path = snapshot_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

    manifest = get_snapshot(base, snapshot_id)
    if not manifest:
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

    restored = 0
    deleted = 0
    actions: List[str] = []

    # Restore backed up files
    for file_info in manifest.files:
        rel_path = file_info["path"]
        file_type = file_info["type"]
        backed_up = file_info.get("backed_up", False)

        target = base / rel_path

        if backed_up and file_type == "file":
            backup_path = files_dir / rel_path
            if backup_path.exists():
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, target)
                actions.append(f"RESTORE  {rel_path}")
                restored += 1

    # Restore directories
    for dir_path in manifest.directories:
        target = base / dir_path
        if not target.exists():
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
            actions.append(f"MKDIR    {dir_path}")
            restored += 1

    # Delete files that were created by the operation
    # We need to figure out what was created - look at current state vs snapshot
    # For simplicity, we'll delete files that weren't in the snapshot but exist now
    # This is a bit tricky - for now, we'll just restore what was backed up

    return {
        "restored": restored,
        "deleted": deleted,
        "actions": actions,
        "dry_run": dry_run,
        "snapshot_id": snapshot_id,
    }


def delete_snapshot(base: Path, snapshot_id: str) -> bool:
    """Delete a specific snapshot.

    Returns True if deleted, False if not found.
    """
    base = base.resolve()
    snapshot_dir = _get_snapshots_dir(base) / snapshot_id

    if not snapshot_dir.exists():
        return False

    shutil.rmtree(snapshot_dir)
    return True
