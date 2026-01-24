"""Tests for seed_cli.snapshot module.

Comprehensive tests for snapshot and restore functionality.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from seed_cli.snapshot import (
    SNAPSHOTS_DIR,
    MAX_SNAPSHOTS,
    SnapshotManifest,
    _generate_snapshot_id,
    _get_snapshots_dir,
    _cleanup_old_snapshots,
    create_snapshot,
    list_snapshots,
    get_snapshot,
    revert_snapshot,
    delete_snapshot,
)
from seed_cli.planning import PlanResult, PlanStep


# -----------------------------------------------------------------------------
# Tests for _generate_snapshot_id
# -----------------------------------------------------------------------------

def test_generate_snapshot_id_format():
    """Snapshot ID should start with 'snap_' prefix."""
    snapshot_id = _generate_snapshot_id()
    assert snapshot_id.startswith("snap_")


def test_generate_snapshot_id_contains_timestamp():
    """Snapshot ID should contain a numeric timestamp."""
    snapshot_id = _generate_snapshot_id()
    timestamp_part = snapshot_id.replace("snap_", "")
    assert timestamp_part.isdigit()


def test_generate_snapshot_id_unique():
    """Consecutive snapshot IDs should be unique (different timestamps)."""
    id1 = _generate_snapshot_id()
    time.sleep(0.002)  # Small delay to ensure different timestamp
    id2 = _generate_snapshot_id()
    assert id1 != id2


# -----------------------------------------------------------------------------
# Tests for _get_snapshots_dir
# -----------------------------------------------------------------------------

def test_get_snapshots_dir_returns_correct_path(tmp_path):
    """Should return the correct .seed/snapshots path."""
    result = _get_snapshots_dir(tmp_path)
    expected = tmp_path / SNAPSHOTS_DIR
    assert result == expected


def test_get_snapshots_dir_does_not_create_dir(tmp_path):
    """Should not create the directory, just return the path."""
    result = _get_snapshots_dir(tmp_path)
    assert not result.exists()


# -----------------------------------------------------------------------------
# Tests for _cleanup_old_snapshots
# -----------------------------------------------------------------------------

def test_cleanup_old_snapshots_does_nothing_when_empty(tmp_path):
    """Should handle non-existent snapshots directory gracefully."""
    _cleanup_old_snapshots(tmp_path)  # Should not raise


def test_cleanup_old_snapshots_keeps_under_max(tmp_path):
    """Should keep snapshots when count is under MAX_SNAPSHOTS."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    # Create 5 snapshots
    for i in range(5):
        snap_dir = snapshots_dir / f"snap_{i}"
        snap_dir.mkdir()
        manifest = {
            "id": f"snap_{i}",
            "created_at": time.time() + i,
            "operation": "test",
            "spec_path": None,
            "base_path": str(tmp_path),
            "files": [],
            "directories": [],
            "plan_summary": None,
        }
        (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    _cleanup_old_snapshots(tmp_path, keep=10)

    remaining = list(snapshots_dir.iterdir())
    assert len(remaining) == 5


def test_cleanup_old_snapshots_removes_oldest(tmp_path):
    """Should remove oldest snapshots, keeping newest ones."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    # Create 5 snapshots with increasing timestamps
    for i in range(5):
        snap_dir = snapshots_dir / f"snap_{i}"
        snap_dir.mkdir()
        manifest = {
            "id": f"snap_{i}",
            "created_at": float(i),  # snap_0 is oldest, snap_4 is newest
            "operation": "test",
            "spec_path": None,
            "base_path": str(tmp_path),
            "files": [],
            "directories": [],
            "plan_summary": None,
        }
        (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    _cleanup_old_snapshots(tmp_path, keep=3)

    remaining = [d.name for d in snapshots_dir.iterdir() if d.is_dir()]
    # Should keep snap_4, snap_3, snap_2 (newest)
    assert len(remaining) == 3
    assert "snap_4" in remaining
    assert "snap_3" in remaining
    assert "snap_2" in remaining
    assert "snap_1" not in remaining
    assert "snap_0" not in remaining


# -----------------------------------------------------------------------------
# Tests for create_snapshot
# -----------------------------------------------------------------------------

def test_create_snapshot_returns_id(tmp_path):
    """Should return a snapshot ID."""
    plan = PlanResult(steps=[], add=0, change=0, delete=0, delete_skipped=0)
    snapshot_id = create_snapshot(tmp_path, plan, "test")
    assert snapshot_id.startswith("snap_")


def test_create_snapshot_creates_directory_structure(tmp_path):
    """Should create .seed/snapshots/<id>/files directory."""
    plan = PlanResult(steps=[], add=0, change=0, delete=0, delete_skipped=0)
    snapshot_id = create_snapshot(tmp_path, plan, "test")

    snapshot_dir = tmp_path / SNAPSHOTS_DIR / snapshot_id
    assert snapshot_dir.exists()
    assert (snapshot_dir / "files").exists()
    assert (snapshot_dir / "manifest.json").exists()


def test_create_snapshot_backs_up_files_for_update(tmp_path):
    """Should backup files that will be updated."""
    # Create file to be updated
    file_path = tmp_path / "important.txt"
    file_path.write_text("original content")

    plan = PlanResult(
        steps=[PlanStep("update", "important.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )
    snapshot_id = create_snapshot(tmp_path, plan, "test")

    # Check backup was created
    backup_path = tmp_path / SNAPSHOTS_DIR / snapshot_id / "files" / "important.txt"
    assert backup_path.exists()
    assert backup_path.read_text() == "original content"


def test_create_snapshot_backs_up_files_for_delete(tmp_path):
    """Should backup files that will be deleted."""
    # Create file to be deleted
    file_path = tmp_path / "to_delete.txt"
    file_path.write_text("delete me")

    plan = PlanResult(
        steps=[PlanStep("delete", "to_delete.txt", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )
    snapshot_id = create_snapshot(tmp_path, plan, "test")

    backup_path = tmp_path / SNAPSHOTS_DIR / snapshot_id / "files" / "to_delete.txt"
    assert backup_path.exists()
    assert backup_path.read_text() == "delete me"


def test_create_snapshot_backs_up_directory_contents_on_delete(tmp_path):
    """Should backup all files in a directory being deleted."""
    # Create directory with files
    dir_path = tmp_path / "delete_dir"
    dir_path.mkdir()
    (dir_path / "file1.txt").write_text("content1")
    (dir_path / "file2.txt").write_text("content2")

    plan = PlanResult(
        steps=[PlanStep("delete", "delete_dir", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )
    snapshot_id = create_snapshot(tmp_path, plan, "test")

    files_dir = tmp_path / SNAPSHOTS_DIR / snapshot_id / "files"
    assert (files_dir / "delete_dir" / "file1.txt").exists()
    assert (files_dir / "delete_dir" / "file2.txt").exists()


def test_create_snapshot_preserves_nested_paths(tmp_path):
    """Should preserve nested directory structure in backups."""
    # Create nested directory structure
    nested_dir = tmp_path / "deep" / "nested" / "path"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "file.txt"
    nested_file.write_text("deeply nested")

    plan = PlanResult(
        steps=[PlanStep("update", "deep/nested/path/file.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )
    snapshot_id = create_snapshot(tmp_path, plan, "test")

    backup_path = tmp_path / SNAPSHOTS_DIR / snapshot_id / "files" / "deep" / "nested" / "path" / "file.txt"
    assert backup_path.exists()
    assert backup_path.read_text() == "deeply nested"


def test_create_snapshot_manifest_contains_metadata(tmp_path):
    """Should create manifest with correct metadata."""
    plan = PlanResult(
        steps=[PlanStep("create", "new.txt", "missing")],
        add=1, change=0, delete=0, delete_skipped=0,
    )
    snapshot_id = create_snapshot(tmp_path, plan, "apply", "/path/to/spec.tree")

    manifest_path = tmp_path / SNAPSHOTS_DIR / snapshot_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["id"] == snapshot_id
    assert manifest["operation"] == "apply"
    assert manifest["spec_path"] == "/path/to/spec.tree"
    assert "created_at" in manifest
    assert manifest["plan_summary"]["add"] == 1


def test_create_snapshot_triggers_cleanup(tmp_path):
    """Should cleanup old snapshots after creating new one."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    # Create MAX_SNAPSHOTS + 1 existing snapshots
    for i in range(MAX_SNAPSHOTS + 1):
        snap_dir = snapshots_dir / f"old_snap_{i}"
        snap_dir.mkdir()
        (snap_dir / "files").mkdir()
        manifest = {
            "id": f"old_snap_{i}",
            "created_at": float(i),
            "operation": "test",
            "spec_path": None,
            "base_path": str(tmp_path),
            "files": [],
            "directories": [],
            "plan_summary": None,
        }
        (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create a new snapshot (should trigger cleanup)
    plan = PlanResult(steps=[], add=0, change=0, delete=0, delete_skipped=0)
    create_snapshot(tmp_path, plan, "test")

    # Should now have MAX_SNAPSHOTS
    remaining = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    assert len(remaining) == MAX_SNAPSHOTS


def test_create_snapshot_handles_nonexistent_paths(tmp_path):
    """Should handle gracefully when files in plan don't exist."""
    plan = PlanResult(
        steps=[PlanStep("delete", "nonexistent.txt", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )
    # Should not raise
    snapshot_id = create_snapshot(tmp_path, plan, "test")
    assert snapshot_id.startswith("snap_")


# -----------------------------------------------------------------------------
# Tests for list_snapshots
# -----------------------------------------------------------------------------

def test_list_snapshots_returns_empty_when_none(tmp_path):
    """Should return empty list when no snapshots exist."""
    result = list_snapshots(tmp_path)
    assert result == []


def test_list_snapshots_returns_manifests(tmp_path):
    """Should return list of SnapshotManifest objects."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    snap_dir = snapshots_dir / "snap_123"
    snap_dir.mkdir()
    manifest = {
        "id": "snap_123",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [],
        "directories": [],
        "plan_summary": None,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = list_snapshots(tmp_path)
    assert len(result) == 1
    assert isinstance(result[0], SnapshotManifest)
    assert result[0].id == "snap_123"


def test_list_snapshots_sorted_newest_first(tmp_path):
    """Should return snapshots sorted by creation time, newest first."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    for i in range(3):
        snap_dir = snapshots_dir / f"snap_{i}"
        snap_dir.mkdir()
        manifest = {
            "id": f"snap_{i}",
            "created_at": float(i),  # snap_2 is newest
            "operation": "test",
            "spec_path": None,
            "base_path": str(tmp_path),
            "files": [],
            "directories": [],
            "plan_summary": None,
        }
        (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = list_snapshots(tmp_path)
    assert result[0].id == "snap_2"
    assert result[1].id == "snap_1"
    assert result[2].id == "snap_0"


def test_list_snapshots_skips_invalid_manifests(tmp_path):
    """Should skip snapshots with corrupted manifests."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    # Valid snapshot
    valid_dir = snapshots_dir / "snap_valid"
    valid_dir.mkdir()
    valid_manifest = {
        "id": "snap_valid",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [],
        "directories": [],
        "plan_summary": None,
    }
    (valid_dir / "manifest.json").write_text(json.dumps(valid_manifest))

    # Invalid snapshot (corrupted JSON)
    invalid_dir = snapshots_dir / "snap_invalid"
    invalid_dir.mkdir()
    (invalid_dir / "manifest.json").write_text("not valid json")

    result = list_snapshots(tmp_path)
    assert len(result) == 1
    assert result[0].id == "snap_valid"


def test_list_snapshots_skips_non_directories(tmp_path):
    """Should skip non-directory items in snapshots folder."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    # Create a file (not a directory)
    (snapshots_dir / "not_a_snapshot.txt").write_text("hello")

    # Create valid snapshot
    snap_dir = snapshots_dir / "snap_123"
    snap_dir.mkdir()
    manifest = {
        "id": "snap_123",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [],
        "directories": [],
        "plan_summary": None,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = list_snapshots(tmp_path)
    assert len(result) == 1


# -----------------------------------------------------------------------------
# Tests for get_snapshot
# -----------------------------------------------------------------------------

def test_get_snapshot_returns_manifest(tmp_path):
    """Should return SnapshotManifest for existing snapshot."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snapshots_dir.mkdir(parents=True)

    snap_dir = snapshots_dir / "snap_123"
    snap_dir.mkdir()
    manifest = {
        "id": "snap_123",
        "created_at": 12345.0,
        "operation": "match",
        "spec_path": "/spec.tree",
        "base_path": str(tmp_path),
        "files": [{"path": "a.txt", "type": "file", "backed_up": True}],
        "directories": ["dir1"],
        "plan_summary": {"add": 1, "change": 0, "delete": 0},
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = get_snapshot(tmp_path, "snap_123")
    assert result is not None
    assert result.id == "snap_123"
    assert result.operation == "match"
    assert result.spec_path == "/spec.tree"
    assert len(result.files) == 1


def test_get_snapshot_returns_none_for_missing(tmp_path):
    """Should return None when snapshot doesn't exist."""
    result = get_snapshot(tmp_path, "nonexistent_snap")
    assert result is None


# -----------------------------------------------------------------------------
# Tests for revert_snapshot
# -----------------------------------------------------------------------------

def test_revert_snapshot_restores_files(tmp_path):
    """Should restore backed up files."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snap_dir = snapshots_dir / "snap_123"
    files_dir = snap_dir / "files"
    files_dir.mkdir(parents=True)

    # Backup a file
    (files_dir / "restored.txt").write_text("restored content")

    manifest = {
        "id": "snap_123",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [{"path": "restored.txt", "type": "file", "backed_up": True}],
        "directories": [],
        "plan_summary": None,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = revert_snapshot(tmp_path, "snap_123")

    assert result["restored"] == 1
    assert (tmp_path / "restored.txt").exists()
    assert (tmp_path / "restored.txt").read_text() == "restored content"


def test_revert_snapshot_restores_directories(tmp_path):
    """Should restore backed up directories."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snap_dir = snapshots_dir / "snap_123"
    snap_dir.mkdir(parents=True)
    (snap_dir / "files").mkdir()

    manifest = {
        "id": "snap_123",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [],
        "directories": ["restored_dir"],
        "plan_summary": None,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = revert_snapshot(tmp_path, "snap_123")

    assert result["restored"] >= 1
    assert (tmp_path / "restored_dir").exists()
    assert (tmp_path / "restored_dir").is_dir()


def test_revert_snapshot_dry_run_no_changes(tmp_path):
    """Dry run should not make any changes."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snap_dir = snapshots_dir / "snap_123"
    files_dir = snap_dir / "files"
    files_dir.mkdir(parents=True)

    (files_dir / "test.txt").write_text("content")

    manifest = {
        "id": "snap_123",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [{"path": "test.txt", "type": "file", "backed_up": True}],
        "directories": [],
        "plan_summary": None,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = revert_snapshot(tmp_path, "snap_123", dry_run=True)

    assert result["dry_run"] is True
    assert result["restored"] == 1  # Counts what would be restored
    assert not (tmp_path / "test.txt").exists()  # But file not actually created


def test_revert_snapshot_raises_for_missing(tmp_path):
    """Should raise FileNotFoundError for missing snapshot."""
    with pytest.raises(FileNotFoundError):
        revert_snapshot(tmp_path, "nonexistent_snap")


def test_revert_snapshot_restores_nested_files(tmp_path):
    """Should restore files with nested paths correctly."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snap_dir = snapshots_dir / "snap_123"
    files_dir = snap_dir / "files"
    nested_backup = files_dir / "a" / "b" / "c"
    nested_backup.mkdir(parents=True)
    (nested_backup / "nested.txt").write_text("nested content")

    manifest = {
        "id": "snap_123",
        "created_at": time.time(),
        "operation": "test",
        "spec_path": None,
        "base_path": str(tmp_path),
        "files": [{"path": "a/b/c/nested.txt", "type": "file", "backed_up": True}],
        "directories": [],
        "plan_summary": None,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest))

    result = revert_snapshot(tmp_path, "snap_123")

    assert (tmp_path / "a" / "b" / "c" / "nested.txt").exists()
    assert (tmp_path / "a" / "b" / "c" / "nested.txt").read_text() == "nested content"


# -----------------------------------------------------------------------------
# Tests for delete_snapshot
# -----------------------------------------------------------------------------

def test_delete_snapshot_removes_snapshot(tmp_path):
    """Should remove snapshot directory."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snap_dir = snapshots_dir / "snap_123"
    snap_dir.mkdir(parents=True)
    (snap_dir / "manifest.json").write_text("{}")

    result = delete_snapshot(tmp_path, "snap_123")

    assert result is True
    assert not snap_dir.exists()


def test_delete_snapshot_returns_false_for_missing(tmp_path):
    """Should return False when snapshot doesn't exist."""
    result = delete_snapshot(tmp_path, "nonexistent_snap")
    assert result is False


def test_delete_snapshot_removes_all_contents(tmp_path):
    """Should remove snapshot directory including all contents."""
    snapshots_dir = tmp_path / SNAPSHOTS_DIR
    snap_dir = snapshots_dir / "snap_123"
    files_dir = snap_dir / "files" / "deep" / "nested"
    files_dir.mkdir(parents=True)
    (files_dir / "file.txt").write_text("content")
    (snap_dir / "manifest.json").write_text("{}")

    result = delete_snapshot(tmp_path, "snap_123")

    assert result is True
    assert not snap_dir.exists()


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

def test_full_snapshot_workflow(tmp_path):
    """Test complete create -> list -> get -> revert -> delete workflow."""
    # Create some files
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    # Create snapshot with planned updates
    plan = PlanResult(
        steps=[
            PlanStep("update", "file1.txt", "checksum_drift"),
            PlanStep("delete", "file2.txt", "extra"),
        ],
        add=0, change=1, delete=1, delete_skipped=0,
    )
    snapshot_id = create_snapshot(tmp_path, plan, "test")

    # Verify snapshot created
    snapshots = list_snapshots(tmp_path)
    assert len(snapshots) == 1
    assert snapshots[0].id == snapshot_id

    # Get specific snapshot
    snap = get_snapshot(tmp_path, snapshot_id)
    assert snap is not None
    assert snap.operation == "test"

    # Simulate file changes
    (tmp_path / "file1.txt").write_text("modified content")
    (tmp_path / "file2.txt").unlink()

    # Revert
    result = revert_snapshot(tmp_path, snapshot_id)
    assert result["restored"] >= 1

    # Verify files restored
    assert (tmp_path / "file1.txt").read_text() == "content1"
    assert (tmp_path / "file2.txt").read_text() == "content2"

    # Delete snapshot
    assert delete_snapshot(tmp_path, snapshot_id) is True
    assert get_snapshot(tmp_path, snapshot_id) is None
