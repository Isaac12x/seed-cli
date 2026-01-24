"""Tests for seed_cli.structure_lock module.

Comprehensive tests for structure locking and version management functionality.
"""

import json
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from seed_cli.structure_lock import (
    STRUCTURES_DIR,
    LOCK_FILE,
    StructureLockInfo,
    _ensure_structures_dir,
    _get_lock_file,
    _load_lock_info,
    _save_lock_info,
    list_versions,
    get_next_version,
    get_status,
    set_lock,
    switch_version,
    watch,
)


# -----------------------------------------------------------------------------
# Tests for _ensure_structures_dir
# -----------------------------------------------------------------------------

def test_ensure_structures_dir_creates_directory(tmp_path):
    """Should create .seed/structures directory."""
    result = _ensure_structures_dir(tmp_path)
    assert result.exists()
    assert result.is_dir()
    assert result == tmp_path / STRUCTURES_DIR


def test_ensure_structures_dir_returns_existing(tmp_path):
    """Should return existing directory without error."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    result = _ensure_structures_dir(tmp_path)
    assert result == structures_dir


def test_ensure_structures_dir_creates_parent_dirs(tmp_path):
    """Should create parent .seed directory if needed."""
    result = _ensure_structures_dir(tmp_path)
    assert (tmp_path / ".seed").exists()


# -----------------------------------------------------------------------------
# Tests for _get_lock_file
# -----------------------------------------------------------------------------

def test_get_lock_file_returns_correct_path(tmp_path):
    """Should return correct lock file path."""
    result = _get_lock_file(tmp_path)
    assert result == tmp_path / LOCK_FILE


def test_get_lock_file_does_not_create(tmp_path):
    """Should not create the lock file."""
    result = _get_lock_file(tmp_path)
    assert not result.exists()


# -----------------------------------------------------------------------------
# Tests for _load_lock_info
# -----------------------------------------------------------------------------

def test_load_lock_info_empty_when_no_file(tmp_path):
    """Should return empty dict when lock file doesn't exist."""
    result = _load_lock_info(tmp_path)
    assert result == {}


def test_load_lock_info_returns_contents(tmp_path):
    """Should return contents of lock file."""
    lock_file = tmp_path / LOCK_FILE
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    info = {"active_version": "v1", "locked_at": 12345.0}
    lock_file.write_text(json.dumps(info))

    result = _load_lock_info(tmp_path)
    assert result == info


# -----------------------------------------------------------------------------
# Tests for _save_lock_info
# -----------------------------------------------------------------------------

def test_save_lock_info_creates_file(tmp_path):
    """Should create lock file with info."""
    info = {"active_version": "v1", "locked_at": time.time()}

    _save_lock_info(tmp_path, info)

    lock_file = tmp_path / LOCK_FILE
    assert lock_file.exists()
    saved = json.loads(lock_file.read_text())
    assert saved["active_version"] == "v1"


def test_save_lock_info_creates_parent_dirs(tmp_path):
    """Should create parent directories."""
    info = {"test": "data"}

    _save_lock_info(tmp_path, info)

    assert (tmp_path / ".seed").exists()


def test_save_lock_info_overwrites_existing(tmp_path):
    """Should overwrite existing lock info."""
    _save_lock_info(tmp_path, {"version": "v1"})
    _save_lock_info(tmp_path, {"version": "v2"})

    result = _load_lock_info(tmp_path)
    assert result["version"] == "v2"


# -----------------------------------------------------------------------------
# Tests for list_versions
# -----------------------------------------------------------------------------

def test_list_versions_empty_when_no_dir(tmp_path):
    """Should return empty list when structures dir doesn't exist."""
    result = list_versions(tmp_path)
    assert result == []


def test_list_versions_returns_tree_files(tmp_path):
    """Should return .tree files in structures directory."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")
    (structures_dir / "v2.tree").write_text("content")

    result = list_versions(tmp_path)
    versions = [v for v, _ in result]

    assert "v1" in versions
    assert "v2" in versions


def test_list_versions_sorted_by_number(tmp_path):
    """Should sort versions by numeric value."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    # Create out of order
    (structures_dir / "v10.tree").write_text("content")
    (structures_dir / "v2.tree").write_text("content")
    (structures_dir / "v1.tree").write_text("content")

    result = list_versions(tmp_path)
    versions = [v for v, _ in result]

    assert versions == ["v1", "v2", "v10"]


def test_list_versions_ignores_non_tree_files(tmp_path):
    """Should ignore non-.tree files."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")
    (structures_dir / "notes.txt").write_text("notes")
    (structures_dir / "v2.json").write_text("{}")

    result = list_versions(tmp_path)
    versions = [v for v, _ in result]

    assert len(versions) == 1
    assert "v1" in versions


def test_list_versions_returns_paths(tmp_path):
    """Should return (version, path) tuples."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")

    result = list_versions(tmp_path)

    assert len(result) == 1
    assert result[0][0] == "v1"
    assert result[0][1] == structures_dir / "v1.tree"


def test_list_versions_handles_non_standard_names(tmp_path):
    """Should handle versions without 'v' prefix with sort fallback."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")
    (structures_dir / "custom.tree").write_text("content")

    result = list_versions(tmp_path)
    versions = [v for v, _ in result]

    assert "v1" in versions
    assert "custom" in versions


# -----------------------------------------------------------------------------
# Tests for get_next_version
# -----------------------------------------------------------------------------

def test_get_next_version_first(tmp_path):
    """Should return 'v1' when no versions exist."""
    result = get_next_version(tmp_path)
    assert result == "v1"


def test_get_next_version_increments(tmp_path):
    """Should increment from highest version."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")
    (structures_dir / "v2.tree").write_text("content")

    result = get_next_version(tmp_path)
    assert result == "v3"


def test_get_next_version_handles_gaps(tmp_path):
    """Should use highest version even with gaps."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")
    (structures_dir / "v5.tree").write_text("content")

    result = get_next_version(tmp_path)
    assert result == "v6"


# -----------------------------------------------------------------------------
# Tests for get_status
# -----------------------------------------------------------------------------

def test_get_status_no_lock(tmp_path):
    """Should return status with no active lock."""
    result = get_status(tmp_path)

    assert isinstance(result, StructureLockInfo)
    assert result.active_version is None
    assert result.versions == []


def test_get_status_with_lock(tmp_path):
    """Should return status with active lock info."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("content")

    _save_lock_info(tmp_path, {
        "active_version": "v1",
        "spec_path": ".seed/structures/v1.tree",
        "locked_at": 12345.0,
    })

    result = get_status(tmp_path)

    assert result.active_version == "v1"
    assert result.spec_path == ".seed/structures/v1.tree"
    assert result.locked_at == 12345.0
    assert "v1" in result.versions


def test_get_status_lists_all_versions(tmp_path):
    """Should include all available versions."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("content")
    (structures_dir / "v2.tree").write_text("content")
    (structures_dir / "v3.tree").write_text("content")

    result = get_status(tmp_path)

    assert "v1" in result.versions
    assert "v2" in result.versions
    assert "v3" in result.versions


# -----------------------------------------------------------------------------
# Tests for set_lock
# -----------------------------------------------------------------------------

def test_set_lock_copies_spec(tmp_path):
    """Should copy spec to structures directory."""
    spec_file = tmp_path / "myspec.tree"
    spec_file.write_text("src/\nfile.py")

    version, path = set_lock(tmp_path, str(spec_file))

    assert path.exists()
    assert path.read_text() == "src/\nfile.py"


def test_set_lock_auto_increments_version(tmp_path):
    """Should auto-increment version when not specified."""
    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    version1, _ = set_lock(tmp_path, str(spec_file))
    version2, _ = set_lock(tmp_path, str(spec_file))

    assert version1 == "v1"
    assert version2 == "v2"


def test_set_lock_uses_specified_version(tmp_path):
    """Should use specified version name."""
    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    version, path = set_lock(tmp_path, str(spec_file), version="v5")

    assert version == "v5"
    assert path.name == "v5.tree"


def test_set_lock_adds_v_prefix(tmp_path):
    """Should add 'v' prefix if not present."""
    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    version, _ = set_lock(tmp_path, str(spec_file), version="3")

    assert version == "v3"


def test_set_lock_updates_lock_info(tmp_path):
    """Should update lock info file."""
    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    set_lock(tmp_path, str(spec_file))

    info = _load_lock_info(tmp_path)
    assert info["active_version"] == "v1"
    assert "locked_at" in info


def test_set_lock_raises_for_missing_spec(tmp_path):
    """Should raise FileNotFoundError for missing spec."""
    with pytest.raises(FileNotFoundError):
        set_lock(tmp_path, "/nonexistent/spec.tree")


# -----------------------------------------------------------------------------
# Tests for switch_version
# -----------------------------------------------------------------------------

def test_switch_version_requires_dangerous(tmp_path):
    """Should require dangerous flag to apply changes."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("file.txt")

    with pytest.raises(RuntimeError, match="dangerous=True"):
        switch_version(tmp_path, "v1", dangerous=False)


def test_switch_version_dry_run_returns_plan(tmp_path):
    """Should return plan without applying on dry_run."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("file.txt")

    result = switch_version(tmp_path, "v1", dry_run=True)

    assert result["dry_run"] is True
    assert "plan" in result
    assert result["version"] == "v1"


def test_switch_version_normalizes_version_name(tmp_path):
    """Should add 'v' prefix if missing."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("file.txt")

    result = switch_version(tmp_path, "1", dry_run=True)

    assert result["version"] == "v1"


def test_switch_version_raises_for_missing(tmp_path):
    """Should raise for non-existent version."""
    with pytest.raises(FileNotFoundError, match="Version not found"):
        switch_version(tmp_path, "v99", dangerous=True)


def test_switch_version_updates_lock_info(tmp_path):
    """Should update lock info after successful switch."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("file.txt")
    (structures_dir / "v2.tree").write_text("file.txt\notherfile.txt")

    # Initial lock to v1
    _save_lock_info(tmp_path, {"active_version": "v1"})

    # Switch to v2
    switch_version(tmp_path, "v2", dangerous=True)

    info = _load_lock_info(tmp_path)
    assert info["active_version"] == "v2"


# -----------------------------------------------------------------------------
# Tests for watch
# -----------------------------------------------------------------------------

def test_watch_requires_active_lock(tmp_path):
    """Should raise if no structure lock is active."""
    with pytest.raises(RuntimeError, match="No structure lock is active"):
        watch(tmp_path, interval=0.1)


def test_watch_requires_spec_exists(tmp_path):
    """Should raise if locked spec file doesn't exist."""
    _save_lock_info(tmp_path, {
        "active_version": "v1",
        "spec_path": ".seed/structures/v1.tree",
        "locked_at": time.time(),
    })

    with pytest.raises(RuntimeError, match="Locked spec not found"):
        watch(tmp_path, interval=0.1)


def test_watch_calls_callback(tmp_path):
    """Should call callback with status messages."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("file.txt")

    _save_lock_info(tmp_path, {
        "active_version": "v1",
        "spec_path": ".seed/structures/v1.tree",
        "locked_at": time.time(),
    })

    messages = []

    def callback(msg_type, message):
        messages.append((msg_type, message))
        if len(messages) >= 3:  # Stop after receiving initial messages
            raise KeyboardInterrupt()

    try:
        watch(tmp_path, interval=0.1, callback=callback)
    except KeyboardInterrupt:
        pass

    # Should have received info messages
    assert any(msg_type == "info" for msg_type, _ in messages)


def test_watch_detects_drift(tmp_path):
    """Should detect and fix structure drift."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)
    (structures_dir / "v1.tree").write_text("required.txt")

    _save_lock_info(tmp_path, {
        "active_version": "v1",
        "spec_path": ".seed/structures/v1.tree",
        "locked_at": time.time(),
    })

    messages = []
    checks = [0]

    def callback(msg_type, message):
        messages.append((msg_type, message))
        checks[0] += 1
        if checks[0] >= 5:
            raise KeyboardInterrupt()

    try:
        watch(tmp_path, interval=0.05, callback=callback)
    except KeyboardInterrupt:
        pass

    # Should have created the required file
    assert (tmp_path / "required.txt").exists()


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

def test_full_lock_workflow(tmp_path):
    """Test complete set -> status -> switch workflow."""
    # Create initial spec
    spec_file = tmp_path / "spec1.tree"
    spec_file.write_text("file1.txt")

    # Set lock
    version1, path1 = set_lock(tmp_path, str(spec_file))
    assert version1 == "v1"

    # Check status
    status = get_status(tmp_path)
    assert status.active_version == "v1"
    assert "v1" in status.versions

    # Create second spec
    spec_file2 = tmp_path / "spec2.tree"
    spec_file2.write_text("file1.txt\nfile2.txt")

    # Set new version
    version2, path2 = set_lock(tmp_path, str(spec_file2))
    assert version2 == "v2"

    # Status should show both versions
    status = get_status(tmp_path)
    assert status.active_version == "v2"
    assert "v1" in status.versions
    assert "v2" in status.versions

    # List versions
    versions = list_versions(tmp_path)
    assert len(versions) == 2


def test_version_sorting_comprehensive(tmp_path):
    """Test comprehensive version sorting."""
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    # Create many versions
    for i in [1, 2, 3, 10, 11, 20, 100]:
        (structures_dir / f"v{i}.tree").write_text(f"v{i}")

    result = list_versions(tmp_path)
    versions = [v for v, _ in result]

    # Should be in numeric order
    assert versions == ["v1", "v2", "v3", "v10", "v11", "v20", "v100"]


def test_switch_applies_structure(tmp_path):
    """Test that switch actually applies the structure."""
    # Create two versions
    structures_dir = tmp_path / STRUCTURES_DIR
    structures_dir.mkdir(parents=True)

    (structures_dir / "v1.tree").write_text("onlyinv1.txt")
    (structures_dir / "v2.tree").write_text("onlyinv2.txt")

    # Set initial lock
    _save_lock_info(tmp_path, {
        "active_version": "v1",
        "spec_path": ".seed/structures/v1.tree",
        "locked_at": time.time(),
    })

    # Apply v1
    switch_version(tmp_path, "v1", dangerous=True)
    assert (tmp_path / "onlyinv1.txt").exists()

    # Switch to v2
    switch_version(tmp_path, "v2", dangerous=True)
    assert (tmp_path / "onlyinv2.txt").exists()
    # v1 file should be removed (match behavior)
    assert not (tmp_path / "onlyinv1.txt").exists()
