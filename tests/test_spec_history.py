"""Tests for seed_cli.spec_history module.

Comprehensive tests for versioned spec capture and comparison functionality.
"""

import os
from pathlib import Path
from datetime import datetime

import pytest

from seed_cli.spec_history import (
    SPECS_DIR,
    _get_specs_dir,
    _get_versions,
    _get_next_version,
    _specs_differ,
    capture_spec,
    list_spec_versions,
    get_spec_version,
    get_current_spec,
    diff_spec_versions,
)


# -----------------------------------------------------------------------------
# Tests for _get_specs_dir
# -----------------------------------------------------------------------------

def test_get_specs_dir_creates_directory(tmp_path):
    """Should create .seed/specs directory."""
    result = _get_specs_dir(tmp_path)
    assert result.exists()
    assert result.is_dir()
    assert result == tmp_path / SPECS_DIR


def test_get_specs_dir_returns_existing(tmp_path):
    """Should return existing directory without error."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    result = _get_specs_dir(tmp_path)
    assert result == specs_dir


def test_get_specs_dir_creates_parent_dirs(tmp_path):
    """Should create parent directories if needed."""
    result = _get_specs_dir(tmp_path)
    assert (tmp_path / ".seed").exists()


# -----------------------------------------------------------------------------
# Tests for _get_versions
# -----------------------------------------------------------------------------

def test_get_versions_empty_dir(tmp_path):
    """Should return empty list for empty directory."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    result = _get_versions(specs_dir)
    assert result == []


def test_get_versions_returns_sorted(tmp_path):
    """Should return versions sorted by number."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    # Create versions out of order
    (specs_dir / "v3.tree").write_text("v3")
    (specs_dir / "v1.tree").write_text("v1")
    (specs_dir / "v10.tree").write_text("v10")
    (specs_dir / "v2.tree").write_text("v2")

    result = _get_versions(specs_dir)

    assert result[0][0] == 1  # v1
    assert result[1][0] == 2  # v2
    assert result[2][0] == 3  # v3
    assert result[3][0] == 10  # v10 (not v10 before v2)


def test_get_versions_ignores_non_version_files(tmp_path):
    """Should ignore files that don't match version pattern."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("v1")
    (specs_dir / "current.tree").write_text("current")
    (specs_dir / "notes.txt").write_text("notes")
    (specs_dir / "abc.tree").write_text("abc")

    result = _get_versions(specs_dir)

    assert len(result) == 1
    assert result[0][0] == 1


def test_get_versions_returns_paths(tmp_path):
    """Should return (version, path) tuples."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("v1")

    result = _get_versions(specs_dir)

    assert len(result) == 1
    assert result[0][0] == 1
    assert result[0][1] == specs_dir / "v1.tree"


# -----------------------------------------------------------------------------
# Tests for _get_next_version
# -----------------------------------------------------------------------------

def test_get_next_version_first(tmp_path):
    """Should return 1 when no versions exist."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    result = _get_next_version(specs_dir)
    assert result == 1


def test_get_next_version_increments(tmp_path):
    """Should return highest version + 1."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("v1")
    (specs_dir / "v2.tree").write_text("v2")
    (specs_dir / "v5.tree").write_text("v5")

    result = _get_next_version(specs_dir)
    assert result == 6


# -----------------------------------------------------------------------------
# Tests for _specs_differ
# -----------------------------------------------------------------------------

def test_specs_differ_identical():
    """Should return False for identical specs."""
    spec = "src/\nfile.py"
    assert _specs_differ(spec, spec) is False


def test_specs_differ_whitespace_normalized():
    """Should ignore whitespace differences."""
    spec1 = "src/\n  file.py  \n\n"
    spec2 = "src/\nfile.py"
    assert _specs_differ(spec1, spec2) is False


def test_specs_differ_different_content():
    """Should return True for different content."""
    spec1 = "src/\nfile1.py"
    spec2 = "src/\nfile2.py"
    assert _specs_differ(spec1, spec2) is True


def test_specs_differ_additional_lines():
    """Should detect additional lines."""
    spec1 = "src/\nfile.py"
    spec2 = "src/\nfile.py\nextra.py"
    assert _specs_differ(spec1, spec2) is True


def test_specs_differ_empty_lines_ignored():
    """Should ignore empty lines."""
    spec1 = "src/\n\n\nfile.py"
    spec2 = "src/\nfile.py"
    assert _specs_differ(spec1, spec2) is False


# -----------------------------------------------------------------------------
# Tests for capture_spec
# -----------------------------------------------------------------------------

def test_capture_spec_creates_first_version(tmp_path):
    """Should create v1.tree on first capture."""
    # Create some files to capture
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("main")

    result = capture_spec(tmp_path)

    assert result is not None
    version, path = result
    assert version == 1
    assert path.name == "v1.tree"
    assert path.exists()


def test_capture_spec_increments_version(tmp_path):
    """Should increment version number on each capture."""
    # Create files
    (tmp_path / "file1.txt").write_text("file1")

    result1 = capture_spec(tmp_path, force=True)
    assert result1[0] == 1

    # Add more files and capture again
    (tmp_path / "file2.txt").write_text("file2")
    result2 = capture_spec(tmp_path)

    assert result2[0] == 2


def test_capture_spec_skips_if_unchanged(tmp_path):
    """Should not create new version if content unchanged."""
    (tmp_path / "file.txt").write_text("content")

    result1 = capture_spec(tmp_path)
    assert result1 is not None

    # Capture again without changes
    result2 = capture_spec(tmp_path)
    assert result2 is None  # No new version


def test_capture_spec_force_creates_even_if_unchanged(tmp_path):
    """Should create new version when force=True even if unchanged."""
    (tmp_path / "file.txt").write_text("content")

    result1 = capture_spec(tmp_path)
    result2 = capture_spec(tmp_path, force=True)

    assert result2 is not None
    assert result2[0] == 2


def test_capture_spec_includes_timestamp_header(tmp_path):
    """Should include timestamp in captured spec."""
    (tmp_path / "file.txt").write_text("content")

    result = capture_spec(tmp_path)
    content = result[1].read_text()

    assert "# Captured:" in content


def test_capture_spec_creates_current_symlink(tmp_path):
    """Should create/update current.tree symlink."""
    (tmp_path / "file.txt").write_text("content")

    capture_spec(tmp_path)

    current_path = tmp_path / SPECS_DIR / "current.tree"
    assert current_path.exists()


def test_capture_spec_updates_current_on_new_version(tmp_path):
    """Should update current.tree to point to latest version."""
    (tmp_path / "file1.txt").write_text("content1")
    capture_spec(tmp_path)

    (tmp_path / "file2.txt").write_text("content2")
    capture_spec(tmp_path)

    current_path = tmp_path / SPECS_DIR / "current.tree"
    # Current should reference v2 content (has both files)
    content = current_path.read_text()
    assert "file2.txt" in content


def test_capture_spec_with_ignore_patterns(tmp_path):
    """Should respect ignore patterns."""
    (tmp_path / "file.txt").write_text("content")
    (tmp_path / "ignored.log").write_text("log")

    result = capture_spec(tmp_path, ignore=["*.log"])

    content = result[1].read_text()
    assert "file.txt" in content
    assert "ignored.log" not in content


# -----------------------------------------------------------------------------
# Tests for list_spec_versions
# -----------------------------------------------------------------------------

def test_list_spec_versions_empty(tmp_path):
    """Should return empty list when no specs."""
    result = list_spec_versions(tmp_path)
    assert result == []


def test_list_spec_versions_returns_all(tmp_path):
    """Should return all versions with timestamps."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("# Captured: 2024-01-01T10:00:00\nfile1.txt")
    (specs_dir / "v2.tree").write_text("# Captured: 2024-01-02T10:00:00\nfile2.txt")

    result = list_spec_versions(tmp_path)

    assert len(result) == 2
    assert result[0][0] == 1  # version number
    assert result[1][0] == 2


def test_list_spec_versions_extracts_timestamp(tmp_path):
    """Should extract timestamp from header."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    timestamp = "2024-03-15T14:30:00"
    (specs_dir / "v1.tree").write_text(f"# Captured: {timestamp}\nfile.txt")

    result = list_spec_versions(tmp_path)

    assert result[0][2] == timestamp


def test_list_spec_versions_handles_missing_timestamp(tmp_path):
    """Should handle specs without timestamp header."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("file.txt")

    result = list_spec_versions(tmp_path)

    assert result[0][2] == "unknown"


# -----------------------------------------------------------------------------
# Tests for get_spec_version
# -----------------------------------------------------------------------------

def test_get_spec_version_returns_content(tmp_path):
    """Should return content of specific version."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    content = "# Header\nfile.txt"
    (specs_dir / "v1.tree").write_text(content)

    result = get_spec_version(tmp_path, 1)
    assert result == content


def test_get_spec_version_returns_none_for_missing(tmp_path):
    """Should return None for non-existent version."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    result = get_spec_version(tmp_path, 99)
    assert result is None


def test_get_spec_version_specific_version(tmp_path):
    """Should get specific version content."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("version1")
    (specs_dir / "v2.tree").write_text("version2")
    (specs_dir / "v3.tree").write_text("version3")

    assert get_spec_version(tmp_path, 2) == "version2"


# -----------------------------------------------------------------------------
# Tests for get_current_spec
# -----------------------------------------------------------------------------

def test_get_current_spec_returns_latest(tmp_path):
    """Should return latest version."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("version1")
    (specs_dir / "v2.tree").write_text("version2")

    result = get_current_spec(tmp_path)

    assert result is not None
    assert result[0] == 2  # latest version number
    assert result[1] == "version2"


def test_get_current_spec_returns_none_when_empty(tmp_path):
    """Should return None when no versions exist."""
    result = get_current_spec(tmp_path)
    assert result is None


def test_get_current_spec_handles_gaps(tmp_path):
    """Should return highest version even with gaps."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    # v1 and v5 exist, v2-v4 missing
    (specs_dir / "v1.tree").write_text("version1")
    (specs_dir / "v5.tree").write_text("version5")

    result = get_current_spec(tmp_path)
    assert result[0] == 5


# -----------------------------------------------------------------------------
# Tests for diff_spec_versions
# -----------------------------------------------------------------------------

def test_diff_spec_versions_shows_added(tmp_path):
    """Should show added paths."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("file1.txt")
    (specs_dir / "v2.tree").write_text("file1.txt\nfile2.txt")

    result = diff_spec_versions(tmp_path, 1, 2)

    assert "file2.txt" in result["added"]


def test_diff_spec_versions_shows_removed(tmp_path):
    """Should show removed paths."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("file1.txt\nfile2.txt")
    (specs_dir / "v2.tree").write_text("file1.txt")

    result = diff_spec_versions(tmp_path, 1, 2)

    assert "file2.txt" in result["removed"]


def test_diff_spec_versions_shows_unchanged(tmp_path):
    """Should show unchanged paths."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("common.txt\nold.txt")
    (specs_dir / "v2.tree").write_text("common.txt\nnew.txt")

    result = diff_spec_versions(tmp_path, 1, 2)

    assert "common.txt" in result["unchanged"]


def test_diff_spec_versions_ignores_comments(tmp_path):
    """Should ignore comment lines in diff."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("# Comment\nfile.txt")
    (specs_dir / "v2.tree").write_text("# Different comment\nfile.txt")

    result = diff_spec_versions(tmp_path, 1, 2)

    # Comment differences should not appear
    assert len(result["added"]) == 0
    assert len(result["removed"]) == 0


def test_diff_spec_versions_raises_for_missing(tmp_path):
    """Should raise for non-existent version."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("file.txt")

    with pytest.raises(ValueError, match="Version not found"):
        diff_spec_versions(tmp_path, 1, 99)


def test_diff_spec_versions_sorted_results(tmp_path):
    """Should return sorted lists."""
    specs_dir = tmp_path / SPECS_DIR
    specs_dir.mkdir(parents=True)

    (specs_dir / "v1.tree").write_text("a.txt\nc.txt\nb.txt")
    (specs_dir / "v2.tree").write_text("b.txt")

    result = diff_spec_versions(tmp_path, 1, 2)

    assert result["removed"] == ["a.txt", "c.txt"]
    assert result["unchanged"] == ["b.txt"]


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

def test_full_spec_history_workflow(tmp_path):
    """Test complete capture -> list -> get -> diff workflow."""
    # Initial state
    (tmp_path / "file1.txt").write_text("f1")

    # First capture
    result1 = capture_spec(tmp_path)
    assert result1[0] == 1

    # Add file and capture
    (tmp_path / "file2.txt").write_text("f2")
    result2 = capture_spec(tmp_path)
    assert result2[0] == 2

    # List versions
    versions = list_spec_versions(tmp_path)
    assert len(versions) == 2

    # Get specific version
    v1_content = get_spec_version(tmp_path, 1)
    assert "file1.txt" in v1_content
    assert "file2.txt" not in v1_content

    # Get current
    current = get_current_spec(tmp_path)
    assert current[0] == 2
    assert "file2.txt" in current[1]

    # Diff
    diff = diff_spec_versions(tmp_path, 1, 2)
    assert "file2.txt" in diff["added"]
    assert "file1.txt" in diff["unchanged"]
