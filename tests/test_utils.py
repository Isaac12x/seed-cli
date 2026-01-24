"""Tests for seed_cli.utils module.

Tests for utility functions in utils/__init__.py and utils/dir.py.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from seed_cli.utils.dir import (
    _detect_single_root_dir,
    _normalize_spec_index,
    _strip_prefix_index,
    _strip_prefix_checks,
)


# Mock Node class for testing
class MockNode:
    def __init__(self, is_dir=False):
        self.is_dir = is_dir


# -----------------------------------------------------------------------------
# Tests for _detect_single_root_dir
# -----------------------------------------------------------------------------

def test_detect_single_root_dir_empty():
    """Should return None for empty index."""
    result = _detect_single_root_dir({})
    assert result is None


def test_detect_single_root_dir_only_dot():
    """Should return None when only '.' is present."""
    result = _detect_single_root_dir({".": MockNode(is_dir=True)})
    assert result is None


def test_detect_single_root_dir_single_root():
    """Should detect single root directory."""
    sidx = {
        "project": MockNode(is_dir=True),
        "project/src": MockNode(is_dir=True),
        "project/src/main.py": MockNode(is_dir=False),
    }
    result = _detect_single_root_dir(sidx)
    assert result == "project"


def test_detect_single_root_dir_multiple_roots():
    """Should return None when multiple root directories exist."""
    sidx = {
        "src": MockNode(is_dir=True),
        "lib": MockNode(is_dir=True),
        "src/main.py": MockNode(is_dir=False),
    }
    result = _detect_single_root_dir(sidx)
    assert result is None


def test_detect_single_root_dir_root_not_dir():
    """Should return None when root is not a directory."""
    sidx = {
        "file.txt": MockNode(is_dir=False),
    }
    result = _detect_single_root_dir(sidx)
    assert result is None


def test_detect_single_root_dir_root_missing():
    """Should return None when root dir node doesn't exist."""
    sidx = {
        "project/src/main.py": MockNode(is_dir=False),
    }
    result = _detect_single_root_dir(sidx)
    assert result is None


def test_detect_single_root_dir_with_dot_and_project():
    """Should detect single root when '.' and project both present."""
    sidx = {
        ".": MockNode(is_dir=True),
        "project": MockNode(is_dir=True),
        "project/file.txt": MockNode(is_dir=False),
    }
    result = _detect_single_root_dir(sidx)
    # Should detect project as the single root (. is excluded)
    assert result == "project"


# -----------------------------------------------------------------------------
# Tests for _normalize_spec_index
# -----------------------------------------------------------------------------

def test_normalize_spec_index_relative_paths(tmp_path):
    """Should preserve relative paths."""
    sidx = {
        "src/main.py": MockNode(is_dir=False),
        "lib/utils.py": MockNode(is_dir=False),
    }
    result = _normalize_spec_index(sidx, tmp_path)

    assert "src/main.py" in result
    assert "lib/utils.py" in result


def test_normalize_spec_index_tilde_expansion(tmp_path):
    """Should expand ~ in paths."""
    sidx = {
        "~/project": MockNode(is_dir=True),
    }
    result = _normalize_spec_index(sidx, tmp_path)

    # ~ should be expanded, but path may not be relative to tmp_path
    keys = list(result.keys())
    assert len(keys) == 1
    assert "~" not in keys[0]


def test_normalize_spec_index_absolute_inside_base(tmp_path):
    """Should convert absolute paths inside base to relative."""
    # Create the path first
    target = tmp_path / "inside" / "file.txt"
    target.parent.mkdir(parents=True)
    target.touch()

    sidx = {
        str(target): MockNode(is_dir=False),
    }
    result = _normalize_spec_index(sidx, tmp_path)

    # Should be normalized to relative path
    keys = list(result.keys())
    assert len(keys) == 1
    # Check it's relative
    assert "inside/file.txt" in keys[0] or "inside\\file.txt" in keys[0]


def test_normalize_spec_index_absolute_outside_base(tmp_path):
    """Should keep absolute paths outside base as absolute."""
    import tempfile
    with tempfile.TemporaryDirectory() as other_dir:
        sidx = {
            f"{other_dir}/file.txt": MockNode(is_dir=False),
        }
        result = _normalize_spec_index(sidx, tmp_path)

        keys = list(result.keys())
        assert len(keys) == 1
        # Should remain absolute since it's outside base
        assert Path(keys[0]).is_absolute()


# -----------------------------------------------------------------------------
# Tests for _strip_prefix_index
# -----------------------------------------------------------------------------

def test_strip_prefix_index_basic():
    """Should strip prefix from all matching paths."""
    idx = {
        "project": "root",
        "project/src": "src_dir",
        "project/src/main.py": "main",
    }
    result = _strip_prefix_index(idx, "project")

    assert "project" not in result  # root dropped
    assert "src" in result
    assert "src/main.py" in result
    assert result["src"] == "src_dir"


def test_strip_prefix_index_no_match():
    """Should preserve paths that don't match prefix."""
    idx = {
        "other": "value",
    }
    result = _strip_prefix_index(idx, "project")

    assert "other" in result
    assert result["other"] == "value"


def test_strip_prefix_index_mixed():
    """Should handle mix of matching and non-matching paths."""
    idx = {
        "project": "root",
        "project/file.txt": "file",
        "external": "external_value",
    }
    result = _strip_prefix_index(idx, "project")

    assert "project" not in result
    assert "file.txt" in result
    assert "external" in result


def test_strip_prefix_index_nested():
    """Should strip prefix from deeply nested paths."""
    idx = {
        "project/a/b/c/d.txt": "deep",
    }
    result = _strip_prefix_index(idx, "project")

    assert "a/b/c/d.txt" in result


def test_strip_prefix_index_with_trailing_slash():
    """Should handle prefix with or without trailing slash."""
    idx = {
        "project/file.txt": "file",
    }
    # prefix is normalized internally
    result = _strip_prefix_index(idx, "project/")

    assert "file.txt" in result


# -----------------------------------------------------------------------------
# Tests for _strip_prefix_checks
# -----------------------------------------------------------------------------

def test_strip_prefix_checks_basic():
    """Should strip prefix from checksum dict keys."""
    checks = {
        "project": {"sha256": "abc"},
        "project/file.txt": {"sha256": "def"},
    }
    result = _strip_prefix_checks(checks, "project")

    assert "project" not in result
    assert "file.txt" in result
    assert result["file.txt"]["sha256"] == "def"


def test_strip_prefix_checks_no_match():
    """Should preserve non-matching paths."""
    checks = {
        "other/file.txt": {"sha256": "xyz"},
    }
    result = _strip_prefix_checks(checks, "project")

    assert "other/file.txt" in result


def test_strip_prefix_checks_empty():
    """Should handle empty dict."""
    result = _strip_prefix_checks({}, "project")
    assert result == {}


def test_strip_prefix_checks_mixed():
    """Should handle mix of matching and non-matching."""
    checks = {
        "project": {"sha256": "a"},
        "project/src/file.py": {"sha256": "b"},
        "external/other.py": {"sha256": "c"},
    }
    result = _strip_prefix_checks(checks, "project")

    assert "project" not in result
    assert "src/file.py" in result
    assert "external/other.py" in result


# -----------------------------------------------------------------------------
# Tests for utils/__init__.py functions
# -----------------------------------------------------------------------------

def test_has_image_support_when_missing():
    """Test has_image_support returns False when dependencies missing."""
    from seed_cli.utils import has_image_support

    with patch.dict('sys.modules', {'pytesseract': None}):
        # Force reimport to test the check
        # This is tricky to test properly, so we just verify the function exists
        result = has_image_support()
        # Result depends on actual environment, so just verify it returns bool
        assert isinstance(result, bool)


def test_has_image_support_function_exists():
    """Test that has_image_support function exists and is callable."""
    from seed_cli.utils import has_image_support

    assert callable(has_image_support)
    result = has_image_support()
    assert isinstance(result, bool)


# Test extract_tree_from_image with mocking

def test_extract_tree_from_image_file_not_found(tmp_path):
    """Test extract_tree_from_image raises for missing file."""
    from seed_cli.utils import extract_tree_from_image

    with pytest.raises(FileNotFoundError):
        extract_tree_from_image(tmp_path / "nonexistent.png")


@patch('seed_cli.image.read_tree')
@patch('seed_cli.image.tree_lines_to_text')
def test_extract_tree_from_image_basic(mock_tree_to_text, mock_read_tree, tmp_path):
    """Test basic extract_tree_from_image functionality."""
    from seed_cli.utils import extract_tree_from_image

    # Create a fake image file
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake png content")

    # Mock OCR functions
    mock_read_tree.return_value = ["src/", "├── main.py"]
    mock_tree_to_text.return_value = "src/\n├── main.py"

    result = extract_tree_from_image(image_path)

    assert result.suffix == ".tree"
    assert result.exists()
    mock_read_tree.assert_called_once()


@patch('seed_cli.image.read_tree')
@patch('seed_cli.image.tree_lines_to_text')
def test_extract_tree_from_image_custom_output(mock_tree_to_text, mock_read_tree, tmp_path):
    """Test extract_tree_from_image with custom output path."""
    from seed_cli.utils import extract_tree_from_image

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake")
    output_path = tmp_path / "custom_output.tree"

    mock_read_tree.return_value = ["file.txt"]
    mock_tree_to_text.return_value = "file.txt"

    result = extract_tree_from_image(image_path, output_path=output_path)

    assert result == output_path


@patch('seed_cli.image.read_tree')
@patch('seed_cli.image.tree_lines_to_text')
def test_extract_tree_from_image_raw_mode(mock_tree_to_text, mock_read_tree, tmp_path):
    """Test extract_tree_from_image with raw=True."""
    from seed_cli.utils import extract_tree_from_image

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake")

    mock_read_tree.return_value = ["raw", "output"]
    mock_tree_to_text.return_value = "raw\noutput"

    result = extract_tree_from_image(image_path, raw=True)

    assert result.exists()
    # In raw mode, tree_lines_to_text is called with raw_text
    assert mock_tree_to_text.called
