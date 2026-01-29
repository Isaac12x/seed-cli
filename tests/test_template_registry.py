"""Tests for seed_cli.template_registry module.

Comprehensive tests for template management functionality.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from seed_cli.template_registry import (
    TEMPLATES_DIR_NAME,
    REGISTRY_FILE,
    META_FILE,
    TemplateMetadata,
    get_templates_dir,
    load_registry,
    save_registry,
    add_template,
    add_local_template,
    remove_template,
    list_templates,
    get_template,
    get_template_spec_path,
    list_versions,
    add_version,
    set_current_version,
    lock_template,
    unlock_template,
    parse_github_url,
    _get_template_dir,
    _load_meta,
    _save_meta,
    _get_next_version,
)


# -----------------------------------------------------------------------------
# Tests for TemplateMetadata
# -----------------------------------------------------------------------------

def test_template_metadata_defaults():
    """Should create metadata with default values."""
    meta = TemplateMetadata(
        name="test",
        source="https://github.com/user/repo",
        current_version="v1",
    )
    assert meta.name == "test"
    assert meta.source == "https://github.com/user/repo"
    assert meta.current_version == "v1"
    assert meta.locked is False
    assert isinstance(meta.created_at, float)
    assert meta.versions == []


def test_template_metadata_to_dict():
    """Should convert metadata to dict."""
    meta = TemplateMetadata(
        name="test",
        source="https://github.com/user/repo",
        current_version="v1",
        locked=True,
        versions=["v1", "v2"],
    )
    data = meta.to_dict()
    assert data["name"] == "test"
    assert data["locked"] is True
    assert data["versions"] == ["v1", "v2"]


def test_template_metadata_from_dict():
    """Should create metadata from dict."""
    data = {
        "name": "test",
        "source": "https://github.com/user/repo",
        "current_version": "v1",
        "locked": False,
        "created_at": 12345.0,
        "versions": ["v1"],
    }
    meta = TemplateMetadata.from_dict(data)
    assert meta.name == "test"
    assert meta.created_at == 12345.0


# -----------------------------------------------------------------------------
# Tests for get_templates_dir
# -----------------------------------------------------------------------------

def test_get_templates_dir_creates_directory(tmp_path, monkeypatch):
    """Should create templates directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = get_templates_dir()

    assert result.exists()
    assert result.is_dir()
    assert result == tmp_path / ".seed" / TEMPLATES_DIR_NAME


def test_get_templates_dir_returns_existing(tmp_path, monkeypatch):
    """Should return existing directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    templates_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME
    templates_dir.mkdir(parents=True)

    result = get_templates_dir()
    assert result == templates_dir


# -----------------------------------------------------------------------------
# Tests for load_registry / save_registry
# -----------------------------------------------------------------------------

def test_load_registry_empty_when_no_file(tmp_path, monkeypatch):
    """Should return empty dict when registry doesn't exist."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = load_registry()
    assert result == {}


def test_save_and_load_registry(tmp_path, monkeypatch):
    """Should save and load registry correctly."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    meta = TemplateMetadata(
        name="test",
        source="https://github.com/user/repo",
        current_version="v1",
        versions=["v1"],
    )
    registry = {"test": meta}

    save_registry(registry)
    loaded = load_registry()

    assert "test" in loaded
    assert loaded["test"].name == "test"
    assert loaded["test"].versions == ["v1"]


def test_load_registry_handles_invalid_json(tmp_path, monkeypatch):
    """Should handle corrupted registry file."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create invalid JSON
    templates_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME
    templates_dir.mkdir(parents=True)
    (templates_dir / REGISTRY_FILE).write_text("not valid json")

    result = load_registry()
    assert result == {}


# -----------------------------------------------------------------------------
# Tests for parse_github_url
# -----------------------------------------------------------------------------

def test_parse_github_url_blob():
    """Should parse GitHub blob URL."""
    url = "https://github.com/user/repo/blob/main/path/to/spec.tree"
    result = parse_github_url(url)

    assert result is not None
    assert result["owner"] == "user"
    assert result["repo"] == "repo"
    assert result["ref"] == "main"
    assert result["path"] == "path/to/spec.tree"
    assert result["type"] == "blob"


def test_parse_github_url_tree():
    """Should parse GitHub tree URL."""
    url = "https://github.com/user/repo/tree/v1.0/specs"
    result = parse_github_url(url)

    assert result is not None
    assert result["type"] == "tree"
    assert result["path"] == "specs"


def test_parse_github_url_without_protocol():
    """Should handle URL without https://."""
    url = "github.com/user/repo/blob/main/spec.tree"
    result = parse_github_url(url)

    assert result is not None
    assert result["owner"] == "user"


def test_parse_github_url_invalid():
    """Should return None for invalid URLs."""
    assert parse_github_url("https://gitlab.com/user/repo") is None
    assert parse_github_url("not a url") is None
    assert parse_github_url("https://github.com/user") is None


# -----------------------------------------------------------------------------
# Tests for _get_next_version
# -----------------------------------------------------------------------------

def test_get_next_version_first(tmp_path, monkeypatch):
    """Should return v1 for empty directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    template_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "test"
    template_dir.mkdir(parents=True)

    result = _get_next_version(template_dir)
    assert result == "v1"


def test_get_next_version_increments(tmp_path, monkeypatch):
    """Should increment from highest version."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    template_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "test"
    template_dir.mkdir(parents=True)

    (template_dir / "v1.tree").write_text("content")
    (template_dir / "v2.tree").write_text("content")

    result = _get_next_version(template_dir)
    assert result == "v3"


def test_get_next_version_handles_gaps(tmp_path, monkeypatch):
    """Should use highest version even with gaps."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    template_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "test"
    template_dir.mkdir(parents=True)

    (template_dir / "v1.tree").write_text("content")
    (template_dir / "v5.tree").write_text("content")

    result = _get_next_version(template_dir)
    assert result == "v6"


# -----------------------------------------------------------------------------
# Tests for add_local_template
# -----------------------------------------------------------------------------

def test_add_local_template(tmp_path, monkeypatch):
    """Should add a template from local file."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create a local spec file
    spec_file = tmp_path / "myspec.tree"
    spec_file.write_text("src/\nfile.py")

    meta = add_local_template(str(spec_file), "mytemplate")

    assert meta.name == "mytemplate"
    assert meta.current_version == "v1"
    assert "v1" in meta.versions
    assert meta.source == f"local:{spec_file}"

    # Check file was copied
    template_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "mytemplate"
    assert (template_dir / "v1.tree").exists()


def test_add_local_template_auto_version(tmp_path, monkeypatch):
    """Should auto-increment version."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    meta1 = add_local_template(str(spec_file), "test")
    meta2 = add_local_template(str(spec_file), "test")

    assert meta1.current_version == "v1"
    assert meta2.current_version == "v2"
    assert "v1" in meta2.versions
    assert "v2" in meta2.versions


def test_add_local_template_explicit_version(tmp_path, monkeypatch):
    """Should use explicit version name."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    meta = add_local_template(str(spec_file), "test", version="v5")

    assert meta.current_version == "v5"


def test_add_local_template_file_not_found(tmp_path, monkeypatch):
    """Should raise FileNotFoundError for missing file."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        add_local_template("/nonexistent/spec.tree", "test")


def test_add_local_template_locked_error(tmp_path, monkeypatch):
    """Should raise ValueError for locked template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    # Add and lock template
    add_local_template(str(spec_file), "test")
    lock_template("test")

    with pytest.raises(ValueError, match="locked"):
        add_local_template(str(spec_file), "test")


# -----------------------------------------------------------------------------
# Tests for remove_template
# -----------------------------------------------------------------------------

def test_remove_template(tmp_path, monkeypatch):
    """Should remove a template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")

    result = remove_template("test")

    assert result is True
    assert get_template("test") is None

    # Check directory was removed
    template_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "test"
    assert not template_dir.exists()


def test_remove_template_not_found(tmp_path, monkeypatch):
    """Should return False for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = remove_template("nonexistent")
    assert result is False


def test_remove_template_locked_error(tmp_path, monkeypatch):
    """Should raise ValueError for locked template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    lock_template("test")

    with pytest.raises(ValueError, match="locked"):
        remove_template("test")


# -----------------------------------------------------------------------------
# Tests for list_templates
# -----------------------------------------------------------------------------

def test_list_templates_empty(tmp_path, monkeypatch):
    """Should return empty list when no templates."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = list_templates()
    assert result == []


def test_list_templates(tmp_path, monkeypatch):
    """Should list all templates."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test1")
    add_local_template(str(spec_file), "test2")

    result = list_templates()

    names = [t.name for t in result]
    assert "test1" in names
    assert "test2" in names


# -----------------------------------------------------------------------------
# Tests for get_template
# -----------------------------------------------------------------------------

def test_get_template(tmp_path, monkeypatch):
    """Should get template metadata."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")

    result = get_template("test")

    assert result is not None
    assert result.name == "test"


def test_get_template_not_found(tmp_path, monkeypatch):
    """Should return None for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = get_template("nonexistent")
    assert result is None


# -----------------------------------------------------------------------------
# Tests for get_template_spec_path
# -----------------------------------------------------------------------------

def test_get_template_spec_path(tmp_path, monkeypatch):
    """Should return path to spec file."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")

    result = get_template_spec_path("test")

    assert result is not None
    assert result.exists()
    assert result.name == "v1.tree"


def test_get_template_spec_path_with_version(tmp_path, monkeypatch):
    """Should return path for specific version."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    add_local_template(str(spec_file), "test")

    result = get_template_spec_path("test", "v1")

    assert result is not None
    assert result.name == "v1.tree"


def test_get_template_spec_path_not_found(tmp_path, monkeypatch):
    """Should return None for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = get_template_spec_path("nonexistent")
    assert result is None


# -----------------------------------------------------------------------------
# Tests for list_versions
# -----------------------------------------------------------------------------

def test_list_versions(tmp_path, monkeypatch):
    """Should list all versions of a template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    add_local_template(str(spec_file), "test")
    add_local_template(str(spec_file), "test")

    result = list_versions("test")

    versions = [v for v, _ in result]
    assert versions == ["v1", "v2", "v3"]


def test_list_versions_not_found(tmp_path, monkeypatch):
    """Should return empty list for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = list_versions("nonexistent")
    assert result == []


# -----------------------------------------------------------------------------
# Tests for add_version
# -----------------------------------------------------------------------------

def test_add_version(tmp_path, monkeypatch):
    """Should add a new version to existing template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    spec_file2 = tmp_path / "spec2.tree"
    spec_file2.write_text("updated content")

    add_local_template(str(spec_file), "test")
    version = add_version("test", str(spec_file2))

    assert version == "v2"

    meta = get_template("test")
    assert "v2" in meta.versions


def test_add_version_explicit_name(tmp_path, monkeypatch):
    """Should use explicit version name."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    version = add_version("test", str(spec_file), "v10")

    assert version == "v10"


def test_add_version_template_not_found(tmp_path, monkeypatch):
    """Should raise ValueError for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    with pytest.raises(ValueError, match="not found"):
        add_version("nonexistent", str(spec_file))


def test_add_version_locked_error(tmp_path, monkeypatch):
    """Should raise ValueError for locked template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    lock_template("test")

    with pytest.raises(ValueError, match="locked"):
        add_version("test", str(spec_file))


# -----------------------------------------------------------------------------
# Tests for set_current_version
# -----------------------------------------------------------------------------

def test_set_current_version(tmp_path, monkeypatch):
    """Should set current version."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    add_local_template(str(spec_file), "test")

    set_current_version("test", "v1")

    meta = get_template("test")
    assert meta.current_version == "v1"


def test_set_current_version_not_found(tmp_path, monkeypatch):
    """Should raise ValueError for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(ValueError, match="Template not found"):
        set_current_version("nonexistent", "v1")


def test_set_current_version_invalid(tmp_path, monkeypatch):
    """Should raise ValueError for non-existent version."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")

    with pytest.raises(ValueError, match="Version not found"):
        set_current_version("test", "v99")


# -----------------------------------------------------------------------------
# Tests for lock_template / unlock_template
# -----------------------------------------------------------------------------

def test_lock_template(tmp_path, monkeypatch):
    """Should lock a template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    lock_template("test")

    meta = get_template("test")
    assert meta.locked is True


def test_lock_template_with_version(tmp_path, monkeypatch):
    """Should set version when locking."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    add_local_template(str(spec_file), "test")

    lock_template("test", version="v1")

    meta = get_template("test")
    assert meta.locked is True
    assert meta.current_version == "v1"


def test_unlock_template(tmp_path, monkeypatch):
    """Should unlock a template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    add_local_template(str(spec_file), "test")
    lock_template("test")
    unlock_template("test")

    meta = get_template("test")
    assert meta.locked is False


def test_lock_template_not_found(tmp_path, monkeypatch):
    """Should raise ValueError for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(ValueError, match="not found"):
        lock_template("nonexistent")


def test_unlock_template_not_found(tmp_path, monkeypatch):
    """Should raise ValueError for non-existent template."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(ValueError, match="not found"):
        unlock_template("nonexistent")


# -----------------------------------------------------------------------------
# Tests for _load_meta / _save_meta
# -----------------------------------------------------------------------------

def test_save_and_load_meta(tmp_path, monkeypatch):
    """Should save and load meta.json correctly."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    meta = TemplateMetadata(
        name="test",
        source="local:test.tree",
        current_version="v1",
        locked=True,
        versions=["v1", "v2"],
    )

    _save_meta(meta)
    loaded = _load_meta("test")

    assert loaded is not None
    assert loaded.name == "test"
    assert loaded.locked is True
    assert loaded.versions == ["v1", "v2"]


def test_load_meta_not_found(tmp_path, monkeypatch):
    """Should return None for non-existent meta.json."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = _load_meta("nonexistent")
    assert result is None


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

def test_full_workflow(tmp_path, monkeypatch):
    """Test complete template lifecycle."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create spec files
    spec1 = tmp_path / "spec1.tree"
    spec1.write_text("src/")

    spec2 = tmp_path / "spec2.tree"
    spec2.write_text("src/\nlib/")

    # Add template
    meta = add_local_template(str(spec1), "myproject")
    assert meta.name == "myproject"
    assert meta.current_version == "v1"

    # List templates
    templates = list_templates()
    assert len(templates) == 1

    # Get spec path
    path = get_template_spec_path("myproject")
    assert path.read_text() == "src/"

    # Add version
    add_version("myproject", str(spec2))
    meta = get_template("myproject")
    assert "v2" in meta.versions

    # List versions
    versions = list_versions("myproject")
    assert len(versions) == 2

    # Set current
    set_current_version("myproject", "v1")
    meta = get_template("myproject")
    assert meta.current_version == "v1"

    # Lock
    lock_template("myproject", version="v2")
    meta = get_template("myproject")
    assert meta.locked is True
    assert meta.current_version == "v2"

    # Unlock
    unlock_template("myproject")
    meta = get_template("myproject")
    assert meta.locked is False

    # Remove
    remove_template("myproject")
    assert get_template("myproject") is None


def test_sanitize_name(tmp_path, monkeypatch):
    """Should sanitize template names."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    spec_file.write_text("content")

    meta = add_local_template(str(spec_file), "my-template!@#$%")

    # Name should be sanitized
    assert "!" not in meta.name
    assert "@" not in meta.name


# -----------------------------------------------------------------------------
# Tests for GitHub fetching (mocked)
# -----------------------------------------------------------------------------

def test_add_template_from_github_mocked(tmp_path, monkeypatch):
    """Should add template from GitHub URL (mocked)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Mock the fetch function
    def mock_fetch(url, dest_dir, name=None):
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / (name or "spec.tree")
        dest_file.write_text("github content")
        return dest_file, "spec.tree"

    monkeypatch.setattr(
        "seed_cli.template_registry.fetch_from_github",
        mock_fetch
    )

    meta = add_template(
        "https://github.com/user/repo/blob/main/spec.tree",
        name="github-template"
    )

    assert meta.name == "github-template"
    assert "github.com" in meta.source
