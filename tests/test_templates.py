import pytest
from pathlib import Path
from seed_cli.templates import (
    validate_template_dir,
    render_template_dir,
    TemplateError,
)


def test_validate_template_dir_missing(tmp_path):
    with pytest.raises(TemplateError):
        validate_template_dir(tmp_path / "missing")


def test_render_template_dir_basic(tmp_path):
    tmpl = tmp_path / "tmpl"
    tmpl.mkdir()
    (tmpl / "{{name}}.txt").write_text("hello {{name}}")  # type: ignore

    out = tmp_path / "out"
    render_template_dir(tmpl, out, {"name": "world"})

    f = out / "world.txt"
    assert f.exists()
    assert f.read_text() == "hello world"


def test_render_template_dir_no_overwrite(tmp_path):
    tmpl = tmp_path / "tmpl"
    tmpl.mkdir()
    (tmpl / "a.txt").write_text("one")  # type: ignore

    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("two")

    render_template_dir(tmpl, out, {}, overwrite=False)
    assert (out / "a.txt").read_text() == "two"


def test_render_template_dir_overwrite(tmp_path):
    tmpl = tmp_path / "tmpl"
    tmpl.mkdir()
    (tmpl / "a.txt").write_text("one")  # type: ignore

    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("two")

    render_template_dir(tmpl, out, {}, overwrite=True)
    assert (out / "a.txt").read_text() == "one"


# Additional tests for improved coverage

from seed_cli.templates import iter_template_files, install_git_hook


def test_validate_template_dir_not_a_directory(tmp_path):
    """Test validation fails when path is a file, not directory."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")

    with pytest.raises(TemplateError, match="not a directory"):
        validate_template_dir(file_path)


def test_validate_template_dir_success(tmp_path):
    """Test validation succeeds for valid directory."""
    tmpl = tmp_path / "templates"
    tmpl.mkdir()

    # Should not raise
    validate_template_dir(tmpl)


def test_iter_template_files_empty_dir(tmp_path):
    """Test iter_template_files on empty directory."""
    tmpl = tmp_path / "empty"
    tmpl.mkdir()

    files = list(iter_template_files(tmpl))
    assert files == []


def test_iter_template_files_with_content(tmp_path):
    """Test iter_template_files returns all files and dirs."""
    tmpl = tmp_path / "tmpl"
    tmpl.mkdir()
    (tmpl / "file1.txt").write_text("content1")
    (tmpl / "subdir").mkdir()
    (tmpl / "subdir" / "file2.txt").write_text("content2")

    files = list(iter_template_files(tmpl))
    names = [f.name for f in files]

    assert "file1.txt" in names
    assert "subdir" in names
    assert "file2.txt" in names


def test_iter_template_files_nested(tmp_path):
    """Test iter_template_files with deeply nested structure."""
    tmpl = tmp_path / "tmpl"
    nested = tmpl / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "deep.txt").write_text("deep")

    files = list(iter_template_files(tmpl))

    assert len(files) >= 4  # a, b, c, deep.txt


def test_render_template_dir_nested_vars(tmp_path):
    """Test rendering with variables in nested paths."""
    tmpl = tmp_path / "tmpl"
    (tmpl / "{{project}}").mkdir(parents=True)
    (tmpl / "{{project}}" / "{{file}}.py").write_text("# {{project}} - {{file}}")

    out = tmp_path / "out"
    render_template_dir(tmpl, out, {"project": "myproj", "file": "main"})

    rendered = out / "myproj" / "main.py"
    assert rendered.exists()
    assert rendered.read_text() == "# myproj - main"


def test_render_template_dir_creates_parent_dirs(tmp_path):
    """Test that render creates parent directories automatically."""
    tmpl = tmp_path / "tmpl"
    tmpl.mkdir()
    (tmpl / "nested" / "path").mkdir(parents=True)
    (tmpl / "nested" / "path" / "file.txt").write_text("content")

    out = tmp_path / "out"
    # out doesn't exist yet
    render_template_dir(tmpl, out, {})

    assert (out / "nested" / "path" / "file.txt").exists()


def test_render_template_dir_empty_vars(tmp_path):
    """Test rendering with empty vars dict."""
    tmpl = tmp_path / "tmpl"
    tmpl.mkdir()
    (tmpl / "static.txt").write_text("no variables here")

    out = tmp_path / "out"
    render_template_dir(tmpl, out, {})

    assert (out / "static.txt").read_text() == "no variables here"


def test_install_git_hook_success(tmp_path):
    """Test installing a git hook in a valid git repo."""
    # Create .git/hooks structure
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)

    install_git_hook(tmp_path, "pre-commit")

    hook_path = hooks_dir / "pre-commit"
    assert hook_path.exists()
    content = hook_path.read_text()
    assert "seed plan" in content
    # Check it's executable (on Unix)
    import os
    assert os.access(hook_path, os.X_OK)


def test_install_git_hook_not_git_repo(tmp_path):
    """Test install_git_hook fails when not a git repo."""
    # No .git directory
    with pytest.raises(RuntimeError, match="Not a git repository"):
        install_git_hook(tmp_path, "pre-commit")


def test_install_git_hook_custom_name(tmp_path):
    """Test installing a hook with custom name."""
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)

    install_git_hook(tmp_path, "pre-push")

    assert (hooks_dir / "pre-push").exists()
