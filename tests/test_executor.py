from pathlib import Path
from seed_cli.executor import execute_plan, _backup_file_if_has_content, _get_backup_dir
from seed_cli.planning import PlanResult, PlanStep
from seed_cli.checksums import load_checksums


def test_executor_create_and_mkdir(tmp_path):
    plan = PlanResult(
        steps=[
            PlanStep("mkdir", "a", "missing"),
            PlanStep("create", "a/file.txt", "missing"),
        ],
        add=2, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path)
    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "a/file.txt").exists()
    assert res["created"] == 2

    checks = load_checksums(tmp_path)
    assert "a/file.txt" in checks


def test_executor_update(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("old")

    plan = PlanResult(
        steps=[PlanStep("update", "x.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=True)
    assert res["updated"] == 1
    assert load_checksums(tmp_path)["x.txt"]["sha256"]


def test_executor_skip(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("skip", "x.txt", "manual")],
        add=0, change=0, delete=0, delete_skipped=1,
    )
    res = execute_plan(plan, tmp_path)
    assert res["skipped"] == 1


def test_executor_delete_requires_dangerous(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")

    plan = PlanResult(
        steps=[PlanStep("delete", "x.txt", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )

    try:
        execute_plan(plan, tmp_path)
        assert False, "delete should have failed"
    except RuntimeError:
        pass


def test_executor_delete(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")

    plan = PlanResult(
        steps=[PlanStep("delete", "x.txt", "extra")],
        add=0, change=0, delete=1, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, dangerous=True)
    assert not f.exists()
    assert res["deleted"] == 1


def test_executor_gitkeep(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("mkdir", "d", "missing")],
        add=1, change=0, delete=0, delete_skipped=0,
    )

    execute_plan(plan, tmp_path, gitkeep=True)
    assert (tmp_path / "d/.gitkeep").exists()


def test_executor_dry_run(tmp_path):
    plan = PlanResult(
        steps=[PlanStep("create", "x.txt", "missing")],
        add=1, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, dry_run=True)
    assert not (tmp_path / "x.txt").exists()
    assert res["created"] == 1


def test_executor_backs_up_file_with_content_on_update(tmp_path):
    """Test that files with content are backed up before update operations."""
    # Create a file with content
    f = tmp_path / "important.txt"
    original_content = "This is important content that should be backed up"
    f.write_text(original_content)

    plan = PlanResult(
        steps=[PlanStep("update", "important.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=True)
    assert res["updated"] == 1
    assert res["backed_up"] == 1

    # Check that a backup was created in .seed/backups
    backups_dir = tmp_path / ".seed" / "backups"
    assert backups_dir.exists()

    # Find the backup file
    backup_dirs = list(backups_dir.iterdir())
    assert len(backup_dirs) == 1

    backup_file = backup_dirs[0] / "important.txt"
    assert backup_file.exists()
    assert backup_file.read_text() == original_content


def test_executor_backs_up_file_with_content_on_create(tmp_path):
    """Test that existing files with content are backed up during create operations."""
    # Create a file with content (simulating a race condition or unexpected state)
    f = tmp_path / "surprise.txt"
    original_content = "Unexpected existing content"
    f.write_text(original_content)

    # Plan says to create, but file already exists with content
    plan = PlanResult(
        steps=[PlanStep("create", "surprise.txt", "missing")],
        add=1, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path)
    assert res["created"] == 1
    assert res["backed_up"] == 1

    # Verify backup was created
    backups_dir = tmp_path / ".seed" / "backups"
    assert backups_dir.exists()

    backup_dirs = list(backups_dir.iterdir())
    assert len(backup_dirs) == 1

    backup_file = backup_dirs[0] / "surprise.txt"
    assert backup_file.exists()
    assert backup_file.read_text() == original_content


def test_executor_no_backup_for_empty_file(tmp_path):
    """Test that empty files are not backed up."""
    # Create an empty file
    f = tmp_path / "empty.txt"
    f.touch()

    plan = PlanResult(
        steps=[PlanStep("update", "empty.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=True)
    assert res["updated"] == 1
    assert res["backed_up"] == 0

    # No backup should be created
    backups_dir = tmp_path / ".seed" / "backups"
    assert not backups_dir.exists()


def test_executor_no_backup_on_dry_run(tmp_path):
    """Test that dry run doesn't create backups."""
    f = tmp_path / "x.txt"
    f.write_text("content")

    plan = PlanResult(
        steps=[PlanStep("update", "x.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, dry_run=True)
    assert res["updated"] == 1
    assert res["backed_up"] == 0

    # No backup should be created on dry run
    backups_dir = tmp_path / ".seed" / "backups"
    assert not backups_dir.exists()


def test_executor_multiple_files_same_backup_dir(tmp_path):
    """Test that multiple files in same execution share one backup directory."""
    # Create multiple files with content
    (tmp_path / "a.txt").write_text("content a")
    (tmp_path / "b.txt").write_text("content b")
    (tmp_path / "c.txt").write_text("content c")

    plan = PlanResult(
        steps=[
            PlanStep("update", "a.txt", "checksum_drift"),
            PlanStep("update", "b.txt", "checksum_drift"),
            PlanStep("update", "c.txt", "checksum_drift"),
        ],
        add=0, change=3, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=True)
    assert res["updated"] == 3
    assert res["backed_up"] == 3

    # All backups should be in a single directory
    backups_dir = tmp_path / ".seed" / "backups"
    backup_dirs = list(backups_dir.iterdir())
    assert len(backup_dirs) == 1

    # Verify all files are in that directory
    backup_dir = backup_dirs[0]
    assert (backup_dir / "a.txt").read_text() == "content a"
    assert (backup_dir / "b.txt").read_text() == "content b"
    assert (backup_dir / "c.txt").read_text() == "content c"


def test_executor_nested_paths_preserved_in_backup(tmp_path):
    """Test that nested file paths are preserved correctly in backups."""
    # Create nested directory structure with files
    nested_dir = tmp_path / "deep" / "nested" / "path"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "file.txt"
    nested_file.write_text("deeply nested content")

    plan = PlanResult(
        steps=[PlanStep("update", "deep/nested/path/file.txt", "checksum_drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=True)
    assert res["backed_up"] == 1

    # Verify backup preserves the nested path
    backups_dir = tmp_path / ".seed" / "backups"
    backup_dirs = list(backups_dir.iterdir())
    assert len(backup_dirs) == 1

    backup_file = backup_dirs[0] / "deep" / "nested" / "path" / "file.txt"
    assert backup_file.exists()
    assert backup_file.read_text() == "deeply nested content"


def test_executor_template_dir_backs_up_existing_files(tmp_path):
    """Test that template directory copying backs up existing files with content."""
    import shutil

    # Create a file that will be overwritten by template
    existing = tmp_path / "template_file.txt"
    existing.write_text("original content before template")

    # Create a template directory with a file that will overwrite existing
    template_dir = tmp_path / "_templates"
    template_dir.mkdir()
    template_file = template_dir / "template_file.txt"
    template_file.write_text("new content from template")

    # Empty plan - we're testing template_dir behavior
    plan = PlanResult(steps=[], add=0, change=0, delete=0, delete_skipped=0)

    res = execute_plan(plan, tmp_path, template_dir=template_dir, force=True)
    assert res["backed_up"] == 1

    # Verify backup was created
    backups_dir = tmp_path / ".seed" / "backups"
    backup_dirs = list(backups_dir.iterdir())
    assert len(backup_dirs) == 1

    backup_file = backup_dirs[0] / "template_file.txt"
    assert backup_file.exists()
    assert backup_file.read_text() == "original content before template"

    # Verify template was applied
    assert existing.read_text() == "new content from template"


# Tests for _backup_file_if_has_content helper function

def test_backup_helper_returns_none_for_nonexistent_file(tmp_path):
    """Test that backup helper returns None when file doesn't exist."""
    nonexistent = tmp_path / "nonexistent.txt"
    result = _backup_file_if_has_content(nonexistent, tmp_path)
    assert result is None


def test_backup_helper_returns_none_for_directory(tmp_path):
    """Test that backup helper returns None for directories."""
    directory = tmp_path / "somedir"
    directory.mkdir()
    result = _backup_file_if_has_content(directory, tmp_path)
    assert result is None


def test_backup_helper_returns_none_for_empty_file(tmp_path):
    """Test that backup helper returns None for empty files."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    result = _backup_file_if_has_content(empty_file, tmp_path)
    assert result is None


def test_backup_helper_creates_backup_dir_if_not_provided(tmp_path):
    """Test that backup helper creates backup dir when not provided."""
    file_with_content = tmp_path / "content.txt"
    file_with_content.write_text("some content")

    result = _backup_file_if_has_content(file_with_content, tmp_path)

    assert result is not None
    assert result.exists()
    assert result.read_text() == "some content"
    assert ".seed/backups" in str(result)


def test_backup_helper_uses_provided_backup_dir(tmp_path):
    """Test that backup helper uses provided backup directory."""
    file_with_content = tmp_path / "content.txt"
    file_with_content.write_text("some content")

    custom_backup_dir = tmp_path / "custom_backups"
    custom_backup_dir.mkdir()

    result = _backup_file_if_has_content(file_with_content, tmp_path, custom_backup_dir)

    assert result is not None
    assert result.exists()
    assert str(custom_backup_dir) in str(result)


def test_get_backup_dir_creates_timestamped_directory(tmp_path):
    """Test that _get_backup_dir creates a timestamped backup directory."""
    backup_dir = _get_backup_dir(tmp_path)

    assert backup_dir.exists()
    assert backup_dir.is_dir()
    assert ".seed/backups/backup_" in str(backup_dir)


# Tests for optional item handling

def test_executor_skip_optional_flag(tmp_path):
    """Test that --skip-optional flag skips optional items."""
    plan = PlanResult(
        steps=[
            PlanStep("mkdir", "required_dir", "missing", optional=False),
            PlanStep("mkdir", "optional_dir", "missing", optional=True),
            PlanStep("create", "required.txt", "missing", optional=False),
            PlanStep("create", "optional.txt", "missing", optional=True),
        ],
        add=4, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, skip_optional=True)

    assert (tmp_path / "required_dir").exists()
    assert not (tmp_path / "optional_dir").exists()
    assert (tmp_path / "required.txt").exists()
    assert not (tmp_path / "optional.txt").exists()
    assert res["created"] == 2
    assert res["skipped"] == 2


def test_executor_include_optional_flag(tmp_path):
    """Test that --yes flag creates all optional items."""
    plan = PlanResult(
        steps=[
            PlanStep("mkdir", "optional_dir", "missing", optional=True),
            PlanStep("create", "optional.txt", "missing", optional=True),
        ],
        add=2, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, include_optional=True)

    assert (tmp_path / "optional_dir").exists()
    assert (tmp_path / "optional.txt").exists()
    assert res["created"] == 2
    assert res["skipped"] == 0


def test_executor_non_interactive_skips_optional(tmp_path):
    """Test that non-interactive mode without --yes skips optional."""
    plan = PlanResult(
        steps=[
            PlanStep("create", "optional.txt", "missing", optional=True),
        ],
        add=1, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, interactive=False)

    assert not (tmp_path / "optional.txt").exists()
    assert res["skipped"] == 1


def test_executor_skipped_optional_parent_skips_children(tmp_path):
    """Test that children under skipped optional parent are also skipped."""
    plan = PlanResult(
        steps=[
            PlanStep("mkdir", "optional_parent", "missing", optional=True),
            PlanStep("create", "optional_parent/child.txt", "missing", optional=False),
        ],
        add=2, change=0, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, skip_optional=True)

    assert not (tmp_path / "optional_parent").exists()
    assert not (tmp_path / "optional_parent" / "child.txt").exists()
    assert res["skipped"] == 2


def test_executor_optional_update(tmp_path):
    """Test that optional update operations respect skip_optional flag."""
    # Create existing file
    (tmp_path / "config.txt").write_text("original")

    plan = PlanResult(
        steps=[
            PlanStep("update", "config.txt", "drift", optional=True),
        ],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, skip_optional=True, force=True)

    # File should not be updated when skip_optional=True
    assert res["skipped"] == 1


def test_executor_unknown_operation_raises(tmp_path):
    """Test that unknown operation raises ValueError."""
    plan = PlanResult(
        steps=[
            PlanStep("unknown_op", "file.txt", "reason"),
        ],
        add=0, change=0, delete=0, delete_skipped=0,
    )

    try:
        execute_plan(plan, tmp_path)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown plan operation" in str(e)


def test_executor_delete_directory(tmp_path):
    """Test that delete operation handles directories correctly."""
    # Create a directory with contents
    dir_path = tmp_path / "to_delete"
    dir_path.mkdir()
    (dir_path / "file.txt").write_text("content")

    plan = PlanResult(
        steps=[
            PlanStep("delete", "to_delete", "extra"),
        ],
        add=0, change=0, delete=1, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, dangerous=True)

    assert not dir_path.exists()
    assert res["deleted"] == 1


def test_executor_update_without_force_on_existing(tmp_path):
    """Test that update on existing file works without force."""
    f = tmp_path / "existing.txt"
    f.write_text("content")

    plan = PlanResult(
        steps=[PlanStep("update", "existing.txt", "drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=False)
    assert res["updated"] == 1


def test_executor_update_without_force_on_nonexistent(tmp_path):
    """Test that update without force on non-existing file does nothing."""
    plan = PlanResult(
        steps=[PlanStep("update", "nonexistent.txt", "drift")],
        add=0, change=1, delete=0, delete_skipped=0,
    )

    res = execute_plan(plan, tmp_path, force=False)
    # File shouldn't be created without force if it doesn't exist
    assert res["updated"] == 1
    assert not (tmp_path / "nonexistent.txt").exists()


def test_executor_backup_file_outside_base(tmp_path):
    """Test backup function handles files outside base directory gracefully."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("external content")
        external_path = Path(f.name)

    try:
        result = _backup_file_if_has_content(external_path, tmp_path)
        # Should backup using just the filename since it's outside base
        assert result is not None
        assert result.name == external_path.name
    finally:
        external_path.unlink()


def test_executor_template_dir_creates_nested_dirs(tmp_path):
    """Test that template_dir creates nested directory structure."""
    # Create template with nested structure
    template_dir = tmp_path / "_templates"
    nested = template_dir / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("content")

    plan = PlanResult(steps=[], add=0, change=0, delete=0, delete_skipped=0)

    execute_plan(plan, tmp_path, template_dir=template_dir, force=True)

    # Verify nested structure was created
    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


def test_executor_template_dir_respects_force_false(tmp_path):
    """Test that template_dir doesn't overwrite when force=False."""
    # Create existing file
    (tmp_path / "existing.txt").write_text("original")

    # Create template with same file
    template_dir = tmp_path / "_templates"
    template_dir.mkdir()
    (template_dir / "existing.txt").write_text("from template")

    plan = PlanResult(steps=[], add=0, change=0, delete=0, delete_skipped=0)

    execute_plan(plan, tmp_path, template_dir=template_dir, force=False)

    # Original file should be preserved
    assert (tmp_path / "existing.txt").read_text() == "original"
