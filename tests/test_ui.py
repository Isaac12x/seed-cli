from seed_cli.ui import render_summary, render_list, render_kv, Summary, format_step


# Helper class to create step-like objects for testing format_step
class MockStep:
    def __init__(self, op, path, reason=None, optional=False):
        self.op = op
        self.path = path
        self.reason = reason
        self.optional = optional


# -----------------------------------------------------------------------------
# Tests for format_step
# -----------------------------------------------------------------------------

def test_format_step_basic():
    """Test basic step formatting with uppercase operation."""
    step = MockStep("mkdir", "src/components")
    result = format_step(step)
    assert "MKDIR" in result
    assert "src/components" in result


def test_format_step_with_reason():
    """Test step formatting includes reason in parentheses."""
    step = MockStep("create", "file.txt", reason="missing")
    result = format_step(step)
    assert "CREATE" in result
    assert "file.txt" in result
    assert "(missing)" in result


def test_format_step_optional_marker():
    """Test optional step has ? marker."""
    step = MockStep("create", "optional.txt", optional=True)
    result = format_step(step)
    assert "?" in result
    assert "optional.txt" in result


def test_format_step_mkdir_operation():
    """Test mkdir operation formatting."""
    step = MockStep("mkdir", "new_dir", reason="missing")
    result = format_step(step)
    assert result.startswith("MKDIR")


def test_format_step_create_operation():
    """Test create operation formatting."""
    step = MockStep("create", "new_file.py", reason="missing")
    result = format_step(step)
    assert result.startswith("CREATE")


def test_format_step_update_operation():
    """Test update operation formatting."""
    step = MockStep("update", "existing.py", reason="checksum_drift")
    result = format_step(step)
    assert result.startswith("UPDATE")
    assert "(checksum_drift)" in result


def test_format_step_delete_operation():
    """Test delete operation formatting."""
    step = MockStep("delete", "extra.txt", reason="extra")
    result = format_step(step)
    assert result.startswith("DELETE")


def test_format_step_skip_operation():
    """Test skip operation formatting."""
    step = MockStep("skip", "manual_file.txt", reason="manual")
    result = format_step(step)
    assert result.startswith("SKIP")


def test_format_step_alignment():
    """Test that operation is padded for alignment."""
    step = MockStep("mkdir", "path")
    result = format_step(step)
    # Operation should be 8 chars wide
    assert result.startswith("MKDIR   ")


def test_format_step_no_reason():
    """Test formatting when reason is None."""
    step = MockStep("create", "file.txt", reason=None)
    result = format_step(step)
    assert "(" not in result  # No parentheses if no reason


def test_format_step_optional_with_reason():
    """Test optional step with reason shows both."""
    step = MockStep("update", "config.json", reason="drift", optional=True)
    result = format_step(step)
    assert "?" in result
    assert "(drift)" in result


# -----------------------------------------------------------------------------
# Existing tests for render functions
# -----------------------------------------------------------------------------

def test_render_summary_plain():
    s = Summary(created=1, updated=2, deleted=3, skipped=4)
    out = render_summary(s)
    assert "Created" in out
    assert "4" in out


def test_render_list():
    out = render_list("Missing", ["a", "b"])
    assert "a" in out and "b" in out


def test_render_list_empty():
    out = render_list("Missing", [])
    assert "none" in out.lower()


def test_render_kv():
    out = render_kv("Vars", {"a": "1"})
    assert "a" in out and "1" in out


def test_render_kv_empty():
    """Test render_kv with empty dict returns 'none'."""
    out = render_kv("Empty", {})
    assert "none" in out.lower()


def test_render_summary_with_backed_up():
    """Test render_summary includes backed_up count when > 0."""
    s = Summary(created=1, updated=2, deleted=0, skipped=0, backed_up=5)
    out = render_summary(s)
    assert "5" in out
    # Either "Backed Up" or "backed_up" should appear
    assert "back" in out.lower()


def test_render_summary_all_zeros():
    """Test render_summary with all zero counts."""
    s = Summary(created=0, updated=0, deleted=0, skipped=0)
    out = render_summary(s)
    assert "0" in out


def test_render_list_single_item():
    """Test render_list with single item."""
    out = render_list("Files", ["single.txt"])
    assert "single.txt" in out


def test_render_list_many_items():
    """Test render_list with multiple items."""
    items = ["file1.txt", "file2.txt", "file3.txt", "file4.txt"]
    out = render_list("Many Files", items)
    for item in items:
        assert item in out


def test_render_kv_multiple_entries():
    """Test render_kv with multiple key-value pairs."""
    kv = {"key1": "value1", "key2": "value2", "key3": "value3"}
    out = render_kv("Config", kv)
    for k, v in kv.items():
        assert k in out
        assert v in out


def test_render_kv_special_characters():
    """Test render_kv handles special characters in values."""
    kv = {"path": "/usr/local/bin", "pattern": "*.txt"}
    out = render_kv("Paths", kv)
    assert "/usr/local/bin" in out
    assert "*.txt" in out


def test_format_step_long_path():
    """Test format_step with long path."""
    step = MockStep("create", "very/long/nested/path/to/some/file.txt", reason="missing")
    result = format_step(step)
    assert "very/long/nested/path/to/some/file.txt" in result


def test_format_step_special_chars_in_path():
    """Test format_step with special characters in path."""
    step = MockStep("create", "file-with_special.chars.txt", reason="missing")
    result = format_step(step)
    assert "file-with_special.chars.txt" in result


# -----------------------------------------------------------------------------
# Tests for render functions with _RICH disabled
# -----------------------------------------------------------------------------

def test_render_summary_without_rich():
    """Test render_summary without rich library (plain text)."""
    import seed_cli.ui as ui_module
    original_rich = ui_module._RICH

    try:
        ui_module._RICH = False
        s = Summary(created=5, updated=3, deleted=1, skipped=2)
        out = render_summary(s)

        assert "Created: 5" in out
        assert "Updated: 3" in out
        assert "Deleted: 1" in out
        assert "Skipped: 2" in out
    finally:
        ui_module._RICH = original_rich


def test_render_summary_without_rich_with_backup():
    """Test render_summary without rich includes backed_up when > 0."""
    import seed_cli.ui as ui_module
    original_rich = ui_module._RICH

    try:
        ui_module._RICH = False
        s = Summary(created=1, updated=0, deleted=0, skipped=0, backed_up=3)
        out = render_summary(s)

        assert "Backed Up: 3" in out
    finally:
        ui_module._RICH = original_rich


def test_render_list_without_rich():
    """Test render_list without rich library (plain text)."""
    import seed_cli.ui as ui_module
    original_rich = ui_module._RICH

    try:
        ui_module._RICH = False
        out = render_list("Items", ["item1", "item2", "item3"])

        assert "Items:" in out
        assert "- item1" in out
        assert "- item2" in out
        assert "- item3" in out
    finally:
        ui_module._RICH = original_rich


def test_render_kv_without_rich():
    """Test render_kv without rich library (plain text)."""
    import seed_cli.ui as ui_module
    original_rich = ui_module._RICH

    try:
        ui_module._RICH = False
        out = render_kv("Settings", {"key1": "val1", "key2": "val2"})

        assert "Settings:" in out
        assert "key1: val1" in out
        assert "key2: val2" in out
    finally:
        ui_module._RICH = original_rich


# Test with rich enabled (if available)

def test_render_summary_with_rich_if_available():
    """Test render_summary with rich library (if available)."""
    import seed_cli.ui as ui_module

    if ui_module._RICH:
        s = Summary(created=2, updated=1, deleted=0, skipped=0)
        out = render_summary(s)
        # Rich output should contain the values
        assert "2" in out
        assert "1" in out


def test_render_list_with_rich_if_available():
    """Test render_list with rich library (if available)."""
    import seed_cli.ui as ui_module

    if ui_module._RICH:
        out = render_list("Files", ["a.txt", "b.txt"])
        assert "a.txt" in out
        assert "b.txt" in out


def test_render_kv_with_rich_if_available():
    """Test render_kv with rich library (if available)."""
    import seed_cli.ui as ui_module

    if ui_module._RICH:
        out = render_kv("Config", {"port": "8080"})
        assert "port" in out
        assert "8080" in out
