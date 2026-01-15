from seed_cli.ui import render_summary, render_list, render_kv, Summary


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
