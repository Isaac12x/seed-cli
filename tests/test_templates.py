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
