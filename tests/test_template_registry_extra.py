from pathlib import Path

import pytest

import seed_cli.template_registry as template_registry
from seed_cli.template_registry import (
    TEMPLATES_DIR_NAME,
    add_local_template,
    fetch_from_github,
    get_template_config_path,
    get_template_content_dir,
    install_default_templates,
    _load_meta,
)


def test_get_template_content_dir_and_versioned_config_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    source_dir = tmp_path / "template"
    files_dir = source_dir / "files"
    files_dir.mkdir(parents=True)
    (source_dir / "spec.tree").write_text("src/\nfile.py", encoding="utf-8")
    (source_dir / ".seed-template.yml").write_text("message: hi", encoding="utf-8")
    (files_dir / "main.py").write_text("print('hi')", encoding="utf-8")

    add_local_template(str(source_dir), "demo")

    content_dir = get_template_content_dir("demo")
    config_path = get_template_config_path("demo")

    assert content_dir == tmp_path / ".seed" / TEMPLATES_DIR_NAME / "demo" / "v1"
    assert config_path == tmp_path / ".seed" / TEMPLATES_DIR_NAME / "demo" / "v1.config.yml"


def test_get_template_config_path_falls_back_to_content_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec_file = tmp_path / "spec.tree"
    content_dir = tmp_path / "content"
    spec_file.write_text("src/\nfile.py", encoding="utf-8")
    content_dir.mkdir()
    (content_dir / "copier.yml").write_text("subdirectory: src", encoding="utf-8")

    add_local_template(str(spec_file), "demo", content_dir=str(content_dir))

    assert get_template_config_path("demo") == (
        content_dir.parent
        / ".seed"
        / TEMPLATES_DIR_NAME
        / "demo"
        / "v1"
        / "copier.yml"
    )


def test_load_meta_returns_none_for_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    template_dir = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "broken"
    template_dir.mkdir(parents=True)
    (template_dir / "meta.json").write_text("{invalid", encoding="utf-8")

    assert _load_meta("broken") is None


def test_fetch_from_github_uses_gh_cli_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(template_registry, "_has_gh_cli", lambda: True)
    monkeypatch.setattr(
        template_registry,
        "_fetch_with_gh",
        lambda owner, repo, ref, path: "src/\nmain.py",
    )

    dest_file, original_name = fetch_from_github(
        "https://github.com/user/repo/blob/main/spec.tree",
        tmp_path,
        name="renamed.tree",
    )

    assert original_name == "spec.tree"
    assert dest_file.name == "renamed.tree"
    assert dest_file.read_text(encoding="utf-8") == "src/\nmain.py"


def test_fetch_from_github_falls_back_to_git_when_gh_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(template_registry, "_has_gh_cli", lambda: True)

    def raise_gh(*args, **kwargs):
        raise RuntimeError("gh failed")

    def fake_git(owner, repo, ref, path, dest_dir):
        fetched = dest_dir / "spec.tree"
        fetched.write_text("src/\nmain.py", encoding="utf-8")
        return fetched

    monkeypatch.setattr(template_registry, "_fetch_with_gh", raise_gh)
    monkeypatch.setattr(template_registry, "_fetch_with_git", fake_git)

    dest_file, original_name = fetch_from_github(
        "https://github.com/user/repo/blob/main/spec.tree",
        tmp_path,
        name="custom.tree",
    )

    assert original_name == "spec.tree"
    assert dest_file.name == "custom.tree"
    assert dest_file.read_text(encoding="utf-8") == "src/\nmain.py"


def test_fetch_from_github_wraps_git_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(template_registry, "_has_gh_cli", lambda: False)
    monkeypatch.setattr(
        template_registry,
        "_fetch_with_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("git failed")),
    )

    with pytest.raises(RuntimeError, match="Failed to fetch from GitHub: git failed"):
        fetch_from_github(
            "https://github.com/user/repo/blob/main/spec.tree",
            tmp_path,
        )


def test_install_default_templates_returns_when_resource_lookup_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(template_registry, "_defaults_installed", False)
    monkeypatch.setattr(
        template_registry.importlib.resources,
        "files",
        lambda package_name: (_ for _ in ()).throw(TypeError("no resources")),
    )

    install_default_templates()

    assert not (tmp_path / ".seed" / TEMPLATES_DIR_NAME / "registry.json").exists()


def test_install_default_templates_skips_preexisting_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(template_registry, "_defaults_installed", False)

    existing_spec = tmp_path / "existing.tree"
    existing_spec.write_text("src/\napp.py", encoding="utf-8")
    add_local_template(str(existing_spec), "demo")

    resources_root = tmp_path / "pkg_root"
    default_template_dir = resources_root / "resources" / "default_templates" / "demo"
    default_template_dir.mkdir(parents=True)
    (default_template_dir / "spec.tree").write_text("overwritten?", encoding="utf-8")

    monkeypatch.setattr(
        template_registry.importlib.resources,
        "files",
        lambda package_name: resources_root,
    )

    install_default_templates()

    spec_path = tmp_path / ".seed" / TEMPLATES_DIR_NAME / "demo" / "v1.tree"
    assert spec_path.read_text(encoding="utf-8") == "src/\napp.py"
