from importlib.metadata import PackageNotFoundError

import seed_cli


def test_read_version_from_pyproject_returns_none_when_file_is_missing(monkeypatch):
    def raise_os_error(self, encoding="utf-8"):
        raise OSError("missing file")

    monkeypatch.setattr(seed_cli.Path, "read_text", raise_os_error)

    assert seed_cli._read_version_from_pyproject() is None


def test_read_version_from_pyproject_returns_none_when_project_has_no_version(monkeypatch):
    monkeypatch.setattr(
        seed_cli.Path,
        "read_text",
        lambda self, encoding="utf-8": "[project]\nname = 'seed-cli'\n[tool.pytest]\n",
    )

    assert seed_cli._read_version_from_pyproject() is None


def test_get_version_falls_back_to_installed_metadata(monkeypatch):
    monkeypatch.setattr(seed_cli, "_read_version_from_pyproject", lambda: None)
    monkeypatch.setattr(seed_cli, "version", lambda package_name: f"{package_name}-9.9.9")

    assert seed_cli.get_version() == "seed-cli-9.9.9"


def test_get_version_returns_zero_when_package_metadata_is_missing(monkeypatch):
    monkeypatch.setattr(seed_cli, "_read_version_from_pyproject", lambda: None)

    def raise_not_found(_: str) -> str:
        raise PackageNotFoundError("seed-cli")

    monkeypatch.setattr(seed_cli, "version", raise_not_found)

    assert seed_cli.get_version() == "0.0.0"
