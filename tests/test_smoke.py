from pathlib import Path

import seed_cli


def project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    in_project_section = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[project]":
            in_project_section = True
            continue
        if in_project_section and line.startswith("["):
            break
        if in_project_section and line.startswith("version"):
            _, value = line.split("=", 1)
            return value.strip().strip("\"'")
    raise AssertionError("Could not find [project].version in pyproject.toml")


def test_smoke_import():
    import seed_cli
    assert seed_cli.__version__


def test_get_version_prefers_repo_pyproject(monkeypatch):
    def fail_if_called(_: str) -> str:
        raise AssertionError("installed package metadata should not be consulted in a source checkout")

    monkeypatch.setattr(seed_cli, "version", fail_if_called)

    assert seed_cli.get_version() == project_version()
