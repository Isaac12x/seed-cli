from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE_NAME = "seed-cli"


def _read_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
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
    except OSError:
        return None
    return None


def get_version() -> str:
    local_version = _read_version_from_pyproject()
    if local_version:
        return local_version
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0"


__version__ = get_version()
