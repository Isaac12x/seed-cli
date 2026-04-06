"""Security helpers for filesystem path validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def normalize_relpath(path: str) -> str:
    """Normalize and validate a plan/spec relative path.

    Rejects absolute paths, parent traversal, empty paths, and obvious
    cross-platform absolute path encodings.
    """
    if not isinstance(path, str):
        raise ValueError(f"Path must be a string, got: {type(path).__name__}")

    raw = path.strip()
    if not raw:
        raise ValueError("Path cannot be empty")
    if "\x00" in raw:
        raise ValueError("Path contains NUL byte")
    if raw.startswith("\\\\") or _WINDOWS_DRIVE_RE.match(raw):
        raise ValueError(f"Absolute path is not allowed: {path}")

    p = Path(raw)
    if p.is_absolute():
        raise ValueError(f"Absolute path is not allowed: {path}")

    parts = [part for part in p.parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Invalid path: {path}")
    if any(part == ".." for part in parts):
        raise ValueError(f"Parent traversal is not allowed: {path}")

    return Path(*parts).as_posix()


def safe_target_path(base: Path, relpath: str) -> Path:
    """Resolve `relpath` under `base`, rejecting escapes."""
    base = base.resolve()
    norm = normalize_relpath(relpath)
    target = (base / norm).resolve(strict=False)
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"Path escapes base directory: {relpath}") from exc
    return target


def validate_plan_paths(plan, base: Path) -> None:
    """Validate and normalize plan step paths in-place."""
    for step in getattr(plan, "steps", []):
        step.path = normalize_relpath(step.path)
        safe_target_path(base, step.path)
