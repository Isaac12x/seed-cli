"""Helpers for seed text-spec file formats."""

from __future__ import annotations

from pathlib import Path


TREE_LIKE_SUFFIXES: tuple[str, ...] = (".tree", ".seed")
DIRECTORY_SPEC_CANDIDATES: tuple[str, ...] = ("spec.seed", "spec.tree")


def is_tree_like_spec_path(path: Path | str) -> bool:
    """Return True when path uses a tree-like spec suffix."""
    return Path(path).suffix.lower() in TREE_LIKE_SUFFIXES


def preferred_tree_like_suffix(path: Path | str | None, *, default: str = ".tree") -> str:
    """Return a supported tree-like suffix for path or a default."""
    if path is None:
        return default

    suffix = Path(path).suffix.lower()
    if suffix in TREE_LIKE_SUFFIXES:
        return suffix
    return default


def resolve_directory_spec(directory: Path) -> Path | None:
    """Resolve a directory-backed spec file such as spec.seed or spec.tree."""
    for candidate_name in DIRECTORY_SPEC_CANDIDATES:
        candidate = directory / candidate_name
        if candidate.is_file():
            return candidate
    return None


def resolve_tree_like_path(path: Path) -> Path | None:
    """Resolve a path to an existing tree-like spec file when possible."""
    if path.is_file():
        return path

    if path.is_dir():
        return resolve_directory_spec(path)

    if path.suffix.lower() in TREE_LIKE_SUFFIXES:
        return path if path.exists() else None

    for suffix in TREE_LIKE_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate

    return None


def strip_tree_like_suffix(path: Path) -> Path:
    """Remove a supported tree-like suffix from path."""
    if path.suffix.lower() in TREE_LIKE_SUFFIXES:
        return path.with_suffix("")
    return path


def versioned_spec_path(directory: Path, version: str, *, suffix: str = ".tree") -> Path:
    """Return the canonical on-disk versioned spec path."""
    return directory / f"{version}{suffix}"

