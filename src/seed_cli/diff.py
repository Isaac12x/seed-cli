

"""seed_cli.diff

Diff the filesystem against the spec nodes.

This is intended to be a *human-friendly* report surface:
- missing: items present in spec but absent in filesystem
- extra: items present in filesystem but absent in spec
- type_mismatch: paths where both exist but one is file and other is dir
- drift: checksum drift for files (only when a recorded checksum exists)

Notes:
- Drift detection relies on `.seed/checksums.json` recorded by executor.
- If no checksum exists for a file, we do not mark it as drift (unknown baseline).
- Ignore patterns follow the same defaults as planning (e.g. .git/, .seed/).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import fnmatch

from .parsers import Node
from .checksums import load_checksums, sha256

DEFAULT_IGNORE = [
    ".git",
    ".git/**",
    "git",
    "git/**",
    ".seed",
    ".seed/**",
    "seed",
    "seed/**",
    "**/.DS_Store",
    "**/__pycache__/**",
]


@dataclass(frozen=True)
class DiffResult:
    missing: List[str]
    extra: List[str]
    type_mismatch: List[str]
    drift: List[str]

    def is_clean(self) -> bool:
        return not (self.missing or self.extra or self.type_mismatch or self.drift)


def _norm(p: Path) -> str:
    return p.as_posix().lstrip("./")


def _ignored(rel: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _fs_index(base: Path, ignore: List[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    if not base.exists():
        return idx
    for p in base.rglob("*"):
        rel = _norm(p.relative_to(base))
        if not rel:
            continue
        # Check if path or any parent matches ignore pattern
        parts = rel.split("/")
        ignored = False
        for i in range(len(parts)):
            check_path = "/".join(parts[:i+1])
            if _ignored(check_path, ignore) or _ignored(check_path + "/", ignore):
                ignored = True
                break
        if ignored:
            continue
        idx[rel] = "dir" if p.is_dir() else "file"
    return idx


def _spec_index(nodes: List[Node], ignore: List[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for n in nodes:
        rel = _norm(n.relpath)
        if not rel or _ignored(rel, ignore):
            continue
        idx[rel] = "dir" if n.is_dir else "file"
    return idx


def _is_sublevel_of_spec(rel: str, sidx: Dict[str, str]) -> bool:
    """Check if a path is a sublevel (child) of a directory defined in the spec.

    If spec defines 'src/' as a directory, then 'src/foo.py' is a sublevel
    and should not be reported as extra.
    """
    parts = rel.split("/")
    # Check each parent level
    for i in range(1, len(parts)):
        parent = "/".join(parts[:i])
        if parent in sidx and sidx[parent] == "dir":
            return True
    return False


def diff(
    spec_nodes: List[Node],
    base: Path,
    ignore: Optional[List[str]] = None,
    skip_sublevels: bool = False,
) -> DiffResult:
    """Diff filesystem against spec.

    Args:
        spec_nodes: Nodes from parsed spec
        base: Base directory to compare
        ignore: Additional ignore patterns
        skip_sublevels: If True, don't report extras inside directories defined in spec
    """
    ignore_patterns = (ignore or []) + DEFAULT_IGNORE
    sidx = _spec_index(spec_nodes, ignore_patterns)
    fidx = _fs_index(base, ignore_patterns)
    checks = load_checksums(base)

    missing: List[str] = []
    extra: List[str] = []
    type_mismatch: List[str] = []
    drift: List[str] = []

    # missing + type mismatches
    for rel in sorted(sidx.keys()):
        st = sidx[rel]
        ft = fidx.get(rel)
        if ft is None:
            missing.append(rel)
        elif ft != st:
            type_mismatch.append(rel)

    # extras
    for rel in sorted(fidx.keys()):
        if rel not in sidx:
            # Skip if this is a sublevel of a spec directory (when flag is set)
            if skip_sublevels and _is_sublevel_of_spec(rel, sidx):
                continue
            extra.append(rel)

    # drift (only for files with recorded checksum)
    for rel in sorted(sidx.keys()):
        if sidx[rel] != "file":
            continue
        if rel in type_mismatch or rel in missing:
            continue
        recorded = checks.get(rel, {}).get("sha256")
        if not recorded:
            continue
        p = base / rel
        if p.exists() and p.is_file():
            current = sha256(p)
            if current != recorded:
                drift.append(rel)

    return DiffResult(missing=missing, extra=extra, type_mismatch=type_mismatch, drift=drift)
