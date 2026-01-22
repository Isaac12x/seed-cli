"""seed_cli.spec_history

Automatic versioned spec capture after filesystem modifications.

After each apply/sync/match operation, captures the resulting filesystem state
as a versioned spec file in .seed/specs/.

Structure:
    .seed/
    └── specs/
        ├── v1.tree      # first captured spec
        ├── v2.tree      # after second modification
        ├── v3.tree      # etc.
        └── current.tree # symlink or copy of latest version
"""

from pathlib import Path
from typing import List, Optional, Tuple
import re
import shutil
from datetime import datetime

from .capture import capture_nodes, to_tree_text
from .parsers import Node


SPECS_DIR = ".seed/specs"
VERSION_PATTERN = re.compile(r"^v(\d+)\.tree$")


def _get_specs_dir(base: Path) -> Path:
    """Get or create the specs directory."""
    specs_dir = base / SPECS_DIR
    specs_dir.mkdir(parents=True, exist_ok=True)
    return specs_dir


def _get_versions(specs_dir: Path) -> List[Tuple[int, Path]]:
    """Get all existing versions sorted by version number."""
    versions = []
    for p in specs_dir.iterdir():
        match = VERSION_PATTERN.match(p.name)
        if match:
            versions.append((int(match.group(1)), p))
    return sorted(versions, key=lambda x: x[0])


def _get_next_version(specs_dir: Path) -> int:
    """Get the next version number."""
    versions = _get_versions(specs_dir)
    if not versions:
        return 1
    return versions[-1][0] + 1


def _specs_differ(spec1: str, spec2: str) -> bool:
    """Check if two specs are different (ignoring whitespace variations)."""
    # Normalize and compare
    lines1 = set(line.strip() for line in spec1.strip().splitlines() if line.strip())
    lines2 = set(line.strip() for line in spec2.strip().splitlines() if line.strip())
    return lines1 != lines2


def capture_spec(
    base: Path,
    source_spec: Optional[str] = None,
    ignore: Optional[List[str]] = None,
    force: bool = False,
) -> Optional[Tuple[int, Path]]:
    """Capture current filesystem state as a versioned spec.

    Args:
        base: Base directory to capture
        source_spec: Original spec content (if available, used for comparison)
        ignore: Additional ignore patterns
        force: Create new version even if unchanged

    Returns:
        Tuple of (version_number, spec_path) if new version created, None otherwise
    """
    specs_dir = _get_specs_dir(base)

    # Capture current state
    nodes = capture_nodes(base, ignore=ignore)
    spec_content = to_tree_text(nodes)

    # Add header with timestamp
    header = f"# Captured: {datetime.now().isoformat()}\n"
    spec_with_header = header + spec_content

    # Check if different from latest version
    versions = _get_versions(specs_dir)
    if versions and not force:
        latest_path = versions[-1][1]
        latest_content = latest_path.read_text()
        # Strip header for comparison
        latest_lines = [l for l in latest_content.splitlines() if not l.startswith("#")]
        if not _specs_differ(spec_content, "\n".join(latest_lines)):
            # No changes, don't create new version
            return None

    # Create new version
    version = _get_next_version(specs_dir)
    spec_path = specs_dir / f"v{version}.tree"
    spec_path.write_text(spec_with_header)

    # Update current symlink/copy
    current_path = specs_dir / "current.tree"
    if current_path.exists() or current_path.is_symlink():
        current_path.unlink()

    # Use symlink on Unix, copy on Windows
    try:
        current_path.symlink_to(f"v{version}.tree")
    except OSError:
        # Fallback to copy if symlinks not supported
        shutil.copy2(spec_path, current_path)

    return (version, spec_path)


def list_spec_versions(base: Path) -> List[Tuple[int, Path, str]]:
    """List all captured spec versions.

    Returns:
        List of (version, path, timestamp) tuples
    """
    specs_dir = base / SPECS_DIR
    if not specs_dir.exists():
        return []

    result = []
    for version, path in _get_versions(specs_dir):
        # Extract timestamp from header
        content = path.read_text()
        timestamp = "unknown"
        for line in content.splitlines():
            if line.startswith("# Captured:"):
                timestamp = line.replace("# Captured:", "").strip()
                break
        result.append((version, path, timestamp))

    return result


def get_spec_version(base: Path, version: int) -> Optional[str]:
    """Get content of a specific spec version."""
    specs_dir = base / SPECS_DIR
    spec_path = specs_dir / f"v{version}.tree"
    if spec_path.exists():
        return spec_path.read_text()
    return None


def get_current_spec(base: Path) -> Optional[Tuple[int, str]]:
    """Get the current (latest) spec version.

    Returns:
        Tuple of (version, content) or None if no specs exist
    """
    specs_dir = base / SPECS_DIR
    if not specs_dir.exists():
        return None

    versions = _get_versions(specs_dir)
    if not versions:
        return None

    version, path = versions[-1]
    return (version, path.read_text())


def diff_spec_versions(base: Path, v1: int, v2: int) -> dict:
    """Compare two spec versions.

    Returns:
        Dict with 'added', 'removed', 'unchanged' lists of paths
    """
    content1 = get_spec_version(base, v1)
    content2 = get_spec_version(base, v2)

    if content1 is None or content2 is None:
        raise ValueError(f"Version not found: v{v1 if content1 is None else v2}")

    # Parse paths from content (skip comments)
    paths1 = set(
        line.strip() for line in content1.splitlines()
        if line.strip() and not line.startswith("#")
    )
    paths2 = set(
        line.strip() for line in content2.splitlines()
        if line.strip() and not line.startswith("#")
    )

    return {
        "added": sorted(paths2 - paths1),
        "removed": sorted(paths1 - paths2),
        "unchanged": sorted(paths1 & paths2),
    }
