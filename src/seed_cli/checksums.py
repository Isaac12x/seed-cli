

"""seed_cli.checksums

Checksum storage and utilities.

Responsibilities:
- Compute file checksums (SHA-256)
- Load/save checksum state from .seed/checksums.json
- Safe, deterministic format for drift detection

This module is used by:
- executor (record after apply)
- diff (detect drift)
"""

from pathlib import Path
from typing import Dict, Any
import hashlib
import json


CHECKSUMS_FILE = ".seed/checksums.json"


def sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(base: Path) -> Dict[str, Dict[str, Any]]:
    """Load checksum state from disk.

    Returns mapping:
      relpath -> { sha256: str, annotation: Optional[str] }
    """
    p = base / CHECKSUMS_FILE
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_checksums(base: Path, checksums: Dict[str, Dict[str, Any]]) -> None:
    """Persist checksum state to disk."""
    p = base / CHECKSUMS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(checksums, indent=2))
