"""Wrappers around the real `gbrain` CLI.

All wrappers are best-effort: if the binary is missing or the call fails, we
return ``None`` / surface a diagnostic so the seed exporter can continue with
manual instructions (PRD R5: feature-detect, never hard-fail).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


GBRAIN_BIN_ENV = "GBRAIN_BIN"


def find_binary() -> Optional[str]:
    explicit = os.environ.get(GBRAIN_BIN_ENV)
    if explicit:
        return explicit if Path(explicit).exists() else None
    return shutil.which("gbrain")


@dataclass
class GbrainResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _run(argv: Sequence[str], *, cwd: Optional[Path] = None, timeout: float = 30.0) -> GbrainResult:
    binary = find_binary()
    if not binary:
        return GbrainResult(ok=False, stderr="gbrain binary not found on PATH")
    try:
        proc = subprocess.run(
            [binary, *argv],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
        return GbrainResult(ok=False, stderr=f"failed to invoke gbrain: {e}")
    return GbrainResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
    )


def version() -> Optional[str]:
    result = _run(["--version"])
    if not result.ok:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = line.split()
    return parts[1] if len(parts) >= 2 else None


def schema_validate(name: str) -> GbrainResult:
    return _run(["schema", "validate", name])


def schema_use(name: str) -> GbrainResult:
    return _run(["schema", "use", name])


def schema_active() -> GbrainResult:
    return _run(["schema", "active"])


def schema_list() -> GbrainResult:
    return _run(["schema", "list"])


def schema_show_json(name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    argv = ["schema", "show", "--json"]
    if name:
        argv.append(name)
    result = _run(argv)
    if not result.ok or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def submit_unify_types(target_pack: str, *, dry_run: bool = True) -> GbrainResult:
    params = json.dumps({"target_pack": target_pack})
    argv = ["jobs", "submit", "unify-types", "--allow-protected", "--params", params]
    if dry_run:
        argv.append("--dry-run")
    return _run(argv, timeout=60.0)


def sync(cwd: Optional[Path] = None) -> GbrainResult:
    return _run(["sync"], cwd=cwd, timeout=120.0)


def list_pages(*, page_type: Optional[str] = None) -> GbrainResult:
    argv = ["list", "--json"]
    if page_type:
        argv.extend(["--type", page_type])
    return _run(argv, timeout=60.0)


def is_available() -> bool:
    return find_binary() is not None
