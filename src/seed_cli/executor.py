

"""seed_cli.executor

Applies a PlanResult to the filesystem.

Responsibilities:
- Execute mkdir / create / update / delete steps
- Respect dry-run, force, dangerous flags
- Create .gitkeep for empty directories when requested
- Apply templates (directory copy) before execution
- Record checksums after successful execution
- Invoke plugin hooks (if provided)

This module is intentionally imperative and side-effectful.
"""

from pathlib import Path
from typing import Optional, Dict, List
import shutil

from .planning import PlanResult, PlanStep
from .checksums import sha256, load_checksums, save_checksums


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _touch(path: Path) -> None:
    path.touch(exist_ok=True)


def execute_plan(
    plan: PlanResult,
    base: Path,
    dangerous: bool = False,
    force: bool = False,
    dry_run: bool = False,
    gitkeep: bool = False,
    template_dir: Optional[Path] = None,
    plugins: Optional[List[object]] = None,
) -> Dict[str, int]:
    """Execute a plan against the filesystem.

    Returns counters: {created, updated, deleted, skipped}
    """
    counters = {"created": 0, "updated": 0, "deleted": 0, "skipped": 0}

    plugins = plugins or []

    if template_dir and not dry_run:
        if template_dir.exists():
            for item in template_dir.rglob("*"):
                rel = item.relative_to(template_dir)
                target = base / rel
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if force or not target.exists():
                        shutil.copy2(item, target)

    checks = load_checksums(base)

    for step in plan.steps:
        target = base / step.path

        if step.op == "skip":
            counters["skipped"] += 1
            continue

        if step.op == "mkdir":
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
                if gitkeep:
                    keep = target / ".gitkeep"
                    keep.touch(exist_ok=True)
            counters["created"] += 1
            continue

        if step.op in ("create", "update"):
            if not dry_run:
                _ensure_parent(target)
                if step.op == "create":
                    _touch(target)
                else:
                    if force or target.exists():
                        _touch(target)
            counters["created" if step.op == "create" else "updated"] += 1

            if not dry_run and target.exists() and target.is_file():
                checks[step.path] = {
                    "sha256": sha256(target),
                    "annotation": step.annotation,
                }
            continue

        if step.op == "delete":
            if not dangerous:
                raise RuntimeError("Refusing to delete without dangerous=True")
            if not dry_run and target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            counters["deleted"] += 1
            checks.pop(step.path, None)
            continue

        raise ValueError(f"Unknown plan operation: {step.op}")

    if not dry_run:
        save_checksums(base, checks)

    return counters
