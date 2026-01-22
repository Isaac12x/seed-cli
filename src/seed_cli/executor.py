

"""seed_cli.executor

Applies a PlanResult to the filesystem.

Responsibilities:
- Execute mkdir / create / update / delete steps
- Respect dry-run, force, dangerous flags
- Create .gitkeep for empty directories when requested
- Apply templates (directory copy) before execution
- Record checksums after successful execution
- Invoke plugin hooks (if provided)
- Prompt for optional items (marked with ?)

This module is intentionally imperative and side-effectful.
"""

from pathlib import Path
from typing import Optional, Dict, List, Set
import shutil
import sys

from .planning import PlanResult, PlanStep
from .checksums import sha256, load_checksums, save_checksums


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _touch(path: Path) -> None:
    path.touch(exist_ok=True)


def _prompt_optional(step: PlanStep) -> bool:
    """Prompt user whether to create an optional item.

    Returns True if user wants to create it, False otherwise.
    """
    item_type = "directory" if step.op == "mkdir" else "file"
    prompt = f"Create optional {item_type} '{step.path}'? [y/N] "

    try:
        response = input(prompt).strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()  # newline after ^C
        return False


def execute_plan(
    plan: PlanResult,
    base: Path,
    dangerous: bool = False,
    force: bool = False,
    dry_run: bool = False,
    gitkeep: bool = False,
    template_dir: Optional[Path] = None,
    plugins: Optional[List[object]] = None,
    interactive: bool = True,
    skip_optional: bool = False,
    include_optional: bool = False,
) -> Dict[str, int]:
    """Execute a plan against the filesystem.

    Args:
        plan: The plan to execute
        base: Base directory
        dangerous: Allow dangerous operations (deletions)
        force: Force overwrite existing files
        dry_run: Preview without executing
        gitkeep: Create .gitkeep in empty directories
        template_dir: Directory containing file templates
        plugins: List of plugins to invoke
        interactive: If True, prompt for optional items. If False, behavior depends on other flags.
        skip_optional: If True, skip all optional items without prompting.
        include_optional: If True, create all optional items without prompting (--yes flag).

    Returns counters: {created, updated, deleted, skipped}
    """
    counters = {"created": 0, "updated": 0, "deleted": 0, "skipped": 0}

    plugins = plugins or []

    # Track skipped optional paths so we can skip their children too
    skipped_optional_paths: Set[str] = set()

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

        # Check if this path is under a skipped optional parent
        if any(step.path.startswith(skipped + "/") for skipped in skipped_optional_paths):
            counters["skipped"] += 1
            continue

        if step.op == "skip":
            counters["skipped"] += 1
            continue

        # Handle optional items
        if step.optional and step.op in ("mkdir", "create", "update"):
            if include_optional:
                # --yes flag: create all optional items without prompting
                pass  # Continue to create the item
            elif skip_optional:
                # --skip-optional flag: skip all optional items
                counters["skipped"] += 1
                if step.op == "mkdir":
                    skipped_optional_paths.add(step.path)
                continue
            elif interactive and not dry_run:
                # Interactive mode: prompt user
                if not _prompt_optional(step):
                    counters["skipped"] += 1
                    if step.op == "mkdir":
                        skipped_optional_paths.add(step.path)
                    continue
            elif not interactive:
                # Non-interactive mode without --yes: skip optional items
                counters["skipped"] += 1
                if step.op == "mkdir":
                    skipped_optional_paths.add(step.path)
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
