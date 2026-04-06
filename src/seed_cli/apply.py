

"""seed_cli.apply

High-level apply orchestration.

Responsibilities:
- Read spec or plan file
- Parse spec when needed
- Build plan (or load plan.json)
- Acquire state lock (if enabled)
- Execute plan via executor
- Persist resulting plan if requested

This is the glue between CLI and core engines.
"""
 
from pathlib import Path
import json
from typing import Optional

from .plugins import SeedPlugin
from .parsers import read_input, parse_any
from .planning import plan as build_plan, PlanResult
from .executor import execute_plan
from .state.local import LocalStateBackend
from .lock_heartbeat import LockHeartbeat
from .project_templates import prune_project_template_nodes
from .security import validate_plan_paths


def _load_plan(path: Path) -> PlanResult:
    doc = json.loads(path.read_text())
    steps = []
    for s in doc.get("steps", []):
        from .planning import PlanStep
        steps.append(PlanStep(**s))
    summ = doc.get("summary", {})
    return PlanResult(
        steps=steps,
        add=summ.get("add", 0),
        change=summ.get("change", 0),
        delete=summ.get("delete", 0),
        delete_skipped=summ.get("delete_skipped", 0),
    )


def apply(
    spec_or_plan_path: str,
    base: Path,
    *,
    dangerous: bool = False,
    plugins: Optional[list[SeedPlugin]] = None,
    force: bool = False,
    dry_run: bool = False,
    gitkeep: bool = False,
    template_dir: Optional[Path] = None,
    vars: Optional[dict] = None,
    ignore: Optional[list[str]] = None,
    allow_delete: bool = False,
    targets: Optional[list[str]] = None,
    target_mode: str = "prefix",
    lock: bool = True,
    lock_ttl: int = 300,
    lock_timeout: int = 30,
    lock_renew: int = 10,
    snapshot: bool = True,
    interactive: bool = True,
    skip_optional: bool = False,
    include_optional: bool = False,
    step_hooks: Optional[list[object]] = None,
    template_exclude: Optional[list[str]] = None,
    template_skip_if_exists: Optional[list[str]] = None,
) -> dict:
    """Apply a spec or plan.

    If `spec_or_plan_path` ends with `.json`, it is treated as a saved plan.
    Otherwise it is parsed as a spec and planned before execution.

    Creates a snapshot before making changes (unless dry_run or snapshot=False).
    Use `seed revert` to undo changes.
    """
    base = base.resolve()
    path = Path(spec_or_plan_path)
    plugins = plugins or []
    step_hooks = step_hooks or []


    # Load or build plan
    captured_spec_content: Optional[str] = None
    if path.suffix == ".json":
        plan = _load_plan(path)
    else:
        from .capture import to_tree_text
        from .parsers import parse_spec
        _, nodes = parse_spec(spec_or_plan_path, vars=vars, base=base)
        captured_spec_content = to_tree_text(nodes)
        nodes_for_plan = prune_project_template_nodes(nodes)
        plan = build_plan(
            nodes_for_plan,
            base,
            ignore=ignore,
            allow_delete=allow_delete,
            targets=targets,
            target_mode=target_mode,
            include_extras=allow_delete,
        )
    validate_plan_paths(plan, base)

    context = {
        "base": base,
        "plan": plan,
        "plugins": plugins,
        "cmd": "apply",
    }

    for p in plugins:
        p.before_build(plan, context)

    # Create snapshot before making changes
    snapshot_id = None
    if snapshot and not dry_run and plan.steps:
        from .snapshot import create_snapshot
        snapshot_id = create_snapshot(base, plan, "apply", spec_or_plan_path)

    # State + locking
    heartbeat = None
    lock_id = None
    backend = None

    if lock and not dry_run:
        backend = LocalStateBackend(base)
        info = backend.acquire_lock(lock_ttl, lock_timeout)
        lock_id = info.get("lock_id")
        heartbeat = LockHeartbeat(
            renew_fn=lambda: backend.renew_lock(lock_id, lock_ttl),
            interval=lock_renew,
        )
        heartbeat.start()

    try:
        result = execute_plan(
            plan,
            base,
            dangerous=dangerous,
            force=force,
            dry_run=dry_run,
            gitkeep=gitkeep,
            template_dir=template_dir,
            plugins=plugins,
            interactive=interactive,
            skip_optional=skip_optional,
            include_optional=include_optional,
            vars=vars,
            step_hooks=step_hooks,
            template_exclude=template_exclude,
            template_skip_if_exists=template_skip_if_exists,
        )
    finally:
        if heartbeat:
            heartbeat.stop()
        if backend and lock_id:
            backend.release_lock(lock_id)

    for p in plugins:
        p.after_build(context)

    # Capture versioned spec after successful execution
    if not dry_run:
        from .spec_history import capture_spec
        spec_result = capture_spec(base, source_spec=captured_spec_content, ignore=ignore)
        if spec_result:
            result["spec_version"] = spec_result[0]
            result["spec_path"] = str(spec_result[1])

    if snapshot_id:
        result["snapshot_id"] = snapshot_id

    return result
