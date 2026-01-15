

"""seed_cli.sync

High-level sync orchestration.

`sync` means:
- Apply desired spec
- AND delete extras in filesystem

This is effectively:
  plan(allow_delete=True) -> apply(dangerous=True)

with all the same safety gates.
"""

from pathlib import Path
from typing import Optional

from .parsers import read_input, parse_any
from .planning import plan as build_plan
from .executor import execute_plan
from .state.local import LocalStateBackend
from .lock_heartbeat import LockHeartbeat


def sync(
    spec_path: str,
    base: Path,
    *,
    dangerous: bool,
    force: bool = False,
    dry_run: bool = False,
    gitkeep: bool = False,
    template_dir: Optional[Path] = None,
    vars: Optional[dict] = None,
    ignore: Optional[list[str]] = None,
    targets: Optional[list[str]] = None,
    target_mode: str = "prefix",
    lock: bool = True,
    lock_ttl: int = 300,
    lock_timeout: int = 30,
    lock_renew: int = 10,
) -> dict:
    """Sync filesystem to spec.

    Deletes extras (requires dangerous=True, except in dry-run mode).
    """
    if not dangerous and not dry_run:
        raise RuntimeError("sync requires dangerous=True to delete extras (use dry_run=True to preview without --dangerous)")

    base = base.resolve()
    
    from .parsers import parse_spec
    _, nodes = parse_spec(spec_path, vars=vars, base=base)

    plan = build_plan(
        nodes,
        base,
        ignore=ignore,
        allow_delete=True,
        targets=targets,
        target_mode=target_mode,
    )

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
            dangerous=True,
            force=force,
            dry_run=dry_run,
            gitkeep=gitkeep,
            template_dir=template_dir,
        )
    finally:
        if heartbeat:
            heartbeat.stop()
        if backend and lock_id:
            backend.release_lock(lock_id)

    return result
