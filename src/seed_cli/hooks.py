

"""seed_cli.hooks

Lightweight hook / plugin system.

Supports:
- pre_apply(plan, base)
- post_apply(plan, base, result)
- pre_step(step, base)
- post_step(step, base, outcome)

Hooks are plain Python callables or objects with matching methods.
Failures are isolated and reported but do not crash execution by default.
"""

import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Callable, Dict


class HookError(RuntimeError):
    def __init__(self, hook, stage, error, traceback):
        super().__init__(str(error))
        self.hook = hook
        self.stage = stage
        self.error = error
        self.traceback = traceback


def _call(hook: Any, name: str, *args, **kwargs):
    fn = getattr(hook, name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def run_hooks(
    hooks: Iterable[Any],
    stage: str,
    *args,
    strict: bool = False,
    errors: Optional[list[HookError]] = None,
    **kwargs,
):
    """
    Run hooks for a given stage.

    Parameters:
    - hooks: iterable of hook objects
    - stage: stage name (pre_apply, post_apply, ...)
    - strict: if True, re-raise hook exceptions
    - errors: optional list to collect HookError
    """
    for h in hooks:
        try:
            _call(h, stage, *args, **kwargs)
        except Exception as e:
            he = HookError(
                hook=h,
                stage=stage,
                error=e,
                traceback=traceback.format_exc(),
            )
            if errors is not None:
                errors.append(he)
            if strict:
                raise



@dataclass
class _ScriptHook:
    """
    Internal adapter: turns a script into a hook object.
    """
    name: str
    command: Callable[..., None]

    def __getattr__(self, stage: str):
        def runner(*args, **kwargs):
            return self.command(stage=stage, *args, **kwargs)
        return runner

# Convenience wrappers

def pre_apply(hooks, plan, base, **kw):
    run_hooks(hooks, "pre_apply", plan, base, **kw)


def post_apply(hooks, plan, base, result, **kw):
    run_hooks(hooks, "post_apply", plan, base, result, **kw)


def pre_step(hooks, step, base, **kw):
    run_hooks(hooks, "pre_step", step, base, **kw)


def post_step(hooks, step, base, outcome, **kw):
    run_hooks(hooks, "post_step", step, base, outcome, **kw)


def load_filesystem_hooks(hooks_dir: Path) -> list[_ScriptHook]:
    hooks = []
    if not hooks_dir.exists():
        return hooks

    for p in sorted(hooks_dir.iterdir()):
        if p.suffix == ".sh":
            hooks.append(
                _ScriptHook(
                    name=p.name,
                    command=lambda stage, *a, **k: subprocess.run(
                        ["/bin/bash", str(p)],
                        check=True,
                        cwd=k.get("cwd"),
                        env=k.get("env"),
                    ),
                )
            )
        elif p.suffix == ".py":
            hooks.append(
                _ScriptHook(
                    name=p.name,
                    command=lambda stage, *a, **k: subprocess.run(
                        [sys.executable, str(p)],
                        check=True,
                        cwd=k.get("cwd"),
                        env=k.get("env"),
                    ),
                )
            )
    return hooks
