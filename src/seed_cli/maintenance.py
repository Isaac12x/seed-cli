"""Manifest-driven repository and system maintenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .templating import apply_vars


DISCOVER_FILENAMES = {
    "maintenance.yml",
    "maintenance.yaml",
    "project.yml",
    "project.yaml",
    "service.yml",
    "service.yaml",
}

KIND_ALIASES = {
    "repo": "repository",
    "repository": "repository",
    "service": "service",
    "system": "system",
    "integration": "integration",
    "project": "project",
}

DEFAULT_GOALS = {
    "repository": ["ensure_path"],
    "service": ["ensure_paths"],
    "system": ["ensure_path"],
    "integration": ["ensure_path"],
}


@dataclass
class MaintenanceAction:
    name: str
    tool: str = "command"
    args: List[str] = field(default_factory=list)
    command: Optional[str] = None
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    shell: bool = False


@dataclass
class MaintenanceTarget:
    name: str
    kind: str
    path: Optional[str]
    goals: List[str] = field(default_factory=list)
    actions: List[MaintenanceAction] = field(default_factory=list)
    config_dir: Optional[str] = None
    data_dir: Optional[str] = None
    compose_file: Optional[str] = None
    deploy_engine: Optional[str] = None
    launch_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None


@dataclass
class MaintenanceStep:
    op: str  # check_path | run
    target: str
    name: str
    reason: str
    path: Optional[str] = None
    tool: Optional[str] = None
    argv: Optional[List[str]] = None
    command: Optional[str] = None
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    shell: bool = False

    def display(self) -> str:
        if self.op == "check_path":
            return f"CHECK    {self.target}: {self.path} ({self.reason})"

        if self.shell and self.command:
            cmd = self.command
        else:
            cmd = shlex.join(self.argv or [])
        cwd = f" [cwd={self.cwd}]" if self.cwd else ""
        return f"RUN      {self.target}: {cmd}{cwd} ({self.reason})"


@dataclass
class MaintenancePlan:
    sources: List[Path]
    targets: List[MaintenanceTarget]
    steps: List[MaintenanceStep]

    @property
    def checks(self) -> int:
        return sum(1 for step in self.steps if step.op == "check_path")

    @property
    def commands(self) -> int:
        return sum(1 for step in self.steps if step.op == "run")

    def to_text(self) -> str:
        lines = [
            (
                f"Maintenance plan: {len(self.targets)} targets, "
                f"{self.checks} checks, {self.commands} commands"
            ),
            "",
            "Actions:",
        ]
        for step in self.steps:
            lines.append(step.display())
        return "\n".join(lines)


def discover_maintenance_manifests(path: Path) -> List[Path]:
    target = path.expanduser().resolve(strict=False)
    if not target.exists():
        raise FileNotFoundError(f"Manifest path not found: {path}")

    if target.is_file():
        return [target]

    manifests = sorted(
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file() and candidate.name in DISCOVER_FILENAMES
    )
    if not manifests:
        raise FileNotFoundError(
            f"No maintenance manifests found under directory: {target}"
        )
    return manifests


def build_maintenance_plan(path: str | Path) -> MaintenancePlan:
    manifests = discover_maintenance_manifests(Path(path))
    targets: List[MaintenanceTarget] = []
    for manifest in manifests:
        targets.extend(_load_targets_from_manifest(manifest))

    steps: List[MaintenanceStep] = []
    seen: set[tuple[Any, ...]] = set()
    for target in targets:
        for step in _expand_target(target):
            signature = _step_signature(step)
            if signature in seen:
                continue
            seen.add(signature)
            steps.append(step)

    return MaintenancePlan(sources=manifests, targets=targets, steps=steps)


def execute_maintenance_plan(
    plan: MaintenancePlan,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    summary = {"checked": 0, "executed": 0, "skipped": 0}

    for step in plan.steps:
        if step.op == "check_path":
            if not dry_run:
                if not step.path:
                    raise ValueError(f"Check step '{step.name}' is missing a path")
                if not Path(step.path).expanduser().exists():
                    raise FileNotFoundError(
                        f"[{step.target}] required path is missing: {step.path}"
                    )
            summary["checked"] += 1
            continue

        if step.op != "run":
            raise ValueError(f"Unknown maintenance operation: {step.op}")

        if dry_run:
            summary["executed"] += 1
            continue

        if _is_git_pull_ff_only_step(step):
            dirty_files = _git_worktree_dirty_files(step.cwd)
            if dirty_files:
                raise RuntimeError(
                    f"[{step.target}] refusing git pull --ff-only because "
                    f"{_format_dirty_worktree_message(dirty_files)}"
                )

        env = dict(os.environ)
        env.update(step.env)
        if step.shell:
            subprocess.run(
                step.command,
                shell=True,
                cwd=step.cwd,
                env=env,
                check=True,
            )
        else:
            subprocess.run(
                step.argv or [],
                cwd=step.cwd,
                env=env,
                check=True,
            )
        summary["executed"] += 1

    return summary


def _is_git_pull_ff_only_step(step: MaintenanceStep) -> bool:
    return not step.shell and (step.argv or [])[:3] == ["git", "pull", "--ff-only"]


def _git_worktree_dirty_files(cwd: Optional[str]) -> List[str]:
    if not cwd:
        return []

    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=normal"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )

    dirty_files: List[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty_files.append(path)
    return dirty_files


def _format_dirty_worktree_message(paths: List[str]) -> str:
    if len(paths) == 1:
        return f"the worktree already has local changes in {paths[0]}"

    if len(paths) <= 3:
        joined = ", ".join(paths)
        return f"the worktree already has local changes in {len(paths)} files: {joined}"

    preview = ", ".join(paths[:3])
    remaining = len(paths) - 3
    return (
        "the worktree already has local changes in "
        f"{len(paths)} files, including {preview}, and {remaining} more"
    )


def _load_targets_from_manifest(path: Path) -> List[MaintenanceTarget]:
    doc = _read_manifest_document(path)

    if "targets" in doc:
        raw_targets = doc["targets"]
        if not isinstance(raw_targets, list):
            raise ValueError(f"'targets' must be a list in {path}")

        maintenance = _parse_maintenance_block(doc.get("maintenance"), path)
        targets: List[MaintenanceTarget] = []
        for index, raw in enumerate(raw_targets):
            if not isinstance(raw, dict):
                raise ValueError(f"Target {index} in {path} must be an object")
            item = dict(raw)
            if "goals" not in item and maintenance["goals"] is not None:
                item["goals"] = maintenance["goals"]
            if "actions" not in item and maintenance["actions"] is not None:
                item["actions"] = maintenance["actions"]
            targets.append(_parse_target(item, source=path))
        return targets

    kind = _normalize_kind(doc.get("kind") or doc.get("type") or "system")
    if kind == "project":
        return _parse_project_manifest(doc, path)

    item = dict(doc)
    maintenance = _parse_maintenance_block(item.pop("maintenance", None), path)
    if maintenance["goals"] is not None:
        item["goals"] = maintenance["goals"]
    if maintenance["actions"] is not None:
        item["actions"] = maintenance["actions"]
    item["kind"] = kind
    return [_parse_target(item, source=path, allow_implicit_path=True)]


def _read_manifest_document(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        doc = json.loads(text)
    else:
        doc = yaml.safe_load(text)
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ValueError(f"Maintenance manifest must be an object: {path}")
    return dict(doc)


def _parse_project_manifest(doc: Dict[str, Any], source: Path) -> List[MaintenanceTarget]:
    maintenance = _parse_maintenance_block(doc.get("maintenance"), source)
    repos = doc.get("repos", [])
    if repos and not isinstance(repos, list):
        raise ValueError(f"'repos' must be a list in {source}")

    project_root = _resolve_optional_path(doc.get("path"), source.parent)
    targets: List[MaintenanceTarget] = []

    for index, repo in enumerate(repos):
        if not isinstance(repo, dict):
            raise ValueError(f"Repo {index} in {source} must be an object")

        repo_doc = dict(repo)
        repo_maintenance = _parse_maintenance_block(repo_doc.pop("maintenance", None), source)
        repo_doc["kind"] = "repository"
        if "goals" not in repo_doc:
            repo_doc["goals"] = (
                repo_maintenance["goals"]
                if repo_maintenance["goals"] is not None
                else maintenance["goals"]
            )
        if "actions" not in repo_doc:
            repo_doc["actions"] = (
                repo_maintenance["actions"]
                if repo_maintenance["actions"] is not None
                else maintenance["actions"]
            )

        if "name" not in repo_doc and doc.get("name"):
            raw_repo_path = str(repo_doc.get("path", f"repo-{index + 1}"))
            repo_doc["name"] = f"{doc['name']}:{Path(raw_repo_path).name}"

        targets.append(
            _parse_target(
                repo_doc,
                source=source,
                path_base=project_root or source.parent,
            )
        )

    if targets:
        return targets

    fallback = dict(doc)
    fallback["kind"] = "system"
    fallback.pop("repos", None)
    fallback.pop("maintenance", None)
    if maintenance["goals"] is not None:
        fallback["goals"] = maintenance["goals"]
    if maintenance["actions"] is not None:
        fallback["actions"] = maintenance["actions"]
    return [_parse_target(fallback, source=source, allow_implicit_path=True)]


def _parse_target(
    raw: Dict[str, Any],
    *,
    source: Path,
    path_base: Optional[Path] = None,
    allow_implicit_path: bool = False,
) -> MaintenanceTarget:
    kind = _normalize_kind(raw.get("kind") or raw.get("type") or "system")
    if kind == "project":
        kind = "system"

    target_base = path_base or source.parent
    path = _resolve_optional_path(raw.get("path"), target_base)
    config_dir = _resolve_optional_path(raw.get("config_dir"), source.parent)
    data_dir = _resolve_optional_path(raw.get("data_dir"), source.parent)

    if not path and kind == "service":
        path = config_dir or data_dir

    if not path and allow_implicit_path:
        path = str(source.parent.resolve(strict=False))

    actions = _parse_actions(raw.get("actions"), source)
    goals = _parse_goals(
        raw.get("goals"),
        kind=kind,
        has_known_paths=bool(path or config_dir or data_dir),
        has_actions=bool(actions),
    )

    compose_base = Path(path) if path else source.parent
    compose_file = _resolve_optional_path(raw.get("compose_file"), compose_base)

    name = str(raw.get("name") or Path(path or source.parent.as_posix()).name or source.stem)
    metadata = {
        key: value
        for key, value in raw.items()
        if key not in {"goals", "actions", "maintenance"}
    }

    return MaintenanceTarget(
        name=name,
        kind=kind,
        path=path,
        goals=goals,
        actions=actions,
        config_dir=config_dir,
        data_dir=data_dir,
        compose_file=compose_file,
        deploy_engine=_optional_text(raw.get("deploy_engine")),
        launch_agent=_optional_text(raw.get("launch_agent")),
        metadata=metadata,
        source=str(source),
    )


def _parse_goals(
    raw_goals: Any,
    *,
    kind: str,
    has_known_paths: bool,
    has_actions: bool,
) -> List[str]:
    if raw_goals is None:
        if not has_known_paths and has_actions:
            return []
        return list(DEFAULT_GOALS.get(kind, []))

    if not isinstance(raw_goals, list):
        raise ValueError("'goals' must be a list")
    return [str(goal) for goal in raw_goals]


def _parse_actions(raw_actions: Any, source: Path) -> List[MaintenanceAction]:
    if raw_actions is None:
        return []
    if not isinstance(raw_actions, list):
        raise ValueError(f"'actions' must be a list in {source}")

    actions: List[MaintenanceAction] = []
    for index, item in enumerate(raw_actions):
        if isinstance(item, str):
            actions.append(
                MaintenanceAction(
                    name=f"action-{index + 1}",
                    command=item,
                    shell=True,
                )
            )
            continue

        if not isinstance(item, dict):
            raise ValueError(f"Action {index} in {source} must be an object or string")

        args = item.get("args", item.get("run", []))
        if args is None:
            args = []
        if not isinstance(args, list):
            raise ValueError(f"Action {index} args in {source} must be a list")

        env = item.get("env", {})
        if env is None:
            env = {}
        if not isinstance(env, dict):
            raise ValueError(f"Action {index} env in {source} must be an object")

        action = MaintenanceAction(
            name=str(item.get("name") or f"action-{index + 1}"),
            tool=str(item.get("tool", "command")),
            args=[str(arg) for arg in args],
            command=_optional_text(item.get("command")),
            cwd=_optional_text(item.get("cwd")),
            env={str(key): str(value) for key, value in env.items()},
            shell=bool(item.get("shell", False)),
        )

        if not action.command and not action.args:
            raise ValueError(f"Action {index} in {source} must define args or command")

        actions.append(action)

    return actions


def _parse_maintenance_block(raw: Any, source: Path) -> Dict[str, Optional[Any]]:
    if raw is None:
        return {"goals": None, "actions": None}
    if not isinstance(raw, dict):
        raise ValueError(f"'maintenance' must be an object in {source}")
    goals = raw.get("goals")
    actions = raw.get("actions")
    return {
        "goals": goals,
        "actions": actions,
    }


def _expand_target(target: MaintenanceTarget) -> List[MaintenanceStep]:
    steps: List[MaintenanceStep] = []
    for raw_goal in target.goals:
        goal = _normalize_goal(raw_goal)
        steps.extend(_expand_goal(target, goal))

    for action in target.actions:
        steps.append(_action_to_step(target, action))

    return steps


def _expand_goal(target: MaintenanceTarget, goal: str) -> List[MaintenanceStep]:
    if goal == "ensure-path":
        path = _require_target_path(target, goal)
        return [
            MaintenanceStep(
                op="check_path",
                target=target.name,
                name="ensure-path",
                reason=goal,
                path=path,
            )
        ]

    if goal == "ensure-paths":
        paths = [target.path, target.config_dir, target.data_dir]
        unique_paths = []
        for path in paths:
            if path and path not in unique_paths:
                unique_paths.append(path)
        if not unique_paths:
            raise ValueError(f"Goal '{goal}' requires at least one path for target '{target.name}'")
        return [
            MaintenanceStep(
                op="check_path",
                target=target.name,
                name="ensure-paths",
                reason=goal,
                path=path,
            )
            for path in unique_paths
        ]

    if target.kind == "repository":
        repo_path = _require_target_path(target, goal)
        if goal == "git-fetch":
            return [_command_step(target, goal, "git", ["fetch", "--all", "--prune"], cwd=repo_path)]
        if goal == "git-status":
            return [_command_step(target, goal, "git", ["status", "--short", "--branch"], cwd=repo_path)]
        if goal == "git-pull-ff-only":
            return [_command_step(target, goal, "git", ["pull", "--ff-only"], cwd=repo_path)]

    if target.kind == "service":
        service_path = target.path
        if goal == "compose-pull":
            return [_compose_step(target, goal, ["pull"], cwd=service_path)]
        if goal == "compose-up":
            return [_compose_step(target, goal, ["up", "-d"], cwd=service_path)]
        if goal == "launchctl-restart":
            if not target.launch_agent:
                raise ValueError(
                    f"Goal '{goal}' requires launch_agent for target '{target.name}'"
                )
            return [
                _command_step(
                    target,
                    goal,
                    "launchctl",
                    ["kickstart", "-k", target.launch_agent],
                    cwd=service_path,
                )
            ]

    raise ValueError(f"Unsupported goal '{goal}' for target '{target.name}' ({target.kind})")


def _compose_step(
    target: MaintenanceTarget,
    reason: str,
    extra_args: List[str],
    *,
    cwd: Optional[str],
) -> MaintenanceStep:
    if target.deploy_engine:
        engine = _normalize_goal(target.deploy_engine)
        if engine not in {"docker-compose", "docker", "docker-compose-v2"}:
            raise ValueError(
                f"Compose goals require a docker compose engine for target '{target.name}'"
            )

    args: List[str] = []
    if target.compose_file:
        args.extend(["-f", target.compose_file])
    args.extend(extra_args)
    return _command_step(target, reason, "docker-compose", args, cwd=cwd)


def _action_to_step(target: MaintenanceTarget, action: MaintenanceAction) -> MaintenanceStep:
    render_vars = _target_vars(target)
    cwd = apply_vars(action.cwd, render_vars, mode="loose") if action.cwd else target.path
    env = {
        key: apply_vars(value, render_vars, mode="loose")
        for key, value in action.env.items()
    }

    command = apply_vars(action.command, render_vars, mode="loose") if action.command else None
    args = [apply_vars(arg, render_vars, mode="loose") for arg in action.args]

    if action.shell:
        if command is None:
            command = shlex.join(_render_tool_command(action.tool, args))
        return MaintenanceStep(
            op="run",
            target=target.name,
            name=action.name,
            reason=f"action:{action.name}",
            tool=action.tool,
            command=command,
            cwd=cwd,
            env=env,
            shell=True,
        )

    if command is not None and not args and action.tool == "command":
        argv = shlex.split(command)
    elif command is not None and not args:
        argv = _render_tool_command(action.tool, shlex.split(command))
    else:
        argv = _render_tool_command(action.tool, args)

    return MaintenanceStep(
        op="run",
        target=target.name,
        name=action.name,
        reason=f"action:{action.name}",
        tool=action.tool,
        argv=argv,
        cwd=cwd,
        env=env,
    )


def _command_step(
    target: MaintenanceTarget,
    reason: str,
    tool: str,
    args: List[str],
    *,
    cwd: Optional[str],
) -> MaintenanceStep:
    return MaintenanceStep(
        op="run",
        target=target.name,
        name=reason,
        reason=reason,
        tool=tool,
        argv=_render_tool_command(tool, args),
        cwd=cwd,
    )


def _render_tool_command(tool: str, args: List[str]) -> List[str]:
    normalized = _normalize_goal(tool)
    if normalized in {"command", ""}:
        return list(args)
    if normalized in {"docker-compose", "docker-compose-v2", "docker"}:
        return ["docker", "compose", *args]
    return [*shlex.split(tool), *args]


def _target_vars(target: MaintenanceTarget) -> Dict[str, str]:
    out: Dict[str, str] = {
        "name": target.name,
        "target_name": target.name,
        "kind": target.kind,
        "target_kind": target.kind,
    }
    if target.path:
        out["path"] = target.path
        out["target_path"] = target.path
    if target.config_dir:
        out["config_dir"] = target.config_dir
    if target.data_dir:
        out["data_dir"] = target.data_dir
    if target.compose_file:
        out["compose_file"] = target.compose_file
    if target.deploy_engine:
        out["deploy_engine"] = target.deploy_engine
    if target.launch_agent:
        out["launch_agent"] = target.launch_agent

    for key, value in target.metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = str(value)
    return out


def _step_signature(step: MaintenanceStep) -> tuple[Any, ...]:
    return (
        step.op,
        step.target,
        step.path,
        tuple(step.argv or []),
        step.command,
        step.cwd,
        step.shell,
    )


def _require_target_path(target: MaintenanceTarget, goal: str) -> str:
    if not target.path:
        raise ValueError(f"Goal '{goal}' requires 'path' for target '{target.name}'")
    return target.path


def _resolve_optional_path(value: Any, base_dir: Path) -> Optional[str]:
    text = _optional_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    return str(path)


def _normalize_kind(value: Any) -> str:
    key = _normalize_goal(str(value))
    normalized = KIND_ALIASES.get(key)
    if not normalized:
        raise ValueError(f"Unsupported maintenance target kind: {value}")
    return normalized


def _normalize_goal(value: str) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
