"""Copier-style scaffolding helpers for template execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .templating import apply_vars


def load_data_file(path: Optional[str]) -> Dict[str, object]:
    """Load variables from a JSON/YAML data file."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Data file must contain a top-level mapping/object")
    return dict(data)


def load_template_config(path: Optional[Path]) -> Dict[str, object]:
    """Load a template config file (copier.yml or seed config)."""
    if not path:
        return {}
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ValueError(f"Invalid template config format in {path}")
    return dict(doc)


def _extract_questions(config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Extract question definitions from a copier-style config."""
    explicit = config.get("questions")
    if isinstance(explicit, dict):
        out: Dict[str, Dict[str, object]] = {}
        for name, spec in explicit.items():
            if isinstance(spec, dict):
                out[name] = dict(spec)
            else:
                out[name] = {"default": spec}
        return out

    out = {}
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _coerce_value(name: str, value, spec: Dict[str, object]):
    qtype = str(spec.get("type", "str")).lower()
    if value is None:
        return None
    if qtype in ("str", "string"):
        coerced = str(value)
    elif qtype in ("int", "integer"):
        coerced = int(value)
    elif qtype in ("float", "number"):
        coerced = float(value)
    elif qtype in ("bool", "boolean"):
        coerced = _coerce_bool(value)
    else:
        coerced = value

    choices = spec.get("choices")
    if isinstance(choices, list) and choices:
        if coerced not in choices:
            raise ValueError(f"Invalid choice for '{name}': {coerced} (choices: {choices})")
    return coerced


def resolve_template_vars(
    *,
    config: Dict[str, object],
    cli_vars: Optional[Dict[str, object]] = None,
    data_vars: Optional[Dict[str, object]] = None,
    defaults: bool = False,
    non_interactive: bool = False,
) -> Dict[str, object]:
    """Resolve final template variables using config questions + overrides."""
    cli_vars = dict(cli_vars or {})
    data_vars = dict(data_vars or {})

    merged: Dict[str, object] = {}
    merged.update(data_vars)
    merged.update(cli_vars)

    for name, spec in _extract_questions(config).items():
        if name in merged:
            merged[name] = _coerce_value(name, merged[name], spec)
            continue

        default = spec.get("default")
        required = bool(spec.get("required", False))
        prompt_text = spec.get("help") or spec.get("prompt") or name

        if non_interactive or defaults:
            if default is None and required:
                raise ValueError(f"Missing required template variable: {name}")
            if default is not None:
                merged[name] = _coerce_value(name, default, spec)
            continue

        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{suffix}: ").strip()
        if not raw:
            raw = default
        if raw is None:
            if required:
                raise ValueError(f"Missing required template variable: {name}")
            continue
        merged[name] = _coerce_value(name, raw, spec)

    return merged


def template_tasks(config: Dict[str, object]) -> List[str]:
    """Extract task commands from config."""
    raw_tasks = config.get("_tasks", config.get("tasks", []))
    if raw_tasks is None:
        return []
    if not isinstance(raw_tasks, list):
        raise ValueError("Template tasks must be a list")

    out: List[str] = []
    for item in raw_tasks:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and isinstance(item.get("command"), str):
            out.append(item["command"])
        else:
            raise ValueError(f"Invalid task entry: {item!r}")
    return out


def template_exclude(config: Dict[str, object]) -> List[str]:
    vals = config.get("_exclude", [])
    if not vals:
        return []
    if isinstance(vals, list):
        return [str(v) for v in vals]
    raise ValueError("Template _exclude must be a list")


def template_skip_if_exists(config: Dict[str, object]) -> List[str]:
    vals = config.get("_skip_if_exists", [])
    if not vals:
        return []
    if isinstance(vals, list):
        return [str(v) for v in vals]
    raise ValueError("Template _skip_if_exists must be a list")


def template_answers_file(config: Dict[str, object], override: Optional[str]) -> Optional[str]:
    if override:
        return override
    value = config.get("_answers_file")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def write_answers(path: Path, vars_dict: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(vars_dict, sort_keys=True), encoding="utf-8")


def run_template_tasks(
    tasks: List[str],
    *,
    vars_dict: Dict[str, object],
    cwd: Path,
    unsafe: bool,
) -> List[str]:
    """Run template tasks after rendering."""
    if not tasks:
        return []
    if not unsafe:
        return []

    executed: List[str] = []
    render_vars = {k: str(v) for k, v in vars_dict.items()}
    for task in tasks:
        cmd = apply_vars(task, render_vars, mode="loose")
        subprocess.run(cmd, shell=True, cwd=cwd, check=True)
        executed.append(cmd)
    return executed
