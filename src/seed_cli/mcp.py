"""Minimal Model Context Protocol server for seed-cli.

The implementation intentionally stays dependency-light: it speaks JSON-RPC over
stdio and exposes seed's existing deterministic plan/check/apply operations as
MCP tools.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from seed_cli import get_version
from seed_cli.apply import apply as apply_spec
from seed_cli.diff import diff as diff_spec
from seed_cli.parsers import parse_spec
from seed_cli.planning import plan as build_plan

PROTOCOL_VERSION = "2025-06-18"


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _spec_aware_ignore(spec: str, base: Path, ignore: list[str] | None = None) -> list[str]:
    patterns = list(ignore or [])
    if spec == "-":
        return patterns

    spec_path = Path(spec)
    if not spec_path.is_absolute():
        spec_path = (base / spec_path).resolve()
    try:
        patterns.append(spec_path.relative_to(base.resolve()).as_posix())
    except ValueError:
        pass
    return patterns


def _resolve_spec(spec: str, base: Path) -> str:
    if spec == "-":
        return spec
    spec_path = Path(spec)
    if spec_path.is_absolute():
        return str(spec_path)
    return str((base / spec_path).resolve())


def _plan_status(plan) -> str:
    return "match" if not plan.steps else "drift"


def _plan_payload(plan) -> dict[str, Any]:
    raw = plan.to_json()
    status = _plan_status(plan)
    return {
        "status": status,
        "exitCode": 0 if status == "match" else 2,
        "create": [s.path for s in plan.steps if s.op in ("mkdir", "create")],
        "update": [s.path for s in plan.steps if s.op == "update"],
        "delete": [s.path for s in plan.steps if s.op == "delete"],
        "skipped_delete": [
            s.path for s in plan.steps if s.op == "skip" and s.reason == "extra"
        ],
        "rename": [],
        "errors": [],
        "summary": raw["summary"],
        "steps": raw["steps"],
    }


def _diff_payload(result) -> dict[str, Any]:
    status = "match" if result.is_clean() else "drift"
    return {
        "status": status,
        "exitCode": 0 if status == "match" else 2,
        "missing": result.missing,
        "extra": result.extra,
        "type_mismatch": result.type_mismatch,
        "drift": result.drift,
        "errors": [],
    }


def _text_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": _json_payload(payload),
            }
        ]
    }


def _tool_schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


TOOLS = [
    {
        "name": "seed_plan",
        "description": "Calculate filesystem drift from a seed tree specification.",
        "inputSchema": _tool_schema(
            ["spec"],
            {
                "spec": {"type": "string", "description": "Path to filesystem.tree"},
                "base": {"type": "string", "default": "."},
                "vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                },
            },
        ),
    },
    {
        "name": "seed_check",
        "description": "Return whether the filesystem matches a seed tree specification.",
        "inputSchema": _tool_schema(
            ["spec"],
            {
                "spec": {"type": "string", "description": "Path to filesystem.tree"},
                "base": {"type": "string", "default": "."},
                "vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                },
            },
        ),
    },
    {
        "name": "seed_apply",
        "description": (
            "Apply a seed tree specification. Deletion is disabled unless "
            "prune is explicitly true."
        ),
        "inputSchema": _tool_schema(
            ["spec"],
            {
                "spec": {"type": "string", "description": "Path to filesystem.tree"},
                "base": {"type": "string", "default": "."},
                "vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                },
                "prune": {"type": "boolean", "default": False},
                "dry_run": {"type": "boolean", "default": False},
            },
        ),
    },
]


def _seed_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    spec = str(arguments["spec"])
    base = Path(arguments.get("base", ".")).resolve()
    vars_ = dict(arguments.get("vars") or {})
    spec_path = _resolve_spec(spec, base)
    _, nodes = parse_spec(spec_path, vars=vars_, base=base)
    plan = build_plan(nodes, base, ignore=_spec_aware_ignore(spec, base))
    return _plan_payload(plan)


def _seed_check(arguments: dict[str, Any]) -> dict[str, Any]:
    spec = str(arguments["spec"])
    base = Path(arguments.get("base", ".")).resolve()
    vars_ = dict(arguments.get("vars") or {})
    spec_path = _resolve_spec(spec, base)
    _, nodes = parse_spec(spec_path, vars=vars_, base=base)
    result = diff_spec(nodes, base, ignore=_spec_aware_ignore(spec, base))
    return _diff_payload(result)


def _seed_apply(arguments: dict[str, Any]) -> dict[str, Any]:
    spec = str(arguments["spec"])
    base = Path(arguments.get("base", ".")).resolve()
    vars_ = dict(arguments.get("vars") or {})
    prune = bool(arguments.get("prune", False))
    dry_run = bool(arguments.get("dry_run", False))
    spec_path = _resolve_spec(spec, base)
    result = apply_spec(
        spec_path,
        base,
        dangerous=prune,
        dry_run=dry_run,
        vars=vars_,
        ignore=_spec_aware_ignore(spec, base),
        allow_delete=prune,
        interactive=False,
    )
    snapshot_id = result.pop("snapshot_id", None)
    spec_version = result.pop("spec_version", None)
    spec_path = result.pop("spec_path", None)
    payload: dict[str, Any] = {
        "status": "dry_run" if dry_run else "applied",
        "exitCode": 0,
        **result,
        "errors": [],
    }
    if snapshot_id:
        payload["snapshot_id"] = snapshot_id
    if spec_version:
        payload["spec_version"] = spec_version
        payload["spec_path"] = spec_path
    return payload


TOOL_HANDLERS = {
    "seed_plan": _seed_plan,
    "seed_check": _seed_check,
    "seed_apply": _seed_apply,
}


def _response(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message."""
    message_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return _response(
            message_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "seed-cli", "version": get_version()},
            },
        )

    if method == "tools/list":
        return _response(message_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return _error(message_id, -32602, f"Unknown tool: {name}")
        try:
            return _response(message_id, _text_result(handler(arguments)))
        except Exception as exc:
            return _response(
                message_id,
                {
                    "isError": True,
                    "content": [{"type": "text", "text": str(exc)}],
                },
            )

    return _error(message_id, -32601, f"Method not found: {method}")


def run_stdio(input_stream: TextIO | None = None, output_stream: TextIO | None = None) -> int:
    """Run the MCP server over newline-delimited JSON-RPC stdio."""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            response = handle_message(json.loads(line))
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"Parse error: {exc}")
        if response is not None:
            output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
            output_stream.flush()
    return 0
