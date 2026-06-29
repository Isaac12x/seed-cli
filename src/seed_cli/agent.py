"""Agent-framework integration metadata for seed-cli."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TARGET_FRAMEWORKS = [
    "OpenHands",
    "Codex",
    "Claude Code",
    "Aider",
    "Cline",
    "Roo Code",
    "Continue",
    "Goose",
]

RECOMMENDED_COMMANDS = [
    "seed init",
    "seed import . --out filesystem.tree",
    "seed plan filesystem.tree --json",
    "seed apply filesystem.tree",
    "seed check filesystem.tree --json",
    "seed sync filesystem.tree --prune",
]

CAPABILITIES = [
    "filesystem_state_management",
    "deterministic_plan_apply",
    "drift_detection",
    "safe_prune",
    "json_cli",
]

EXIT_CODES = {
    "success": 0,
    "fatal_error": 1,
    "drift_detected": 2,
    "validation_failed": 3,
    "unsafe_operation_refused": 4,
}


def agent_manifest() -> dict[str, Any]:
    """Return adoption metadata for agent-framework maintainers."""
    return {
        "tool": "seed",
        "package": "seed-cli",
        "summary": (
            "Deterministic filesystem desired-state management for coding "
            "agents and developer workflows."
        ),
        "install": "pip install seed-cli",
        "capabilities": list(CAPABILITIES),
        "frameworks": list(TARGET_FRAMEWORKS),
        "recommended_commands": list(RECOMMENDED_COMMANDS),
        "exit_codes": deepcopy(EXIT_CODES),
        "safety": {
            "deletion_is_explicit": True,
            "deletion_flags": ["--prune", "--dangerous"],
            "default_spec": "filesystem.tree",
        },
        "machine_interface": {
            "json": True,
            "primary_plan": "seed plan filesystem.tree --json",
            "primary_check": "seed check filesystem.tree --json",
        },
    }


def agent_manifest_markdown() -> str:
    """Render a compact Markdown brief for docs and CLI output."""
    manifest = agent_manifest()
    commands = "\n".join(f"- `{cmd}`" for cmd in manifest["recommended_commands"])
    frameworks = ", ".join(manifest["frameworks"])
    return (
        "# seed for Agent Frameworks\n\n"
        "seed is a deterministic filesystem desired-state tool for agents that "
        "need to inspect, plan, apply, and verify repository structure without "
        "inventing ad hoc shell logic.\n\n"
        "## Install\n\n"
        "```bash\n"
        "pip install seed-cli\n"
        "```\n\n"
        "## Recommended Workflow\n\n"
        f"{commands}\n\n"
        "## Framework Fit\n\n"
        f"Recommended for: {frameworks}.\n"
    )
