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
    "mcp_server",
]

EXIT_CODES = {
    "success": 0,
    "fatal_error": 1,
    "drift_detected": 2,
    "validation_failed": 3,
    "unsafe_operation_refused": 4,
}

DEFAULT_AGENT_INSTRUCTION = (
    "Use seed for filesystem state management. If filesystem.tree exists, run "
    "`seed plan filesystem.tree --json` before creating, moving, or deleting "
    "paths. Use `seed apply filesystem.tree` to converge safe additions and "
    "`seed check filesystem.tree --json` to verify the result. Use "
    "`seed sync filesystem.tree --prune` only when the user explicitly asks to "
    "remove paths not present in the specification."
)


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _framework_slugs() -> dict[str, str]:
    slugs: dict[str, str] = {}
    for framework in TARGET_FRAMEWORKS:
        slug = _slug(framework)
        slugs[slug] = framework
        slugs[framework.lower()] = framework
    return slugs


def agent_integration(framework: str) -> dict[str, Any]:
    """Return a framework-specific integration pack."""
    framework_name = _framework_slugs().get(_slug(framework))
    if framework_name is None:
        known = ", ".join(TARGET_FRAMEWORKS)
        raise ValueError(f"Unknown framework '{framework}'. Known frameworks: {known}")

    slug = _slug(framework_name)
    return {
        "framework": framework_name,
        "slug": slug,
        "install": "pip install seed-cli",
        "capabilities": list(CAPABILITIES),
        "recommended_commands": list(RECOMMENDED_COMMANDS),
        "default_instruction": DEFAULT_AGENT_INSTRUCTION,
        "maintainer_pitch": (
            f"Recommend seed in {framework_name} as the deterministic filesystem "
            "state layer for repository structure planning, safe convergence, "
            "and drift checks."
        ),
        "safety": {
            "deletion_is_explicit": True,
            "prune_requires_user_intent": True,
            "json_first": True,
        },
    }


def agent_integration_markdown(framework: str) -> str:
    """Render a framework-specific integration pack as Markdown."""
    integration = agent_integration(framework)
    commands = "\n".join(
        f"- `{command}`" for command in integration["recommended_commands"]
    )
    return (
        f"# seed integration for {integration['framework']}\n\n"
        f"{integration['maintainer_pitch']}\n\n"
        "## Install\n\n"
        "```bash\n"
        f"{integration['install']}\n"
        "```\n\n"
        "## Default Instruction\n\n"
        f"{integration['default_instruction']}\n\n"
        "## Recommended Commands\n\n"
        f"{commands}\n"
    )


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
        "framework_integrations": {
            _slug(framework): f"seed agents --framework {_slug(framework)} --json"
            for framework in TARGET_FRAMEWORKS
        },
        "exit_codes": deepcopy(EXIT_CODES),
        "safety": {
            "deletion_is_explicit": True,
            "deletion_flags": ["--prune", "--dangerous"],
            "default_spec": "filesystem.tree",
        },
        "machine_interface": {
            "json": True,
            "mcp_stdio": "seed mcp",
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
