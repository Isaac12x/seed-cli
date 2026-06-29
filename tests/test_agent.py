from seed_cli.agent import (
    agent_integration,
    agent_manifest,
    agent_manifest_markdown,
)


def test_agent_manifest_names_target_frameworks():
    manifest = agent_manifest()

    assert manifest["tool"] == "seed"
    assert "filesystem_state_management" in manifest["capabilities"]
    for framework in [
        "OpenHands",
        "Codex",
        "Claude Code",
        "Aider",
        "Cline",
        "Roo Code",
        "Continue",
        "Goose",
    ]:
        assert framework in manifest["frameworks"]


def test_agent_manifest_markdown_includes_install_and_json_workflow():
    markdown = agent_manifest_markdown()

    assert "pip install seed-cli" in markdown
    assert "seed plan filesystem.tree --json" in markdown
    assert "seed apply filesystem.tree" in markdown


def test_agent_integration_returns_framework_specific_pack():
    integration = agent_integration("codex")

    assert integration["framework"] == "Codex"
    assert integration["slug"] == "codex"
    assert "seed plan filesystem.tree --json" in integration["default_instruction"]
    assert "seed check filesystem.tree --json" in integration["default_instruction"]
    assert integration["recommended_commands"] == [
        "seed init",
        "seed import . --out filesystem.tree",
        "seed plan filesystem.tree --json",
        "seed apply filesystem.tree",
        "seed check filesystem.tree --json",
        "seed sync filesystem.tree --prune",
    ]
