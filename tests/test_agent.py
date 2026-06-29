from seed_cli.agent import agent_manifest, agent_manifest_markdown


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
