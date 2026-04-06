from pathlib import Path

import pytest

import seed_cli.maintenance as maintenance


def test_parse_project_manifest_without_repos_falls_back_to_single_system_target(tmp_path):
    source = tmp_path / "project.yml"
    source.write_text("name: demo\n", encoding="utf-8")

    targets = maintenance._parse_project_manifest(
        {
            "name": "demo",
            "path": "./workspace",
            "maintenance": {
                "goals": ["ensure_path"],
                "actions": [{"name": "echo", "command": "echo hello", "shell": True}],
            },
        },
        source,
    )

    assert len(targets) == 1
    assert targets[0].kind == "system"
    assert targets[0].name == "demo"
    assert targets[0].goals == ["ensure_path"]
    assert [action.name for action in targets[0].actions] == ["echo"]


def test_parse_target_uses_service_config_dir_as_default_path(tmp_path):
    source = tmp_path / "service.yml"
    source.write_text("name: demo\n", encoding="utf-8")

    target = maintenance._parse_target(
        {"kind": "service", "config_dir": "./config"},
        source=source,
    )

    assert target.kind == "service"
    assert target.path == str((tmp_path / "config").resolve())
    assert target.config_dir == str((tmp_path / "config").resolve())


def test_parse_goals_skips_default_path_checks_for_action_only_targets():
    assert maintenance._parse_goals(
        None,
        kind="system",
        has_known_paths=False,
        has_actions=True,
    ) == []


def test_parse_actions_normalizes_string_and_object_entries(tmp_path):
    actions = maintenance._parse_actions(
        [
            "echo hi",
            {
                "name": "print-env",
                "tool": "python -m demo",
                "args": ["--count", 2],
                "env": {"DEBUG": True},
                "cwd": " ./service ",
            },
        ],
        tmp_path / "maintenance.yml",
    )

    assert actions[0].shell is True
    assert actions[0].command == "echo hi"
    assert actions[1].tool == "python -m demo"
    assert actions[1].args == ["--count", "2"]
    assert actions[1].env == {"DEBUG": "True"}
    assert actions[1].cwd == "./service"


def test_expand_goal_ensure_paths_deduplicates_paths():
    target = maintenance.MaintenanceTarget(
        name="svc",
        kind="service",
        path="/tmp/service",
        config_dir="/tmp/service",
        data_dir="/tmp/data",
    )

    steps = maintenance._expand_goal(target, "ensure-paths")

    assert [step.path for step in steps] == ["/tmp/service", "/tmp/data"]


def test_expand_goal_launchctl_restart_requires_launch_agent():
    target = maintenance.MaintenanceTarget(
        name="svc",
        kind="service",
        path="/tmp/service",
    )

    with pytest.raises(ValueError, match="requires launch_agent"):
        maintenance._expand_goal(target, "launchctl-restart")


def test_compose_step_rejects_unknown_engine():
    target = maintenance.MaintenanceTarget(
        name="svc",
        kind="service",
        path="/tmp/service",
        deploy_engine="podman-compose",
    )

    with pytest.raises(ValueError, match="docker compose engine"):
        maintenance._compose_step(target, "compose-pull", ["pull"], cwd=target.path)


def test_action_to_step_renders_shell_and_command_variants():
    target = maintenance.MaintenanceTarget(
        name="svc",
        kind="service",
        path="/tmp/service",
        metadata={"project": "demo"},
    )

    shell_step = maintenance._action_to_step(
        target,
        maintenance.MaintenanceAction(
            name="compose",
            tool="docker-compose",
            args=["pull"],
            shell=True,
        ),
    )
    argv_step = maintenance._action_to_step(
        target,
        maintenance.MaintenanceAction(
            name="custom",
            command="python -m http.server 9000",
        ),
    )

    assert shell_step.shell is True
    assert shell_step.command == "docker compose pull"
    assert shell_step.cwd == "/tmp/service"
    assert argv_step.argv == ["python", "-m", "http.server", "9000"]


def test_format_dirty_worktree_message_handles_multiple_shapes():
    assert maintenance._format_dirty_worktree_message(["README.md"]) == (
        "the worktree already has local changes in README.md"
    )
    assert "2 files: README.md, setup.py" in maintenance._format_dirty_worktree_message(
        ["README.md", "setup.py"]
    )
    assert "and 1 more" in maintenance._format_dirty_worktree_message(
        ["README.md", "setup.py", "src/app.py", "tests/test_app.py"]
    )
