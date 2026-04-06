import sys
from types import SimpleNamespace

import pytest

from seed_cli.maintenance import build_maintenance_plan, execute_maintenance_plan


def test_build_maintenance_plan_from_workspace_manifest(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    service_dir = tmp_path / "service"
    service_dir.mkdir()

    data_dir = tmp_path / "local" / "services" / "demo"
    data_dir.mkdir(parents=True)

    manifest = tmp_path / "maintenance.yml"
    manifest.write_text(
        """
targets:
  - name: workspace-repo
    kind: repository
    path: ./repo
    goals:
      - ensure_path
      - git_status
  - name: demo-service
    kind: service
    path: ./service
    config_dir: ./service/config
    data_dir: ./local/services/demo
    compose_file: compose.yml
    deploy_engine: docker-compose
    launch_agent: user/com.example.demo
    goals:
      - ensure_paths
      - compose_pull
      - launchctl_restart
""",
        encoding="utf-8",
    )

    plan = build_maintenance_plan(manifest)

    assert len(plan.targets) == 2
    assert plan.checks == 4
    assert plan.commands == 3
    assert any(step.argv == ["git", "status", "--short", "--branch"] for step in plan.steps)
    assert any(
        step.argv == ["docker", "compose", "-f", str(service_dir / "compose.yml"), "pull"]
        for step in plan.steps
    )
    assert any(
        step.argv == ["launchctl", "kickstart", "-k", "user/com.example.demo"]
        for step in plan.steps
    )


def test_project_manifest_expands_repositories_relative_to_project_root(tmp_path):
    project_root = tmp_path / "product-x"
    web_repo = project_root / "repos" / "web-app"
    api_repo = project_root / "repos" / "api"
    web_repo.mkdir(parents=True)
    api_repo.mkdir(parents=True)

    manifest = tmp_path / "project.yml"
    manifest.write_text(
        """
name: product-x
type: project
path: ./product-x
maintenance:
  goals:
    - git_fetch
repos:
  - name: web-app
    path: repos/web-app
  - name: api
    path: repos/api
""",
        encoding="utf-8",
    )

    plan = build_maintenance_plan(manifest)

    assert len(plan.targets) == 2
    assert {target.kind for target in plan.targets} == {"repository"}
    assert {target.path for target in plan.targets} == {
        str(web_repo.resolve()),
        str(api_repo.resolve()),
    }
    assert plan.checks == 0
    assert plan.commands == 2
    assert all(step.argv[:2] == ["git", "fetch"] for step in plan.steps)


def test_execute_maintenance_plan_runs_expected_commands(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    manifest = tmp_path / "maintenance.yml"
    manifest.write_text(
        f"""
targets:
  - name: py-task
    kind: system
    path: ./repo
    goals:
      - ensure_path
    actions:
      - name: write-marker
        tool: "{sys.executable}"
        args:
          - -c
          - "print('seed-maintenance')"
""",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("seed_cli.maintenance.subprocess.run", fake_run)

    plan = build_maintenance_plan(manifest)
    result = execute_maintenance_plan(plan, dry_run=False)

    assert result == {"checked": 1, "executed": 1, "skipped": 0}
    assert len(calls) == 1
    assert calls[0][0][0] == sys.executable
    assert calls[0][1]["cwd"] == str(repo_dir.resolve())


def test_execute_maintenance_plan_refuses_git_pull_with_dirty_worktree(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    manifest = tmp_path / "maintenance.yml"
    manifest.write_text(
        """
targets:
  - name: dirty-repo
    kind: repository
    path: ./repo
    goals:
      - git_pull_ff_only
""",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd == ["git", "status", "--short", "--untracked-files=normal"]:
            return SimpleNamespace(
                returncode=0,
                stdout=" M README.md\n?? tests/test_cli.py\n",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("seed_cli.maintenance.subprocess.run", fake_run)

    plan = build_maintenance_plan(manifest)
    with pytest.raises(RuntimeError, match="README.md"):
        execute_maintenance_plan(plan, dry_run=False)

    assert calls == [
        (
            ["git", "status", "--short", "--untracked-files=normal"],
            {
                "cwd": str(repo_dir.resolve()),
                "capture_output": True,
                "text": True,
                "check": True,
            },
        )
    ]


def test_execute_maintenance_plan_runs_git_pull_when_worktree_is_clean(
    tmp_path, monkeypatch
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    manifest = tmp_path / "maintenance.yml"
    manifest.write_text(
        """
targets:
  - name: clean-repo
    kind: repository
    path: ./repo
    goals:
      - git_pull_ff_only
""",
        encoding="utf-8",
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd == ["git", "status", "--short", "--untracked-files=normal"]:
            return SimpleNamespace(returncode=0, stdout="")
        if cmd == ["git", "pull", "--ff-only"]:
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("seed_cli.maintenance.subprocess.run", fake_run)

    plan = build_maintenance_plan(manifest)
    result = execute_maintenance_plan(plan, dry_run=False)

    assert result == {"checked": 0, "executed": 1, "skipped": 0}
    assert [call[0] for call in calls] == [
        ["git", "status", "--short", "--untracked-files=normal"],
        ["git", "pull", "--ff-only"],
    ]
