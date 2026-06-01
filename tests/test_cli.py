import json
import os
import subprocess
import sys
from pathlib import Path


def project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    in_project_section = False
    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[project]":
            in_project_section = True
            continue
        if in_project_section and line.startswith("["):
            break
        if in_project_section and line.startswith("version"):
            _, value = line.split("=", 1)
            return value.strip().strip("\"'")
    raise AssertionError("Could not find [project].version in pyproject.toml")


def run(cmd, cwd):
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    p = subprocess.run(
        [sys.executable, "-m", "seed_cli.cli"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    # Combine stdout and stderr for easier checking
    output = p.stdout + p.stderr
    return p.returncode, output, p.stderr


def write_template_registry(seed_home: Path, names: list[str]) -> None:
    templates_dir = seed_home / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        name: {
            "name": name,
            "source": "test",
            "current_version": "v1",
            "locked": False,
            "created_at": 0,
            "versions": ["v1"],
            "content_url": None,
        }
        for name in names
    }
    (templates_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def test_cli_plan(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("a/file.txt")
    code, out, err = run(["plan", "spec.tree"], tmp_path)
    assert code == 0
    assert "Plan:" in out


def test_cli_diff(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("a/file.txt")
    code, out, err = run(["diff", "spec.tree"], tmp_path)
    assert code == 1
    assert "Missing" in out or "missing" in out.lower()


def test_cli_apply(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("x.txt")
    code, out, err = run(["apply", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "x.txt").exists()


def test_cli_apply_seed_spec(tmp_path):
    spec = tmp_path / "spec.seed"
    spec.write_text("vendor/ !service\nvendor/README.md")
    code, out, err = run(["apply", "spec.seed"], tmp_path)
    assert code == 0
    assert (tmp_path / "vendor").is_dir()
    assert (tmp_path / "vendor" / "README.md").exists()


def test_cli_doctor(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("a.txt\na.txt")
    code, out, err = run(["doctor", "spec.tree"], tmp_path)
    assert code == 1
    assert "duplicate" in out


def test_cli_no_command(tmp_path):
    code, out, err = run([], tmp_path)
    assert code == 1
    assert "no command provided" in out
    assert "Available commands" in out
    assert "Plan & Apply:" in out


def test_cli_no_command_applies_single_seed_spec_and_moves_files(tmp_path):
    spec = tmp_path / "Project.seed"
    spec.write_text("src/app.py\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")

    code, out, err = run([], tmp_path)

    assert code == 0
    assert "Auto-applying spec: Project.seed" in out
    assert not (tmp_path / "app.py").exists()
    assert (
        (tmp_path / "src" / "app.py").read_text(encoding="utf-8")
        == "print('hello')\n"
    )


def test_cli_no_command_errors_when_multiple_default_specs(tmp_path):
    (tmp_path / "a.tree").write_text("a.txt", encoding="utf-8")
    (tmp_path / "b.seed").write_text("b.txt", encoding="utf-8")

    code, out, err = run([], tmp_path)

    assert code == 1
    assert "multiple default specs found" in out


def test_cli_version(tmp_path):
    code, out, err = run(["--version"], tmp_path)
    assert code == 0
    assert out.strip() == f"seed {project_version()}"


def test_cli_help_groups_top_level_commands(tmp_path):
    code, out, err = run(["--help"], tmp_path)

    assert code == 0
    for heading in [
        "Plan & Apply:",
        "Templates:",
        "State & History:",
        "Maintenance:",
        "Export & Utilities:",
    ]:
        assert heading in out

    assert "templates  Manage reusable templates. (alias: template)" in out
    assert "template   Manage reusable templates." not in out
    assert out.index("Plan & Apply:") < out.index("Templates:")
    assert out.index("Templates:") < out.index("State & History:")


def test_cli_template_help_groups_subcommands(tmp_path):
    code, out, err = run(["templates", "--help"], tmp_path)

    assert code == 0
    assert "Browse:" in out
    assert "Apply:" in out
    assert "Manage:" in out
    assert out.index("Browse:") < out.index("Apply:")
    assert out.index("Apply:") < out.index("Manage:")


def test_cli_capture(tmp_path):
    (tmp_path / "test.txt").write_text("content")
    code, out, err = run(["capture"], tmp_path)
    assert code == 0
    assert "test.txt" in out


def test_cli_capture_json(tmp_path):
    (tmp_path / "test.txt").write_text("content")
    code, out, err = run(["capture", "--json"], tmp_path)
    assert code == 0
    assert "entries" in out
    assert "test.txt" in out


def test_cli_capture_out(tmp_path):
    (tmp_path / "test.txt").write_text("content")
    code, out, err = run(["capture", "--out", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "spec.tree").exists()
    assert "test.txt" in (tmp_path / "spec.tree").read_text()


def test_cli_specs_watch_parser():
    from seed_cli.cli import build_parser

    args = build_parser().parse_args(["specs", "watch", "--interval", "0.25"])

    assert args.cmd == "specs"
    assert args.specs_action == "watch"
    assert args.interval == 0.25


def test_cli_export_tree(tmp_path):
    (tmp_path / "test.txt").write_text("content")
    code, out, err = run(["export", "tree", "--out", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "spec.tree").exists()


def test_cli_export_with_input(tmp_path):
    spec = tmp_path / "input.tree"
    spec.write_text("a/file.txt")
    code, out, err = run(["export", "tree", "--input", "input.tree", "--out", "output.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "output.tree").exists()
    assert "a/file.txt" in (tmp_path / "output.tree").read_text()


def test_cli_lock_status(tmp_path):
    code, out, err = run(["lock", "status"], tmp_path)
    assert code == 0
    assert "No structure lock active" in out


def test_cli_state_lock_no_lock(tmp_path):
    code, out, err = run(["utils", "state-lock"], tmp_path)
    assert code == 0
    assert "No execution lock found" in out


def test_cli_state_lock_renew_no_lock(tmp_path):
    code, out, err = run(["utils", "state-lock", "--renew"], tmp_path)
    # Should handle gracefully or show error
    assert code in (0, 1)


def test_cli_hooks_install(tmp_path):
    import subprocess
    # Create a fake .git directory
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    code, out, err = run(["hooks", "install"], tmp_path)
    assert code == 0
    assert "pre-commit" in out or "Installed git hook" in out


def test_cli_sync_dry_run_no_dangerous(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("x.txt")
    code, out, err = run(["sync", "spec.tree", "--dry-run"], tmp_path)
    # Should work without --dangerous in dry-run mode
    assert code == 0


def test_cli_maintain_dry_run(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

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
""",
        encoding="utf-8",
    )

    code, out, err = run(["maintain", "maintenance.yml"], tmp_path)

    assert code == 0
    assert "DRY RUN" in out
    assert "Maintenance plan:" in out
    assert "git status --short --branch" in out


def test_cli_diff_type_mismatch(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("a/")
    (tmp_path / "a").write_text("file")  # Create as file instead of dir
    code, out, err = run(["diff", "spec.tree"], tmp_path)
    assert code == 1
    # Should show type mismatch
    assert "Type Mismatch" in out or "type_mismatch" in out.lower()


def test_cli_apply_plan_delete_with_dangerous(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("x")

    plan = tmp_path / "plan.json"
    plan.write_text(
        (
            '{"summary":{"add":0,"change":0,"delete":1,"delete_skipped":0},'
            '"steps":[{"op":"delete","path":"victim.txt","reason":"test",'
            '"annotation":null,"depends_on":null,"note":null,"optional":false}]}'
        )
    )

    code, out, err = run(["apply", "plan.json", "--dangerous"], tmp_path)
    assert code == 0
    assert not victim.exists()


def test_parse_spec_file_does_not_register_project_template(tmp_path):
    from seed_cli.cli import parse_spec_file

    (tmp_path / ".git").mkdir()
    spec = tmp_path / "spec.tree"
    spec.write_text(
        "features/\n"
        "├── <name>/\n"
        "│   └── api/\n"
        "│       └── route.ts\n"
    )

    parse_spec_file(str(spec), {}, tmp_path, [], {"base": tmp_path, "plugins": [], "cmd": "plan"})

    registered = tmp_path / ".seed" / "templates" / "spec.tree"
    assert not registered.exists()


def test_cli_register_creates_project_template_support_files(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )

    code, out, err = run(["register", "spec.tree"], tmp_path)

    assert code == 0
    assert (tmp_path / ".seed" / "templates" / "spec.tree").exists()
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree").exists()
    assert "Registered spec:" in out


def test_cli_register_mirrors_plain_tree_spec_without_project_templates(tmp_path):
    spec = tmp_path / "plain.tree"
    spec.write_text("src/\n└── main.py\n")

    code, out, err = run(["register", "plain.tree"], tmp_path)

    assert code == 0
    assert (tmp_path / ".seed" / "templates" / "plain.tree").exists()
    assert "Registered spec:" in out
    assert "Registered project template:" not in out


def test_cli_create_with_project_template_from_nested_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    spec = tmp_path / "spec.tree"
    spec.write_text(
        "features/\n"
        "├── <name>/\n"
        "│   └── api/\n"
        "│       └── route.ts\n"
    )

    code, out, err = run(["register", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / ".seed" / "templates" / "spec.tree").exists()

    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)

    code, out, err = run(
        ["create", "--template", str((tmp_path / ".seed" / "templates" / "spec.tree").resolve()), "name=users"],
        nested,
    )

    assert code == 0
    assert (nested / "users" / "api").is_dir()
    assert (nested / "users" / "api" / "route.ts").exists()


def test_cli_create_rejects_relative_template_flag_path(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("<name>/\n└── route.ts\n")

    code, out, err = run(["create", "--template", "spec.tree", "users"], tmp_path)

    assert code == 1
    assert "--template must be a full path" in out


def test_cli_apply_cleans_stale_materialized_project_template_dir(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )
    stale_dir = tmp_path / "features" / "<name>" / "api"
    stale_dir.mkdir(parents=True)
    (stale_dir / "route.ts").write_text("legacy")

    code, out, err = run(["apply", "spec.tree"], tmp_path)

    assert code == 0
    assert not (tmp_path / "features" / "<name>").exists()
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree").exists()


def test_cli_create_with_registered_project_template(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "└── features/\n"
        "    └── <name>/\n"
        "        └── api/\n"
        "            └── route.ts\n"
    )

    code, out, err = run(["apply", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree").exists()
    assert not (tmp_path / "features" / "<name>").exists()

    features_dir = tmp_path / "features"
    code, out, err = run(["create", "--project", "users"], features_dir)

    assert code == 0
    assert (features_dir / "users" / "api").is_dir()
    assert (features_dir / "users" / "api" / "route.ts").exists()


def test_cli_create_with_registered_project_template_replaces_multiple_placeholders(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text(
        ".\n"
        "└── features/\n"
        "    └── <domain>/\n"
        "        └── <name>/\n"
        "            └── route.ts\n"
    )

    code, out, err = run(["apply", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "domain.tree").exists()

    code, out, err = run(["create", "--project", "domain", "domain=billing", "name=invoices"], tmp_path / "features")

    assert code == 0
    assert (tmp_path / "features" / "billing" / "invoices" / "route.ts").exists()
    assert not (tmp_path / "features" / "billing" / "<name>").exists()


def test_cli_create_with_registered_placeholder_filename_template(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("features/<name>.ts\n")

    code, out, err = run(["apply", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / "features" / ".seed" / "templates" / "project" / "name.tree").exists()

    code, out, err = run(["create", "--project", "name", "name=users"], tmp_path / "features")

    assert code == 0
    assert (tmp_path / "features" / "users.ts").exists()


def test_cli_create_finds_project_template_without_flag(tmp_path):
    template_dir = tmp_path / ".seed" / "templates" / "project"
    template_dir.mkdir(parents=True)
    (template_dir / "project.tree").write_text(
        ".\n"
        "└── <name>/\n"
        "    └── api/\n"
        "        └── route.ts\n"
    )

    code, out, err = run(["create", "project", "test"], tmp_path)

    assert code == 0
    assert (tmp_path / "test" / "api").is_dir()
    assert (tmp_path / "test" / "api" / "route.ts").exists()


def test_cli_templates_list_shows_project_templates_first(tmp_path, monkeypatch):
    monkeypatch.setenv("SEED_HOME", str(tmp_path / "seed-home"))
    write_template_registry(
        tmp_path / "seed-home",
        ["fastapi", "python-package", "node-typescript", "ralph", "stored"],
    )

    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / ".git").mkdir()
    template_dir = project_root / ".seed" / "templates" / "project"
    template_dir.mkdir(parents=True)
    (template_dir / "component.tree").write_text("<name>/\n└── route.ts\n")

    code, out, err = run(["templates", "list"], project_root)

    assert code == 0
    assert "Project templates:" in out
    assert "  component" in out
    assert out.index("Project templates:") < out.index("Stored templates:")


def test_cli_templates_use_discovers_project_template_and_uses_folder(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / ".git").mkdir()
    template_dir = project_root / ".seed" / "templates" / "project"
    template_dir.mkdir(parents=True)
    (template_dir / "component.tree").write_text(
        ".\n"
        "└── <name>/\n"
        "    └── api/\n"
        "        └── route.ts\n"
    )

    code, out, err = run(["templates", "use", "component", "users"], project_root)

    assert code == 0
    assert (project_root / "users" / "api" / "route.ts").exists()


def test_cli_template_use_registry_template_uses_folder_argument(tmp_path, monkeypatch):
    seed_home = tmp_path / "seed-home"
    monkeypatch.setenv("SEED_HOME", str(seed_home))
    write_template_registry(
        seed_home,
        ["fastapi", "python-package", "node-typescript", "ralph", "component"],
    )
    template_dir = seed_home / "templates" / "component"
    template_dir.mkdir(parents=True)
    (template_dir / "v1.tree").write_text(
        ".\n"
        "└── <name>/\n"
        "    └── api/\n"
        "        └── route.ts\n"
    )

    code, out, err = run(["template", "use", "component", "users", "--yes"], tmp_path)

    assert code == 0
    assert (tmp_path / "users" / "api" / "route.ts").exists()
