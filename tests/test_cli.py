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


def test_cli_version(tmp_path):
    code, out, err = run(["--version"], tmp_path)
    assert code == 0
    assert out.strip() == f"seed {project_version()}"


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


def test_parse_spec_file_registers_project_template(tmp_path):
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
    assert registered.exists()
    assert registered.read_text() == spec.read_text()


def test_cli_create_with_project_template_from_nested_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    spec = tmp_path / "spec.tree"
    spec.write_text(
        "features/\n"
        "├── <name>/\n"
        "│   └── api/\n"
        "│       └── route.ts\n"
    )

    code, out, err = run(["plan", "spec.tree"], tmp_path)
    assert code == 0
    assert (tmp_path / ".seed" / "templates" / "spec.tree").exists()

    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)

    code, out, err = run(
        ["create", "--template", ".seed/templates/spec.tree", "name=users"],
        nested,
    )

    assert code == 0
    assert (nested / "users" / "api").is_dir()
    assert (nested / "users" / "api" / "route.ts").exists()


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
