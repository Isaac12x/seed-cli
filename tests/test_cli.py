import subprocess
import sys
from pathlib import Path

def run(cmd, cwd):
    p = subprocess.run(
        [sys.executable, "-m", "seed_cli.cli"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
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


def test_cli_diff_type_mismatch(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("a/")
    (tmp_path / "a").write_text("file")  # Create as file instead of dir
    code, out, err = run(["diff", "spec.tree"], tmp_path)
    assert code == 1
    # Should show type mismatch
    assert "Type Mismatch" in out or "type_mismatch" in out.lower()
