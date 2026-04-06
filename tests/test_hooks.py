from pathlib import Path
from seed_cli.hooks import run_hooks, pre_apply, post_apply, pre_step, post_step, HookError, load_filesystem_hooks


class H:
    def __init__(self):
        self.calls = []

    def pre_apply(self, plan, base):
        self.calls.append("pre_apply")


class Bad:
    def pre_apply(self, plan, base):
        raise RuntimeError("boom")


def test_basic_hook_call():
    h = H()
    pre_apply([h], plan=None, base=None)
    assert "pre_apply" in h.calls


def test_hook_error_collected():
    h = H()
    b = Bad()
    errs = []
    hooks = load_filesystem_hooks(Path("hooks"))
    run_hooks([h, b], "pre_apply", None, None, errors=errs)
    assert len(errs) == 1
    assert isinstance(errs[0].error, RuntimeError)


def test_hook_strict_raises():
    b = Bad()
    try:
        run_hooks([b], "pre_apply", None, None, strict=True)
        assert False, "expected error"
    except RuntimeError:
        pass


def test_filesystem_hooks_use_distinct_scripts(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    out = tmp_path / "out.txt"

    a = hooks_dir / "a.sh"
    b = hooks_dir / "b.sh"
    a.write_text(f"#!/bin/sh\necho a >> '{out}'\n")
    b.write_text(f"#!/bin/sh\necho b >> '{out}'\n")
    a.chmod(0o755)
    b.chmod(0o755)

    hooks = load_filesystem_hooks(hooks_dir)
    run_hooks(hooks, "pre_apply", cwd=tmp_path, strict=True)

    lines = out.read_text().strip().splitlines()
    assert lines == ["a", "b"]


def test_load_filesystem_hooks_missing_dir(tmp_path):
    hooks = load_filesystem_hooks(tmp_path / "missing")
    assert hooks == []


def test_python_hook_receives_stage_env(tmp_path):
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    out = tmp_path / "stage.txt"
    py = hooks_dir / "stage.py"
    py.write_text(
        "import os\n"
        f"open(r'{out}', 'w').write(os.environ.get('SEED_HOOK_STAGE', ''))\n"
    )
    py.chmod(0o755)

    hooks = load_filesystem_hooks(hooks_dir)
    run_hooks(hooks, "post_apply", cwd=tmp_path, strict=True)
    assert out.read_text() == "post_apply"


def test_hook_wrapper_helpers():
    h = H()
    pre_apply([h], None, None)
    post_apply([h], None, None, {})
    pre_step([h], None, None)
    post_step([h], None, None, "ok")
