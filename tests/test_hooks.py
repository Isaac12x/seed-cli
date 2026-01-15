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
