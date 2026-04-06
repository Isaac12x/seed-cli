from pathlib import Path
from seed_cli.sync import sync


class _VetoDeletePlugin:
    def before_build(self, plan, context):
        return None

    def after_build(self, context):
        return None

    def before_sync_delete(self, relpath, context):
        return False


def test_sync_deletes_extras(tmp_path):
    # filesystem has extra file
    (tmp_path / "extra.txt").write_text("x")

    spec = tmp_path / "spec.tree"
    spec.write_text("")

    res = sync(str(spec), tmp_path, dangerous=True, ignore=["spec.tree"])
    assert not (tmp_path / "extra.txt").exists()
    # May delete spec.tree too if not ignored, so check >= 1
    assert res["deleted"] >= 1


def test_sync_requires_dangerous(tmp_path):
    spec = tmp_path / "spec.tree"
    spec.write_text("")
    try:
        sync(str(spec), tmp_path, dangerous=False)
        assert False, "sync without dangerous should fail"
    except RuntimeError:
        pass


def test_sync_targets(tmp_path):
    # extras in different dirs
    (tmp_path / "a").mkdir()
    (tmp_path / "a/x.txt").write_text("x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b/y.txt").write_text("y")

    spec = tmp_path / "spec.tree"
    spec.write_text("")

    # only sync target a
    res = sync(str(spec), tmp_path, dangerous=True, targets=["a"])
    assert not (tmp_path / "a/x.txt").exists()
    assert (tmp_path / "b/y.txt").exists()


def test_sync_plugin_can_veto_deletion(tmp_path):
    (tmp_path / "extra.txt").write_text("x")
    spec = tmp_path / "spec.tree"
    spec.write_text("")

    res = sync(
        str(spec),
        tmp_path,
        dangerous=True,
        ignore=["spec.tree"],
        plugins=[_VetoDeletePlugin()],
    )
    assert (tmp_path / "extra.txt").exists()
    assert res["deleted"] == 0
    assert res["skipped"] >= 1
