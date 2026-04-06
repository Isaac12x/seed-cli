import sys

from seed_cli.plugins.loader import load_plugins
from seed_cli.plugins.base import SeedPlugin


def test_load_plugins_from_module(tmp_path):
    # Create a fake plugin module
    mod = tmp_path / "myplugin.py"
    mod.write_text(
        """
from seed_cli.plugins.base import SeedPlugin

class MyPlugin(SeedPlugin):
    name = "example"
"""
    )

    sys.path.insert(0, str(tmp_path))
    try:
        plugins = load_plugins(modules=["myplugin"])
        assert len(plugins) == 1
        assert plugins[0].name == "example"
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_plugin_load_failure_isolated(monkeypatch):
    plugins = load_plugins(modules=["nonexistent.module"])
    assert plugins == []
