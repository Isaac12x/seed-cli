from seed_cli.plugins.base import SeedPlugin


def test_seed_plugin_default_hooks_are_noops():
    plugin = SeedPlugin()
    context = {"target": "demo"}
    nodes = ["node"]
    plan = ["step"]

    assert plugin.before_parse("spec", context) == "spec"
    assert plugin.after_parse(nodes, context) is None
    assert plugin.before_plan(nodes, context) is None
    assert plugin.after_plan(plan, context) is None
    assert plugin.before_build(plan, context) is None
    assert plugin.after_build(context) is None
    assert plugin.before_sync_delete("README.md", context) is True
