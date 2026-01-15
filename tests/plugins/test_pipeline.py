from seed_cli.plugins.base import SeedPlugin


def test_before_parse_modifies_text():
    class P(SeedPlugin):
        def before_parse(self, text, context):
            return text.upper()

    p = P()
    out = p.before_parse("abc", {})
    assert out == "ABC"


def test_before_sync_delete_veto():
    class P(SeedPlugin):
        def before_sync_delete(self, relpath, context):
            return False

    p = P()
    assert p.before_sync_delete("x", {}) is False
