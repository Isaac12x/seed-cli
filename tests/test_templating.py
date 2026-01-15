import pytest
from seed_cli.templating import apply_vars, TemplateError


def test_basic_substitution():
    text = "{{name}}/file.txt"
    out = apply_vars(text, {"name": "demo"})
    assert out == "demo/file.txt"


def test_default_value():
    text = "{{name|fallback}}/file.txt"
    out = apply_vars(text, {})
    assert out == "fallback/file.txt"


def test_missing_strict():
    with pytest.raises(TemplateError):
        apply_vars("{{x}}", {})


def test_missing_loose():
    out = apply_vars("{{x}}", {}, mode="loose")
    assert out == "{{x}}"


def test_multiple_vars():
    text = "{{a}}/{{b|x}}"
    out = apply_vars(text, {"a": "1"})
    assert out == "1/x"
