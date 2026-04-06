import json
from pathlib import Path

import pytest

from seed_cli.scaffold import (
    load_data_file,
    load_template_config,
    resolve_template_vars,
    template_tasks,
    template_exclude,
    template_skip_if_exists,
    template_answers_file,
    write_answers,
    run_template_tasks,
)


def test_load_data_file_yaml(tmp_path):
    p = tmp_path / "vars.yml"
    p.write_text("name: demo\ncount: 3\n")
    data = load_data_file(str(p))
    assert data["name"] == "demo"
    assert data["count"] == 3


def test_load_data_file_json(tmp_path):
    p = tmp_path / "vars.json"
    p.write_text(json.dumps({"name": "demo", "flag": True}))
    data = load_data_file(str(p))
    assert data["flag"] is True


def test_load_data_file_none_returns_empty():
    assert load_data_file(None) == {}


def test_load_data_file_empty_doc(tmp_path):
    p = tmp_path / "vars.yml"
    p.write_text("")
    assert load_data_file(str(p)) == {}


def test_load_data_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data_file(str(tmp_path / "missing.yml"))


def test_load_data_file_requires_mapping(tmp_path):
    p = tmp_path / "vars.yml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="top-level mapping"):
        load_data_file(str(p))


def test_load_template_config_valid(tmp_path):
    p = tmp_path / "copier.yml"
    p.write_text("_tasks:\n  - echo hi\n")
    cfg = load_template_config(p)
    assert cfg["_tasks"] == ["echo hi"]


def test_load_template_config_missing_and_empty(tmp_path):
    assert load_template_config(None) == {}
    assert load_template_config(tmp_path / "missing.yml") == {}

    p = tmp_path / "empty.yml"
    p.write_text("")
    assert load_template_config(p) == {}


def test_load_template_config_invalid_format(tmp_path):
    p = tmp_path / "copier.yml"
    p.write_text("- not-a-dict\n")
    with pytest.raises(ValueError, match="Invalid template config format"):
        load_template_config(p)


def test_resolve_template_vars_defaults_non_interactive():
    config = {
        "project_name": {"type": "str", "default": "demo"},
        "port": {"type": "int", "default": 8080},
        "enabled": {"type": "bool", "default": True},
    }
    vars_dict = resolve_template_vars(
        config=config,
        cli_vars={"project_name": "acme"},
        data_vars={},
        defaults=True,
        non_interactive=True,
    )
    assert vars_dict["project_name"] == "acme"
    assert vars_dict["port"] == 8080
    assert vars_dict["enabled"] is True


def test_resolve_template_vars_choices_and_required():
    config = {
        "flavor": {"type": "str", "choices": ["vanilla", "choco"], "required": True},
    }
    with pytest.raises(ValueError, match="Missing required template variable"):
        resolve_template_vars(
            config=config,
            cli_vars={},
            data_vars={},
            defaults=False,
            non_interactive=True,
        )
    with pytest.raises(ValueError, match="Invalid choice"):
        resolve_template_vars(
            config=config,
            cli_vars={"flavor": "mint"},
            data_vars={},
            defaults=True,
            non_interactive=True,
        )


def test_resolve_template_vars_interactive(monkeypatch):
    config = {"name": {"type": "str", "help": "Name", "required": True}}
    monkeypatch.setattr("builtins.input", lambda prompt: "demo")
    out = resolve_template_vars(
        config=config,
        cli_vars={},
        data_vars={},
        defaults=False,
        non_interactive=False,
    )
    assert out["name"] == "demo"


def test_resolve_template_vars_interactive_default_blank(monkeypatch):
    config = {"port": {"type": "int", "default": 1234}}
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    out = resolve_template_vars(
        config=config,
        cli_vars={},
        data_vars={},
        defaults=False,
        non_interactive=False,
    )
    assert out["port"] == 1234


def test_resolve_template_vars_explicit_question_non_dict():
    config = {"questions": {"name": "demo"}}
    out = resolve_template_vars(
        config=config,
        cli_vars={},
        data_vars={},
        defaults=True,
        non_interactive=True,
    )
    assert out["name"] == "demo"


def test_resolve_template_vars_coercion_edges():
    with pytest.raises(ValueError, match="Invalid boolean value"):
        resolve_template_vars(
            config={"enabled": {"type": "bool"}},
            cli_vars={"enabled": "maybe"},
            data_vars={},
            defaults=True,
            non_interactive=True,
        )

    out = resolve_template_vars(
        config={"ratio": {"type": "float", "default": "1.5"}, "raw": {"type": "custom", "default": 9}},
        cli_vars={},
        data_vars={},
        defaults=True,
        non_interactive=True,
    )
    assert out["ratio"] == 1.5
    assert out["raw"] == 9


def test_resolve_template_vars_questions_section():
    config = {
        "questions": {
            "count": {"type": "int", "default": 2},
            "enabled": {"type": "bool", "default": "yes"},
        }
    }
    out = resolve_template_vars(
        config=config,
        cli_vars={},
        data_vars={"count": "4"},
        defaults=True,
        non_interactive=True,
    )
    assert out["count"] == 4
    assert out["enabled"] is True


def test_template_tasks_and_config_helpers():
    config = {
        "_tasks": ["echo hello", {"command": "echo world"}],
        "_exclude": ["**/.DS_Store"],
        "_skip_if_exists": ["README.md"],
        "_answers_file": ".seed/answers.yml",
    }
    assert template_tasks(config) == ["echo hello", "echo world"]
    assert template_exclude(config) == ["**/.DS_Store"]
    assert template_skip_if_exists(config) == ["README.md"]
    assert template_answers_file(config, None) == ".seed/answers.yml"
    assert template_answers_file(config, "answers.yml") == "answers.yml"

    assert template_tasks({"tasks": ["echo x"]}) == ["echo x"]
    assert template_tasks({"_tasks": None}) == []
    assert template_exclude({}) == []
    assert template_skip_if_exists({}) == []
    assert template_answers_file({}, None) is None


def test_template_helpers_invalid_values():
    with pytest.raises(ValueError, match="must be a list"):
        template_tasks({"_tasks": "echo hi"})
    with pytest.raises(ValueError, match="Invalid task entry"):
        template_tasks({"_tasks": [123]})
    with pytest.raises(ValueError, match="must be a list"):
        template_exclude({"_exclude": "x"})
    with pytest.raises(ValueError, match="must be a list"):
        template_skip_if_exists({"_skip_if_exists": "x"})


def test_write_answers_and_run_tasks(tmp_path):
    answers = tmp_path / "answers.yml"
    write_answers(answers, {"project_name": "demo"})
    assert "project_name: demo" in answers.read_text()

    tasks = ["printf 'hello {{project_name}}' > task.txt"]
    executed = run_template_tasks(
        tasks,
        vars_dict={"project_name": "demo"},
        cwd=tmp_path,
        unsafe=True,
    )
    assert executed
    assert (tmp_path / "task.txt").read_text() == "hello demo"


def test_run_tasks_skips_when_not_unsafe(tmp_path):
    executed = run_template_tasks(
        ["printf 'x' > nope.txt"],
        vars_dict={"name": "demo"},
        cwd=tmp_path,
        unsafe=False,
    )
    assert executed == []
    assert not (tmp_path / "nope.txt").exists()


def test_run_tasks_with_empty_task_list(tmp_path):
    assert run_template_tasks([], vars_dict={}, cwd=tmp_path, unsafe=True) == []
