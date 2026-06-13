from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from seed_cli.gbrain import gbrain_cli


def test_find_binary_prefers_existing_environment_override(tmp_path, monkeypatch):
    binary = tmp_path / "gbrain"
    binary.write_text("")
    monkeypatch.setenv(gbrain_cli.GBRAIN_BIN_ENV, str(binary))
    monkeypatch.setattr(gbrain_cli.shutil, "which", lambda name: "/ignored/gbrain")

    assert gbrain_cli.find_binary() == str(binary)


def test_find_binary_rejects_missing_override_and_uses_path_without_one(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(gbrain_cli.GBRAIN_BIN_ENV, str(tmp_path / "missing"))
    assert gbrain_cli.find_binary() is None

    monkeypatch.delenv(gbrain_cli.GBRAIN_BIN_ENV)
    monkeypatch.setattr(gbrain_cli.shutil, "which", lambda name: "/usr/local/bin/gbrain")
    assert gbrain_cli.find_binary() == "/usr/local/bin/gbrain"


def test_run_reports_missing_binary(monkeypatch):
    monkeypatch.setattr(gbrain_cli, "find_binary", lambda: None)

    result = gbrain_cli._run(["schema", "active"])

    assert result.ok is False
    assert result.stderr == "gbrain binary not found on PATH"


def test_run_captures_process_result_and_options(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 3, stdout="output\n", stderr="problem\n")

    monkeypatch.setattr(gbrain_cli, "find_binary", lambda: "/usr/bin/gbrain")
    monkeypatch.setattr(gbrain_cli.subprocess, "run", fake_run)

    result = gbrain_cli._run(["sync"], cwd=tmp_path, timeout=12.5)

    assert result == gbrain_cli.GbrainResult(
        ok=False,
        stdout="output\n",
        stderr="problem\n",
        returncode=3,
    )
    assert captured["argv"] == ["/usr/bin/gbrain", "sync"]
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert captured["kwargs"]["timeout"] == 12.5
    assert captured["kwargs"]["capture_output"] is True


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["gbrain"], 1),
        FileNotFoundError("missing"),
        PermissionError("denied"),
    ],
)
def test_run_converts_invocation_errors_to_results(error, monkeypatch):
    monkeypatch.setattr(gbrain_cli, "find_binary", lambda: "/usr/bin/gbrain")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(gbrain_cli.subprocess, "run", fail)

    result = gbrain_cli._run(["--version"])

    assert result.ok is False
    assert result.stderr.startswith("failed to invoke gbrain:")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (gbrain_cli.GbrainResult(ok=False), None),
        (gbrain_cli.GbrainResult(ok=True, stdout=""), None),
        (gbrain_cli.GbrainResult(ok=True, stdout="gbrain 2.4.1\n"), "2.4.1"),
    ],
)
def test_version_parses_successful_output(result, expected, monkeypatch):
    monkeypatch.setattr(gbrain_cli, "_run", lambda argv: result)

    assert gbrain_cli.version() == expected


def test_schema_commands_delegate_to_run(monkeypatch):
    calls = []
    sentinel = gbrain_cli.GbrainResult(ok=True)

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return sentinel

    monkeypatch.setattr(gbrain_cli, "_run", fake_run)

    assert gbrain_cli.schema_validate("pack") is sentinel
    assert gbrain_cli.schema_use("pack") is sentinel
    assert gbrain_cli.schema_active() is sentinel
    assert gbrain_cli.schema_list() is sentinel
    assert calls == [
        (["schema", "validate", "pack"], {}),
        (["schema", "use", "pack"], {}),
        (["schema", "active"], {}),
        (["schema", "list"], {}),
    ]


def test_schema_show_json_handles_named_valid_and_invalid_responses(monkeypatch):
    results = iter(
        [
            gbrain_cli.GbrainResult(ok=True, stdout='{"name": "pack"}'),
            gbrain_cli.GbrainResult(ok=False, stdout='{"ignored": true}'),
            gbrain_cli.GbrainResult(ok=True, stdout="{invalid"),
        ]
    )
    calls = []

    def fake_run(argv):
        calls.append(argv)
        return next(results)

    monkeypatch.setattr(gbrain_cli, "_run", fake_run)

    assert gbrain_cli.schema_show_json("pack") == {"name": "pack"}
    assert gbrain_cli.schema_show_json() is None
    assert gbrain_cli.schema_show_json() is None
    assert calls == [
        ["schema", "show", "--json", "pack"],
        ["schema", "show", "--json"],
        ["schema", "show", "--json"],
    ]


def test_long_running_commands_build_expected_arguments(tmp_path, monkeypatch):
    calls = []
    sentinel = gbrain_cli.GbrainResult(ok=True)

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return sentinel

    monkeypatch.setattr(gbrain_cli, "_run", fake_run)

    assert gbrain_cli.submit_unify_types("pack", dry_run=True) is sentinel
    assert gbrain_cli.submit_unify_types("pack", dry_run=False) is sentinel
    assert gbrain_cli.sync(tmp_path) is sentinel
    assert gbrain_cli.list_pages(page_type="person") is sentinel
    assert gbrain_cli.list_pages() is sentinel

    params = json.dumps({"target_pack": "pack"})
    assert calls == [
        (
            [
                "jobs",
                "submit",
                "unify-types",
                "--allow-protected",
                "--params",
                params,
                "--dry-run",
            ],
            {"timeout": 60.0},
        ),
        (
            [
                "jobs",
                "submit",
                "unify-types",
                "--allow-protected",
                "--params",
                params,
            ],
            {"timeout": 60.0},
        ),
        (["sync"], {"cwd": tmp_path, "timeout": 120.0}),
        (["list", "--json", "--type", "person"], {"timeout": 60.0}),
        (["list", "--json"], {"timeout": 60.0}),
    ]


def test_is_available_reflects_binary_discovery(monkeypatch):
    monkeypatch.setattr(gbrain_cli, "find_binary", lambda: "/usr/bin/gbrain")
    assert gbrain_cli.is_available() is True

    monkeypatch.setattr(gbrain_cli, "find_binary", lambda: None)
    assert gbrain_cli.is_available() is False
