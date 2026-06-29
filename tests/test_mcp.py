import json
from io import StringIO
from pathlib import Path

from seed_cli.mcp import handle_message, run_stdio


def test_mcp_initialize_reports_tools_capability():
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "seed-cli"
    assert response["result"]["capabilities"]["tools"] == {}


def test_mcp_tools_list_includes_seed_plan_check_apply():
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )

    tool_names = {tool["name"] for tool in response["result"]["tools"]}
    assert {"seed_plan", "seed_check", "seed_apply"}.issubset(tool_names)


def test_mcp_seed_plan_returns_structured_drift_payload(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text(".\n└── src/\n    └── app.py\n", encoding="utf-8")

    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "seed_plan",
                "arguments": {
                    "spec": "filesystem.tree",
                    "base": str(tmp_path),
                },
            },
        }
    )

    content = response["result"]["content"][0]
    payload = json.loads(content["text"])
    assert content["type"] == "text"
    assert payload["status"] == "drift"
    assert payload["create"] == ["src", "src/app.py"]
    assert payload["exitCode"] == 2


def test_mcp_seed_apply_converges_without_pruning(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text(".\n└── src/\n    └── app.py\n", encoding="utf-8")
    (tmp_path / "extra").mkdir()

    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "seed_apply",
                "arguments": {
                    "spec": "filesystem.tree",
                    "base": str(tmp_path),
                },
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["status"] == "applied"
    assert (tmp_path / "src" / "app.py").exists()
    assert (tmp_path / "extra").exists()


def test_mcp_seed_check_reports_match_after_apply(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text(".\n└── src/\n", encoding="utf-8")
    (tmp_path / "src").mkdir()

    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "seed_check",
                "arguments": {"spec": "filesystem.tree", "base": str(tmp_path)},
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["status"] == "match"
    assert payload["exitCode"] == 0


def test_mcp_seed_apply_prune_deletes_extras(tmp_path):
    spec = tmp_path / "filesystem.tree"
    spec.write_text(".\n", encoding="utf-8")
    (tmp_path / "extra").mkdir()

    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "seed_apply",
                "arguments": {
                    "spec": "filesystem.tree",
                    "base": str(tmp_path),
                    "prune": True,
                },
            },
        }
    )

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["deleted"] == 1
    assert not (tmp_path / "extra").exists()


def test_mcp_unknown_method_and_tool_return_errors():
    method_response = handle_message(
        {"jsonrpc": "2.0", "id": 7, "method": "unknown", "params": {}}
    )
    tool_response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert method_response["error"]["code"] == -32601
    assert tool_response["error"]["code"] == -32602


def test_mcp_notifications_do_not_emit_response():
    assert (
        handle_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        is None
    )


def test_mcp_stdio_handles_json_lines_and_parse_errors():
    input_stream = StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list"}) + "\n"
        "{bad json\n"
    )
    output_stream = StringIO()

    code = run_stdio(input_stream, output_stream)

    lines = output_stream.getvalue().splitlines()
    assert code == 0
    assert json.loads(lines[0])["result"]["tools"]
    assert json.loads(lines[1])["error"]["code"] == -32700
