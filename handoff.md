# Handoff - 2026-06-29

Implemented the PRD-aligned agent-first filesystem state surfaces on branch
`feature/codex/prd-agent-filesystem-state`.

## Changed

- Added `seed init`, `seed import`, `seed check`, `seed validate`, `seed repair`,
  `seed graph`, and `seed agents`.
- Added JSON output for `plan`, `check`, `diff`, `doctor`, `apply`, and `sync`.
- Added explicit pruning spelling with `seed apply --prune` and
  `seed sync --prune`.
- Added `src/seed_cli/agent.py` and `docs/agents.md` for framework adoption by
  OpenHands, Codex, Claude Code, Aider, Cline, Roo Code, Continue, and Goose.
- Added `seed agents --framework <name>` for framework-specific integration
  packs with maintainer pitch, default instruction text, commands, and safety
  notes.
- Added `seed mcp`, a dependency-light stdio MCP server exposing `seed_plan`,
  `seed_check`, and `seed_apply` for frameworks that support MCP.
- Added `seed agents --framework <name> --format proposal` to generate
  upstream-ready issue or PR text for framework maintainers.
- Updated `README.md`, `CHANGELOG.md`, and the Superpowers implementation plan.

## Verification

- `uv run pytest -o addopts='' tests/test_agent.py tests/test_cli.py::test_cli_plan_json_reports_drift_with_exit_code_2 tests/test_cli.py::test_cli_check_json_uses_drift_exit_code tests/test_cli.py::test_cli_init_writes_default_filesystem_tree tests/test_cli.py::test_cli_import_alias_writes_tree_spec tests/test_cli.py::test_cli_validate_and_repair_aliases_doctor tests/test_cli.py::test_cli_graph_outputs_mermaid tests/test_cli.py::test_cli_apply_prune_deletes_extras_explicitly tests/test_cli.py::test_cli_sync_prune_is_safe_deletion_spelling tests/test_cli.py::test_cli_agents_json_manifest -q`

Result: `11 passed`.

- `uv run pytest -o addopts='' tests/test_cli.py tests/test_agent.py tests/test_capture.py tests/test_doctor.py tests/test_graphviz.py -q`

Result: `66 passed`.

- `uv run pytest -q`

Result: `661 passed`, coverage `85.41%`.

- `uv run pytest -o addopts='' tests/test_agent.py::test_agent_integration_returns_framework_specific_pack tests/test_cli.py::test_cli_agents_framework_json_pack -q`

Result: `2 passed`.

- `uv run pytest -o addopts='' tests/test_cli.py::test_cli_help_groups_top_level_commands tests/test_cli.py::test_build_parser_knows_agent_first_commands tests/test_mcp.py -q`

Result: `6 passed`.

- `uv run pytest -o addopts='' tests/test_agent.py::test_agent_proposal_markdown_is_upstream_ready tests/test_cli.py::test_cli_agents_framework_proposal_format -q`

Result: `2 passed`.

## Notes

- A targeted `uv run pytest tests/test_cli.py -q` before changes passed all CLI
  tests but failed the repository coverage gate because it ran only one file.
- Full-suite verification passed locally after the argparse compatibility patch.
