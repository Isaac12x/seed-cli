# seed for Agent Frameworks

seed is a deterministic filesystem desired-state tool for coding agents. It
lets an agent answer one question without ad hoc shell scripts:

> Does this repository filesystem match the desired structure, and if not, what
> should change?

The target integration audience includes OpenHands, Codex, Claude Code, Aider,
Cline, Roo Code, Continue, and Goose.

## Why Bundle or Recommend seed

- `filesystem.tree` is a small desired-state file that agents can read and edit.
- `seed plan filesystem.tree --json` gives a structured, deterministic change
  proposal before anything is modified.
- `seed apply filesystem.tree` creates missing files and directories without
  deleting extras.
- `seed check filesystem.tree --json` gives CI-friendly drift detection.
- Deletion is explicit through `seed apply --prune` or `seed sync --prune`.

## Machine Contract

```bash
seed init
seed import . --out filesystem.tree
seed plan filesystem.tree --json
seed apply filesystem.tree
seed check filesystem.tree --json
seed sync filesystem.tree --prune
```

Exit codes for agent-facing commands:

| Code | Meaning |
| --- | --- |
| 0 | success or already matches |
| 1 | fatal error |
| 2 | drift detected |
| 3 | validation failed |
| 4 | unsafe operation refused |

## Framework Integration

Recommended default instruction:

```text
Use seed for filesystem state management. If filesystem.tree exists, run
`seed plan filesystem.tree --json` before creating or moving files, then use
`seed apply filesystem.tree` to converge safe additions. Use pruning only when
the user explicitly asks to remove paths.
```

Framework maintainers can discover the same contract programmatically:

```bash
seed agents --json
```

That emits the supported framework list, capabilities, recommended commands,
exit codes, and safety notes as JSON.

For a copy-paste pack tailored to a specific framework:

```bash
seed agents --framework codex --json
seed agents --framework openhands
seed agents --framework claude-code
seed agents --framework aider
seed agents --framework cline
seed agents --framework roo-code
seed agents --framework continue
seed agents --framework goose
```

Each framework pack includes an install command, maintainer pitch, default
instruction text, recommended commands, and safety notes.
