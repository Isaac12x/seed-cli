# Native File Manager Actions Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make `seed` available to non-technical users from native macOS Finder and Windows File Explorer right-click menus, using the OS picker/confirmation UI for paths and `seed-cli` for the actual filesystem operations.

**Architecture:** Add a cross-platform `seed desktop` command group that installs, uninstalls, and invokes native file-manager integrations. Keep Seed's core Python planning/apply/capture logic as the source of truth; the OS integrations should only collect selected paths/spec files and dispatch to `seed desktop run <action> ...`. For macOS, ship Finder Quick Actions backed by Automator-compatible shell scripts or Shortcuts-compatible scripts; for Windows, ship registry-backed Explorer context-menu verbs that call a small Python launcher and optionally PowerShell for UI prompts.

**Tech Stack:** Python 3.10+, argparse, existing `seed_cli.cli`, `pathlib`, `subprocess`, macOS Finder Quick Actions/Services, Windows Explorer registry context menus under HKCU, PowerShell dialogs/toasts where needed, pytest with platform-specific dry-run tests.

---

## Current State

- Package entrypoint is `seed = "seed_cli.cli:main"` in `pyproject.toml`.
- CLI is centralized in `src/seed_cli/cli.py` using `argparse`.
- Existing high-value commands for file-manager actions:
  - `seed capture --out project.tree`
  - `seed plan spec.tree --out plan.json`
  - `seed apply spec-or-plan --dry-run`
  - `seed apply spec-or-plan`
  - `seed diff spec.tree`
  - `seed sync spec.tree --dry-run`
  - `seed match spec.tree --dry-run`
  - `seed templates ...`
- `capture_nodes`, `to_tree_text`, `to_json`, and `to_dot` already support turning a selected folder into a spec representation in `src/seed_cli/capture.py`.
- Planning and execution are already separate; this is good for right-click UX because destructive actions can preview first.
- There is no current desktop/file-manager integration layer.
- There is no current GUI prompt abstraction for selecting a spec file, output path, or confirmation from a native file manager.

## Product Shape

### Right-click actions to expose first

Keep the first version small and safe:

1. **Seed: Capture folder as `.tree`**
   - Context: selected folder.
   - OS UI asks where to save the generated `.tree` file.
   - Runs: `seed desktop run capture --selection <folder> --out <chosen.tree>`.

2. **Seed: Preview plan from spec**
   - Context: selected `.tree`, `.yaml`, `.json`, `.dot`, `.png`, `.jpg`, or `.jpeg` spec file.
   - OS UI asks for target folder if not obvious.
   - Runs: `seed desktop run plan --spec <file> --base <folder>`.
   - Opens a readable text/HTML plan output for non-technical users.

3. **Seed: Apply spec to folder**
   - Context: selected spec file or selected folder containing a known spec.
   - Always performs dry-run preview first.
   - Requires native confirmation before actual write.
   - Runs: `seed desktop run apply --spec <file> --base <folder> --confirm`.

4. **Seed: Compare folder to spec**
   - Context: selected folder or spec file.
   - Runs: `seed desktop run diff --spec <file> --base <folder>`.

5. **Seed: Revert last Seed change**
   - Context: selected folder.
   - Runs: existing `seed revert --base <folder>` after native confirmation.

Defer `sync`, `match`, and `maintain --execute` from the first menu unless behind an advanced submenu because they are more destructive or developer-oriented.

## Recommended Implementation Path

### Task 1: Add a `desktop` command group skeleton

**Objective:** Add platform-neutral CLI entrypoints without installing anything yet.

**Files:**
- Modify: `src/seed_cli/cli.py`
- Create: `src/seed_cli/desktop/__init__.py`
- Create: `src/seed_cli/desktop/actions.py`
- Test: `tests/test_desktop_actions.py`

**CLI shape:**

```bash
seed desktop install
seed desktop uninstall
seed desktop status
seed desktop run capture --selection /path/to/folder --out /path/to/project.tree
seed desktop run plan --spec /path/to/spec.tree --base /path/to/project
seed desktop run diff --spec /path/to/spec.tree --base /path/to/project
seed desktop run apply --spec /path/to/spec.tree --base /path/to/project --confirm
```

**Implementation notes:**
- `install`, `uninstall`, and `status` dispatch by `platform.system()` to macOS or Windows modules.
- `run` should be platform-neutral and call Seed internals or existing command functions.
- Use `--dry-run` by default for write actions unless `--confirm` is passed.

**Verification:**

```bash
python -m pytest tests/test_desktop_actions.py -v
python -m seed_cli.cli desktop status
```

### Task 2: Add a platform-neutral action runner

**Objective:** Ensure native integrations call one stable internal Seed command instead of duplicating shell logic.

**Files:**
- Modify: `src/seed_cli/desktop/actions.py`
- Test: `tests/test_desktop_actions.py`

**Behavior:**
- `capture(selection, out)` validates selected path is a directory, calls `capture_nodes(Path(selection))`, writes `to_tree_text(nodes)`.
- `plan(spec, base, out=None)` uses existing parse/plan path and writes a text plan to a temp file if `out` is absent.
- `diff(spec, base)` returns a text report using existing `diff` logic.
- `apply(spec, base, confirm=False)` runs a dry-run first. If `confirm` is false, return preview and do not write.

**UX contract:** Return an `ActionResult` dataclass:

```python
@dataclass
class ActionResult:
    ok: bool
    title: str
    message: str
    output_path: Path | None = None
```

The macOS/Windows wrappers can show `title` and `message` in native UI.

### Task 3: Implement macOS Finder Quick Action installer

**Objective:** Install user-level Finder Quick Actions that appear in right-click menus without requiring admin permissions.

**Files:**
- Create: `src/seed_cli/desktop/macos.py`
- Create templates under: `src/seed_cli/resources/desktop/macos/`
- Modify: `pyproject.toml` force-include resources.
- Test: `tests/test_desktop_macos.py`

**Recommended macOS mechanism:** Finder Quick Actions / Services installed into:

```text
~/Library/Services/Seed Capture Folder.workflow
~/Library/Services/Seed Preview Plan.workflow
~/Library/Services/Seed Apply Spec.workflow
~/Library/Services/Seed Diff With Spec.workflow
~/Library/Services/Seed Revert Last Change.workflow
```

Each workflow should be an Automator workflow package containing an `Info.plist` and a shell script action. Finder passes selected files as arguments to the shell script.

**Why this route:**
- Uses Finder's native right-click UI.
- User-level install; no admin prompt.
- Does not require a long-running app.
- Works well for non-technical users after `seed desktop install`.

**Shell script pattern:**

```bash
#!/bin/zsh
set -euo pipefail
SEED_BIN="${SEED_BIN:-seed}"
for selected in "$@"; do
  "$SEED_BIN" desktop run capture --selection "$selected"
done
```

**Native UI:**
- Use AppleScript via `osascript` for file/folder pickers and confirmation:
  - `choose file name with prompt "Save Seed spec as..." default name "project.tree"`
  - `choose folder with prompt "Choose the folder Seed should compare/apply against"`
  - `display dialog "Seed will apply these changes..." buttons {"Cancel", "Apply"}`
- Reveal output in Finder with `open -R <output_path>`.

**Status checks:**
- `seed desktop status` should report which workflows are installed and whether the `seed` binary is resolvable.

### Task 4: Implement Windows Explorer context-menu installer

**Objective:** Install per-user Explorer context-menu actions without requiring admin permissions.

**Files:**
- Create: `src/seed_cli/desktop/windows.py`
- Create: `src/seed_cli/resources/desktop/windows/seed_desktop_launcher.ps1`
- Test: `tests/test_desktop_windows.py`

**Recommended Windows mechanism:** Per-user registry keys under HKCU:

```text
HKCU\Software\Classes\Directory\shell\SeedCapture
HKCU\Software\Classes\Directory\shell\SeedDiff
HKCU\Software\Classes\*\shell\SeedPreviewPlan
HKCU\Software\Classes\*\shell\SeedApplySpec
```

Each `command` should invoke Python/Seed through a stable launcher:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<installed launcher.ps1>" -Action Capture -Selection "%1"
```

**Why this route:**
- Uses Explorer's native right-click menu.
- HKCU install avoids admin rights.
- Works for folders and files.
- Can be installed/uninstalled from `seed desktop install` / `seed desktop uninstall`.

**Native UI:**
- Use PowerShell `System.Windows.Forms` dialogs for folder/file selection and confirmations.
- Use `Start-Process notepad.exe <plan-output>` or open generated text/html in the default app.
- Use Windows notifications later; first version can use message boxes.

**Safety:**
- `Apply` must show dry-run output first and ask user to confirm.
- `sync` and `match` should not be installed by default.

### Task 5: Add installer resource handling

**Objective:** Make integration templates available after `pip install seed-cli`.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/seed_cli/resources/desktop/...`
- Test: `tests/test_desktop_resources.py`

**Implementation:**
- Add resource directories to Hatch force-include or package data.
- Use `importlib.resources.files("seed_cli.resources.desktop")` to copy scripts/templates.
- Install generated wrappers into a user config directory:
  - macOS: `~/Library/Application Support/seed-cli/desktop/`
  - Windows: `%APPDATA%\seed-cli\desktop\`

### Task 6: Add menu/action configuration

**Objective:** Let users/admins control which actions are installed.

**Files:**
- Create: `src/seed_cli/desktop/config.py`
- Test: `tests/test_desktop_config.py`

**Config file:**

```yaml
actions:
  capture: true
  preview_plan: true
  diff: true
  apply: true
  revert: true
  sync: false
  match: false
```

**Commands:**

```bash
seed desktop install --only capture,diff
seed desktop install --with-advanced
seed desktop uninstall
seed desktop status --json
```

### Task 7: Add docs for non-technical setup

**Objective:** Make the feature understandable for users who do not live in terminals.

**Files:**
- Modify: `README.md`
- Create: `docs/desktop-actions.md`

**Content:**
- One-time install:

```bash
pip install seed-cli
seed desktop install
```

- macOS instructions: Finder → right-click folder/spec → Quick Actions → Seed...
- Windows instructions: Explorer → right-click folder/spec → Seed...
- How to uninstall:

```bash
seed desktop uninstall
```

- Safety explanation: Seed previews before writes; destructive operations are not installed by default.

## Native UI Design Details

### macOS Finder

Finder does not have a simple cross-version public API for arbitrary third-party context menu entries from Python. The practical user-level path is Finder Quick Actions/Services. They appear in Finder's context menu under **Quick Actions** or **Services**, and they receive selected paths.

If later we want top-level Finder context menu items, that requires a Finder Sync Extension inside a signed `.app`, which is a much bigger packaging project. Do not start there.

### Windows Explorer

Explorer supports context menu verbs through the registry. The first version should use static verbs under HKCU because they are simple, reversible, and do not require admin permissions. A full COM `IExplorerCommand` extension would be more native and dynamic, but it requires compiled code/signing and is overkill for v1.

### Linux

Not requested, but the same `seed desktop run ...` layer can later support:
- Nautilus scripts under `~/.local/share/nautilus/scripts/`
- KDE Dolphin service menus under `~/.local/share/kio/servicemenus/`

## Open Product Decisions

1. Should `Apply Spec` be installed by default, or should v1 only install safe read-only actions (`Capture`, `Preview Plan`, `Diff`) unless the user passes `--with-write-actions`?
2. Should generated plan previews be `.txt`, `.html`, or both? `.html` is friendlier; `.txt` is simpler and safer.
3. On macOS, is **Quick Actions** acceptable, or do we eventually need a signed `.app` with Finder Sync for true top-level context menu entries?
4. Should selected folders automatically look for `.seed/current.tree` or ask the user to choose a spec every time?
5. Should the installer support packaged app builds (PyInstaller/Briefcase) in addition to `pip install`?

## Verification Commands

Run focused tests first:

```bash
python -m pytest tests/test_desktop_actions.py tests/test_desktop_resources.py -v
```

Run platform dry-run tests on each OS:

```bash
python -m pytest tests/test_desktop_macos.py -v      # macOS only
python -m pytest tests/test_desktop_windows.py -v    # Windows only
```

Run full suite before release:

```bash
python -m pytest
```

Manual verification:

- macOS:
  1. `seed desktop install`
  2. Open Finder.
  3. Right-click a folder.
  4. Confirm Seed actions appear under Quick Actions/Services.
  5. Run Capture and verify a `.tree` file is created.
  6. Run `seed desktop uninstall` and verify actions disappear.

- Windows:
  1. `seed desktop install`
  2. Restart Explorer or refresh shell associations if needed.
  3. Right-click a folder/spec file.
  4. Confirm Seed actions appear.
  5. Run Capture and Preview Plan.
  6. Run `seed desktop uninstall` and verify registry entries are removed.

## Definition of Done

- `seed desktop install`, `status`, and `uninstall` work on macOS and Windows without admin rights.
- Finder/Explorer actions appear in native right-click UI.
- Actions use OS-native file/folder prompts and confirmation dialogs.
- All writes go through existing Seed planning/apply logic.
- Apply previews changes before writing and requires confirmation.
- Uninstall cleans up workflows, launchers, and registry keys created by Seed.
- Docs explain setup and usage for non-technical users.
