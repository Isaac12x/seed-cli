# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Added `seed register <spec>` in `src/seed_cli/cli.py` to:
  - mirror `.tree` specs into `.seed/templates/`
  - extract nested project templates into `.seed/templates/project/`
  - remove stale literal placeholder paths such as `<name>/` left by older apply runs

### Changed
- Refactored project-template registration in `src/seed_cli/project_templates.py`:
  - introduced explicit registration result/cleanup helpers
  - moved stale placeholder cleanup into the shared registration flow
  - mirrors any `.tree` spec, even when it has no template subtree to extract
- Updated `apply()` in `src/seed_cli/apply.py`:
  - runs the shared registration flow for spec inputs after snapshot creation and under the apply lock
  - removes previously materialized literal template subtrees before executing the pruned plan
- Updated documentation in `README.md` and `docs/index.html`:
  - documents `seed register`
  - clarifies that `seed apply <spec>` auto-registers project-local templates

### Fixed
- Fixed a project-template recovery gap where rerunning `seed apply FILENAME.tree` on older worktrees could leave literal `<NAME>` directories in place instead of converting them into `.seed` support files.

### Tests
- Added coverage for explicit registration and stale template cleanup in:
  - `tests/test_apply.py`
  - `tests/test_cli.py`
  - `tests/test_project_templates.py`

## [1.0.8] - 2026-04-06

### Added
- Added `src/seed_cli/security.py` with path-hardening helpers:
  - `normalize_relpath()`
  - `safe_target_path()`
  - `validate_plan_paths()`
- Added manifest-driven maintenance orchestration in `src/seed_cli/maintenance.py` and `seed maintain`:
  - manifest discovery for `maintenance.yml`, `project.yml`, and `service.yml`
  - repository, service, system, integration, and project target expansion
  - built-in maintenance goals such as `git_fetch`, `git_status`, `git_pull_ff_only`, `compose_pull`, `compose_up`, and `launchctl_restart`
  - execution guardrails that refuse `git pull --ff-only` on dirty worktrees
- Added Copier-style scaffolding helpers in `src/seed_cli/scaffold.py`:
  - data file loading (`JSON`/`YAML`)
  - template config loading
  - question/default/non-interactive variable resolution
  - `_tasks` extraction and gated execution
  - `_exclude` and `_skip_if_exists` support
  - `_answers_file` resolution and answers file writing
- Added project-local template registration and lookup in `src/seed_cli/project_templates.py`:
  - mirrors template-capable specs into `.seed/templates/`
  - extracts nested template subtrees into `.seed/templates/project/`
  - resolves registered project template names and paths for `seed create`
- Added template config discovery and storage support in `src/seed_cli/template_registry.py`:
  - recognizes `copier.yml`, `copier.yaml`, `.seed-template.yml`, `.seed-template.yaml`
  - persists versioned template config as `<version>.config.yml`
  - exposes `get_template_config_path()`
- Added `SEED_HOME` environment variable support for template storage root in `template_registry`.
- Added new `templates use` CLI flags in `src/seed_cli/cli.py`:
  - `--data-file`
  - `--defaults`
  - `--non-interactive`
  - `--answers-file`
  - `--unsafe`
  - `--overwrite`

### Changed
- Extended `apply()` signature and behavior in `src/seed_cli/apply.py`:
  - supports `step_hooks`, `template_exclude`, `template_skip_if_exists`
  - validates plan paths before execution
  - avoids mutable default for `plugins`
- Extended `sync()` in `src/seed_cli/sync.py`:
  - supports `plugins` and `step_hooks`
  - validates plan paths before execution
  - runs plugin `before_build`/`after_build` lifecycle
- Extended `match()` in `src/seed_cli/match.py`:
  - supports `plugins` and `step_hooks`
  - validates plan paths before execution
  - runs plugin `before_build`/`after_build` lifecycle
- Enhanced `execute_plan()` in `src/seed_cli/executor.py`:
  - validates/normalizes plan paths before execution
  - resolves all execution targets through `safe_target_path()`
  - supports template exclude/skip-if-exists patterns
  - runs `pre_step`/`post_step` hooks
  - applies plugin deletion veto (`before_sync_delete`) on delete steps
- Updated template usage flow in `src/seed_cli/cli.py`:
  - resolves variables from config + data file + CLI vars
  - supports answers file generation with base-directory containment checks
  - runs template tasks only with `--unsafe`
  - passes overwrite/exclude/skip rules into apply pipeline
- Expanded `seed create` in `src/seed_cli/cli.py`:
  - supports `--template` paths rooted in `.seed/templates/`
  - supports `--project` for registered project-local templates
  - adds shell completion for project template names and paths
- Refreshed project documentation:
  - rewrote `README.md` to cover maintenance workflows, template registry usage, copier-style scaffolding, and safety model details
  - updated `docs/index.html` to match the expanded CLI surface
- Metadata cleanup:
  - synchronized package version to `1.0.8`
  - now derives `src/seed_cli/__init__.py` version from `pyproject.toml` when available
  - removed Python 3.6-3.9 classifiers from `pyproject.toml` to match `requires-python >=3.10`
- Updated `.gitignore`:
  - ignores `examples/`, `data/`, and nested Python cache directories

### Fixed
- Fixed image spec parsing runtime bug in `src/seed_cli/parsers.py`:
  - replaced undefined image parse call with `parse_image()`
  - now raises clearer contextual parse errors for image specs
- Fixed `apply --dangerous` wiring bug in CLI:
  - CLI now passes `args.dangerous` into `apply()`
- Fixed filesystem hook closure bug in `src/seed_cli/hooks.py`:
  - each discovered script now executes its own path correctly
- Improved filesystem hook environment:
  - exported `SEED_HOOK_STAGE` to script hooks

### Security
- Hardened filesystem write/delete safety:
  - rejects absolute paths, parent traversal, NUL bytes, and invalid relative paths in plan steps
  - enforces base-directory containment for operation targets
  - enforces containment checks for answers file output path
  - validates plan paths in `apply`, `sync`, and `match` before execution

### Tests
- Added `tests/test_security.py`:
  - path traversal and absolute path rejection
  - unsafe plan JSON rejection
  - symlink escape containment tests
- Added `tests/test_scaffold.py`:
  - data/config loading
  - variable resolution semantics
  - tasks/exclude/skip/answers helpers
  - task execution gating with `unsafe`
- Updated `tests/test_cli.py`:
  - verifies apply plan delete works with `--dangerous`
- Updated `tests/test_hooks.py`:
  - verifies distinct script execution
  - verifies stage env propagation
  - validates hook helper wrappers and missing-hooks behavior
- Updated `tests/test_sync.py`:
  - verifies plugin deletion veto behavior
- Added `tests/test_maintenance.py`:
  - workspace, project, and service manifest planning coverage
  - maintenance command execution coverage
  - dirty-worktree protection for `git pull --ff-only`
- Added `tests/test_project_templates.py`:
  - project template registration, extraction, and lookup coverage
- Full test suite status after changes:
  - `507 passed`
  - coverage `83.89%` (below gate `85%` when run with repo defaults)
