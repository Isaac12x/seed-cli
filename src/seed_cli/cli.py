# src/seed_cli/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seed_cli.logging import setup_logging, get_logger
from seed_cli.ui import Summary, render_summary, render_list
from seed_cli.parsers import read_input, parse_any
from seed_cli.includes import resolve_includes
from seed_cli.templating import apply_vars
from seed_cli.capture import capture_nodes, to_tree_text, to_json, to_dot
from seed_cli.exporter import (
    export_tree,
    export_json_spec,
    export_plan,
    export_dot,
)
from seed_cli.planning import plan as build_plan
from seed_cli.diff import diff
from seed_cli.apply import apply
from seed_cli.sync import sync
from seed_cli.doctor import doctor
from seed_cli.graphviz import plan_to_dot
from seed_cli.image import read_tree
from seed_cli.hooks import run_hooks, load_filesystem_hooks
from seed_cli.templates import install_git_hook
from seed_cli.plugins import load_plugins

log = get_logger("cli")
DEFAULT_TTL = 30
DEFAULT_LOCK_TIMEOUT = 10

def parse_vars(values):
    out = {}
    for v in values or []:
        if "=" in v:
            k, val = v.split("=", 1)
            out[k] = val
    return out


def parse_spec_file(spec_path: str, vars: dict, base: Path, plugins: list, context: dict) -> tuple[Path, list]:
    """Parse a spec file (text, image, or graphviz) into nodes.
    
    Handles:
    - Text files (.tree, .yaml, .json)
    - Image files (.png, .jpg, .jpeg)
    - Graphviz files (.dot)
    
    Applies includes, vars, and plugin hooks.
    
    Returns:
        tuple: (spec_path, nodes)
    """
    from seed_cli.parsers import parse_spec
    
    spec = Path(spec_path)
    
    # For image files, parse directly (includes/vars handled by read_tree -> parse_any)
    if spec.suffix.lower() in (".png", ".jpg", ".jpeg"):
        _, nodes = parse_spec(spec_path, vars=vars, base=base)
        return spec, nodes
    
    # For DOT files, parse directly (vars handled by parse_spec)
    if spec.suffix.lower() == ".dot":
        _, nodes = parse_spec(spec_path, vars=vars, base=base)
        return spec, nodes
    
    # For text files, apply includes and vars before parsing
    text = read_input(spec_path)
    text = resolve_includes(text, spec)
    text = apply_vars(text, vars)
    
    # Apply plugin hooks before parsing
    for p in plugins:
        text = p.before_parse(text, context)
    
    # Parse with vars=None since we've already applied them
    # parse_any will still apply includes, but that's idempotent
    _, nodes = parse_any(spec_path, text, vars=None, base=base)
    return spec, nodes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "seed",
        description="Terraform-inspired filesystem orchestration tool",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Extra ignore patterns (glob). Can be specified multiple times.",
    )
    p.add_argument(
        "--targets",
        action="append",
        default=[],
        help="Extra targets (glob). Can be specified multiple times.",
    )
    p.add_argument(
        "--target-mode",
        choices=["prefix", "exact"],
        default="prefix",
        help="Target mode (prefix or exact). Default: prefix",
    )

    sub = p.add_subparsers(dest="cmd", required=False, help="Available commands")

    # plan
    sp = sub.add_parser(
        "plan",
        description="Parse spec, run plugin parse + plan lifecycle, and output plan",
        help="Parse spec and generate execution plan",
    )
    sp.add_argument("spec", help="Spec file (.tree, .yaml, .json, .dot, or image)")
    sp.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sp.add_argument("--vars", action="append", help="Template variables (key=value)")
    sp.add_argument("--out", help="Output plan to file (JSON format)")
    sp.add_argument("--dot", action="store_true", help="Output plan as Graphviz DOT format")
    sp.add_argument("--no-skip", action="store_true", dest="no_skip",
                    help="Hide SKIP lines in output for cleaner view")

    # apply
    sa = sub.add_parser(
        "apply",
        description="Acquire state lock, run hooks (pre_apply, post_apply), execute plan, and run plugin build lifecycle",
        help="Execute plan to create/update files and directories",
    )
    sa.add_argument("spec", help="Spec file or plan.json")
    sa.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sa.add_argument("--dangerous", action="store_true", help="Allow dangerous operations")
    sa.add_argument("--dry-run", action="store_true", help="Show what would be executed without making changes")
    sa.add_argument("--yes", "-y", action="store_true", help="Create all optional items (marked with ?) without prompting")
    sa.add_argument("--skip-optional", action="store_true", help="Skip all optional items (marked with ?) without prompting")

    # sync
    ss = sub.add_parser(
        "sync",
        description="Same as apply, but also deletes extraneous files. Plugins may veto deletions.",
        help="Execute plan and delete extraneous files (dangerous, gated)",
    )
    ss.add_argument("spec", help="Spec file")
    ss.add_argument("--base", default=".", help="Base directory (default: current directory)")
    ss.add_argument("--dangerous", action="store_true", help="Required flag to enable sync (dangerous operation). Not required when using --dry-run")
    ss.add_argument("--dry-run", action="store_true", help="Show what would be executed without making changes")
    ss.add_argument("--yes", "-y", action="store_true", help="Create all optional items (marked with ?) without prompting")
    ss.add_argument("--skip-optional", action="store_true", help="Skip all optional items (marked with ?) without prompting")

    # diff
    sd = sub.add_parser(
        "diff",
        description="Compare spec with filesystem and show missing, extra, and drifted paths",
        help="Compare spec with filesystem state",
    )
    sd.add_argument("spec", help="Spec file")
    sd.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sd.add_argument("--ignore", action="append", default=[], metavar="PATTERN",
                    help="Ignore paths matching pattern (can be specified multiple times)")
    sd.add_argument("--no-sublevels", action="store_true",
                    help="Hide extras that are inside directories defined in the spec")

    # match
    sm = sub.add_parser(
        "match",
        description="Modify filesystem to match the spec. Creates missing items and deletes extras. "
                    "Use `...` in spec to mark directories where extra files are allowed.",
        help="Modify filesystem to match spec (dangerous)",
    )
    sm.add_argument("spec", help="Spec file")
    sm.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sm.add_argument("--dangerous", action="store_true",
                    help="Required flag to enable match (will create/delete files). Not required with --dry-run")
    sm.add_argument("--dry-run", action="store_true",
                    help="Preview changes without modifying filesystem")
    sm.add_argument("--yes", "-y", action="store_true",
                    help="Create all optional items (marked with ?) without prompting")
    sm.add_argument("--skip-optional", action="store_true",
                    help="Skip all optional items (marked with ?) without prompting")

    # create - instantiate template structures
    sc_create = sub.add_parser(
        "create",
        description="Create a new instance of a template directory structure. "
                    "Use with specs containing <varname>/ template patterns.",
        help="Create new instance of template structure",
    )
    sc_create.add_argument("spec", help="Spec file containing template")
    sc_create.add_argument("values", nargs="+",
                           help="Template values as varname=value (e.g., version_id=v3)")
    sc_create.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sc_create.add_argument("--dry-run", action="store_true",
                           help="Preview what would be created")

    # revert - undo last operation
    sr = sub.add_parser(
        "revert",
        description="Revert filesystem to a previous snapshot. Snapshots are created "
                    "automatically before apply/match/sync operations.",
        help="Revert to previous snapshot (undo)",
    )
    sr.add_argument("snapshot_id", nargs="?", help="Snapshot ID to revert to (default: latest)")
    sr.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sr.add_argument("--list", action="store_true", dest="list_snapshots",
                    help="List available snapshots")
    sr.add_argument("--dry-run", action="store_true",
                    help="Preview what would be reverted")
    sr.add_argument("--delete", metavar="ID", help="Delete a specific snapshot")

    # doctor
    sdoc = sub.add_parser(
        "doctor",
        description="Lint spec and optionally auto-fix issues",
        help="Lint spec for issues",
    )
    sdoc.add_argument("spec", help="Spec file")
    sdoc.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sdoc.add_argument("--fix", action="store_true", help="Automatically fix issues when possible")

    # capture
    sc = sub.add_parser(
        "capture",
        description="Capture current filesystem state as a spec",
        help="Capture current filesystem state",
    )
    sc.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sc.add_argument("--json", action="store_true", help="Output in JSON format")
    sc.add_argument("--dot", action="store_true", help="Output in Graphviz DOT format")
    sc.add_argument("--out", help="Output file path. If not specified, output is printed to stdout")

    # export
    se = sub.add_parser(
        "export",
        description="Export filesystem state or plan in various formats",
        help="Export filesystem state or plan",
    )
    se.add_argument("kind", choices=["tree", "json", "plan", "dot"], help="Export format")
    se.add_argument("--input", help="Input spec or plan file (default: capture from filesystem)")
    se.add_argument("--out", required=True, help="Output file path")
    se.add_argument("--base", default=".", help="Base directory (default: current directory)")

    # lock - structure locking
    sl = sub.add_parser(
        "lock",
        description="Lock filesystem structure to a spec. Supports versioning and watch mode.",
        help="Lock filesystem structure to a spec",
    )
    lock_sub = sl.add_subparsers(dest="lock_action", help="Lock action")

    # lock set <spec> - set/create a structure lock
    lock_set = lock_sub.add_parser(
        "set",
        description="Set the active structure spec (creates version if needed)",
        help="Set active structure spec",
    )
    lock_set.add_argument("spec", help="Spec file to lock to")
    lock_set.add_argument("--base", default=".", help="Base directory")
    lock_set.add_argument("--version", "-v", help="Version name (default: auto-increment)")

    # lock watch - continuous enforcement
    lock_watch = lock_sub.add_parser(
        "watch",
        description="Watch filesystem and enforce structure continuously",
        help="Watch and enforce structure",
    )
    lock_watch.add_argument("--base", default=".", help="Base directory")
    lock_watch.add_argument("--interval", type=float, default=1.0, help="Check interval in seconds")

    # lock list - list versions
    lock_list = lock_sub.add_parser(
        "list",
        description="List available structure versions",
        help="List structure versions",
    )
    lock_list.add_argument("--base", default=".", help="Base directory")

    # lock upgrade <version>
    lock_upgrade = lock_sub.add_parser(
        "upgrade",
        description="Upgrade to a newer structure version",
        help="Upgrade to version",
    )
    lock_upgrade.add_argument("version", help="Version to upgrade to")
    lock_upgrade.add_argument("--base", default=".", help="Base directory")
    lock_upgrade.add_argument("--dangerous", action="store_true", help="Apply changes (required)")
    lock_upgrade.add_argument("--dry-run", action="store_true", help="Preview changes")

    # lock downgrade <version>
    lock_downgrade = lock_sub.add_parser(
        "downgrade",
        description="Downgrade to an older structure version",
        help="Downgrade to version",
    )
    lock_downgrade.add_argument("version", help="Version to downgrade to")
    lock_downgrade.add_argument("--base", default=".", help="Base directory")
    lock_downgrade.add_argument("--dangerous", action="store_true", help="Apply changes (required)")
    lock_downgrade.add_argument("--dry-run", action="store_true", help="Preview changes")

    # lock status - show current lock
    lock_status = lock_sub.add_parser(
        "status",
        description="Show current structure lock status",
        help="Show lock status",
    )
    lock_status.add_argument("--base", default=".", help="Base directory")

    # hooks
    sh = sub.add_parser(
        "hooks",
        description="Install git hooks (e.g. pre-commit)",
        help="Manage git hooks",
    )
    sh.add_argument(
        "action",
        choices=["install"],
        help="Action to perform",
    )
    sh.add_argument(
        "--hook",
        action="append",
        help="Hook name to install (e.g. pre-commit). Defaults to pre-commit if not specified.",
    )

    # specs - view captured spec history
    ssp = sub.add_parser(
        "specs",
        description="View and manage automatically captured spec versions",
        help="View captured spec history",
    )
    specs_sub = ssp.add_subparsers(dest="specs_action", help="Specs action")

    # specs list
    specs_list = specs_sub.add_parser(
        "list",
        description="List all captured spec versions",
        help="List spec versions",
    )
    specs_list.add_argument("--base", default=".", help="Base directory")

    # specs show
    specs_show = specs_sub.add_parser(
        "show",
        description="Show content of a spec version",
        help="Show spec content",
    )
    specs_show.add_argument("version", nargs="?", help="Version to show (e.g., 1 or v1). Default: latest")
    specs_show.add_argument("--base", default=".", help="Base directory")

    # specs diff
    specs_diff = specs_sub.add_parser(
        "diff",
        description="Compare two spec versions",
        help="Diff spec versions",
    )
    specs_diff.add_argument("v1", help="First version (e.g., 1 or v1)")
    specs_diff.add_argument("v2", help="Second version (e.g., 2 or v2)")
    specs_diff.add_argument("--base", default=".", help="Base directory")

    # templates - manage reusable specs from GitHub
    stpl = sub.add_parser(
        "templates",
        description="Manage reusable spec templates from GitHub",
        help="Manage spec templates",
    )
    templates_sub = stpl.add_subparsers(dest="templates_action", help="Templates action")

    # templates list
    tpl_list = templates_sub.add_parser(
        "list",
        description="List all stored templates",
        help="List templates",
    )

    # templates add <github_url>
    tpl_add = templates_sub.add_parser(
        "add",
        description="Add a template from GitHub URL or local file",
        help="Add a template",
    )
    tpl_add.add_argument("source", help="GitHub URL or local file path")
    tpl_add.add_argument("--name", "-n", help="Name for the template (default: derived from URL/filename)")
    tpl_add.add_argument("--version", "-v", help="Version name (default: auto-increment)")

    # templates remove <name>
    tpl_remove = templates_sub.add_parser(
        "remove",
        description="Remove a template",
        help="Remove a template",
    )
    tpl_remove.add_argument("name", help="Template name to remove")

    # templates use <name>
    tpl_use = templates_sub.add_parser(
        "use",
        description="Apply a template to the filesystem",
        help="Use a template",
    )
    tpl_use.add_argument("name", help="Template name to use")
    tpl_use.add_argument("--version", "-v", help="Version to use (default: current)")
    tpl_use.add_argument("--base", default=".", help="Base directory (default: current directory)")
    tpl_use.add_argument("--vars", action="append", help="Template variables (key=value)")
    tpl_use.add_argument("--yes", "-y", action="store_true", help="Apply without prompting")
    tpl_use.add_argument("--dry-run", action="store_true", help="Show plan without applying")

    # templates show <name>
    tpl_show = templates_sub.add_parser(
        "show",
        description="Show template content",
        help="Show template content",
    )
    tpl_show.add_argument("name", help="Template name")
    tpl_show.add_argument("--version", "-v", help="Version to show (default: current)")

    # templates lock <name>
    tpl_lock = templates_sub.add_parser(
        "lock",
        description="Lock or unlock a template",
        help="Lock/unlock template",
    )
    tpl_lock.add_argument("name", help="Template name")
    tpl_lock.add_argument("--version", "-v", help="Version to set as current before locking")
    tpl_lock.add_argument("--unlock", action="store_true", help="Unlock instead of lock")

    # templates versions <name>
    tpl_versions = templates_sub.add_parser(
        "versions",
        description="Manage template versions",
        help="Manage versions",
    )
    tpl_versions.add_argument("name", help="Template name")
    tpl_versions.add_argument("--add", metavar="PATH", help="Add a new version from file")
    tpl_versions.add_argument("--name", dest="version_name", metavar="VERSION", help="Name for the new version (with --add)")
    tpl_versions.add_argument("--set-current", metavar="VERSION", help="Set version as current")

    # utils
    sut = sub.add_parser(
        "utils",
        description="Utility functions for common operations",
        help="Utility functions",
    )
    utils_sub = sut.add_subparsers(dest="util_action", required=True, help="Utility action")
    
    # extract-tree subcommand
    extract_tree = utils_sub.add_parser(
        "extract-tree",
        description="Extract tree structure from an image using OCR",
        help="Extract tree structure from image",
    )
    extract_tree.add_argument("image", help="Path to image file (.png, .jpg, .jpeg)")
    extract_tree.add_argument("--out", help="Output .tree file path (default: image path with .tree extension)")
    extract_tree.add_argument("--vars", action="append", help="Template variables (key=value)")
    extract_tree.add_argument("--raw", action="store_true", help="Output raw OCR text without cleaning (for debugging)")

    # state-lock subcommand (moved from top-level lock)
    state_lock = utils_sub.add_parser(
        "state-lock",
        description="Manage execution state locks (for concurrent access control)",
        help="Manage execution state locks",
    )
    state_lock.add_argument("--base", default=".", help="Base directory")
    state_lock.add_argument("--renew", action="store_true", help="Renew existing lock")
    state_lock.add_argument("--force-unlock", action="store_true", help="Force unlock (use if process crashed)")

    return p


def main(argv=None) -> int:
    parser = build_parser()

    # Enable tab completion (requires: activate-global-python-argcomplete)
    import argcomplete
    argcomplete.autocomplete(parser)

    args = parser.parse_args(argv or sys.argv[1:])
    
    # If no command provided, show available commands
    if not args.cmd:
        print("seed: error: no command provided\n")
        print("Available commands:")
        subparsers = parser._subparsers._group_actions[0]
        for name, subparser in sorted(subparsers.choices.items()):
            help_text = getattr(subparser, 'help', '') or getattr(subparser, 'description', '')
            if help_text:
                print(f"  {name:12} {help_text}")
            else:
                print(f"  {name}")
        print("\nUse 'seed <command> -h' for help on a specific command.")
        return 1
    
    setup_logging(args.verbose, args.debug)


    base = Path(getattr(args, "base", ".")).resolve()
    vars = parse_vars(getattr(args, "vars", []))

    plugins = load_plugins()
    context = {
        "base": base,
        "plugins": plugins,
        "cmd": args.cmd,
    }

    
    # ---------------- PLAN ----------------
    if args.cmd == "plan":
        try:
            _, nodes = parse_spec_file(args.spec, vars, base, plugins, context)

            for p in plugins:
                p.after_parse(nodes, context)

            for p in plugins:
                p.before_plan(nodes, context)

            plan = build_plan(
                nodes, 
                base,
                ignore=args.ignore,
                allow_delete=False,
                targets=args.targets,
                target_mode=args.target_mode,
            )

            for p in plugins:
                p.after_plan(plan, context)

            if args.dot:
                print(plan_to_dot(plan))
                return 0

            if args.out:
                export_plan(plan, Path(args.out))
                return 0

            show_skip = not getattr(args, "no_skip", False)
            print(plan.to_text(show_skip=show_skip))
            return 0
        except Exception as e:
            log.error(f"Error planning: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    # ---------------- APPLY / SYNC ----------------
    if args.cmd in ("apply", "sync"):
        hooks = plugins + load_filesystem_hooks(base / "hooks")
        try:
            run_hooks(hooks, "pre_apply", cwd=base, strict=True)

            if args.cmd == "apply":
                result = apply(
                    args.spec,
                    base,
                    plugins=plugins,
                    dry_run=args.dry_run,
                    include_optional=args.yes,
                    skip_optional=args.skip_optional,
                )
            else:
                # For sync, --dangerous is required unless --dry-run is used
                if not args.dry_run and not args.dangerous:
                    print("seed sync: error: --dangerous flag is required for sync operations (use --dry-run to preview without --dangerous)")
                    return 1
                result = sync(
                    args.spec,
                    base,
                    dangerous=args.dangerous,
                    dry_run=args.dry_run,
                    include_optional=args.yes,
                    skip_optional=args.skip_optional,
                )

            run_hooks(hooks, f"post_{args.cmd}", strict=True, cwd=base)

            # Extract extra fields before creating Summary
            snapshot_id = result.pop("snapshot_id", None)
            spec_version = result.pop("spec_version", None)
            spec_path = result.pop("spec_path", None)
            summary = Summary(**result)
            print(render_summary(summary))
            if spec_version:
                print(f"\nSpec captured: v{spec_version} ({spec_path})")
            if snapshot_id:
                print(f"Snapshot created: {snapshot_id}")
                print("Use 'seed revert' to undo changes")
            return 0
        except Exception as e:
            log.error(f"Error {args.cmd}: {e}")
            return 1
  
    # ---------------- DIFF ----------------
    if args.cmd == "diff":
        _, nodes = parse_spec_file(args.spec, vars, base, plugins, context)
        res = diff(nodes, base, ignore=args.ignore, skip_sublevels=args.no_sublevels)
        print(render_list("Missing", res.missing))
        print(render_list("Extra", res.extra))
        print(render_list("Type Mismatch", res.type_mismatch))
        print(render_list("Drift", res.drift))
        return 0 if res.is_clean() else 1

    # ---------------- MATCH ----------------
    if args.cmd == "match":
        from seed_cli.match import match as do_match, match_plan

        # Require --dangerous unless --dry-run
        if not args.dry_run and not args.dangerous:
            print("seed match: error: --dangerous flag is required (will create/delete files)")
            print("Use --dry-run to preview changes without --dangerous")
            return 1

        try:
            if args.dry_run:
                # Show plan without executing
                plan = match_plan(
                    args.spec,
                    base,
                    vars=vars,
                    ignore=args.ignore,
                    targets=args.targets,
                    target_mode=args.target_mode,
                )
                print("DRY RUN - No changes will be made\n")
                print(plan.to_text())
                return 0
            else:
                result = do_match(
                    args.spec,
                    base,
                    dangerous=args.dangerous,
                    dry_run=False,
                    vars=vars,
                    ignore=args.ignore,
                    targets=args.targets,
                    target_mode=args.target_mode,
                    include_optional=args.yes,
                    skip_optional=args.skip_optional,
                )
                # Extract extra fields before creating Summary
                snapshot_id = result.pop("snapshot_id", None)
                spec_version = result.pop("spec_version", None)
                spec_path = result.pop("spec_path", None)
                summary = Summary(**result)
                print(render_summary(summary))
                if spec_version:
                    print(f"\nSpec captured: v{spec_version} ({spec_path})")
                if snapshot_id:
                    print(f"Snapshot created: {snapshot_id}")
                    print("Use 'seed revert' to undo changes")
                return 0
        except Exception as e:
            log.error(f"Error during match: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    # ---------------- CREATE ----------------
    if args.cmd == "create":
        from seed_cli.match import create_from_template

        # Parse template values
        template_values = {}
        for val in args.values:
            if "=" not in val:
                print(f"Error: Invalid value format '{val}'. Use varname=value")
                return 1
            k, v = val.split("=", 1)
            template_values[k] = v

        try:
            result = create_from_template(
                args.spec,
                base,
                template_values,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                print("DRY RUN - Would create:\n")
                for p in result["paths"]:
                    print(f"  {p}")
                print(f"\nTotal: {result['created']} items")
            else:
                print(f"Created {result['created']} items:")
                for p in result["paths"]:
                    print(f"  {p}")
            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            log.error(f"Error during create: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    # ---------------- REVERT ----------------
    if args.cmd == "revert":
        from seed_cli.snapshot import list_snapshots, get_snapshot, revert_snapshot, delete_snapshot
        import time as time_module

        # Delete a snapshot
        if args.delete:
            if delete_snapshot(base, args.delete):
                print(f"Deleted snapshot: {args.delete}")
            else:
                print(f"Snapshot not found: {args.delete}")
            return 0

        # List snapshots
        if args.list_snapshots:
            snapshots = list_snapshots(base)
            if not snapshots:
                print("No snapshots available.")
                print("\nSnapshots are created automatically before apply/match/sync operations.")
                return 0

            print("Available snapshots (newest first):\n")
            for snap in snapshots:
                age = time_module.time() - snap.created_at
                if age < 60:
                    age_str = f"{int(age)}s ago"
                elif age < 3600:
                    age_str = f"{int(age/60)}m ago"
                elif age < 86400:
                    age_str = f"{int(age/3600)}h ago"
                else:
                    age_str = f"{int(age/86400)}d ago"

                summary = ""
                if snap.plan_summary:
                    s = snap.plan_summary
                    summary = f" (+{s.get('add',0)} ~{s.get('change',0)} -{s.get('delete',0)})"

                print(f"  {snap.id}  {snap.operation:8}{summary}  {age_str}")
            print(f"\nUse 'seed revert <snapshot_id>' to restore")
            return 0

        # Revert to a snapshot
        snapshot_id = args.snapshot_id
        if not snapshot_id:
            # Get the latest snapshot
            snapshots = list_snapshots(base)
            if not snapshots:
                print("No snapshots available to revert to.")
                return 1
            snapshot_id = snapshots[0].id
            print(f"Reverting to latest snapshot: {snapshot_id}")

        try:
            result = revert_snapshot(base, snapshot_id, dry_run=args.dry_run)

            if args.dry_run:
                print(f"DRY RUN - Would revert snapshot {snapshot_id}:\n")
                for action in result["actions"]:
                    print(f"  {action}")
                print(f"\nTotal: {result['restored']} items would be restored")
            else:
                print(f"Reverted to snapshot: {snapshot_id}")
                print(f"Restored {result['restored']} items")
                for action in result["actions"]:
                    print(f"  {action}")
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            log.error(f"Error during revert: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    # ---------------- DOCTOR ----------------
    if args.cmd == "doctor":
        _, nodes = parse_spec_file(args.spec, vars, base, plugins, context)
        issues = doctor(nodes, base, fix=args.fix)
        if issues:
            print(render_list("Issues", issues))
            return 1
        print("Spec is healthy.")
        return 0

    # ---------------- CAPTURE ----------------
    if args.cmd == "capture":
        nodes = capture_nodes(base)
        if args.dot:
            output = to_dot(nodes)
        elif args.json:
            output = to_json(nodes)
        else:
            output = to_tree_text(nodes)

        if args.out:
            Path(args.out).write_text(output)
        else:
            print(output)
        return 0

    # ---------------- SPECS ----------------
    if args.cmd == "specs":
        from seed_cli.spec_history import (
            list_spec_versions,
            get_spec_version,
            get_current_spec,
            diff_spec_versions,
        )

        # Default to list if no action specified
        action = args.specs_action or "list"

        if action == "list":
            versions = list_spec_versions(base)
            if not versions:
                print("No captured specs found.")
                print("Specs are automatically captured after apply/sync/match operations.")
                return 0
            print("Captured spec versions:\n")
            for version, path, timestamp in versions:
                print(f"  v{version}  {timestamp}")
            print(f"\nTotal: {len(versions)} versions")
            print(f"Location: {base / '.seed/specs/'}")
            return 0

        if action == "show":
            version_arg = getattr(args, "version", None)
            if version_arg:
                # Parse version number (accept "1", "v1", etc.)
                v = int(version_arg.lstrip("v"))
                content = get_spec_version(base, v)
                if content is None:
                    print(f"Version v{v} not found.")
                    return 1
                print(content)
            else:
                # Show latest
                result = get_current_spec(base)
                if result is None:
                    print("No captured specs found.")
                    return 1
                version, content = result
                print(f"# Current spec: v{version}\n")
                print(content)
            return 0

        if action == "diff":
            v1 = int(args.v1.lstrip("v"))
            v2 = int(args.v2.lstrip("v"))
            try:
                result = diff_spec_versions(base, v1, v2)
                if result["added"]:
                    print(f"Added in v{v2}:")
                    for p in result["added"]:
                        print(f"  + {p}")
                if result["removed"]:
                    print(f"\nRemoved in v{v2}:")
                    for p in result["removed"]:
                        print(f"  - {p}")
                if not result["added"] and not result["removed"]:
                    print(f"No differences between v{v1} and v{v2}")
                return 0
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        return 0

    # ---------------- EXPORT ----------------
    if args.cmd == "export":
        out = Path(args.out)
        
        # Get nodes: from input file if provided, otherwise capture from filesystem
        if args.input:
            from seed_cli.parsers import parse_spec
            _, nodes = parse_spec(args.input, vars=vars, base=base)
        else:
            nodes = capture_nodes(base)
        
        if args.kind == "tree":
            export_tree(nodes, out)
        elif args.kind == "json":
            export_json_spec(nodes, out)
        elif args.kind == "plan":
            export_plan(build_plan(nodes, base), out)
        elif args.kind == "dot":
            export_dot(build_plan(nodes, base), out)
        return 0

    # ---------------- LOCK (structure locking) ----------------
    if args.cmd == "lock":
        from seed_cli.structure_lock import (
            set_lock, get_status, list_versions, switch_version, watch
        )
        import time as time_module

        action = getattr(args, "lock_action", None)

        # Default: show status if no action
        if not action or action == "status":
            status = get_status(base)
            if not status.active_version:
                print("No structure lock active.")
                print("\nUse 'seed lock set <spec>' to lock to a structure.")
            else:
                print(f"Active version: {status.active_version}")
                print(f"Spec: {status.spec_path}")
                if status.locked_at:
                    print(f"Locked at: {time_module.ctime(status.locked_at)}")
            if status.versions:
                print(f"\nAvailable versions: {', '.join(status.versions)}")
            return 0

        if action == "set":
            try:
                version, stored_path = set_lock(
                    base,
                    args.spec,
                    version=getattr(args, "version", None),
                )
                print(f"Structure locked to version: {version}")
                print(f"Stored at: {stored_path.relative_to(base)}")
                return 0
            except Exception as e:
                log.error(f"Error setting lock: {e}")
                return 1

        if action == "list":
            versions = list_versions(base)
            if not versions:
                print("No structure versions found.")
                print("\nUse 'seed lock set <spec>' to create the first version.")
                return 0

            status = get_status(base)
            print("Available structure versions:")
            for v, path in versions:
                active = " (active)" if v == status.active_version else ""
                print(f"  {v}{active} - {path.name}")
            return 0

        if action in ("upgrade", "downgrade"):
            version = args.version
            if not args.dry_run and not args.dangerous:
                print(f"seed lock {action}: error: --dangerous flag required")
                print("Use --dry-run to preview changes")
                return 1

            try:
                if args.dry_run:
                    result = switch_version(base, version, dry_run=True)
                    plan = result["plan"]
                    print(f"DRY RUN - Preview {action} to {result['version']}\n")
                    print(plan.to_text())
                else:
                    result = switch_version(base, version, dangerous=True)
                    print(f"Switched to version: {version}")
                    summary = Summary(
                        created=result.get("created", 0),
                        updated=result.get("updated", 0),
                        deleted=result.get("deleted", 0),
                        skipped=result.get("skipped", 0),
                    )
                    print(render_summary(summary))
                return 0
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except Exception as e:
                log.error(f"Error during {action}: {e}")
                return 1

        if action == "watch":
            try:
                watch(base, interval=args.interval)
            except RuntimeError as e:
                print(f"Error: {e}")
                return 1
            except KeyboardInterrupt:
                pass
            return 0

        print(f"Unknown lock action: {action}")
        return 1

    # ---------------- TEMPLATES ----------------
    if args.cmd == "templates":
        from seed_cli.template_registry import (
            list_templates,
            get_template,
            get_template_spec_path,
            add_template,
            add_local_template,
            remove_template,
            list_versions as list_template_versions,
            add_version,
            set_current_version,
            lock_template,
            unlock_template,
            parse_github_url,
        )
        import time as time_module
        from datetime import datetime

        action = getattr(args, "templates_action", None)

        # Default: list if no action specified
        if not action or action == "list":
            templates = list_templates()
            if not templates:
                print("No templates stored.")
                print("\nUse 'seed templates add <github_url>' to add a template.")
                return 0

            print("Stored templates:\n")
            for tmpl in templates:
                locked_str = " [LOCKED]" if tmpl.locked else ""
                print(f"  {tmpl.name}{locked_str}")
                print(f"    Version: {tmpl.current_version} ({len(tmpl.versions)} total)")
                print(f"    Source: {tmpl.source}")
                created = datetime.fromtimestamp(tmpl.created_at).strftime("%Y-%m-%d %H:%M")
                print(f"    Created: {created}")
                print()
            return 0

        if action == "add":
            source = args.source
            name = getattr(args, "name", None)
            version = getattr(args, "version", None)

            try:
                # Check if it's a GitHub URL or local file
                if parse_github_url(source):
                    meta = add_template(source, name=name, version=version)
                    print(f"Added template: {meta.name}")
                    print(f"  Version: {meta.current_version}")
                    print(f"  Source: {meta.source}")
                else:
                    # Local file
                    if not name:
                        name = Path(source).stem
                    meta = add_local_template(source, name, version=version)
                    print(f"Added template: {meta.name}")
                    print(f"  Version: {meta.current_version}")
                    print(f"  Source: {meta.source}")
                return 0
            except ValueError as e:
                print(f"Error: {e}")
                return 1
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except RuntimeError as e:
                print(f"Error: {e}")
                return 1
            except Exception as e:
                log.error(f"Error adding template: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                return 1

        if action == "remove":
            name = args.name
            try:
                if remove_template(name):
                    print(f"Removed template: {name}")
                else:
                    print(f"Template not found: {name}")
                    return 1
                return 0
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        if action == "use":
            name = args.name
            version = getattr(args, "version", None)
            use_base = Path(getattr(args, "base", ".")).resolve()
            use_vars = parse_vars(getattr(args, "vars", []))
            yes = getattr(args, "yes", False)
            dry_run = getattr(args, "dry_run", False)

            # Get template spec path
            spec_path = get_template_spec_path(name, version)
            if not spec_path:
                print(f"Template not found: {name}")
                if version:
                    print(f"Version: {version}")
                return 1

            meta = get_template(name)
            used_version = version or meta.current_version
            print(f"Using template: {name} ({used_version})")
            print(f"Base directory: {use_base}")
            print()

            try:
                # Parse spec and build plan
                _, nodes = parse_spec_file(str(spec_path), use_vars, use_base, plugins, context)

                for p in plugins:
                    p.after_parse(nodes, context)

                for p in plugins:
                    p.before_plan(nodes, context)

                plan = build_plan(
                    nodes,
                    use_base,
                    ignore=args.ignore,
                    allow_delete=False,
                    targets=args.targets,
                    target_mode=args.target_mode,
                )

                for p in plugins:
                    p.after_plan(plan, context)

                # Show plan
                print(plan.to_text())

                if dry_run:
                    print("\nDRY RUN - No changes applied")
                    return 0

                # Prompt for confirmation unless --yes
                if not yes:
                    response = input("\nApply? [y/N]: ").strip().lower()
                    if response not in ("y", "yes"):
                        print("Aborted.")
                        return 0

                # Apply the template
                result = apply(
                    str(spec_path),
                    use_base,
                    plugins=plugins,
                    dry_run=False,
                    vars=use_vars,
                    ignore=args.ignore,
                    targets=args.targets,
                    target_mode=args.target_mode,
                )

                # Extract extra fields before creating Summary
                snapshot_id = result.pop("snapshot_id", None)
                spec_version = result.pop("spec_version", None)
                result_spec_path = result.pop("spec_path", None)
                summary = Summary(**result)
                print(render_summary(summary))
                if spec_version:
                    print(f"\nSpec captured: v{spec_version} ({result_spec_path})")
                if snapshot_id:
                    print(f"Snapshot created: {snapshot_id}")
                    print("Use 'seed revert' to undo changes")
                return 0
            except Exception as e:
                log.error(f"Error using template: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                return 1

        if action == "show":
            name = args.name
            version = getattr(args, "version", None)

            spec_path = get_template_spec_path(name, version)
            if not spec_path:
                print(f"Template not found: {name}")
                if version:
                    print(f"Version: {version}")
                return 1

            meta = get_template(name)
            used_version = version or meta.current_version
            print(f"# Template: {name} ({used_version})")
            print(f"# Source: {meta.source}")
            locked_str = " [LOCKED]" if meta.locked else ""
            print(f"# Status:{locked_str}")
            print()
            print(spec_path.read_text())
            return 0

        if action == "lock":
            name = args.name
            version = getattr(args, "version", None)
            do_unlock = getattr(args, "unlock", False)

            try:
                if do_unlock:
                    unlock_template(name)
                    print(f"Unlocked template: {name}")
                else:
                    lock_template(name, version)
                    msg = f"Locked template: {name}"
                    if version:
                        msg += f" at version {version}"
                    print(msg)
                return 0
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        if action == "versions":
            name = args.name
            add_path = getattr(args, "add", None)
            version_name = getattr(args, "version_name", None)
            set_current = getattr(args, "set_current", None)

            meta = get_template(name)
            if not meta:
                print(f"Template not found: {name}")
                return 1

            # Add a new version
            if add_path:
                try:
                    new_version = add_version(name, add_path, version_name)
                    print(f"Added version: {new_version}")
                    return 0
                except (ValueError, FileNotFoundError) as e:
                    print(f"Error: {e}")
                    return 1

            # Set current version
            if set_current:
                try:
                    set_current_version(name, set_current)
                    print(f"Set current version: {set_current}")
                    return 0
                except ValueError as e:
                    print(f"Error: {e}")
                    return 1

            # List versions
            versions = list_template_versions(name)
            if not versions:
                print(f"No versions found for template: {name}")
                return 1

            print(f"Versions for template '{name}':\n")
            for v, path in versions:
                current = " (current)" if v == meta.current_version else ""
                print(f"  {v}{current}")
            print(f"\nTotal: {len(versions)} versions")
            return 0

        print(f"Unknown templates action: {action}")
        return 1

    # ---------------- HOOKS (git) ----------------
    if args.cmd == "hooks":
        if args.action == "install":
            hooks = args.hook or ["pre-commit"]
            for h in hooks:
                install_git_hook(base, h)
                print(f"Installed git hook: {h}")
        return 0

    # ---------------- UTILS ----------------
    if args.cmd == "utils":
        from seed_cli.utils import extract_tree_from_image, has_image_support

        if args.util_action == "extract-tree":
            # Check if image support is available
            if not has_image_support():
                print("Error: Image extraction requires optional dependencies.")
                print("Please install: pip install seed-cli[image]")
                return 1

            image_path = Path(args.image)
            output_path = Path(args.out) if args.out else None
            vars_dict = parse_vars(getattr(args, "vars", []))

            try:
                result_path = extract_tree_from_image(
                    image_path,
                    output_path,
                    vars=vars_dict if vars_dict else None,
                    raw=getattr(args, "raw", False),
                )
                if getattr(args, "raw", False):
                    print(f"Successfully extracted raw OCR text to: {result_path}")
                else:
                    print(f"Successfully extracted tree structure to: {result_path}")
                return 0
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
            except RuntimeError as e:
                print(f"Error: {e}")
                return 1
            except Exception as e:
                log.error(f"Error extracting tree from image: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                return 1

        if args.util_action == "state-lock":
            # Execution state lock management (moved from top-level lock)
            from seed_cli.state.local import LocalStateBackend
            import time as time_module

            backend = LocalStateBackend(base)
            lock_info = backend.lock_status()

            if args.force_unlock:
                if not lock_info:
                    print("No execution lock found to unlock.")
                    return 0
                backend.force_unlock()
                print("Execution lock force-unlocked.")
                return 0

            if not lock_info:
                print("No execution lock found.")
                return 0

            if args.renew:
                lock_id = lock_info.get("lock_id")
                if not lock_id:
                    print("Error: Cannot renew lock - no lock_id found")
                    return 1
                try:
                    backend.renew_lock(lock_id, DEFAULT_TTL)
                    print(f"Lock renewed: {lock_id}")
                except Exception as e:
                    print(f"Error renewing lock: {e}")
                    return 1
            else:
                # Show lock status
                lock_id = lock_info.get("lock_id", "unknown")
                pid = lock_info.get("pid", "unknown")
                created_at = lock_info.get("created_at", 0)
                expires_at = lock_info.get("expires_at", 0)

                if expires_at:
                    remaining = max(0, int(expires_at - time_module.time()))
                    print(f"Lock ID: {lock_id}")
                    print(f"PID: {pid}")
                    print(f"Created: {time_module.ctime(created_at) if created_at else 'unknown'}")
                    print(f"Expires in: {remaining} seconds")
                else:
                    print(f"Lock ID: {lock_id}")
                    print(f"PID: {pid}")
            return 0

        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
