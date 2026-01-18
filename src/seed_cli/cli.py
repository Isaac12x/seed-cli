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
from seed_cli.image import parse_image, extract_text_from_image_cv2
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
    
    # For image files, parse directly (includes/vars handled by parse_image -> parse_any)
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

    # diff
    sd = sub.add_parser(
        "diff",
        description="Compare spec with filesystem and show missing, extra, and drifted paths",
        help="Compare spec with filesystem state",
    )
    sd.add_argument("spec", help="Spec file")
    sd.add_argument("--base", default=".", help="Base directory (default: current directory)")

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

    # lock
    sl = sub.add_parser(
        "lock",
        description="Manual lock control via CLI",
        help="Manage state locks",
    )
    sl.add_argument("--base", default=".", help="Base directory (default: current directory)")
    sl.add_argument("--renew", action="store_true", help="Renew existing lock")
    sl.add_argument("--force-unlock", action="store_true", help="Force unlock (use if process crashed)")

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

    return p


def main(argv=None) -> int:
    parser = build_parser()
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

            print(plan.to_text())
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
                )

            run_hooks(hooks, f"post_{args.cmd}", strict=True, cwd=base)
            summary = Summary(**result)
            print(render_summary(summary))
            return 0
        except Exception as e:
            log.error(f"Error {args.cmd}: {e}")
            return 1
  
    # ---------------- DIFF ----------------
    if args.cmd == "diff":
        _, nodes = parse_spec_file(args.spec, vars, base, plugins, context)
        res = diff(nodes, base)
        print(render_list("Missing", res.missing))
        print(render_list("Extra", res.extra))
        print(render_list("Type Mismatch", res.type_mismatch))
        print(render_list("Drift", res.drift))
        return 0 if res.is_clean() else 1

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

    # ---------------- LOCK ----------------
    if args.cmd == "lock":
        from seed_cli.state.local import LocalStateBackend
        import time
        
        backend = LocalStateBackend(base)
        lock_info = backend.lock_status()
        
        if args.force_unlock:
            if not lock_info:
                print("No lock found to unlock.")
                return 0
            backend.force_unlock()
            print("Lock force-unlocked.")
            return 0
        
        if not lock_info:
            print("No lock found.")
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
                remaining = max(0, int(expires_at - time.time()))
                print(f"Lock ID: {lock_id}")
                print(f"PID: {pid}")
                print(f"Created: {time.ctime(created_at) if created_at else 'unknown'}")
                print(f"Expires in: {remaining} seconds")
            else:
                print(f"Lock ID: {lock_id}")
                print(f"PID: {pid}")
        return 0

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
        
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
