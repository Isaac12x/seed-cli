
"""seed_cli.match

Match filesystem to spec, creating missing items and deleting extras.

Key difference from sync:
- Respects `...` markers in spec that allow extra files in specific directories
- Directories containing `...` will not have their extra contents deleted
- Supports template directories with `<varname>/` syntax for repeating structures

Usage in spec:
    src/
    ├── lib/
    │   ├── index.ts
    │   └── ...          # extra files allowed in lib/
    ├── components/
    │   └── Button.tsx   # no extras allowed here
    └── ...              # extra files allowed at root level

Template directories:
    files/
    ├── <version_id>/    # matches any directory name
    │   ├── data.json    # each matched dir must have this structure
    │   └── meta/
    └── ...
"""

from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple
import fnmatch
import re

from .parsers import Node
from .planning import plan as build_plan, PlanResult, PlanStep, DEFAULT_IGNORE, _norm
from .executor import execute_plan
from .state.local import LocalStateBackend
from .lock_heartbeat import LockHeartbeat

_TEMPLATE_PATTERN = re.compile(r"<[a-zA-Z_][a-zA-Z0-9_]*>")


def _extract_extras_allowed(nodes: List[Node]) -> Set[str]:
    """Extract directories that allow extra files (marked with `...`).

    Returns set of directory paths that allow extras.
    """
    extras_allowed: Set[str] = set()

    for node in nodes:
        path = node.relpath.as_posix()
        # Check for ... marker nodes (created by parser)
        if path.endswith("/...") or path == "...":
            # Parent directory allows extras
            parent = str(Path(path).parent)
            if parent == ".":
                extras_allowed.add("")  # root level
            else:
                extras_allowed.add(parent)
        # Also check annotation
        elif node.annotation == "extras":
            parent = str(Path(path).parent)
            if parent == ".":
                extras_allowed.add("")
            else:
                extras_allowed.add(parent)

    return extras_allowed


def _extract_templates(nodes: List[Node]) -> Dict[str, List[Node]]:
    """Extract template directory patterns and their child structures.

    Returns dict mapping template path patterns to their child nodes.
    E.g., {"files/<version_id>": [Node("files/<version_id>/data.json"), ...]}
    """
    templates: Dict[str, List[Node]] = {}

    for node in nodes:
        path = node.relpath.as_posix()
        annotation = node.annotation or ""

        if annotation.startswith("template:"):
            # This is a template directory marker
            templates[path] = []

    # Now collect children of each template
    for node in nodes:
        path = node.relpath.as_posix()
        for template_path in templates:
            if path.startswith(template_path + "/"):
                templates[template_path].append(node)

    return templates


def _expand_templates(
    nodes: List[Node],
    templates: Dict[str, List[Node]],
    base: Path,
) -> List[Node]:
    """Expand template patterns to match actual directories on filesystem.

    For each template pattern like `files/<version_id>/`, find all actual
    directories at that level and create concrete nodes for each.
    """
    if not templates:
        return nodes

    expanded: List[Node] = []
    template_paths = set(templates.keys())

    # Add non-template nodes as-is (but skip template children, we'll expand them)
    for node in nodes:
        path = node.relpath.as_posix()

        # Skip template marker nodes
        if path in template_paths:
            continue

        # Skip children of templates (we'll expand them)
        is_template_child = any(path.startswith(tp + "/") for tp in template_paths)
        if is_template_child:
            continue

        # Check if path contains a template variable
        if _TEMPLATE_PATTERN.search(path):
            continue

        expanded.append(node)

    # Expand each template
    for template_path, children in templates.items():
        # Find the parent directory of the template
        parent_path = str(Path(template_path).parent)
        if parent_path == ".":
            parent_dir = base
        else:
            parent_dir = base / parent_path

        if not parent_dir.exists():
            # Parent doesn't exist - will be created, but no dirs to expand
            continue

        # Find all actual directories at the template level
        for item in parent_dir.iterdir():
            if not item.is_dir():
                continue

            actual_name = item.name
            # Create the concrete path by replacing the template variable
            template_var = Path(template_path).name  # e.g., "<version_id>"

            # Add the directory itself
            concrete_dir_path = f"{parent_path}/{actual_name}" if parent_path != "." else actual_name
            expanded.append(Node(
                relpath=Path(concrete_dir_path),
                is_dir=True,
                comment=None,
                annotation=None,
            ))

            # Add expanded children
            for child in children:
                child_path = child.relpath.as_posix()
                # Replace template path with concrete path
                concrete_path = child_path.replace(template_path, concrete_dir_path)
                expanded.append(Node(
                    relpath=Path(concrete_path),
                    is_dir=child.is_dir,
                    comment=child.comment,
                    annotation=child.annotation if child.annotation and not child.annotation.startswith("template:") else None,
                ))

    return expanded


def _filter_templates_from_extras(extras_allowed: Set[str], templates: Dict[str, List[Node]]) -> Set[str]:
    """Add template parent directories to extras_allowed.

    Template directories implicitly allow extras at their level since
    there can be multiple instances.
    """
    result = set(extras_allowed)

    for template_path in templates:
        parent = str(Path(template_path).parent)
        if parent == ".":
            result.add("")
        else:
            result.add(parent)

    return result


def _is_under_extras_allowed(path: str, extras_allowed: Set[str]) -> bool:
    """Check if a path is under a directory that allows extras."""
    if "" in extras_allowed:
        # Root allows extras - check if path is a direct child of root
        if "/" not in path:
            return True

    path_parts = Path(path).parts
    # Check each parent level
    for i in range(len(path_parts)):
        parent = "/".join(path_parts[:i]) if i > 0 else ""
        if parent in extras_allowed:
            return True

    return False


def _filter_plan_for_extras(plan: PlanResult, extras_allowed: Set[str]) -> PlanResult:
    """Filter out delete operations for paths under extras-allowed directories."""
    filtered_steps: List[PlanStep] = []
    delete_removed = 0

    for step in plan.steps:
        if step.op == "delete":
            if _is_under_extras_allowed(step.path, extras_allowed):
                # Skip this delete - it's in an extras-allowed location
                delete_removed += 1
                continue
        filtered_steps.append(step)

    return PlanResult(
        steps=filtered_steps,
        add=plan.add,
        change=plan.change,
        delete=plan.delete - delete_removed,
        delete_skipped=plan.delete_skipped + delete_removed,
    )


def _filter_nodes_for_planning(nodes: List[Node]) -> List[Node]:
    """Remove marker nodes before planning (they're not real files).

    Removes:
    - ... marker nodes (extras allowed)
    - Template marker nodes (template:<varname>)
    - Paths containing template variables (<varname>)
    """
    filtered = []
    for n in nodes:
        path = n.relpath.as_posix()
        # Skip ... markers
        if path.endswith("..."):
            continue
        # Skip template markers
        if n.annotation and n.annotation.startswith("template:"):
            continue
        # Skip paths with unexpanded template variables
        if _TEMPLATE_PATTERN.search(path):
            continue
        filtered.append(n)
    return filtered


def match(
    spec_path: str,
    base: Path,
    *,
    dangerous: bool,
    dry_run: bool = False,
    force: bool = False,
    gitkeep: bool = False,
    template_dir: Optional[Path] = None,
    vars: Optional[dict] = None,
    ignore: Optional[list[str]] = None,
    targets: Optional[list[str]] = None,
    target_mode: str = "prefix",
    lock: bool = True,
    lock_ttl: int = 300,
    lock_timeout: int = 30,
    lock_renew: int = 10,
    interactive: bool = True,
    skip_optional: bool = False,
    include_optional: bool = False,
) -> dict:
    """Match filesystem to spec.

    Creates missing files/directories and deletes extras.
    Respects `...` markers that allow extra files in specific directories.

    Args:
        spec_path: Path to spec file
        base: Base directory
        dangerous: Must be True to allow deletions (or use dry_run)
        dry_run: Preview changes without executing
        force: Force overwrite of existing files
        gitkeep: Create .gitkeep in empty directories
        template_dir: Directory for file templates
        vars: Template variables
        ignore: Additional ignore patterns
        targets: Target paths to scope operations
        target_mode: 'prefix' or 'exact'
        lock: Use state locking
        lock_ttl: Lock TTL in seconds
        lock_timeout: Lock acquisition timeout
        lock_renew: Lock renewal interval

    Returns:
        dict with created, updated, deleted, skipped counts
    """
    if not dangerous and not dry_run:
        raise RuntimeError(
            "match requires --dangerous flag to modify filesystem "
            "(use --dry-run to preview changes)"
        )

    base = base.resolve()

    # Parse spec
    from .parsers import parse_spec
    _, nodes = parse_spec(spec_path, vars=vars, base=base)

    # Extract directories that allow extras
    extras_allowed = _extract_extras_allowed(nodes)

    # Extract and expand templates
    templates = _extract_templates(nodes)
    if templates:
        nodes = _expand_templates(nodes, templates, base)
        # Template parent dirs implicitly allow extras
        extras_allowed = _filter_templates_from_extras(extras_allowed, templates)

    # Remove marker nodes before planning
    nodes_for_plan = _filter_nodes_for_planning(nodes)

    # Build plan with deletions enabled
    plan = build_plan(
        nodes_for_plan,
        base,
        ignore=ignore,
        allow_delete=True,
        targets=targets,
        target_mode=target_mode,
    )

    # Filter out deletions for extras-allowed directories
    plan = _filter_plan_for_extras(plan, extras_allowed)

    # Create snapshot before making changes
    snapshot_id = None
    if not dry_run and plan.steps:
        from .snapshot import create_snapshot
        snapshot_id = create_snapshot(base, plan, "match", spec_path)

    # Locking
    heartbeat = None
    lock_id = None
    backend = None

    if lock and not dry_run:
        backend = LocalStateBackend(base)
        info = backend.acquire_lock(lock_ttl, lock_timeout)
        lock_id = info.get("lock_id")
        heartbeat = LockHeartbeat(
            renew_fn=lambda: backend.renew_lock(lock_id, lock_ttl),
            interval=lock_renew,
        )
        heartbeat.start()

    try:
        result = execute_plan(
            plan,
            base,
            dangerous=True,  # We've already gated on dangerous flag
            force=force,
            dry_run=dry_run,
            gitkeep=gitkeep,
            template_dir=template_dir,
            interactive=interactive,
            skip_optional=skip_optional,
            include_optional=include_optional,
        )
    finally:
        if heartbeat:
            heartbeat.stop()
        if backend and lock_id:
            backend.release_lock(lock_id)

    # Capture versioned spec after successful execution
    if not dry_run:
        from .spec_history import capture_spec
        spec_result = capture_spec(base, ignore=ignore)
        if spec_result:
            result["spec_version"] = spec_result[0]
            result["spec_path"] = str(spec_result[1])

    if snapshot_id:
        result["snapshot_id"] = snapshot_id

    return result


def create_from_template(
    spec_path: str,
    base: Path,
    template_values: Dict[str, str],
    *,
    dry_run: bool = False,
) -> Dict:
    """Create a new instance of a template directory structure.

    Args:
        spec_path: Path to spec file containing template
        base: Base directory
        template_values: Dict mapping template variable names to values
                        e.g., {"version_id": "v3"}
        dry_run: Preview without creating

    Returns:
        dict with created count and list of paths
    """
    base = base.resolve()

    from .parsers import parse_spec
    _, nodes = parse_spec(spec_path, base=base)

    # Find template nodes and their children
    templates = _extract_templates(nodes)

    if not templates:
        raise ValueError("No template patterns (<varname>/) found in spec")

    # Find matching template for provided values
    created_paths: List[str] = []

    for template_path, children in templates.items():
        # Extract variable name from template path (e.g., "files/<version_id>" -> "version_id")
        template_name = Path(template_path).name  # e.g., "<version_id>"
        var_match = re.match(r"<([a-zA-Z_][a-zA-Z0-9_]*)>", template_name)
        if not var_match:
            continue

        var_name = var_match.group(1)
        if var_name not in template_values:
            continue

        value = template_values[var_name]

        # Calculate the parent path
        parent_path = str(Path(template_path).parent)
        if parent_path == ".":
            concrete_dir_path = value
        else:
            concrete_dir_path = f"{parent_path}/{value}"

        # Create the directory
        concrete_dir = base / concrete_dir_path
        if not dry_run:
            concrete_dir.mkdir(parents=True, exist_ok=True)
        created_paths.append(concrete_dir_path)

        # Create children
        for child in children:
            child_path = child.relpath.as_posix()
            # Replace template path with concrete path
            concrete_path = child_path.replace(template_path, concrete_dir_path)

            # Skip marker nodes
            if concrete_path.endswith("..."):
                continue

            target = base / concrete_path
            if child.is_dir:
                if not dry_run:
                    target.mkdir(parents=True, exist_ok=True)
                created_paths.append(concrete_path)
            else:
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                created_paths.append(concrete_path)

    return {
        "created": len(created_paths),
        "paths": created_paths,
        "dry_run": dry_run,
    }


def match_plan(
    spec_path: str,
    base: Path,
    *,
    vars: Optional[dict] = None,
    ignore: Optional[list[str]] = None,
    targets: Optional[list[str]] = None,
    target_mode: str = "prefix",
) -> PlanResult:
    """Generate a match plan without executing.

    Useful for previewing what match would do.
    """
    base = base.resolve()

    from .parsers import parse_spec
    _, nodes = parse_spec(spec_path, vars=vars, base=base)

    extras_allowed = _extract_extras_allowed(nodes)

    # Extract and expand templates
    templates = _extract_templates(nodes)
    if templates:
        nodes = _expand_templates(nodes, templates, base)
        extras_allowed = _filter_templates_from_extras(extras_allowed, templates)

    nodes_for_plan = _filter_nodes_for_planning(nodes)

    plan = build_plan(
        nodes_for_plan,
        base,
        ignore=ignore,
        allow_delete=True,
        targets=targets,
        target_mode=target_mode,
    )

    return _filter_plan_for_extras(plan, extras_allowed)
