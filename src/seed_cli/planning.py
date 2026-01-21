

"""seed_cli.planning

Deterministic planning engine that compares a *spec* (list of Nodes) to the
filesystem and produces an execution plan (PlanResult) consisting of PlanSteps.

Implemented behaviors (scaffold → real):

- Adds: missing directories/files in spec produce mkdir/create steps
- Changes:
  - type mismatch (file vs dir) -> update step
  - checksum drift (if checksum recorded) -> update step
  - checksum drift on @manual -> skip step (protected)
- Deletes:
  - extras in filesystem become delete steps if allow_delete=True
  - otherwise become skip steps with a note
- Targets:
  - When targets are provided, only emit steps within targets
  - Target closure: ensure parent directories needed for targeted paths exist
- Ordering:
  - mkdir parents first (increasing depth)
  - create/update/skip next
  - delete last (deeper first)

This module is used as the single source of truth for apply/sync execution.
"""

from dataclasses import dataclass
from seed_cli.ui import format_step
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import fnmatch

from .parsers import Node
from .checksums import load_checksums, sha256
from .utils.dir import _detect_single_root_dir, _strip_prefix_index, _strip_prefix_checks, _normalize_spec_index
from .logging import get_logger

log = get_logger("planning")


DEFAULT_IGNORE = [
    ".git/**",
    ".seed/**",
    "**/.DS_Store",
    "**/__pycache__/**",
]


@dataclass
class PlanStep:
    op: str  # mkdir|create|update|delete|skip
    path: str
    reason: str
    annotation: Optional[str] = None  # generated|manual|None
    depends_on: Optional[List[str]] = None
    note: Optional[str] = None


@dataclass
class PlanResult:
    steps: List[PlanStep]
    add: int
    change: int
    delete: int
    delete_skipped: int

    def to_text(self, show_skip: bool = True) -> str:
        from .ui import format_step

        lines: List[str] = []
        lines.append(
            f"Plan: {self.add} to add, {self.change} to change, {self.delete} to delete"
            + (f" ({self.delete_skipped} deletions skipped)" if self.delete_skipped else "")
        )
        lines.append("")
        lines.append("Actions:")
        for s in self.steps:
            if not show_skip and s.op == "skip":
                continue
            lines.append(format_step(s))
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "summary": {
                "add": self.add,
                "change": self.change,
                "delete": self.delete,
                "delete_skipped": self.delete_skipped,
            },
            "steps": [s.__dict__ for s in self.steps],
        }


def _norm(p: Path) -> str:
    if isinstance(p, str):
        return p.lstrip("./")
    return p.as_posix().lstrip("./")


def _ignored(rel: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _target_match(rel: str, targets: List[str], mode: str) -> bool:
    if not targets:
        return True
    rel = rel.rstrip("/")
    norm_targets = [t.rstrip("/").lstrip("./") for t in targets if t]
    if mode == "exact":
        return any(rel == t for t in norm_targets)
    return any(rel == t or rel.startswith(t + "/") for t in norm_targets)


def _parents(rel: str) -> List[str]:
    p = Path(rel)
    out: List[str] = []
    for i in range(1, len(p.parts)):
        out.append(Path(*p.parts[:i]).as_posix())
    return out


def _fs_index(base: Path, ignore: List[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    if not base.exists():
        return idx
    for p in base.rglob("*"):
        rel = _norm(p.relative_to(base))
        if not rel or _ignored(rel, ignore):
            continue
        idx[rel] = "dir" if p.is_dir() else "file"
    return idx


def _spec_index(nodes: List[Node], ignore: List[str]) -> Dict[str, Node]:
    idx: Dict[str, Node] = {}
    for n in nodes:
        rel = _norm(n.relpath)
        if not rel or _ignored(rel, ignore):
            continue
        idx[rel] = n
    return idx


def plan(
    spec_nodes: List[Node],
    base: Path,
    ignore: Optional[List[str]] = None,
    allow_delete: bool = False,
    targets: Optional[List[str]] = None,
    target_mode: str = "prefix",
) -> PlanResult:
    """Create an execution plan.

    Args:
        spec_nodes: Nodes describing desired filesystem structure.
        base: Base directory to compare against.
        ignore: Extra ignore patterns (glob). Defaults include .git and .seed.
        allow_delete: Whether to emit delete steps for filesystem extras.
        targets: Optional list of target prefixes or exact paths to scope planning.
        target_mode: 'prefix' (default) or 'exact'.

    Returns:
        PlanResult with ordered PlanStep list and summary counts.
    """
    ignore_patterns = (ignore or []) + DEFAULT_IGNORE
    targets = targets or []

    # 1. Build indices
    sidx = _spec_index(spec_nodes, ignore_patterns)

    log.debug("sidx keys (after build):")
    for k in sorted(sidx.keys()):
        log.debug("  %s", k)

    # 2. NORMALIZE spec paths against base (tilde / absolute → relative)
    sidx = _normalize_spec_index(sidx, base)

    log.debug("sidx keys (after normalize):")
    for k in sorted(sidx.keys()):
        log.debug("  %s", k)

    # 3. STEP 0 — root collapse invariant
    # If spec root resolves to ".", skip it entirely
    if "." in sidx and sidx["."].is_dir:
        sidx = {k: v for k, v in sidx.items() if k != "."}

    # 3b. Only collapse root if base directory name matches the root prefix
    roots = {Path(k).parts[0] for k in sidx if k and "/" in k}
    if len(roots) == 1:
        root = next(iter(roots))
        # Collapse if base directory name matches the root prefix
        # This handles the case where we're inside the root directory
        if base.name == root and root in sidx and sidx[root].is_dir:
            sidx = _strip_prefix_index(sidx, root)

    # 4. Now build filesystem index
    fidx = _fs_index(base, ignore_patterns)
    checks = load_checksums(base)  # {rel: {sha256:..., annotation:...}}

    # 5. Everything below stays EXACTLY the same
    steps: List[PlanStep] = []
    add = change = delete = delete_skipped = 0

    # Spec-driven ops (add/change/drift)
    for rel, node in sorted(sidx.items(), key=lambda kv: kv[0]):
        if not _target_match(rel, targets, target_mode):
            continue

        stype = "dir" if node.is_dir else "file"
        ftype = fidx.get(rel)

        deps: List[str] = []
        parent = str(Path(rel).parent) if "/" in rel else ""
        if parent and parent != ".":
            deps.append(parent)

        if ftype is None:
            if stype == "dir":
                steps.append(PlanStep("mkdir", rel, "missing", annotation=node.annotation, depends_on=deps or None))
            else:
                steps.append(PlanStep("create", rel, "missing", annotation=node.annotation, depends_on=deps or None))
            add += 1
            continue

        if ftype != stype:
            steps.append(
                PlanStep(
                    "update",
                    rel,
                    f"type_mismatch(fs={ftype},spec={stype})",
                    annotation=node.annotation,
                    depends_on=deps or None,
                )
            )
            change += 1
            continue

        if stype == "file":
            target = base / rel
            if target.exists() and target.is_file():
                current = sha256(target)
                recorded = checks.get(rel, {}).get("sha256")
                if recorded and recorded != current:
                    if node.annotation == "manual":
                        steps.append(PlanStep("skip", rel, "checksum_drift", annotation="manual", note="protected (@manual)"))
                    else:
                        steps.append(PlanStep("update", rel, "checksum_drift", annotation=node.annotation))
                        change += 1

    # Ensure parent dirs exist in plan (always, not just for targets)
    required_parents: Set[str] = set()
    for s in steps:
        required_parents.update(_parents(s.path))

    # add mkdirs for missing parents (even if not in spec list)
    for parent in sorted(required_parents, key=lambda p: (len(Path(p).parts), p)):
        if fidx.get(parent) == "dir":
            continue
        if any(st.op == "mkdir" and st.path == parent for st in steps):
            continue
        deps = _parents(parent)
        steps.append(PlanStep("mkdir", parent, "missing_parent", depends_on=deps or None))
        add += 1

    present = {s.path for s in steps}
    for s in steps:
        if s.depends_on:
            s.depends_on = [d for d in s.depends_on if d in present]

    # Extras: delete/skip
    extras = [rel for rel in fidx.keys() if rel not in sidx]
    extras.sort(key=lambda r: len(Path(r).parts), reverse=True)
    for rel in extras:
        if targets and not _target_match(rel, targets, target_mode):
            continue
        if allow_delete:
            steps.append(PlanStep("delete", rel, "extra"))
            delete += 1
        else:
            steps.append(PlanStep("skip", rel, "extra", note="SKIPPED: deletions not allowed"))
            delete_skipped += 1

    # Deduplicate (defensive) and order
    seen: Set[str] = set()
    uniq: List[PlanStep] = []
    for s in steps:
        if s.path in seen:
            continue
        seen.add(s.path)
        uniq.append(s)

    def rank(s: PlanStep) -> Tuple[int, int, str]:
        depth = len(Path(s.path).parts)
        if s.op == "mkdir":
            return (0, depth, s.path)
        if s.op == "delete":
            return (2, -depth, s.path)
        return (1, depth, s.path)

    uniq.sort(key=rank)

    return PlanResult(steps=uniq, add=add, change=change, delete=delete, delete_skipped=delete_skipped)
