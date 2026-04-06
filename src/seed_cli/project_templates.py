"""Helpers for project-local template registration and lookup."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, TYPE_CHECKING

from .logging import get_logger

if TYPE_CHECKING:
    from .parsers import Node


log = get_logger("project_templates")

SEED_DIR_NAME = ".seed"
PROJECT_TEMPLATES_DIR_NAME = "templates"
PROJECT_TEMPLATE_GROUP = "project"


@dataclass
class ProjectTemplateRegistrationResult:
    mirrored_spec: Path | None
    project_templates: list[Path]
    deleted_paths: list[Path]

    @property
    def changed(self) -> bool:
        return bool(self.mirrored_spec or self.project_templates or self.deleted_paths)


def _iter_ancestors(start: Path) -> List[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    return [current, *current.parents]


def find_project_root(start: Path) -> Path:
    """Find the nearest project root for project-local .seed data."""
    ancestors = _iter_ancestors(start)

    for candidate in ancestors:
        if (candidate / SEED_DIR_NAME).is_dir():
            return candidate

    for candidate in ancestors:
        if (candidate / ".git").exists():
            return candidate

    return ancestors[0]


def get_project_seed_dir(start: Path, *, create: bool = False) -> Path:
    """Return the project .seed directory, walking up from start."""
    seed_dir = find_project_root(start) / SEED_DIR_NAME
    if create:
        seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def get_project_templates_dir(start: Path, *, create: bool = False) -> Path:
    """Return the project-local template directory under .seed/."""
    templates_dir = get_project_seed_dir(start, create=create) / PROJECT_TEMPLATES_DIR_NAME
    if create:
        templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def get_registered_project_templates_dir(start: Path, *, create: bool = False) -> Path:
    """Return the project-template namespace under the nearest project .seed/."""
    project_templates_dir = get_project_templates_dir(start, create=create) / PROJECT_TEMPLATE_GROUP
    if create:
        project_templates_dir.mkdir(parents=True, exist_ok=True)
    return project_templates_dir


def get_local_project_templates_dir(start: Path, *, create: bool = False) -> Path:
    """Return the local project-template directory rooted exactly at start/.seed/."""
    local_templates_dir = start.resolve() / SEED_DIR_NAME / PROJECT_TEMPLATES_DIR_NAME / PROJECT_TEMPLATE_GROUP
    if create:
        local_templates_dir.mkdir(parents=True, exist_ok=True)
    return local_templates_dir


def has_template_subtree(nodes: Iterable["Node"]) -> bool:
    """Return True when the spec contains a template dir with children."""
    return bool(_template_subtree_roots(nodes))


def _render_node_name(node: "Node") -> str:
    name = node.relpath.name
    if node.is_dir:
        name += "/"
    if node.optional:
        name += " ?"
    if node.annotation and not node.annotation.startswith("template:"):
        name += f" @{node.annotation}"
    if node.comment:
        name += f" ({node.comment})"
    return name


def _template_subtree_roots(nodes: Iterable["Node"]) -> list[Path]:
    node_list = list(nodes)
    template_paths = {
        node.relpath
        for node in node_list
        if (node.annotation or "").startswith("template:")
    }
    if not template_paths:
        return []

    return sorted(
        {
            template_path
            for template_path in template_paths
            if any(template_path in node.relpath.parents for node in node_list)
        },
        key=lambda path: path.as_posix(),
    )


def _iter_template_subtrees(nodes: Iterable["Node"]) -> Iterator[tuple[str, Path, Path, list["Node"]]]:
    node_list = list(nodes)
    template_roots = set(_template_subtree_roots(node_list))
    for node in node_list:
        annotation = node.annotation or ""
        if not annotation.startswith("template:") or node.relpath not in template_roots:
            continue

        template_path = node.relpath
        subtree = [
            child
            for child in node_list
            if child.relpath == template_path or template_path in child.relpath.parents
        ]

        yield annotation.split(":", 1)[1], template_path, template_path.parent, subtree


def prune_project_template_nodes(nodes: Iterable["Node"]) -> list["Node"]:
    """Drop extracted template subtree nodes from concrete apply planning."""
    node_list = list(nodes)
    template_roots = set(_template_subtree_roots(node_list))
    if not template_roots:
        return node_list

    return [
        node
        for node in node_list
        if not any(node.relpath == template_root or template_root in node.relpath.parents for template_root in template_roots)
    ]


def _rebase_template_subtree(nodes: Iterable["Node"], parent: Path) -> list["Node"]:
    rebased: list["Node"] = []
    for node in nodes:
        relpath = node.relpath if parent == Path(".") else node.relpath.relative_to(parent)
        rebased.append(type(node)(
            relpath=relpath,
            is_dir=node.is_dir,
            comment=node.comment,
            annotation=node.annotation,
            optional=node.optional,
        ))
    return rebased


def _render_tree_text(nodes: Iterable["Node"]) -> str:
    node_list = list(nodes)
    children_by_parent: dict[Path, list["Node"]] = {}
    for node in node_list:
        children_by_parent.setdefault(node.relpath.parent, []).append(node)

    lines = ["."]

    def walk(parent: Path, prefix: str = "") -> None:
        children = sorted(
            children_by_parent.get(parent, []),
            key=lambda node: (node.relpath.as_posix(), 0 if node.is_dir else 1),
        )
        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            branch = "└── " if is_last else "├── "
            lines.append(f"{prefix}{branch}{_render_node_name(child)}")
            if child.is_dir:
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(child.relpath, child_prefix)

    walk(Path("."))
    return "\n".join(lines) + "\n"


def _write_project_template_subtrees(nodes: Iterable["Node"], start: Path) -> list[Path]:
    written: list[Path] = []

    for template_name, _, parent_relpath, subtree in _iter_template_subtrees(nodes):
        parent_dir = start.resolve() if parent_relpath == Path(".") else (start.resolve() / parent_relpath)
        templates_dir = get_local_project_templates_dir(parent_dir, create=True)
        destination = templates_dir / f"{template_name}.tree"
        rebased_nodes = _rebase_template_subtree(subtree, parent_relpath)
        content = _render_tree_text(rebased_nodes)
        destination.write_text(content, encoding="utf-8")
        written.append(destination)
        log.debug("Registered project subtree template %s -> %s", template_name, destination)

    return written


def materialized_project_template_paths(nodes: Iterable["Node"], start: Path) -> list[Path]:
    """Return literal template placeholder paths that should not exist after registration."""
    materialized_paths: list[Path] = []
    seen: set[Path] = set()

    for _, template_path, _, _ in _iter_template_subtrees(nodes):
        candidate = (start.resolve() / template_path).resolve()
        if candidate not in seen:
            seen.add(candidate)
            materialized_paths.append(candidate)

    return materialized_paths


def delete_materialized_project_templates(nodes: Iterable["Node"], start: Path) -> list[Path]:
    """Delete literal placeholder directories/files such as <name>/ created by older apply flows."""
    deleted: list[Path] = []

    for target in materialized_project_template_paths(nodes, start):
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted.append(target)
        log.debug("Deleted materialized project template path %s", target)

    return deleted


def register_project_template(spec_path: Path | str, nodes: Iterable["Node"], start: Path) -> Path | None:
    """Mirror a .tree spec into the project .seed directory."""
    spec = Path(spec_path).resolve()
    if not spec.is_file():
        return None

    if spec.suffix != ".tree":
        return None

    seed_dir = get_project_seed_dir(start, create=True)
    try:
        spec.relative_to(seed_dir)
        return spec
    except ValueError:
        pass

    project_root = seed_dir.parent
    templates_dir = get_project_templates_dir(project_root, create=True)

    try:
        relative_spec = spec.relative_to(project_root)
        if relative_spec.parts and relative_spec.parts[0] == SEED_DIR_NAME:
            return spec
        destination = templates_dir / relative_spec
    except ValueError:
        destination = templates_dir / spec.name

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec, destination)
    log.debug("Registered project template %s -> %s", spec, destination)
    return destination


def register_spec_project_templates(
    spec_path: Path | str,
    nodes: Iterable["Node"],
    start: Path,
    *,
    cleanup_materialized: bool = False,
) -> ProjectTemplateRegistrationResult:
    """Register a template-capable spec and optionally clean up stale literal template paths."""
    node_list = list(nodes)
    mirrored_spec = register_project_template(spec_path, node_list, start)
    project_templates = _write_project_template_subtrees(node_list, start) if has_template_subtree(node_list) else []
    deleted_paths = delete_materialized_project_templates(node_list, start) if cleanup_materialized else []
    return ProjectTemplateRegistrationResult(
        mirrored_spec=mirrored_spec,
        project_templates=project_templates,
        deleted_paths=deleted_paths,
    )


def resolve_project_template_path(template_path: str, start: Path) -> Path:
    """Resolve a template path, treating .seed/... as project-root relative."""
    def resolve_tree_candidate(candidate: Path) -> Path | None:
        if candidate.is_file():
            return candidate

        if candidate.is_dir():
            nested_candidate = candidate / f"{candidate.name}.tree"
            if nested_candidate.is_file():
                return nested_candidate

        if candidate.suffix != ".tree":
            tree_candidate = candidate.with_suffix(".tree")
            if tree_candidate.is_file():
                return tree_candidate

        return None

    raw = Path(template_path)
    if raw.is_absolute():
        resolved = resolve_tree_candidate(raw)
        return resolved or raw

    if raw.parts and raw.parts[0] == SEED_DIR_NAME:
        seed_dir = get_project_seed_dir(start)
        candidate = seed_dir / Path(*raw.parts[1:])
        resolved = resolve_tree_candidate(candidate)
        return resolved or candidate

    direct_candidate = (start / raw).resolve()
    resolved = resolve_tree_candidate(direct_candidate)
    if resolved:
        return resolved

    candidate = get_project_templates_dir(start) / raw
    resolved = resolve_tree_candidate(candidate)
    if resolved:
        return resolved

    return direct_candidate


def iter_registered_project_template_dirs(start: Path) -> Iterator[Path]:
    """Yield local project-template directories from nearest scope outward."""
    seen: set[Path] = set()
    for candidate in _iter_ancestors(start):
        directory = candidate / SEED_DIR_NAME / PROJECT_TEMPLATES_DIR_NAME / PROJECT_TEMPLATE_GROUP
        if directory.is_dir():
            resolved = directory.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def list_registered_project_templates(start: Path) -> List[Path]:
    """List all registered project templates visible from start."""
    templates: List[Path] = []
    seen: set[Path] = set()
    for directory in iter_registered_project_template_dirs(start):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    templates.append(resolved)
    return templates


def resolve_registered_project_template(template_name: str, start: Path) -> Path:
    """Resolve a registered project template by name from nearest scope outward."""
    raw = Path(template_name)
    candidate_names = [raw]
    if raw.suffix != ".tree":
        candidate_names.insert(0, raw.with_suffix(".tree"))

    for directory in iter_registered_project_template_dirs(start):
        for candidate_name in candidate_names:
            candidate = directory / candidate_name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(template_name)


def complete_registered_project_template_names(prefix: str, start: Path) -> List[str]:
    """List visible registered project template names for argcomplete."""
    suggestions: set[str] = set()
    for path in list_registered_project_templates(start):
        for directory in iter_registered_project_template_dirs(start):
            try:
                rel = path.relative_to(directory)
                break
            except ValueError:
                continue
        else:
            continue

        suggestion = rel.as_posix()
        if rel.suffix == ".tree":
            suggestion = rel.with_suffix("").as_posix()
        suggestions.add(suggestion)

    normalized_prefix = prefix or ""
    return sorted(s for s in suggestions if not normalized_prefix or s.startswith(normalized_prefix))


def complete_project_template_paths(prefix: str, start: Path) -> List[str]:
    """List project-local template path suggestions for argcomplete."""
    seed_dir = get_project_seed_dir(start)
    templates_dir = seed_dir / PROJECT_TEMPLATES_DIR_NAME
    if not templates_dir.exists():
        return []

    suggestions = {f"{SEED_DIR_NAME}/{PROJECT_TEMPLATES_DIR_NAME}/"}
    for path in templates_dir.rglob("*"):
        rel = path.relative_to(seed_dir)
        rendered = Path(SEED_DIR_NAME) / rel
        suggestion = rendered.as_posix()
        if path.is_dir():
            suggestion += "/"
        suggestions.add(suggestion)

    normalized_prefix = prefix or ""
    return sorted(s for s in suggestions if not normalized_prefix or s.startswith(normalized_prefix))
