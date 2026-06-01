"""Helpers for project-local template registration and lookup."""

from __future__ import annotations

import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

from .logging import get_logger
from .parsers import Node, render_node_text
from .spec_formats import (
    TREE_LIKE_SUFFIXES,
    preferred_tree_like_suffix,
    resolve_tree_like_path,
    strip_tree_like_suffix,
)


log = get_logger("project_templates")

SEED_DIR_NAME = ".seed"
PROJECT_TEMPLATES_DIR_NAME = "templates"
PROJECT_TEMPLATE_GROUP = "project"
_TEMPLATE_VAR_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>")
_BARE_TEMPLATE_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*_ID$")
_TEMPLATE_ANNOTATION_RE = re.compile(r"^template:([a-zA-Z_][a-zA-Z0-9_]*)$")


@dataclass
class ProjectTemplateRegistrationResult:
    mirrored_spec: Path | None
    project_templates: list[Path]
    deleted_paths: list[Path]

    @property
    def changed(self) -> bool:
        return bool(self.mirrored_spec or self.project_templates or self.deleted_paths)


@dataclass(frozen=True)
class TemplateSubtree:
    name: str
    relpath: Path
    parent: Path
    nodes: list[Node]


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
    """Return True when the spec contains a placeholder-backed template."""
    return bool(_template_subtree_roots(nodes))


def _annotation_template_name(annotation: str | None) -> str | None:
    if not annotation:
        return None
    match = _TEMPLATE_ANNOTATION_RE.match(annotation)
    return match.group(1) if match else None


def _placeholder_names(text: str) -> list[str]:
    names: list[str] = []
    for name in _TEMPLATE_VAR_RE.findall(text):
        if name not in names:
            names.append(name)
    if _BARE_TEMPLATE_VAR_RE.match(text) and text not in names:
        names.append(text)
    return names


def _template_filename_part(name: str) -> str:
    if name.isupper() and name.endswith("_ID"):
        name = name[:-3]
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return normalized or "template"


def _template_name_from_path(path: Path) -> str | None:
    names = _placeholder_names(path.name)
    if not names:
        return None
    return "-".join(_template_filename_part(name) for name in names)


def template_variable_names(nodes: Iterable["Node"]) -> list[str]:
    """Return unique placeholder variable names found in template annotations or paths."""
    names: list[str] = []
    for node in nodes:
        annotation_name = _annotation_template_name(node.annotation)
        if annotation_name and annotation_name not in names:
            names.append(annotation_name)
        for part in node.relpath.parts:
            for name in _placeholder_names(part):
                if name not in names:
                    names.append(name)
    return names


def path_has_template_variable(path: Path | str) -> bool:
    """Return True when a path contains an angle or bare project-template variable."""
    return any(_placeholder_names(part) for part in Path(path).parts)


def render_template_path(path: str, template_values: dict[str, str]) -> str | None:
    """Render angle placeholders and bare uppercase *_ID path segments."""
    missing_value = False

    def replace_angle(match: re.Match[str]) -> str:
        nonlocal missing_value
        name = match.group(1)
        if name not in template_values:
            missing_value = True
            return match.group(0)
        return template_values[name]

    rendered = _TEMPLATE_VAR_RE.sub(replace_angle, path)

    parts: list[str] = []
    for part in rendered.split("/"):
        if _BARE_TEMPLATE_VAR_RE.match(part):
            if part not in template_values:
                missing_value = True
                parts.append(part)
            else:
                parts.append(template_values[part])
        else:
            parts.append(part)

    return None if missing_value else "/".join(parts)


def _first_placeholder_root(path: Path) -> Path | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if _placeholder_names(part):
            return Path(*parts[: index + 1])
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _template_subtree_root_specs(nodes: Iterable["Node"]) -> list[tuple[str, Path]]:
    node_list = list(nodes)
    candidates: dict[Path, str] = {}

    for node in node_list:
        annotation_name = _annotation_template_name(node.annotation)
        if annotation_name:
            candidates[node.relpath] = _template_filename_part(annotation_name)

        placeholder_root = _first_placeholder_root(node.relpath)
        if placeholder_root:
            template_name = _template_name_from_path(placeholder_root)
            if template_name:
                candidates.setdefault(placeholder_root, template_name)

    selected: list[tuple[str, Path]] = []
    for path, name in sorted(candidates.items(), key=lambda item: (len(item[0].parts), item[0].as_posix())):
        if any(_is_relative_to(path, selected_path) and path != selected_path for _, selected_path in selected):
            continue
        if not any(_is_relative_to(node.relpath, path) for node in node_list):
            continue
        selected.append((name, path))

    return sorted(selected, key=lambda item: item[1].as_posix())


def _template_subtree_roots(nodes: Iterable["Node"]) -> list[Path]:
    return [path for _, path in _template_subtree_root_specs(nodes)]


def _iter_template_subtrees(nodes: Iterable["Node"]) -> Iterator[TemplateSubtree]:
    node_list = list(nodes)
    for template_name, template_path in _template_subtree_root_specs(node_list):
        subtree = [
            child
            for child in node_list
            if _is_relative_to(child.relpath, template_path)
        ]
        if not subtree:
            continue

        yield TemplateSubtree(
            name=template_name,
            relpath=template_path,
            parent=template_path.parent,
            nodes=subtree,
        )


def iter_template_subtrees(nodes: Iterable["Node"]) -> Iterator[TemplateSubtree]:
    """Yield inferred project-template subtrees for annotated or placeholder paths."""
    yield from _iter_template_subtrees(nodes)


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
            metadata=dict(node.metadata),
        ))
    return rebased


def _with_implicit_directory_nodes(nodes: Iterable["Node"]) -> list["Node"]:
    node_list = list(nodes)
    nodes_by_path = {node.relpath: node for node in node_list}

    for node in node_list:
        for parent in node.relpath.parents:
            if parent == Path("."):
                break
            nodes_by_path.setdefault(parent, Node(relpath=parent, is_dir=True))

    return list(nodes_by_path.values())


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
            lines.append(f"{prefix}{branch}{render_node_text(child, basename=True)}")
            if child.is_dir:
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(child.relpath, child_prefix)

    walk(Path("."))
    return "\n".join(lines) + "\n"


def _template_scope_prefix(path: Path) -> str | None:
    for part in path.parts:
        if part in ("", "."):
            continue
        if _placeholder_names(part):
            return None
        return _template_filename_part(part)
    return None


def _project_scope_template_names(subtrees: list[TemplateSubtree]) -> list[str]:
    name_counts = Counter(subtree.name for subtree in subtrees)
    used: Counter[str] = Counter()
    names: list[str] = []

    for subtree in subtrees:
        base_name = subtree.name
        if name_counts[base_name] > 1:
            prefix = _template_scope_prefix(subtree.relpath)
            if prefix:
                base_name = f"{prefix}_{base_name}"

        name = base_name
        used[name] += 1
        if used[name] > 1:
            name = f"{base_name}_{used[name]}"
        names.append(name)

    return names


def _write_template_subtree(
    subtree: TemplateSubtree,
    destination: Path,
) -> None:
    rebased_nodes = _with_implicit_directory_nodes(_rebase_template_subtree(subtree.nodes, subtree.parent))
    content = _render_tree_text(rebased_nodes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def _write_project_template_subtrees(
    nodes: Iterable["Node"],
    start: Path,
    *,
    spec_suffix: str = ".tree",
) -> list[Path]:
    written: list[Path] = []
    written_set: set[Path] = set()
    subtrees = list(_iter_template_subtrees(nodes))
    project_scope_names = _project_scope_template_names(subtrees)

    def write_once(subtree: TemplateSubtree, destination: Path) -> None:
        resolved = destination.resolve()
        if resolved in written_set:
            return
        _write_template_subtree(subtree, destination)
        written_set.add(resolved)
        written.append(destination)

    for subtree, project_scope_name in zip(subtrees, project_scope_names):
        parent_relpath = subtree.parent
        parent_dir = start.resolve() if parent_relpath == Path(".") else (start.resolve() / parent_relpath)
        templates_dir = get_local_project_templates_dir(parent_dir, create=True)
        destination = templates_dir / f"{subtree.name}{spec_suffix}"
        write_once(subtree, destination)
        log.debug("Registered project subtree template %s -> %s", subtree.name, destination)

        project_templates_dir = get_registered_project_templates_dir(start, create=True)
        project_destination = project_templates_dir / f"{project_scope_name}{spec_suffix}"
        write_once(subtree, project_destination)
        log.debug("Registered project subtree template %s -> %s", project_scope_name, project_destination)

    return written


def materialized_project_template_paths(nodes: Iterable["Node"], start: Path) -> list[Path]:
    """Return literal template placeholder paths that should not exist after registration."""
    materialized_paths: list[Path] = []
    seen: set[Path] = set()

    for subtree in _iter_template_subtrees(nodes):
        candidate = (start.resolve() / subtree.relpath).resolve()
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
    """Mirror a tree-like spec into the project .seed directory."""
    spec = Path(spec_path).resolve()
    if not spec.is_file():
        return None

    if spec.suffix.lower() not in TREE_LIKE_SUFFIXES:
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
    spec_suffix = preferred_tree_like_suffix(spec_path)
    project_templates = (
        _write_project_template_subtrees(node_list, start, spec_suffix=spec_suffix)
        if has_template_subtree(node_list)
        else []
    )
    deleted_paths = delete_materialized_project_templates(node_list, start) if cleanup_materialized else []
    return ProjectTemplateRegistrationResult(
        mirrored_spec=mirrored_spec,
        project_templates=project_templates,
        deleted_paths=deleted_paths,
    )


def resolve_project_template_path(template_path: str, start: Path) -> Path:
    """Resolve a template path, treating .seed/... as project-root relative."""
    def resolve_tree_candidate(candidate: Path) -> Path | None:
        resolved = resolve_tree_like_path(candidate)
        if resolved:
            return resolved

        if candidate.is_dir():
            for suffix in TREE_LIKE_SUFFIXES:
                nested_candidate = candidate / f"{candidate.name}{suffix}"
                if nested_candidate.is_file():
                    return nested_candidate

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
    seen_relative_names: set[Path] = set()
    for directory in iter_registered_project_template_dirs(start):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                resolved = path.resolve()
                rel = path.relative_to(directory)
                if resolved in seen or rel in seen_relative_names:
                    continue
                seen.add(resolved)
                seen_relative_names.add(rel)
                templates.append(resolved)
    return templates


def resolve_registered_project_template(template_name: str, start: Path) -> Path:
    """Resolve a registered project template by name from nearest scope outward."""
    raw = Path(template_name)
    candidate_names = [raw]
    if raw.suffix.lower() not in TREE_LIKE_SUFFIXES:
        for suffix in TREE_LIKE_SUFFIXES:
            candidate_names.insert(0, raw.with_suffix(suffix))

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
        if rel.suffix.lower() in TREE_LIKE_SUFFIXES:
            suggestion = strip_tree_like_suffix(rel).as_posix()
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
