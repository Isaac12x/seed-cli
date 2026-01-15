

"""seed_cli.templates

Template directory handling.

Purpose:
- Validate template directories
- Render template trees into target filesystem
- Support variable substitution in filenames and file contents

This is used by executor (pre-apply injection).
"""

from pathlib import Path
from typing import Dict, Iterable
import shutil

from .templating import apply_vars


class TemplateError(RuntimeError):
    pass


def validate_template_dir(template_dir: Path) -> None:
    if not template_dir.exists():
        raise TemplateError(f"Template directory not found: {template_dir}")
    if not template_dir.is_dir():
        raise TemplateError(f"Template path is not a directory: {template_dir}")


def iter_template_files(template_dir: Path) -> Iterable[Path]:
    """Yield all files and directories inside template_dir."""
    for p in template_dir.rglob("*"):
        yield p


def render_template_dir(
    template_dir: Path,
    target_dir: Path,
    vars: Dict[str, str],
    *,
    overwrite: bool = False,
) -> None:
    """Render a template directory into target_dir.

    - Filenames are templated
    - File contents are templated
    - Directories are created automatically
    """
    validate_template_dir(template_dir)

    for src in iter_template_files(template_dir):
        rel = src.relative_to(template_dir)
        rel_str = apply_vars(rel.as_posix(), vars, mode="strict")
        dst = target_dir / rel_str

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            continue

        content = src.read_text(encoding="utf-8")
        rendered = apply_vars(content, vars, mode="strict")
        dst.write_text(rendered, encoding="utf-8")


def install_git_hook(base: Path, name: str) -> None:
    """Install a git hook from a template.
    
    Creates a pre-commit hook that runs `seed plan` to validate the spec.
    """
    git_dir = base / ".git" / "hooks"
    if not git_dir.exists():
        raise RuntimeError("Not a git repository")

    hook_path = git_dir / name
    hook_path.write_text(
        "#!/bin/sh\n"
        "seed plan || exit 1\n"
    )
    hook_path.chmod(0o755)
