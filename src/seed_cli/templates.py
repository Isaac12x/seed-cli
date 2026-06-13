

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


GBRAIN_POST_APPLY_SCRIPT = """#!/bin/sh
# Installed by `seed hooks install --gbrain`. Re-export the gbrain pack after
# `seed apply` mutates structure so the brain stays in sync with the spec.
if [ "$SEED_HOOK_STAGE" != "post_apply" ]; then
  exit 0
fi
SPEC="${SEED_GBRAIN_SPEC:-__SPEC__}"
NAME="${SEED_GBRAIN_NAME:-__NAME__}"
ACTIVATE="${SEED_GBRAIN_ACTIVATE:-__ACTIVATE__}"
EXTRA="${SEED_GBRAIN_EXTRA:-}"
if [ ! -f "$SPEC" ]; then
  echo "[gbrain hook] spec '$SPEC' not found; skipping" >&2
  exit 0
fi
exec seed export gbrain "$SPEC" --name "$NAME" --install --activate "$ACTIVATE" $EXTRA
"""


GBRAIN_PRE_PUSH_SCRIPT = """#!/bin/sh
# Installed by `seed hooks install --gbrain`. Fails the push if the committed
# pack hash diverges from the current spec hash (PRD AC7).
SPEC="${SEED_GBRAIN_SPEC:-__SPEC__}"
PACK="${SEED_GBRAIN_PACK:-__PACK__}"
SOURCE_JSON="${SEED_GBRAIN_SOURCE_JSON:-__SOURCE_JSON__}"
if [ ! -f "$SPEC" ] || [ ! -f "$PACK" ] || [ ! -f "$SOURCE_JSON" ]; then
  exit 0
fi
SPEC_HASH=$(python3 - "$SPEC" <<'PY'
import sys, hashlib
text = open(sys.argv[1]).read()
out = []
for line in text.splitlines():
    line = line.rstrip()
    if not line or line.lstrip().startswith('#'):
        continue
    out.append(line)
print(hashlib.sha256('\\n'.join(out).encode()).hexdigest())
PY
)
RECORDED=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('spec_hash',''))" "$SOURCE_JSON")
if [ "$SPEC_HASH" != "$RECORDED" ]; then
  echo "gbrain pack is stale: re-run \\`seed export gbrain $SPEC --install\\` and commit before pushing." >&2
  exit 1
fi
"""


def install_gbrain_post_apply_hook(
    base: Path,
    *,
    spec: str = "brain.seed",
    name: str = "brain-pack",
    activate_mode: str = "repo",
    extra: str = "",
) -> Path:
    """Install a `<base>/hooks/post_apply_gbrain.sh` script."""
    hooks_dir = base / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    script_path = hooks_dir / "post_apply_gbrain.sh"
    script = (
        GBRAIN_POST_APPLY_SCRIPT
        .replace("__SPEC__", spec)
        .replace("__NAME__", name)
        .replace("__ACTIVATE__", activate_mode)
    )
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def install_gbrain_pre_push_hook(
    base: Path,
    *,
    spec: str = "brain.seed",
    pack: str = ".gbrain/pack/pack.yaml",
    source_json: str = ".gbrain/pack/source.json",
) -> Path:
    """Install ``.git/hooks/pre-push`` that blocks pushes when the pack is stale."""
    git_dir = base / ".git" / "hooks"
    if not git_dir.exists():
        raise RuntimeError("Not a git repository")
    hook_path = git_dir / "pre-push"
    script = (
        GBRAIN_PRE_PUSH_SCRIPT
        .replace("__SPEC__", spec)
        .replace("__PACK__", pack)
        .replace("__SOURCE_JSON__", source_json)
    )
    hook_path.write_text(script, encoding="utf-8")
    hook_path.chmod(0o755)
    return hook_path
