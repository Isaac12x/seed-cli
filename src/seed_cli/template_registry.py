"""seed_cli.template_registry

Template management - store and manage reusable specs from GitHub.

Features:
- Fetch specs from GitHub URLs
- Version management per template
- Template locking
- Local storage in ~/.seed/templates/

Structure:
    ~/.seed/
    └── templates/
        ├── registry.json         # {name: TemplateMetadata}
        └── <template-name>/
            ├── meta.json         # name, source, current_version, locked, versions[]
            ├── v1.tree
            └── v2.tree

Usage:
    seed templates add https://github.com/user/repo/blob/main/spec.tree --name myspec
    seed templates list
    seed templates use myspec
    seed templates lock myspec
    seed templates versions myspec --add newspec.tree --name v2
"""

import importlib.resources
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .logging import get_logger

log = get_logger("template_registry")

TEMPLATES_DIR_NAME = "templates"
REGISTRY_FILE = "registry.json"
META_FILE = "meta.json"


@dataclass
class TemplateMetadata:
    """Metadata for a stored template."""
    name: str
    source: str  # GitHub URL
    current_version: str  # "v1", "main", etc.
    locked: bool = False
    created_at: float = field(default_factory=time.time)
    versions: List[str] = field(default_factory=list)
    content_url: Optional[str] = None  # GitHub tree URL for fetching content

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateMetadata":
        # Handle older meta.json files that don't have content_url
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def get_templates_dir() -> Path:
    """Get the templates directory (~/.seed/templates/).

    Creates the directory if it doesn't exist.
    """
    seed_dir = Path.home() / ".seed"
    templates_dir = seed_dir / TEMPLATES_DIR_NAME
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def _get_registry_path() -> Path:
    """Get path to the registry.json file."""
    return get_templates_dir() / REGISTRY_FILE


_defaults_installed = False


def install_default_templates() -> None:
    """Install bundled default templates into ~/.seed/templates/ if not already present.

    Templates are shipped inside the package under resources/default_templates/.
    Each subdirectory contains a spec.tree (and optionally a files/ dir).
    Only installs templates that don't already exist in the registry.
    """
    global _defaults_installed
    if _defaults_installed:
        return
    _defaults_installed = True

    registry = load_registry()

    # Locate bundled templates via importlib.resources
    try:
        resources_pkg = importlib.resources.files("seed_cli") / "resources" / "default_templates"
    except (TypeError, ModuleNotFoundError):
        return

    # Iterate over bundled template directories
    try:
        entries = list(resources_pkg.iterdir())
    except (FileNotFoundError, OSError):
        return

    for entry in entries:
        if not entry.is_dir():
            continue
        tpl_name = entry.name

        # Skip if already registered
        if tpl_name in registry:
            continue

        spec_resource = entry / "spec.tree"
        if not spec_resource.is_file():
            continue

        # Read spec content from package resources
        try:
            spec_content = spec_resource.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Install into ~/.seed/templates/<name>/
        template_dir = _get_template_dir(tpl_name)
        template_dir.mkdir(parents=True, exist_ok=True)

        version = "v1"
        dest_path = template_dir / f"{version}.tree"
        dest_path.write_text(spec_content)

        # Copy files/ content dir if present
        content_url = None
        files_resource = entry / "files"
        if files_resource.is_dir():
            content_dest = template_dir / version
            if content_dest.exists():
                shutil.rmtree(content_dest)
            # Use shutil.copytree for Path-like resources (works on filesystem paths)
            try:
                shutil.copytree(str(files_resource), str(content_dest))
            except (OSError, shutil.Error):
                pass

        # Check for source.json and fetch content from GitHub
        source_json_resource = entry / "source.json"
        if source_json_resource.is_file():
            try:
                source_data = json.loads(source_json_resource.read_text(encoding="utf-8"))
                content_url = source_data.get("content_url")
                if content_url:
                    content_dest = template_dir / version
                    try:
                        fetch_dir_from_github(content_url, content_dest)
                        log.info(f"Fetched content for '{tpl_name}' from {content_url}")
                    except Exception as e:
                        log.warning(
                            f"Failed to fetch content for '{tpl_name}' from {content_url}: {e}. "
                            "Installing structure-only."
                        )
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                log.warning(f"Error reading source.json for {tpl_name}: {e}")

        meta = TemplateMetadata(
            name=tpl_name,
            source="builtin",
            current_version=version,
            locked=False,
            versions=[version],
            content_url=content_url,
        )
        _save_meta(meta)
        registry[tpl_name] = meta

    save_registry(registry)


def load_registry() -> Dict[str, TemplateMetadata]:
    """Load the template registry from disk.

    Returns:
        Dict mapping template names to TemplateMetadata
    """
    registry_path = _get_registry_path()
    if not registry_path.exists():
        return {}

    try:
        data = json.loads(registry_path.read_text())
        return {
            name: TemplateMetadata.from_dict(meta)
            for name, meta in data.items()
        }
    except (json.JSONDecodeError, KeyError) as e:
        log.warning(f"Error loading registry: {e}")
        return {}


def save_registry(registry: Dict[str, TemplateMetadata]) -> None:
    """Save the template registry to disk."""
    registry_path = _get_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        name: meta.to_dict()
        for name, meta in registry.items()
    }
    registry_path.write_text(json.dumps(data, indent=2))


def _get_template_dir(name: str) -> Path:
    """Get the directory for a specific template."""
    return get_templates_dir() / name


def _load_meta(name: str) -> Optional[TemplateMetadata]:
    """Load metadata for a template from its meta.json file."""
    meta_path = _get_template_dir(name) / META_FILE
    if not meta_path.exists():
        return None

    try:
        data = json.loads(meta_path.read_text())
        return TemplateMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        log.warning(f"Error loading meta for {name}: {e}")
        return None


def _save_meta(meta: TemplateMetadata) -> None:
    """Save metadata for a template to its meta.json file."""
    template_dir = _get_template_dir(meta.name)
    template_dir.mkdir(parents=True, exist_ok=True)

    meta_path = template_dir / META_FILE
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2))


def parse_github_url(url: str) -> Optional[Dict[str, str]]:
    """Parse a GitHub URL to extract owner, repo, ref, and path.

    Supports formats:
    - https://github.com/owner/repo/blob/ref/path/to/file.tree
    - https://github.com/owner/repo/tree/ref/path/to/dir
    - github.com/owner/repo/blob/ref/path/to/file.tree

    Returns:
        Dict with keys: owner, repo, ref, path, type (blob/tree)
        None if URL doesn't match expected format
    """
    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "www.github.com"):
        return None

    # Match /owner/repo/blob|tree/ref/path...
    pattern = r'^/([^/]+)/([^/]+)/(blob|tree)/([^/]+)(?:/(.*))?$'
    match = re.match(pattern, parsed.path)

    if not match:
        return None

    owner, repo, url_type, ref, path = match.groups()
    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "path": path or "",
        "type": url_type,  # "blob" for file, "tree" for directory
    }


def _has_gh_cli() -> bool:
    """Check if gh CLI is available."""
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _fetch_with_gh(owner: str, repo: str, ref: str, path: str) -> str:
    """Fetch file content using gh CLI (handles auth)."""
    # Use gh api to fetch raw content
    api_path = f"repos/{owner}/{repo}/contents/{path}?ref={ref}"

    result = subprocess.run(
        ["gh", "api", api_path, "-q", ".content"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"gh api failed: {result.stderr}")

    # Content is base64 encoded
    import base64
    content = base64.b64decode(result.stdout.strip()).decode("utf-8")
    return content


def _fetch_with_git(owner: str, repo: str, ref: str, path: str, dest_dir: Path) -> Path:
    """Fetch file using git sparse-checkout (no auth needed for public repos)."""
    repo_url = f"https://github.com/{owner}/{repo}.git"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize sparse checkout
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
             "--branch", ref, repo_url, str(tmpdir / "repo")],
            capture_output=True,
            timeout=60,
            check=True,
        )

        repo_dir = tmpdir / "repo"

        # Set up sparse checkout
        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        # Add the specific path
        parent_dir = str(Path(path).parent) if "/" in path else "."
        subprocess.run(
            ["git", "sparse-checkout", "set", parent_dir],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        # Checkout
        subprocess.run(
            ["git", "checkout"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        # Copy the file to destination
        src_file = repo_dir / path
        if not src_file.exists():
            raise FileNotFoundError(f"File not found in repo: {path}")

        dest_file = dest_dir / src_file.name
        shutil.copy2(src_file, dest_file)
        return dest_file


def fetch_from_github(url: str, dest_dir: Path, name: Optional[str] = None) -> Tuple[Path, str]:
    """Fetch a spec file from GitHub.

    Args:
        url: GitHub URL to the spec file
        dest_dir: Directory to store the fetched file
        name: Optional name override for the file

    Returns:
        Tuple of (local_path, original_filename)

    Raises:
        ValueError: If URL is invalid
        RuntimeError: If fetching fails
    """
    parsed = parse_github_url(url)
    if not parsed:
        raise ValueError(f"Invalid GitHub URL: {url}")

    if parsed["type"] != "blob":
        raise ValueError("URL must point to a file (blob), not a directory (tree)")

    owner = parsed["owner"]
    repo = parsed["repo"]
    ref = parsed["ref"]
    path = parsed["path"]

    if not path:
        raise ValueError("URL must include a path to a file")

    original_name = Path(path).name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try gh CLI first (handles auth for private repos)
    if _has_gh_cli():
        try:
            content = _fetch_with_gh(owner, repo, ref, path)
            dest_name = name or original_name
            dest_file = dest_dir / dest_name
            dest_file.write_text(content)
            return dest_file, original_name
        except Exception as e:
            log.debug(f"gh CLI failed, falling back to git: {e}")

    # Fallback to git sparse-checkout
    try:
        fetched = _fetch_with_git(owner, repo, ref, path, dest_dir)
        if name and name != fetched.name:
            new_path = dest_dir / name
            fetched.rename(new_path)
            return new_path, original_name
        return fetched, original_name
    except Exception as e:
        raise RuntimeError(f"Failed to fetch from GitHub: {e}")


def _fetch_dir_with_git(owner: str, repo: str, ref: str, path: str, dest_dir: Path) -> Path:
    """Fetch a directory from GitHub using git sparse-checkout.

    Args:
        owner: Repository owner
        repo: Repository name
        ref: Git ref (branch, tag, commit)
        path: Path to directory within the repo
        dest_dir: Local destination directory

    Returns:
        Path to the fetched directory contents (dest_dir)
    """
    repo_url = f"https://github.com/{owner}/{repo}.git"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
             "--branch", ref, repo_url, str(tmpdir / "repo")],
            capture_output=True,
            timeout=120,
            check=True,
        )

        repo_dir = tmpdir / "repo"

        subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        subprocess.run(
            ["git", "sparse-checkout", "set", path],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        subprocess.run(
            ["git", "checkout"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        src_dir = repo_dir / path
        if not src_dir.exists() or not src_dir.is_dir():
            raise FileNotFoundError(f"Directory not found in repo: {path}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        # Copy contents of the fetched directory into dest_dir
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(str(src_dir), str(dest_dir))
        return dest_dir


def fetch_dir_from_github(url: str, dest_dir: Path) -> Path:
    """Fetch a directory from a GitHub tree URL.

    Args:
        url: GitHub tree URL (e.g., https://github.com/owner/repo/tree/ref/path)
        dest_dir: Local destination directory

    Returns:
        Path to the local copy

    Raises:
        ValueError: If URL is invalid or not a tree URL
        RuntimeError: If fetching fails
    """
    parsed = parse_github_url(url)
    if not parsed:
        raise ValueError(f"Invalid GitHub URL: {url}")

    if parsed["type"] != "tree":
        raise ValueError("URL must point to a directory (tree), not a file (blob)")

    owner = parsed["owner"]
    repo = parsed["repo"]
    ref = parsed["ref"]
    path = parsed["path"]

    if not path:
        raise ValueError("URL must include a path to a directory")

    try:
        return _fetch_dir_with_git(owner, repo, ref, path, dest_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch directory from GitHub: {e}")


def update_template(name: str, content_url: Optional[str] = None) -> TemplateMetadata:
    """Re-fetch content from a template's content_url, or set a new one.

    Args:
        name: Template name
        content_url: Optional new content URL to set (local path or GitHub tree URL).
                     If provided, replaces the existing content_url.

    Returns:
        Updated TemplateMetadata

    Raises:
        ValueError: If template not found or has no content_url
        RuntimeError: If fetching fails
    """
    meta = _load_meta(name)
    if not meta:
        raise ValueError(f"Template not found: {name}")

    # Use new content_url if provided, otherwise use existing
    url = content_url or meta.content_url
    if not url:
        raise ValueError(f"Template '{name}' has no content_url to update from")

    template_dir = _get_template_dir(name)
    content_dest = template_dir / meta.current_version

    fetch_content_to_dir(url, content_dest)

    # If a new content_url was provided, persist it
    if content_url:
        meta.content_url = content_url
        _save_meta(meta)
        _save_source_json(name, content_url)

        registry = load_registry()
        registry[name] = meta
        save_registry(registry)

    return meta


def _save_source_json(name: str, content_url: str) -> None:
    """Write source.json into a template's directory."""
    template_dir = _get_template_dir(name)
    template_dir.mkdir(parents=True, exist_ok=True)
    source_json_path = template_dir / "source.json"
    source_json_path.write_text(json.dumps({"content_url": content_url}, indent=2))


def fetch_content_to_dir(content_url: str, dest_dir: Path) -> Path:
    """Fetch content from a local path or GitHub tree URL into dest_dir.

    Args:
        content_url: Local directory path or GitHub tree URL
        dest_dir: Destination directory

    Returns:
        Path to dest_dir
    """
    local_path = Path(content_url)
    if local_path.exists() and local_path.is_dir():
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(str(local_path), str(dest_dir))
        return dest_dir

    # Otherwise treat as GitHub URL
    return fetch_dir_from_github(content_url, dest_dir)


def _get_next_version(template_dir: Path) -> str:
    """Get the next version number for a template."""
    if not template_dir.exists():
        return "v1"

    existing = []
    for f in template_dir.iterdir():
        if f.suffix == ".tree" and f.stem.startswith("v"):
            match = re.match(r"v(\d+)", f.stem)
            if match:
                existing.append(int(match.group(1)))

    if not existing:
        return "v1"
    return f"v{max(existing) + 1}"


def add_template(
    source: str,
    name: Optional[str] = None,
    version: Optional[str] = None,
    content_url: Optional[str] = None,
) -> TemplateMetadata:
    """Add a template from a GitHub URL.

    Args:
        source: GitHub URL to the spec file
        name: Optional name for the template (defaults to filename without extension)
        version: Optional version name (defaults to auto-increment)
        content_url: Optional content URL (local path or GitHub tree URL) to fetch content from

    Returns:
        TemplateMetadata for the added template

    Raises:
        ValueError: If URL is invalid or template already exists
        RuntimeError: If fetching fails
    """
    parsed = parse_github_url(source)
    if not parsed:
        raise ValueError(f"Invalid GitHub URL: {source}")

    # Determine template name
    if not name:
        path = parsed.get("path", "")
        if path:
            name = Path(path).stem  # filename without extension
        else:
            name = parsed["repo"]

    # Sanitize name
    name = re.sub(r'[^\w\-]', '_', name)

    # If URL points to a directory (tree), fetch it and look for spec.tree + source.json
    if parsed["type"] == "tree":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir) / "fetched"
            fetch_dir_from_github(source, tmpdir_path)

            # Look for spec.tree inside
            spec_file = tmpdir_path / "spec.tree"
            if not spec_file.exists():
                raise ValueError(f"No spec.tree found in directory at {source}")

            # Delegate to add_local_template with the fetched directory
            return add_local_template(str(tmpdir_path), name, version=version, content_url=content_url)

    template_dir = _get_template_dir(name)

    # Check if template exists and is locked
    existing_meta = _load_meta(name)
    if existing_meta and existing_meta.locked:
        raise ValueError(f"Template '{name}' is locked. Unlock it first.")

    # Determine version
    if not version:
        version = _get_next_version(template_dir)
    elif not version.startswith("v"):
        version = f"v{version}"

    # Fetch the spec file
    version_filename = f"{version}.tree"
    fetched_path, _ = fetch_from_github(source, template_dir, version_filename)

    # Fetch content from content_url if provided
    if content_url:
        content_dest = template_dir / version
        try:
            fetch_content_to_dir(content_url, content_dest)
            log.info(f"Fetched content from {content_url}")
        except Exception as e:
            log.warning(f"Failed to fetch content from {content_url}: {e}")
            log.warning("Installing structure-only (empty files)")

    # Create or update metadata
    if existing_meta:
        meta = existing_meta
        if version not in meta.versions:
            meta.versions.append(version)
        meta.current_version = version
        if content_url:
            meta.content_url = content_url
    else:
        meta = TemplateMetadata(
            name=name,
            source=source,
            current_version=version,
            locked=False,
            versions=[version],
            content_url=content_url,
        )

    _save_meta(meta)

    # Persist source.json
    if content_url:
        _save_source_json(name, content_url)

    # Update registry
    registry = load_registry()
    registry[name] = meta
    save_registry(registry)

    return meta


def add_local_template(
    spec_path: str,
    name: str,
    version: Optional[str] = None,
    content_dir: Optional[str] = None,
    content_url: Optional[str] = None,
) -> TemplateMetadata:
    """Add a template from a local spec file or directory.

    Args:
        spec_path: Path to the local spec file (.tree) or directory containing spec.tree
        name: Name for the template
        version: Optional version name (defaults to auto-increment)
        content_dir: Optional path to directory with file contents to copy
        content_url: Optional content URL (local path or GitHub tree URL) to fetch content from

    Returns:
        TemplateMetadata for the added template
    """
    spec_file = Path(spec_path)

    # If spec_path is a directory, look for spec.tree and files/ inside
    source_json_data = None
    if spec_file.is_dir():
        source_dir = spec_file
        spec_file = source_dir / "spec.tree"
        if not spec_file.exists():
            raise FileNotFoundError(
                f"No spec.tree found in directory: {spec_path}"
            )
        # Auto-detect files/ subdirectory as content_dir
        files_dir = source_dir / "files"
        if content_dir is None and files_dir.exists() and files_dir.is_dir():
            content_dir = str(files_dir)
        # Check for source.json
        source_json_path = source_dir / "source.json"
        if source_json_path.exists():
            try:
                source_json_data = json.loads(source_json_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Error reading source.json: {e}")
    elif not spec_file.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    # Sanitize name
    name = re.sub(r'[^\w\-]', '_', name)

    template_dir = _get_template_dir(name)
    template_dir.mkdir(parents=True, exist_ok=True)

    # Check if template exists and is locked
    existing_meta = _load_meta(name)
    if existing_meta and existing_meta.locked:
        raise ValueError(f"Template '{name}' is locked. Unlock it first.")

    # Determine version
    if not version:
        version = _get_next_version(template_dir)
    elif not version.startswith("v"):
        version = f"v{version}"

    # Copy the spec file
    dest_path = template_dir / f"{version}.tree"
    shutil.copy2(spec_file, dest_path)

    # Copy content directory if provided
    if content_dir:
        content_src = Path(content_dir)
        if content_src.exists() and content_src.is_dir():
            content_dest = template_dir / version
            if content_dest.exists():
                shutil.rmtree(content_dest)
            shutil.copytree(content_src, content_dest)

    # Resolve content_url: explicit param > source.json > None
    if not content_url and source_json_data and not content_dir:
        content_url = source_json_data.get("content_url")

    # Fetch content from content_url if present and no local content_dir was copied
    if content_url and not content_dir:
        content_dest = template_dir / version
        try:
            fetch_content_to_dir(content_url, content_dest)
            log.info(f"Fetched content from {content_url}")
        except Exception as e:
            log.warning(f"Failed to fetch content from {content_url}: {e}")
            log.warning("Installing structure-only (empty files)")

    # Create or update metadata
    if existing_meta:
        meta = existing_meta
        if version not in meta.versions:
            meta.versions.append(version)
        meta.current_version = version
        if content_url:
            meta.content_url = content_url
        # Preserve original source if it was from GitHub
        if not meta.source.startswith("local:"):
            pass  # Keep the GitHub source
        else:
            meta.source = f"local:{spec_path}"
    else:
        meta = TemplateMetadata(
            name=name,
            source=f"local:{spec_path}",
            current_version=version,
            locked=False,
            versions=[version],
            content_url=content_url,
        )

    _save_meta(meta)

    # Persist source.json so `templates update` knows where to fetch
    if content_url:
        _save_source_json(name, content_url)

    # Update registry
    registry = load_registry()
    registry[name] = meta
    save_registry(registry)

    return meta


def remove_template(name: str) -> bool:
    """Remove a template.

    Args:
        name: Template name

    Returns:
        True if removed, False if not found

    Raises:
        ValueError: If template is locked
    """
    registry = load_registry()

    if name not in registry:
        return False

    meta = registry[name]
    if meta.locked:
        raise ValueError(f"Template '{name}' is locked. Unlock it first.")

    # Remove template directory
    template_dir = _get_template_dir(name)
    if template_dir.exists():
        shutil.rmtree(template_dir)

    # Remove from registry
    del registry[name]
    save_registry(registry)

    return True


def list_templates() -> List[TemplateMetadata]:
    """List all templates.

    Returns:
        List of TemplateMetadata for all templates
    """
    install_default_templates()
    registry = load_registry()
    return list(registry.values())


def get_template(name: str) -> Optional[TemplateMetadata]:
    """Get metadata for a specific template.

    Args:
        name: Template name

    Returns:
        TemplateMetadata or None if not found
    """
    install_default_templates()
    registry = load_registry()
    return registry.get(name)


def get_template_spec_path(name: str, version: Optional[str] = None) -> Optional[Path]:
    """Get the path to a template's spec file.

    Args:
        name: Template name
        version: Optional version (defaults to current_version)

    Returns:
        Path to the .tree file or None if not found
    """
    meta = get_template(name)
    if not meta:
        return None

    if version is None:
        version = meta.current_version
    elif not version.startswith("v"):
        version = f"v{version}"

    spec_path = _get_template_dir(name) / f"{version}.tree"
    if spec_path.exists():
        return spec_path
    return None


def get_template_content_dir(name: str, version: Optional[str] = None) -> Optional[Path]:
    """Get the path to a template's content directory.

    Content directories contain actual file contents that mirror the tree structure.
    Located at ~/.seed/templates/<name>/<version>/.

    Args:
        name: Template name
        version: Optional version (defaults to current_version)

    Returns:
        Path to the content directory or None if not found
    """
    meta = get_template(name)
    if not meta:
        return None

    if version is None:
        version = meta.current_version
    elif not version.startswith("v"):
        version = f"v{version}"

    content_dir = _get_template_dir(name) / version
    if content_dir.exists() and content_dir.is_dir():
        return content_dir
    return None


def list_versions(name: str) -> List[Tuple[str, Path]]:
    """List all versions of a template.

    Args:
        name: Template name

    Returns:
        List of (version, path) tuples sorted by version number
    """
    meta = get_template(name)
    if not meta:
        return []

    template_dir = _get_template_dir(name)
    versions = []

    for version in meta.versions:
        path = template_dir / f"{version}.tree"
        if path.exists():
            versions.append((version, path))

    # Sort by version number
    def version_key(item):
        v = item[0]
        match = re.match(r'v(\d+)', v)
        if match:
            return int(match.group(1))
        return 0

    return sorted(versions, key=version_key)


def add_version(
    name: str,
    spec_path: str,
    version_name: Optional[str] = None,
    content_dir: Optional[str] = None,
) -> str:
    """Add a new version to an existing template.

    Args:
        name: Template name
        spec_path: Path to the spec file to add
        version_name: Optional version name
        content_dir: Optional path to directory with file contents to copy

    Returns:
        The version name that was added

    Raises:
        ValueError: If template not found or is locked
        FileNotFoundError: If spec_path doesn't exist
    """
    meta = get_template(name)
    if not meta:
        raise ValueError(f"Template not found: {name}")

    if meta.locked:
        raise ValueError(f"Template '{name}' is locked. Unlock it first.")

    spec_file = Path(spec_path)
    if not spec_file.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    template_dir = _get_template_dir(name)

    # Determine version
    if not version_name:
        version_name = _get_next_version(template_dir)
    elif not version_name.startswith("v"):
        version_name = f"v{version_name}"

    # Copy the spec file
    dest_path = template_dir / f"{version_name}.tree"
    shutil.copy2(spec_file, dest_path)

    # Copy content directory if provided
    if content_dir:
        content_src = Path(content_dir)
        if content_src.exists() and content_src.is_dir():
            content_dest = template_dir / version_name
            if content_dest.exists():
                shutil.rmtree(content_dest)
            shutil.copytree(content_src, content_dest)

    # Update metadata
    if version_name not in meta.versions:
        meta.versions.append(version_name)

    _save_meta(meta)

    # Update registry
    registry = load_registry()
    registry[name] = meta
    save_registry(registry)

    return version_name


def set_current_version(name: str, version: str) -> None:
    """Set the current version for a template.

    Args:
        name: Template name
        version: Version to set as current

    Raises:
        ValueError: If template or version not found
    """
    meta = get_template(name)
    if not meta:
        raise ValueError(f"Template not found: {name}")

    if not version.startswith("v"):
        version = f"v{version}"

    if version not in meta.versions:
        raise ValueError(f"Version not found: {version}")

    meta.current_version = version
    _save_meta(meta)

    # Update registry
    registry = load_registry()
    registry[name] = meta
    save_registry(registry)


def lock_template(name: str, version: Optional[str] = None) -> None:
    """Lock a template to prevent modifications.

    Args:
        name: Template name
        version: Optional version to set as current before locking

    Raises:
        ValueError: If template not found
    """
    meta = get_template(name)
    if not meta:
        raise ValueError(f"Template not found: {name}")

    if version:
        if not version.startswith("v"):
            version = f"v{version}"
        if version not in meta.versions:
            raise ValueError(f"Version not found: {version}")
        meta.current_version = version

    meta.locked = True
    _save_meta(meta)

    # Update registry
    registry = load_registry()
    registry[name] = meta
    save_registry(registry)


def unlock_template(name: str) -> None:
    """Unlock a template to allow modifications.

    Args:
        name: Template name

    Raises:
        ValueError: If template not found
    """
    meta = get_template(name)
    if not meta:
        raise ValueError(f"Template not found: {name}")

    meta.locked = False
    _save_meta(meta)

    # Update registry
    registry = load_registry()
    registry[name] = meta
    save_registry(registry)
