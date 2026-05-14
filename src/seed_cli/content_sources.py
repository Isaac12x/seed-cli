"""Helpers for fetching external content declared in spec nodes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Iterator, TYPE_CHECKING

from .logging import get_logger

if TYPE_CHECKING:
    from .parsers import Node


log = get_logger("content_sources")

_URL_RE = re.compile(
    r"((?:git\+)?https?://[^\s)]+|ssh://[^\s)]+|git://[^\s)]+|git@[^\s)]+)"
)


@dataclass(frozen=True)
class NodeContentSource:
    """External content source declared on a directory node comment."""

    relpath: Path
    url: str


def extract_url_candidate(text: str | None) -> str | None:
    """Return the first URL-like token embedded in comment text."""
    if not text:
        return None

    match = _URL_RE.search(text.strip())
    if not match:
        return None

    return match.group(1).rstrip(",.;")


def find_node_content_sources(nodes: Iterable["Node"]) -> list[NodeContentSource]:
    """Collect directory nodes whose comments contain a remote content source."""
    sources: list[NodeContentSource] = []
    seen: set[tuple[str, str]] = set()

    for node in nodes:
        if not node.is_dir:
            continue

        url = None
        if isinstance(node.metadata, dict):
            candidate = node.metadata.get("url")
            if candidate:
                url = str(candidate).strip()

        if not url:
            url = extract_url_candidate(node.comment)

        if not url:
            continue

        relpath = Path(node.relpath)
        key = (relpath.as_posix(), url)
        if key in seen:
            continue

        seen.add(key)
        sources.append(NodeContentSource(relpath=relpath, url=url))

    return sources


def materialize_content_sources(
    sources: Iterable[NodeContentSource],
    dest_dir: Path,
    *,
    strict: bool = False,
) -> list[Path]:
    """Fetch a collection of content sources into dest_dir."""
    from .template_registry import fetch_content_to_dir

    written: list[Path] = []
    root = dest_dir.resolve()

    for source in sources:
        target_dir = root if source.relpath.as_posix() == "." else root / source.relpath
        try:
            fetch_content_to_dir(source.url, target_dir)
            written.append(target_dir)
        except Exception as exc:
            if strict:
                raise
            log.warning(
                "Failed to fetch content for '%s' from %s: %s",
                source.relpath.as_posix(),
                source.url,
                exc,
            )

    return written


def materialize_node_content_sources(
    nodes: Iterable["Node"],
    dest_dir: Path,
    *,
    strict: bool = False,
) -> list[Path]:
    """Fetch any directory-comment content sources declared in nodes."""
    return materialize_content_sources(
        find_node_content_sources(nodes),
        dest_dir,
        strict=strict,
    )


@contextmanager
def runtime_template_dir(
    nodes: Iterable["Node"],
    template_dir: Path | None,
    *,
    enabled: bool = True,
) -> Iterator[Path | None]:
    """Build a temporary template directory with nested remote content overlaid."""
    if not enabled:
        yield template_dir
        return

    sources = find_node_content_sources(nodes)
    if not sources:
        yield template_dir
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_dir = Path(tmpdir) / "content"

        if template_dir and template_dir.exists():
            shutil.copytree(template_dir, merged_dir)
        else:
            merged_dir.mkdir(parents=True, exist_ok=True)

        materialize_content_sources(sources, merged_dir)
        yield merged_dir
