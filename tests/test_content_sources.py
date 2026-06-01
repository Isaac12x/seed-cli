from pathlib import Path

from seed_cli.content_sources import (
    NodeContentSource,
    extract_url_candidate,
    find_node_content_sources,
    runtime_template_dir,
)
from seed_cli.parsers import Node


def test_extract_url_candidate_finds_http_and_git_urls():
    assert extract_url_candidate("https://github.com/acme/repo.git") == "https://github.com/acme/repo.git"
    assert extract_url_candidate("source: git@github.com:acme/repo.git") == "git@github.com:acme/repo.git"
    assert extract_url_candidate("not a url") is None


def test_find_node_content_sources_only_uses_directory_comments():
    nodes = [
        Node(Path("vendor"), True, comment="https://github.com/acme/repo.git"),
        Node(Path("vendor/file.txt"), False, comment="https://example.com/file.txt"),
        Node(Path("docs"), True, comment="notes only"),
    ]

    assert find_node_content_sources(nodes) == [
        NodeContentSource(relpath=Path("vendor"), url="https://github.com/acme/repo.git")
    ]


def test_find_node_content_sources_uses_seed_metadata_url():
    nodes = [
        Node(
            Path("vendor"),
            True,
            comment="notes only",
            metadata={"url": "https://github.com/acme/repo.git"},
        ),
    ]

    assert find_node_content_sources(nodes) == [
        NodeContentSource(relpath=Path("vendor"), url="https://github.com/acme/repo.git")
    ]


def test_runtime_template_dir_overlays_nested_remote_content(tmp_path, monkeypatch):
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "existing.txt").write_text("base", encoding="utf-8")

    nodes = [
        Node(Path("vendor"), True, comment="https://github.com/acme/repo.git"),
    ]

    def fake_materialize(sources, dest_dir, *, strict=False):
        vendor_dir = dest_dir / "vendor"
        vendor_dir.mkdir(parents=True, exist_ok=True)
        (vendor_dir / "README.md").write_text("fetched", encoding="utf-8")
        return [vendor_dir]

    monkeypatch.setattr("seed_cli.content_sources.materialize_content_sources", fake_materialize)

    with runtime_template_dir(nodes, template_dir, enabled=True) as merged_dir:
        assert merged_dir is not None
        assert (merged_dir / "existing.txt").read_text(encoding="utf-8") == "base"
        assert (merged_dir / "vendor" / "README.md").read_text(encoding="utf-8") == "fetched"
