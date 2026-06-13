import pytest
from pathlib import Path
from seed_cli.parsers import parse_any, parse_tree_text


def test_parse_simple_tree():
    text = """
    a/
    a/file.txt
    b/file2.txt
    """
    nodes = parse_tree_text(text)
    paths = {(n.relpath.as_posix(), n.is_dir) for n in nodes}
    # Parser creates root "." when it sees a root dir
    assert (".", True) in paths or ("a", True) in paths
    assert ("a/file.txt", False) in paths
    assert ("b/file2.txt", False) in paths


def test_parse_ascii_tree():
    text = """
    root/
    ├── a/
    │   └── file.txt
    └── b.txt
    """
    nodes = parse_tree_text(text)
    paths = {n.relpath.as_posix() for n in nodes}
    # Parser creates root "." and paths relative to it
    assert "." in paths or "root" in paths
    assert "a" in paths or "root/a" in paths
    assert "a/file.txt" in paths or "root/a/file.txt" in paths or "file.txt" in paths
    assert "b.txt" in paths or "root/b.txt" in paths


def test_parse_comment_and_annotation():
    text = "a/file.db (encrypted) (@manual)"
    nodes = parse_tree_text(text)
    n = nodes[0]
    assert n.comment == "encrypted"
    assert n.annotation == "manual"


def test_parse_seed_inline_metadata():
    text = "vendor/ !service +remote +template -> https://github.com/acme/repo.git (@manual)"
    nodes = parse_tree_text(text)
    n = nodes[0]
    assert n.annotation == "manual"
    assert n.metadata == {
        "kind": "service",
        "tags": ["remote", "template"],
        "url": "https://github.com/acme/repo.git",
    }


def test_parse_seed_brace_group_with_shared_extension():
    nodes = parse_tree_text("memories/{global,facts,episodes}.jsonl")

    assert {n.relpath.as_posix() for n in nodes} == {
        "memories/global.jsonl",
        "memories/facts.jsonl",
        "memories/episodes.jsonl",
    }
    assert all(not n.is_dir for n in nodes)


def test_parse_seed_brace_group_with_mixed_files_and_directories():
    nodes = parse_tree_text(
        "services/<service-id>/{service.json,knowledge/,prompts/,tools.json}"
    )

    assert {(n.relpath.as_posix(), n.is_dir) for n in nodes} == {
        ("services/<service-id>/service.json", False),
        ("services/<service-id>/knowledge", True),
        ("services/<service-id>/prompts", True),
        ("services/<service-id>/tools.json", False),
    }


def test_parse_seed_brace_group_with_directory_suffix():
    nodes = parse_tree_text("common/global/{knowledge,policies,prompts}/")

    assert {(n.relpath.as_posix(), n.is_dir) for n in nodes} == {
        ("common/global/knowledge", True),
        ("common/global/policies", True),
        ("common/global/prompts", True),
    }


def test_parse_seed_multiple_brace_groups_as_cartesian_product():
    nodes = parse_tree_text("{people,teams}/{active,archived}.json")

    assert {n.relpath.as_posix() for n in nodes} == {
        "people/active.json",
        "people/archived.json",
        "teams/active.json",
        "teams/archived.json",
    }


def test_parse_seed_brace_group_strips_alternative_whitespace():
    nodes = parse_tree_text("audit/{ billing_events, memory_writes }.jsonl")

    assert {n.relpath.as_posix() for n in nodes} == {
        "audit/billing_events.jsonl",
        "audit/memory_writes.jsonl",
    }


def test_parse_seed_brace_group_inherits_inline_metadata():
    nodes = parse_tree_text(
        "vendor/{api,worker}/ !service +remote (@manual) (shared services)"
    )

    assert {n.relpath.as_posix() for n in nodes} == {
        "vendor/api",
        "vendor/worker",
    }
    assert all(n.is_dir for n in nodes)
    assert all(n.annotation == "manual" for n in nodes)
    assert all(n.comment == "shared services" for n in nodes)
    assert all(n.metadata == {"kind": "service", "tags": ["remote"]} for n in nodes)


def test_parse_seed_nested_template_directory_path():
    nodes = parse_tree_text("services/<service-id>/")

    assert len(nodes) == 1
    assert nodes[0].relpath.as_posix() == "services/<service-id>"
    assert nodes[0].is_dir
    assert nodes[0].annotation == "template:service-id"


@pytest.mark.parametrize(
    "text",
    [
        "literal/{name}.txt",
        "literal/{left,right.txt",
        "literal/left,right}.txt",
    ],
)
def test_parse_seed_non_expandable_braces_remain_literal(text):
    nodes = parse_tree_text(text)

    assert [n.relpath.as_posix() for n in nodes] == [text]


def test_parse_tree_skips_unicode_guides_and_preserves_at_prefixed_dirs():
    text = """
    .
    ├── tracking/
    │   └── status/
    │
    ├── @userfiles/
    │   └── incoming/
    └── archive/
    """
    nodes = parse_tree_text(text)

    indexed = {(n.relpath.as_posix(), n.is_dir, n.annotation) for n in nodes}

    assert (".", True, None) in indexed
    assert ("tracking", True, None) in indexed
    assert ("tracking/status", True, None) in indexed
    assert ("@userfiles", True, None) in indexed
    assert ("@userfiles/incoming", True, None) in indexed
    assert ("archive", True, None) in indexed
    assert not any(n.relpath.as_posix() == "│" for n in nodes)
    assert not any(n.annotation == "userfiles" for n in nodes)


def test_parse_structured_json():
    import json
    doc = {
        "entries": [
            {"path": "a/", "type": "dir"},
            {"path": "a/file.txt", "type": "file", "annotation": "generated"},
        ]
    }
    root, nodes = parse_any("spec.json", json.dumps(doc))
    assert root.as_posix() == "."
    assert len(nodes) == 2
    assert nodes[1].annotation == "generated"


def test_parse_structured_yaml():
    text = """
    entries:
      - path: a/
        type: dir
      - path: a/file.txt
        type: file
        annotation: manual
    """
    root, nodes = parse_any("spec.yaml", text)
    assert len(nodes) == 2
    assert nodes[1].annotation == "manual"


def test_parse_structured_metadata_fields():
    text = """
    entries:
      - path: vendor/
        type: dir
        kind: service
        tags:
          - remote
          - template
        url: https://github.com/acme/repo.git
    """
    _, nodes = parse_any("spec.seed", text)
    assert nodes[0].metadata == {
        "kind": "service",
        "tags": ["remote", "template"],
        "url": "https://github.com/acme/repo.git",
    }


def test_parse_templating():
    text = "{{name}}/file.txt"
    _, nodes = parse_any("spec.tree", text, vars={"name": "demo"})
    assert nodes[0].relpath.as_posix() == "demo/file.txt"


def test_parse_spec_text_file(tmp_path):
    from seed_cli.parsers import parse_spec
    
    spec = tmp_path / "spec.tree"
    spec.write_text("a/file.txt")
    _, nodes = parse_spec(str(spec), base=tmp_path)
    assert len(nodes) >= 1
    # Should have the file, may or may not have explicit dir
    assert any(n.relpath.as_posix() == "a/file.txt" and not n.is_dir for n in nodes)


def test_parse_spec_seed_file(tmp_path):
    from seed_cli.parsers import parse_spec

    spec = tmp_path / "spec.seed"
    spec.write_text("vendor/ !service -> https://github.com/acme/repo.git")
    _, nodes = parse_spec(str(spec), base=tmp_path)
    assert nodes[0].metadata["kind"] == "service"
    assert nodes[0].metadata["url"] == "https://github.com/acme/repo.git"


def test_parse_spec_image_file_requires_ocr(tmp_path):
    from seed_cli.parsers import parse_spec
    
    img = tmp_path / "spec.png"
    img.write_bytes(b"not an image")
    
    # Should raise an error when trying to parse invalid image
    with pytest.raises((RuntimeError, ValueError, Exception), match="(OCR|image|PIL|Image)"):
        parse_spec(str(img))


def test_read_input_rejects_images(tmp_path):
    from seed_cli.parsers import read_input
    
    img = tmp_path / "spec.png"
    img.write_bytes(b"fake image")
    
    with pytest.raises(ValueError, match="Image file detected"):
        read_input(str(img))
