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
