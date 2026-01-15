from pathlib import Path
from seed_cli.doctor import doctor
from seed_cli.parsers import Node


def test_doctor_duplicate():
    nodes = [
        Node(Path("a.txt"), False),
        Node(Path("a.txt"), False),
    ]
    issues = doctor(nodes, Path("."), fix=False)
    assert "duplicate: a.txt" in issues


def test_doctor_duplicate_fix():
    nodes = [
        Node(Path("a.txt"), False),
        Node(Path("a.txt"), False),
    ]
    issues = doctor(nodes, Path("."), fix=True)
    assert "duplicate: a.txt" in issues
    assert len(nodes) == 1


def test_doctor_invalid_annotation():
    nodes = [Node(Path("a.txt"), False, annotation="bad")]
    issues = doctor(nodes, Path("."), fix=False)
    assert "invalid annotation @bad on a.txt" in issues


def test_doctor_invalid_annotation_fix():
    nodes = [Node(Path("a.txt"), False, annotation="bad")]
    issues = doctor(nodes, Path("."), fix=True)
    assert nodes[0].annotation is None


def test_doctor_parent_file_conflict():
    nodes = [
        Node(Path("a"), False),
        Node(Path("a/b.txt"), False),
    ]
    issues = doctor(nodes, Path("."), fix=False)
    assert any("parent is file" in i for i in issues)
