import pytest
from seed_cli.schema import validate_document, SchemaError


def test_valid_schema():
    doc = {
        "entries": [
            {"path": "a/", "type": "dir"},
            {"path": "a/file.txt", "type": "file", "annotation": "manual"},
        ]
    }
    validate_document(doc)


def test_missing_entries():
    with pytest.raises(SchemaError):
        validate_document({})


def test_invalid_entries_type():
    with pytest.raises(SchemaError):
        validate_document({"entries": {}})


def test_missing_path():
    with pytest.raises(SchemaError):
        validate_document({"entries": [{}]})


def test_invalid_type():
    with pytest.raises(SchemaError):
        validate_document({"entries": [{"path": "x", "type": "bad"}]})


def test_invalid_annotation():
    with pytest.raises(SchemaError):
        validate_document({"entries": [{"path": "x", "annotation": "bad"}]})


def test_invalid_comment():
    with pytest.raises(SchemaError):
        validate_document({"entries": [{"path": "x", "comment": 123}]})
