# src/seed_cli/schema.py


"""seed_cli.schema

Schema validation for structured specs (YAML / JSON).
"""

from typing import Dict, Any

VALID_TYPES = {"file", "dir"}
VALID_ANNOTATIONS = {"manual", "generated"}


class SchemaError(ValueError):
    pass


def _err(msg: str) -> None:
    raise SchemaError(msg)


def validate_document(doc: Dict[str, Any]) -> None:
    if not isinstance(doc, dict):
        _err("document must be an object")

    if "entries" not in doc:
        _err("document must contain 'entries'")

    entries = doc["entries"]
    if not isinstance(entries, list):
        _err("'entries' must be a list")

    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            _err(f"entry {i} must be an object")

        if "path" not in e:
            _err(f"entry {i} missing 'path'")

        path = e["path"]
        if not isinstance(path, str) or not path:
            _err(f"entry {i} 'path' must be a non-empty string")

        if "type" in e:
            t = e["type"]
            if t not in VALID_TYPES:
                _err(f"entry {i} invalid type '{t}' (must be file|dir)")

        if "annotation" in e:
            a = e["annotation"]
            if a not in VALID_ANNOTATIONS:
                _err(f"entry {i} invalid annotation '{a}'")

        if "comment" in e and not isinstance(e["comment"], str):
            _err(f"entry {i} 'comment' must be string")
