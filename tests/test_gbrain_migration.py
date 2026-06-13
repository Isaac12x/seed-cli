"""Forward-migration (successor pack) tests (PRD M3)."""

from __future__ import annotations

from pathlib import Path

import yaml

from seed_cli.gbrain import export_gbrain
from seed_cli.gbrain.migration import compute_mapping_rules, parse_spec_text


def _spec(tmp_path: Path, body: str, name: str = "brain.seed") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


SPEC_V1 = """\
.
├── people/
│   └── <person>/ !person
└── projects/
    └── <name>/ !project
"""

SPEC_V2_RENAMED = """\
.
├── people/
│   └── <person>/ !person
└── deals/
    └── <name>/ !deal
"""


def test_compute_mapping_rules_removed_prefix_falls_back_to_note(tmp_path):
    prev = parse_spec_text(SPEC_V1)
    new = parse_spec_text(SPEC_V2_RENAMED)
    rules = compute_mapping_rules(prev, new, name="x")
    # projects/ disappeared -> project should be retyped to note
    fall = next((r for r in rules if r["from_type"] == "project" and r["to_type"] == "note"), None)
    assert fall is not None
    assert fall["path_filter"] == "projects/%"


def test_export_with_predecessor_spec_emits_migration_block(tmp_path):
    """AC implicit: a captured prior version triggers successor emission."""
    # Seed prior version into .seed/specs/
    specs_dir = tmp_path / ".seed" / "specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "v1.tree").write_text(SPEC_V1, encoding="utf-8")
    # current spec on disk
    current = _spec(tmp_path, SPEC_V2_RENAMED)
    result = export_gbrain(
        spec=str(current),
        base=tmp_path,
        name="brain-pack",
        migrate="prompt",
        skip_validate=True,
    )
    assert result.manifest.get("migration_from") == {"pack": "brain-pack", "version": "0.1.x"}
    rules = result.manifest.get("mapping_rules") or []
    # catch-all is always present
    assert any(r.get("from_type") == "*unknown*" for r in rules)
    # plus the projects/ -> note retype
    assert any(r.get("from_type") == "project" and r.get("to_type") == "note" for r in rules)
    # message hints at unify-types
    assert "unify-types" in (result.migration_message or "")


def test_migrate_off_skips_predecessor_lookup(tmp_path):
    specs_dir = tmp_path / ".seed" / "specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "v1.tree").write_text(SPEC_V1, encoding="utf-8")
    current = _spec(tmp_path, SPEC_V2_RENAMED)
    result = export_gbrain(
        spec=str(current),
        base=tmp_path,
        migrate="off",
        skip_validate=True,
    )
    assert "migration_from" not in result.manifest


def test_migrate_from_explicit_path(tmp_path):
    prior = tmp_path / "prior.seed"
    prior.write_text(SPEC_V1, encoding="utf-8")
    current = _spec(tmp_path, SPEC_V2_RENAMED)
    result = export_gbrain(
        spec=str(current),
        base=tmp_path,
        migrate="prompt",
        migrate_from=str(prior),
        skip_validate=True,
    )
    assert result.manifest.get("migration_from", {}).get("version") == "prior"


def test_identical_spec_does_not_emit_migration_block(tmp_path):
    specs_dir = tmp_path / ".seed" / "specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "v1.tree").write_text(SPEC_V1, encoding="utf-8")
    current = _spec(tmp_path, SPEC_V1)
    result = export_gbrain(
        spec=str(current),
        base=tmp_path,
        migrate="prompt",
        skip_validate=True,
    )
    assert "migration_from" not in result.manifest
