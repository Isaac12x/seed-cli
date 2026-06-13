"""Reverse-drift (`seed amend`) tests (PRD M4, AC8-AC10)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from seed_cli.gbrain import amend


SPEC_BASE = """\
.
├── people/
│   └── <person>/ !person
└── projects/
    └── <name>/ !project
"""


def _write_spec(tmp_path: Path) -> Path:
    p = tmp_path / "brain.seed"
    p.write_text(SPEC_BASE, encoding="utf-8")
    return p


def test_amend_adopts_filesystem_drift_into_spec(tmp_path):
    """AC8: gbrain creates a folder; amend folds it back into the spec."""
    spec = _write_spec(tmp_path)
    # Simulate gbrain creating a new folder not in the spec.
    drift_dir = tmp_path / "incidents"
    drift_dir.mkdir()
    (drift_dir / "outage-2026-06-13.md").write_text("# outage\n", encoding="utf-8")

    result = amend(spec=str(spec), base=tmp_path, policy="adopt")

    assert any(c.path.startswith("incidents") for c in result.changes)
    assert result.spec_rewritten
    new_text = spec.read_text()
    assert "incidents" in new_text


def test_amend_quarantine_parks_drift_under_inbox(tmp_path):
    """AC10: quarantine policy parks unowned structure under the catch-all."""
    spec = _write_spec(tmp_path)
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "thing.md").write_text("x", encoding="utf-8")

    result = amend(
        spec=str(spec),
        base=tmp_path,
        policy="quarantine",
        quarantine_dir="_inbox/",
    )
    new_text = spec.read_text()
    assert "_inbox/" in new_text
    assert any(c.policy == "quarantine" for c in result.changes)


def test_amend_ignore_writes_persistent_ignore_file(tmp_path):
    spec = _write_spec(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "noise.log").write_text("x", encoding="utf-8")

    result = amend(spec=str(spec), base=tmp_path, policy="ignore")
    ignore_path = tmp_path / ".seed" / "gbrain" / "ignore.yml"
    assert ignore_path.exists()
    doc = yaml.safe_load(ignore_path.read_text())
    assert any(p.startswith("logs") for p in doc["ignore"])
    # Spec untouched
    assert spec.read_text() == SPEC_BASE


def test_amend_skips_already_ignored_paths(tmp_path):
    spec = _write_spec(tmp_path)
    ignore_dir = tmp_path / ".seed" / "gbrain"
    ignore_dir.mkdir(parents=True)
    (ignore_dir / "ignore.yml").write_text(
        yaml.safe_dump({"ignore": ["logs"]}),
        encoding="utf-8",
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "x.log").write_text("x", encoding="utf-8")

    result = amend(spec=str(spec), base=tmp_path, policy="adopt", dry_run=True)
    assert not any(c.path.startswith("logs") for c in result.changes)


def test_amend_from_gbrain_does_not_duplicate_known_types(tmp_path):
    """AC9: types already declared by gbrain shouldn't be re-created."""
    spec = _write_spec(tmp_path)
    (tmp_path / "meetings").mkdir()
    (tmp_path / "meetings" / "standup.md").write_text("x", encoding="utf-8")

    fake_pack = {
        "page_types": [
            {"name": "meeting", "path_prefixes": ["meetings/"]},
        ]
    }
    with patch("seed_cli.gbrain.amend.gbrain_cli.schema_show_json", return_value=fake_pack):
        result = amend(
            spec=str(spec),
            base=tmp_path,
            policy="adopt",
            from_fs=True,
            from_gbrain=True,
        )

    meetings = [c for c in result.changes if c.path.startswith("meetings")]
    assert meetings, "expected at least one drift entry for meetings/"
    for change in meetings:
        # the kind should be carried over from the active gbrain pack
        assert change.kind == "meeting"


def test_amend_reexport_regenerates_pack(tmp_path):
    """AC8 (continued): --reexport produces an updated pack manifest."""
    spec = _write_spec(tmp_path)
    (tmp_path / "incidents").mkdir()
    (tmp_path / "incidents" / "x.md").write_text("x", encoding="utf-8")

    result = amend(
        spec=str(spec),
        base=tmp_path,
        policy="adopt",
        reexport=True,
    )
    assert result.reexport_summary is not None
    assert (tmp_path / ".gbrain" / "pack" / "pack.yaml").exists()


def test_amend_dry_run_does_not_write(tmp_path):
    spec = _write_spec(tmp_path)
    original = spec.read_text()
    (tmp_path / "stray").mkdir()

    result = amend(spec=str(spec), base=tmp_path, policy="adopt", dry_run=True)
    assert result.changes
    assert not result.spec_rewritten
    assert spec.read_text() == original
