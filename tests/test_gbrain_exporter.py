"""End-to-end tests for `seed export gbrain` (PRD M1 AC1-AC4)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from seed_cli.gbrain import export_gbrain


SAMPLE_SPEC = """
people/
└── <person>/ !person +contact
companies/
└── <company>/ !company
projects/
└── <name>/ !project +active
notes/        ...
vendor/
└── api/ !service +remote -> https://github.com/acme/api.git
""".strip()


def _write_spec(path: Path, text: str = SAMPLE_SPEC) -> Path:
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_export_writes_pack_yaml_and_audit_files(tmp_path):
    spec = _write_spec(tmp_path / "brain.seed")
    result = export_gbrain(
        spec=str(spec),
        base=tmp_path,
        skip_validate=True,
    )
    assert result.pack_path.exists()
    assert result.kindmap_lock_path.exists()
    assert result.source_json_path.exists()

    manifest = yaml.safe_load(result.pack_path.read_text())
    assert manifest["api_version"] == "gbrain-schema-pack-v1"
    assert manifest["extends"] == "gbrain-base"
    names = [pt["name"] for pt in manifest["page_types"]]
    assert "person" in names and "company" in names and "project" in names

    source = json.loads(result.source_json_path.read_text())
    assert source["spec_hash"] == result.spec_hash


def test_rerun_on_unchanged_spec_is_byte_identical(tmp_path):
    """AC2: deterministic compilation."""
    spec = _write_spec(tmp_path / "brain.seed")
    out_dir = tmp_path / "pack"
    result_a = export_gbrain(spec=str(spec), base=tmp_path, out=out_dir, skip_validate=True)
    text_a = result_a.pack_path.read_text()
    result_b = export_gbrain(spec=str(spec), base=tmp_path, out=out_dir, skip_validate=True)
    text_b = result_b.pack_path.read_text()
    assert text_a == text_b


def test_unknown_kind_does_not_fail_and_emits_diagnostic(tmp_path):
    """AC3: graceful default for unknown kinds."""
    spec = _write_spec(
        tmp_path / "brain.seed",
        "widgets/\n└── <widget>/ !novelty-kind",
    )
    result = export_gbrain(spec=str(spec), base=tmp_path, skip_validate=True)
    assert result.pack_path.exists()
    names = [pt["name"] for pt in result.manifest["page_types"]]
    assert "novelty-kind" in names
    assert any("novelty-kind" in d for d in result.diagnostics)


def test_install_copies_pack_to_home_root(tmp_path):
    spec = _write_spec(tmp_path / "brain.seed")
    home_root = tmp_path / "home_packs"
    result = export_gbrain(
        spec=str(spec),
        base=tmp_path,
        install=True,
        home_schema_packs=home_root,
        skip_validate=True,
    )
    assert (home_root / result.manifest["name"] / "pack.yaml").exists()


def test_activate_repo_writes_gbrain_yml(tmp_path):
    """AC4: --activate repo populates gbrain.yml schema: block (tier 5)."""
    spec = _write_spec(tmp_path / "brain.seed")
    result = export_gbrain(
        spec=str(spec),
        base=tmp_path,
        name="my-brain-pack",
        activate_mode="repo",
        skip_validate=True,
    )
    yml = tmp_path / "gbrain.yml"
    assert yml.exists()
    doc = yaml.safe_load(yml.read_text())
    assert doc["schema"]["pack"] == "my-brain-pack"
    assert result.activation is not None
    assert result.activation.repo_yaml == yml


def test_dry_run_writes_nothing(tmp_path):
    spec = _write_spec(tmp_path / "brain.seed")
    result = export_gbrain(spec=str(spec), base=tmp_path, dry_run=True, skip_validate=True)
    assert not result.pack_path.exists()
    assert not result.source_json_path.exists()
    assert result.manifest["page_types"]


def test_version_from_spec_uses_v_marker(tmp_path):
    spec = _write_spec(
        tmp_path / "brain.seed",
        "# spec v7\npeople/\n└── <person>/ !person",
    )
    result = export_gbrain(
        spec=str(spec),
        base=tmp_path,
        version_from="spec",
        skip_validate=True,
    )
    assert result.manifest["version"].startswith("7.")


def test_literal_version_override(tmp_path):
    spec = _write_spec(tmp_path / "brain.seed")
    result = export_gbrain(
        spec=str(spec),
        base=tmp_path,
        version_from="2.3.4",
        skip_validate=True,
    )
    assert result.manifest["version"] == "2.3.4"
