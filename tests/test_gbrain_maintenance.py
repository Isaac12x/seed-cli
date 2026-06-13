"""Tests for the `gbrain-sync` maintenance goal and brain hooks (PRD M2)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from seed_cli.maintenance import build_maintenance_plan
from seed_cli.templates import (
    install_gbrain_post_apply_hook,
    install_gbrain_pre_push_hook,
)


def _write_manifest(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_gbrain_sync_emits_export_and_reindex_steps(tmp_path):
    brain_path = tmp_path / "brain"
    brain_path.mkdir()
    manifest = tmp_path / "maintenance.yml"
    _write_manifest(manifest, {
        "targets": [
            {
                "name": "my-brain",
                "kind": "project",
                "path": str(brain_path),
                "goals": ["gbrain-sync"],
                "gbrain": {
                    "spec": "brain.seed",
                    "name": "my-brain-pack",
                    "activate": "repo",
                    "migrate": "off",
                    "run_sync": True,
                },
            }
        ]
    })
    plan = build_maintenance_plan(manifest)
    names = [step.name for step in plan.steps]
    assert "gbrain-export" in names
    assert "gbrain-reindex" in names
    assert "gbrain-unify-types" not in names


def test_gbrain_sync_migrate_auto_emits_unify_types(tmp_path):
    brain_path = tmp_path / "brain"
    brain_path.mkdir()
    manifest = tmp_path / "maintenance.yml"
    _write_manifest(manifest, {
        "targets": [
            {
                "name": "my-brain",
                "kind": "project",
                "path": str(brain_path),
                "goals": ["gbrain-sync"],
                "gbrain": {
                    "spec": "brain.seed",
                    "name": "my-brain-pack",
                    "activate": "repo",
                    "migrate": "auto",
                    "run_sync": False,
                },
            }
        ]
    })
    plan = build_maintenance_plan(manifest)
    unify = next((s for s in plan.steps if s.name == "gbrain-unify-types"), None)
    assert unify is not None
    assert unify.argv[:5] == ["gbrain", "jobs", "submit", "unify-types", "--allow-protected"]
    payload = json.loads(unify.argv[unify.argv.index("--params") + 1])
    assert payload == {"target_pack": "my-brain-pack"}


def test_gbrain_sync_export_command_carries_pack_options(tmp_path):
    brain_path = tmp_path / "brain"
    brain_path.mkdir()
    manifest = tmp_path / "maintenance.yml"
    _write_manifest(manifest, {
        "targets": [
            {
                "name": "x",
                "kind": "project",
                "path": str(brain_path),
                "goals": ["gbrain-sync"],
                "gbrain": {
                    "spec": "brain.seed",
                    "name": "x-pack",
                    "extends": "gbrain-base",
                    "activate": "both",
                    "kindmap": ".seed/gbrain/kindmap.yml",
                    "version_from": "hash",
                    "run_sync": False,
                },
            }
        ]
    })
    plan = build_maintenance_plan(manifest)
    export = next(s for s in plan.steps if s.name == "gbrain-export")
    assert export.argv[:5] == ["seed", "export", "gbrain", "brain.seed", "--install"]
    assert "--activate" in export.argv and "both" in export.argv
    assert "--name" in export.argv and "x-pack" in export.argv
    assert "--extends" in export.argv and "gbrain-base" in export.argv
    assert "--kindmap" in export.argv
    assert "--version-from" in export.argv


def test_install_gbrain_post_apply_hook_writes_executable_script(tmp_path):
    path = install_gbrain_post_apply_hook(
        tmp_path,
        spec="brain.seed",
        name="my-pack",
        activate_mode="repo",
    )
    assert path.exists()
    text = path.read_text()
    assert "SEED_HOOK_STAGE" in text
    assert "post_apply" in text
    assert "brain.seed" in text
    assert "my-pack" in text
    assert path.stat().st_mode & 0o111


def test_install_gbrain_pre_push_hook_writes_git_hook(tmp_path):
    git = tmp_path / ".git" / "hooks"
    git.mkdir(parents=True)
    path = install_gbrain_pre_push_hook(tmp_path)
    assert path == git / "pre-push"
    assert path.exists()
    text = path.read_text()
    assert "stale" in text.lower()
    assert path.stat().st_mode & 0o111


def test_pre_push_hook_blocks_when_spec_hash_diverges(tmp_path):
    """AC7: pre-push hook fails when the committed pack hash != current spec hash."""
    git = tmp_path / ".git" / "hooks"
    git.mkdir(parents=True)
    install_gbrain_pre_push_hook(
        tmp_path,
        spec="brain.seed",
        pack=".gbrain/pack/pack.yaml",
        source_json=".gbrain/pack/source.json",
    )
    (tmp_path / "brain.seed").write_text("people/\n", encoding="utf-8")
    pack_dir = tmp_path / ".gbrain" / "pack"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.yaml").write_text("api_version: gbrain-schema-pack-v1\n")
    (pack_dir / "source.json").write_text(
        json.dumps({"spec_hash": "deadbeef"}),
        encoding="utf-8",
    )
    import subprocess
    result = subprocess.run(
        ["/bin/sh", str(git / "pre-push")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "stale" in (result.stderr + result.stdout).lower()
