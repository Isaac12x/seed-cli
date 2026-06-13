"""Compiler unit tests for the gbrain exporter (PRD M1 AC1-AC3)."""

from pathlib import Path

from seed_cli.gbrain.compiler import compile_pack
from seed_cli.gbrain.kindmap import load_kindmap
from seed_cli.gbrain.manifest import dump_manifest, lint_manifest
from seed_cli.parsers import Node


def _node(rel: str, *, kind: str | None = None, tags=(), annotation=None) -> Node:
    is_dir = rel.endswith("/")
    metadata: dict = {}
    if kind:
        metadata["kind"] = kind
    if tags:
        metadata["tags"] = list(tags)
    return Node(
        relpath=Path(rel.rstrip("/")),
        is_dir=is_dir,
        annotation=annotation,
        metadata=metadata,
    )


def test_compiles_typed_placeholder_dirs_into_page_types():
    nodes = [
        _node("people/"),
        _node("people/<person>/", kind="person", tags=["contact"], annotation="template:person"),
        _node("companies/"),
        _node("companies/<company>/", kind="company", annotation="template:company"),
        _node("projects/"),
        _node("projects/<name>/", kind="project", tags=["active"], annotation="template:name"),
    ]
    pack = compile_pack(nodes, name="brain-pack", version="0.0.1+abc12345")
    names = [pt["name"] for pt in pack.manifest["page_types"]]
    assert "person" in names
    assert "company" in names
    assert "project" in names

    by_name = {pt["name"]: pt for pt in pack.manifest["page_types"]}
    assert by_name["person"]["primitive"] == "entity"
    assert by_name["person"]["expert_routing"] is True
    assert by_name["person"]["path_prefixes"] == ["people/"]
    assert by_name["company"]["path_prefixes"] == ["companies/"]
    assert by_name["project"]["primitive"] == "concept"


def test_unknown_kind_degrades_to_derived_concept_with_diagnostic():
    nodes = [
        _node("widgets/"),
        _node("widgets/foo/", kind="quux-thing"),
    ]
    pack = compile_pack(nodes, name="x", version="0.0.1")
    names = [pt["name"] for pt in pack.manifest["page_types"]]
    assert "quux-thing" in names
    derived = next(pt for pt in pack.manifest["page_types"] if pt["name"] == "quux-thing")
    assert derived["primitive"] == "concept"
    assert any("quux-thing" in d for d in pack.diagnostics)


def test_unknown_kind_never_aborts_compilation():
    nodes = [_node("foo/"), _node("foo/bar/", kind="totally-novel")]
    pack = compile_pack(nodes, name="x", version="0.0.1")
    assert pack.manifest["page_types"], "expected at least one page type"


def test_catch_all_mapping_rule_is_always_emitted():
    pack = compile_pack([_node("notes/")], name="x", version="0.0.1")
    rules = pack.manifest["mapping_rules"]
    assert any(
        r.get("from_type") == "*unknown*" and r.get("to_type") == "note"
        for r in rules
    )


def test_deterministic_output_for_unchanged_input():
    nodes = [
        _node("people/"),
        _node("people/<person>/", kind="person", annotation="template:person"),
        _node("companies/"),
        _node("companies/<company>/", kind="company", annotation="template:company"),
    ]
    pack_a = compile_pack(nodes, name="x", version="0.0.1+abc12345")
    pack_b = compile_pack(nodes, name="x", version="0.0.1+abc12345")
    assert dump_manifest(pack_a.manifest) == dump_manifest(pack_b.manifest)


def test_lint_accepts_compiler_output():
    nodes = [_node("people/"), _node("people/<person>/", kind="person", annotation="template:person")]
    pack = compile_pack(nodes, name="x", version="0.0.1")
    assert lint_manifest(pack.manifest) == []


def test_kindmap_override_via_extra_path(tmp_path):
    kindmap_path = tmp_path / "kindmap.yml"
    kindmap_path.write_text(
        "widget:\n  type: widget\n  primitive: media\n  extractable: true\n"
    )
    nodes = [_node("widgets/"), _node("widgets/x/", kind="widget")]
    pack = compile_pack(
        nodes,
        name="x",
        version="0.0.1",
        kindmap=load_kindmap(extra_path=kindmap_path),
    )
    widget = next(pt for pt in pack.manifest["page_types"] if pt["name"] == "widget")
    assert widget["primitive"] == "media"
    assert widget["extractable"] is True


def test_top_level_typed_dir_uses_own_prefix():
    nodes = [_node("notes/", kind="note")]
    pack = compile_pack(nodes, name="x", version="0.0.1")
    note = next(pt for pt in pack.manifest["page_types"] if pt["name"] == "note")
    assert note["path_prefixes"] == ["notes/"]


def test_duplicate_prefixes_dedup_within_same_type():
    nodes = [
        _node("people/"),
        _node("people/<person>/", kind="person", annotation="template:person"),
        _node("people/<contact>/", kind="contact", annotation="template:contact"),
    ]
    pack = compile_pack(nodes, name="x", version="0.0.1")
    person = next(pt for pt in pack.manifest["page_types"] if pt["name"] == "person")
    assert person["path_prefixes"] == ["people/"]
