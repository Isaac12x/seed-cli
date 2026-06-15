import json
from pathlib import Path

import pytest

from seed_cli import conversion
from seed_cli.parsers import Node, parse_spec, parse_tree_text


def _semantic_nodes(nodes):
    return {
        (
            node.relpath.as_posix(),
            node.is_dir,
            node.comment,
            node.annotation,
            node.optional,
            json.dumps(node.metadata, sort_keys=True),
        )
        for node in nodes
    }


def test_render_compact_seed_groups_shared_extensions():
    nodes = [
        Node(Path("memories/global.jsonl"), False),
        Node(Path("memories/facts.jsonl"), False),
        Node(Path("memories/episodes.jsonl"), False),
    ]

    assert (
        conversion.render_compact_seed(nodes)
        == "memories/{episodes,facts,global}.jsonl\n"
    )


def test_render_compact_seed_collapses_chains_and_mixed_leaves():
    nodes = [
        Node(Path("services/<service-id>/service.json"), False),
        Node(Path("services/<service-id>/knowledge"), True),
        Node(Path("services/<service-id>/prompts"), True),
        Node(Path("services/<service-id>/tools.json"), False),
    ]

    assert conversion.render_compact_seed(nodes) == (
        "services/<service-id>/{knowledge/,prompts/,service.json,tools.json}\n"
    )


def test_render_compact_seed_groups_directory_siblings():
    nodes = [
        Node(Path("common/global/knowledge"), True),
        Node(Path("common/global/policies"), True),
        Node(Path("common/global/prompts"), True),
    ]

    assert (
        conversion.render_compact_seed(nodes)
        == "common/global/{knowledge,policies,prompts}/\n"
    )


def test_render_compact_seed_splits_mixed_siblings_when_suffix_factoring_is_shorter():
    nodes = [
        Node(Path("memories/global.jsonl"), False),
        Node(Path("memories/facts.jsonl"), False),
        Node(Path("memories/episodes.jsonl"), False),
        Node(Path("memories/preferences.jsonl"), False),
        Node(Path("memories/relationships.jsonl"), False),
        Node(Path("memories/corrections.jsonl"), False),
        Node(Path("memories/embeddings"), True),
        Node(Path("memories/indexes"), True),
    ]

    assert conversion.render_compact_seed(nodes) == (
        "memories/{corrections,episodes,facts,global,preferences,relationships}.jsonl\n"
        "memories/{embeddings,indexes}/\n"
    )


def test_render_compact_seed_collapses_single_path_chain():
    nodes = [
        Node(
            Path("conversations/whatsapp/YYYY/MM/session_<id>.jsonl"),
            False,
        )
    ]

    assert conversion.render_compact_seed(nodes) == (
        "conversations/whatsapp/YYYY/MM/session_<id>.jsonl\n"
    )


def test_render_compact_seed_groups_compatible_metadata():
    nodes = parse_tree_text(
        "vendor/api/ !service +remote @manual (shared)\n"
        "vendor/worker/ !service +remote @manual (shared)\n"
    )

    assert conversion.render_compact_seed(nodes) == (
        "vendor/{api,worker}/ @manual !service +remote (shared)\n"
    )


def test_render_compact_seed_keeps_incompatible_metadata_separate():
    nodes = parse_tree_text(
        "vendor/api/ !service\n"
        "vendor/worker/ !worker\n"
    )

    assert conversion.render_compact_seed(nodes) == (
        "vendor/api/ !service\n"
        "vendor/worker/ !worker\n"
    )


def test_render_compact_seed_omits_plain_redundant_parent_directories():
    nodes = [
        Node(Path("memories"), True),
        Node(Path("memories/global.jsonl"), False),
        Node(Path("memories/facts.jsonl"), False),
    ]

    assert conversion.render_compact_seed(nodes) == (
        "memories/{facts,global}.jsonl\n"
    )


def test_render_compact_seed_preserves_meaningful_parent_directories():
    nodes = [
        Node(Path("vendor"), True, metadata={"kind": "service"}),
        Node(Path("vendor/README.md"), False),
    ]

    assert conversion.render_compact_seed(nodes) == (
        "vendor/ !service\n"
        "vendor/README.md\n"
    )


def test_render_compact_seed_is_deterministic_and_deduplicates_nodes():
    nodes = [
        Node(Path("b.txt"), False),
        Node(Path("a.txt"), False),
        Node(Path("b.txt"), False),
    ]

    assert conversion.render_compact_seed(nodes) == "{a,b}.txt\n"


def test_render_compact_seed_empty_input_is_empty():
    assert conversion.render_compact_seed([]) == ""


def test_render_compact_seed_round_trips_semantic_nodes():
    source = (
        "memories/global.jsonl @manual !memory +shared\n"
        "memories/facts.jsonl @manual !memory +shared\n"
        "services/api/ !service\n"
        "services/worker/ !worker\n"
    )
    source_nodes = parse_tree_text(source)
    rendered = conversion.render_compact_seed(source_nodes)
    reparsed_nodes = parse_tree_text(rendered)

    assert _semantic_nodes(reparsed_nodes) == _semantic_nodes(source_nodes)


def test_render_compact_seed_matches_requested_hierarchy_shape():
    source = """~/.hermes/gbrain/
├── common/
│   ├── global/
│   │   ├── knowledge/
│   │   ├── policies/
│   │   └── prompts/
│   └── services/
│       └── <service-id>/
│           ├── service.json
│           ├── knowledge/
│           ├── prompts/
│           └── tools.json
└── people/
    └── <person_id>/
        ├── identity.json
        ├── entitlements.json
        ├── allowance.json
        ├── profile.md
        ├── communication_style.md
        ├── system_overlay.md
        ├── memories/
        │   ├── global.jsonl
        │   ├── facts.jsonl
        │   ├── episodes.jsonl
        │   ├── preferences.jsonl
        │   ├── relationships.jsonl
        │   ├── corrections.jsonl
        │   ├── embeddings/
        │   └── indexes/
        ├── conversations/
        │   └── whatsapp/
        │       └── YYYY/
        │           └── MM/
        │               └── session_<id>.jsonl
        ├── services/
        │   └── <service-id>/
        │       ├── allowance.json
        │       ├── memory.jsonl
        │       ├── conversations/
        │       ├── files/
        │       ├── state/
        │       └── outputs/
        ├── gbrain/
        │   ├── graph.sqlite
        │   ├── vector.sqlite|chroma/
        │   └── recall_cache.json
        └── audit/
            ├── billing_events.jsonl
            ├── memory_writes.jsonl
            └── service_access.jsonl
"""

    assert conversion.render_compact_seed(parse_tree_text(source)) == (
        "common/global/{knowledge,policies,prompts}/\n"
        "common/services/<service-id>/\n"
        "common/services/<service-id>/{knowledge/,prompts/,service.json,tools.json}\n"
        "people/<person_id>/\n"
        "people/<person_id>/audit/{billing_events,memory_writes,service_access}.jsonl\n"
        "people/<person_id>/conversations/whatsapp/YYYY/MM/session_<id>.jsonl\n"
        "people/<person_id>/gbrain/{graph.sqlite,recall_cache.json,vector.sqlite|chroma/}\n"
        "people/<person_id>/memories/"
        "{corrections,episodes,facts,global,preferences,relationships}.jsonl\n"
        "people/<person_id>/memories/{embeddings,indexes}/\n"
        "people/<person_id>/services/<service-id>/\n"
        "people/<person_id>/services/<service-id>/"
        "{allowance.json,conversations/,files/,memory.jsonl,outputs/,state/}\n"
        "people/<person_id>/"
        "{allowance.json,communication_style.md,entitlements.json,"
        "identity.json,profile.md,system_overlay.md}\n"
    )


def test_convert_tree_to_seed_defaults_to_same_stem(tmp_path):
    source = tmp_path / "nested" / "brain.tree"
    source.parent.mkdir()
    source.write_text(
        "memories/global.jsonl\nmemories/facts.jsonl\n",
        encoding="utf-8",
    )

    output = conversion.convert_tree_to_seed(source)

    assert output == source.with_suffix(".seed")
    assert output.read_text(encoding="utf-8") == (
        "memories/{facts,global}.jsonl\n"
    )


def test_convert_tree_to_seed_writes_explicit_nested_output(tmp_path):
    source = tmp_path / "brain.tree"
    output = tmp_path / "generated" / "brain.seed"
    source.write_text("a.txt\nb.txt\n", encoding="utf-8")

    result = conversion.convert_tree_to_seed(source, output)

    assert result == output
    assert output.read_text(encoding="utf-8") == "{a,b}.txt\n"


def test_convert_tree_to_seed_output_round_trips_through_parse_spec(tmp_path):
    source = tmp_path / "brain.tree"
    source.write_text(
        "audit/billing_events.jsonl\n"
        "audit/memory_writes.jsonl\n"
        "audit/service_access.jsonl\n",
        encoding="utf-8",
    )

    output = conversion.convert_tree_to_seed(source)
    _, source_nodes = parse_spec(str(source), base=tmp_path)
    _, output_nodes = parse_spec(str(output), base=tmp_path)

    assert _semantic_nodes(output_nodes) == _semantic_nodes(source_nodes)


def test_convert_tree_to_seed_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="Tree input file not found"):
        conversion.convert_tree_to_seed(tmp_path / "missing.tree")


def test_convert_tree_to_seed_rejects_directory_input(tmp_path):
    source = tmp_path / "directory.tree"
    source.mkdir()

    with pytest.raises(ValueError, match="must be a file"):
        conversion.convert_tree_to_seed(source)


def test_convert_tree_to_seed_rejects_non_tree_input(tmp_path):
    source = tmp_path / "brain.txt"
    source.write_text("a.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must use the \.tree suffix"):
        conversion.convert_tree_to_seed(source)


def test_convert_tree_to_seed_rejects_non_seed_output(tmp_path):
    source = tmp_path / "brain.tree"
    source.write_text("a.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must use the \.seed suffix"):
        conversion.convert_tree_to_seed(source, tmp_path / "brain.txt")


@pytest.mark.parametrize("suffix", [".tree", ".seed"])
def test_collapse_spec_accepts_tree_and_seed_files(tmp_path, suffix):
    source = tmp_path / f"brain{suffix}"
    source.write_text(
        "memories/global.jsonl\n"
        "memories/facts.jsonl\n"
        "memories/episodes.jsonl\n",
        encoding="utf-8",
    )

    assert conversion.collapse_spec(source) == (
        "memories/{episodes,facts,global}.jsonl\n"
    )


def test_collapse_spec_writes_explicit_tree_or_seed_output(tmp_path):
    source = tmp_path / "brain.seed"
    output = tmp_path / "shared" / "brain.tree"
    source.write_text("a.txt\nb.txt\n", encoding="utf-8")

    rendered = conversion.collapse_spec(source, output)

    assert rendered == "{a,b}.txt\n"
    assert output.read_text(encoding="utf-8") == rendered


def test_collapse_spec_rejects_non_tree_like_input(tmp_path):
    source = tmp_path / "brain.txt"
    source.write_text("a.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must use a \.tree or \.seed suffix"):
        conversion.collapse_spec(source)


def test_collapse_spec_rejects_non_tree_like_output(tmp_path):
    source = tmp_path / "brain.seed"
    source.write_text("a.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must use a \.tree or \.seed suffix"):
        conversion.collapse_spec(source, tmp_path / "brain.txt")
