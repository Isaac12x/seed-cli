from seed_cli.image import read_tree, tree_lines_to_text

TREE_IMAGE = "/mnt/data/pseo-tree.jpeg"
EXPECTED_TREE = "/mnt/data/pseo-correct.tree"


def test_tree_matches_expected():
    result = read_tree_image(TREE_IMAGE)
    text = tree_to_text(result)

    with open(EXPECTED_TREE, "r", encoding="utf-8") as f:
        expected = f.read().strip()

    assert text.strip() == expected


def test_confidence_and_depth_present():
    result = read_tree_image(TREE_IMAGE)

    for line in result:
        assert "confidence" in line
        assert "depth" in line
        assert 0.0 <= line["confidence"] <= 1.0
        assert isinstance(line["depth"], int)
