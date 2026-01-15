from pathlib import Path
from seed_cli.checksums import sha256, load_checksums, save_checksums


def test_sha256(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h = sha256(f)
    assert isinstance(h, str)
    assert len(h) == 64


def test_load_missing(tmp_path):
    data = load_checksums(tmp_path)
    assert data == {}


def test_save_and_load(tmp_path):
    data = {
        "a.txt": {"sha256": "abc", "annotation": None}
    }
    save_checksums(tmp_path, data)
    loaded = load_checksums(tmp_path)
    assert loaded == data


def test_corrupt_file(tmp_path):
    p = tmp_path / ".seed" / "checksums.json"
    p.parent.mkdir()
    p.write_text("not json")
    loaded = load_checksums(tmp_path)
    assert loaded == {}
