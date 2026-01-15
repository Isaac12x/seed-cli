from pathlib import Path
import pytest
from seed_cli.state.local import LocalStateBackend, LocalStateError


def test_acquire_and_release(tmp_path):
    backend = LocalStateBackend(tmp_path)
    lock_info = backend.acquire_lock(ttl_seconds=10, timeout_seconds=1)
    assert lock_info.get("lock_id")
    assert (tmp_path / ".seed" / "lock.json").exists()
    
    backend.release_lock(lock_info["lock_id"])
    assert not (tmp_path / ".seed" / "lock.json").exists()


def test_double_lock_fails(tmp_path):
    backend = LocalStateBackend(tmp_path)
    lock_info = backend.acquire_lock(ttl_seconds=10, timeout_seconds=1)
    
    with pytest.raises(LocalStateError, match="Timed out"):
        backend.acquire_lock(ttl_seconds=10, timeout_seconds=0.1)
    
    backend.release_lock(lock_info["lock_id"])
