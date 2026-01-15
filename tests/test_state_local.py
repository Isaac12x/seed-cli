import time
from pathlib import Path
from seed_cli.state.local import LocalStateBackend, LocalStateError


def test_state_load_save(tmp_path):
    b = LocalStateBackend(tmp_path)
    b.save({"a": 1})
    assert b.load() == {"a": 1}


def test_lock_acquire_and_release(tmp_path):
    b = LocalStateBackend(tmp_path)
    info = b.acquire_lock(ttl_seconds=1, timeout_seconds=1)
    assert info["lock_id"]
    status = b.lock_status()
    assert bool(status) is True  # Lock exists
    assert status.get("lock_id") == info["lock_id"]

    b.release_lock(info["lock_id"])
    status2 = b.lock_status()
    assert bool(status2) is False  # Lock doesn't exist


def test_lock_expiry(tmp_path):
    b = LocalStateBackend(tmp_path)
    info = b.acquire_lock(ttl_seconds=0.2, timeout_seconds=1)
    time.sleep(0.3)
    status = b.lock_status()
    # Check if lock is expired by comparing expires_at to current time
    assert status.get("expires_at", 0) < time.time()

    # next acquire should steal
    info2 = b.acquire_lock(ttl_seconds=1, timeout_seconds=1)
    assert info2["lock_id"] != info["lock_id"]


def test_lock_renew(tmp_path):
    b = LocalStateBackend(tmp_path)
    info = b.acquire_lock(ttl_seconds=0.2, timeout_seconds=1)
    b.renew_lock(info["lock_id"], ttl_seconds=1)
    status = b.lock_status()
    assert status["expires_at"] > time.time()


def test_force_unlock(tmp_path):
    b = LocalStateBackend(tmp_path)
    b.acquire_lock(ttl_seconds=10, timeout_seconds=1)
    b.force_unlock()
    status = b.lock_status()
    assert bool(status) is False  # Lock doesn't exist


def test_lock_timeout(tmp_path):
    b1 = LocalStateBackend(tmp_path)
    b2 = LocalStateBackend(tmp_path)
    info = b1.acquire_lock(ttl_seconds=5, timeout_seconds=1)

    try:
        b2.acquire_lock(ttl_seconds=1, timeout_seconds=0.2)
        assert False, "expected timeout"
    except (TimeoutError, LocalStateError):
        pass

    b1.release_lock(info["lock_id"])
