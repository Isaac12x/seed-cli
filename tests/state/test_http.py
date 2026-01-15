import requests
import pytest
from seed_cli.state.http import HTTPStateBackend, HTTPStateError


class DummyResp:
    def __init__(self, code=200, text="ok", json_data=None):
        self.status_code = code
        self.text = text
        self._json_data = json_data or {"lock_id": "test-123", "expires_at": 1000}
    
    def json(self):
        return self._json_data


def test_http_lock_acquire(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: DummyResp())
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: DummyResp())
    backend = HTTPStateBackend("http://example.com")
    lock_info = backend.acquire_lock(ttl_seconds=10, timeout_seconds=1)
    assert lock_info.get("lock_id")


def test_http_lock_acquire_fail(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **kwargs: DummyResp(500, "err"))
    backend = HTTPStateBackend("http://example.com")
    with pytest.raises(HTTPStateError):
        backend.acquire_lock(ttl_seconds=10, timeout_seconds=1)
