

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict

from .base import StateBackend


STATE_DIR = ".seed"
STATE_FILE = "state.json"
LOCK_FILE = "lock.json"


class LocalStateError(RuntimeError):
    pass


class LocalStateBackend(StateBackend):
    def __init__(self, base: Path):
        self.base = base
        self.state_path = base / STATE_DIR / STATE_FILE
        self.lock_path = base / STATE_DIR / LOCK_FILE

    # -------- State --------

    def load(self) -> Dict:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text())

    def save(self, state: Dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    # -------- Locking --------

    def acquire_lock(self, ttl_seconds: int, timeout_seconds: int) -> Dict:
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            if not self.lock_path.exists():
                return self._create_lock(ttl_seconds)
            if self._lock_expired():
                self.force_unlock()
                return self._create_lock(ttl_seconds)
            time.sleep(0.2)

        raise LocalStateError("Timed out acquiring lock")

    def _create_lock(self, ttl_seconds: int) -> Dict:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock = {
            "lock_id": str(uuid.uuid4()),
            "pid": os.getpid(),
            "created_at": time.time(),
            "expires_at": time.time() + ttl_seconds,
        }

        self.lock_path.write_text(json.dumps(lock, indent=2))
        return lock

    def renew_lock(self, lock_id: str, ttl_seconds: int) -> None:
        lock = self.lock_status()
        if lock.get("lock_id") != lock_id:
            raise LocalStateError("Lock ID mismatch")

        lock["expires_at"] = time.time() + ttl_seconds
        self.lock_path.write_text(json.dumps(lock, indent=2))

    def release_lock(self, lock_id: str) -> None:
        lock = self.lock_status()
        if lock.get("lock_id") != lock_id:
            raise LocalStateError("Lock ID mismatch")
        self.force_unlock()

    def lock_status(self) -> Dict:
        if not self.lock_path.exists():
            return {}
        return json.loads(self.lock_path.read_text())

    def force_unlock(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()

    def _lock_expired(self) -> bool:
        lock = self.lock_status()
        return bool(lock) and lock.get("expires_at", 0) < time.time()
