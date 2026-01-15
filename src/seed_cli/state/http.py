

import requests
from typing import Dict

from .base import StateBackend


class HTTPStateError(RuntimeError):
    pass


class HTTPStateBackend(StateBackend):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    # -------- State --------

    def load(self) -> Dict:
        r = requests.get(f"{self.base_url}/state")
        if r.status_code != 200:
            raise HTTPStateError(r.text)
        return r.json()

    def save(self, state: Dict) -> None:
        r = requests.put(f"{self.base_url}/state", json=state)
        if r.status_code != 200:
            raise HTTPStateError(r.text)

    # -------- Locking --------

    def acquire_lock(self, ttl_seconds: int, timeout_seconds: int) -> Dict:
        r = requests.post(
            f"{self.base_url}/lock",
            json={"ttl": ttl_seconds, "timeout": timeout_seconds},
        )
        if r.status_code != 200:
            raise HTTPStateError(r.text)
        return r.json()

    def renew_lock(self, lock_id: str, ttl_seconds: int) -> None:
        r = requests.post(
            f"{self.base_url}/lock/renew",
            json={"lock_id": lock_id, "ttl": ttl_seconds},
        )
        if r.status_code != 200:
            raise HTTPStateError(r.text)

    def release_lock(self, lock_id: str) -> None:
        r = requests.post(
            f"{self.base_url}/lock/release",
            json={"lock_id": lock_id},
        )
        if r.status_code != 200:
            raise HTTPStateError(r.text)

    def lock_status(self) -> Dict:
        r = requests.get(f"{self.base_url}/lock")
        if r.status_code != 200:
            raise HTTPStateError(r.text)
        return r.json()

    def force_unlock(self) -> None:
        r = requests.delete(f"{self.base_url}/lock")
        if r.status_code != 200:
            raise HTTPStateError(r.text)
