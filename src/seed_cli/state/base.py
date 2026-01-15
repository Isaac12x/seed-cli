

from abc import ABC, abstractmethod
from typing import Dict


class StateBackend(ABC):
    """
    Abstract backend for state + locking.
    """

    # -------- State --------

    @abstractmethod
    def load(self) -> Dict:
        """Load persisted state."""
        raise NotImplementedError

    @abstractmethod
    def save(self, state: Dict) -> None:
        """Persist state."""
        raise NotImplementedError

    # -------- Locking --------

    @abstractmethod
    def acquire_lock(
        self,
        ttl_seconds: int,
        timeout_seconds: int,
    ) -> Dict:
        """
        Acquire a lock.

        Returns:
            dict with at least: { lock_id, expires_at }
        """
        raise NotImplementedError

    @abstractmethod
    def renew_lock(self, lock_id: str, ttl_seconds: int) -> None:
        """Renew an existing lock."""
        raise NotImplementedError

    @abstractmethod
    def release_lock(self, lock_id: str) -> None:
        """Release a lock."""
        raise NotImplementedError

    @abstractmethod
    def lock_status(self) -> Dict:
        """Return current lock status."""
        raise NotImplementedError

    @abstractmethod
    def force_unlock(self) -> None:
        """Forcefully remove the lock."""
        raise NotImplementedError
