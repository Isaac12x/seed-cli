

"""seed_cli.lock_heartbeat

State lock heartbeat to prevent stale locks during long-running operations.

This module:
- periodically updates a lock's timestamp
- runs in a background thread
- can be stopped cleanly

Used by:
- apply
- sync
- plan (optional, for remote state)
"""

import threading
import time
from typing import Callable, Optional


class HeartbeatError(RuntimeError):
    pass


class LockHeartbeat:
    def __init__(
        self,
        renew_fn: Callable[[], None],
        *,
        interval: float = 5.0,
    ) -> None:
        self._renew_fn = renew_fn
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._renew_fn()
            except Exception as e:
                raise HeartbeatError("Lock heartbeat renewal failed") from e

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._interval * 2)
