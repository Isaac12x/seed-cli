import time
from seed_cli.lock_heartbeat import LockHeartbeat


def test_heartbeat_calls_renew():
    calls = []

    def renew():
        calls.append(time.time())

    hb = LockHeartbeat(renew_fn=renew, interval=0.05)
    hb.start()
    time.sleep(0.12)
    hb.stop()

    assert len(calls) >= 2


def test_heartbeat_stop():
    calls = []

    def renew():
        calls.append(1)

    hb = LockHeartbeat(renew_fn=renew, interval=0.05)
    hb.start()
    time.sleep(0.08)
    hb.stop()
    count = len(calls)
    time.sleep(0.1)
    assert len(calls) == count
