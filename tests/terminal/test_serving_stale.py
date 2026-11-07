"""#3977 #4872: a managed-runtime record whose owner host is gone is not a model that is running.

`runtime.active_managed_instances()` reports a record as "stale" when the owner host that wrote it is gone (its lease is
not held). `updater.serving_now()` used to return every non-stopped record, stale ones included, so `strata update`
refused with "a model is running, so nothing was changed" although nothing was loaded - and the stale record could stay
for ever (the app asked the person to unload a model they never loaded).

    python -m pytest tests/terminal/test_serving_stale.py -q

The record here is a real one on disk (a private state file in a temporary runtime folder), not a mock: the recovery the
fix does is the same `strata runtime stop` path a person used by hand to clear it (#3977's workaround).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import profiles, runtime, updater

WINDOWS = os.name == "nt"


class StaleRuntimeRecord(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-serving-stale-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(self.root / "runtime"),
                                           "STRATA_PROFILE_DIR": str(self.root / "profiles")})
        env.start()
        self.addCleanup(env.stop)

    def stale_record(self, name="app-1", child_pid=999999):
        """A record shaped like one an owner host leaves when it is killed: a private state file, its child marker,
        and no live host holding either lease."""
        runtime.runtime_dir()
        state_path, socket_path, _ = runtime._paths(name)
        token = "a" * 64
        state = {"schema_version": 1, "profile": name,
                 "socket": "tcp:127.0.0.1:50000" if WINDOWS else str(socket_path),
                 "token": token, "host_pid": 999999}
        descriptor = runtime._write_new(state_path, (json.dumps(state) + "\n").encode())
        os.close(descriptor)                        # the host that would hold this lease is gone
        marker = runtime._child_marker(name, token)
        marker.write_text(json.dumps({"host_token": token, "phase": "owned", "pid": child_pid,
                                      "ready": str(state_path.parent / f"{name}-fake.ready.json")}),
                          encoding="utf-8")
        marker.chmod(0o600)
        return name

    def test_a_lost_owner_host_is_recovered_instead_of_reported_as_a_serving_model(self):
        name = self.stale_record()
        self.assertEqual((runtime.status(name)["state"], runtime.status(name)["error"]), ("stale", "owner_host_lost"))
        self.assertEqual(updater.serving_now(), [])
        self.assertEqual(runtime.status(name)["state"], "stopped")   # the leftover record was cleared, not ignored

    def test_a_run_that_is_starting_or_ready_is_still_a_serving_model(self):
        for state in ("starting", "ready"):
            with mock.patch.object(runtime, "active_managed_instances",
                                   return_value=[{"profile": "app-1", "state": state}]):
                self.assertEqual(updater.serving_now(), [{"profile": "app-1", "state": state}])

    def test_a_stale_record_whose_child_is_still_there_is_still_reported(self):
        # A lost owner can leave a live server behind. Recovery must not clear that record; the engine may be serving.
        with mock.patch.object(runtime, "active_managed_instances",
                               return_value=[{"profile": "app-1", "state": "stale"}]), \
                mock.patch.object(runtime, "stop", side_effect=profiles.ProfileError(
                    "lost host child is still present; retry after it exits", 6)):
            self.assertEqual(updater.serving_now(), [{"profile": "app-1", "state": "stale"}])


if __name__ == "__main__":
    unittest.main()