"""Windows runtime state cleanup with a transient file reader."""

import errno
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from strata_control import runtime


@unittest.skipUnless(os.name == "nt", "Windows readers can temporarily block deletion")
class RuntimeStateCleanupTests(unittest.TestCase):
    def test_host_shutdown_removes_state_after_a_reader_releases_it(self):
        with tempfile.TemporaryDirectory(prefix="strata-runtime-share-") as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.dict(os.environ, {
                "STRATA_RUNTIME_DIR": str(root / "runtime"),
                "STRATA_STATE_DIR": str(root / "state"),
            }))
            host = runtime.Host("app-1", shutdown_timeout=0)
            listener = mock.Mock()
            listener.bind.return_value = "tcp:127.0.0.1:9111"
            reader = None
            sharing_failures = []

            def stop_with_reader_open():
                nonlocal reader
                reader = stack.enter_context(host.state_path.open("rb"))
                host.running = False
                return None

            real_unlink = Path.unlink

            def release_reader_after_failed_unlink(path, *args, **kwargs):
                try:
                    return real_unlink(path, *args, **kwargs)
                except PermissionError as error:
                    if path == host.state_path and error.winerror == 32:
                        # Release on the actual OS refusal so the first attempt
                        # always sees contention without relying on thread timing.
                        sharing_failures.append(error.winerror)
                        reader.close()
                    raise

            listener.accept.side_effect = stop_with_reader_open
            activity_lease = runtime.acquire_activity_lease(shared=True)
            try:
                stack.enter_context(mock.patch.object(runtime, "acquire_activity_lease", return_value=activity_lease))
                stack.enter_context(mock.patch.object(runtime.owner_transport, "Listener", return_value=listener))
                stack.enter_context(mock.patch.object(runtime.signal, "signal"))
                stack.enter_context(mock.patch.object(host, "_startup", return_value=None))
                stack.enter_context(mock.patch.object(Path, "unlink", release_reader_after_failed_unlink))

                self.assertEqual(host.serve(), 0)
                self.assertEqual(sharing_failures, [32])
                self.assertFalse(host.state_path.exists())
            finally:
                # The failing baseline leaves the activity descriptor open.
                try:
                    os.close(activity_lease)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
