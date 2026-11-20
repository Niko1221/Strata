"""Conversation isolation and fail-closed native capability acknowledgement."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

from serve import private_cache, server


class PrivateCacheTests(unittest.TestCase):
    def test_harness_affinity_is_bounded_and_explicit_chat_wins(self):
        session = private_cache.request_session({}, 'session:chat-a')
        self.assertEqual(session, private_cache.request_session({}, 'session:chat-a'))
        self.assertNotEqual(session, private_cache.request_session({}, 'session:chat-b'))
        self.assertRegex(session, r'^harness_[a-f0-9]{64}$')
        self.assertEqual(private_cache.request_session({'strata_cache_session': 'conversation_00001'}, 'other'), 'conversation_00001')
        self.assertIsNone(private_cache.request_session({'strata_cache_session': None}, 'other'))
        for invalid in ('', 'x' * 129, 'chat\nSTOP', 'a b', None, 1):
            self.assertIsNone(private_cache.request_session({}, invalid))

    def test_scope_binds_both_key_and_chat_without_exposing_them(self):
        chat = 'conversation_00000001'
        a = private_cache.conversation_scope('first-user-key', chat)
        self.assertEqual(a, private_cache.conversation_scope('first-user-key', chat))
        self.assertRegex(a, r'^[a-f0-9]{64}$')
        self.assertNotEqual(a, private_cache.conversation_scope('other-user-key', chat))
        self.assertNotEqual(a, private_cache.conversation_scope('first-user-key', chat + '2'))
        self.assertIsNone(private_cache.conversation_scope('', chat))
        self.assertIsNone(private_cache.conversation_scope('key', None))
        for bad in ('', 'short', 'x' * 129, 'valid_session_000\nSTOP', {}, 7):
            with self.assertRaises(ValueError):
                private_cache.conversation_scope('key', bad)

    def test_config_cannot_remove_required_acknowledgement(self):
        cfg = {'arch': 'qwen35moe', 'args': ['--prompt-cache', '0', '--conversation-cache-mib', '0',
                '--private-conversation-cache'], 'env': {'STRATA_REQUIRE_Q36_PRIVATE_CACHE': '0'}}
        self.assertEqual(server.child_env(cfg)['STRATA_REQUIRE_Q36_PRIVATE_CACHE'], '2')
        self.assertFalse(private_cache.capability_confirmed({}, 2))
        self.assertFalse(private_cache.capability_confirmed({'q36_private_cache': 1}, 2))
        self.assertTrue(private_cache.capability_confirmed({'q36_private_cache': 2}, 2))

    def test_old_engine_ready_is_rejected_and_scoped_engine_receives_only_digest(self):
        with tempfile.TemporaryDirectory(prefix='strata-private-cache-') as tmp:
            script = Path(tmp) / 'engine.py'
            events = Path(tmp) / 'events.txt'
            env = dict(os.environ, STRATA_REQUIRE_Q36_PRIVATE_CACHE='2')
            real_popen = server.subprocess.Popen
            def start():
                with mock.patch.object(server.subprocess, 'Popen', lambda cmd, **kw:
                        real_popen([sys.executable, '-u', str(script), *cmd[1:]], **kw)):
                    return server.StrataEngine('strata', [], env=env)
            script.write_text("print('READY 4096 stop',flush=True)\ninput()\n", encoding='utf-8')
            with self.assertRaisesRegex(server.EngineDied, 'private conversation cache'):
                start()
            script.write_text("import sys\nfrom pathlib import Path\n"
                "print('INFO q36_private_cache=2',flush=True)\nprint('READY 4096 stop',flush=True)\n"
                "for line in sys.stdin:\n"
                f" Path({str(events)!r}).write_text(line)\n"
                " print('T 42',flush=True)\n"
                " print('DONE 1 2 1.0 1.0 length 0 0 0 0 0 0 0 0.0 2 0',flush=True)\n", encoding='utf-8')
            # An actual pipe verifies startup acknowledgement. Generation's command is
            # captured without waiting for a fake inference loop to finish.
            engine = start()
            try:
                self.assertEqual(engine.info['q36_private_cache'], 2)
                cancel = threading.Event()
                cancel._strata_cache_scope = private_cache.conversation_scope('secret-key', 'conversation_00000001')
                iterator = engine.generate([1, 2], 1, {}, cancel)
                self.assertEqual(next(iterator), 42)
                sent = events.read_text()
                self.assertIn('cache_scope=' + cancel._strata_cache_scope, sent)
                self.assertNotIn('secret-key', sent)
                self.assertNotIn('conversation_00000001', sent)
                list(iterator)
            finally:
                engine.close()


if __name__ == '__main__':
    unittest.main()
