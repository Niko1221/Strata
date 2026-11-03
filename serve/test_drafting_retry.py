"""GUI-FB3 item 2(b): an Auto start that picked DFlash2 and ended before the engine was ready is started once more with the MTP, in one quiet line. A start the person
named DFlash2 for fails with the engine's own reason. A real process and a real pipe (a fake engine program), like serve/test_fatal_recovery.py."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.server import EngineDied, StrataEngine
from strata_control import drafting

FAKE = r'''
import sys
from pathlib import Path
args = sys.argv[1:]
Path(args[args.index('--starts') + 1]).open('a').write(' '.join(args) + '\n')
if '--draft' in args and '--die-on-draft' in args:
    sys.stderr.write('strata generate: --draft dflash2 drafts without the MTP layer and needs --draft-model PACK, --spec 2 or more and --mtp DIR (for DIR/draft_vocab.bin)\n')
    sys.stderr.flush()
    raise SystemExit(2)
if '--die-always' in args:
    sys.stderr.write('strata generate: something else is wrong\n')
    sys.stderr.flush()
    raise SystemExit(2)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
'''


class AutoRetriesWithTheMtp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        py = self.root / "native.py"
        if os.name == "nt":
            py.write_text(FAKE)
            self.script = self.root / "native.cmd"
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script = py
            py.write_text("#!" + sys.executable + "\n" + FAKE)
            py.chmod(0o700)
        self.starts = self.root / "starts.txt"
        self.log = str(self.root / "engine.log")
        self.first = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-on-draft", "--draft", "dflash2", "--draft-model", "/p", "--draft-fallback", "none",
                      "--mtp-draft-vocab", "/d/mtp/rt/draft_vocab.bin"]
        self.mtp = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-on-draft", "--mtp", "/d/mtp/rt", "--spec", "4"]
        drafting.take_note()
        self.addCleanup(drafting.take_note)

    def started(self):
        return self.starts.read_text().splitlines() if self.starts.exists() else []

    def test_auto_that_picked_dflash2_starts_again_with_the_mtp_and_says_so_quietly(self):
        engine = StrataEngine(str(self.script), self.first, log=self.log, retry_args=self.mtp)
        self.addCleanup(engine.close)
        self.assertTrue(engine.alive())
        starts = self.started()
        self.assertEqual(len(starts), 2, "one start with DFlash2, one more with the MTP, never a third")
        self.assertIn("--draft dflash2", starts[0])
        self.assertNotIn("--draft", starts[1])
        self.assertEqual(engine.spawn[1], self.mtp, "a later restart of this engine uses the arguments that worked")
        self.assertEqual(engine.info["drafting"], "mtp")
        self.assertEqual(engine.info["drafting_note"], "Drafting: MTP (DFlash2 didn't start)", "the one quiet line the Models screen shows")
        text = Path(self.log).read_text(encoding="utf-8")
        self.assertIn("starting it again with the MTP", text)
        self.assertIn("needs --draft-model PACK", text, "the engine's own reason stays in the log")

    def test_a_start_with_no_retry_arguments_fails_with_the_engines_reason(self):
        """Named by the person (or nothing to retry with): a real error that says why."""
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), self.first, log=self.log)
        self.assertIn("exited before it was ready", str(caught.exception))
        self.assertIn("needs --draft-model PACK", str(caught.exception))
        self.assertEqual(len(self.started()), 1)

    def test_the_retry_is_one_start_only(self):
        both = [*self.first, "--die-always"]
        retry = [*self.mtp, "--die-always"]
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), both, log=self.log, retry_args=retry)
        self.assertIn("something else is wrong", str(caught.exception), "the second start's reason, not the first one's")
        self.assertEqual(len(self.started()), 2)

    def test_a_start_that_works_is_not_started_twice(self):
        ok = ["--pack", "/d/packs/g", "--starts", str(self.starts)]
        engine = StrataEngine(str(self.script), ok, log=self.log, retry_args=self.mtp)
        self.addCleanup(engine.close)
        self.assertEqual(len(self.started()), 1)
        self.assertEqual(engine.info.get("drafting_note", ""), "")

    def test_the_same_arguments_are_not_a_retry(self):
        with self.assertRaises(EngineDied):
            StrataEngine(str(self.script), self.first, log=self.log, retry_args=list(self.first))
        self.assertEqual(len(self.started()), 1)


if __name__ == "__main__":
    unittest.main()
