"""The engine's own "DFlash 2 not loaded: it would move part of the model off the card (...)" line reaches the person, and `--draft on` forces the drafter through.

The Inkling engine (HIP now, CUDA through the residency-first drafter rule) leaves DFlash 2 out when loading it would move a model that fits whole into the slower hybrid
mode, prints one line to its log (stderr) and says `--draft on` loads it anyway.  Control must show that line (the Terminal start and the app's start log read the server's
console lines; the Monitor reads the drafting note) and must pass the force through (`strata launch --draft on`, `--set draft=on`, the Advanced screen).  A stub engine prints the
line, as the real one does, unless it is started with `--draft on`.

    python -m pytest serve/test_inkling_draft_veto_line.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import inkling_kind, server  # noqa: E402
from strata_cli.launch import settings as S  # noqa: E402

LINE = ("[ink-hip] DFlash 2 not loaded: it would move part of the model off the card (the experts need 18.20 GiB, 16.90 GiB fit with the drafter's 536 MiB, "
        "17.45 GiB without it): decode and prompt reading get slower than the drafter's gain. `--draft on` loads it anyway.")
SENTENCE = LINE[len("[ink-hip] "):]

STUB = '''import sys
argv = sys.argv[1:]
forced = any(argv[i] == "--draft" and argv[i + 1] == "on" for i in range(len(argv) - 1))
print("INFO engine=0.0.0 drafting=" + ("dspark" if forced else "off"), flush=True)
if not forced:
    sys.stderr.write(%r + "\\n")
    sys.stderr.flush()
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
''' % LINE


class TheLine(unittest.TestCase):
    def test_it_is_found_without_its_bracketed_prefix_and_the_last_one_wins(self):
        self.assertEqual(inkling_kind.veto_line("a\n" + LINE + "\nb\n"), SENTENCE)
        self.assertEqual(inkling_kind.veto_line("[ink] DFlash 2 not loaded: first\n" + LINE), SENTENCE)
        self.assertEqual(inkling_kind.veto_line("DFlash 2 not loaded: bare\n"), "DFlash 2 not loaded: bare")
        self.assertEqual(inkling_kind.veto_line(""), "")
        self.assertEqual(inkling_kind.veto_line(None), "")
        self.assertEqual(inkling_kind.veto_line("[ink] DFlash 2 loaded: 536 MiB\n"), "")

    def test_the_drafting_note_is_the_engines_sentence_when_it_left_the_drafter_out(self):
        for mode in ("dflash2", "on", "dspark"):
            got = inkling_kind.drafting_info(["--draft", mode], {"drafting": "off"}, LINE)
            self.assertEqual(got, {"drafting": "off", "drafting_note": SENTENCE}, mode)
        # an engine that says nothing keeps the old sentences
        old = inkling_kind.drafting_info(["--draft", "dflash2"], {"drafting": "off"}, "")
        self.assertEqual(old["drafting_note"], inkling_kind.DRAFTING_LEFT_OFF_NOTE)
        # `--draft on` is a drafter that was asked for, and one that runs says so
        self.assertEqual(inkling_kind.drafting_info(["--draft", "on"], {"drafting": "dspark"}, "")["drafting"], "dflash2")
        # --draft off is the person's own choice: no engine line is repeated at them
        self.assertEqual(inkling_kind.drafting_info(["--draft", "off"], {"drafting": "off"}, LINE), {"drafting": "off", "drafting_note": ""})


class StartOutput(unittest.TestCase):
    def start(self, draft: list[str]):
        with tempfile.TemporaryDirectory() as d:
            script, log = Path(d) / "stub.py", Path(d) / "engine.log"
            script.write_text(STUB, encoding="utf-8")
            log.write_text("", encoding="utf-8")
            real = server.subprocess.Popen
            out = io.StringIO()
            with mock.patch.object(server.subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)), \
                    contextlib.redirect_stdout(out):
                eng = server.StrataEngine("strata-inkling", ["--native", "m.strata", *draft], log=str(log))
                try:
                    info = dict(eng.info)
                finally:
                    eng.close()
            return out.getvalue(), info

    def test_the_start_prints_the_engines_line_where_the_person_reads_the_start(self):
        said, info = self.start(["--draft", "dflash2"])
        self.assertIn("[strata] " + SENTENCE, said.splitlines())
        self.assertEqual(info.get("drafting"), "off")
        self.assertEqual(info.get("drafting_note"), SENTENCE)

    def test_draft_on_forces_it_and_the_start_says_nothing_about_leaving_it_out(self):
        said, info = self.start(["--draft", "on"])
        self.assertNotIn("DFlash 2 not loaded", said)
        self.assertEqual(info.get("drafting"), "dflash2")
        self.assertEqual(info.get("drafting_note"), "")


class ForceThroughControl(unittest.TestCase):
    def test_the_word_on_reaches_the_engine_as_typed(self):
        # `strata launch --set draft=on` and the Advanced screen: the word used to read as "present" and the engine got a bare --draft
        args, env = S.as_argv_changes({"--draft": "on"}, frozenset(), S.WORD_VALUE_FLAGS)
        self.assertEqual(args, {"--draft": "on"})
        self.assertEqual(env, {})
        from strata_control import tune_knobs
        self.assertEqual(tune_knobs.apply_to_argv(["--native", "m.strata", "--draft", "dflash2"], args, frozenset()),
                         ["--native", "m.strata", "--draft", "on"])
        self.assertEqual(S.as_argv_changes({"--draft": "off"}, frozenset(), S.WORD_VALUE_FLAGS)[0], {"--draft": "off"})
        self.assertEqual(S.as_argv_changes({"--draft": ""}, frozenset(), S.WORD_VALUE_FLAGS)[0], {})

    def test_the_servers_inkling_arguments_keep_the_forced_word(self):
        cfg = {"arch": "inkling2_loop", "exe": "component:strata-inkling", "args": ["--native", "m.strata", "--draft", "on"], "drafting": "auto"}
        with contextlib.redirect_stdout(io.StringIO()):
            got = inkling_kind.engine_args(cfg, None)
        at = got.index("--draft")
        self.assertEqual(got[at + 1], "on")
        self.assertEqual(got.count("--draft"), 1)

    def test_the_launch_command_takes_draft_on(self):
        import argparse
        from strata_cli.launch import cmd
        top = argparse.ArgumentParser()
        cmd.add_parsers(top.add_subparsers(dest="command"))
        ns = top.parse_args(["launch", "--draft", "on", "--model", "iq3_s"])
        self.assertEqual(ns.draft, "on")
        self.assertEqual(top.parse_args(["launch", "--drafting", "dflash2"]).drafting, "dflash2")     # the longer options still parse
        self.assertEqual(top.parse_args(["launch", "--draft-vocab", "en"]).draft_vocab, "en")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            top.parse_args(["launch", "--draft", "maybe"])


if __name__ == "__main__":
    unittest.main()
