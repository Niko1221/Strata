"""serve/test_crash_loop_restart.py: #6896 #6830 #6849.  An engine that dies at its start or on its first request was started again by every request
that found it dead, for ever (375 failed starts in five hours, no message). Three deaths within ten minutes stop the restarts, and the 503 says
what the engine said last, where its log is and, for the Q8_0 tensor-core fault, the way out.
CPU only, no engine: Service.ensure_loaded runs on a stand-in whose engine dies each time it is started."""
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402


class DyingEngine:
    """Every start 'works' and the process is dead again at the next request (a first-request fault), or the start itself fails."""

    def __init__(self, last_err, log_path=None, fail_start=False):
        self.last_err, self.log_path, self.fail_start = last_err, log_path, fail_start
        self.proc = types.SimpleNamespace(pid=100)
        self.restarts = 0
        self.unloaded = False

    def exit_code(self):
        return 3

    def restart(self):
        self.restarts += 1
        self.proc = types.SimpleNamespace(pid=100 + self.restarts)      # a new process, dead again when the next request comes
        if self.fail_start:
            raise RuntimeError("the engine did not start")

    def _crash_tail(self):
        return self.last_err or ""


def service(engine):
    svc = types.SimpleNamespace(engine=engine, before_load=None, min_free_vram_mib=0, restore_after_load=None, vision=None, vram_reserve=None,
                                loaded=lambda: False, _vision_down=lambda: False)
    for name in ("FATAL_GPU_MAX", "FATAL_GPU_WINDOW_S", "CRASH_LOOP_MAX", "CRASH_LOOP_WINDOW_S", "LAYER0_WORDS"):
        setattr(svc, name, getattr(S.Service, name))
    for name in ("_note_fatal_gpu_exit", "_note_crash_loop", "_engine_last_words", "_crash_loop_text", "ensure_loaded"):
        setattr(svc, name, getattr(S.Service, name).__get__(svc))
    return svc


class CrashLoop(unittest.TestCase):
    def ask(self, svc, n):
        """n requests, each finding the engine dead: (restarts made, the refusals)."""
        refused = []
        for _ in range(n):
            try:
                svc.ensure_loaded()
            except S.EngineDied as e:
                refused.append(str(e))
            except RuntimeError:
                pass                                   # a start that failed: the request fails, the next one finds it dead
        return refused

    def test_an_engine_that_dies_every_time_is_started_twice_then_refused_with_its_last_words(self):
        with tempfile.TemporaryDirectory() as d:
            log = str(Path(d) / "engine.log")
            eng = DyingEngine("verify: something else broke", log_path=log)
            svc = service(eng)
            refused = self.ask(svc, 50)
            self.assertEqual(eng.restarts, 2, "restarted for the first two deaths, never again")
            self.assertEqual(len(refused), 48)
            text = refused[0]
            self.assertIn("The engine stopped 3 times in 10 minutes", text)
            self.assertIn("verify: something else broke", text)
            self.assertIn(log, text)
            self.assertNotIn("STRATA_HC_Q8_TC", text, "the workaround is for its own fault only")
            self.assertEqual(set(refused), {text}, "the same plain message every time")

    def test_a_start_that_itself_fails_is_limited_the_same_way(self):
        eng = DyingEngine("out of memory", fail_start=True)
        refused = self.ask(service(eng), 50)
        self.assertEqual(eng.restarts, 2)
        self.assertEqual(len(refused), 48)

    def test_the_q8_tensor_core_fault_names_the_way_out(self):
        for words in ("verify: layer 0 never rang (operation not supported on global/shared address space)",
                      "operation not supported on global/shared address space"):
            with self.subTest(words=words):
                refused = self.ask(service(DyingEngine(words)), 5)
                self.assertTrue(refused)
                self.assertIn("STRATA_HC_Q8_TC=0", refused[0])
                self.assertIn("STRATA_HC_Q8_BF16TC=0", refused[0])
                self.assertIn("0.2.21", refused[0])
                self.assertIn(words, refused[0])

    def test_the_fault_line_above_the_last_one_in_the_log_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("strata serve: verify: layer 0 never rang (operation not supported on global/shared address space)\nstrata: exiting\n", encoding="utf-8")
            refused = self.ask(service(DyingEngine("strata: exiting", log_path=str(log))), 5)
            self.assertIn("STRATA_HC_Q8_TC=0", refused[0])

    def test_after_the_window_it_tries_again(self):
        eng = DyingEngine("boom")
        svc = service(eng)
        self.ask(svc, 5)
        self.assertEqual(eng.restarts, 2)
        svc._crash_times[:] = [t - S.Service.CRASH_LOOP_WINDOW_S - 1 for t in svc._crash_times]
        self.ask(svc, 1)
        self.assertEqual(eng.restarts, 3, "the old deaths aged out: this request starts it once more")

    def test_deaths_far_apart_never_stop_a_working_engine(self):
        eng = DyingEngine("boom")
        svc = service(eng)
        for _ in range(10):
            self.ask(svc, 1)
            svc._crash_times[:] = [t - S.Service.CRASH_LOOP_WINDOW_S - 1 for t in svc._crash_times]     # ten minutes pass between deaths
        self.assertEqual(eng.restarts, 10)

    def test_an_engine_program_that_cannot_be_run_keeps_its_own_plain_error(self):
        """No process ever existed (a missing program): every request gets "the engine could not be started: ..." at once, never the loop message
        (serve.test_ready_timeout.SpawnFailure asks four requests in a row)."""
        eng = DyingEngine("boom")
        eng.proc = None

        def restart():
            eng.restarts += 1
            raise OSError(2, "No such file or directory")
        eng.restart = restart
        svc = service(eng)
        for _ in range(10):
            with self.assertRaises(S.EngineDied) as cm:
                svc.ensure_loaded()
            self.assertIn("the engine could not be started", str(cm.exception))
        self.assertEqual(eng.restarts, 10)

    def test_an_unloaded_model_is_not_a_crash(self):
        eng = DyingEngine("boom")
        eng.unloaded = True
        svc = service(eng)
        self.ask(svc, 10)
        self.assertEqual(eng.restarts, 10, "loading it again after an unload is not counted")


if __name__ == "__main__":
    unittest.main()
