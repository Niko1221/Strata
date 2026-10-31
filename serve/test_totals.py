"""The token counts the Monitor tab prints, kept between starts, and the prices that turn them into the dollar
figure printed beside them (serve/server.py's totals_* helpers, and the price keys of serve/runconfig.py).
No GPU and no model pack: these only read and write one small JSON file.

    python -m unittest serve.test_totals -v
"""
import tempfile
import unittest
from pathlib import Path

from serve.server import (TOTALS_KEYS, totals_add_saved, totals_load_file, totals_path_of,
                          totals_reset_file, totals_write_file)
def fresh(since=1000.0):
    """The totals of a server that has just started, as Service.__init__ makes them."""
    return {"since": since, "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
            "prompt_ms": 0.0, "decode_ms": 0.0, "drafts_offered": 0, "drafts_accepted": 0}


class TotalsPath(unittest.TestCase):
    def test_next_to_the_run_config(self):
        """A run config names the file: a model's counters live beside its own settings, as its shared settings do."""
        self.assertEqual(totals_path_of("", "strata-m.json"), "strata-m.totals.json")

    def test_a_given_path_wins(self):
        self.assertEqual(totals_path_of("/data/t.json", "strata-m.json"), "/data/t.json")

    def test_turned_off(self):
        self.assertIsNone(totals_path_of("none", "strata-m.json"))
        self.assertIsNone(totals_path_of("off", "strata-m.json"))

    def test_no_run_config_counts_one_start(self):
        """No run config and no --totals-file: nowhere to keep them, so the counters stay as they were (#35)."""
        self.assertIsNone(totals_path_of("", None))


class TotalsFile(unittest.TestCase):
    def test_no_file_is_a_first_start(self):
        self.assertEqual(totals_load_file(Path(tempfile.gettempdir()) / "strata-surely-not-here.json"), {})

    def test_a_broken_file_is_a_first_start(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.totals.json"
            p.write_text("{not json", encoding="utf-8")
            self.assertEqual(totals_load_file(p), {})

    def test_saved_counts_are_added_and_the_window_moves_back(self):
        saved = {"since": 500.0, "requests": 3, "prompt_tokens": 300, "reused": 250, "output_tokens": 40,
                 "prompt_ms": 100.5, "decode_ms": 500.25, "drafts_offered": 8, "drafts_accepted": 5}
        t = totals_add_saved(fresh(1000.0), saved)
        self.assertEqual((t["requests"], t["prompt_tokens"], t["reused"], t["output_tokens"]), (3, 300, 250, 40))
        self.assertEqual((t["prompt_ms"], t["decode_ms"]), (100.5, 500.25))
        self.assertEqual((t["drafts_offered"], t["drafts_accepted"]), (8, 5))
        self.assertEqual(t["since"], 500.0)        # the printed line reads the whole lifetime, not one start

    def test_only_numbers_are_added(self):
        """A file another version wrote, or a hand-edit that got something wrong, cannot spoil a start."""
        t = totals_add_saved(fresh(), {"requests": "three", "reused": None, "prompt_ms": 5, "from": "the moon"})
        self.assertEqual(t["requests"], 0)
        self.assertEqual(t["reused"], 0)
        self.assertEqual(t["prompt_ms"], 5)
        self.assertEqual(t["since"], 1000.0)       # a window the file does not give stays as it was

    def test_a_newer_since_does_not_move_the_window_forward(self):
        self.assertEqual(totals_add_saved(fresh(1000.0), {"since": 4000.0})["since"], 1000.0)

    def test_write_then_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.totals.json"
            t = fresh(1000.0)
            t.update({"requests": 4, "prompt_tokens": 4000, "reused": 3900, "output_tokens": 60,
                      "prompt_ms": 1200.5, "decode_ms": 700.25, "drafts_offered": 9, "drafts_accepted": 6})
            totals_write_file(p, t)
            saved = totals_load_file(p)
            self.assertEqual({k: saved[k] for k in TOTALS_KEYS}, {k: t[k] for k in TOTALS_KEYS})
            self.assertEqual(saved["since"], 1000.0)

    def test_two_starts_add_up(self):
        """The point of it all: the second start counts on from the first, and the window is the first's."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.totals.json"
            first = fresh(1000.0)
            first.update({"requests": 2, "prompt_tokens": 200, "reused": 150, "output_tokens": 20})
            totals_write_file(p, first)            # written when the last of those requests ended
            second = totals_add_saved(fresh(2000.0), totals_load_file(p))
            second.update({"requests": 3, "prompt_tokens": 250, "reused": 190, "output_tokens": 25})
            totals_write_file(p, second)
            saved = totals_load_file(p)
            self.assertEqual(saved["requests"], 3)
            self.assertEqual(saved["prompt_tokens"], 250)
            self.assertEqual(saved["output_tokens"], 25)
            self.assertEqual(saved["since"], 1000.0)

class TotalsWindow(unittest.TestCase):
    """`--totals-reset` closes a costing window, so one session's cost can be read on its own."""

    def test_the_window_that_ended_is_moved_aside_not_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.totals.json"
            first = fresh(1000.0)
            first.update({"requests": 4, "prompt_tokens": 400, "output_tokens": 40})
            totals_write_file(p, first)
            was = totals_reset_file(p)
            self.assertEqual(was.name, "m.totals.prev.json")
            self.assertEqual(totals_load_file(p), {})                   # this start begins from zero again
            self.assertEqual(totals_load_file(was)["requests"], 4)      # and the window before it still reads

    def test_no_file_is_nothing_to_reset(self):
        """A first start, or a server counting one start only, has no window to close."""
        self.assertIsNone(totals_reset_file(Path(tempfile.gettempdir()) / "strata-surely-not-here-again.json"))


if __name__ == "__main__":
    unittest.main()
