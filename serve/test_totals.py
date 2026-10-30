"""The token counts the Monitor tab prints, kept between starts, and the prices that turn them into the dollar figure
printed beside them: the plain input and output prices, and the two cache rates that price a prompt's parts the way
an API does (serve/server.py's totals_* helpers, and the price keys of serve/runconfig.py).
No GPU and no model pack: these only read and write one small JSON file.

    python -m unittest serve.test_totals -v
"""
import re
import tempfile
import unittest
from pathlib import Path

from serve import runconfig
from serve.server import (TOTALS_KEYS, totals_add_saved, totals_cost, totals_load_file, totals_path_of,
                          totals_reset_file, totals_with_cost, totals_write_file)


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


class Cost(unittest.TestCase):
    """The dollar figure the Monitor prints: the counted tokens at two prices."""

    def test_no_price_means_no_cost(self):
        """Nothing configured is no cost at all, so the page prints no figure rather than a $0.00 that would
        read as a model that works for free."""
        self.assertEqual(totals_cost(fresh(), 0, 0), 0.0)
        t = totals_with_cost(fresh(), None, None)
        self.assertEqual((t["cost_input_per_million"], t["cost_output_per_million"], t["cost_usd"]), (0, 0, 0.0))

    def test_a_million_tokens_costs_its_price(self):
        """A price is what a million tokens costs, which is how the APIs price a model: 2M read and 100,000
        written at $5 and $25 per million is 2 x $5 + 0.1 x $25."""
        t = fresh()
        t.update({"prompt_tokens": 2_000_000, "output_tokens": 100_000})
        self.assertEqual(totals_cost(t, 5, 25), 12.5)

    def test_a_read_the_cache_held_is_paid_for_as_input(self):
        """Without a cache price the tokens the cache held are priced as plain input, which is what an API without
        prompt caching charges - and exactly what this figure did before the cache rates existed."""
        self.assertEqual(totals_cost({"prompt_tokens": 1_000_000, "reused": 999_000, "output_tokens": 0}, 5, 25), 5.0)

    def test_a_re_read_is_paid_at_the_cache_price(self):
        """An API prices a token the cache already held far below a new one: of this 1,000,000-token prompt
        990,000 were re-read at $0.50 and 10,000 were new at $6.25, so the prompt costs $0.5575 - where pricing
        every prompt token as plain input would have said $5."""
        self.assertEqual(totals_cost({"prompt_tokens": 1_000_000, "reused": 990_000, "output_tokens": 0},
                                     5, 25, 6.25, 0.5), 0.5575)

    def test_an_added_token_is_paid_at_the_write_price(self):
        """With nothing cached every prompt token is one an API would have had to write to its cache, so
        1,000,000 of them cost 1,000,000 x $6.25 rather than the $5 an uncached prompt costs."""
        self.assertEqual(totals_cost({"prompt_tokens": 1_000_000, "reused": 0, "output_tokens": 0},
                                     5, 25, 6.25, 0.5), 6.25)

    def test_the_prices_of_a_real_price_list_on_a_day_of_agent_turns(self):
        """$5, $25, $6.25 written and $0.50 read over 2,000,000 prompt tokens of which the cache held 1,900,000,
        plus 100,000 written: 0.1 x $6.25 + 1.9 x $0.50 + 0.1 x $25 = $4.075, where the two plain prices alone
        say $12.50 because they price a re-read as if it were a new token."""
        self.assertEqual(totals_cost({"prompt_tokens": 2_000_000, "reused": 1_900_000, "output_tokens": 100_000},
                                     5, 25, 6.25, 0.5), 4.075)

    def test_a_free_cache_read_is_a_price_and_not_a_missing_one(self):
        """An explicit 0 on the cache read says those tokens cost nothing, while leaving the rate out falls back to
        the input price - the two must not mean the same thing."""
        held = {"prompt_tokens": 1_000_000, "reused": 900_000, "output_tokens": 0}
        self.assertEqual(totals_cost(held, 5, 25, 6.25, 0), 0.625)      # 100,000 new at $6.25, the rest for nothing
        self.assertEqual(totals_cost(held, 5, 25, 6.25, None), 5.125)   # the 900,000 fall back to the $5 input price

    def test_the_response_carries_the_prices_and_the_cost(self):
        """GET /metrics' totals: the cost without the reader having to multiply, and the prices beside it, so a
        page can show which prices a figure came from. The counts stay where they were."""
        t = fresh()
        t.update({"prompt_tokens": 1_500_000, "reused": 1_400_000, "output_tokens": 250_000})
        out = totals_with_cost(t, 5, 25)
        self.assertEqual(out["cost_usd"], 13.75)            # 1.5 million at $5 = $7.50, 250,000 at $25 = $6.25
        self.assertEqual((out["cost_input_per_million"], out["cost_output_per_million"]), (5, 25))
        self.assertEqual(out["prompt_tokens"], 1_500_000)

    def test_the_page_is_told_the_rate_each_kind_of_token_is_paid_at(self):
        """A page should not have to know the fallback rule: what comes back is the rate the tokens are actually
        paid at, so a cache price that was never set reads back as the plain input price it fell back to."""
        t = totals_with_cost(fresh(), 5, 25, 6.25, None)
        self.assertEqual((t["cost_cache_write_per_million"], t["cost_cache_hit_per_million"]), (6.25, 5))

    def test_the_cost_covers_every_start_the_file_remembers(self):
        """The cost outlives a restart with the counts, because it is worked out of them and never kept beside
        them: the totals file stays counts only."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.totals.json"
            first = fresh(1000.0)
            first.update({"requests": 2, "prompt_tokens": 1_000_000, "output_tokens": 100_000})
            totals_write_file(p, first)
            self.assertNotIn("cost_usd", totals_load_file(p))
            second = totals_add_saved(fresh(2000.0), totals_load_file(p))
            second.update({"prompt_tokens": 1_500_000, "output_tokens": 250_000})
            self.assertEqual(totals_cost(second, 5, 25), 13.75)    # 1.5 million read + 250,000 written, both starts


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


class PriceSettings(unittest.TestCase):
    """The prices are run config keys, so the page's own Settings view reads and changes them (#564)."""

    def test_both_prices_are_editable_from_the_page(self):
        listed = [k["key"] for k in runconfig.view({}, "strata-m.json")["keys"]]
        self.assertIn("cost_input_per_million", listed)
        self.assertIn("cost_output_per_million", listed)

    def test_a_price_set_from_the_page_is_written_and_read_back(self):
        cfg, changed = runconfig.apply({}, {"cost_input_per_million": 5, "cost_output_per_million": 25})
        self.assertEqual(sorted(changed), ["cost_input_per_million", "cost_output_per_million"])
        self.assertEqual((cfg["cost_input_per_million"], cfg["cost_output_per_million"]), (5, 25))

    def test_only_a_price_of_at_least_nothing_is_a_price(self):
        """A price is a number of dollars, not a text, not a switch, and not what a purchase order says."""
        for bad in (-1, "five", True, [], 0):
            if bad == 0:                                   # 0 is the one price that means "no cost": allowed
                self.assertEqual(runconfig.apply({}, {"cost_input_per_million": 0})[1], ["cost_input_per_million"])
                continue
            with self.assertRaises(ValueError, msg=repr(bad)):
                runconfig.apply({}, {"cost_input_per_million": bad})
        cfg, changed = runconfig.apply({"cost_input_per_million": 5}, {"cost_input_per_million": None})
        self.assertEqual((cfg, changed), ({}, ["cost_input_per_million"]))    # null clears it, as every key does

    def test_the_two_prices_travel_with_their_own_name_only(self):
        """A change to a price touches that key and nothing else in the file - the engine's args, the network
        keys, and the other price all stay as they were."""
        was = {"model": "m.gguf", "args": ["--n-gpu-layers", "99"], "api_key": "k", "cost_output_per_million": 25}
        cfg, changed = runconfig.apply(was, {"cost_input_per_million": 5})
        self.assertEqual(changed, ["cost_input_per_million"])
        self.assertEqual({k: v for k, v in cfg.items() if k != "cost_input_per_million"},
                         {k: v for k, v in was.items() if k != "cost_input_per_million"})

    def test_the_cache_prices_are_editable_from_the_page_too(self):
        """Four prices, four fields: the Settings view is built from the same list, so the two cache rates are in
        it beside the plain ones rather than only in the file."""
        listed = [k["key"] for k in runconfig.view({}, "strata-m.json")["keys"]]
        self.assertIn("cost_cache_write_per_million", listed)
        self.assertIn("cost_cache_hit_per_million", listed)

    def test_a_cache_price_is_dollars_and_cents(self):
        """A real price list reads $6.25 and $0.50, so a price is a number of dollars and cents, not whole
        dollars rounded."""
        cfg, changed = runconfig.apply({}, {"cost_cache_write_per_million": 6.25,
                                            "cost_cache_hit_per_million": 0.5})
        self.assertEqual((cfg["cost_cache_write_per_million"], cfg["cost_cache_hit_per_million"]), (6.25, 0.5))
        self.assertEqual(sorted(changed), ["cost_cache_hit_per_million", "cost_cache_write_per_million"])

    def test_a_cache_price_of_nothing_is_a_price_of_nothing(self):
        """0 is a price here as it is on the other two, where it means those tokens cost nothing rather than
        "no price was set"."""
        cfg, changed = runconfig.apply({}, {"cost_cache_hit_per_million": 0})
        self.assertEqual((cfg["cost_cache_hit_per_million"], changed), (0, ["cost_cache_hit_per_million"]))


class WhatTheSettingsCardLooksLike(unittest.TestCase):
    """The page's Model settings card is a grid of a help line and a field (#564).  The help is the server's own
    words and may be long, so the card must give way in its label and never in its field: sizing the label column to
    max-content once let one long help line squeeze every field of the card, the four prices among them, down to the
    width of its own spinner arrows.  Neither the style nor the words may bring that back.
    """

    LONGEST = 140      # the longest help line the card was laid out beside before it broke (mtp: 137)

    @classmethod
    def setUpClass(cls):
        cls.css = (Path(__file__).resolve().parent / "web" / "app.css").read_text(encoding="utf-8")

    def test_the_field_column_keeps_a_floor_of_its_own(self):
        rule = re.search(r"\n\.cfg-form \{[^}]*grid-template-columns:([^;]+);", self.css)
        self.assertIsNotNone(rule, "the card's grid rule is gone")
        cols = rule.group(1)
        self.assertNotIn("max-content", cols,               # the label wraps rather than pushing the field out
                         f"the label column is sized to its content again: {cols}")
        self.assertRegex(cols, r"1fr\)\s*minmax\(\s*[\d.]+",  # and the field keeps a width whatever the label says
                         f"the field column has no floor: {cols}")

    def test_no_help_line_is_a_wall_of_text(self):
        """A help line is both the field's label and its tooltip, so it stays short enough to read in one row."""
        for key, _, help_text in runconfig.EDITABLE:
            self.assertLessEqual(len(help_text), self.LONGEST,
                                 f"{key}'s help is {len(help_text)} characters, over the {self.LONGEST} the card fits")


if __name__ == "__main__":
    unittest.main()
