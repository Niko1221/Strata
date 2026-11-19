import unittest

from serve import tool_name_constraint as tnc


class _Tok:
    """A Qwen-shaped stand-in: one id per character, single ids for the structural markers.

    The real tokenizer emits one id for each of <think>, </think>, <tool_call> and
    </tool_call> and merges regular text greedily; here a plain character is one id, which is
    the simplest case the wire format must express (trigger + one tail per declared name)."""
    SPECIALS = {tnc._THINK_OPEN: 200, tnc._THINK_CLOSE: 201,
                tnc._CALL_OPEN: 202, tnc._CALL_CLOSE: 203}

    def encode(self, text, parse_special=False):
        out, i = [], 0
        while i < len(text):
            for s, sid in sorted(self.SPECIALS.items(), key=lambda kv: -len(kv[0])):
                if parse_special and text.startswith(s, i):
                    out.append(sid)
                    i += len(s)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out


class _NoMarkers(_Tok):
    """A model whose vocabulary has no <tool_call> markers: the feature must stay off."""

    def encode(self, text, parse_special=False):
        if parse_special and text in self.SPECIALS:
            return []
        return super().encode(text, parse_special)


class _MergedEq(_Tok):
    """A BPE-like merge: `=` and the name's first letter become one id (1000 + letter), as `="` + letter does for invoke."""

    def encode(self, text, parse_special=False):
        ids = super().encode(text, parse_special)
        out, i = [], 0
        while i < len(ids):
            if ids[i] == ord("=") and i + 1 < len(ids) and chr(ids[i + 1]).isalpha():
                out.append(1000 + ids[i + 1])
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out


def _chars(text):
    return [ord(c) for c in text]


def _parse(record):
    """-> version, initial, think_open, think_close, call_open, call_close, [(trigger, [tails])]"""
    parts = record.split("|")
    rows = lambda p: [int(x) for x in p.split(",")] if p else []   # noqa: E731
    forms = []
    for p in parts[6:]:
        if p.startswith("#"):
            forms.append((rows(p[1:]), []))
        else:
            forms[-1][1].append(rows(p))
    return parts[0], parts[1], rows(parts[2]), rows(parts[3]), rows(parts[4]), rows(parts[5]), forms


class Build(unittest.TestCase):
    def test_both_forms_share_the_markers(self):
        record = tnc.build(_Tok(), ["write", "read"], thinking=True)
        version, initial, think_open, think_close, call_open, call_close, forms = _parse(record)
        self.assertEqual((version, initial), ("v2", "1"))
        self.assertEqual((think_open, think_close, call_open, call_close), ([200], [201], [202], [203]))
        (fn_trigger, fn_tails), (inv_trigger, inv_tails) = forms
        self.assertEqual(fn_trigger, _chars("<function="))
        self.assertEqual(fn_tails, [_chars("write>"), _chars("read>"), _chars("function_calls>")])
        self.assertEqual(inv_trigger, _chars('<invoke name="'))
        self.assertEqual(inv_tails, [_chars('write">'), _chars('read">')])

    def test_shared_name_prefix_extends_the_trigger_only_to_the_opener(self):
        # 'get_' is common to both names, but the bare opener bounds the trigger: the first name token stays masked
        record = tnc.build(_Tok(), ["get_weather", "get_temp"], thinking=True)
        (_, fn_tails), (inv_trigger, inv_tails) = _parse(record)[6]
        self.assertIn(_chars("get_weather>"), fn_tails)
        self.assertEqual(inv_trigger, _chars('<invoke name="'))
        self.assertEqual(inv_tails, [_chars('get_weather">'), _chars('get_temp">')])

    def test_a_single_name_is_still_masked_from_its_first_token(self):
        # with one declared tool the trigger must stop at the opener, or a hallucinated name would pass
        record = tnc.build(_Tok(), ["write"], thinking=True)
        (fn_trigger, fn_tails), (inv_trigger, inv_tails) = _parse(record)[6]
        self.assertEqual(fn_trigger, _chars("<function="))
        self.assertEqual(inv_trigger, _chars('<invoke name="'))
        self.assertEqual(inv_tails, [_chars('write">')])

    def test_a_merged_opener_token_moves_into_the_tails(self):
        record = tnc.build(_MergedEq(), ["bash", "read"], thinking=False)
        (fn_trigger, fn_tails), _ = _parse(record)[6]
        self.assertEqual(fn_trigger, _chars("<function"))   # `=b` / `=r` are one id each: the trigger stops before them
        self.assertEqual(fn_tails[0], [1000 + ord("b")] + _chars("ash>"))

    def test_a_wrapper_tag_the_trigger_fires_on_stays_writable(self):
        # Qwen's BPE: the <function= trigger is `<` `function`, so `<function_calls>` arms it too
        record = tnc.build(_MergedEq(), ["bash"], thinking=False)
        (fn_trigger, fn_tails), _ = _parse(record)[6]
        self.assertEqual(fn_trigger, _chars("<function"))
        self.assertIn(_chars("_calls>"), fn_tails)
        # with the char tokenizer the trigger is `<function=`, which `<function_calls>` never completes
        (fn_trigger, fn_tails), _ = _parse(tnc.build(_Tok(), ["bash"], thinking=False))[6]
        self.assertEqual(fn_trigger, _chars("<function="))
        self.assertNotIn(_chars("_calls>"), fn_tails)

    def test_the_wrapper_name_is_not_doubled(self):
        record = tnc.build(_Tok(), ["function_calls", "bash"], thinking=False)
        (_, fn_tails), _ = _parse(record)[6]
        self.assertEqual(fn_tails.count(_chars("function_calls>")), 1)

    def test_thinking_flag_becomes_the_initial_state(self):
        self.assertEqual(_parse(tnc.build(_Tok(), ["write"], thinking=False))[1], "0")

    def test_no_names_disables(self):
        for names in ([], [""], [None], [1], ["", None]):
            self.assertIsNone(tnc.build(_Tok(), names, True))

    def test_missing_call_markers_disables(self):
        self.assertIsNone(tnc.build(_NoMarkers(), ["write"], True))

    def test_missing_think_markers_just_disables_reasoning_tracking(self):
        class NoThink(_Tok):
            def encode(self, text, parse_special=False):
                if parse_special and text in (tnc._THINK_OPEN, tnc._THINK_CLOSE):
                    return []
                return super().encode(text, parse_special)

        _, _, think_open, think_close, call_open, call_close, forms = _parse(tnc.build(NoThink(), ["write"], True))
        self.assertEqual((think_open, think_close, call_open, call_close), ([], [], [202], [203]))
        self.assertEqual(len(forms), 2)

    def test_a_form_that_is_one_token_is_left_out(self):
        class OneToken(_Tok):
            def encode(self, text, parse_special=False):
                if text.startswith(tnc._TRIGGER):
                    return [999]
                return super().encode(text, parse_special)

        forms = _parse(tnc.build(OneToken(), ["write"], True))[6]
        self.assertEqual([f[0] for f in forms], [_chars('<invoke name="')])   # only the invoke form is sent


class EngineSupports(unittest.TestCase):
    class Eng:
        def __init__(self, info, split=False):
            self.info, self.split_engine = info, split

    def test_only_an_engine_reporting_the_format(self):
        self.assertTrue(tnc.engine_supports(self.Eng({"tool_name_constraint": 2})))
        self.assertFalse(tnc.engine_supports(self.Eng({"tool_name_constraint": 1})))   # an older format
        self.assertFalse(tnc.engine_supports(self.Eng({"tool_name_constraint": 0})))   # --batch: its windows take no mask
        self.assertFalse(tnc.engine_supports(self.Eng({"logit_bias": 1})))             # SYCL / Metal / an old engine
        self.assertFalse(tnc.engine_supports(self.Eng({"tool_name_constraint": 2}, split=True)))
        self.assertFalse(tnc.engine_supports(None))


class Setting(unittest.TestCase):
    def test_off_unless_named(self):
        import os
        from serve.server_parts.http_api import tool_name_constraint_setting
        old = os.environ.pop("STRATA_TOOL_NAME_CONSTRAINT", None)
        try:
            self.assertFalse(tool_name_constraint_setting({}))
            self.assertTrue(tool_name_constraint_setting({"tool_name_constraint": True}))
            os.environ["STRATA_TOOL_NAME_CONSTRAINT"] = "1"
            self.assertTrue(tool_name_constraint_setting({}))
            self.assertFalse(tool_name_constraint_setting({"tool_name_constraint": False}))   # the config wins
            with self.assertRaises(SystemExit):
                tool_name_constraint_setting({"tool_name_constraint": "yes"})
        finally:
            os.environ.pop("STRATA_TOOL_NAME_CONSTRAINT", None)
            if old is not None:
                os.environ["STRATA_TOOL_NAME_CONSTRAINT"] = old


if __name__ == "__main__":
    unittest.main()
