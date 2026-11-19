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


def _chars(text):
    return [ord(c) for c in text]


def _parse(record):
    parts = record.split("|")
    assert parts[0] in ("0", "1"), parts[0]
    rows = [[int(x) for x in p.split(",")] if p else [] for p in parts[1:]]
    return parts[0], rows[0], rows[1], rows[2], rows[3], rows[4], rows[5:]


class Build(unittest.TestCase):
    def test_two_names_share_the_trigger(self):
        record = tnc.build(_Tok(), ["write", "read"], thinking=True)
        initial, trigger, think_open, think_close, call_open, call_close, tails = _parse(record)
        self.assertEqual(initial, "1")
        self.assertEqual(trigger, _chars("<function="))
        self.assertEqual(tails, [_chars("write>"), _chars("read>")])
        self.assertEqual(think_open, [200])
        self.assertEqual(think_close, [201])
        self.assertEqual(call_open, [202])
        self.assertEqual(call_close, [203])

    def test_shared_name_prefix_extends_the_trigger(self):
        # 'get_' is common to both names, so the trigger is "<function=get_" and the tails are
        # only the divergent rest: the engine can't accept a token from the other branch.
        record = tnc.build(_Tok(), ["get_weather", "get_temp"], thinking=True)
        _, trigger, _, _, _, _, tails = _parse(record)
        self.assertEqual(trigger, _chars("<function=get_"))
        self.assertEqual(tails, [_chars("weather>"), _chars("temp>")])

    def test_a_single_name_ends_at_its_last_token(self):
        record = tnc.build(_Tok(), ["write"], thinking=True)
        _, trigger, _, _, _, _, tails = _parse(record)
        self.assertEqual(trigger, _chars("<function=write"))
        self.assertEqual(tails, [[ord(">")]])

    def test_thinking_flag_becomes_the_initial_state(self):
        record = tnc.build(_Tok(), ["write"], thinking=False)
        self.assertEqual(_parse(record)[0], "0")

    def test_no_names_disables(self):
        for names in ([], [""], [None], [1], [""], ["", None]):
            self.assertIsNone(tnc.build(_Tok(), names, True))

    def test_missing_call_markers_disables(self):
        self.assertIsNone(tnc.build(_NoMarkers(), ["write"], True))

    def test_missing_think_markers_just_disables_reasoning_tracking(self):
        # A model that can call tools but has no <think> block still constrains; the reasoning
        # markers are empty so the engine tracks only the <tool_call> envelope.
        class NoThink(_Tok):
            def encode(self, text, parse_special=False):
                if parse_special and text in (tnc._THINK_OPEN, tnc._THINK_CLOSE):
                    return []
                return super().encode(text, parse_special)

        record = tnc.build(NoThink(), ["write"], True)
        _, trigger, think_open, think_close, call_open, call_close, tails = _parse(record)
        self.assertEqual(trigger, _chars("<function=write"))
        self.assertEqual(tails, [[ord(">")]])
        self.assertEqual(think_open, [])
        self.assertEqual(think_close, [])
        self.assertEqual(call_open, [202])
        self.assertEqual(call_close, [203])

    def test_the_whole_sequence_in_one_token_is_refused(self):
        # If <function=NAME> merges into a single id there is no trigger/name split to mask.
        class OneToken(_Tok):
            def encode(self, text, parse_special=False):
                if text.startswith(tnc._TRIGGER):
                    return [999]
                return super().encode(text, parse_special)

        self.assertIsNone(tnc.build(OneToken(), ["write"], True))


if __name__ == "__main__":
    unittest.main()
