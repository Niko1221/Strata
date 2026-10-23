"""serve/turn_splice.py - the reply the model wrote, sent back as the ids it wrote (opt-in "splice_turns").

A chat client sends the whole conversation every turn and the server renders and encodes it again.  The engine
continues from the state it holds only when the new prompt starts with every token it holds: the previous prompt and
every token it generated (the recurrent state of a hybrid model cannot step back).  The previous reply comes back
as text, and encoding that text does not always give the ids the model sampled: at temperature 1 the model now and
then writes a piece of text in tokens other than the tokenizer's own split (".\n" as "." + "\n", a file's last line
and the blank line before </parameter> in two tokens instead of one).  The text is the same, the ids are not, and
the engine reads the whole conversation again - 80-105K tokens, 30-45 s - about once in 8000 generated tokens
(measured on two long agent runs with a Qwen3.6-35B-A3B fine-tune (Q4_K_M) on an RTX 5080: every turn the engine re-read had a re-encoded
reply of another length than the one generated; every turn it continued had the same).

With "splice_turns" the server keeps, per recent conversation, the prompt it sent and the ids generated after it.
When a new prompt is that prompt's text plus the reply rendered back (the template's own turn, up to its
<|im_end|>) and more, and the reply's text is the generated ids' text, the new prompt's ids are the ids sent then,
the generated ids and the encode of the rest: the engine's held tokens are their start again.  "parsed" also takes
a reply whose text differs but which reads as the same reasoning, answer and calls (whitespace the output parser
drops, a call's arguments written in another JSON spacing): the prompt then shows the model its reply as written.

A control token the model wrote inside its reply (an <|im_start|>, say) comes back as its text, and the encode marks
that text as text (encode_prompt): such a reply is never spliced, so a role boundary the model was talked into
writing stays text in the next prompt.
"""
from __future__ import annotations

import json
import threading
from array import array
from collections import Counter


def common_prefix(a, b) -> int:
    """Length of the longest common prefix of two id sequences (lists or arrays), by slices (C speed)."""
    n = min(len(a), len(b))
    lo, step = 0, 4096
    while lo < n:
        hi = min(n, lo + step)
        if a[lo:hi] != b[lo:hi]:
            break
        lo, step = hi, step * 2
    else:
        return n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class _Turn:
    """One request's prompt and what the engine generated after it."""
    __slots__ = ("text", "canon", "sent", "raw", "thinking", "tools")

    def __init__(self, text, canon, sent, thinking, tools):
        self.text = text                    # the rendered prompt
        self.canon = array("i", canon)      # its encode
        self.sent = array("i", sent)        # the ids sent (an earlier splice makes them differ from canon)
        self.raw = array("i")               # the ids generated after it (the stop token included)
        self.thinking, self.tools = thinking, tools


class TurnSplicer:
    """See the module docstring.  `mode`: "text" (the reply's text must be the generated text), "parsed", or "off"
    (the turns are kept for diagnose() only)."""

    def __init__(self, tok, stop_ids, im_end: int, keep: int = 4, mode: str = "text", parser=None):
        self.tok, self.stop_ids, self.im_end = tok, set(stop_ids), im_end
        self.control_ids = set()            # the control tokens' ids (<|im_start|>, <|im_end|>, ...)
        for t in getattr(tok, "control_tokens", ()):
            one = tok.encode(t, parse_special=True)
            if len(one) == 1:
                self.control_ids.add(one[0])
        self.keep, self.mode = max(1, keep), mode
        self.parser = parser                # frontend.OutputParser ("parsed")
        self.entries: list[_Turn] = []      # the newest first
        self.pending: list[tuple[object, _Turn]] = []   # (the ids list a run will get, its turn)
        self.lock = threading.Lock()
        self.spliced = 0                    # prompts that got a reply's ids back (for tests and /metrics)

    # ---------------------------------------------------------------- before the run
    def splice(self, text: str, ids: list[int], thinking: bool, tools) -> list[int] | None:
        """The ids to send for prompt `text` (encoded as `ids`): None when no kept turn continues into it."""
        if self.mode == "off":
            return None
        with self.lock:
            entries = list(self.entries)
        for e in entries:
            if len(text) <= len(e.text) or not text.startswith(e.text) or not e.raw:
                continue
            c = len(e.canon)
            if array("i", ids[:c]) != e.canon:     # the encode merged across the old prompt's end
                continue
            try:
                end = ids.index(self.im_end, c)    # the reply's <|im_end|> (one inside a message is text: marked)
            except ValueError:
                continue
            body = e.raw[:-1] if e.raw[-1] in self.stop_ids else e.raw
            if not self._controls_kept(ids[c:end], body) or not self._same(ids[c:end], body, e):
                continue
            with self.lock:
                self.spliced += 1
            return list(e.sent) + list(body) + ids[end:]
        return None

    def _controls_kept(self, turn, body) -> bool:
        """Every control token of the generated reply is one the encode of its text has too (see the docstring)."""
        if not self.control_ids:
            return True
        want = Counter(t for t in body if t in self.control_ids)
        return not want or not (want - Counter(t for t in turn if t in self.control_ids))

    def _same(self, turn, body, e: _Turn) -> bool:
        if list(turn) == list(body):
            return True
        a, b = self.tok.decode(list(turn)), self.tok.decode(list(body))
        if a == b:
            return True
        return self.mode == "parsed" and self.parser is not None and self._read(a, e) == self._read(b, e)

    def _read(self, text: str, e: _Turn):
        """A reply as the client gets it: its reasoning, its answer and its calls."""
        p = self.parser(thinking=e.thinking, tools=e.tools)
        evs = p.feed(text) + p.finish("stop")
        reasoning = "".join(ev.text or "" for ev in evs if ev.kind == "reasoning").strip()
        content = "".join(ev.text or "" for ev in evs if ev.kind == "content").strip()
        calls = [(ev.call.name, json.dumps(ev.call.arguments, sort_keys=True, ensure_ascii=False))
                 for ev in evs if ev.kind == "tool_call"]
        return reasoning, content, calls

    def expect(self, sent: list[int], text: str, canon: list[int], thinking: bool, tools) -> None:
        """A run of `sent` (the very list object Service.run gets) is about to start for prompt `text`."""
        with self.lock:
            self.pending = [(sent, _Turn(text, canon, sent, thinking, tools))] + self.pending[:2 * self.keep]

    # ---------------------------------------------------------------- after the run
    def generated(self, sent: list[int], raw: list[int], ok: bool) -> None:
        """The run of `sent` generated `raw`; `ok`: the engine holds sent + raw (all but the last id).  A run that
        was cancelled, recovered with another prompt or lost the engine's state is not kept."""
        with self.lock:
            k = next((i for i, (s, _) in enumerate(self.pending) if s is sent), None)
            if k is None:
                return
            turn = self.pending.pop(k)[1]
            if not ok or not raw:
                return
            turn.raw = array("i", raw)
            # the turn this one continues is replaced: its prompt is the start of this one's
            self.entries = [turn] + [e for e in self.entries
                                     if not turn.text.startswith(e.text)][:self.keep - 1]

    # ---------------------------------------------------------------- the diagnostic
    def diagnose(self, ids: list[int], width: int = 48) -> str | None:
        """When `ids` does not continue the tokens the engine holds for any kept turn but shares a long start with
        one: where they part, with the text around it (opt-in: it prints conversation text)."""
        with self.lock:
            entries = list(self.entries)
        best, held = 0, None
        for e in entries:
            h = list(e.sent) + list(e.raw[:-1])
            n = common_prefix(h, ids)
            if n == len(h) and n < len(ids):
                return None                         # it continues them: nothing to say
            if n > best:
                best, held = n, (h, len(e.sent))
        if held is None or best < 1024:
            return None
        h, sent = held
        where = f"in the generated reply (its token {best - sent} of {len(h) - sent})" if best >= sent else \
            f"in the prompt before the reply ({sent - best} tokens before its end)"
        d = self.tok.decode
        return (f"the prompt ({len(ids)} tokens) shares {best} of the {len(h)} tokens held for this conversation; "
                f"they part {where}: held {d(h[max(0, best - 8):best])!r} | {d(h[best:best + 8])!r}  "
                f"prompt {d(ids[max(0, best - 8):best])!r} | {d(ids[best:best + 8])!r}  "
                f"(ids held {h[best:best + 4]} prompt {ids[best:best + 4]}; text before: "
                f"{d(ids[max(0, best - width):best])!r})")
