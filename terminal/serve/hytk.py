"""Translate hy4-flash reasoning and tool-call markers for the streaming output parser.

The chat template supplies the marker suffix. Tool names and argument keys can span chunks."""
from __future__ import annotations

import re

from serve.frontend import OutputParser

# `{%- set HYTK = ':6124c78e' %}`, single or double quoted, any spacing.
_HYTK = re.compile(r"""set\s+HYTK\s*=\s*(['"])(.*?)\1""")

_TAG_MAX = 64          # every marker below is far shorter; a longer run starting with '<' is text


def suffix_from_template(text: str) -> str | None:
    """The template's HYTK, or None when this is not an hy4-flash template.

    "" is a real answer and not a failure: a template that builds the markers without a suffix spells them
    `<think>` and `<tool_call>`, which this module reads too."""
    if not text:
        return None
    m = _HYTK.search(text)
    if m is not None:
        return m.group(2)
    # No HYTK, but the hy4-flash markers may still be there literally (a hand-edited template). The arg_key/arg_value
    # pair is the one shape no other family here uses, so it is what identifies the grammar.
    m = re.search(r"<arg_key([^>]*)>", text)
    return m.group(1) if m is not None else None


class Markers:
    """Every literal hy4-flash writes, built from one suffix."""

    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        s = suffix
        self.think_open, self.think_close = f"<think{s}>", f"</think{s}>"
        self.calls_open, self.calls_close = f"<tool_calls{s}>", f"</tool_calls{s}>"
        self.call_open, self.call_close = f"<tool_call{s}>", f"</tool_call{s}>"
        self.key_open, self.key_close = f"<arg_key{s}>", f"</arg_key{s}>"
        self.val_open, self.val_close = f"<arg_value{s}>", f"</arg_value{s}>"
        # The turn markers. They are single tokens and the engine stops on hy_end's id, so they do not normally
        # reach this - but a template rendered into a prompt the model continues can echo one, and a control
        # marker shown to a user as text is worse than a dropped one.
        self.turn = tuple(f"<｜hy_{p}{s}｜>" for p in ("start", "middle", "end"))
        self.all = (self.think_open, self.think_close, self.calls_open, self.calls_close, self.call_open,
                    self.call_close, self.key_open, self.key_close, self.val_open, self.val_close) + self.turn

    def could_still_be_a_tag(self, s: str) -> bool:
        """`s` starts with '<' and has no '>' yet: it is held only if it is the start of a real marker.

        An exact prefix test over the marker list, not a regex of what a tag might look like: the markers are
        known strings, and holding back anything else would delay ordinary text that happens to start with '<'."""
        return len(s) <= _TAG_MAX and any(t.startswith(s) for t in self.all)


class HyTranslator:
    """hy4-flash text -> the Qwen tool-call text, a delta at a time. Text outside a call passes through unchanged."""

    def __init__(self, markers: Markers | None = None):
        self.m = markers or Markers()
        self.buf = ""
        # Parse tool names and arguments incrementally so delimiters may span input chunks.
        self.state = "text"
        self.name = ""
        self.key = ""

    @property
    def held(self) -> str:
        """Text held back because it may be the start of a marker."""
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        """The end of the output: whatever is held is text."""
        return self._run(final=True)

    # ------------------------------------------------------------------------------------------------
    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            lt = self.buf.find("<")
            if lt < 0:
                self._text(self.buf, out)
                self.buf = ""
                break
            if lt:
                self._text(self.buf[:lt], out)
                self.buf = self.buf[lt:]
            tag = self._tag(out)
            if tag == "wait":
                if final:
                    self._text(self.buf, out)
                    self.buf = ""
                break
            if tag is None:                     # an ordinary '<'
                self._text("<", out)
                self.buf = self.buf[1:]
        if final:
            self._close_open(out)
        return "".join(out)

    def _text(self, text: str, out: list[str]) -> None:
        if self.state == "name":
            self.name += text
            return
        if self.state == "key":
            self.key += text
            return
        if self.state in ("calls", "args", "afterkey"):
            if text.strip() == "":
                return                          # the layout between tags
            out.append(text)                    # stray text inside the block: shown, and the block goes on
            return
        out.append(text)

    def _open_call(self, out: list[str]) -> None:
        """Write the call's opening now that its name is complete."""
        out.append(f"<tool_call>\n<function={self.name.strip()}>\n")
        self.name = ""
        self.state = "args"

    def _close_value(self, out: list[str]) -> None:
        if self.state == "value":
            out.append("\n</parameter>\n")
            self.state = "args"

    def _close_call(self, out: list[str]) -> None:
        """End whatever call is open, in any of the states one can be open in."""
        if self.state == "name":                # a call with no arguments at all
            self._open_call(out)
        self._close_value(out)
        if self.state in ("args", "key", "afterkey"):
            out.append("</function>\n</tool_call>\n")
        self.state = "calls"

    def _close_open(self, out: list[str]) -> None:
        """The turn ended mid-call (a stop, a length cap): close it so the parser sees a complete call rather
        than discarding what was written."""
        if self.state != "text":
            self._close_call(out)
            self.state = "text"

    def _tag(self, outl: list[str]):
        """Handle the marker at the start of self.buf. Returns True (one was consumed), "wait" (it may still
        become one) or None (it is not one)."""
        s = self.buf
        gt = s.find(">")
        if gt < 0:
            return "wait" if self.m.could_still_be_a_tag(s) else None
        head = s[:gt + 1]
        if head not in self.m.all:
            return None
        out = outl.append
        eat = lambda: setattr(self, "buf", s[gt + 1:])   # noqa: E731 - one line, used a dozen times below
        m = self.m

        if head in m.turn:                      # a control marker: never text
            eat()
            return True
        if head == m.think_open:
            eat()
            out("<think>")
            return True
        if head == m.think_close:
            # Inside a call the reasoning cannot be ending; a stray close tag there is text.
            if self.state != "text":
                return None
            eat()
            out("</think>")
            return True
        if head == m.calls_open:
            if self.state != "text":
                return None
            eat()
            self.state = "calls"
            return True
        if head == m.calls_close:
            if self.state == "text":
                return None
            eat()
            self._close_call(outl)
            self.state = "text"
            return True
        if head == m.call_open:
            if self.state == "text":
                # The block's opening tag was left out. A call is still a call: the model writing
                # <tool_call> without <tool_calls> is the commonest malformation of this grammar.
                self.state = "calls"
            elif self.state != "calls":
                self._close_call(outl)           # the previous call's closing tag was left out
            eat()
            self.state, self.name = "name", ""
            return True
        if head == m.call_close:
            if self.state == "text":
                return None
            eat()
            self._close_call(outl)
            return True
        if head == m.key_open:
            if self.state == "name":
                self._open_call(outl)            # the name ends where the first key begins
            if self.state == "value":
                self._close_value(outl)          # the value's closing tag was left out
            if self.state != "args":
                return None
            eat()
            self.state, self.key = "key", ""
            return True
        if head == m.key_close:
            if self.state != "key":
                return None
            eat()
            self.state = "afterkey"
            return True
        if head == m.val_open:
            if self.state == "key":             # the key's closing tag was left out
                self.state = "afterkey"
            if self.state != "afterkey":
                return None
            eat()
            self.state = "value"
            out(f"<parameter={self.key.strip()}>\n")
            self.key = ""
            return True
        if head == m.val_close:
            if self.state != "value":
                return None
            eat()
            self._close_value(outl)
            return True
        return None


class HyOutputParser(OutputParser):
    """OutputParser for hy4-flash: the model's own markers are read as the Qwen-form calls they are.

    `markers` is a class attribute because the suffix is a property of the model being served, set once by
    `configure` when the family is chosen - the same shape as frontend.use_family."""

    markers = Markers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hy = HyTranslator(type(self).markers)

    def feed(self, delta: str):
        return super().feed(self.hy.feed(delta))

    def finish(self, reason: str | None = None):
        rest = self.hy.flush()
        out = super().feed(rest) if rest else []
        return out + super().finish(reason)

    @property
    def clean(self) -> bool:
        """No marker is held back anywhere: a clean point to stop the thinking at."""
        return not self.buf and not self.hy.held


def configure(suffix: str) -> None:
    """Set the suffix every HyOutputParser from here on reads."""
    HyOutputParser.markers = Markers(suffix)


def to_hy_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening in hy4-flash's spelling."""
    if not force:
        return force
    m = HyOutputParser.markers
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    if name.endswith(">\n"):
        name = name[:-2]
    return m.calls_open + "\n" + m.call_open + name
