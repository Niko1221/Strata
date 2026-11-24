"""Server regression tests for hy4-flash chat, tool calls and picture inputs."""
from __future__ import annotations

import io
import json
import struct
import threading
from pathlib import Path
from unittest import mock

import pytest

from serve import frontend, hytk
from serve.hy4_flash_pictures import Hy4FlashPictures
from serve.server import ByteTokenizer, MockEngine, Service, arch_from_tokenizer, engine_args
from tools.strata_tokenizer import Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "hy4_flash" / "chat_template.jinja"
TK = ":6124c78e"


def translate(text: str, suffix: str = TK, chunk: int | None = None) -> str:
    """`text` through the translator, optionally in `chunk`-byte pieces."""
    t = hytk.HyTranslator(hytk.Markers(suffix))
    if chunk is None:
        return t.feed(text) + t.flush()
    out = [t.feed(text[i:i + chunk]) for i in range(0, len(text), chunk)]
    out.append(t.flush())
    return "".join(out)


def every_split(text: str, suffix: str = TK) -> str:
    """The same text fed one byte at a time; the result every other split must equal."""
    return translate(text, suffix, chunk=1)


def test_suffix_comes_from_the_real_template():
    assert hytk.suffix_from_template(FIXTURE.read_text(encoding="utf-8")) == TK


def test_suffix_other_spellings():
    assert hytk.suffix_from_template('{%- set HYTK = ":abc123" %}') == ":abc123"
    assert hytk.suffix_from_template("{%-set HYTK=':x' -%}") == ":x"
    # A template that builds the markers with no suffix at all: "" is an answer, not a failure.
    assert hytk.suffix_from_template("{%- set HYTK = '' %}") == ""
    # Not an hy4-flash template.
    assert hytk.suffix_from_template("{% for m in messages %}{{ m.content }}{% endfor %}") is None
    assert hytk.suffix_from_template("") is None


def test_suffix_from_a_hand_edited_template_without_hytk():
    """No HYTK, but the arg_key/arg_value pair is there literally - the one shape no other family writes."""
    assert hytk.suffix_from_template("...<arg_key:zz>{k}</arg_key:zz>...") == ":zz"


def test_thinking_tags_are_translated():
    got = translate("thinking out loud</think:6124c78e>the answer")
    assert got == "thinking out loud</think>the answer"
    assert every_split("thinking out loud</think:6124c78e>the answer") == got


def test_open_think_tag_too():
    assert translate("<think:6124c78e>why</think:6124c78e>so") == "<think>why</think>so"


def test_text_that_only_looks_like_a_marker_is_text():
    for text in ("a < b", "1<2 and 3>2", "</think>", "<think>", "<tool_call>", "x</thinking:6124c78e>y"):
        assert translate(text) == text, text
        assert every_split(text) == text, text


def test_a_marker_with_the_wrong_suffix_is_text():
    """Another release's suffix read with this one's: plain text, which is the honest outcome - and the reason
    the suffix is read from the template rather than guessed."""
    assert translate("a</think:deadbeef>b") == "a</think:deadbeef>b"


CALL = (f"<tool_calls{TK}>\n"
        f"<tool_call{TK}>get_weather"
        f"<arg_key{TK}>city</arg_key{TK}><arg_value{TK}>Zagreb</arg_value{TK}>"
        f"<arg_key{TK}>days</arg_key{TK}><arg_value{TK}>3</arg_value{TK}>"
        f"</tool_call{TK}>\n"
        f"</tool_calls{TK}>")


def test_one_call_two_arguments():
    assert translate(CALL) == ("<tool_call>\n<function=get_weather>\n"
                               "<parameter=city>\nZagreb\n</parameter>\n"
                               "<parameter=days>\n3\n</parameter>\n"
                               "</function>\n</tool_call>\n")


def test_a_call_is_the_same_however_it_is_split():
    whole = translate(CALL)
    for chunk in range(1, len(CALL) + 1):
        assert translate(CALL, chunk=chunk) == whole, f"split every {chunk} bytes"


def test_nothing_is_emitted_before_a_marker_is_complete():
    """The invariant a client depends on: a partial marker is held, never shown."""
    t = hytk.HyTranslator(hytk.Markers(TK))
    assert t.feed("answer<tool_cal") == "answer"
    assert t.held == "<tool_cal"
    assert t.feed(f"ls{TK}>") == ""      # completes <tool_calls...>: consumed, nothing shown
    assert t.held == ""


def test_the_tool_name_is_never_shown_as_text():
    t = hytk.HyTranslator(hytk.Markers(TK))
    out = t.feed(f"<tool_calls{TK}><tool_call{TK}>get_wea")
    assert out == ""                     # the name is being collected, not streamed
    out += t.feed(f"ther<arg_key{TK}>")
    assert "<function=get_weather>" in out


def test_two_calls_in_one_block():
    text = (f"<tool_calls{TK}>"
            f"<tool_call{TK}>a<arg_key{TK}>x</arg_key{TK}><arg_value{TK}>1</arg_value{TK}></tool_call{TK}>"
            f"<tool_call{TK}>b<arg_key{TK}>y</arg_key{TK}><arg_value{TK}>2</arg_value{TK}></tool_call{TK}>"
            f"</tool_calls{TK}>")
    got = translate(text)
    assert got.count("<tool_call>") == 2
    assert "<function=a>" in got and "<function=b>" in got
    assert every_split(text) == got


def test_a_call_with_no_arguments():
    text = f"<tool_calls{TK}><tool_call{TK}>now</tool_call{TK}></tool_calls{TK}>"
    assert translate(text) == "<tool_call>\n<function=now>\n</function>\n</tool_call>\n"
    assert every_split(text) == translate(text)


def test_a_json_value_passes_through_as_written():
    """A non-string argument is JSON in the template's own convention; the translator does not touch it."""
    value = json.dumps({"a": [1, 2], "b": "x"})
    text = (f"<tool_calls{TK}><tool_call{TK}>f<arg_key{TK}>o</arg_key{TK}>"
            f"<arg_value{TK}>{value}</arg_value{TK}></tool_call{TK}></tool_calls{TK}>")
    assert value in translate(text)


def test_a_value_containing_angle_brackets():
    value = "if (a<b) { return a>b; }"
    text = (f"<tool_calls{TK}><tool_call{TK}>f<arg_key{TK}>code</arg_key{TK}>"
            f"<arg_value{TK}>{value}</arg_value{TK}></tool_call{TK}></tool_calls{TK}>")
    assert value in translate(text)
    assert every_split(text) == translate(text)


def test_a_call_without_its_block_is_still_a_call():
    """The commonest malformation of this grammar: <tool_call> with no <tool_calls> around it."""
    text = (f"<tool_call{TK}>f<arg_key{TK}>k</arg_key{TK}><arg_value{TK}>v</arg_value{TK}>"
            f"</tool_call{TK}>")
    assert translate(text) == ("<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n"
                               "</function>\n</tool_call>\n")


def test_a_missing_call_close_is_closed_by_the_next_call():
    text = (f"<tool_calls{TK}><tool_call{TK}>a<arg_key{TK}>x</arg_key{TK}><arg_value{TK}>1</arg_value{TK}>"
            f"<tool_call{TK}>b<arg_key{TK}>y</arg_key{TK}><arg_value{TK}>2</arg_value{TK}></tool_call{TK}>"
            f"</tool_calls{TK}>")
    got = translate(text)
    assert got.count("</tool_call>") == 2 and got.count("<tool_call>") == 2


def test_a_turn_that_ends_mid_call_still_yields_the_call():
    """A stop or a length cap inside a call: flush() closes it rather than dropping what was written."""
    text = f"<tool_calls{TK}><tool_call{TK}>f<arg_key{TK}>k</arg_key{TK}><arg_value{TK}>half"
    got = translate(text)
    assert "<function=f>" in got and "<parameter=k>" in got and "half" in got
    assert got.rstrip().endswith("</tool_call>")


def test_a_missing_value_close_is_closed_by_the_next_key():
    text = (f"<tool_calls{TK}><tool_call{TK}>f<arg_key{TK}>a</arg_key{TK}><arg_value{TK}>1"
            f"<arg_key{TK}>b</arg_key{TK}><arg_value{TK}>2</arg_value{TK}></tool_call{TK}></tool_calls{TK}>")
    got = translate(text)
    assert "<parameter=a>" in got and "<parameter=b>" in got and got.count("</parameter>") == 2


def test_turn_markers_are_dropped_not_shown():
    assert translate(f"hello<｜hy_end{TK}｜>") == "hello"
    assert translate(f"<｜hy_start{TK}｜>assistant<｜hy_middle{TK}｜>hi") == "assisthi".replace("assist", "assistant")


def test_the_bare_spelling_parses_too():
    text = ("<tool_calls><tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>"
            "</tool_call></tool_calls>")
    assert translate(text, suffix="") == ("<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n"
                                          "</function>\n</tool_call>\n")


def test_forced_call_opening_is_written_in_hy4_flash_spelling():
    hytk.configure(TK)
    assert hytk.to_hy_force("<tool_call>\n<function=get_weather>\n") == \
        f"<tool_calls{TK}>\n<tool_call{TK}>get_weather"
    assert hytk.to_hy_force(None) is None
    assert hytk.to_hy_force("") == ""


def test_the_parser_reports_a_tool_call():
    hytk.configure(TK)
    p = hytk.HyOutputParser(thinking=True, tools=[{"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}])
    events = list(p.feed(f"I should look it up.</think{TK}>")) + list(p.feed(CALL)) + list(p.finish("stop"))
    kinds = [e.kind for e in events]
    assert "reasoning" in kinds, kinds
    calls = [e for e in events if e.kind == "tool_call"]
    assert len(calls) == 1, kinds
    call = calls[0]
    name = getattr(call, "name", None) or getattr(getattr(call, "call", None), "name", None)
    assert name == "get_weather", call
    args = getattr(call, "arguments", None) or getattr(getattr(call, "call", None), "arguments", None)
    if isinstance(args, str):
        args = json.loads(args)
    assert args == {"city": "Zagreb", "days": 3}, args


def test_the_parser_splits_thinking_from_the_answer():
    hytk.configure(TK)
    p = hytk.HyOutputParser(thinking=True)
    events = list(p.feed(f"let me think.</think{TK}>The capital of France is Paris.")) + list(p.finish("stop"))
    reasoning = "".join(e.text for e in events if e.kind == "reasoning")
    content = "".join(e.text for e in events if e.kind == "content")
    assert reasoning.strip() == "let me think."
    assert content.strip() == "The capital of France is Paris."


def test_clean_is_false_while_a_marker_is_held():
    hytk.configure(TK)
    p = hytk.HyOutputParser(thinking=False)
    list(p.feed("text<tool_cal"))
    assert not p.clean
    list(p.feed(f"ls{TK}>"))
    assert p.clean


def test_every_effort_word_renders_against_the_real_template():
    """hy4-flash's template RAISES on a level it does not know, so a mapping that sends "high" to "xhigh" (the Qwen
    table) is a 500 on a request that spelled it. This renders the real template for every spelling a client
    may use and requires each to come out as one of the four levels the template accepts."""
    import jinja2

    from serve import frontend

    source = FIXTURE.read_text(encoding="utf-8")

    def render(kwargs):
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 undefined=jinja2.ChainableUndefined,
                                 extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(AssertionError(m))
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, **kw)
        return env.from_string(source).render(messages=[{"role": "user", "content": "hi"}],
                                              add_generation_prompt=True, **kwargs)

    frontend.use_family("hy4-flash")
    try:
        for word in ("low", "medium", "high", "xhigh", "max", "maximum", "none", "off", "false", "disabled"):
            kwargs = frontend.effort_kwargs(word)
            out = render(kwargs)                      # raises AssertionError if the template refuses the level
            assert "reasoning_effort:" in out, (word, kwargs)
        # "off" must actually turn the thinking off, which for hy4-flash is no_think and NOT enable_thinking.
        off = frontend.effort_kwargs("none")
        assert off == {"reasoning_effort": "no_think"}, off
        out = render(off)
        assert out.endswith("<think:6124c78e></think:6124c78e>"), out[-60:]
        # and a thinking turn leaves the tag open for the model to continue
        assert render(frontend.effort_kwargs("high")).endswith("<think:6124c78e>")
    finally:
        frontend.use_family(None)


def test_the_qwen_words_still_work_for_qwen():
    """use_family(None) must put the old table back: this file must not change another family's behaviour."""
    from serve import frontend

    frontend.use_family("hy4-flash")
    frontend.use_family(None)
    assert frontend.effort_kwargs("high") == {"reasoning_effort": "xhigh"}
    assert frontend.effort_kwargs("none") == {"enable_thinking": False}


# A tool call can immediately follow reasoning and end at the turn marker.
TOOL_REPLY = f"Check the forecast.</think{TK}>" + CALL.replace("\n", "") + f"<｜hy_end{TK}｜>"


def test_tool_reply_hides_control_markers():
    hytk.configure(TK)
    parser = hytk.HyOutputParser(thinking=True, tools=[{"name": "get_weather"}])
    events = list(parser.feed(TOOL_REPLY)) + list(parser.finish("stop"))
    assert "".join(e.text for e in events if e.kind == "reasoning") == "Check the forecast."
    assert TK not in "".join(e.text for e in events if e.kind == "content")
    assert sum(e.kind == "tool_call" for e in events) == 1


def test_tool_reply_streamed_one_character_at_a_time():
    hytk.configure(TK)
    text = TOOL_REPLY
    schemas = [{"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}}}]
    whole = hytk.HyOutputParser(thinking=True, tools=schemas)
    a = list(whole.feed(text)) + list(whole.finish("stop"))
    piece = hytk.HyOutputParser(thinking=True, tools=schemas)
    b = []
    for ch in text:
        b += list(piece.feed(ch))
    b += list(piece.finish("stop"))

    def calls(evs):
        out = []
        for e in evs:
            if e.kind != "tool_call":
                continue
            name = getattr(e, "name", None) or getattr(getattr(e, "call", None), "name", None)
            args = getattr(e, "arguments", None) or getattr(getattr(e, "call", None), "arguments", None)
            out.append((name, json.loads(args) if isinstance(args, str) else args))
        return out

    assert calls(a) == calls(b) != []
    assert "".join(e.text for e in a if e.kind == "reasoning") == \
           "".join(e.text for e in b if e.kind == "reasoning")


@pytest.fixture
def service():
    tok = ByteTokenizer()
    tok.special_ids = {"eos": 120025}
    template = frontend.ChatTemplate(FIXTURE)
    svc = Service(MockEngine(tok, "hello"), tok, template)
    svc.set_family("hy4-flash")
    yield svc
    frontend.use_family(None)


@pytest.mark.parametrize("extra", [
    {"reasoning_effort": "none"}, {"reasoning_effort": "no_think"},
    {"reasoning": {"effort": "none"}},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"}},
])
def test_no_think_outputs_content_not_reasoning(service, extra):
    messages, tools, kwargs = frontend.openai_to_messages({"messages": [{"role": "user", "content": "hi"}], **extra})
    ids, thinking, _ = service.prepare(messages, tools, kwargs, 32)
    assert not thinking
    assert service.tok.decode(ids).endswith("<think:6124c78e></think:6124c78e>")
    parser = service.parser_cls(thinking=thinking)
    events = parser.feed("hello") + parser.finish("stop")
    assert any(e.kind == "content" and e.text == "hello" for e in events)


def test_anthropic_disabled_and_budget_use_hy4_flash_levels(service):
    req = {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
    messages, tools, kwargs = frontend.anthropic_to_messages(req)
    assert service.prepare(messages, tools, kwargs, 32)[1] is False
    assert frontend.budget_effort(4096) == {"reasoning_effort": "high"}
    service.template.render(messages, **frontend.budget_effort(4096))


def test_exported_special_id_keys_and_arch_are_normalized():
    tok = Tokenizer(["a", "END"], [], [1, 3], "hy_v4", {"tokenizer.ggml.eos_token_id": 1})
    assert tok.special_ids["eos"] == 1
    assert arch_from_tokenizer(None, "hy_v4") == "hy4-flash"
    assert arch_from_tokenizer("qwen4exp", "hy_v4") == "qwen4exp"


def test_reasoning_budget_closes_the_hy4_flash_marker(service):
    service.engine = MockEngine(service.tok, ["thinking " * 100, "the answer"])
    service.stop_ids = set(service.tok.encode("<|im_end|>", parse_special=True))
    messages, tools, kwargs = frontend.openai_to_messages({"messages": [{"role": "user", "content": "hi"}]})
    ids, thinking, limit = service.prepare(messages, tools, kwargs, 256)
    result = list(service.run(ids, thinking, tools, limit, {"reasoning_budget_tokens": 8}, threading.Event()))
    prompt = service.tok.decode(service.engine.last_prompt)
    assert "</think:6124c78e>" in prompt
    content = "".join(event.text for kind, event in result if kind == "event" and event.kind == "content")
    assert "the answer" in content


def test_hy4_flash_does_not_run_qwen_argument_rewriters():
    cfg = {"arch": "hy4-flash", "args": ["--context", "8192"], "gpu": 0, "parallel": 1}
    with mock.patch("serve.server.drafting.apply", side_effect=AssertionError("Qwen drafter")), \
         mock.patch("serve.server.vram_fit.fit_start", side_effect=AssertionError("Qwen fit")):
        assert engine_args(cfg) == cfg["args"]
    with pytest.raises(ValueError, match="one inference slot"):
        engine_args({**cfg, "parallel": 2})


def png(size, color):
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", size, color).save(image, "PNG")
    return image.getvalue()


def test_hy4_flash_bundle_matches_token_spans_and_rgb(tmp_path):
    encoder = Hy4FlashPictures(tmp_path)
    pad = encoder.image_token
    ids, bundle = encoder.build([1, pad, 2, pad, 3], [png((64, 64), "red"), png((96, 64), "blue")])
    assert ids == [1] + [pad] * 6 + [2] + [pad] * 8 + [3]
    with open(bundle, "rb") as f:
        assert f.read(8) == b"DSI1" + struct.pack("<i", 2)
        for start, width, rgb in ((1, 64, b"\xff\x00\x00"), (8, 96, b"\x00\x00\xff")):
            assert struct.unpack("<qii", f.read(16)) == (start, width, 64)
            assert f.read(3 * width * 64) == rgb * (width * 64)
        assert not f.read()


def test_hy4_flash_downscale_at_patch_boundary(tmp_path):
    width, height, _ = Hy4FlashPictures(tmp_path).prepare(png((2078, 2078), "red"))
    assert (width, height) == (2048, 2048)


def test_hy4_flash_invalid_image(tmp_path):
    with pytest.raises(ValueError, match="could not read"):
        Hy4FlashPictures(tmp_path).prepare(b"not a picture")
