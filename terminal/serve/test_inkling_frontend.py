"""strata-ink2's side of Inkling 2: its reply form through Ink2OutputParser, its effort words, its thinking close and the {% generation %}
tag.  strata-inkling (upstream) keeps InklingOutputParser and thinking off: serve/test_inkling_dialect.py."""
import random

import pytest

from serve import frontend
from serve.frontend import (INK_CALL, INK_END, INK_HEAD, INK_STOP, INK_TEXT, INK_THINK, ChatTemplate, Ink2OutputParser, InklingOutputParser,
                            effort_kwargs, family_text, openai_to_messages, parser_for_template, use_family)

TOOLS = [{"name": n, "description": "", "parameters": {"type": "object", "properties": {}}} for n in ("get_weather", "a", "b")]
CASES = {
    "think+text": (INK_HEAD + INK_THINK + "17*23 = 391." + INK_END + INK_HEAD + INK_TEXT + "391" + INK_END + INK_STOP,
                   ("17*23 = 391.", "391", [])),
    "text only": (INK_HEAD + INK_TEXT + "a <b>c</b> d<|weird|>e" + INK_END + INK_STOP, ("", "a <b>c</b> d<|weird|>e", [])),
    "think+call": (INK_HEAD + INK_THINK + "Need it." + INK_END + INK_HEAD + "get_weather" + INK_CALL
                   + '{"name":"get_weather","args":{"city":"Paris"}}' + INK_END + INK_STOP,
                   ("Need it.", "", [("get_weather", {"city": "Paris"})])),
    "two calls": (INK_HEAD + INK_THINK + "x" + INK_END + INK_HEAD + "a" + INK_CALL + '{"name":"a","args":{}}' + INK_END
                  + INK_HEAD + "b" + INK_CALL + '{"name":"b","args":{"k":1}}' + INK_END + INK_STOP,
                  ("x", "", [("a", {}), ("b", {"k": 1})])),
}


@pytest.fixture
def inkling():
    use_family("inkling2", thinks=True)
    yield
    use_family(None)


def _run(stream: str, rng: random.Random):
    p = Ink2OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
    evs, i = [], 0
    while i < len(stream):
        k = rng.randint(1, 7)
        evs += p.feed(stream[i:i + k])
        i += k
    evs += p.finish("stop")
    return ("".join(e.text for e in evs if e.kind == "reasoning"), "".join(e.text for e in evs if e.kind == "content"),
            [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"])


@pytest.mark.parametrize("name", CASES)
def test_reply_form_any_split(name):
    stream, want = CASES[name]
    rng = random.Random(name)
    for _ in range(100):                 # markers split anywhere across deltas
        assert _run(stream, rng) == want


def test_parser_chosen_by_engine(tmp_path):
    """The same template for both engines: strata-inkling (known by its program's name) gets upstream's parser and thinking off,
    any other engine on Inkling 2's template is strata-ink2."""
    from serve.server import ByteTokenizer, Service
    from serve.test_vram import VramEngine
    t = tmp_path / "t.jinja"
    t.write_text("{#- <|content_thinking|> <|content_invoke_tool_json|> -#}{{ reasoning_effort }}")
    assert parser_for_template(t.read_text()) is InklingOutputParser
    tok = ByteTokenizer()
    try:
        svc = Service(VramEngine(tok), tok, ChatTemplate(t))
        svc.set_family(None)
        assert (svc.family, svc.parser_cls, frontend._INK_THINKS) == ("inkling2", Ink2OutputParser, True)
        assert svc.template.render([], **effort_kwargs("high")) == "high"
        assert svc.template.caps["supports_preserve_reasoning"] is False   # (this template drops it; the official one keeps it)
        eng = VramEngine(tok)
        eng.inkling_engine = True
        svc = Service(eng, tok, ChatTemplate(t))
        svc.set_family("inkling2")
        assert (svc.family, svc.parser_cls, frontend._INK_THINKS) == ("inkling2", InklingOutputParser, False)
        assert svc.template.render([], **effort_kwargs("high")) == "none"
    finally:
        use_family(None)


def test_family_text_and_effort(inkling):
    assert family_text("\n\nDone thinking.\n</think>\n\n") == "\n\nDone thinking." + INK_END + INK_HEAD + INK_TEXT
    assert effort_kwargs(False) == {"reasoning_effort": "none"}
    assert effort_kwargs("high") == {"reasoning_effort": "high"}


def test_qwen_unchanged():
    use_family(None)
    assert family_text("a\n</think>\n\n") == "a\n</think>\n\n"
    assert effort_kwargs(False) == {"enable_thinking": False}
    assert effort_kwargs("high") == {"reasoning_effort": "xhigh"}


def test_thinking_off_is_the_effort_none(inkling):
    """Inkling 2's template reads no enable_thinking: "thinking off" from any client is its effort "none" (the model still writes a short
    thinking message, which the parser keeps out of the answer)."""
    from serve.frontend import anthropic_to_messages
    q = [{"role": "user", "content": "hi"}]
    assert openai_to_messages({"messages": q, "chat_template_kwargs": {"enable_thinking": False}})[2] == {"reasoning_effort": "none"}
    assert anthropic_to_messages({"messages": q, "thinking": {"type": "disabled"}})[2] == {"reasoning_effort": "none"}
    use_family(None)
    assert openai_to_messages({"messages": q, "chat_template_kwargs": {"enable_thinking": False}})[2] == {"enable_thinking": False}
    assert anthropic_to_messages({"messages": q, "thinking": {"type": "disabled"}})[2] == {"enable_thinking": False}


def test_tool_result_named_from_its_call(inkling):
    msgs, _, _ = openai_to_messages({"messages": [
        {"role": "user", "content": "w?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                             "function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "12"}]})
    assert msgs[-1]["name"] == "get_weather"
    use_family(None)
    msgs, _, _ = openai_to_messages({"messages": [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "12"}]})
    assert "name" not in msgs[-1]


def test_forced_call_opening():
    from serve.frontend import forced_call, inkling_force
    tools = [{"name": "get_weather"}, {"name": "other"}]
    named = inkling_force(forced_call({"type": "function", "function": {"name": "get_weather"}}, tools))
    assert named == INK_HEAD + "get_weather" + INK_CALL + '{"name":"get_weather","args":'
    assert inkling_force(forced_call("required", tools)) == INK_HEAD
    assert inkling_force(None) is None
    p = Ink2OutputParser(thinking=True, tools=TOOLS, stream_tools=True)   # the opening, then the model's arguments
    evs = p.feed(INK_HEAD + INK_THINK + "t" + INK_END) + p.feed(named) + p.feed('{"city":"Lyon"}}' + INK_END + INK_STOP)
    evs += p.finish("stop")
    assert [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"] == [("get_weather", {"city": "Lyon"})]


def test_generation_tag(tmp_path):
    t = tmp_path / "t.jinja"
    t.write_text("{% for m in messages %}{% generation %}[{{ m.content }}]{% endgeneration %}{% endfor %}")
    assert ChatTemplate(t).render([{"role": "user", "content": "hi"}]) == "[hi]"
    assert frontend._FAMILY is None
