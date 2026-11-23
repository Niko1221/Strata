"""Unit tests for serve/inkling_dialect.py, serve/inkling_kind.py, and frontend Inkling-2-Mini integration."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from serve import frontend, inkling_dialect, inkling_kind, toolformat
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402


class TestInklingDialect(unittest.TestCase):
    def test_streaming_thinking_discarded_and_text_byte_by_byte(self):
        raw = (
            "<|message_model|><|content_thinking|>Let me check 2 + 2 = 4.<|end_message|>"
            "<|message_model|><|content_text|>The answer is 4.<|end_message|>"
            "<|content_model_end_sampling|>"
        )
        # Feed 1 character at a time: raw <|content_thinking|> is silently discarded (no thinking mode), text streams cleanly
        d = inkling_dialect.InklingDialect()
        out = "".join(d.feed(ch) for ch in raw) + d.feed(final=True)
        self.assertEqual(out, "The answer is 4.")
        self.assertEqual(d.calls, 0)

    def test_tool_call_preserves_xml_chars_quotes_and_closing_tags(self):
        code = (
            'def check(a, b):\n'
            '    xml = "<tag attr=\\"x\\">a < b && c > d </parameter> </tool_call>"\n'
            "    return xml + 'single'\n"
        )
        payload = json.dumps({"name": "write_file", "args": {"path": "src/a&b.py", "content": code, "overwrite": True, "lines": [1, 2, 3]}})
        raw = (
            "<|message_model|><|content_thinking|>Writing patch with < and > and </tool_call>.<|end_message|>"
            f"<|message_model|>write_file<|content_invoke_tool_json|>{payload}<|end_message|>"
            "<|content_model_end_sampling|>"
        )
        # Feed raw Inkling-2-Mini tokens in awkward 3-char chunks directly into InklingOutputParser
        p = frontend.InklingOutputParser(tools=[{"name": "write_file", "parameters": {"type": "object"}}])
        reasoning_parts, content_parts, tool_calls = [], [], []
        for i in range(0, len(raw), 3):
            for ev in p.feed(raw[i:i + 3]):
                if ev.kind == "reasoning":
                    reasoning_parts.append(ev.text)
                elif ev.kind == "content":
                    content_parts.append(ev.text)
                elif ev.kind == "tool_call" and ev.call is not None:
                    tool_calls.append({"name": ev.call.name, "arguments": ev.call.arguments})
        for ev in p.finish():
            if ev.kind == "reasoning":
                reasoning_parts.append(ev.text)
            elif ev.kind == "content":
                content_parts.append(ev.text)
            elif ev.kind == "tool_call" and ev.call is not None:
                tool_calls.append({"name": ev.call.name, "arguments": ev.call.arguments})

        self.assertEqual("".join(reasoning_parts), "")
        self.assertEqual("".join(content_parts), "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "write_file")
        self.assertEqual(tool_calls[0]["arguments"]["path"], "src/a&b.py")
        self.assertEqual(tool_calls[0]["arguments"]["content"], code)
        self.assertIs(tool_calls[0]["arguments"]["overwrite"], True)
        self.assertEqual(tool_calls[0]["arguments"]["lines"], [1, 2, 3])

    def test_multiple_parallel_tool_calls_in_one_turn(self):
        raw = (
            '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"a.py"}}<|end_message|>'
            '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"b.py"}}<|end_message|>'
            '<|content_model_end_sampling|>'
        )
        p = frontend.InklingOutputParser(tools=[{"name": "read_file", "parameters": {"type": "object"}}])
        events = p.feed(raw) + p.finish()
        calls = [ev.call for ev in events if ev.kind == "tool_call" and ev.call is not None]
        self.assertEqual([c.arguments["path"] for c in calls], ["a.py", "b.py"])

    def test_no_thinking_mode_always_off(self):
        # Inkling-2-Mini has no thinking mode: default is off ("none") and every input normalizes to "none"
        for level in (None, "", "none", "off", "minimal", "low", "medium", "high", "max", "xhigh", 0.0, 0.9):
            self.assertEqual(inkling_dialect.normalize_effort(level), "none")
        self.assertEqual(inkling_dialect.template_kwargs({}), {"enable_thinking": False, "reasoning_effort": "none"})
        for level in ("none", "minimal", "low", "medium", "high", "max"):
            got = inkling_dialect.template_kwargs({"reasoning_effort": level})
            self.assertEqual(got, {"enable_thinking": False, "reasoning_effort": "none"})
        self.assertEqual(inkling_dialect.template_kwargs({"enable_thinking": True, "thinking_mode": "high"}), {"enable_thinking": False, "reasoning_effort": "none"})

    def test_chat_template_empty_kwargs_forces_effort_zero_and_caps_no_reasoning(self):
        import tempfile
        tpl_src = (
            "{%- set effort_map = {'none': 0.0, 'high': 0.9} -%}"
            "{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"
            "{%- set num = effort_map[eff] if eff is string else eff -%}"
            "<|message_system|><|content_text|>Thinking effort level: {% if num == 0.0 %}0{% else %}{{ num }}{% endif %}<|end_message|>"
            "{%- if tools -%}<|message_system|>tool_declare<|content_xml|>{{ tools | tojson }}<|end_message|>{%- endif -%}"
            "{%- for m in messages -%}"
            "{%- if m.role == 'system' -%}<|message_system|><|content_text|>{{ m.content }}<|end_message|>"
            "{%- elif m.role == 'user' -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>"
            "{%- elif m.role == 'assistant' -%}"
            "{%- if m.tool_calls -%}{%- for tc in m.tool_calls -%}<|message_model|>{{ tc.function.name }}<|content_invoke_tool_json|>{{ tc.function.arguments | tojson }}<|end_message|>{%- endfor -%}{%- endif -%}"
            "{%- if m.content -%}<|message_model|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
            "{%- elif m.role == 'tool' -%}<|message_tool|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
            "{%- endfor -%}"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chat_template.jinja"
            p.write_text(tpl_src, encoding="utf-8")
            tpl = frontend.ChatTemplate(p)
            rendered = tpl.render([{"role": "user", "content": "hi"}])
            self.assertIn("Thinking effort level: 0<|end_message|>", rendered)
            self.assertFalse(tpl.caps["supports_preserve_reasoning"])
            self.assertTrue(tpl.caps["supports_tools"])

    def test_prepare_messages_converts_json_string_tool_args_to_dicts(self):
        msgs = [
            {"role": "user", "content": "check file"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "plan",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
        ]
        out = inkling_dialect.prepare_messages(msgs)
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], {"path": "a.py"})
        # Original input is not mutated
        self.assertIsInstance(msgs[1]["tool_calls"][0]["function"]["arguments"], str)

    def test_toolformat_detects_inkling2(self):
        sample = "<|message_system|>tool_declare<|content_xml|>...<|content_invoke_tool_json|>"
        self.assertEqual(toolformat.tool_format(sample), "inkling2")

    def test_inkling_kind_detection_and_arg_validation(self):
        self.assertTrue(inkling_kind.is_inkling({"arch": "inkling2_loop"}))
        self.assertTrue(inkling_kind.is_inkling({"exe": "/opt/strata/strata-inkling"}))
        self.assertTrue(inkling_kind.is_inkling({"exe": r"C:\strata\strata-inkling.exe"}))
        self.assertFalse(inkling_kind.is_inkling({"exe": "/opt/strata/strata-hy4"}))

        info = inkling_kind.parse_info("arch=inkling2_loop layers=24 visits=48 eos=200006 image_pad=200054 parallel=64")
        self.assertEqual(info["arch"], "inkling2_loop")
        self.assertEqual(int(info["image_pad"]), 200054)
        self.assertEqual(int(info["parallel"]), 64)

        class _FakeEngine:
            inkling_engine = True

        self.assertIsNone(inkling_kind.refuse(_FakeEngine(), {"parallel": 8}))
        with self.assertRaises(ValueError):
            inkling_kind.refuse(_FakeEngine(), {"logprobs": True})

        built = inkling_kind.engine_args(
            {"args": ["--native", "Inkling-2-Mini-SQ-20.strata", "--max-context", "32768", "--vision", "Inkling-2-Mini-Vision.strata"], "parallel": 32, "gpu": [0, 1]},
            announce=False,
        )
        self.assertIn("--parallel", built)
        self.assertEqual(built[built.index("--parallel") + 1], "32")
        self.assertIn("--gpus", built)
        self.assertEqual(built[built.index("--gpus") + 1], "1,0")
        self.assertIn("--layer-split", built)
        self.assertEqual(built[built.index("--layer-split") + 1], "auto")

    def test_multigpu_main_first_ordinals_and_stale_vram_strip_6631(self):
        base = ["--native", "Inkling-2-Mini-SQ-15.strata", "--max-context", "32768"]
        # 5080 (GPU 0) = Main, 5070 Ti (GPU 1) = Helper -> child_env sets CUDA_VISIBLE_DEVICES=1,0 ->
        # local ordinal 1 is Main (5080) and local 0 is Helper (5070 Ti), so --gpus MUST be 1,0!
        cfg_roles = {
            "arch": "inkling2_loop",
            "backend": "cuda",
            "gpu_roles": {"0": "main", "1": "helper"},
            "args": base + ["--vram-experts-gb", "10.50", "--ram-experts-gb", "1.78"],
        }
        got = inkling_kind.engine_args(cfg_roles, announce=False)
        self.assertEqual(got[got.index("--gpus") + 1], "1,0")
        self.assertEqual(got[got.index("--layer-split") + 1], "auto")
        self.assertNotIn("--vram-experts-gb", got)
        self.assertNotIn("--ram-experts-gb", got)

        # Bare count --gpus 2 written by setup is rewritten to Main-first 1,0 unless explicitly in user_args
        cfg_setup = {
            "arch": "inkling2_loop",
            "backend": "cuda",
            "gpu_roles": {"0": "main", "1": "helper"},
            "args": base + ["--gpus", "2"],
        }
        got_setup = inkling_kind.engine_args(cfg_setup, announce=False)
        self.assertEqual(got_setup[got_setup.index("--gpus") + 1], "1,0")

        # Explicit user_args --gpus / --vram-experts-gb are respected verbatim
        cfg_user = {
            "arch": "inkling2_loop",
            "backend": "cuda",
            "gpu": [0, 1],
            "args": list(base),
            "user_args": ["--gpus", "0,1", "--vram-experts-gb", "9.00"],
        }
        got_user = inkling_kind.engine_args(cfg_user, announce=False)
        self.assertEqual(got_user[got_user.index("--gpus") + 1], "0,1")
        self.assertIn("--vram-experts-gb", got_user)

        # 3 & 4 CUDA GPUs emit Main-first local ordinals + --layer-split auto; HIP/SYCL cap at 2 GPUs (Main + last Helper)
        for gpu_sel, bkd, want in (
            ([0, 1, 2], "cuda", "2,0,1"),
            ([0, 1, 2, 3], "cuda", "3,0,1,2"),
            ([0, 1], "hip", "1,0"),
            ([0, 1, 2], "hip", "2,1"),
            ([0, 1], "sycl", "1,0"),
        ):
            c = {"arch": "inkling2_loop", "backend": bkd, "gpu": gpu_sel, "args": list(base)}
            a = inkling_kind.engine_args(c, announce=False)
            self.assertEqual(a[a.index("--gpus") + 1], want, (gpu_sel, bkd, a))
            if bkd == "cuda":
                self.assertEqual(a[a.index("--layer-split") + 1], "auto")
            else:
                self.assertNotIn("--layer-split", a)

        # Explicit --ep-split keeps EP2 without injecting --layer-split auto
        cfg_ep = {"arch": "inkling2_loop", "backend": "cuda", "gpu": [0, 1], "args": base + ["--ep-split", "0.5"]}
        got_ep = inkling_kind.engine_args(cfg_ep, announce=False)
        self.assertEqual(got_ep[got_ep.index("--gpus") + 1], "1,0")
        self.assertNotIn("--layer-split", got_ep)

    def test_dspark_drafting_off_flag_and_drafting_info_6633(self):
        base = ["--native", "Inkling-2-Mini-SQ-20.strata", "--max-context", "32768"]
        # drafting: "off" MUST emit --draft off so strata-inkling does not default to auto/dspark
        got_off = inkling_kind.engine_args({"arch": "inkling2_loop", "drafting": "off", "args": list(base)}, announce=False)
        self.assertIn("--draft", got_off)
        self.assertEqual(got_off[got_off.index("--draft") + 1], "off")
        self.assertEqual(inkling_kind.drafting_info(got_off, {"drafter": "dspark"}), {"drafting": "off", "drafting_note": ""})

        # --draft dspark or INFO drafter=dspark reports drafting: "dspark"
        got_on = base + ["--draft", "dspark"]
        self.assertEqual(inkling_kind.drafting_info(got_on, {"drafter": "dspark"}), {"drafting": "dspark", "drafting_note": ""})
        self.assertEqual(inkling_kind.drafting_info(base, {"drafter": "dspark"}), {"drafting": "dspark", "drafting_note": ""})
        self.assertEqual(inkling_kind.drafting_info(base, {"drafter": "none"}), {"drafting": "off", "drafting_note": ""})

    def test_finish_prompt_skips_hidden_thinking_for_plain_and_tool_turns_6612_6627(self):
        base_prompt = (
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            "<|message_user|><|content_text|>Say hello in five words.<|end_message|>"
        )
        plain = inkling_dialect.finish_prompt(base_prompt, has_tools=False)
        self.assertTrue(plain.endswith("<|message_model|><|content_text|>"))
        # Idempotent
        self.assertEqual(inkling_dialect.finish_prompt(plain, has_tools=False), plain)

        # With tools: closes an empty thinking turn so model immediately chooses text or tool invocation
        with_tools = inkling_dialect.finish_prompt(base_prompt, has_tools=True)
        self.assertTrue(with_tools.endswith("<|message_model|><|content_thinking|><|end_message|>"))
        self.assertEqual(inkling_dialect.finish_prompt(with_tools, has_tools=True), with_tools)

        # Already ending with <|message_model|>
        self.assertEqual(
            inkling_dialect.finish_prompt(base_prompt + "<|message_model|>", has_tools=False),
            base_prompt + "<|message_model|><|content_text|>",
        )
        self.assertEqual(
            inkling_dialect.finish_prompt(base_prompt + "<|message_model|>", has_tools=True),
            base_prompt + "<|message_model|><|content_thinking|><|end_message|>",
        )
        # Forced tool call leaves prompt untouched
        self.assertEqual(inkling_dialect.finish_prompt(base_prompt, has_tools=True, forced=True), base_prompt)

    def test_dialect_hardening_prevents_empty_replies_on_missing_tags_and_yaml_6620(self):
        # 1. Model emits <|message_model|> followed directly by YAML/prose (omitting <|content_text|>):
        # header state flushes on newline / close instead of discarding the entire reply!
        yaml_reply = (
            "<|message_model|>Her er problemerne i din docker-compose-fil:\n"
            "```yaml\nservices:\n  db:\n    image: postgres:16\n```\n"
            "<|end_message|><|content_model_end_sampling|>"
        )
        p = frontend.InklingOutputParser(tools=[
            {"name": "search_web", "parameters": {"type": "object"}},
            {"name": "fetch_article", "parameters": {"type": "object"}},
        ])
        evs = []
        for i in range(0, len(yaml_reply), 4):
            evs.extend(p.feed(yaml_reply[i:i + 4]))
        evs.extend(p.finish())
        content = "".join(ev.text for ev in evs if ev.kind == "content")
        self.assertIn("postgres:16", content)
        self.assertIn("Her er problemerne", content)

        # 2. Short single-line reply with <|message_model|>...<|end_message|> (no <|content_text|>) still flushes
        short_missing_tag = "<|message_model|>Hello world!<|end_message|>"
        self.assertEqual(inkling_dialect.translate(short_missing_tag), "Hello world!")

        # 3. Direct <|content_text|> transition out of <|content_thinking|> (omitting <|end_message|><|message_model|>)
        direct_exit = (
            "<|message_model|><|content_thinking|>internal plan"
            "<|content_text|>Visible YAML answer.<|end_message|>"
        )
        self.assertEqual(inkling_dialect.translate(direct_exit), "Visible YAML answer.")

        # 4. Truncated non-JSON inside <|content_invoke_tool_json|> on final=True releases text instead of bogus raw call
        truncated_prose = "<|message_model|>search_web<|content_invoke_tool_json|>services:\n  db:\n    image: postgres:16"
        d = inkling_dialect.InklingDialect()
        out = d.feed(truncated_prose, final=True)
        self.assertIn("postgres:16", out)
        self.assertNotIn("<tool_call>", out)

        # 5. Model continuing directly from finish_prompt's <|message_model|><|content_text|> prefill
        prefilled_out = "Red<|end_message|><|content_model_end_sampling|>"
        self.assertEqual(inkling_dialect.translate(prefilled_out), "Red")

    def test_quoted_tags_in_thinking_and_text_never_leak_or_truncate_6675(self):
        """#6675: quoting <|message_model|>, <|content_thinking|>, or <|end_message|> in thinking or
        visible text must neither leak hidden thinking into content nor strip/truncate the answer."""
        # 1. Exact #6675 scenario: thinking quotes both tags, then visible answer quotes both tags in a code span + DONE
        raw_stream = (
            "<|message_model|><|content_thinking|>"
            "User wants: 'The header token is `<|message_model|>` and ends with `<|end_message|>`.'. "
            "These look like special tokens. Repeating them is benign? Need double backticks. Final only."
            "<|end_message|>"
            "<|message_model|><|content_text|>"
            "``The header token is `<|message_model|>` and ends with `<|end_message|>`.``\nDONE"
            "<|end_message|><|content_model_end_sampling|>"
        )
        want = "``The header token is `<|message_model|>` and ends with `<|end_message|>`.``\nDONE"
        for step in (1, 3, 7, 1000):
            d = inkling_dialect.InklingDialect()
            out = "".join(d.feed(raw_stream[i:i + step]) for i in range(0, len(raw_stream), step)) + d.feed(final=True)
            self.assertEqual(out, want, f"step={step}")
            self.assertNotIn("These look like special tokens", out)

        # 2. Explaining serve/inkling_dialect.py: quoting <|message_model|> followed by <|content_thinking|> in visible text
        explain = (
            "<|message_model|><|content_text|>"
            "Text before `<|message_model|>` and `<|content_thinking|>` and text after."
            "<|end_message|><|content_model_end_sampling|>"
        )
        self.assertEqual(
            inkling_dialect.translate(explain),
            "Text before `<|message_model|>` and `<|content_thinking|>` and text after.",
        )

        # 3. Continuing directly from finish_prompt()'s <|message_model|><|content_text|> prefill
        prefilled = (
            "``The header token is `<|message_model|>` and ends with `<|end_message|>`.``\nDONE"
            "<|end_message|><|content_model_end_sampling|>"
        )
        self.assertEqual(inkling_dialect.translate(prefilled), want)

        # 4. Adjacent quoted + real <|end_message|> at the end of the answer
        adjacent = "<|message_model|><|content_text|>The tag is <|end_message|><|end_message|><|content_model_end_sampling|>"
        self.assertEqual(inkling_dialect.translate(adjacent), "The tag is <|end_message|>")

        # 5. Token-aware feed_token(): ordinary BPE tokens never trigger tags even for <|content_model_end_sampling|>
        d_tok = inkling_dialect.InklingDialect()
        toks = [
            ("<|message_model|>", True),
            ("<|content_thinking|>", True),
            ("Hidden <|end_message|> thinking <|content_text|> still hidden", False),
            ("<|end_message|>", True),
            ("<|message_model|>", True),
            ("<|content_text|>", True),
            ("Literal tags: <|message_model|>, <|content_thinking|>, <|end_message|>, <|content_model_end_sampling|>", False),
            ("<|end_message|>", True),
            ("<|content_model_end_sampling|>", True),
        ]
        chunks = []
        for piece, sp in toks:
            chunks.extend(d_tok.feed_token(piece, is_special=sp))
        chunks.extend(d_tok.feed_channels("", final=True))
        got_content = "".join(c for k, c in chunks if k == "content")
        self.assertEqual(
            got_content,
            "Literal tags: <|message_model|>, <|content_thinking|>, <|end_message|>, <|content_model_end_sampling|>",
        )

        # 6. Quoted <|message_model|> inside tool call JSON args does not split/corrupt the tool call
        payload = json.dumps({"name": "write_file", "args": {"content": "MSG = '<|message_model|>'"}})
        tool_raw = f"<|message_model|>write_file<|content_invoke_tool_json|>{payload}<|end_message|><|content_model_end_sampling|>"
        p = frontend.InklingOutputParser(tools=[{"name": "write_file", "parameters": {"type": "object"}}])
        evs = p.feed(tool_raw) + p.finish("stop")
        calls = [e.call for e in evs if e.kind == "tool_call" and e.call is not None]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments["content"], "MSG = '<|message_model|>'")

    def test_truncated_thinking_on_length_surfaces_reasoning_6682_and_bool_draft_off_6670(self):
        # #6682: if max_tokens runs out inside <|content_thinking|> with zero visible answer, surface reasoning
        p_cut = frontend.InklingOutputParser()
        evs = p_cut.feed("<|message_model|><|content_thinking|>Working through the algorithm step by step")
        evs += p_cut.finish("length")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "")
        self.assertIn("Working through the algorithm", "".join(e.text for e in evs if e.kind == "reasoning"))

        # Normal stop or completed answer still discards thinking (preserves nothink contract)
        p_ok = frontend.InklingOutputParser()
        evs_ok = p_ok.feed("<|message_model|><|content_thinking|>Internal.<|end_message|><|message_model|><|content_text|>42<|end_message|>")
        evs_ok += p_ok.finish("length")
        self.assertEqual("".join(e.text for e in evs_ok if e.kind == "content"), "42")
        self.assertEqual("".join(e.text for e in evs_ok if e.kind == "reasoning"), "")

        # #6670: boolean False / "off" / "none" all emit --draft off
        base = ["--native", "Inkling-2-Mini-SQ-20.strata", "--max-context", "32768"]
        for val in ("off", False, "none"):
            self.assertFalse(inkling_kind.wants_drafter({"drafting": val}), val)
            args = inkling_kind.engine_args({"arch": "inkling2_loop", "drafting": val, "args": list(base)}, announce=False)
            self.assertEqual(args[args.index("--draft") + 1], "off", val)

    def test_parser_state_follows_the_thinking_channel(self):
        """The server reads parser.state to tell the model's thinking from its answer (thinking_n, #123's budget
        wrap-up, #1053's close-retry). The dialect keeps the thinking to itself, so the state follows the channel
        the dialect is in: by feed() and by feed_token()."""
        p = frontend.InklingOutputParser()
        self.assertEqual(p.state, "content")
        for ch in "<|message_model|>":                # a tag that arrives a character at a time
            p.feed(ch)
        self.assertEqual(p.state, "content")           # the message header is not the thinking
        for ch in "<|content_thinking|>":
            p.feed(ch)
        self.assertEqual(p.state, "reasoning")
        self.assertEqual(p.feed("Working it out."), [])     # the thinking stays hidden
        self.assertEqual(p.state, "reasoning")
        evs = p.feed("<|end_message|><|message_model|><|content_text|>The answer is 4.")
        self.assertEqual(p.state, "content")
        self.assertEqual([(e.kind, e.text) for e in evs], [("content", "The answer is 4.")])

        q = frontend.InklingOutputParser()             # the token-aware path the server uses with the Inkling engine
        seen = []
        for piece, special in (("<|message_model|>", True), ("<|content_thinking|>", True), ("hidden", False),
                               ("<|end_message|>", True), ("<|message_model|>", True), ("<|content_text|>", True)):
            q.feed_token(piece, is_special=special)
            seen.append(q.state)
        self.assertEqual(seen, ["content", "reasoning", "reasoning", "reasoning", "content", "content"])

        r = frontend.InklingOutputParser()             # a reply cut inside its thinking is in the reasoning state
        r.feed("<|message_model|><|content_thinking|>still thinking")
        r.finish("length")
        self.assertEqual(r.state, "reasoning")


# A small Inkling-2-Mini template: the channel tags the dialect reads, and no thinking block of its own.
_TEMPLATE = (
    "<|message_system|><|content_text|>You are a helpful assistant.<|end_message|>"
    "{%- if tools -%}<|message_system|>tool_declare<|content_xml|>{{ tools | tojson }}<|end_message|>{%- endif -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>"
    "{%- elif m.role == 'assistant' -%}"
    "{%- for tc in m.tool_calls or [] -%}<|message_model|>{{ tc.function.name }}<|content_invoke_tool_json|>"
    "{{ tc.function.arguments | tojson }}<|end_message|>{%- endfor -%}"
    "{%- if m.content -%}<|message_model|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
    "{%- elif m.role == 'tool' -%}<|message_tool|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
    "{%- endfor -%}"
)


class _InklingTokenizer(ByteTokenizer):
    """ByteTokenizer with Inkling-2-Mini's channel tags for special tokens."""
    SPECIALS = ByteTokenizer.SPECIALS + [inkling_dialect.MSG_MODEL, inkling_dialect.CONTENT_THINK,
                                         inkling_dialect.CONTENT_TEXT, inkling_dialect.END_MSG,
                                         inkling_dialect.CONTENT_TOOL_JSON, inkling_dialect.END_SAMPLING,
                                         inkling_dialect.ENDOFTEXT]


class _ThinkingInklingEngine(MockEngine):
    """An Inkling-2-Mini that opens a real `<|content_thinking|>` turn (whatever opener the prompt ends with) and
    would think until max_tokens. A prompt that ends that turn the way the model does (#123's wrap-up, #1053's
    close-retry) gets ANSWER. Records every prompt it is given."""
    inkling_engine = True
    THOUGHT = "Let me think step by step about two plus two. " * 4          # 184 tokens (one per byte)
    ANSWER = "The answer is 4."

    def __init__(self, tok):
        super().__init__(tok, "x", max_context=8192)
        self.prompts = []

    def reply(self, closed: bool) -> str:
        return self.ANSWER if closed else f"{inkling_dialect.MSG_MODEL}{inkling_dialect.CONTENT_THINK}{self.THOUGHT}"

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        raw = self.tok.decode(ids)
        close = inkling_dialect.close_thinking()
        # closed: its own thinking is in the prompt and that turn is ended (a turn without tools opens with the
        # same tags, before anything was thought)
        closed = raw.endswith(close) and self.THOUGHT[:10] in raw[:-len(close)]
        end = self.tok.encode(inkling_dialect.END_SAMPLING, parse_special=True)
        for t in (self.tok.encode(self.reply(closed), parse_special=True) + end)[:max_new]:
            if cancel.is_set():
                return
            yield t


class _InklingCallingEngine(_ThinkingInklingEngine):
    """The same model, answering the closed thinking with a tool invocation (the empty text message in between is
    the model's own: a call opens a new message)."""
    CALL = (f"{inkling_dialect.END_MSG}{inkling_dialect.MSG_MODEL}bash{inkling_dialect.CONTENT_TOOL_JSON}"
            '{"name":"bash","args":{"cmd":"ls"}}' + inkling_dialect.END_MSG)

    def reply(self, closed: bool) -> str:
        return self.CALL if closed else super().reply(closed)


class InklingThinkingControls(unittest.TestCase):
    """#123 and #1053 on Inkling-2-Mini. On the model, a turn that needs planning opens a real `<|content_thinking|`
    turn (a tool turn does, after the empty one the prompt ends with), and the server reads parser.state to bound
    it. The thinking is closed with the model's own end of a thinking message: a Qwen `</think>` written inside
    that channel closes nothing there."""

    BASH = {"type": "function", "function": {"name": "bash", "description": "Runs a shell command.",
                                             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                            "required": ["cmd"]}}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tpl = Path(self.tmp.name) / "chat_template.jinja"
        tpl.write_text(_TEMPLATE, encoding="utf-8")
        self.tok = _InklingTokenizer()
        self.template = frontend.ChatTemplate(tpl)
        self.start(_ThinkingInklingEngine)

    def tearDown(self):
        self.stop()
        self.tmp.cleanup()

    def start(self, engine_cls):
        self.engine = engine_cls(self.tok)
        self.svc = Service(self.engine, self.tok, self.template)
        self.assertIs(self.svc.parser_cls, frontend.InklingOutputParser)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, **extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 400, **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_the_budget_ends_the_thinking_the_way_the_model_does(self):
        code, b = self.chat(tools=[self.BASH], reasoning_budget_tokens=20)
        self.assertEqual(code, 200, b)
        c = b["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), (_ThinkingInklingEngine.ANSWER, "stop"))
        first, second = self.engine.prompts
        opener = self.tok.encode(f"{inkling_dialect.MSG_MODEL}{inkling_dialect.CONTENT_THINK}", parse_special=True)
        close = self.tok.encode(inkling_dialect.close_thinking(), parse_special=True)
        # the prompt ends with an empty thinking turn; the model opened a real one
        self.assertTrue(self.tok.decode(first).endswith(f"{inkling_dialect.MSG_MODEL}{inkling_dialect.CONTENT_THINK}"
                                                        f"{inkling_dialect.END_MSG}"))
        self.assertEqual(second[:len(first)], first)                  # continued from the prefix the engine holds
        thought = second[len(first):-len(close)]
        self.assertEqual(thought[:len(opener)], opener)
        self.assertTrue(0 < len(thought) - len(opener) <= 20)           # the thinking the model was given before the wrap-up
        self.assertEqual(second[-len(close):], close)                  # the model's end of a thinking message
        self.assertNotIn("</think>", self.tok.decode(second))
        self.assertEqual(b["usage"]["completion_tokens"],
                         len(second) - len(first) + len(_ThinkingInklingEngine.ANSWER) + 1)
        self.assertFalse(c["message"].get("reasoning_content"))       # an answered reply keeps its thinking hidden

    def test_without_a_budget_the_thinking_is_left_to_the_model(self):
        code, b = self.chat(tools=[self.BASH], max_tokens=60)
        self.assertEqual(code, 200, b)
        c = b["choices"][0]
        self.assertEqual((c["finish_reason"], len(self.engine.prompts)), ("length", 1))
        self.assertFalse(c["message"].get("content"))
        self.assertTrue((c["message"].get("reasoning_content") or "").startswith("Let me think step by step"))   # #6682

    def test_the_thinking_tokens_are_counted(self):
        """The Responses API reports the thinking as reasoning tokens; the server's count reads parser.state."""
        self.svc.reasoning_budget_tokens = 20
        body = {"model": "m", "input": "2+2?", "max_output_tokens": 400}
        req = urllib.request.Request(self.base + "/v1/responses", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            b = json.loads(r.read().decode())
        generated = len(self.engine.prompts[1]) - len(self.engine.prompts[0])
        self.assertTrue(0 < b["usage"]["output_tokens_details"]["reasoning_tokens"] <= generated)

    def test_a_call_made_after_the_wrap_up_is_a_call(self):
        self.stop()
        self.start(_InklingCallingEngine)
        code, b = self.chat(tools=[self.BASH], reasoning_budget_tokens=20)
        self.assertEqual(code, 200, b)
        c = b["choices"][0]
        self.assertEqual((c["finish_reason"], len(self.engine.prompts)), ("tool_calls", 2))
        self.assertFalse(c["message"].get("content"))
        call = c["message"]["tool_calls"][0]["function"]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("bash", {"cmd": "ls"}))

    def test_a_reply_that_stops_inside_its_thinking_is_closed_once_and_answers(self):
        self.engine.THOUGHT = "Let me think: two plus two is four."
        self.svc.reasoning_close_retry = True
        code, b = self.chat()
        self.assertEqual(code, 200, b)
        c = b["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), (_ThinkingInklingEngine.ANSWER, "stop"))
        first, second = self.engine.prompts
        self.assertEqual(second[:len(first)], first)
        self.assertTrue(self.tok.decode(second).endswith(inkling_dialect.close_thinking()))
        self.assertNotIn("</think>", self.tok.decode(second))

    def test_without_the_flag_that_reply_stays_empty(self):
        self.engine.THOUGHT = "Let me think: two plus two is four."
        code, b = self.chat()
        self.assertEqual(code, 200, b)
        c = b["choices"][0]
        self.assertEqual((c["finish_reason"], len(self.engine.prompts)), ("stop", 1))
        self.assertFalse(c["message"].get("content"))


if __name__ == "__main__":
    unittest.main()
