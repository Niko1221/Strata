"""Chat history normalization before rendering the model's prompt."""
import copy
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service


class EmptyAssistantHistory(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_openai_empty_replies_are_not_rendered(self):
        for content in (None, "", " \n\t", [], [{"type": "text", "text": ""}]):
            for preserve in (True, False):
                with self.subTest(content=content, preserve_thinking=preserve):
                    history = []
                    for _ in range(3):
                        history.extend([{"role": "user", "content": "haz un ls"},
                                        {"role": "assistant", "content": content,
                                         "reasoning_content": "Let me run ls."}])
                    history.append({"role": "user", "content": "haz un ls"})
                    messages, tools, kwargs = openai_to_messages({"messages": history})
                    kwargs["preserve_thinking"] = preserve
                    clean = [m for m in messages if m["role"] == "user"]
                    self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                                     self.service.encode_prompt(clean, tools, kwargs))

    def test_anthropic_thinking_only_replies_are_not_rendered(self):
        messages, tools, kwargs = anthropic_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me run ls."}]},
            {"role": "user", "content": "haz un ls"},
        ]})
        self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                         self.service.encode_prompt([messages[0], messages[2]], tools, kwargs))

    def test_text_and_tool_calls_are_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "text", "text": "Running ls."}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "run", "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "content": "file.txt"},
            {"role": "user", "content": "thanks"},
        ]})
        before = copy.deepcopy(messages)
        prompt = self.template.render(messages, tools=tools, **kwargs)
        self.assertIn("Running ls.", prompt)
        self.assertIn("<function=run>", prompt)
        self.assertIn("file.txt", prompt)
        self.assertEqual(messages, before)

    def test_image_only_assistant_turn_is_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "describe the image"},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "thanks"},
        ]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(messages, tools=tools, **kwargs))

    def test_final_empty_assistant_turn_is_kept(self):
        prompt = self.template.render([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": ""}], add_generation_prompt=False)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_empty_user_and_tool_turns_are_kept(self):
        prompt = self.template.render([{"role": "user", "content": ""},
                                       {"role": "tool", "content": ""},
                                       {"role": "user", "content": "hi"}])
        self.assertEqual(prompt.count("<|im_start|>user\n"), 3)
        self.assertIn("<tool_response>\n\n</tool_response>", prompt)


class UnreadContentParts(unittest.TestCase):
    """A content part the server does not read is refused naming it and where it sits, as /v1/responses does; chat
    and messages used to drop it without a word, and the model answered about a file it had never seen."""

    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_chat_file_and_audio_parts_are_left_out(self):
        parts = ({"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AA"}},
                 {"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}})
        plain = [{"role": "user", "content": "summarize"}, {"role": "user", "content": "this one"}]
        for part in parts:
            with self.subTest(kind=part["type"]):
                got = openai_to_messages({"messages": [
                    {"role": "user", "content": "summarize"},
                    {"role": "user", "content": [{"type": "text", "text": "this one"}, part]}]})
                want = openai_to_messages({"messages": [
                    {"role": "user", "content": "summarize"},
                    {"role": "user", "content": [{"type": "text", "text": "this one"}]}]})
                self.assertEqual(got, want)

    def test_messages_blocks_real_clients_send_are_left_out(self):
        # tool_reference (Claude Code's tool search), documents, search results, server tool blocks, signed thinking
        extra = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA"}},
                 {"type": "search_result", "source": "s", "title": "t", "content": [{"type": "text", "text": "x"}]},
                 {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
                 {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                 {"type": "container_upload", "file_id": "f"}]
        for block in extra:
            with self.subTest(kind=block["type"]):
                got = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}, block]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}, block]}]})
                want = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}]}]})
                self.assertEqual(got, want)

    def test_messages_tool_reference_in_a_tool_result_is_left_out(self):
        def run(extra):
            return anthropic_to_messages({"messages": [
                {"role": "user", "content": "find a tool"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "found"}] + extra}]}]})
        self.assertEqual(run([{"type": "tool_reference", "tool_name": "mcp__x"}]), run([]))

    def test_a_system_prompt_with_an_image_part_is_read_as_text_as_before(self):
        got = openai_to_messages({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "be brief"},
                                           {"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "hi"}]})
        self.assertEqual(got[0][0]["content"], "be brief")

    def test_messages_redacted_thinking_is_dropped(self):
        msgs = {"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "ZW50cnk="},
                                              {"type": "text", "text": "Running ls."}]},
            {"role": "user", "content": "thanks"}]}
        without = copy.deepcopy(msgs)
        del without["messages"][1]["content"][0]
        with_block = anthropic_to_messages(msgs)
        without_block = anthropic_to_messages(without)
        self.assertEqual(with_block[0], without_block[0])
        self.assertEqual(self.service.encode_prompt(with_block[0], with_block[1], with_block[2]),
                         self.service.encode_prompt(without_block[0], without_block[1], without_block[2]))

    def test_chat_refusal_part_is_read_as_its_text(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "I cannot do that."}]},
            {"role": "user", "content": "ok"}]})
        self.assertEqual(messages[1]["content"], "I cannot do that.")
        self.assertIn("I cannot do that.", self.template.render(messages, tools=tools, **kwargs))

    def test_text_and_image_requests_render_as_before(self):
        parts = [{"role": "user", "content": [{"type": "text", "text": "haz un ls"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "file.txt"}]}]
        plain = [{"role": "user", "content": "haz un ls"}, {"role": "assistant", "content": "file.txt"}]
        for convert in (openai_to_messages, anthropic_to_messages):
            with self.subTest(convert=convert.__name__):
                self.assertEqual(self.service.encode_prompt(*convert({"messages": parts})),
                                 self.service.encode_prompt(*convert({"messages": plain})))
        chat = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image_url", "image_url": "image.png"}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(chat[0], tools=chat[1], **chat[2]))
        msgs = anthropic_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image", "source": {"type": "url", "url": "image.png"}}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(msgs[0], tools=msgs[1], **msgs[2]))


if __name__ == "__main__":
    unittest.main()


class EmptyTurnsSwitch(unittest.TestCase):
    def test_keep_empty_turns_restores_the_old_prompt(self):
        import os
        from unittest import mock
        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                {"role": "user", "content": "b"}]
        clean = template.render([msgs[0], msgs[2]])
        self.assertEqual(template.render(msgs), clean)
        with mock.patch.dict(os.environ, {"STRATA_KEEP_EMPTY_TURNS": "1"}):
            self.assertNotEqual(template.render(msgs), clean)


class ForcedCallOpening(unittest.TestCase):
    def test_required_with_one_tool_names_it(self):
        from serve.frontend import forced_call
        one, two = [{"name": "a"}], [{"name": "a"}, {"name": "b"}]
        self.assertTrue(forced_call("required", one).endswith("<function=a>\n"))
        self.assertTrue(forced_call("required", two).endswith("<function="))
        self.assertIsNone(forced_call("auto", one))
