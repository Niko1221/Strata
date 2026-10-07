"""Video normalization/template tests; no decoder, model, or GPU."""
from pathlib import Path
import unittest

from serve.frontend import (ChatTemplate, anthropic_to_messages, images_of, media_of, openai_to_messages)
from serve.responses import ResponsesError, input_messages

ROOT = Path(__file__).resolve().parents[1]


class VideoFrontend(unittest.TestCase):
    def test_openai_order_and_legacy_image_accessor(self):
        parts = [{"type": "text", "text": "before"},
                 {"type": "image_url", "image_url": {"url": "picture.png"}},
                 {"type": "video_url", "video_url": {"url": "clip.mp4"}},
                 {"type": "text", "text": "after"},
                 {"type": "video", "video": "second.mkv"}]
        messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": parts}]})
        self.assertEqual(media_of(messages), [("image", "picture.png"), ("video", "clip.mp4"), ("video", "second.mkv")])
        self.assertEqual(images_of(messages), ["picture.png"])
        rendered = ChatTemplate(ROOT / "serve/chat_template.jinja").render(messages)
        self.assertLess(rendered.index("before"), rendered.index("<|image_pad|>"))
        self.assertLess(rendered.index("<|image_pad|>"), rendered.index("<|video_pad|>"))
        self.assertLess(rendered.index("<|video_pad|>"), rendered.index("after"))
        self.assertEqual(rendered.count("<|video_pad|>"), 2)

    def test_anthropic_url_and_base64_extensions(self):
        parts = [{"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "eA=="}},
                 {"type": "video", "source": {"type": "url", "url": "https://example.test/clip.mkv"}}]
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": parts}]})
        self.assertEqual(media_of(messages), [("video", "data:video/mp4;base64,eA=="),
                                             ("video", "https://example.test/clip.mkv")])

    def test_responses_preserves_video_and_order(self):
        messages = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "look"}, {"type": "input_video", "video_url": "clip.mp4"},
            {"type": "input_image", "image_url": "still.png"}]}]})
        self.assertEqual(media_of(messages), [("video", "clip.mp4"), ("image", "still.png")])

    def test_non_user_roles_fail_even_after_late_system_conversion(self):
        for role in ("system", "developer", "assistant", "tool"):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    openai_to_messages({"messages": [{"role": "user", "content": "first"},
                        {"role": role, "content": [{"type": "video_url", "video_url": "clip.mp4"}]}]})
                with self.assertRaises(ResponsesError):
                    input_messages({"input": [{"role": role, "content": [{"type": "input_video", "video_url": "clip.mp4"}]}]})

    def test_nested_tool_video_is_not_dropped(self):
        with self.assertRaises(ValueError):
            anthropic_to_messages({"messages": [{"role": "user", "content": [{"type": "tool_result",
                "tool_use_id": "x", "content": [{"type": "video", "source": {"type": "url", "url": "clip.mp4"}}]}]}]})

    def test_bad_forms_fail_instead_of_disappearing(self):
        bad = [{"type": "video_url", "video_url": []}, {"type": "video", "video": True},
               {"type": "video_frames", "frames": ["one.jpg"]}, {"type": "video", "video": ""},
               {"type": "video", "video": "clip.mp4", "fps": 4},
               {"type": "video_url", "video_url": {"url": "clip.mp4", "nframes": 8}},
               {"type": "video", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}},
               {"type": "video", "source": {"type": "camera", "url": "x"}}]
        for part in bad:
            with self.subTest(part=part):
                with self.assertRaises(ValueError):
                    openai_to_messages({"messages": [{"role": "user", "content": [part]}]})
        with self.assertRaises(ResponsesError):
            input_messages({"input": [{"role": "user", "content": [{"type": "input_video", "video_url": []}]}]})

    def test_video_only_user_message_survives_rendering(self):
        messages, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": "clip.mp4"}]}]})
        rendered = ChatTemplate(ROOT / "serve/chat_template.jinja").render(messages)
        self.assertIn("<|vision_start|><|video_pad|><|vision_end|>", rendered)


if __name__ == "__main__":
    unittest.main()
