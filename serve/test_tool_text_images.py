"""A tool that hands its picture back as base64 in its TEXT: the picture goes to the encoder, not to the context.

Chatbox and the other clients built on the AI SDK run MCP servers themselves, and for a chat-completions upstream
the SDK JSON-stringifies a `content` tool output - so a picture an MCP tool returned arrives as one long base64 run
inside the tool message (a 2 MB photo: 1,491,022 tokens, a prompt no context holds and the conversation dies).
serve/server.py's `_lift_pictures_from_tool_text` takes such a picture out of the text and gives it to the vision
encoder, the way serve/mcp.py does for a picture on Strata's own MCP path.

    python -m unittest serve.test_tool_text_images -v
"""
import base64
import json
import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]

# The lift checks the magic bytes rather than decoding the whole picture (the encoder itself refuses a broken one,
# and `_note_unreadable_tool_images` turns that into a note), so a picture here is its header plus enough body to
# pass the 1024-character floor - a real PNG of that size is a 32x32 one.
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1500).decode()
JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x01" * 1400).decode()
NOT_A_PICTURE = base64.b64encode(b"a zip, not a picture" * 90).decode()
# A tool's own text that starts with a bitmap's 2-byte magic: the case a reviewer of #2343 hit, where the log line
# was taken away from the model and handed to the encoder as a 3,016-byte "bitmap".
BMW = base64.b64encode(b"BMW service log " + b"x" * 3000).decode()
BMP = base64.b64encode(b"BM" + b"\x00" * 1500).decode()
GIF = base64.b64encode(b"GIF89a" + b"\x00" * 1500).decode()


class FakeVision:
    """A stand-in encoder: every picture "encodes" to three rows in one file, and it remembers what it was given."""

    def __init__(self, d):
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")
        self.seen: list = []

    def encode(self, source):
        self.seen.append(source)
        return self.rows, 3


def conversation(result):
    """One tool call and its result, as a client that ran the tool itself sends them back."""
    return [
        {"role": "user", "content": "what is in my photo?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "img__read_image", "arguments": {"path": "/home/ds/x.jpg"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": result},
    ]


class ToolTextPictures(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "ok", max_context=131072)
        self.tmp = tempfile.TemporaryDirectory()
        self.vision = FakeVision(self.tmp.name)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                           vision=self.vision)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, messages):
        ids, _, _ = self.svc.prepare(messages, None, {}, 64)
        return ids, self.tok.decode(ids)

    @staticmethod
    def chatbox_text(*blobs, mime="image/png"):
        """The AI SDK's `content` tool output, JSON-stringified for a chat-completions upstream."""
        blocks = [{"type": "text", "text": "JPEG 4320x2432, resized to 1200x676, converted to png"}]
        blocks += [{"type": "image-data", "data": b, "mediaType": mime} for b in blobs]
        return json.dumps({"content": blocks}, ensure_ascii=False)

    def test_a_picture_in_a_tools_text_becomes_a_picture(self):
        ids, prompt = self.prepare(conversation(self.chatbox_text(PNG)))
        self.assertIn("JPEG 4320x2432, resized to 1200x676", prompt)
        self.assertIn("[image (image/png, 1,508 bytes)]", prompt)
        self.assertIn("<|vision_start|><|image_pad|>", prompt)
        self.assertNotIn(PNG, prompt)                                   # the base64 is out of the text
        self.assertEqual(self.vision.seen, [f"data:image/png;base64,{PNG}"] * 2)   # checked, then encoded
        self.assertLess(len(ids), 1200)                                 # not the ~2,100 the base64 would cost

    def test_a_data_url_in_a_tools_text_becomes_a_picture(self):
        _, prompt = self.prepare(conversation(f"read /home/ds/x.jpg\nhere: data:image/jpeg;base64,{JPEG} done"))
        self.assertIn("[image (image/jpeg, 1,404 bytes)]", prompt)
        self.assertIn("<|vision_start|><|image_pad|>", prompt)
        self.assertNotIn(JPEG, prompt)
        self.assertEqual(self.vision.seen, [f"data:image/jpeg;base64,{JPEG}"] * 2)

    def test_text_that_is_not_a_picture_stays_text(self):
        for text in (self.chatbox_text(NOT_A_PICTURE),                  # long base64, but no image magic
                     "result: " + BMW,                                  # bytes beginning "BMW ": not a bitmap
                     "sha256 " + "a" * 4000):                            # a long run of letters, not a picture
            messages = conversation(text)
            _, prompt = self.prepare(messages)
            with self.subTest(text=text[:40]):
                self.assertEqual(messages[2]["content"], text)          # left exactly as the client sent it
                self.assertNotIn("<|vision_start|>", prompt)
                self.assertEqual(self.vision.seen, [])

    def test_a_bitmap_or_gif_in_a_tools_text_stays_text(self):
        """Only a PNG's or a JPEG's magic is read out of a tool's text: "BM" is 2 bytes, and a GIF's 6 are not worth
        the guessing either (#2343).  A BMP or GIF sent as a real image part still reaches the encoder -
        `Vision.normalize` passes both through to it, as it always did."""
        for blob in (BMP, GIF):
            text = self.chatbox_text(blob)
            messages = conversation(text)
            _, prompt = self.prepare(messages)
            with self.subTest(magic=blob[:8]):
                self.assertEqual(messages[2]["content"], text)          # left exactly as the client sent it
                self.assertNotIn("<|vision_start|>", prompt)
                self.assertEqual(self.vision.seen, [])

    def test_a_picture_a_user_sent_as_text_stays_text(self):
        """Only a tool's text is read this way: a user who pastes base64 means the characters."""
        messages = [{"role": "user", "content": f"data:image/png;base64,{PNG}"}]
        _, prompt = self.prepare(messages)
        self.assertIn(PNG, prompt)
        self.assertNotIn("<|vision_start|>", prompt)

    def test_the_same_caps_as_one_mcp_result(self):
        """Nine pictures in one result: eight go to the model, the ninth is named and left out of the text."""
        _, prompt = self.prepare(conversation(self.chatbox_text(*[PNG] * 9)))
        self.assertEqual(self.vision.seen, [f"data:image/png;base64,{PNG}"] * 16)   # each checked, then encoded
        self.assertIn("[image (image/png, 1,508 bytes): is past the 8 pictures one result may carry, "
                      "not shown to the model]", prompt)
        self.assertNotIn(PNG, prompt)

    def test_without_an_encoder_the_base64_is_dropped_rather_than_kept(self):
        """No encoder to read it: the picture still costs nothing, and the note in its place says why."""
        self.svc.vision = None
        _, prompt = self.prepare(conversation(self.chatbox_text(PNG)))
        self.assertIn("[image omitted: this server has no image encoder]", prompt)
        self.assertNotIn(PNG, prompt)
        self.assertNotIn("<|vision_start|>", prompt)

    def test_a_second_request_of_the_same_conversation_is_unchanged(self):
        """Clients resend the whole history every turn: the lift runs again on every one of those requests."""
        for _ in range(2):
            _, prompt = self.prepare(conversation(self.chatbox_text(PNG)))
            self.assertIn("<|vision_start|><|image_pad|>", prompt)
        self.assertEqual(self.vision.seen, [f"data:image/png;base64,{PNG}"] * 4)    # twice: check, encode, twice


if __name__ == "__main__":
    unittest.main()
