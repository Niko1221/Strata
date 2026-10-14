"""serve/test_video.py - the OpenAI video part (video_url / input_video) reaches the engine as the clip's
frames, read by the encoder itself.

    python -m unittest serve.test_video -v

The criterion is the prompt's size: a video must add its image tokens to the prompt, as a picture does.
A 200 whose prompt_tokens stay at the text alone is a dropped video ("Video: not supported, and it fails
quietly", docs/DETAILS.md) - this file is the test that ends that.  The clip is one media place in the
prompt (the template's one video marker), and its marker expands to the encoder's total image tokens -
one record per merged frame group, in order (ENCV, tools/vision/strata_vision.cpp).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, media_of, openai_to_messages  # noqa: E402
from serve.server import (ByteTokenizer, IMAGE_PAD, MockEngine, Service, VISION_START,  # noqa: E402
                          Vision, serve, video_fps_of)
from serve.test_server import CTX, ImageMarkers  # noqa: E402

TPL = Path(__file__).with_name("chat_template.jinja")
GROUP = VISION_START + IMAGE_PAD                  # one media place in the rendered prompt (the start + its rows)


class FakeVision(ImageMarkers.FakeVision):
    """The image fake plus one encode per video: `n` image tokens per clip, whatever the encoder would give."""

    def __init__(self, d, n=9):
        super().__init__(d)
        self.n = n
        self.videos = []
        self.vrows = self.dir / "vid.sve"
        self.vrows.write_bytes(b"vids")

    def encode_video(self, source, fps=2.0, keep=()):
        self.videos.append((source, fps))
        return self.vrows, self.n


class VideoPart(unittest.TestCase):
    """The wire formats (pi's compat.videoPartFormat) -> a video item whose source the server can read."""

    def setUp(self):
        self.template = ChatTemplate(TPL)

    def test_video_url(self):
        for url in ("clip.mp4", "https://x.example/clip.mp4", "data:video/mp4;base64,AAAA"):
            with self.subTest(url=url):
                msgs, tools, kwargs = openai_to_messages({"messages": [
                    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": url}},
                                                 {"type": "text", "text": "what happens in the clip?"}]}]})
                self.assertEqual(msgs[0]["content"],
                                 [{"type": "video", "source": url},
                                  {"type": "text", "text": "what happens in the clip?"}])
                prompt = self.template.render(msgs, tools=tools, **kwargs)
                self.assertEqual(prompt.count(VISION_START), 1)
                self.assertIn("what happens in the clip?", prompt)

    def test_video_url_bare_string(self):
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "video_url", "video_url": "clip.mp4"}]}]})
        self.assertEqual(msgs[0]["content"], [{"type": "video", "source": "clip.mp4"}])

    def test_input_video(self):
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "input_video", "input_video": {"data": "AAAA"}},
                                         {"type": "text", "text": "what happens in the clip?"}]}]})
        self.assertEqual(msgs[0]["content"][0], {"type": "video", "source": "data:video/mp4;base64,AAAA"})
        prompt = self.template.render(msgs)
        self.assertEqual(prompt.count(VISION_START), 1)

    def test_input_video_mime_type(self):
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "input_video",
                                          "input_video": {"data": "AAAA", "mime_type": "video/webm"}}]}]})
        self.assertEqual(msgs[0]["content"], [{"type": "video", "source": "data:video/webm;base64,AAAA"}])

    def test_plain_text_is_unchanged(self):
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": "what happens?"}]})
        self.assertEqual(msgs[0]["content"], "what happens?")

    def test_tool_and_assistant_messages_take_video(self):
        for role in ("tool", "assistant"):
            with self.subTest(role=role):
                msgs, _, _ = openai_to_messages({"messages": [
                    {"role": "user", "content": "look"},
                    {"role": role, "content": [{"type": "video_url", "video_url": "clip.mp4"}]}]})
                self.assertEqual(msgs[1]["content"], [{"type": "video", "source": "clip.mp4"}])

    def test_media_of_lists_both_kinds_in_prompt_order(self):
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": "a.png"},
                                         {"type": "video_url", "video_url": "clip.mp4"}]}]})
        self.assertEqual(media_of(msgs), [("image", "a.png"), ("video", "clip.mp4")])


class VideoPrompt(unittest.TestCase):
    """The server: a video is one media place whose marker expands to the clip's image tokens - the
    prompt_tokens grow by them (the criterion), not by zero."""

    def setUp(self):
        self.tok = ByteTokenizer()

    def make(self, d, n=9):
        vision = FakeVision(d, n)
        svc = Service(MockEngine(self.tok, "ok", max_context=CTX), self.tok, ChatTemplate(TPL), vision=vision)
        return svc, vision

    def video_msgs(self):
        return openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "video_url", "video_url": "clip.mp4"},
                                         {"type": "text", "text": "what happens in the clip?"}]}]})

    def test_prompt_grows_by_the_video(self):
        with tempfile.TemporaryDirectory() as d:
            svc, vision = self.make(d)
            msgs, tools, kwargs = self.video_msgs()
            ids, _, _ = svc.prepare(msgs, tools, kwargs)
            path = svc.embeddings.path                      # before the next prepare: it resets the thread-local
            plain = openai_to_messages(
                {"messages": [{"role": "user", "content": "what happens in the clip?"}]})[0]
            ids_plain, _, _ = svc.prepare(plain, None, {})
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            self.assertEqual(vision.videos, [("clip.mp4", 2.0)])          # the model's own rate, 2 a second
            self.assertEqual(ids.count(pad) - ids_plain.count(pad), 9)     # the clip's image tokens
            self.assertEqual(ids_plain.count(pad), 0)
            self.assertEqual(self.tok.decode(ids).count(GROUP), 1)         # the video's one media place
            path.unlink(missing_ok=True)

    def test_the_clip_reaches_the_encoder_once(self):
        with tempfile.TemporaryDirectory() as d:
            svc, vision = self.make(d)
            msgs, tools, kwargs = openai_to_messages({"messages": [
                {"role": "user", "content": [{"type": "input_video", "input_video": {"data": "AAAA"}}]}]})
            ids, _, _ = svc.prepare(msgs, tools, kwargs)
            # the clip's source goes to the encoder as-is: its load() decodes the base64, one encode per clip
            self.assertEqual(vision.videos, [("data:video/mp4;base64,AAAA", 2.0)])
            self.assertEqual(ids.count(self.tok.encode(IMAGE_PAD, parse_special=True)[0]), 9)
            svc.embeddings.path.unlink(missing_ok=True)

    def test_image_and_video_share_their_places(self):
        with tempfile.TemporaryDirectory() as d:
            svc, vision = self.make(d, n=9)
            msgs, tools, kwargs = openai_to_messages({"messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": "a.png"},
                                             {"type": "video_url", "video_url": "clip.mp4"},
                                             {"type": "text", "text": "both?"}]}]})
            ids, _, _ = svc.prepare(msgs, tools, kwargs)
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            self.assertEqual(ids.count(pad), 9 + 3)                        # the clip's 9, the picture's 3
            self.assertEqual(self.tok.decode(ids).count(GROUP), 2)         # one media place each, in order
            self.assertEqual(vision.videos, [("clip.mp4", 2.0)])
            svc.embeddings.path.unlink(missing_ok=True)

    def test_without_vision_it_is_said_not_dropped(self):
        svc = Service(MockEngine(self.tok, "ok", max_context=CTX), self.tok, ChatTemplate(TPL))
        msgs, tools, kwargs = self.video_msgs()
        with self.assertRaises(ValueError) as e:
            svc.prepare(msgs, tools, kwargs)
        self.assertIn("vision encoder", str(e.exception))
        self.assertIn("video", str(e.exception))


class VideoEnc(unittest.TestCase):
    """Vision.encode_video against a fake encoder process: the ENCV line, the clip's file cleaned up, the
    result cached by the clip's hash."""

    FAKE = """
from pathlib import Path
import sys
sys.stdout.write("READY 2560\\n")
sys.stdout.flush()
for line in sys.stdin:
    words = line.split()
    if not words:
        continue
    if words[0] == "QUIT":
        break
    if words[0] in ("ENC", "ENCV"):
        out = Path(words[2])
        out.write_bytes(b"rec")
        with open("encv.log", "a") as f:
            f.write(line)
        sys.stdout.write("OK 9 12\\n")
    sys.stdout.flush()
"""

    def make(self, d):
        exe = Path(d) / "fake_encoder.py"
        exe.write_text(self.FAKE, encoding="utf-8")
        v = Vision({"exe": "unused", "mmproj": "mm.gguf", "model": "model.gguf"}, lazy=True)
        v.spawn = ([sys.executable, str(exe), "--mmproj", "mm.gguf", "--model", "model.gguf"], None, None)
        v._start()
        return v

    def test_the_encv_line_and_the_cleanup(self):
        import hashlib
        import threading
        with tempfile.TemporaryDirectory() as d:
            v = self.make(d)
            try:
                data = b"the clip's bytes"
                out, n = v.encode_video(data)
                key = hashlib.sha256(data).hexdigest()[:32]
                self.assertEqual(n, 9)
                self.assertEqual(Path(out).read_bytes(), b"rec")
                self.assertFalse((v.dir / f"{key}.vid").exists())         # the clip's file: gone after the encode
                self.assertIn(f"ENCV {key}.vid {key}.sve 2.0\n", (v.dir / "encv.log").read_text())
                again, again_n = v.encode_video(data)
                self.assertEqual((again, again_n), (out, n))              # cached by the clip's hash
                logs = (v.dir / "encv.log").read_text()
                self.assertEqual(logs.count("ENCV"), 1)
            finally:
                v.close()

    def test_the_encoder_says_err(self):
        with tempfile.TemporaryDirectory() as d:
            v = self.make(d)
            # a clip the fake encoder refuses: same ERR path as a picture's
            (Path(d) / "fake_encoder.py").write_text(
                "import sys\n"
                "sys.stdout.write('READY 2560\\n'); sys.stdout.flush()\n"
                "for line in sys.stdin:\n"
                "    if line.split() and line.split()[0] == 'ENCV':\n"
                "        sys.stdout.write('ERR no video decoder\\n')\n"
                "    sys.stdout.flush()\n", encoding="utf-8")
            v.close()
            v._start()
            try:
                with self.assertRaises(ValueError) as e:
                    v.encode_video(b"AAAA")
                self.assertIn("no video decoder", str(e.exception))
            finally:
                v.close()


class VideoModalities(unittest.TestCase):
    """/v1/models advertises video only while the encoder can actually read a clip (ffmpeg on its PATH)."""

    def setUp(self):
        self.tok = ByteTokenizer()

    def test_advertised_with_ffmpeg(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(self.tok, "ok", max_context=CTX), self.tok, ChatTemplate(TPL),
                          vision=FakeVision(d))
            httpd = serve(svc, port=0)
            try:
                base = f"http://127.0.0.1:{httpd.server_address[1]}"
                with mock.patch("shutil.which", return_value="/bin/ffmpeg"):
                    body = json.load(urllib.request.urlopen(base + "/v1/models"))
                self.assertEqual(body["data"][0]["architecture"]["input_modalities"],
                                 ["text", "image", "video"])
                with mock.patch("shutil.which", return_value=None):
                    body = json.load(urllib.request.urlopen(base + "/v1/models"))
                self.assertEqual(body["data"][0]["architecture"]["input_modalities"], ["text", "image"])
            finally:
                httpd.shutdown()
                httpd.server_close()


class VideoConfig(unittest.TestCase):
    """The vision section's video_fps: frames a second the encoder's mtmd samples a clip at (the model's own)."""

    def test_video_fps(self):
        self.assertEqual(video_fps_of(None), 2.0)                       # the model's default
        self.assertEqual(video_fps_of(4), 4.0)
        self.assertEqual(video_fps_of(0.5), 0.5)
        for bad in ("x", 0, -1, 65, True):
            with self.assertRaises(ValueError):
                video_fps_of(bad)


if __name__ == "__main__":
    unittest.main()
