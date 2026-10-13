"""serve/test_video.py - the OpenAI video part (video_url / input_video) reaches the engine as frames.

    python -m unittest serve.test_video -v

The criterion is the prompt's size: a video must add its frames' image tokens to the prompt, as a picture
does.  A 200 whose prompt_tokens stay at the text alone is a dropped video ("Video: not supported, and it
fails quietly", docs/DETAILS.md) - this file is the test that ends that.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, media_of, openai_to_messages  # noqa: E402
from serve.server import (ByteTokenizer, IMAGE_PAD, MockEngine, Service, VISION_START,  # noqa: E402
                          Vision, serve, video_frames_of, video_fps_of)
from serve.test_server import CTX, ImageMarkers  # noqa: E402

TPL = Path(__file__).with_name("chat_template.jinja")
GROUP = VISION_START + IMAGE_PAD                  # one media place in the rendered prompt (the start + its rows)


class FakeVision(ImageMarkers.FakeVision):
    """The image fake plus frame cutting: `n` fake frames per video, whatever ffmpeg would give."""

    def __init__(self, d, n=3):
        super().__init__(d)
        self.n = n
        self.cut = []

    def extract_frames(self, source, n=None, fps=2.0, cancel=None):
        self.cut.append((source, n, fps))
        return [b"frame%d" % i for i in range(self.n)]


class RealCutVision(ImageMarkers.FakeVision):
    """Cuts with the real ffmpeg: for the error paths, where the cutting is the subject."""

    def extract_frames(self, source, n=None, fps=2.0, cancel=None):
        return Vision.extract_frames(source, n, fps, cancel)


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
    """The server: a video is its frames as pictures - the prompt's media places are filled, so the
    prompt_tokens grow by the frames' image tokens (the criterion), not by zero."""

    def setUp(self):
        self.tok = ByteTokenizer()

    def make(self, d, n=3, real_cut=False):
        vision = RealCutVision(d) if real_cut else FakeVision(d, n)
        svc = Service(MockEngine(self.tok, "ok", max_context=CTX), self.tok, ChatTemplate(TPL), vision=vision)
        return svc, vision

    def video_msgs(self):
        return openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "video_url", "video_url": "clip.mp4"},
                                         {"type": "text", "text": "what happens in the clip?"}]}]})

    def test_prompt_grows_by_the_frames(self):
        with tempfile.TemporaryDirectory() as d:
            svc, vision = self.make(d)
            msgs, tools, kwargs = self.video_msgs()
            ids, _, _ = svc.prepare(msgs, tools, kwargs)
            path = svc.embeddings.path                      # before the next prepare: it resets the thread-local
            plain = openai_to_messages(
                {"messages": [{"role": "user", "content": "what happens in the clip?"}]})[0]
            ids_plain, _, _ = svc.prepare(plain, None, {})
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            self.assertEqual(vision.cut, [("clip.mp4", None, 2.0)])    # no pinned count: the model's own rate, 2 a second
            self.assertEqual(ids.count(pad) - ids_plain.count(pad), 9)  # 3 frames x the fake encoder's 3 rows
            self.assertEqual(ids_plain.count(pad), 0)
            self.assertEqual(self.tok.decode(ids).count(GROUP), 3)     # one media place per frame, in order
            path.unlink(missing_ok=True)

    def test_frames_are_ordinary_pictures_to_the_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            class Watching(FakeVision):
                def __init__(self, d):
                    super().__init__(d, 3)
                    self.encoded = []

                def encode(self, source):
                    self.encoded.append(source)
                    return super().encode(source)
            vision = Watching(d)
            svc = Service(MockEngine(self.tok, "ok", max_context=CTX), self.tok, ChatTemplate(TPL), vision=vision)
            msgs, tools, kwargs = self.video_msgs()
            ids, _, _ = svc.prepare(msgs, tools, kwargs)
            self.assertEqual(vision.encoded, [b"frame0", b"frame1", b"frame2"])
            self.assertEqual(ids.count(self.tok.encode(IMAGE_PAD, parse_special=True)[0]), 9)
            svc.embeddings.path.unlink(missing_ok=True)
    def test_without_vision_it_is_said_not_dropped(self):
        svc = Service(MockEngine(self.tok, "ok", max_context=CTX), self.tok, ChatTemplate(TPL))
        msgs, tools, kwargs = self.video_msgs()
        with self.assertRaises(ValueError) as e:
            svc.prepare(msgs, tools, kwargs)
        self.assertIn("vision encoder", str(e.exception))
        self.assertIn("video", str(e.exception))

    def test_without_ffmpeg_the_error_names_it(self):
        with tempfile.TemporaryDirectory() as d:
            svc, _ = self.make(d, real_cut=True)
            msgs, tools, kwargs = openai_to_messages({"messages": [
                {"role": "user", "content": [{"type": "video_url", "video_url": "data:video/mp4;base64,AAAA"}]}]})
            with mock.patch("shutil.which", return_value=None):
                with self.assertRaises(ValueError) as e:
                    svc.prepare(msgs, tools, kwargs)
            self.assertIn("ffmpeg", str(e.exception))


class VideoModalities(unittest.TestCase):
    """/v1/models advertises video only while the server can actually cut a clip."""

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


class VideoCut(unittest.TestCase):
    """Vision.extract_frames against a real ffmpeg: an even cut over the clip's length, PNG bytes back."""

    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which("ffmpeg")
        if not (cls.ffmpeg and shutil.which("ffprobe")):
            raise unittest.SkipTest("ffmpeg/ffprobe not on the PATH")

    def test_even_frames_as_png(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            subprocess.run([self.ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                            "-i", "testsrc=duration=2:size=32x32:rate=10", str(path)],
                           check=True, capture_output=True)
            frames = Vision.extract_frames(str(path), n=4)
            self.assertEqual(len(frames), 4)
            for f in frames:
                self.assertTrue(f.startswith(b"\x89PNG"))

    def test_short_clip_is_resampled_to_the_requested_cut(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "clip.mp4"
            subprocess.run([self.ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                            "-i", "testsrc=duration=1:size=32x32:rate=2", str(path)],
                           check=True, capture_output=True)
            frames = Vision.extract_frames(str(path), n=8)
            self.assertEqual(len(frames), 8)            # the fps cut resamples the 2 real frames to the asked 8
            self.assertTrue(all(f.startswith(b"\x89PNG") for f in frames))

    def test_unreadable_source_is_a_value_error(self):
        with self.assertRaises(ValueError):
            Vision.extract_frames("data:video/mp4;base64,AAAA", n=4)     # not a video, and it says so

    def clip(self, d, spec):
        path = Path(d) / "clip.mp4"
        subprocess.run([self.ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc=duration={spec}:size=32x32:rate=10", str(path)],
                       check=True, capture_output=True)
        return str(path)

    def test_fps_cut_is_two_a_second(self):
        with tempfile.TemporaryDirectory() as d:
            frames = Vision.extract_frames(self.clip(d, "2"))            # 2 s x 2 fps = 4 frames, the model's own rate
            self.assertEqual(len(frames), 4)

    def test_short_clip_gets_the_minimum(self):
        with tempfile.TemporaryDirectory() as d:
            frames = Vision.extract_frames(self.clip(d, "1"))            # 1 s x 2 = 2 -> the minimum of 4
            self.assertEqual(len(frames), 4)

    def test_odd_count_rounds_to_even(self):
        with tempfile.TemporaryDirectory() as d:
            frames = Vision.extract_frames(self.clip(d, "2.5"))          # 2.5 s x 2 = 5 -> an even count, like the official cut
            self.assertEqual(len(frames), 4)

    def test_long_clip_is_capped_at_64(self):
        with tempfile.TemporaryDirectory() as d:
            frames = Vision.extract_frames(self.clip(d, "60"))           # 60 s x 2 = 120 -> the cap
            self.assertEqual(len(frames), 64)

    def test_pinned_count_overrides_fps(self):
        with tempfile.TemporaryDirectory() as d:
            frames = Vision.extract_frames(self.clip(d, "2"), n=3)       # the config's video_frames: a fixed cut
            self.assertEqual(len(frames), 3)


class VideoConfig(unittest.TestCase):
    """The vision section's video keys: video_fps (the model's own rate) and the pinned video_frames."""

    def test_video_fps(self):
        self.assertEqual(video_fps_of(None), 2.0)                       # the model's default
        self.assertEqual(video_fps_of(4), 4.0)
        self.assertEqual(video_fps_of(0.5), 0.5)
        for bad in ("x", 0, -1, 65, True):
            with self.assertRaises(ValueError):
                video_fps_of(bad)

    def test_video_frames(self):
        self.assertIsNone(video_frames_of(None))                         # unset: the fps cut
        self.assertEqual(video_frames_of(8), 8)
        for bad in ("x", 0, 65, True):
            with self.assertRaises(ValueError):
                video_frames_of(bad)


if __name__ == "__main__":
    unittest.main()
