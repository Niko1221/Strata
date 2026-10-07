"""GLM API/template and vision protocol checks without a model, GPU or downloads."""
import io
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from serve.frontend import ChatTemplate, effort_kwargs
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, Vision, serve, vision_footprint


class GlmTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<|user|>", "<|observation|>", "<|begin_of_image|>", "<|image|>", "<|end_of_image|>"]


class GlmIntegration(unittest.TestCase):
    SRC = ("{% set effort = reasoning_effort if reasoning_effort in ['low', 'high'] else 'max' %}"
           "E={{ effort }}{% for m in messages %}{% if m.content is string %}{{ m.content }}"
           "{% else %}{% for p in m.content %}{% if p.type == 'image' %}"
           "<|begin_of_image|><|image|><|end_of_image|>{% else %}{{ p.text }}{% endif %}{% endfor %}"
           "{% endif %}{% endfor %}{% if add_generation_prompt %}<think>{% endif %}")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        template = self.path / "chat_template.jinja"
        template.write_text(self.SRC, encoding="utf-8")
        self.template = ChatTemplate(template)
        self.tok = GlmTokenizer()
        self.engine = MockEngine(self.tok, "ok", max_context=16384)

    def test_effort_levels(self):
        for effort, expected in [(None, "max"), ("low", "low"), ("medium", "high"), ("high", "max"), ("none", "low")]:
            prompt = self.template.render([], **effort_kwargs(effort))
            self.assertTrue(prompt.startswith("E=" + expected))
            self.assertTrue(prompt.endswith("<think></think>" if effort == "none" else "<think>"))

    def test_stop_ids_and_spelled_markers(self):
        svc = Service(self.engine, self.tok, self.template)
        for marker in ("<|user|>", "<|observation|>"):
            self.assertIn(self.tok.encode(marker, parse_special=True)[0], svc.stop_ids)
            self.engine.script = list(("answer" + marker + "next turn").encode())
            events = list(svc.run([], False, None, 200, {}, threading.Event()))
            self.assertEqual("".join(v.text for k, v in events if k == "event"), "answer")
            self.assertEqual(events[-1][1]["finish"], "stop")
        self.engine.script = list(b"a <|b|> and x<|use")
        events = list(svc.run([], False, None, 200, {}, threading.Event()))
        self.assertEqual("".join(v.text for k, v in events if k == "event"), "a <|b|> and x<|use")

    def test_image_markers_and_tool_image_borrow(self):
        vision = Vision({"exe": "unused", "model": "unused", "mmproj": "unused"}, lazy=True)
        self.addCleanup(vision.shutdown)
        rows = vision.dir / "image.sve"
        rows.write_bytes(b"rows")
        order = []
        vision.lend = lambda: order.append("lend")
        vision.reclaim = lambda: order.append("reclaim")
        vision._start = lambda: order.append("start")
        vision.close = lambda: order.append("close")
        vision.proc = mock.Mock()
        vision.encode = lambda source, keep=(): (rows, 3)
        svc = Service(self.engine, self.tok, self.template, vision=vision)
        messages = [{"role": "tool", "content": [{"type": "text", "text": "<|begin_of_image|><|image|>"},
                                                   {"type": "image", "source": "unused.png"}]}]
        ids, _, _ = svc.prepare(messages, None, {})
        self.assertEqual(ids.count(self.tok.encode("<|image|>", parse_special=True)[0]), 3)
        self.assertEqual(order[:4], ["lend", "start", "close", "reclaim"])
        self.assertEqual(order[4:8], ["lend", "start", "close", "reclaim"])
        self.assertTrue(Path(svc.embeddings.path).is_file())
        svc.drop_embeddings()

    def test_lending_reclaims_after_failed_start(self):
        v = Vision({"exe": "unused", "model": "unused", "mmproj": "unused"}, lazy=True)
        self.addCleanup(v.shutdown)
        v.lend, v.reclaim = mock.Mock(), mock.Mock()
        v._start = mock.Mock(side_effect=RuntimeError("encoder failed"))
        with self.assertRaisesRegex(ValueError, "encoder failed"), v.active():
            pass
        v.lend.assert_called_once()
        v.reclaim.assert_called_once()
        self.assertFalse(v.owed)

    def test_reclaim_failure_blocks_generation(self):
        v = mock.Mock(lend=object(), owed=True)
        v.settle.return_value = False
        svc = Service(self.engine, self.tok, self.template, vision=v)
        with self.assertRaisesRegex(ValueError, "reclaimed"):
            svc.ensure_loaded()

    def test_external_bind_requires_key_before_socket(self):
        svc = Service(self.engine, self.tok, self.template)
        with mock.patch("serve.server.Server") as socket:
            with self.assertRaisesRegex(ValueError, "api-key"):
                serve(svc, host="0.0.0.0")
            socket.assert_not_called()

    def test_control_and_stat_protocol(self):
        e = StrataEngine("unused", ["--glm-pack", "glm-pack"], lazy=True)
        self.assertEqual(e.model_path, "glm-pack")
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        e.lines.put("VLENT 123\n")
        self.assertEqual(e.command("VLEND", "VLENT"), "VLENT 123")
        self.assertEqual(e.proc.stdin.getvalue(), "VLEND\n")
        e.lines.put("ERR still allocated\n")
        with self.assertRaisesRegex(ValueError, "still allocated"):
            e.command("VRECLAIM", "VRECLAIMED")
        e.lines.put(None)
        with self.assertRaises(EngineDied):
            e.command("VLEND", "VLENT")
        for i in range(125):
            e._parse_stat(f"STAT tok_s={i} ram_fetch=1.25 disk=0 promo=2 vram_hit=.75")
        self.assertEqual(len(e.stat_history), 120)
        self.assertEqual(e.stat["tok_s"], 124)
        svc = Service(self.engine, self.tok, self.template)
        self.engine.stat, self.engine.stat_history = e.stat, e.stat_history
        self.assertEqual(svc.metrics()["tiers"]["history"]["ram_fetch"][-1], 1.25)

    def test_footprint_cache_and_bad_probe(self):
        projector = self.path / "mmproj.gguf"
        projector.write_bytes(b"metadata")
        cfg = {"exe": "unused", "mmproj": str(projector), "model": "unused", "max_tokens": 4096, "no_flash_attn": True}
        cache = self.path / "memory.json"
        cache.write_text("[]", encoding="utf-8")
        with mock.patch("serve.server.subprocess.run", return_value=mock.Mock(returncode=0, stdout="MEM 100 200 1000\n")) as probe:
            self.assertEqual(vision_footprint(cfg, {}, cache, 0), (4096, 300 + (256 << 20)))
            self.assertIn("--no-flash-attn", probe.call_args.args[0])
            vision_footprint(cfg, {}, cache, 0)
            probe.assert_called_once()
        cache.unlink()
        with mock.patch("serve.server.subprocess.run", return_value=mock.Mock(returncode=1, stdout="MEM 100 200 1000\n")):
            self.assertIsNone(vision_footprint(cfg, {}, cache, 0))


if __name__ == "__main__":
    unittest.main()
