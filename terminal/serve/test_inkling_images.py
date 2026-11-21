"""Inkling 2's pictures: strata-ink2 reads them itself (its ViT runs in the engine), as Hy4-Flash does.  The server writes
each picture to a file and names it with an image=<path> key on the GEN line; the template's one <|unused_200054|> per
picture is expanded to its image tokens by the engine.  Never the separate picture reader.

    python -m pytest serve/test_inkling_images.py
"""
import base64
import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, openai_to_messages, use_family
from serve.server import ByteTokenizer, Service, StrataEngine
from serve.test_vram import VramEngine

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00"
       b"\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82")
# the image branch of Inkling 2's chat_template.jinja (the model's own file is not in this repository)
TEMPLATE = ('{#- Inkling 2: <|content_thinking|> -#}{%- for m in messages -%}{%- if m.content is string -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>'
            '{%- else -%}{%- for part in m.content -%}{%- if part.type in ("image", "input_image", "image_url") -%}'
            '<|message_user|><|content_image|><|unused_200054|><|end_message|>{%- else -%}'
            '<|message_user|><|content_text|>{{ part.text }}<|end_message|>{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}')


def request():
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "What is this?"}]}]})
    return msgs


class InklingImages(unittest.TestCase):
    def service(self, images):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "t.jinja").write_text(TEMPLATE)
        tok = ByteTokenizer()
        engine = VramEngine(tok)
        engine.loaded = True
        if images is not None:
            engine.info["images"] = images
        svc = Service(engine, tok, ChatTemplate(tmp / "t.jinja"))
        svc.set_family("inkling2")
        self.addCleanup(use_family, None)
        self.addCleanup(svc.drop_embeddings)
        return svc

    def test_the_template_is_known_as_inkling(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "t.jinja").write_text(TEMPLATE)
        tok = ByteTokenizer()
        svc = Service(VramEngine(tok), tok, ChatTemplate(tmp / "t.jinja"))
        svc.set_family(None)
        self.addCleanup(use_family, None)
        self.assertEqual(svc.family, "inkling2")
        self.assertTrue(svc.engine_reads_images())

    def test_images_need_an_engine_that_says_images_1(self):
        self.assertTrue(self.service(1).has_vision())
        self.assertFalse(self.service(0).has_vision())
        self.assertFalse(self.service(None).has_vision())

    def test_a_picture_goes_to_the_engine_as_a_file(self):
        svc = self.service(1)
        svc.prepare(request(), None, {})
        paths = svc.embeddings.hy4
        self.assertEqual(len(paths), 1)
        self.assertEqual(Path(paths[0]).read_bytes(), PNG)          # a PNG goes as it came (the engine decodes it)
        self.assertIsNone(svc.embeddings.path)                      # no embeddings file: the separate reader is not used
        self.assertEqual(svc.render_prompt(request(), None, {}).count("<|unused_200054|>"), 1)   # one placeholder a picture
        svc.drop_embeddings()
        self.assertFalse(Path(paths[0]).exists())

    def test_the_gen_line_names_the_picture(self):
        keys = StrataEngine.sampling_keys({"strata_hy4_images": ["/tmp/a b%.png"]})
        self.assertIn(" image=/tmp/a%20b%25.png", keys)

    def test_an_engine_without_images_is_a_clear_400(self):
        svc = self.service(0)
        with self.assertRaises(ValueError) as cm:
            svc.prepare(request(), None, {})
        self.assertIn("Inkling 2 engine does not read images", str(cm.exception))

    def test_qwen_without_a_reader_is_unchanged(self):
        tok = ByteTokenizer()
        svc = Service(VramEngine(tok), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.set_family(None)
        self.assertFalse(svc.engine_reads_images())
        with self.assertRaises(ValueError) as cm:
            svc.prepare(request(), None, {})
        self.assertIn("started without the vision encoder", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
