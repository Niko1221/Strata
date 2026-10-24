"""The elastic encoder stays down for pictures it has already encoded: the hash cache and its .sve files survive an
unload (Vision.unload keeps them), so a conversation that re-sends its pictures on every turn - which is what most
clients do - must not pay the elastic start (seconds + the VRAM the expert cache gives up) on a cache hit.  A new
picture, alone or alongside cached ones, still wakes it."""
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]


class CachingEncoder:
    """An encoder process stand-in with Vision's cache semantics: every picture encodes to three image tokens and
    keeps its (file, count) in .cache, which - as on a real unload - survives restarts and unloads."""

    def __init__(self, d):
        self.elastic, self.elastic_mib, self.elastic_idle_s = True, 2048, 60.0
        self.up, self.starts, self.unloads, self.encodes = False, 0, 0, 0
        self.dir = Path(d)
        self.cache = {}

    def alive(self):
        return self.up

    def restart(self):
        self.up, self.starts = True, self.starts + 1

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1

    def key_of(self, source):
        return str(source)

    def cached(self, source):
        return self.cache.get(self.key_of(source))

    def encode(self, source, keep=()):
        assert self.up, "a picture reached an encoder that was not started"
        self.encodes += 1
        row = self.dir / f"{source}.sve"
        row.write_bytes(b"rows")
        self.cache[self.key_of(source)] = (row, 3)
        return self.cache[self.key_of(source)]

    def encode_all(self, sources, cancel=None):
        return [self.encode(s) for s in sources]


class CacheWake(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = VramEngine(self.tok)
        self.engine.loaded = True
        self.vision = CachingEncoder(tempfile.mkdtemp())
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)
        self.addCleanup(self.drop_embeddings)

    def drop_embeddings(self):
        path = getattr(getattr(self.svc, "embeddings", None), "path", None)
        if path is not None:
            path.unlink(missing_ok=True)

    def send(self, *sources):
        content = [{"type": "text", "text": "what is it"}] + [{"type": "image", "source": s} for s in sources]
        return self.svc.prepare([{"role": "user", "content": content}], None, {})

    def test_a_re_sent_picture_does_not_wake_the_idle_encoder(self):
        ids1 = self.send("a.png")
        self.assertEqual(self.vision.starts, 1)                          # a new picture: the normal start
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")           # the 60 s idle passed
        ids2 = self.send("a.png")                                        # the next turn re-sends the history
        self.assertEqual(self.vision.starts, 1)                          # already encoded: no start
        self.assertFalse(self.vision.up)
        self.assertFalse(self.svc.vision_held)
        self.assertEqual(self.engine.calls, [2048, 0])                   # and no fresh share request either
        self.assertEqual(ids2, ids1)                                     # the prompt is built the same, from the cache

    def test_a_new_picture_alongside_cached_ones_still_wakes_it(self):
        self.send("a.png")
        self.svc.last_image_at = 0
        self.svc._vision_idle_down()
        self.send("a.png", "b.png")                                      # one hit, one miss
        self.assertEqual(self.vision.starts, 2)
        self.assertTrue(self.vision.up)

    def test_a_cache_hit_while_the_encoder_runs_changes_nothing(self):
        self.send("a.png")
        self.send("a.png")                                               # the encoder is up: nothing to reserve, start or encode
        self.assertEqual(self.vision.starts, 1)
        self.assertEqual(self.engine.calls, [2048])

    def test_the_cache_lookup_needs_no_encoder_process(self):
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True}, lazy=True)
        self.assertFalse(v.alive())                                      # nothing started
        v._ensure_dir()                                                  # the folder exists (cached() guards on it, #3370)
        img = Image.new("RGB", (8, 6), (9, 9, 9))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        self.assertIsNone(v.cached(png))                                 # cold: nobody encoded it
        v.cache[v.key_of(png)] = (Path("x.sve"), 42)                     # what encode() would have stored
        self.assertEqual(v.cached(png), (Path("x.sve"), 42))             # warm: the file alone answers, process down
        self.assertIsNone(v.cached(b"not an image"))                     # unreadable: None - encode() gives that error

    def test_a_cache_hit_keeps_the_picture_from_aging_out(self):
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True}, lazy=True)
        v._ensure_dir()                                                  # the folder exists (cached() guards on it, #3370)
        pngs = []
        for shade in range(3):                                           # three distinct pictures
            img = Image.new("RGB", (8, 6), (shade, shade, shade))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pngs.append(buf.getvalue())
        for png in pngs:                                                 # encoded oldest to newest, as encode() would cache
            v.cache[v.key_of(png)] = (Path(f"{len(v.cache)}.sve"), 3)
        self.assertEqual(v.cached(pngs[0]), (Path("0.sve"), 3))          # the oldest is asked for again (re-sent history)
        self.assertEqual(list(v.cache)[-1], v.key_of(pngs[0]))           # the lookup marks it most recently used, so
                                                                         # the 64-entry eviction (encode, oldest first)
                                                                         # takes a picture nobody re-sent instead

    def test_a_temp_cleaner_removal_makes_the_all_cached_turn_re_encode(self):
        # #3370 made encode() re-create a work folder a temp cleaner removed; the all-cached skip reads the same
        # dict without ever entering encode(), so cached() must carry the same guard - dead .sve paths may not keep
        # answering while the turn that would re-encode is skipped (prepare() would die on them every turn).
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True}, lazy=True)
        img = Image.new("RGB", (8, 6), (9, 9, 9))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        v._ensure_dir()
        v.cache[v.key_of(png)] = (v.dir / "x.sve", 42)
        (v.dir / "x.sve").write_bytes(b"rows")
        self.assertEqual(v.cached(png), (v.dir / "x.sve", 42))           # folder present: the file alone answers
        shutil.rmtree(v.dir)                                             # Storage Sense removed it during the unload
        self.assertIsNone(v.cached(png))                                 # the dead entry is gone with the folder...
        self.assertEqual(v.cache, {})                                    # ...and the dict is cleared, so prepare()
                                                                         # sees a miss and takes the encode path that
                                                                         # re-creates the folder and re-encodes

    def test_a_non_elastic_turn_takes_todays_path_even_all_cached(self):
        self.vision.elastic = False                                      # a plain always-on encoder
        self.vision.up = True                                            # it started with the server, not with a picture
        self.send("a.png")                                               # encode_all ran (a cache hit inside encode())
        self.assertEqual(self.vision.encodes, 1)
        self.send("a.png")                                               # the next turn: today's path unchanged, hits and all
        self.assertEqual(self.vision.encodes, 2)
        self.assertEqual(self.engine.calls, [])                          # no make-room request either: the elastic machinery stays out


if __name__ == "__main__":
    unittest.main()
