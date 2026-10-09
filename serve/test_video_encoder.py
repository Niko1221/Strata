"""Synthetic-payload cache/negotiation tests, never model correctness evidence."""
from dataclasses import replace
from pathlib import Path
import re
import tempfile
import unittest

from serve.media import MediaBundle, MediaKind, VisualSpan, read_bundle, write_bundle
from serve.video import ClipInfo, VideoError, VideoLimitError, VideoRequestBudget
from serve.video_encoder import VideoEncoder

VS, VE, IMAGE, VIDEO = 248053, 248054, 248056, 248057


class Tokenizer:
    """Explicit synthetic lexer: structural tokens are real ids, text is bytes."""
    controls = {"<|vision_start|>": VS, "<|vision_end|>": VE, "<|image_pad|>": IMAGE, "<|video_pad|>": VIDEO}
    def encode(self, text, parse_special=True):
        result=[]
        for part in re.split(r"(<\|[^|]+\|>)",text):
            result.extend([self.controls[part]] if part in self.controls and parse_special else part.encode())
        return result


def clip_info():
    return ClipInfo(5,2.,128,64,2.5,(0,1,2,3,4),(0.,0.5,1.0,1.5,2.0),128,64)


def synthetic_video(info, tok):
    """Fake embeddings: all-zero FP32, NOT a projector forward pass."""
    rows=info.resized_width//32*(info.resized_height//32)
    nx=info.resized_width//32
    tokens,spans=[],[]
    for i in range(0,len(info.times),2):
        a,b=info.times[i],info.times[min(i+1,len(info.times)-1)]
        seconds=(a+b)/2
        tokens+=tok.encode(f"<{seconds:.1f} seconds><|vision_start|>",parse_special=True)
        spans.append(VisualSpan(len(tokens),VIDEO,MediaKind.VIDEO,max(nx,info.resized_height//32),
                               tuple((0,j//nx,j%nx) for j in range(rows)),bytes(rows*2560*4)))
        tokens += [VIDEO]*rows+[VE]
    return MediaBundle(2560,tuple(tokens),tuple(spans))


def ready_encoder(directory, **options):
    cfg={"min_tokens":8,"max_tokens":300,"video":{"enabled":True,**options}}
    e=VideoEncoder(cfg,directory)
    e.identity="synthetic-unit-test-no-model"
    e.reason=None
    return e


class EncoderCache(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
        self.encoder=ready_encoder(self.directory)
        self.addCleanup(self.encoder.close)
        self.tok=Tokenizer();self.info=clip_info()
        self.bundle=synthetic_video(self.info,self.tok)

    def exchange(self, command, budget):
        op,packet,out=command.split()
        self.assertEqual(op,"ENCV")
        with (self.directory/out).open("wb") as stream:
            write_bundle(stream,self.bundle)
        return "VOK 24 3 0"

    def encode(self, source="a", budget=None, exchange=None):
        b=budget or VideoRequestBudget(self.encoder.policy)
        self.info.charge(b)
        return self.encoder.encode(self.directory/"unused.svf",source,self.info,b,
                                   exchange or self.exchange,self.tok)

    def test_validated_publication_hit_costs_and_cleanup(self):
        self.assertEqual(self.encode(),self.bundle)
        self.assertEqual(len(self.encoder.cache),1)
        self.assertGreater(self.encoder.quota.used,0)
        b=VideoRequestBudget(self.encoder.policy)
        hit=self.encoder.cached("a",b,self.tok)
        self.assertEqual(hit,self.bundle)
        self.assertEqual((b.frames,b.tokens,b.embedding_bytes,b.duration_s),(5,24,24*2560*4,2.5))
        self.encoder.close()
        self.assertEqual(self.encoder.quota.used,0)
        self.assertEqual(list(self.directory.iterdir()),[])

    def test_repeated_hits_consume_logical_budget(self):
        self.encode()
        b=VideoRequestBudget(replace(self.encoder.policy,max_frames=6))
        self.encoder.cached("a",b,self.tok)
        with self.assertRaises(VideoLimitError):
            self.encoder.cached("a",b,self.tok)

    def test_byte_eviction(self):
        self.encoder.close()
        self.encoder=ready_encoder(self.directory,cache_bytes=300000)
        self.encode("a");self.encode("b")
        self.assertEqual(len(self.encoder.cache),1)
        self.assertIsNone(self.encoder.cached("a",VideoRequestBudget(self.encoder.policy)))
        self.assertIsNotNone(self.encoder.cached("b",VideoRequestBudget(self.encoder.policy)))

    def test_decoder_identity_changes_cache_key(self):
        first=self.encoder.key("source")
        self.encoder.identity="different-preprocessor-tokenizer-projector"
        self.assertNotEqual(first,self.encoder.key("source"))

    def test_unfinished_or_wrong_reply_never_caches(self):
        def bad(command,budget):
            output=self.directory/command.split()[2]
            output.write_bytes(b"unfinished")
            Path(str(output)+".partial").write_bytes(b"partial")
            return "ERR invalid spool"
        with self.assertRaises(VideoError):
            self.encode(exchange=bad)
        self.assertEqual(len(self.encoder.cache),0)
        self.assertEqual(self.encoder.quota.used,0)
        self.assertEqual(list(self.directory.iterdir()),[])

    def test_tokenizer_mismatch_is_rejected_before_cache(self):
        b=self.bundle
        self.bundle=replace(b,tokens=(255,)+b.tokens[1:])
        with self.assertRaisesRegex(VideoError,"tokenizer disagree"):
            self.encode()
        self.assertEqual(self.encoder.quota.used,0)
        self.assertEqual(len(self.encoder.cache),0)

    def test_request_file_owns_quota_until_close(self):
        with self.encoder.request_artifact(self.bundle,self.encoder.limits()) as owned:
            with owned.path.open("rb") as stream:
                self.assertEqual(read_bundle(stream,qwen4=True),self.bundle)
            self.assertEqual(owned.reserved,self.encoder.quota.used)
        self.assertEqual(self.encoder.quota.used,0)

    def test_video_span_limit_follows_visual_rows(self):
        count = 258
        self.encoder.policy = replace(self.encoder.policy, max_frames=count, max_duration_s=130)
        info = ClipInfo(count, 2., 128, 64, 129, tuple(range(count)),
                        tuple(i / 2 for i in range(count)), 128, 64)
        bundle = synthetic_video(info, self.tok)
        limits = self.encoder.limits()
        self.assertGreaterEqual(limits.max_spans, len(bundle.spans))
        with self.encoder.request_artifact(bundle, limits) as owned:
            with owned.path.open("rb") as source:
                self.assertEqual(read_bundle(source, limits, qwen4=True), bundle)
        self.assertEqual(self.encoder.quota.used, 0)

    def test_unavailable_video_does_not_touch_legacy_encoder(self):
        e=VideoEncoder({"video":{"enabled":True}},self.directory)
        e.configure("ERR old encoder",lambda *args:self.fail("must not configure an old encoder"),
                    VideoRequestBudget(e.policy))
        self.assertFalse(e.available)
        self.assertIn("images only",e.reason)
        bad=VideoEncoder({"video":{"enabled":True,"fps":False}},self.directory)
        self.assertFalse(bad.available)
        self.assertIsNotNone(bad.reason)


if __name__ == "__main__":
    unittest.main()
