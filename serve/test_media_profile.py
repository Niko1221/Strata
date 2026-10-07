"""Model-specific visual binding, separated from the generic SVE2 codec."""
from dataclasses import replace
import io
import unittest

from serve.media import (HEADER, SPAN, MediaBundle, MediaError, MediaKind, VisualSpan, build_positions,
                         encode, read_bundle, splice_media, validate_qwen4)

VS, VE, IMAGE, VIDEO = 248053, 248054, 248056, 248057


def visual(kind=MediaKind.VIDEO):
    pad = VIDEO if kind == MediaKind.VIDEO else IMAGE
    positions = ((0, 0, 0), (0, 0, 1))
    span = VisualSpan(1, pad, kind, 2, positions, bytes(2 * 2560 * 4),
                      2 if kind == MediaKind.IMAGE else 0, 1 if kind == MediaKind.IMAGE else 0)
    return MediaBundle(2560, (VS, pad, pad, VE), (span,))


class ModelProfile(unittest.TestCase):
    def test_valid_image_video_and_generation_positions(self):
        for kind in (MediaKind.IMAGE, MediaKind.VIDEO):
            bundle = visual(kind)
            validate_qwen4(bundle)
            self.assertEqual(build_positions(bundle, 6).positions, ((0,0,0), (1,1,1), (1,1,2),
                                                                    (3,3,3), (4,4,4), (5,5,5)))
            self.assertEqual(read_bundle(io.BytesIO(encode(bundle)), qwen4=True), bundle)

    def test_profile_rejects_generic_valid_mutations(self):
        b = visual()
        cases = [replace(b, tokens=(17, VIDEO, VIDEO, VE)), replace(b, tokens=(VS, VIDEO, VIDEO, 17)),
                 replace(b, width=1, spans=(replace(b.spans[0], embeddings=bytes(8)),)),
                 replace(b, tokens=(VS, IMAGE, IMAGE, VE), spans=(replace(b.spans[0], pad_id=IMAGE),)),
                 replace(b, spans=(replace(b.spans[0], positions=((1,0,0), (1,0,1))),)),
                 replace(b, spans=(replace(b.spans[0], advance=3),))]
        for bad in cases:
            with self.subTest(bad=bad):
                encode(bad)  # still a valid generic transport
                with self.assertRaises(MediaError):
                    validate_qwen4(bad)

    def test_profile_validation_precedes_embedding_reads(self):
        b = replace(visual(), tokens=(17, VIDEO, VIDEO, VE))
        raw = encode(b)
        payload = HEADER.size + 4 * len(b.tokens) + SPAN.size + 12 * len(b.spans[0].positions)
        class HeaderOnly(io.BytesIO):
            def read(self, n=-1):
                if self.tell() >= payload:
                    raise AssertionError("embedding data was read before profile validation")
                return super().read(n)
        with self.assertRaisesRegex(MediaError, "delimiters"):
            read_bundle(HeaderOnly(raw), qwen4=True)

    def test_whole_slot_splicing_keeps_text_order_and_rebases(self):
        ids = (11, VS, IMAGE, VE, 12, VS, VIDEO, VE, 13)
        bundle = splice_media(ids, [(MediaKind.IMAGE, visual(MediaKind.IMAGE)), (MediaKind.VIDEO, visual())])
        self.assertEqual(bundle.tokens, (11, VS, IMAGE, IMAGE, VE, 12, VS, VIDEO, VIDEO, VE, 13))
        self.assertEqual([s.start for s in bundle.spans], [2, 7])
        validate_qwen4(bundle)

    def test_missing_extra_wrong_kind_and_unbound_slots_fail(self):
        ids = (11, VS, VIDEO, VE, 13)
        bad = [(ids, []), (ids, [(MediaKind.IMAGE, visual(MediaKind.IMAGE))]),
               (ids, [(MediaKind.VIDEO, visual()), (MediaKind.VIDEO, visual())]),
               ((11, VIDEO, 13), [(MediaKind.VIDEO, visual())])]
        for tokens, pieces in bad:
            with self.subTest(tokens=tokens):
                with self.assertRaises(MediaError):
                    splice_media(tokens, pieces)


if __name__ == "__main__":
    unittest.main()
