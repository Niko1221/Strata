"""Pinned processor traces, synthetic host rows only; no decoder/model/GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from serve.media import MediaBundle, MediaKind, VisualSpan, build_positions, decode, encode
from tools.vision.video_reference_trace import CHECKPOINT_REVISION, METADATA_HASHES, REFERENCE_REVISION, SOURCE_HASHES

FIXTURE = Path(__file__).resolve().parents[1] / "tools/vision/fixtures/qwen-video-reference.json"


def probe(case):
    spans = []
    for descriptor in case["visual_spans"]:
        kind = MediaKind.IMAGE if descriptor["kind"] == "image" else MediaKind.VIDEO
        rows = descriptor["rows"]
        # Known 4x6 pre-merge grids, merge 2: 2x3 row-major host probes.
        positions = tuple((0, row // 3, row % 3) for row in range(rows))
        spans.append(VisualSpan(descriptor["start"], descriptor["pad_id"], kind, 3, positions, bytes(rows * 4),
                                3 if kind == MediaKind.IMAGE else 0, 2 if kind == MediaKind.IMAGE else 0))
    return MediaBundle(1, tuple(case["token_ids"]), tuple(spans))


class ProcessorTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_provenance_and_unrun_gates_are_explicit(self):
        trace = self.trace
        self.assertEqual(trace["reference_revision"], REFERENCE_REVISION)
        self.assertEqual(trace["checkpoint_revision"], CHECKPOINT_REVISION)
        self.assertEqual(trace["source_sha256"], SOURCE_HASHES)
        self.assertEqual(trace["metadata_sha256"], METADATA_HASHES)
        self.assertEqual(trace["sampling_defaults"], {"fps": 2, "min_frames": 4, "max_frames": 768})
        self.assertEqual(trace["embedding_reference"], "NOT_RUN")
        self.assertEqual(trace["decoder_mtmd_comparison"], "NOT_RUN")
        self.assertTrue(all(c["position_reference"] == "PASS_CPU_TORCH" for c in trace["cases"]))
        self.assertEqual(trace["tokenizer_reference"]["status"], "PASS_RUST_TOKENIZERS")
        self.assertEqual(trace["tokenizer_reference"]["cases"], 6)
        self.assertEqual(trace["tokenizer_reference"]["sha256"],
                         "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3")

    def test_reference_frame_groups_and_timestamp_wrappers(self):
        cases = {case["name"]: case for case in self.trace["cases"]}
        self.assertEqual(cases["one_frame"]["padded_indices"], [0, 0])
        self.assertEqual(cases["odd_three_frames"]["padded_indices"], [0, 1, 2, 2])
        self.assertEqual(cases["odd_three_frames"]["merged_timestamps"], [.25, 1.0])
        self.assertEqual(cases["odd_five_frames"]["merged_timestamps"], [.25, 1.25, 2.0])
        self.assertEqual(len(cases["ten_second_clip"]["sampled_indices"]), 20)
        for case in cases.values():
            padded = case["padded_indices"]
            expected = [(padded[i] / case["source_fps"] + padded[i + 1] / case["source_fps"]) / 2
                        for i in range(0, len(padded), 2)]
            self.assertEqual(case["merged_timestamps"], expected)
            videos = [s for s in case["visual_spans"] if s["kind"] == "video"]
            self.assertEqual(len(videos), len(expected))
            for timestamp in expected:
                wrapper = f"<{timestamp:.1f} seconds><|vision_start|>" + "<|video_pad|>" * 6 + "<|vision_end|>"
                self.assertIn(wrapper, case["expanded_text"])
            for span in videos:
                self.assertEqual(span["rows"], 6)
                self.assertEqual(span["pad_id"], 248057)

    def test_mixed_processor_tokens_survive_transport(self):
        for case in self.trace["cases"]:
            with self.subTest(case=case["name"]):
                bundle = probe(case)
                self.assertEqual(decode(encode(bundle)), bundle)
                plan = build_positions(bundle, len(bundle.tokens) + 2)
                self.assertEqual(len(plan.rows), len(bundle.tokens) + 2)
                self.assertEqual(sum(row is not None for row in plan.rows), sum(s["rows"] for s in case["visual_spans"]))
        mixed = next(c for c in self.trace["cases"] if c["name"] == "mixed_image_video")
        self.assertEqual([s["kind"] for s in mixed["visual_spans"]], ["image", "video", "video", "video"])
        self.assertEqual(mixed["visual_spans"][0]["pad_id"], 248056)

    @unittest.skipUnless(os.environ.get("MEDIA_TEST_EXE"), "C++ parity job needs MEDIA_TEST_EXE")
    def test_cpp_consumes_the_same_processor_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            for case in self.trace["cases"]:
                bundle = probe(case)
                raw = encode(bundle)
                path.write_bytes(raw)
                result = subprocess.run([os.environ["MEDIA_TEST_EXE"], "--roundtrip", str(path)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, raw)
                capacity = len(bundle.tokens) + 2
                result = subprocess.run([os.environ["MEDIA_TEST_EXE"], "--positions", str(path), str(capacity)],
                                        capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = build_positions(bundle, capacity)
                expected = []
                for pos, row in zip(plan.positions, plan.rows):
                    span, index = row if row is not None else (-1, 0)
                    expected.append(f"{pos[0]} {pos[1]} {pos[2]} {span} {index}\n")
                self.assertEqual(result.stdout.decode(), "".join(expected))


if __name__ == "__main__":
    unittest.main()
