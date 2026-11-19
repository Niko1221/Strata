"""Admission tests use tiny local files; no model or GPU is loaded."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import native_cache_capsule as M
import verify_native_pack_binding as V

class ProductionBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.pack = root / "pack"; self.pack.mkdir()
        (self.pack / "dense.bin").write_bytes(b"dense")
        (self.pack / "index.txt").write_bytes(b"index")
        self.source = root / "source.gguf"; self.source.write_bytes(b"source")
        self.record = {"path":str(self.source), "bytes":6,
                       "sha256":M.sha(b"source"), "expected_sha256":M.sha(b"source"),
                       "matches":True, "metadata_unchanged":True,
                       "mtime_ns":self.source.stat().st_mtime_ns}
        self.receipt = root / "source.json"
        self.receipt.write_text(json.dumps({"models":[self.record]}))
        self.source_bytes = self.receipt.read_bytes()
        self.old = {"pack":str(self.pack), "source_receipt_sha256":M.sha(self.source_bytes),
                    "dense_bytes":5, "dense_spans":[], "experts":{}, "index_rows":1,
                    "native_dense_tensors":0, "tokenizer":{},
                    "metadata_sha256":{"index.txt":M.sha(b"index")}}
        binding = root / "binding.json"; binding.write_text(json.dumps(self.old))
        output = root / "output"; output.mkdir()
        self.args = SimpleNamespace(source_receipt=self.receipt,
            source_receipt_sha256=M.sha(self.source_bytes), pack_receipt=binding,
            pack_receipt_sha256=M.sha(binding.read_bytes()), source=[self.source],
            raw_sha256=[self.record["sha256"]], pack=self.pack, output=output,
            capacity=256, rows=32)
        self.model = SimpleNamespace(metadata={"general.architecture":"test",
            "test.block_count":4, "test.full_attention_interval":4,
            "test.attention.head_count":24, "test.attention.head_count_kv":2,
            "test.attention.key_length":256, "test.rope.dimension_count":64})

    def run_producer(self, during_scan=lambda:None, after_scan=lambda:None):
        # Only the heavyweight GGUF/span scan and header parser are substituted.
        # Receipt selection and all production capture/emission code run normally.
        def scan(pack, receipt_path, **kwargs):
            during_scan()
            data = kwargs["receipt_bytes"] if "receipt_bytes" in kwargs else receipt_path.read_bytes()
            self.assertEqual(data, self.source_bytes, "scan consumed replacement receipt")
            return dict(self.old)
        def header(path):
            after_scan()
            return self.model
        with patch.object(V, "verify", side_effect=scan), patch.object(V.G, "GGUFFile", side_effect=header):
            return M.produce(self.args)

    def assert_rejected(self, during_scan=lambda:None, after_scan=lambda:None):
        with self.assertRaises(ValueError):
            self.run_producer(during_scan, after_scan)
        self.assertEqual(list(self.args.output.iterdir()), [], "published unverified capsule")

    def test_receipt_replacement_cannot_change_scan_input(self):
        self.run_producer(lambda:self.receipt.write_bytes(b'{"models":[]}'))

    def test_duplicate_receipt_rejected_by_shared_verifier_entrypoint(self):
        self.receipt.write_bytes(b'{"models":[],"models":[]}')
        with self.assertRaisesRegex(ValueError, "duplicate receipt field"):
            V.verify(self.pack, self.receipt)

    def test_small_file_change_after_scan_rejected(self):
        self.assert_rejected(after_scan=lambda:(self.pack / "index.txt").write_bytes(b"other"))

    def test_membership_addition_after_scan_rejected(self):
        self.assert_rejected(after_scan=lambda:(self.pack / "extra.txt").write_bytes(b"extra"))

    def test_membership_removal_after_scan_rejected(self):
        self.assert_rejected(after_scan=lambda:(self.pack / "index.txt").unlink())

    def test_dense_change_during_scan_rejected(self):
        self.assert_rejected(during_scan=lambda:(self.pack / "dense.bin").write_bytes(b"changed dense"))

    def test_dense_replacement_after_scan_rejected(self):
        def replace():
            path = self.pack / "dense.bin"
            original = path.stat()
            replacement = self.pack / "replacement"
            replacement.write_bytes(b"dense")
            import os
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            replacement.replace(path)
        self.assert_rejected(after_scan=replace)

    def test_source_change_before_publication_rejected(self):
        self.assert_rejected(after_scan=lambda:self.source.write_bytes(b"changed source"))

    def test_unchanged_produces_expected_bound_bytes(self):
        result = self.run_producer()
        blob = (self.args.output / "f32.capsule").read_bytes()
        self.assertEqual(M.sha(blob), result["outputs"]["f32"]["sha256"])
        self.assertEqual(blob[:8], b"STNCAP1\n")
        self.assertIn(M.text("index.txt") + M.metadata(self.pack / "index.txt", True), blob)
        self.assertIn(M.text("dense.bin") + M.metadata(self.pack / "dense.bin", False), blob)
        self.assertEqual(len(list(self.args.output.iterdir())), 3)

class Admission(unittest.TestCase):
    def test_same_verified_receipt_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"receipt.json"
            p.write_bytes(b'{"a":1}')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(M.trusted_json(p,digest),{"a":1})
            with self.assertRaises(ValueError): M.trusted_json(p,"0"*64)
            p.write_bytes(b'{"a":1,"a":2}')
            with self.assertRaises(ValueError): M.trusted_json(p,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_actual_selected_sources_independent(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"source space.gguf";p.write_bytes(b"original")
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            record={"path":str(p),"bytes":8,"sha256":h,"expected_sha256":h,
                    "matches":True,"metadata_unchanged":True,"mtime_ns":p.stat().st_mtime_ns}
            self.assertEqual(M.validate_source(record,p,h)[0],h)
            q=Path(root)/"wrong.gguf";q.write_bytes(b"original")
            with self.assertRaises(ValueError): M.validate_source(record,q,h)
            with self.assertRaises(ValueError): M.validate_source(record,p,"0"*64)
            p.write_bytes(b"changed!")
            # The producer checks the trusted receipt's exact metadata, not a
            # fresh whole-shard hash. Make metadata drift explicit on NTFS too.
            import os
            os.utime(p, ns=(p.stat().st_atime_ns, record['mtime_ns'] + 1_000_000_000))
            with self.assertRaises(ValueError): M.validate_source(record,p,h)

    def test_oversize_is_rejected_before_content_read(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"metadata.json"
            with p.open("wb") as f: f.truncate(33*1024*1024)
            with self.assertRaises(ValueError): M.trusted_json(p,"0"*64)
            with self.assertRaises(ValueError): M.pack_metadata(Path(root),{"metadata.json":"0"*64})
            with self.assertRaises(ValueError): M.pack_metadata(Path(root),{})

    def test_unicode_path_and_canonical_position(self):
        self.assertEqual(M.text("パック / a\\b")[4:].decode("utf-8"),"パック / a\\b")
        with self.assertRaises(ValueError): M.text("bad\0path")
        self.assertNotEqual(M.position_digest(64,1e7),M.position_digest(128,1e7))
        self.assertNotEqual(M.position_digest(64,1e7),M.position_digest(64,1e6))

if __name__ == "__main__": unittest.main()
