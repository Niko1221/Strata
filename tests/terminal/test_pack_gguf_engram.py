"""pack_gguf must never take an Engram file's metadata for the model's: ISTA-DASLab's final Engram says general.architecture 'deepseek41-engram'
(the first release's said 'engram').  The fixture is the real header of DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (its 416 bytes: metadata and the tensor
directory, no data)."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "dsv41" / "engram-final-header.gguf"


def tiny_gguf(path: Path, arch: str, extra: dict | None = None) -> Path:
    """A GGUF with only metadata (string and uint32 values), no tensors."""
    kv = {"general.architecture": arch, **(extra or {})}
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
    for k, v in kv.items():
        out += struct.pack("<Q", len(k)) + k.encode()
        if isinstance(v, str):
            out += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(bytes(out))
    return path


class EngramHeader(unittest.TestCase):
    def test_the_real_final_engram_header_is_an_engram_file(self):
        g = PG.Gguf(FIXTURE)
        self.assertEqual(g.kv["general.architecture"], "deepseek41-engram")
        self.assertTrue(PG.is_engram_file(g))
        self.assertEqual(len(g.tensors), 2)
        self.assertEqual(g.data_start, 416)

    def test_it_is_never_the_group_the_model_metadata_comes_from(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shard = PG.Gguf(tiny_gguf(d / "model-00001-of-00004.gguf", "deepseek41", {"split.count": 4}))
            engram = PG.Gguf(FIXTURE)
            for order in ([engram, shard], [shard, engram]):
                kv = PG.model_kv(order)
                self.assertIsNotNone(kv)
                self.assertEqual(kv["general.architecture"], "deepseek41")
            self.assertIsNone(PG.model_kv([engram]))                    # an Engram alone is not a model

    def test_the_first_releases_engram_is_one_too(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "e.gguf", "engram"))))
            self.assertFalse(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "m.gguf", "deepseek41"))))


class LegacyModelMetadata(unittest.TestCase):
    def test_legacy_engram_names_write_the_same_hash_table(self):
        common = {
            "general.architecture": "deepseek41",
            "deepseek41.engram.max_ngram_size": 2,
            "deepseek41.engram.head_count": 1,
            "deepseek41.engram.token_map": [0, 1, 2],
            "deepseek41.engram.multipliers": [3, 5],
            "deepseek41.engram.primes": [7],
            "deepseek41.engram.offsets": [0],
        }
        legacy = {**common, "deepseek41.engram.layer_ids": [1],
                  "deepseek41.engram.key_length": 256, "deepseek41.engram.pad_id": 2}
        current = {**common, "deepseek41.engram.layers": [1],
                   "deepseek41.engram.head_dim": 256, "deepseek41.engram.pad_token_id": 2,
                   "deepseek41.engram.key_length": 999, "deepseek41.engram.pad_id": 999}
        PG.int_tensors_to_engram([], legacy)
        PG.int_tensors_to_engram([], current)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "legacy.bin", Path(d) / "current.bin"
            PG.write_engram(a, legacy, 3)
            PG.write_engram(b, current, 3)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(struct.unpack("<4s6i", a.read_bytes()[:28]),
                             (b"DSE1", 3, 1, 2, 1, 256, 2))

    def test_source_config_restores_omitted_sparse_geometry(self):
        kv = {"general.architecture": "deepseek41", "deepseek41.block_count": 40,
              "deepseek41.embedding_length": 5120, "deepseek41.expert_count": 384}
        cfg = {"text_config": {"num_hidden_layers": 40, "hidden_size": 5120,
               "n_routed_experts": 384, "kv_source_layer_ids": [2, 8, 14, 20],
               "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
               "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048,
               "candidate_block_size": 8}}
        PG.apply_source_config(kv, cfg)
        geometry = PG.geometry(kv, 129280)
        self.assertEqual(geometry["kv_source_layer_ids"], [2, 8, 14, 20])
        self.assertEqual(geometry["index_source_layer_ids"], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(geometry["candidate_source_layer_id"], 20)
        self.assertEqual(geometry["candidate_topk_blocks"], 2048)
        self.assertEqual(geometry["candidate_block_size"], 8)
        cfg["text_config"]["candidate_block_size"] = 16
        with self.assertRaisesRegex(SystemExit, "candidate_block_size"):
            PG.apply_source_config(kv, cfg)


if __name__ == "__main__":
    unittest.main()
