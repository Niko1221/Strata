"""Synthetic GLM pack and tokenizer contracts; no model download."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_synth_gguf as synth
import iq_pack
from strata_tokenizer import Tokenizer, BYTE_TO_UNICODE


class GlmPack(unittest.TestCase):
    def test_singleton_conv_dimension_is_byte_preserving(self):
        self.assertEqual(iq_pack.index_shape((4, 1, 256, 1)), (4, 256))

    def test_synthetic_trunk_pack_excludes_mtp_and_dense_lead_experts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source, out = root / "model.gguf", root / "pack"
            md, tensors = synth.build(1234, vocab=True)
            md = [(k, synth.kv_u32(7) if k == "glm5-next.block_count" else v) for k, v in md]
            md.extend([("glm5-next.nextn_predict_layers", synth.kv_u32(1)),
                       ("tokenizer.chat_template", synth.kv_str("{{ messages }}"))])
            tensors = [(name, np.ones((1, 256, 1, 4), np.float32)
                        if name == "blk.0.ssm_conv1d_q.weight" else value) for name, value in tensors]
            tensors.append(("blk.6.nextn.eh_proj.weight", np.ones((128, 128), np.float32)))
            tensors.extend((name.replace("blk.1.", "blk.6."), data.copy())
                           for name, data in list(tensors) if name.startswith("blk.1.ffn_") and "_exps." in name)
            synth.write_gguf(source, md, tensors)
            before = source.read_bytes()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(source), "--out", str(out),
                                             "--compat-bf16"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.main(), 0)
            _, rows = iq_pack.read_index(out / "index.txt")
            self.assertFalse(any(n.startswith("blk.6.") for n in rows))
            self.assertEqual(rows["blk.3.attn_k_b.weight"][7:9], [str(48 * 64), "8"])
            self.assertEqual(rows["blk.0.ssm_conv1d_q.weight"][7:9], ["4", "256"])
            self.assertEqual(rows["blk.1.ffn_gate_inp.weight"][2], "4")
            experts = [int(line.split()[0]) for line in (out / "native_experts.txt").read_text().splitlines()
                       if line and not line.startswith("#")]
            self.assertEqual(experts, list(range(1, 6)))
            self.assertEqual(source.read_bytes(), before)
            cfg = json.loads((out / "tokenizer" / "tokenizer.json").read_text())
            self.assertEqual(cfg["pre"], "glm5")
            self.assertTrue(cfg["ignore_merges"])

    def test_unknown_quantized_glm_projection_refused(self):
        from _paths import add_gguf_py
        add_gguf_py()
        from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            writer = GGUFWriter(source, "glm5-next")
            writer.add_tensor("blk.0.unknown.weight", quants.quantize(np.ones((2, 32), np.float32), Q.Q8_0),
                              raw_dtype=Q.Q8_0)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source), True, "glm5-next"), 1)
            self.assertFalse((root / "dense.bin").exists())

    def test_glm_digit_groups_and_exact_vocab_shortcut(self):
        alphabet = list(BYTE_TO_UNICODE.values())
        tk = Tokenizer(alphabet + ["hello", "123", "456"], [], pre="glm5")
        self.assertEqual(tk.encode("hello"), [tk.ids["hello"]])
        self.assertEqual(tk.encode("1234567"), [tk.ids["123"], tk.ids["456"], tk.ids["7"]])
        qwen = Tokenizer(alphabet + ["hello"], [], pre="qwen35")
        self.assertEqual(qwen.encode("hello"), [qwen.ids[x] for x in "hello"])
        for text in ("e\u0301_j", "Olá, 世界! 😀", "\r\n\t  ", "I'm running 1234567", "\0\x7f"):
            self.assertEqual(tk.decode(tk.encode(text)), text)
        with self.assertRaises(ValueError):
            Tokenizer(alphabet, [], pre="unknown")


if __name__ == "__main__":
    unittest.main()
