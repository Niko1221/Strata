"""zero_adapter：`--lora` 那个文件由训练侧生成，推理引擎一行不用改。

为什么值得钉住：这个文件是「只要能推理就能训练」的唯一前提 —— 引擎的 `LORA <file>` 换的是启动时那套
表的内容，没有表就没有换的余地。它必须同时满足两件事，错一件都不会当场报错：

  1. 载荷全零（A = B = 0）。错成小随机数，模型行为照样"看起来正常"，只是每次推理都偏离基座一点点，
     而训练读到的 logp 也会带着这个偏差 —— 没人在日志里看得出来；
  2. 形状与 `lora_to_gguf.py` / `mezo.py` 写出的完全一致（A 是 [n_in, r]、B 是 [r, n_out]，层到投影的
     映射与引擎的 `is_qsa_layer` 同一条公式）。形状不合，换表直接被引擎拒（`lora_update` 的形状检查），
     等于白做。
"""

# --- path bootstrap: this file sits under training/tests/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import tempfile
import unittest

import mezo  # noqa: F401  （把 tools/ 放进 sys.path：gguf_reader / gguf_writer）
import zero_adapter
from gguf_reader import GGUFFile


GEOM = {                      # 4 层，interval 2 ⇒ 层 1、3 是 QSA（attn_output），0、2 是 GDN（ssm_out）
    "block_count": 4,
    "embedding_length": 8,
    "attention.head_count": 2,
    "attention.key_length": 4,
    "full_attention_interval": 2,
    "ssm.inner_size": 6,
}


class TestZeroAdapter(unittest.TestCase):
    def test_payload_is_all_zero_and_shapes_match_the_trainer(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(_p.Path(td) / "zero.gguf")
            info = zero_adapter.write_zero_adapter(out, GEOM, rank=3, alpha=16.0)
            self.assertEqual(info["layers"], 4)
            f = GGUFFile(out)
            self.assertEqual(f.metadata["general.type"], "adapter")
            self.assertEqual(f.metadata["adapter.type"], "lora")
            self.assertEqual(f.metadata["general.architecture"], "qwen4exp")
            self.assertAlmostEqual(float(f.metadata["adapter.lora.alpha"]), 16.0, places=6)

            # 层 → 投影：1、3 是 attn_output（n_in = head_count × key_length = 8），其余是 ssm_out（n_in = 6）
            want = {0: ("ssm_out", 6), 1: ("attn_output", 8), 2: ("ssm_out", 6), 3: ("attn_output", 8)}
            by_name = {t.name: t for t in f.tensors}
            self.assertEqual(len(by_name), 8)                   # 每层一对 a / b
            for l, (which, n_in) in want.items():
                base = f"blk.{l}.{which}.weight"
                ta, tb = by_name[base + ".lora_a"], by_name[base + ".lora_b"]
                # llama.cpp 的 ne 声明：lora_a ne = [n_in, r]，lora_b ne = [r, n_out]
                self.assertEqual(ta.shape, [n_in, 3], base)
                self.assertEqual(tb.shape, [3, 8], base)

            # 载荷全零：文件在 data_start 之后没有一个非零字节
            raw = _p.Path(out).read_bytes()
            self.assertTrue(raw[f.data_start:].strip(b"\x00") == b"",
                            "adapter 载荷出现非零字节：它就不再是恒等，推理与训练的读法都会偏")

    def test_arch_follows_the_model(self):
        """引擎按 `general.architecture` 拒收给别的模型做的 adapter；写死 `qwen4exp` 会让另一族模型用不了。"""
        with tempfile.TemporaryDirectory() as td:
            out = str(_p.Path(td) / "z.gguf")
            zero_adapter.write_zero_adapter(out, GEOM, rank=2, arch="qwen35moe")
            self.assertEqual(GGUFFile(out).metadata["general.architecture"], "qwen35moe")

    def test_rank_is_checked(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                zero_adapter.write_zero_adapter(str(_p.Path(td) / "z.gguf"), GEOM, rank=0)

    def test_layout_matches_the_engine_rule(self):
        """层 → 投影的映射必须与引擎 `is_qsa_layer`（layer % interval == interval-1）一致。"""
        layout = mezo.adapter_layout(GEOM)
        self.assertEqual([layout[l][0] for l in range(4)],
                         ["ssm_out", "attn_output", "ssm_out", "attn_output"])


if __name__ == "__main__":
    unittest.main()
