"""tools/test_memt_audit.py - synthetic fixtures for tools/memt_audit.py.

    python tools/test_memt_audit.py [-v]

The fixtures are header-only safetensors files: the audit never reads tensor
data, so this proves the schema checks without any weights or network.
"""
from __future__ import annotations

import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memt_audit as A  # noqa: E402

H, D, KV, N = 2, 16, 1, 32
R = 16
LAYERS, INTERVAL = 4, 2
EXPERTS, TOP_K, MOE_FF, SHARED_FF = 3, 1, 32, 32
LK, LV, LKD, LVD, CONV = 1, 2, 16, 16, 4
VOCAB = 16


def config():
    full = [i for i in range(LAYERS) if (i + 1) % INTERVAL == 0]
    return {
        "model_type": "memt_qwen3_5_moe",
        "architectures": ["MemTQwen3_5MoeForCausalLM"],
        "torch_dtype": "bfloat16",
        "backbone_configs": {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "dtype": "bfloat16",
                "hidden_size": N,
                "num_attention_heads": H,
                "head_dim": D,
                "num_key_value_heads": KV,
                "num_hidden_layers": LAYERS,
                "full_attention_interval": INTERVAL,
                "num_experts": EXPERTS,
                "num_experts_per_tok": TOP_K,
                "moe_intermediate_size": MOE_FF,
                "shared_expert_intermediate_size": SHARED_FF,
                "linear_num_key_heads": LK,
                "linear_num_value_heads": LV,
                "linear_key_head_dim": LKD,
                "linear_value_head_dim": LVD,
                "linear_conv_kernel_dim": CONV,
                "rms_norm_eps": 1e-6,
                "vocab_size": VOCAB,
                "attn_output_gate": True,
                "layer_types": [
                    "full_attention" if i in full else "linear_attention"
                    for i in range(LAYERS)
                ],
                "mtp_num_hidden_layers": 0,
            },
        },
        "memt": {
            "memory_update_mode": "token",
            "gdn_version": "stable_gdn2",
            "memory_use_short_conv": False,
            "memory_head_mode": "mha",
            "memory_read_scale": "k_inv_sqrt",
            "num_memory_heads": H,
            "num_kv_heads": H,
            "linear_attention_state_policy": "reset_per_chunk",
            "chunk_tokens": 4,
            "tail_policy": "commit",
            "output_gate_type": "silu",
            "output_gate_bias": False,
            "output_gate_softcap_enabled": True,
            "output_gate_softcap_limit": 2.0,
            "stable_gdn2_sqrt_erase_gate": True,
            "backbone_gdn_override": False,
        },
    }


def specs():
    report = A.Report()
    geometry, memory = A.parse_config(config(), report)
    assert report.ok, report.errors
    out = A.base_tensor_specs(geometry, memory)
    return {name: (dtype, tuple(shape)) for name, (dtype, shape) in out.items()}


def write_header(path: Path, entries) -> None:
    header = {
        name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
        for name, (dtype, shape) in entries.items()
    }
    payload = json.dumps(header, sort_keys=True).encode("utf-8")
    payload += b" " * ((8 - len(payload) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


class Fixture:
    def __init__(self, root: Path, cfg=None, table=None, shards=2, headers=True, index=True):
        self.root = root
        cfg = copy.deepcopy(config() if cfg is None else cfg)
        table = dict(specs() if table is None else table)
        self.names = sorted(table)
        names = sorted(table)
        shards = max(1, min(shards, len(names)))
        groups = [names[i::shards] for i in range(shards)]
        weight_map = {}
        for shard_index, group in enumerate(groups):
            shard = "model-%05d.safetensors" % shard_index
            weight_map.update({name: shard for name in group})
            if headers:
                write_header(root / shard,
                             {name: table[name] for name in group})
        (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        if index:
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
            )

    def run(self):
        return A.run(None, None, self.root)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_passes(self):
        fixture = Fixture(self.root)
        report = fixture.run()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.memory["layers"], [1, 3])
        self.assertEqual(report.counts["memory_tensors"], 12)
        self.assertEqual(report.counts["tensors_in_headers"], len(fixture.names))
        self.assertEqual(report.counts["gguf_backbone_tensors"], len(fixture.names) - 12 + LAYERS)
        # pinned independently of base_tensor_specs()
        want = {
            "blk.1.memt.in_proj.weight": (["BF16", [5 * H * D + 2 * R, N]], [N, 5 * H * D + 2 * R]),
            "blk.1.memt.f_down.weight": (["BF16", [H * D, R]], [R, H * D]),
            "blk.1.memt.g_down.weight": (["BF16", [H * D, R]], [R, H * D]),
            "blk.1.memt.decay_scale_log": (["BF16", [H]], [H]),
            "blk.1.memt.decay_bias": (["BF16", [H * D]], [H * D]),
            "blk.1.memt.o_norm.weight": (["BF16", [D]], [D]),
        }
        for name, (hf, gguf) in want.items():
            self.assertEqual(report.gguf_tensors[name]["dtype"], hf[0])
            self.assertEqual(report.gguf_tensors[name]["shape_hf"], hf[1])
            self.assertEqual(report.gguf_tensors[name]["shape_gguf"], gguf)

    def test_headers_absent_is_a_warning(self):
        report = Fixture(self.root, headers=False).run()
        self.assertTrue(report.ok, report.errors)
        self.assertTrue(any("not present" in w for w in report.warnings))
        self.assertEqual(report.counts["tensors_in_headers"], 0)

    def test_missing_memory_tensor(self):
        table = specs()
        del table["model.language_model.layers.1.self_attn.memt.o_norm.weight"]
        report = Fixture(self.root, table=table).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("o_norm.weight" in e for e in report.errors))

    def test_wrong_memory_shape(self):
        table = specs()
        name = "model.language_model.layers.3.self_attn.memt.in_proj.weight"
        dtype, shape = table[name]
        table[name] = (dtype, (shape[0] + 1, shape[1]))
        report = Fixture(self.root, table=table).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("in_proj.weight: shape" in e for e in report.errors))

    def test_wrong_dtype(self):
        table = specs()
        name = "model.language_model.layers.0.linear_attn.conv1d.weight"
        dtype, shape = table[name]
        table[name] = ("F16", shape)
        report = Fixture(self.root, table=table).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("dtype F16" in e for e in report.errors))

    def test_unknown_tensor(self):
        table = specs()
        table["model.language_model.layers.0.mystery.weight"] = ("BF16", (N,))
        report = Fixture(self.root, table=table).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("unknown tensor" in e for e in report.errors))

    def test_unknown_memt_field(self):
        cfg = config()
        cfg["memt"]["gate_rank"] = D
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("gate_rank" in e for e in report.errors))

    def test_missing_memt_field(self):
        cfg = config()
        del cfg["memt"]["output_gate_bias"]
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("output_gate_bias is missing" in e for e in report.errors))

    def test_pinned_text_field_missing_fails(self):
        cfg = config()
        del cfg["backbone_configs"]["text_config"]["dtype"]
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("text_config.dtype" in e for e in report.errors))

    def test_absent_optional_none_field_is_fine(self):
        # quantization_config is pinned to None; absent reads as None, as MemT does
        cfg = config()
        self.assertNotIn("quantization_config", cfg["backbone_configs"]["text_config"])
        report = Fixture(self.root, cfg=cfg).run()
        self.assertTrue(report.ok, report.errors)

    def test_variable_text_field_missing_fails(self):
        cfg = config()
        del cfg["backbone_configs"]["text_config"]["hidden_size"]
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("hidden_size" in e for e in report.errors))

    def test_layer_types_mismatch_is_a_warning(self):
        cfg = config()
        cfg["backbone_configs"]["text_config"]["layer_types"][1] = "linear_attention"
        report = Fixture(self.root, cfg=cfg).run()
        self.assertTrue(report.ok, report.errors)
        self.assertTrue(any("layer_types" in w for w in report.warnings))

    def test_memory_head_count_mismatch(self):
        cfg = config()
        cfg["memt"]["num_memory_heads"] = H + 1
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("num_memory_heads" in e for e in report.errors))

    def test_index_less_reads_headers(self):
        fixture = Fixture(self.root, index=False)
        report = fixture.run()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.counts["tensors_in_headers"], len(fixture.names))
        self.assertTrue(any("index" in w for w in report.warnings))

    def test_index_less_still_checks_shapes(self):
        table = specs()
        name = "model.language_model.layers.1.self_attn.memt.o_norm.weight"
        dtype, shape = table[name]
        table[name] = (dtype, (shape[0] + 1,))
        report = Fixture(self.root, table=table, index=False).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("o_norm.weight: shape" in e for e in report.errors))

    def test_corrupt_header_fails(self):
        Fixture(self.root)
        shard = self.root / "model-00000.safetensors"
        shard.write_bytes(b"short")
        report = A.run(None, None, self.root)
        self.assertFalse(report.ok)
        self.assertTrue(any("safetensors" in e for e in report.errors))

    def test_config_and_index_paths_from_other_directories(self):
        Fixture(self.root)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        cfg = elsewhere / "config.json"
        cfg.write_bytes((self.root / "config.json").read_bytes())
        index = self.root / "model.safetensors.index.json"
        # the old bug resolved shards under config_path.parent (elsewhere), which
        # would leave headers empty; they must come from the index's directory
        report = A.run(cfg, index, None)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.counts["tensors_in_headers"], len(specs()))

    def test_wrong_fixed_value(self):
        cfg = config()
        cfg["memt"]["memory_update_mode"] = "chunk"
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("memory_update_mode" in e for e in report.errors))

    def test_num_kv_heads_mismatch(self):
        cfg = config()
        cfg["memt"]["num_kv_heads"] = H + 1
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("num_kv_heads" in e for e in report.errors))

    def test_a_mixture_of_storage_field_is_refused(self):
        for field, value in (("num_storage_experts", 4), ("storage_router_dim", N), ("storage_expert_count", 2)):
            cfg = config()
            cfg["memt"][field] = value
            report = Fixture(self.root, cfg=cfg).run()
            self.assertFalse(report.ok)
            self.assertTrue(any(field in e and "not a supported field" in e for e in report.errors), report.errors)

    def test_checkpoint_id_null_fails(self):
        cfg = config()
        cfg["memt_checkpoint_id"] = None
        report = Fixture(self.root, cfg=cfg).run()
        self.assertFalse(report.ok)
        self.assertTrue(any("memt_checkpoint_id" in e for e in report.errors))

    def test_cli_exit_one(self):
        table = specs()
        del table["model.language_model.norm.weight"]
        Fixture(self.root, table=table)
        self.assertEqual(A.main([str(self.root)]), 1)

    def test_cli_json(self):
        Fixture(self.root)
        out = self.root / "audit.json"
        code = A.main([str(self.root), "--json", "--out", str(out)])
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["memory"]["layers"], [1, 3])
        self.assertIn("gguf_tensors", payload)
        self.assertEqual(
            payload["gguf_tensors"]["blk.1.memt.in_proj.weight"],
            {"dtype": "BF16",
             "shape_hf": [5 * H * D + 2 * R, N],
             "shape_gguf": [N, 5 * H * D + 2 * R]},
        )

    def test_mapping_table(self):
        table = A.mapping_table()
        self.assertEqual(
            table["top"]["model.language_model.embed_tokens.weight"],
            "token_embd.weight",
        )
        self.assertEqual(table["layer"]["self_attn.q_proj.weight"], "attn_q.weight")
        self.assertEqual(
            table["memory"]["self_attn.memt.in_proj.weight"],
            "blk.N.memt.in_proj.weight",
        )
        self.assertIn("linear_attn.A_log", table["transforms"])
        self.assertIn("-exp(A_log)", table["transforms"]["linear_attn.A_log"])
        self.assertIn("add 1", table["norm_offset"])


if __name__ == "__main__":
    unittest.main()
