r"""tools/test_glm53_pack.py - the packer's arithmetic, over a tiny synthetic container (no model, no GPU).

    .venv/bin/python -m unittest discover -s tools -p test_glm53_pack.py

The container is written here with the same shape the real one has: every tensor stored FLAT, a quantized tensor
as two entries (`name` U8 codes, `name.qs` F32 scales), and the group size solvable only from the two byte spans.
The point is not that the numbers are big - it is that the plane sizes, the blob layout and the refusals are
checked against something computable by hand.

The blob check is the one that matters: `blob_layout` reorders the three role tensors, so it is only right if the
reordered blob still accounts for every source byte.  A layout that drops or duplicates a plane is the same class
of bug as the fp16-scale-width bug `tools/pack_index.py` documents.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import glm53_geometry as GM
import glm53_pack as P

CFG = {
    "model_type": "glm_moe_dsa", "hidden_size": 64, "moe_intermediate_size": 64, "intermediate_size": 64,
    "q_lora_rank": 64, "kv_lora_rank": 64, "qk_head_dim": 64, "qk_nope_head_dim": 64, "qk_rope_head_dim": 0, "v_head_dim": 64,
    "num_attention_heads": 1, "n_routed_experts": 2, "num_experts_per_tok": 2, "num_hidden_layers": 2,
    "first_k_dense_replace": 1, "vocab_size": 8, "index_topk": 8, "index_head_dim": 64, "index_n_heads": 1,
    "topk_method": "noaux_tc", "scoring_func": "sigmoid", "norm_topk_prob": True, "routed_scaling_factor": 2.5,
    "indexer_types": ["shared", "full"],
}

ATTN = (("self_attn.q_a_proj.weight", 64, 64), ("self_attn.q_b_proj.weight", 64, 64),
        ("self_attn.kv_a_proj_with_mqa.weight", 64, 64), ("self_attn.kv_b_proj.weight", 128, 64),
        ("self_attn.o_proj.weight", 64, 64))
NORMS = (("input_layernorm.weight", 64), ("post_attention_layernorm.weight", 64),
         ("self_attn.q_a_layernorm.weight", 64), ("self_attn.kv_a_layernorm.weight", 64))


def quant4(flat, group=64):
    """The container's own encoding: one f32 scale per group, codes stored as q + 8, two per byte, LSB first."""
    n = len(flat)
    ng = (n + group - 1) // group
    out = bytearray(n // 2)
    scales = []
    for g in range(ng):
        part = flat[g * group:(g + 1) * group]
        amax = max(abs(v) for v in part) if part else 0
        s = max(amax / 7.0, 1e-8)
        scales.append(s)
        for j, v in enumerate(part):
            q = max(-8, min(7, round(v / s))) + 8
            if j % 2:
                out[j // 2] |= q << 4
            else:
                out[j // 2] |= q
    return bytes(out), b"".join(struct.pack("<f", s) for s in scales)


def write_safetensors(path: pathlib.Path, entries):
    """entries: [(name, dtype, shape, data_bytes)].  The header records the flat length, as the real one does."""
    header, at = {}, 0
    for name, dtype, shape, data in entries:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + len(data)]}
        at += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, _, _, data in entries:
            fh.write(data)


def quant_entry(name, cfg):
    """One tensor as the container stores it: codes (U8, flat) and scales (F32, flat), both by byte count."""
    sh = GM.shape_of(name, cfg)
    rows, cols = sh[0], (sh[1] if len(sh) > 1 else 1)
    flat = [7, -7, 0, 1] * (rows * cols // 4)
    codes, scales = quant4(flat)
    return [(name, "U8", [len(codes)], codes), (name + ".qs", "F32", [len(scales) // 4], scales)]


def container(root: pathlib.Path) -> pathlib.Path:
    entries = []
    names = ["model.embed_tokens.weight", "lm_head.weight"]
    for l in range(CFG["num_hidden_layers"]):
        sparse = l >= CFG["first_k_dense_replace"]
        if sparse:
            names += [f"model.layers.{l}.mlp.experts.{e}.{r}.weight"
                      for e in range(CFG["n_routed_experts"]) for r in ("gate_proj", "up_proj", "down_proj")]
            names += [f"model.layers.{l}.mlp.gate.weight",
                      f"model.layers.{l}.mlp.gate.e_score_correction_bias"]
        else:
            names += [f"model.layers.{l}.mlp.{r}.weight" for r in ("gate_proj", "up_proj", "down_proj")]
        names += [f"model.layers.{l}.{s}" for s, _ in NORMS]
        names += [f"model.layers.{l}.{s}" for s, _, _ in ATTN]
    for name in sorted(names):
        sh = GM.shape_of(name, CFG)
        if len(sh) == 1:
            entries.append((name, "F32", [sh[0]], b"\x00" * sh[0] * 4))
            continue
        entries.extend(quant_entry(name, CFG))
    write_safetensors(root / "out-00000.safetensors", entries)
    (root / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
    return root

def index_rows(out: pathlib.Path):
    rows = {}
    for line in (out / "index.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        assert len(f) == 19, line
        rows[f[0]] = f
    return rows


class PackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = container(pathlib.Path(self.tmp.name))
        self.out = pathlib.Path(self.tmp.name) / "pack"
        self.b = P.build(self.root)
        P.write(self.out, self.b, True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_planes_fill_every_tensor(self):
        for r in self.b["rows"]:
            self.assertEqual(r["codes_bytes"] + r["scales_bytes"] + r["offset_bytes"], r["src_bytes"], r["name"])

    def test_the_blob_layout_accounts_for_every_expert_byte(self):
        blob = self.b["blob"]
        # One expert in the container is three tensors of codes + scales; the blob is the same bytes reordered.
        self.assertEqual(blob["blob"], 3 * (2048 + 256), blob)
        self.assertEqual(blob["gu_codes"] + blob["d_codes"] + blob["gu_scales"] + blob["d_scales"], blob["blob"])
        # expert_source.cpp reads per[3] = {up_off, up_off, blob - down_off}: the gate/up CODE plane each, then
        # the down SCALE plane.  Those are not the per-role totals - the roles are 2,304 B each, not 2,048 - so
        # the columns are checked against what the engine actually takes from them.
        per = [blob["up_off"], blob["up_off"], blob["blob"] - blob["down_off"]]
        self.assertEqual(per, [blob["gu_codes"] // 2, blob["gu_codes"] // 2, blob["d_scales"]])

    def test_index_rows_parse_and_are_contiguous(self):
        rows = index_rows(self.out)
        self.assertEqual(len(rows), len(self.b["rows"]))
        order = sorted(rows.values(), key=lambda r: int(r[5]))
        for i in range(len(order) - 1):
            span = int(order[i][4])
            self.assertEqual(int(order[i + 1][5]), int(order[i][5]) + (span + 63) // 64 * 64)

    def test_written_bytes_are_the_source_ranges(self):
        data = (self.root / "out-00000.safetensors").read_bytes()
        pack = {P.FILE_ID[f]: (self.out / f).read_bytes() for f in P.FILE_ID}
        for r in self.b["rows"]:
            t = self.b["tmap"][r["name"]]
            src = data[t.data_off + t.src_off:t.data_off + t.src_off + t.nbytes]
            self.assertEqual(pack[r["file"]][r["src_off"]:r["src_off"] + len(src)], src, r["name"])
            if r["scales_bytes"]:
                q = self.b["tmap"][r["name"] + ".qs"]
                src2 = data[q.data_off + q.src_off:q.data_off + q.src_off + q.nbytes]
                self.assertEqual(pack[r["file"]][r["src_off"] + len(src):r["src_off"] + len(src) + len(src2)],
                                 src2, r["name"])

    def test_blob_is_the_reordered_source(self):
        data = (self.root / "out-00000.safetensors").read_bytes()
        blob, got = self.b["blob"], (self.out / "experts.bin").read_bytes()
        e = self.b["experts"][0]
        names = [f"model.layers.{e['layer']}.mlp.experts.{e['expert']}.{r}.weight" for r in P.ROLES]
        tm = self.b["tmap"]
        codes = [(data[tm[n].data_off + tm[n].src_off:tm[n].data_off + tm[n].src_off + tm[n].nbytes]) for n in names]
        scales = [(data[tm[n + ".qs"].data_off + tm[n + ".qs"].src_off:
                      tm[n + ".qs"].data_off + tm[n + ".qs"].src_off + tm[n + ".qs"].nbytes]) for n in names]
        row, ng = blob["row_bytes"], blob["ngroups"]
        want = b""
        for i in range(64):
            want += codes[0][i * row:(i + 1) * row] + codes[1][i * row:(i + 1) * row]
        for i in range(64):
            want += codes[2][i * row:(i + 1) * row]
        for i in range(3):
            for j in range(64):
                want += scales[i][j * ng * 4:(j + 1) * ng * 4]
        self.assertEqual(got[e["offset"]:e["offset"] + blob["blob"]], want)

    def test_native_experts_line_is_the_blob_the_engine_wants(self):
        lines = [l for l in (self.out / "native_experts.txt").read_text(encoding="utf-8").splitlines()
                 if l and not l.startswith("#")]
        self.assertEqual(len(lines), 1, lines)
        layer, gu, dt, off, blob, gate, up, down, shard = lines[0].split()
        self.assertEqual(int(layer), 1)
        self.assertEqual(int(off), 0)   # layer 0 is dense: the first blob starts at 0
        self.assertEqual(int(blob), self.b["blob"]["blob"])
        self.assertEqual(int(up), self.b["blob"]["up_off"])
        self.assertEqual(int(blob) - int(down), self.b["blob"]["d_scales"])

    def test_missing_scale_plane_is_refused(self):
        tmap = dict(self.b["tmap"])
        name = "model.layers.1.mlp.experts.0.gate_proj.weight"
        tmap.pop(name + ".qs")
        with self.assertRaisesRegex(ValueError, "no .qs scale plane"):
            P.planes(tmap[name], tmap, CFG)

    def test_scale_count_that_does_not_match_the_group_is_refused(self):
        t = self.b["tmap"]["model.layers.1.mlp.experts.0.gate_proj.weight"]
        bad = dict(self.b["tmap"])
        bad[t.name + ".qs"] = dataclasses.replace(t, name=t.name + ".qs", dtype="F32", elems=192, nbytes=768)
        with self.assertRaisesRegex(ValueError, "does not divide"):
            P.planes(t, bad, CFG)

    def test_name_with_a_space_is_refused_before_anything_is_written(self):
        root = pathlib.Path(self.tmp.name) / "spaced"
        root.mkdir()
        (root / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
        write_safetensors(root / "out-00000.safetensors", [("model.layers.0.mlp.gate.weight", "F32", [2], b"\x00" * 8)])
        with self.assertRaisesRegex(ValueError, "space"):
            P.build(root)

    def test_layer_with_the_wrong_expert_count_is_refused(self):
        root = pathlib.Path(self.tmp.name) / "short"
        root.mkdir()
        (root / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
        write_safetensors(root / "out-00000.safetensors",
                          [("model.layers.1.mlp.experts.0.gate_proj.weight", "U8", [2048], b"\x00" * 2048),
                           ("model.layers.1.mlp.experts.0.gate_proj.weight.qs", "F32", [64], b"\x00" * 256)])
        with self.assertRaisesRegex(ValueError, "experts, not"):
            P.build(root)


if __name__ == "__main__":
    unittest.main()