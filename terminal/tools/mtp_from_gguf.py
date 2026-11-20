"""terminal/tools/mtp_from_gguf.py - a llama.cpp MTP GGUF (Qwen4Exp, `convert_hf_to_gguf.py --mtp`) back to the BF16
`mtp.*` tensors that terminal/tools/mtp_pack.py reads, in the same folder layout terminal/tools/mtp_fetch.py writes (one
raw BF16 file per tensor plus mtp-manifest.json).

Why: some models ship their MTP head only as llama.cpp's standalone draft GGUF, not as the Hugging Face tensors
Strata's MTP path is built from (a pruned derivative of Qwen3.8-Flash-Next, for example). Every change
llama.cpp's Qwen4Exp converter makes to these tensors can be undone, so this undoes them:
  - names: blk.<n>.* / blk.<n>.nextn.* back to mtp.layers.0.* / mtp.* (the table below);
  - norm weights: the converter stores 1 + w in F32, so w = F32 - 1, rounded to BF16 (lossless except for |w| below
    about 1e-5, where F32's 1 + w already dropped bits: measured on Qwen3.8-Flash-Next, 1 value in 10,240 in two
    tensors, off by ~3e-8);
  - nextn.eh_proj = [fc_embedding | fc_hidden] (torch.cat dim=1), split in half;
  - indexer.q_proj / k_proj = index_qk_proj split by rows, joined again;
  - ffn_gate_exps / ffn_up_exps = experts.gate_up_proj split on dim -2, joined again (gate first);
  - the router and the shared-expert gate are F32 copies of BF16 values (exact back to BF16).
Newer llama.cpp (llama.cpp #31028, #32102) merges hc_*_down with hc_*_inject into hc_*_down_inject; that layout is
split again (the inject part is the last hc_count rows).

A head with another number of routed experts than the canonical 512 (a pruned model's) comes through unchanged; the
engine takes the draft layer's expert count from the packed experts.bin.

Checked by a round trip (2026-11-16): Qwen3.8-Flash-Next's own 31 BF16 MTP tensors -> llama.cpp b38960f `--mtp` ->
this tool gives the original bytes (mtp-manifest.json SHA-256) for every tensor but the two norms above. Run that
check with --verify.

Usage:
    python terminal/tools/mtp_from_gguf.py --gguf <model>-mtp-BF16.gguf --out <dir>
    python terminal/tools/mtp_from_gguf.py --gguf <model>-mtp-BF16.gguf --out <dir> --verify <dir of mtp_fetch.py>
then: python terminal/tools/mtp_pack.py --src <dir> --experts q2_0 --out <dir>/mtp-q2_0.gguf
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

# llama.cpp name after "blk.<n>." -> HF name, and whether it is a norm stored as 1 + w
SIMPLE = {
    "nextn.hc_head_norm.weight": ("mtp.hyper_connection_mixer.hc_norm.weight", True),
    "nextn.hc_head_down.weight": ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", False),
    "nextn.hc_head_up.weight": ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", False),
    "nextn.enorm.weight": ("mtp.pre_fc_norm_embedding.weight", True),
    "nextn.hnorm.weight": ("mtp.pre_fc_norm_hidden.weight", True),
    "hc_attn_inject.weight": ("mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", False),
    "hc_attn_norm.weight": ("mtp.layers.0.attn_hyper_connection.hc_norm.weight", True),
    "hc_attn_down.weight": ("mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", False),
    "hc_attn_up.weight": ("mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", False),
    "hc_ffn_inject.weight": ("mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", False),
    "hc_ffn_norm.weight": ("mtp.layers.0.mlp_hyper_connection.hc_norm.weight", True),
    "hc_ffn_down.weight": ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", False),
    "hc_ffn_up.weight": ("mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", False),
    "ffn_down_exps.weight": ("mtp.layers.0.mlp.experts.down_proj", False),
    "ffn_gate_inp.weight": ("mtp.layers.0.mlp.gate.weight", False),
    "ffn_gate_inp_shexp.weight": ("mtp.layers.0.mlp.shared_expert_gate.weight", False),
    "ffn_down_shexp.weight": ("mtp.layers.0.mlp.shared_expert.down_proj.weight", False),
    "ffn_gate_shexp.weight": ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", False),
    "ffn_up_shexp.weight": ("mtp.layers.0.mlp.shared_expert.up_proj.weight", False),
    "indexer.k_norm.weight": ("mtp.layers.0.self_attn.indexer.k_layernorm.weight", True),
    "indexer.q_norm.weight": ("mtp.layers.0.self_attn.indexer.q_layernorm.weight", True),
    "attn_k_norm.weight": ("mtp.layers.0.self_attn.k_norm.weight", True),
    "attn_q_norm.weight": ("mtp.layers.0.self_attn.q_norm.weight", True),
    "attn_q.weight": ("mtp.layers.0.self_attn.q_proj.weight", False),
    "attn_k.weight": ("mtp.layers.0.self_attn.k_proj.weight", False),
    "attn_v.weight": ("mtp.layers.0.self_attn.v_proj.weight", False),
    "attn_output.weight": ("mtp.layers.0.self_attn.o_proj.weight", False),
}
# token_embd / output are the trunk's (the MTP head shares them); Strata reads its own from the main pack
SHARED = {"token_embd.weight", "output.weight", "output_norm.weight"}


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> BF16 bits, round to nearest even (as torch's .to(bfloat16))."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def load(t) -> np.ndarray:
    """A tensor as BF16 bits (uint16) in HF (numpy) dimension order, or float32 for F32 tensors."""
    shape = [int(x) for x in reversed(t.shape)]
    d = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        return d.view("<u2").reshape(shape)
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return d.view(np.float32).reshape(shape)
    raise ValueError(f"{t.name}: {t.tensor_type.name}; only BF16/F32 files can be restored exactly")


def as_bf16(x: np.ndarray, norm: bool) -> np.ndarray:
    if x.dtype == np.uint16:
        if norm:
            raise ValueError("a norm stored as BF16 is unexpected (the converter writes 1-D tensors as F32)")
        return x
    return bf16_bits(x - np.float32(1) if norm else x)


def restore(path: Path):
    r = gguf.GGUFReader(path)
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
    hc = r.fields.get(f"{arch}.hyper_connection.count")
    hc = int(hc.parts[-1][0]) if hc else 4
    blks = {int(t.name.split(".")[1]) for t in r.tensors if t.name.startswith("blk.")}
    if len(blks) != 1:
        raise ValueError(f"expected one MTP block, found blocks {sorted(blks)}")
    pre = f"blk.{blks.pop()}."
    T = {t.name[len(pre):] if t.name.startswith(pre) else t.name: t for t in r.tensors}
    out, used = {}, set()
    for k, (name, norm) in SIMPLE.items():
        if k in T:
            out[name] = as_bf16(load(T[k]), norm)
            used.add(k)
    # merged layout (newer llama.cpp): [down | inject] along the output rows
    for part, hf in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        k = f"hc_{part}_down_inject.weight"
        if k in T:
            x = as_bf16(load(T[k]), False)
            out[f"mtp.layers.0.{hf}.input_mix_weight_down.weight"] = np.ascontiguousarray(x[:-hc])
            out[f"mtp.layers.0.{hf}.block_inject_weight.weight"] = np.ascontiguousarray(x[-hc:])
            used.add(k)
    eh = as_bf16(load(T["nextn.eh_proj.weight"]), False)
    half = eh.shape[1] // 2
    out["mtp.fc_embedding.weight"] = np.ascontiguousarray(eh[:, :half])
    out["mtp.fc_hidden.weight"] = np.ascontiguousarray(eh[:, half:])
    out["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"] = np.concatenate(
        [as_bf16(load(T["indexer.q_proj.weight"]), False), as_bf16(load(T["indexer.k_proj.weight"]), False)], axis=0)
    out["mtp.layers.0.mlp.experts.gate_up_proj"] = np.concatenate(
        [as_bf16(load(T["ffn_gate_exps.weight"]), False), as_bf16(load(T["ffn_up_exps.weight"]), False)], axis=-2)
    used |= {"nextn.eh_proj.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
             "ffn_gate_exps.weight", "ffn_up_exps.weight"}
    left = set(T) - used - SHARED
    if left:
        raise ValueError(f"tensors this tool does not know (a newer converter?): {sorted(left)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="llama.cpp MTP GGUF (BF16)")
    ap.add_argument("--out", required=True, help="new folder for the tensors and mtp-manifest.json")
    ap.add_argument("--verify", help="a terminal/tools/mtp_fetch.py folder: compare every tensor with its original bytes")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    tensors = restore(Path(a.gguf))
    manifest = []
    for name in sorted(tensors):
        x = np.ascontiguousarray(tensors[name])
        f = Path("tensors") / f"{name}.bin"
        x.tofile(out / f)
        manifest.append({"name": name, "dtype": "BF16", "shape": list(x.shape), "bytes": x.nbytes, "file": str(f),
                         "sha256": hashlib.sha256(x.tobytes()).hexdigest(), "source": Path(a.gguf).name})
        print(f"{name}: {list(x.shape)}", flush=True)
    (out / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} tensors to {out}")
    if not a.verify:
        return 0
    ref = {e["name"]: e for e in json.loads((Path(a.verify) / "mtp-manifest.json").read_text())}
    bad = 0
    for e in manifest:
        r = ref.get(e["name"])
        if r is None:
            print(f"VERIFY {e['name']}: not in the reference"); bad += 1; continue
        if r["shape"] != e["shape"]:
            print(f"VERIFY {e['name']}: shape {e['shape']} vs {r['shape']}"); bad += 1; continue
        if r["sha256"] == e["sha256"]:
            continue
        got = np.fromfile(out / e["file"], dtype="<u2")
        want = np.fromfile(Path(a.verify) / r["file"], dtype="<u2")
        f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
        diff = np.nonzero(got != want)[0]
        print(f"VERIFY {e['name']}: {diff.size} of {want.size} values differ, "
              f"max abs {np.abs(f32(got[diff]) - f32(want[diff])).max():.3g}")
        bad += 1
    missing = set(ref) - {e["name"] for e in manifest}
    for m in sorted(missing):
        print(f"VERIFY {m}: missing"); bad += 1
    print(f"VERIFY: {len(manifest) - bad + len(missing)} of {len(ref)} byte-identical, {bad} differ or missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
