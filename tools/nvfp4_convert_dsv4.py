"""tools/nvfp4_convert_dsv4.py - a ModelOpt NVFP4 DeepSeek V4-Flash checkpoint -> a GGUF Strata can pack.

    python tools/nvfp4_convert_dsv4.py --model <ModelOpt NVFP4 dir> --outfile <model.gguf>

This is llama.cpp's own converter (third_party/llama.cpp at the commit setup.py pins) with the DeepseekV4Model
class patched for the V4-Flash NVFP4 checkpoint and Strata's layout. The stock class converts V4 experts to
MXFP4 from a `.scale` naming that this checkpoint does not use; the checkpoint is ModelOpt NVFP4
(`w{1,2,3}.weight` + 2D `.weight_scale` + scalar `.weight_scale_2` + scalar `.input_scale` per expert), which
the generic base.py NVFP4 path almost handles - except its per-expert merge regex only knows the
`gate_proj|up_proj|down_proj` suffixes and its expert count lookup does not know `n_routed_experts`, so 33,024
individual expert tensors would be written (and the end-of-run flush would try to hold ~138 GB in RAM).

What the patches here do:

  * merge each (layer, projection)'s 256 experts into one NVFP4 tensor named
    `model.layers.N.mlp.experts.{gate,down,up}_proj.weight` (w1->gate, w2->down, w3->up), which the DEEPSEEK4
    tensor map writes as blk.N.ffn_{gate,down,up}_exps.weight - the same shape Strata packs for Qwen - plus the
    per-expert F32 .scale / .input_scale side tensors;
  * skip the MXFP4 expert writer and the MOSTLY_MXFP4_MOE ftype override when the source is NVFP4 (the generic
    path already wrote the experts);
  * keep the rest of the class policy: FP8 (attention, shared experts) dequantized and forced Q8_0, BF16 kept
    BF16 (routers, embed/head sources), F32 kept F32 (hyper-connection tables, attn sinks);
  * force token_embd and output to Q8_0, the same serving choice as the Qwen pack.

The MTP layer (mtp.0.*) is dropped here as llama.cpp does; DSpark MTP export is a separate --dspark run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ENGINE = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ENGINE / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))
sys.path.insert(0, str(LLAMA))

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from conversion import deepseek as DS  # noqa: E402
from conversion.base import LazyTorchTensor  # noqa: E402

T = gguf.GGMLQuantizationType

Q8_NAMES = {"token_embd.weight", "output.weight"}

# layers.N.ffn.experts.E.w{1,2,3}.weight -> (bid, expert_id, proj)
EXPERT_RE = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
PROJ_NAMES = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def install_policy() -> None:
    cls = DS.DeepseekV4Model

    def generate_nvfp4_tensors(self):
        # base.py's version, with the expert pattern and count made V4-aware. Writes one merged NVFP4 tensor
        # per (layer, proj) plus per-expert F32 .scale / .input_scale, and single tensors for anything else.
        expert_blocks: dict = {}
        expert_scales: dict = {}
        expert_input_scales: dict = {}
        expert_shapes: dict = {}
        n_experts = int(self.hparams.get("n_routed_experts") or 0)
        consumed: list[str] = []

        def flush(key):
            blocks = expert_blocks.pop(key)
            scales = expert_scales.pop(key)
            input_scales = expert_input_scales.pop(key)
            shape = expert_shapes.pop(key)
            bid, proj_type = key
            blocks.sort(key=lambda x: x[0])
            merged = np.stack([b[1] for b in blocks], axis=0)
            merged_name = f"model.layers.{bid}.mlp.experts.{proj_type}.weight"
            new_name = self.map_tensor_name(merged_name)
            self.gguf_writer.add_tensor(new_name, merged, raw_dtype=T.NVFP4)
            scales.sort(key=lambda x: x[0])
            self._write_scales_tensor(new_name.replace(".weight", ".scale"), [s[1] for s in scales])
            input_scales.sort(key=lambda x: x[0])
            self._write_scales_tensor(new_name.replace(".weight", ".input_scale"), [s[1] for s in input_scales])
            print(f"nvfp4_convert_dsv4: repacked {new_name} [{len(blocks)}, {shape[0]}, {shape[1]}]", flush=True)
            del blocks, merged

        for name in list(self.model_tensors.keys()):
            if not name.endswith(".weight"):
                continue
            scale_name = name.replace(".weight", ".weight_scale")
            scale2_name = name.replace(".weight", ".weight_scale_2")
            input_scale_name = name.replace(".weight", ".input_scale")
            if scale_name not in self.model_tensors:
                continue
            weight = LazyTorchTensor.to_eager(self.model_tensors[name]())
            scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())
            if scale.ndim < 2:  # FP8 with 1D scales stays for dequant_model
                continue
            scale2 = LazyTorchTensor.to_eager(self.model_tensors.get(scale2_name, lambda: torch.tensor(1.0))())
            input_scale = LazyTorchTensor.to_eager(self.model_tensors.get(input_scale_name, lambda: torch.tensor(1.0))())
            consumed.extend([name, scale_name])
            if scale2_name in self.model_tensors:
                consumed.append(scale2_name)
            if input_scale_name in self.model_tensors:
                consumed.append(input_scale_name)

            m = EXPERT_RE.match(name)
            if m:
                bid, expert_id, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                key = (bid, PROJ_NAMES[proj])
                raw, shape = self._nvfp4_pack(weight, scale)
                expert_blocks.setdefault(key, []).append((expert_id, raw.copy()))
                expert_scales.setdefault(key, []).append((expert_id, float(scale2.float().sum())))
                expert_input_scales.setdefault(key, []).append((expert_id, float(input_scale.float().sum())))
                expert_shapes.setdefault(key, shape)
                if n_experts > 0 and len(expert_blocks[key]) >= n_experts:
                    flush(key)
            else:
                self._repack_nvfp4(name, weight, scale, scale2, input_scale)

        for key in list(expert_blocks.keys()):
            flush(key)

        for name in consumed:
            self.model_tensors.pop(name, None)
        for name in list(self.model_tensors.keys()):
            if name.endswith((".k_scale", ".v_scale")):
                del self.model_tensors[name]

    def generate_extra_tensors(self):
        # NVFP4 experts are already written by generate_nvfp4_tensors; only the hash-routing tables and the
        # MTP eh_proj concat remain from the stock implementation.
        if self._is_nvfp4:
            consumed: list[str] = []
            if not self.mtp_only:
                consumed.extend(self._write_hash_routing_tensors())
            main_layers = self.hparams["num_hidden_layers"]
            for bid in range(main_layers, self.block_count):
                e_name = f"layers.{bid}.nextn.e_proj.weight"
                h_name = f"layers.{bid}.nextn.h_proj.weight"
                if e_name not in self.model_tensors and h_name not in self.model_tensors:
                    continue
                e_proj = LazyTorchTensor.to_eager(self.model_tensors[e_name]())
                h_proj = LazyTorchTensor.to_eager(self.model_tensors[h_name]())
                yield (f"layers.{bid}.nextn.eh_proj.weight", torch.cat((e_proj, h_proj), dim=1).contiguous())
                consumed.extend((e_name, h_name))
            for name in consumed:
                del self.model_tensors[name]
            self._dsv4_mxfp4_generated = True
            return
        yield from generate_extra_tensors_stock(self)

    def prepare_tensors(self):
        super(DS.DeepseekV4Model, self).prepare_tensors()
        if not self._is_nvfp4:
            self._is_mxfp4 = True
            self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE

    base_force = cls.tensor_force_quant

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if n_dims >= 2 and new_name in Q8_NAMES:
            return T.Q8_0
        return base_force(self, name, new_name, bid, n_dims)

    generate_extra_tensors_stock = cls.generate_extra_tensors
    cls._generate_nvfp4_tensors = generate_nvfp4_tensors
    cls.generate_extra_tensors = generate_extra_tensors
    cls.prepare_tensors = prepare_tensors
    cls.tensor_force_quant = tensor_force_quant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the ModelOpt NVFP4 checkpoint directory")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    model = pathlib.Path(a.model)
    out = pathlib.Path(a.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    install_policy()
    import convert_hf_to_gguf as C  # noqa: E402  (after the policy is installed on the class)

    argv = [str(model), "--outtype", "q8_0", "--outfile", str(out), "--use-temp-file"]
    if a.dry_run:
        argv.append("--dry-run")
    sys.argv = ["convert_hf_to_gguf.py"] + argv
    t0 = time.time()
    rc = C.main()
    took = time.time() - t0

    try:
        commit = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {
        "source": str(model),
        "llama_cpp_commit": commit,
        "policy": {
            "experts": "NVFP4 merged per (layer, proj) from w1/w2/w3 + per-expert F32 .scale/.input_scale",
            "fp8": "attention + shared experts dequantized -> Q8_0",
            "q8_0_forced": sorted(Q8_NAMES),
            "bf16": "kept (routers and BF16 sources)",
            "f32": "kept (hyper-connection tables, attn sinks)",
        },
        "mtp": "dropped here; DSpark export is a separate --dspark run",
        "seconds": round(took, 1),
    }
    if not a.dry_run:
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("nvfp4_convert_dsv4: done in %.0f s" % took)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
