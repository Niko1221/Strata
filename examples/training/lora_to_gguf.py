"""把训练产出的 LoRA state_dict 写成 llama.cpp 的 adapter GGUF，让引擎 `--lora` 能加载。

只写引擎真正适配的两类矩阵 —— 它们都是「mixer 输出 → 残差写入」的投影：

    HF `...layers.<l>.self_attn.o_proj`     -> GGUF `blk.<l>.attn_output.weight`
    HF `...layers.<l>.linear_attn.out_proj` -> GGUF `blk.<l>.ssm_out.weight`

其余键（q/k/v、in_proj_*、共享专家）**必须丢掉**：引擎的加载器把「不支持的 tensor」当错误，
因为它宁可不加载，也不愿跑半个 adapter（跑半个得到的是既不像 base 也不像微调版的模型）。
所以训练侧要配合 `--targets o_proj,out_proj` 只训这两类，才谈得上回灌引擎。

形状约定（GGUF 的 ne 是「每维元素数」，dim 0 变化最快）：

    lora_a  ne = [n_in, r]   偏移 (i_in, j_r) = j_r * n_in + i_in  == A 平铺（A 是 (r, n_in) 行主序）
    lora_b  ne = [r, n_out]  偏移 (k_r, j_out) = j_out * r + k     == B 平铺（B 是 (n_out, r) 行主序）

两个都不需要转置，只是 ne 的声明顺序和 numpy 的 shape 直觉相反。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]          # examples/training -> 仓库根
if not (REPO / "tools" / "gguf_writer.py").exists():
    # 早先这个脚本是从 .workbuddy 下的副本跑的（比仓库深一层），那条路径留给它
    REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402

# module 后缀 -> llama.cpp 的 tensor 名（不含 blk.<l>. 前缀）
MAP = {
    "self_attn.o_proj": "attn_output",
    "linear_attn.out_proj": "ssm_out",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=32.0, help="训练时的 lora_alpha（s = alpha / rank）")
    ap.add_argument("--arch", default="qwen4exp",
                    help="写进 general.architecture，**必须等于基座 GGUF 的那个值**：引擎拿它和模型比对，"
                         "不一致直接拒载（'made for X, not Y'，宁可不加载也不跑半个 adapter）。"
                         "Swift-Qwen3.8-Flash-Next = qwen4exp；qwen3.6-35b-a3b = qwen35moe")
    ap.add_argument("--keep", default=",".join(MAP), help="要写出的 module 后缀，逗号分隔")
    ap.add_argument("--allow-empty", action="store_true", help="一个可映射的键都没有时也写出（默认报错）")
    args = ap.parse_args()

    keep = [k.strip() for k in args.keep.split(",") if k.strip()]
    sd = torch.load(args.pt, map_location="cpu")

    # module 名 -> {"A": tensor, "B": tensor}
    mods: dict[str, dict[str, torch.Tensor]] = {}
    skipped: list[str] = []
    for k, v in sd.items():
        if k.endswith(".A"):
            mods.setdefault(k[:-2], {})["A"] = v
        elif k.endswith(".B"):
            mods.setdefault(k[:-2], {})["B"] = v

    layers: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, ab in sorted(mods.items()):
        hit = next((m for m in keep if name.endswith(m)), None)
        if hit is None:
            skipped.append(name)
            continue
        if "A" not in ab or "B" not in ab:
            raise SystemExit(f"{name}: only one of A / B present")
        parts = name.split(".")
        li = None
        # `mtp.layers.0.self_attn.o_proj` 会解析出「第 0 层」——引擎的 MTP 块不在 blk.<l> 命名空间里，
        # 那样写会静默覆盖真正的第 0 层。MTP 只是起草层，本来就不该被适配。
        if parts and parts[0] == "mtp":
            skipped.append(name + " (the MTP draft layer: not adapted)")
            continue
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                li = int(parts[i + 1])
        if li is None:
            skipped.append(name)
            continue
        a = ab["A"].float().numpy()   # (r, n_in)
        b = ab["B"].float().numpy()   # (n_out, r)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise SystemExit(f"{name}: A {a.shape} / B {b.shape} do not agree on the rank")
        if float(np.abs(b).max()) == 0.0:
            skipped.append(name + " (B is zero: never trained)")
            continue
        if li in layers and MAP[hit] in layers[li]:
            raise SystemExit(f"layer {li} {MAP[hit]}: two modules map to one tensor")
        layers.setdefault(li, {})[MAP[hit]] = (a, b)

    n = sum(len(v) for v in layers.values())
    if n == 0 and not args.allow_empty:
        raise SystemExit(f"no adaptor tensor to write (kept {keep}); the state dict has {len(mods)} modules")

    w = GGUFWriter()
    w.add("general.type", "adapter", "string")
    w.add("general.architecture", args.arch, "string")
    w.add("adapter.type", "lora", "string")
    w.add("adapter.lora.alpha", float(args.alpha), "f32")

    total = 0
    for li in sorted(layers):
        for which, (a, b) in sorted(layers[li].items()):
            r, n_in = a.shape
            n_out = b.shape[0]
            base = f"blk.{li}.{which}.weight"
            # SEE the module docstring: the flat bytes are already in GGUF order, only the ne is declared reversed
            w.tensors.append(_Tensor(base + ".lora_a", [n_in, r], "F32",
                                     np.ascontiguousarray(a, dtype="<f4").tobytes()))
            w.tensors.append(_Tensor(base + ".lora_b", [r, n_out], "F32",
                                     np.ascontiguousarray(b, dtype="<f4").tobytes()))
            total += r * n_in + n_out * r

    path = w.write(args.out)
    ranks = sorted({a.shape[0] for v in layers.values() for a, _ in v.values()})
    print(f"wrote {path}: {n} layer(s) adapted, {len(w.tensors)} tensors, rank {ranks}, "
          f"{path.stat().st_size / 2**20:.1f} MiB, alpha {args.alpha:g} "
          f"(s = alpha/rank = {args.alpha / ranks[-1]:.3f} for the widest rank)")
    if skipped:
        print(f"dropped {len(skipped)} module(s) the engine does not adapt:")
        for s in skipped[:10]:
            print("  ", s)
        if len(skipped) > 10:
            print(f"   ... {len(skipped) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
