"""zero_adapter.py — 推理引擎要「顺便能训练」时，`--lora` 指的那个文件。

背景：`LORA <file>` 换的是**引擎启动时那套表**的内容（形状必须不变），所以引擎不带 adapter 时，
`/v1/engine/lora` 直接拒绝。要满足「只要能推理就能训练」，服务启动时就该带着一个 adapter —— 但它
不该由用户手工造。

这个工具造的是**全零 adapter**：A = 0、B = 0。`y + s·B(Ax) = y`，所以带着它服务出去的数值就是基座
自己的（加零唯一还可能改的浮点是 −0 → +0）。它唯一的用途是给换表提供一张可写的表，rank 决定后续
所有步的形状。

    python mezo/zero_adapter.py --gguf <base.gguf> --out zero-r8.gguf --rank 8
    # 然后用它起服务（训练侧不需要再改引擎）：
    #   strata generate --pack ... --lora zero-r8.gguf --serve ...

rank 必须与训练侧一致（`mezo.py --rank`，默认 8；反传线由 `lora_to_gguf.py --rank` 决定）。
写出的 alpha 与 `mezo.py --alpha` 同默认（16），`s = alpha/rank` 乘在 B 上 —— B 全零时它不参与数值，
只影响与后续文件的约定。

**为什么放在训练侧、而不是引擎参数**：推理路径一行不用改。引擎仍然只认 `--lora FILE`（上游行为），
这里只是让那个文件不再需要人手准备。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # training/：objectives（mezo.py 顶部就要它）
sys.path.insert(0, str(HERE))                 # mezo/：本目录的 mezo.py
from mezo import adapter_layout, read_geometry, write_adapter  # noqa: E402


def zero_theta(layout: dict, rank: int) -> dict:
    """全零 A / 全零 B。A 不用随机：B=0 时 A 不出现在输出里，随机只让同一 rank 的两个文件 digest 不同。"""
    theta = {}
    for l, (which, n_in, n_out) in layout.items():
        theta[l] = {which: (np.zeros((rank, n_in), dtype=np.float32),
                            np.zeros((n_out, rank), dtype=np.float32))}
    return theta


def write_zero_adapter(out: str, g: dict, rank: int = 8, alpha: float = 16.0,
                       arch: str = "qwen4exp") -> dict:
    """按几何 `g`（`read_geometry` 的形状）写一个全零 adapter，返回它的布局摘要。

    `arch` 必须与基座一致：引擎按 `general.architecture` 拒收"给别的模型做的 adapter"。
    """
    if rank <= 0:
        raise ValueError(f"rank 必须 ≥ 1，收到 {rank}")
    layout = adapter_layout(g)
    write_adapter(out, layout, zero_theta(layout, rank), alpha, arch)
    return {"path": out, "rank": rank, "alpha": alpha, "layers": len(layout), "arch": arch,
            "n_in": sorted({v[1] for v in layout.values()})}


def model_arch(gguf: str) -> str:
    from gguf_reader import GGUFFile

    a = str(GGUFFile(gguf).metadata.get("general.architecture") or "")
    if not a:
        raise SystemExit(f"{gguf}: metadata 'general.architecture' missing")
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="写一个全零 LoRA adapter：让 --serve 的引擎从第一步就能换表")
    ap.add_argument("--gguf", required=True, help="基座模型的任一 GGUF 分片（只读几何：层数 / 宽度 / 注意力形状）")
    ap.add_argument("--out", required=True, help="输出文件（给引擎的 --lora）")
    ap.add_argument("--rank", type=int, default=8, help="adapter 秩（与训练侧 --rank 一致，默认 8）")
    ap.add_argument("--alpha", type=float, default=16.0, help="lora_alpha（s = alpha / rank；B=0 时不参与数值）")
    a = ap.parse_args(argv)

    g = read_geometry(a.gguf)
    arch = model_arch(a.gguf)
    info = write_zero_adapter(a.out, g, a.rank, a.alpha, arch)
    print(f"写好了 {info['path']}")
    print(f"  arch {info['arch']}  rank {info['rank']}  alpha {info['alpha']:g}  层 {info['layers']}  "
          f"n_in {info['n_in']}  A = B = 0 ⇒ 数值与不带 adapter 相同（加零只可能把 −0 变 +0）")
    print(f"  起服务时带上它：--lora {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
