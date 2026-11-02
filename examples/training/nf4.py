"""NF4 量化/反量化，纯 torch 实现。

为什么不用 bitsandbytes：
  - `bnb.functional.quantize_4bit` 对 3D 输入走的是一条特殊路径，返回的 absmax
    与输入同形（每个元素一个 bf16 尺度），实际占 20 bit/参数，比 8bit 还差；
  - 我们需要按「专家」维度切片反量化（256 个专家只解当前命中的那几个），
    自己实现才能保证 packed / absmax 的形状对 dim 0 可切。

格式与 bnb 对齐，便于互相对照：
  - 每 64 个连续元素一个 block，block 内 absmax = max(|x|)，clamp eps；
  - 归一化后用 bnb 的 NF4 码本做最近邻；
  - 两个 4bit 打成一个 uint8，低半字节在前。
"""

from __future__ import annotations

import torch

# bitsandbytes 的 NF4 码本（16 个值）
NF4_CODE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)

# 相邻码本值的中点，作为桶边界
_BOUNDARIES = (NF4_CODE[:-1] + NF4_CODE[1:]) / 2.0
EPS = 1e-8


def packed_shape(shape: tuple[int, ...], blocksize: int = 64) -> tuple[int, ...]:
    """量化后 packed 张量的形状：最后一维压一半。"""
    assert shape[-1] % blocksize == 0, f"last dim {shape[-1]} not divisible by {blocksize}"
    assert shape[-1] % 2 == 0
    return shape[:-1] + (shape[-1] // 2,)


def absmax_shape(shape: tuple[int, ...], blocksize: int = 64) -> tuple[int, ...]:
    return shape[:-1] + (shape[-1] // blocksize,)


def quantize(x: torch.Tensor, blocksize: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., N) -> (packed uint8 (..., N/2), absmax bf16 (..., N/blocksize))。"""
    dev = x.device
    code = NF4_CODE.to(dev)
    bounds = _BOUNDARIES.to(dev)
    orig = x.shape
    flat = x.reshape(-1, blocksize).float()
    absmax = flat.abs().amax(dim=1).clamp_min(EPS)
    norm = flat / absmax[:, None]
    idx = torch.bucketize(norm.contiguous(), bounds).to(torch.uint8)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous()
    n_blocks = orig[-1] // blocksize
    return packed.view(*orig[:-1], orig[-1] // 2), absmax.view(*orig[:-1], n_blocks).to(torch.bfloat16)


def dequantize(packed: torch.Tensor, absmax: torch.Tensor, shape: tuple[int, ...], blocksize: int = 64) -> torch.Tensor:
    """quantize 的逆。shape 是原始形状（packed 的最后一维 × 2）。"""
    dev = packed.device
    code = NF4_CODE.to(dev)
    packed = packed.reshape(-1, packed.shape[-1])
    n = packed.shape[0]
    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    idx = torch.empty(n, packed.shape[1] * 2, dtype=torch.long, device=dev)
    idx[:, 0::2] = lo
    idx[:, 1::2] = hi
    scale = absmax.reshape(-1, absmax.shape[-1]).to(torch.float32)
    scale = scale.repeat_interleave(blocksize, dim=1)
    out = code[idx] * scale
    return out.reshape(shape).to(torch.bfloat16)


def quantized_bytes(shape: tuple[int, ...], blocksize: int = 64) -> int:
    n = 1
    for d in shape:
        n *= d
    return n // 2 + (n // blocksize) * 2  # uint8 半字节 + bf16 尺度


def _self_test() -> None:
    torch.manual_seed(0)
    for shape in [(256, 1024, 2048), (4, 512, 2048), (2048, 512)]:
        w = (torch.randn(shape, dtype=torch.float32, device="cuda") * 0.02)
        p, a = quantize(w)
        assert tuple(p.shape) == packed_shape(shape), (p.shape, packed_shape(shape))
        assert tuple(a.shape) == absmax_shape(shape), (a.shape, absmax_shape(shape))
        d = dequantize(p, a, shape)
        err = (d.float() - w).abs()
        rel = (err / w.abs().clamp_min(1e-6)).median().item()
        # 只解一个专家，验证 dim 0 可切
        one = dequantize(p[1:2].contiguous(), a[1:2].contiguous(), (1,) + tuple(shape[1:]))
        err1 = (one[0].float() - w[1]).abs().mean().item()
        print(
            f"{shape}: bytes/elem={quantized_bytes(shape) / (torch.tensor(shape).prod().item()):.3f} "
            f"mean_abs_err={err.mean().item():.6f} median_rel={rel:.4f} slice_err={err1:.6f}"
        )
        assert err1 < 0.01, err1


if __name__ == "__main__":
    _self_test()
    print("nf4 self-test ok")
