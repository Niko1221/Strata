"""Fused NF4 matmul（Triton）：读 packed uint8 权重，寄存器内解码，直接出结果。

动机（2026-10-28 阶段 1 数据）：纯 torch 的 nf4.dequantize（~20 kernel）+ F.linear
两步走，每专家把 9.8MB bf16 中间张量写一遍再读一遍，实测解压吞吐仅 13.6GB/s，
是 37s/步里 ~25s 的来源。fused 版不落中间张量，目标 300-600GB/s 等效权重带宽。

数值契约：与 nf4.dequantize+F.linear 逐位对齐做不到（累加顺序不同），契约是
  w = bf16(code[nibble] * scale)   # 与 dequantize 的 bf16 舍入一致
  y = dot(x, w) fp32 累加          # 与 cuBLAS bf16 matmul 同级
自测：随机数据 max|Δ| 应在 1e-2 量级（bf16 舍入 + 求和顺序），见 __main__。

训练契约：基座冻结，只需 dx。forward y = x@W^T，backward dx = dy@W，
都走同一个解码内核；packed/absmax 只读，无 dW。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_w(packed_ptr, absmax_ptr, code_ptr,
              offs_o, offs_k, k0,
              stride_po, stride_ao,
              BO: tl.constexpr, BK: tl.constexpr):
    """解码 W 的 (BO, BK) tile：offs_o 行、全局列 offs_k（NF4，低半字节在前）。
    要求 IN % BK == 0。BK==64 且 k0 按 64 对齐时每块恰落一个 absmax 组（标量快路径）；
    更大的 BK 用 per-element gather（offs_k//64 在块内可变）。"""
    byte_idx = offs_k // 2
    p = tl.load(packed_ptr + offs_o[:, None] * stride_po + byte_idx[None, :])
    nib = tl.where(offs_k[None, :] % 2 == 0, p & 0xF, p >> 4)
    if BK == 64:
        scale = tl.load(absmax_ptr + offs_o * stride_ao + k0 // 64)[:, None]
    else:
        scale = tl.load(absmax_ptr + offs_o[:, None] * stride_ao + (offs_k // 64)[None, :])
    code = tl.load(code_ptr + nib)                                   # LUT gather，L1 命中
    return (code * scale).to(tl.bfloat16)


@triton.jit
def _nf4_fwd_kernel(x_ptr, packed_ptr, absmax_ptr, code_ptr, y_ptr,
                    T, IN, OUT,
                    sx_t, sy_t,
                    stride_po, stride_ao,
                    BT: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    """y (T, OUT) = x (T, IN) @ W^T，W 为 NF4 (OUT, IN)。grid=(cdiv(T,BT), OUT//BO)"""
    pid_t = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_o = pid_o * BO + tl.arange(0, BO)
    tmask = offs_t < T
    acc = tl.zeros((BT, BO), dtype=tl.float32)
    for k0 in range(0, IN, BK):
        offs_k = k0 + tl.arange(0, BK)
        x_tile = tl.load(x_ptr + offs_t[:, None] * sx_t + offs_k[None, :],
                         mask=tmask[:, None], other=0.0)
        w = _decode_w(packed_ptr, absmax_ptr, code_ptr, offs_o, offs_k, k0,
                      stride_po, stride_ao, BO, BK)          # (BO, BK) bf16
        acc += tl.dot(x_tile, tl.trans(w))
    tl.store(y_ptr + offs_t[:, None] * sy_t + offs_o[None, :],
             acc.to(tl.bfloat16), mask=tmask[:, None])


@triton.jit
def _nf4_bwd_kernel(dy_ptr, packed_ptr, absmax_ptr, code_ptr, dx_ptr,
                    T, IN, OUT,
                    sdy_t, sdx_t,
                    stride_po, stride_ao,
                    BT: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    """dx (T, IN) = dy (T, OUT) @ W，W 为 NF4 (OUT, IN)。grid=(cdiv(T,BT), IN//BK)"""
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_k = pid_k * BK + tl.arange(0, BK)
    tmask = offs_t < T
    acc = tl.zeros((BT, BK), dtype=tl.float32)
    for o0 in range(0, OUT, BO):
        offs_o = o0 + tl.arange(0, BO)
        dy_tile = tl.load(dy_ptr + offs_t[:, None] * sdy_t + offs_o[None, :],
                          mask=tmask[:, None], other=0.0)
        w = _decode_w(packed_ptr, absmax_ptr, code_ptr, offs_o, offs_k, pid_k * BK,
                      stride_po, stride_ao, BO, BK)          # (BO, BK) bf16
        acc += tl.dot(dy_tile, w)
    tl.store(dx_ptr + offs_t[:, None] * sdx_t + offs_k[None, :],
             acc.to(tl.bfloat16), mask=tmask[:, None])


def _run(x, packed, absmax, code, backward: bool, BT: int = 16, BO: int = 0, BK: int = 0) -> torch.Tensor:
    IN2 = packed.shape[1]          # IN/2
    IN = IN2 * 2
    OUT = packed.shape[0]
    T = x.shape[0]
    x = x.contiguous()
    if not BK:
        BK = 64 if backward else next(c for c in (256, 128, 64) if IN % c == 0)
    if not BO:
        BO = 64 if backward else 16
    assert IN % BK == 0 and OUT % BO == 0
    if not backward:               # y = x @ W^T
        y = torch.empty(T, OUT, dtype=torch.bfloat16, device=x.device)
        grid = (triton.cdiv(T, BT), OUT // BO)
        _nf4_fwd_kernel[grid](x, packed, absmax, code, y, T, IN, OUT,
                              x.stride(0), y.stride(0),
                              packed.stride(0), absmax.stride(0),
                              BT, BO, BK)
        return y
    dx = torch.empty(T, IN, dtype=torch.bfloat16, device=x.device)   # dx = dy @ W
    grid = (triton.cdiv(T, BT), IN // BK)
    _nf4_bwd_kernel[grid](x, packed, absmax, code, dx, T, IN, OUT,
                          x.stride(0), dx.stride(0),
                          packed.stride(0), absmax.stride(0),
                          BT, BO, BK)
    return dx


class FusedNF4LinearFn(torch.autograd.Function):
    """y = x @ dequant(packed, absmax)^T；backward 只出 dx（基座冻结，无 dW）。"""

    @staticmethod
    def forward(ctx, x, packed, absmax, code):
        ctx.save_for_backward(packed, absmax, code)
        return _run(x, packed, absmax, code, backward=False)

    @staticmethod
    def backward(ctx, dy):
        packed, absmax, code = ctx.saved_tensors
        return _run(dy.contiguous(), packed, absmax, code, backward=True), None, None, None


_CODE_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_code(dev: torch.device) -> torch.Tensor:
    """NF4 码本唯一来源是 nf4.NF4_CODE，这里只做设备搬运，不抄数值。"""
    import nf4
    code = _CODE_CACHE.get(dev)
    if code is None:
        code = nf4.NF4_CODE.to(device=dev, dtype=torch.float32).contiguous()
        _CODE_CACHE[dev] = code
    return code


def fused_nf4_linear(x: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
    """x (T, IN) bf16; packed (OUT, IN/2) uint8; absmax (OUT, IN/64) bf16 → y (T, OUT)。"""
    return FusedNF4LinearFn.apply(x, packed, absmax, _get_code(x.device))


if __name__ == "__main__":
    import nf4, time
    torch.manual_seed(0)
    dev = "cuda"

    for (OUT, IN, tag) in [(1280, 2560, "gu"), (2560, 640, "dn")]:
        w_ref = torch.randn(OUT, IN, device=dev, dtype=torch.float32) * 0.02
        packed, absmax = nf4.quantize(w_ref.to(torch.bfloat16))
        for T in (1, 8, 28):
            x = torch.randn(T, IN, device=dev, dtype=torch.bfloat16)

            # 参考：解压 + cuBLAS
            w_dec = nf4.dequantize(packed, absmax, (OUT, IN))
            y_ref = torch.nn.functional.linear(x, w_dec)

            y = fused_nf4_linear(x, packed, absmax)
            d = (y.float() - y_ref.float()).abs()
            rel = d.max().item() / (y_ref.float().abs().max().item() + 1e-9)
            print(f"{tag} T={T:3d}  max|Δ|={d.max().item():.4e}  rel={rel:.2e}  "
                  f"|y|max={y_ref.abs().max().item():.3f}")

            # 反传对照：dx = dy @ W
            dy = torch.randn(T, OUT, device=dev, dtype=torch.bfloat16)
            dx_ref = dy.float() @ w_dec.float()
            dx = _run(dy.contiguous(), packed, absmax, _get_code(dev), backward=True)
            d2 = (dx.float() - dx_ref.float()).abs().max().item()
            print(f"    bwd  max|Δdx|={d2:.4e}  |dx|max={dx_ref.abs().max().item():.3f}")

    # 吞吐基准：单专家 gu 形状，T=8，扫 (BO, BK) 找最优 CTA 并行度
    OUT, IN, T = 1280, 2560, 8
    w_ref = torch.randn(OUT, IN, device=dev) * 0.02
    packed, absmax = nf4.quantize(w_ref.to(torch.bfloat16))
    x = torch.randn(T, IN, device=dev, dtype=torch.bfloat16)
    code = _get_code(dev)
    w_dec = nf4.dequantize(packed, absmax, (OUT, IN))
    y_ref = torch.nn.functional.linear(x, w_dec)

    def deq_lin():
        return torch.nn.functional.linear(x, nf4.dequantize(packed, absmax, (OUT, IN)))

    def bench(fn, n=300):
        for _ in range(30): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    t = bench(deq_lin)
    print(f"dequant+linear    {t*1e6:8.1f} µs/call   等效 {packed.numel()/t/2**30:6.1f} GiB/s")
    best = (1e9, None)
    for BO in (16, 32, 64):
        for BK in (64, 128, 256):
            if IN % BK or OUT % BO:
                continue
            try:
                y_c = _run(x, packed, absmax, code, False, BO=BO, BK=BK)
                err = (y_c.float() - y_ref.float()).abs().max().item()
                if err > 0.1:
                    print(f"fused BO={BO:3d} BK={BK:3d}  WRONG (max|Δ|={err:.3e})")
                    continue
                t = bench(lambda: _run(x, packed, absmax, code, False, BO=BO, BK=BK))
            except Exception as ex:
                print(f"fused BO={BO:3d} BK={BK:3d}  SKIP ({type(ex).__name__}: {str(ex)[:120]})")
                continue
            gbs = packed.numel() / t / 2**30
            print(f"fused BO={BO:3d} BK={BK:3d}  {t*1e6:8.1f} µs/call   等效 {gbs:6.1f} GiB/s  max|Δ|={err:.1e}")
            if t < best[0]:
                best = (t, (BO, BK))

    # bwd 配置扫
    dy = torch.randn(T, OUT, device=dev, dtype=torch.bfloat16)
    dx_ref = dy.float() @ w_dec.float()
    bb = (1e9, None)
    for BO in (16, 32, 64):
        for BK in (64, 128):
            try:
                dx_c = _run(dy, packed, absmax, code, True, BO=BO, BK=BK)
                err = (dx_c.float() - dx_ref.float()).abs().max().item()
                if err > 0.1:
                    print(f"bwd  BO={BO:3d} BK={BK:3d}  WRONG (max|Δ|={err:.3e})")
                    continue
                t = bench(lambda: _run(dy, packed, absmax, code, True, BO=BO, BK=BK))
            except Exception as ex:
                print(f"bwd  BO={BO:3d} BK={BK:3d}  SKIP ({type(ex).__name__})")
                continue
            print(f"bwd  BO={BO:3d} BK={BK:3d}  {t*1e6:8.1f} µs/call   max|Δ|={err:.1e}")
            if t < bb[0]:
                bb = (t, (BO, BK))
    print(f"best fwd: BO={best[1][0]} BK={best[1][1]}  {best[0]*1e6:.1f} µs | "
          f"best bwd: BO={bb[1][0]} BK={bb[1][1]}  {bb[0]*1e6:.1f} µs")
