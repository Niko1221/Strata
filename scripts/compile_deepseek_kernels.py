#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile standard and fused DeepSeek/FlyDSL RDNA4 (gfx1201) kernels and embed them into
Strata-gfx1201-nvfp4-native/src/kernels/cuda/gfx1201_nvfp4_hsaco.cpp.

Strictly keeps FlyDSL-pr1223-alignwithmain/ and llvm-project/ READ-ONLY.
"""

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

WORKSPACE = Path("/home/no/Desktop/ARGON-STRATA-NVFP4")
FLYDSL_ROOT = WORKSPACE / "FlyDSL-pr1223-alignwithmain"
STRATA_ROOT = WORKSPACE / "Strata-gfx1201-nvfp4-native"
HSACO_CPP_PATH = STRATA_ROOT / "src/kernels/cuda/gfx1201_nvfp4_hsaco.cpp"

os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1201")
os.environ.setdefault("ROCM_PATH", str(WORKSPACE / ".venv/lib/python3.12/site-packages/_rocm_sdk_devel"))
_ROCM_LLVM_BIN = str(WORKSPACE / ".venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin")
if _ROCM_LLVM_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_ROCM_LLVM_BIN}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(FLYDSL_ROOT / "build-fly/python_packages"))
sys.path.insert(0, str(FLYDSL_ROOT))

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as mlir_gpu
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from kernels.gemm.bf16_linear_gfx120x import build_bf16_linear_module
from kernels.norm.hyper_connection_gfx120x import build_hc_premix_module
from kernels.qwen4_glue_gfx120x import (
    _LOG2E,
    _NEG,
    _THREADS,
    _WAVE,
    _bf,
    _e4m3_to_f32,
    _exp,
    _f32_to_e2m1_nib,
    _f32_to_e4m3_byte,
    _idx_norm_rope,
    _nvfp4_8_tiered,
    _ptr,
    _sigmoid_f32,
    _vload4,
    _vload8,
    _vload8_bf16,
    build_bf16_gemv_fast_module,
    build_bf16_gemv_m4_module,
    build_hc_premix_decode_module,
)

P_BF16 = flyc.from_c_void_p(fx.BFloat16, 0)
P_F32 = flyc.from_c_void_p(fx.Float32, 0)
P_I32 = flyc.from_c_void_p(fx.Int32, 0)
P_I64 = flyc.from_c_void_p(fx.Int64, 0)
P_U8 = flyc.from_c_void_p(fx.Uint8, 0)

_INV_LN2 = 1.4426950408889634
_LN2 = 0.6931471805599453


def extract_hsaco_and_meta(cf, default_grid: tuple[int, int, int]):
    ir_text = cf._keepalive._ir_text
    ctx = ir.Context()
    mod = ir.Module.parse(ir_text, context=ctx)
    hsaco = None
    for op in mod.body.operations:
        if op.name == "gpu.binary":
            obj = mlir_gpu.ObjectAttr(op.attributes["objects"][0])
            hsaco = obj.object
            break
    if hsaco is None:
        raise RuntimeError("No gpu.binary found in compiled module")
    m = re.search(
        r'#gpu\.kernel_metadata<"([^"]+)".*?reqd_workgroup_size\s*=\s*array<i32:\s*(\d+),\s*(\d+),\s*(\d+)>',
        ir_text,
    )
    if not m:
        raise RuntimeError("Could not parse kernel_metadata")
    kname, bx, by, bz = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    gx, gy, gz = default_grid
    return hsaco, kname, bx, by, bz, gx, gy, gz


# ---------------------------------------------------------------------------
# KRN-1: bf16_gemv_m8_248320_2560 (Single-pass 8-row wave32-coalesced LM Head GEMV)
# ---------------------------------------------------------------------------
def build_bf16_gemv_m8_module(N: int, K: int) -> Callable[..., None]:
    """Wave32-coalesced multi-token BF16 GEMV for M in 1..8: Y[M, N] = X[M, K] @ W[N, K]^T.
    Reads each row of W [N, K] ONCE from GDDR6 and computes up to 8 tokens in registers simultaneously.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    NN, KK = int(N), int(K)
    assert KK % 128 == 0 and NN % 8 == 0, f"K={KK} must be %128==0 and N={NN} %8==0"
    vec_w = 8 if (KK % 256 == 0) else 4
    vload_fn = _vload8 if vec_w == 8 else _vload4
    step_elems = _WAVE * vec_w
    steps = KK // step_elems
    waves_per_block = _THREADS // _WAVE  # 8
    grid_x = NN // waves_per_block
    elem = fx.BFloat16

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def bf16_gemv_m8_kernel(X: fx.Pointer, W: fx.Pointer, Y: fx.Pointer, M: fx.Int32) -> None:
        x_ptr, w_ptr, y_ptr = _ptr(X, elem), _ptr(W, elem), _ptr(Y, elem)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        row = fx.Int32(gpu.block_idx.x) * fx.Int32(waves_per_block) + wave
        base = fx.Int64(row) * fx.Int64(KK)
        x_offs = [
            fx.Int32(0) if r == 0 else (M > fx.Int32(r)).select(fx.Int32(r * KK), fx.Int32(0))
            for r in range_constexpr(8)
        ]
        accs = [fx.Float32(0.0) for _ in range_constexpr(8)]
        for s in range_constexpr(steps):
            k0 = fx.Int32(s * step_elems) + lane * fx.Int32(vec_w)
            wv = vload_fn(w_ptr, base + fx.Int64(k0))
            for r in range_constexpr(8):
                xvr = vload_fn(x_ptr, x_offs[r] + k0)
                for i in range_constexpr(vec_w):
                    accs[r] = accs[r] + xvr[i] * wv[i]
        for o in (16, 8, 4, 2, 1):
            for r in range_constexpr(8):
                accs[r] = accs[r] + fx.gpu.shuffle_xor(fx.Float32(accs[r]), fx.Int32(o), fx.Int32(_WAVE))
        if lane == fx.Int32(0):
            r64 = fx.Int64(row)
            n64 = fx.Int64(NN)
            fx.ptr_store(accs[0].to(elem), y_ptr + r64)
            for r in range_constexpr(1, 8):
                if M > fx.Int32(r):
                    fx.ptr_store(accs[r].to(elem), y_ptr + fx.Int64(r) * n64 + r64)

    @flyc.jit
    def launch(X: fx.Pointer, W: fx.Pointer, Y: fx.Pointer, M: fx.Int32, stream: fx.Stream = fx.Stream(None)) -> None:
        bf16_gemv_m8_kernel(X, W, Y, M).launch(grid=(grid_x, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    bf16_gemv_m8_kernel.__name__ = f"bf16_gemv_m8_n{NN}_k{KK}"
    return launch


# ---------------------------------------------------------------------------
# KRN-3: qsa_swa_sink_decode_fused_nvfp4 (Single-launch tiered NVFP4 Sink + SWA decode for MTP)
# ---------------------------------------------------------------------------
def build_qsa_swa_sink_decode_fused_nvfp4_module(
    num_heads: int = 24, num_kv_heads: int = 2, head_dim: int = 256
) -> Callable[..., None]:
    """Fused single-block-per-head tiered NVFP4 Sink + Sliding-Window decode attention + output sigmoid gate.
    Eliminates QSA block-key generation, scoring, pinning, top-k, partial buffers, and combine for MTP draft decode.
    Grid: (24, 1, 1), Block: (256, 1, 1).
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    H, KV, D = int(num_heads), int(num_kv_heads), int(head_dim)
    assert D == 8 * _WAVE and H % KV == 0
    group = H // KV
    scale = float(D) ** -0.5
    elem = fx.BFloat16
    n_waves = _THREADS // _WAVE  # 8

    @fx.struct
    class SharedStorage:
        m: fx.Array[fx.Float32, n_waves, 16]
        l: fx.Array[fx.Float32, n_waves, 16]
        acc: fx.Array[fx.Float32, n_waves * D, 16]

    def _exp_f32(x):
        return fmath.exp2(x * fx.Float32(_LOG2E), fastmath="fast")

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def qsa_swa_sink_decode_fused_nvfp4_kernel(
        Q: fx.Pointer,
        KC: fx.Pointer,
        VC: fx.Pointer,
        QG: fx.Pointer,
        OUT: fx.Pointer,
        kc_host_delta: fx.Int64,
        vc_host_delta: fx.Int64,
        pos: fx.Int32,
        vram_prefix: fx.Int32,
        sink_tokens: fx.Int32,
        window_tokens: fx.Int32,
    ) -> None:
        q_ptr = _ptr(Q, elem)
        qg_ptr = _ptr(QG, elem)
        o_ptr = _ptr(OUT, elem)
        seq_k = pos + fx.Int32(1)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        m_s = lds.m.view(fx.make_layout(n_waves, 1))
        l_s = lds.l.view(fx.make_layout(n_waves, 1))
        a_s = lds.acc.view(fx.make_layout(n_waves * D, 1))

        h = fx.Int32(gpu.block_idx.x)
        kvh = h // fx.Int32(group)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        d0 = lane * fx.Int32(8)
        qv = _vload8(q_ptr, h * fx.Int32(D) + d0)

        sink_cnt = fx.min(seq_k, sink_tokens)
        win_start = fx.max(sink_cnt, seq_k - window_tokens)
        total_active = sink_cnt + fx.max(fx.Int32(0), seq_k - win_start)

        init = [fx.Float32(-1.0e30), fx.Float32(0.0)] + [fx.Float32(0.0)] * 8
        st = init
        for idx, c in range(wave, total_active, fx.Int32(n_waves), init=init):
            m, l = c[0], c[1]
            acc = [c[2 + i] for i in range_constexpr(8)]
            ii = fx.Int32(idx)
            tok = (ii < sink_cnt).select(ii, win_start + (ii - sink_cnt))
            kv = _nvfp4_8_tiered(KC, kc_host_delta, tok, pos, vram_prefix, kvh, KV, d0)
            s = fx.Float32(0.0)
            for i in range_constexpr(8):
                s = s + qv[i] * kv[i]
            for o in (1, 2, 4, 8, 16):
                s = s + fx.gpu.shuffle_xor(fx.Float32(s), fx.Int32(o), fx.Int32(_WAVE))
            s = s * fx.Float32(scale)
            mn = fx.max(m, s)
            a = _exp_f32(m - mn)
            p = _exp_f32(s - mn)
            vv = _nvfp4_8_tiered(VC, vc_host_delta, tok, pos, vram_prefix, kvh, KV, d0)
            nxt = [mn, l * a + p] + [acc[i] * a + p * vv[i] for i in range_constexpr(8)]
            st = yield nxt

        if lane == fx.Int32(0):
            fx.memref_store(fx.Float32(st[0]), m_s, wave)
            fx.memref_store(fx.Float32(st[1]), l_s, wave)
        for i in range_constexpr(8):
            fx.memref_store(fx.Float32(st[2 + i]), a_s, wave * fx.Int32(D) + d0 + fx.Int32(i))
        gpu.barrier()

        M = fx.Float32(-1.0e30)
        for w in range_constexpr(n_waves):
            M = fx.max(M, fx.Float32(m_s[w]))
        L = fx.Float32(0.0)
        A = fx.Float32(0.0)
        for w in range_constexpr(n_waves):
            e = _exp_f32(fx.Float32(m_s[w]) - M)
            L = L + fx.Float32(l_s[w]) * e
            A = A + fx.Float32(a_s[fx.Int32(w * D) + tid]) * e

        o = A / L
        g = fx.Float32(fx.ptr_load(qg_ptr + h * fx.Int32(2 * D) + fx.Int32(D) + tid))
        sg = fx.Float32(1.0) / (fx.Float32(1.0) + _exp_f32(fx.Float32(0.0) - g))
        fx.ptr_store((o * sg).to(elem), o_ptr + h * fx.Int32(D) + tid)

    @flyc.jit
    def launch(
        Q: fx.Pointer,
        KC: fx.Pointer,
        VC: fx.Pointer,
        QG: fx.Pointer,
        OUT: fx.Pointer,
        kc_host_delta: fx.Int64,
        vc_host_delta: fx.Int64,
        pos: fx.Int32,
        vram_prefix: fx.Int32,
        sink_tokens: fx.Int32,
        window_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        qsa_swa_sink_decode_fused_nvfp4_kernel(
            Q, KC, VC, QG, OUT, kc_host_delta, vc_host_delta, pos, vram_prefix, sink_tokens, window_tokens
        ).launch(grid=(H, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    qsa_swa_sink_decode_fused_nvfp4_kernel.__name__ = f"qsa_swa_sink_decode_fused_nvfp4_h{H}_kv{KV}_d{D}"
    return launch


# ---------------------------------------------------------------------------
# KRN-4: silu_mul_640_strided (Eliminates moe_sh_gu_extract across all 48 layers)
# ---------------------------------------------------------------------------
def build_silu_mul_strided_module(inter: int = 640) -> Callable[..., None]:
    """Row-strided SwiGLU: Out[r, c] = silu(GU[r * in_stride + c]) * GU[r * in_stride + inter + c].
    Reads directly from fused MoE projection [M, 1856] at offset 0 with in_stride = 1856.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    I = int(inter)
    assert I % 8 == 0
    nvec = I // 8  # 80
    elem = fx.BFloat16

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def silu_mul_strided_kernel(
        GU: fx.Pointer,
        Out: fx.Pointer,
        rows: fx.Int32,
        in_stride: fx.Int32,
    ) -> None:
        gu = _ptr(GU, elem)
        out = _ptr(Out, elem)
        idx = fx.Int32(gpu.block_idx.x) * fx.Int32(_THREADS) + fx.Int32(gpu.thread_idx.x)
        total = rows * fx.Int32(nvec)
        if idx < total:
            row = idx // fx.Int32(nvec)
            col = (idx % fx.Int32(nvec)) * fx.Int32(8)
            g_base = fx.Int64(row) * fx.Int64(in_stride) + fx.Int64(col)
            u_base = g_base + fx.Int64(I)
            o_base = fx.Int64(row) * fx.Int64(I) + fx.Int64(col)
            gv = Vec(fx.make_view(fx.add_offset(gu, g_base), fx.make_layout(8, 1)).load())
            uv = Vec(fx.make_view(fx.add_offset(gu, u_base), fx.make_layout(8, 1)).load())
            outs = []
            for i in range_constexpr(8):
                g = fx.Float32(gv[i])
                u = fx.Float32(uv[i])
                s = g / (fx.Float32(1.0) + fmath.exp2(g * fx.Float32(-_INV_LN2), fastmath="fast"))
                outs.append((s.to(elem).to(fx.Float32) * u).to(elem))
            view = fx.make_view(fx.add_offset(out, o_base), fx.make_layout(8, 1))
            view.store(Vec.from_elements(outs, elem))

    @flyc.jit
    def launch(
        GU: fx.Pointer,
        Out: fx.Pointer,
        rows: fx.Int32,
        in_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        n = rows * fx.Int32(nvec)
        grid = (n + fx.Int32(_THREADS - 1)) // fx.Int32(_THREADS)
        silu_mul_strided_kernel(GU, Out, rows, in_stride).launch(
            grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    silu_mul_strided_kernel.__name__ = f"silu_mul_strided_i{I}"
    return launch


# ---------------------------------------------------------------------------
# KRN-5: gated_deltanet_fused_strided (Supports in_stride=16480 + out-of-place S_IN/S_OUT)
# ---------------------------------------------------------------------------
def build_gated_deltanet_fused_strided_module(
    head_dim_k: int = 128,
    head_dim_v: int = 128,
    num_k_heads: int = 16,
    num_v_heads: int = 48,
    conv_width: int = 4,
    activation: str = "silu",
) -> Callable[..., None]:
    """Fused conv1d + Gated DeltaNet recurrence + gated RMSNorm with:
    1. Out-of-place recurrent state pointers (S_IN -> S_OUT) so speculative verification does not overwrite live state.
    2. Configurable input row stride `in_stride` (e.g., 16480 for fused GDN projection [M, 16480] or 10240 for split),
       completely eliminating gdn_proj_split across all 36 GDN layers!
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    dk, dv, hk, hv, W = (int(x) for x in (head_dim_k, head_dim_v, num_k_heads, num_v_heads, conv_width))
    _COLS = 8
    _ROW_GROUPS = _WAVE // _COLS
    group = hv // hk
    H = W - 1
    waves = dv // _COLS
    threads = waves * _WAVE  # 512
    owned = 2 * dk + dv  # 384
    C = 2 * hk * dk + hv * dv  # 10240
    rpl = dk // _ROW_GROUPS  # 32
    q_scale = float(dk) ** -0.5
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        qkv: fx.Array[fx.Float32, owned, 16]
        red: fx.Array[fx.Float32, waves, 16]

    def _rsum(x):
        x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(_COLS), fx.Int32(_WAVE))
        x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(2 * _COLS), fx.Int32(_WAVE))
        return x

    def _csum(x):
        for o in (1, 2, 4):
            x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(o), fx.Int32(_WAVE))
        return x

    def _silu(x):
        return x / (fx.Float32(1.0) + fmath.exp2(x * fx.Float32(-_LOG2E), fastmath="fast"))

    def _gate(x):
        if const_expr(activation == "silu"):
            return _silu(x)
        return fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp2(x * fx.Float32(-_LOG2E), fastmath="fast"))

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gated_deltanet_fused_strided_kernel(
        X: fx.Pointer,
        Z: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        ALOG: fx.Pointer,
        DTB: fx.Pointer,
        CW: fx.Pointer,
        CSI: fx.Pointer,
        CSO: fx.Pointer,
        S_IN: fx.Pointer,
        S_OUT: fx.Pointer,
        NW: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        in_stride: fx.Int32,
        eps: fx.Float32,
    ) -> None:
        x_ptr = _ptr(X, elem)
        z_ptr = _ptr(Z, elem)
        a_ptr = _ptr(A, elem)
        b_ptr = _ptr(B, elem)
        al_ptr = _ptr(ALOG, elem)
        dt_ptr = _ptr(DTB, elem)
        cw_ptr = _ptr(CW, elem)
        csi_ptr = _ptr(CSI, elem)
        cso_ptr = _ptr(CSO, elem)
        sin_ptr = _ptr(S_IN, fx.Float32)
        sout_ptr = _ptr(S_OUT, fx.Float32)
        nw_ptr = _ptr(NW, elem)
        o_ptr = _ptr(O, elem)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        qkv_s = lds.qkv.view(fx.make_layout(owned, 1))
        red_s = lds.red.view(fx.make_layout(waves, 1))

        T = fx.Int32(seq_len)
        bh = fx.Int32(gpu.block_idx.x)
        head = bh % fx.Int32(hv)
        batch = bh // fx.Int32(hv)
        khead = head // fx.Int32(group)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        c = lane % fx.Int32(_COLS)
        r = lane // fx.Int32(_COLS)
        col = wave * fx.Int32(_COLS) + c
        row0 = r * fx.Int32(rpl)

        is_fused = in_stride == fx.Int32(16480)
        z_stride = is_fused.select(fx.Int32(16480), fx.Int32(hv * dv))
        ab_stride = is_fused.select(fx.Int32(16480), fx.Int32(hv))

        own = tid < fx.Int32(owned)
        slot = own.select(tid, fx.Int32(0))
        is_q = slot < fx.Int32(dk)
        is_k = (slot >= fx.Int32(dk)) & (slot < fx.Int32(2 * dk))
        ch = is_q.select(
            khead * fx.Int32(dk) + slot,
            is_k.select(
                fx.Int32(hk * dk) + khead * fx.Int32(dk) + slot - fx.Int32(dk),
                fx.Int32(2 * hk * dk) + head * fx.Int32(dv) + slot - fx.Int32(2 * dk),
            ),
        )
        taps = [fx.Float32(fx.ptr_load(cw_ptr + ch * fx.Int32(W) + fx.Int32(j))) for j in range_constexpr(W)]
        cs_base = batch * fx.Int32(H * C) + ch
        hist0 = [fx.Float32(fx.ptr_load(csi_ptr + cs_base + fx.Int32(j * C))) for j in range_constexpr(H)]

        s_base = bh * fx.Int32(dk * dv) + row0 * fx.Int32(dv) + col
        st0 = [fx.Float32(fx.ptr_load(sin_ptr + s_base + fx.Int32(i * dv))) for i in range_constexpr(rpl)]

        neg_a = fx.Float32(0.0) - fmath.exp2(fx.Float32(fx.ptr_load(al_ptr + head)) * fx.Float32(_LOG2E), fastmath="fast")
        dt_b = fx.Float32(fx.ptr_load(dt_ptr + head))
        nw = fx.Float32(fx.ptr_load(nw_ptr + col))

        init = st0 + hist0
        carry = init
        for t, carried in range(fx.Int32(0), T, fx.Int32(1), init=init):
            sc = [carried[i] for i in range_constexpr(rpl)]
            hist = [carried[rpl + j] for j in range_constexpr(H)]
            tok = batch * T + fx.Int32(t)

            # 1. conv1d + SiLU for the owned channel
            cur = fx.Float32(fx.ptr_load(x_ptr + fx.Int64(tok) * fx.Int64(in_stride) + fx.Int64(ch)))
            win = hist + [cur]
            acc = win[0] * taps[0]
            for j in range_constexpr(1, W):
                acc = acc + win[j] * taps[j]
            if own:
                fx.memref_store(_bf(_silu(acc)), qkv_s, slot)
            gpu.barrier()

            # 2. recurrence on this wave's 8 columns
            ab_off = fx.Int64(tok) * fx.Int64(ab_stride) + fx.Int64(head)
            kv = [fx.Float32(qkv_s[fx.Int32(dk) + row0 + fx.Int32(i)]) for i in range_constexpr(rpl)]
            qv = [fx.Float32(qkv_s[row0 + fx.Int32(i)]) for i in range_constexpr(rpl)]
            vj = fx.Float32(qkv_s[fx.Int32(2 * dk) + col])
            xg = fx.Float32(fx.ptr_load(a_ptr + ab_off)) + dt_b
            ax = fx.max(xg, fx.Float32(0.0) - xg)
            sp = fx.max(xg, fx.Float32(0.0)) + fmath.log2(
                fx.Float32(1.0) + fmath.exp2((fx.Float32(0.0) - ax) * fx.Float32(_LOG2E), fastmath="fast"),
                fastmath="fast",
            ) * fx.Float32(_LN2)
            g = fmath.exp2(neg_a * sp * fx.Float32(_LOG2E), fastmath="fast")
            bv = fx.Float32(fx.ptr_load(b_ptr + ab_off))
            beta = fx.Float32(1.0) / (
                fx.Float32(1.0) + fmath.exp2((fx.Float32(0.0) - bv) * fx.Float32(_LOG2E), fastmath="fast")
            )
            sq = fx.Float32(0.0)
            sk = fx.Float32(0.0)
            mem = fx.Float32(0.0)
            for i in range_constexpr(rpl):
                sq = sq + qv[i] * qv[i]
                sk = sk + kv[i] * kv[i]
                mem = mem + sc[i] * kv[i]
            sq = _rsum(sq)
            sk = _rsum(sk)
            mem = _rsum(mem)
            rq = fmath.rsqrt(sq + fx.Float32(1e-6), fastmath="fast") * fx.Float32(q_scale)
            rk = fmath.rsqrt(sk + fx.Float32(1e-6), fastmath="fast")
            delta = (vj - mem * rk * g) * beta * rk
            oacc = fx.Float32(0.0)
            nxt = []
            for i in range_constexpr(rpl):
                si = g * sc[i] + kv[i] * delta
                nxt.append(si)
                oacc = oacc + si * qv[i]
            o = _bf(_rsum(oacc) * rq)

            # 3. gated RMSNorm across all Dv columns
            wsum = _csum(o * o)
            if lane == fx.Int32(0):
                fx.memref_store(wsum, red_s, wave)
            gpu.barrier()
            tot = fx.Float32(0.0)
            for w in range_constexpr(waves):
                tot = tot + fx.Float32(red_s[w])
            rstd = fmath.rsqrt(tot / fx.Float32(float(dv)) + eps, fastmath="fast")
            z_off = fx.Int64(tok) * fx.Int64(z_stride) + fx.Int64(head * fx.Int32(dv) + col)
            zv = fx.Float32(fx.ptr_load(z_ptr + z_off))
            y = _bf(o * rstd) * nw * _gate(zv)
            if r == fx.Int32(0):
                out_off = fx.Int64(tok * fx.Int32(hv) + head) * fx.Int64(dv) + fx.Int64(col)
                fx.ptr_store(fx.Float32(y).to(elem), o_ptr + out_off)
            carry = yield nxt + hist[1:] + [cur]

        for i in range_constexpr(rpl):
            fx.ptr_store(fx.Float32(carry[i]), sout_ptr + s_base + fx.Int32(i * dv))
        writes = own & ((slot >= fx.Int32(2 * dk)) | (head % fx.Int32(group) == fx.Int32(0)))
        if writes:
            for j in range_constexpr(H):
                fx.ptr_store(fx.Float32(carry[rpl + j]).to(elem), cso_ptr + cs_base + fx.Int32(j * C))

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Z: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        ALOG: fx.Pointer,
        DTB: fx.Pointer,
        CW: fx.Pointer,
        CSI: fx.Pointer,
        CSO: fx.Pointer,
        S_IN: fx.Pointer,
        S_OUT: fx.Pointer,
        NW: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        in_stride: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        gated_deltanet_fused_strided_kernel(
            X, Z, A, B, ALOG, DTB, CW, CSI, CSO, S_IN, S_OUT, NW, O, seq_len, in_stride, eps
        ).launch(
            grid=(hv, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    gated_deltanet_fused_strided_kernel.__name__ = (
        f"gated_deltanet_fused_strided_k{dk}_v{dv}_hk{hk}_hv{hv}_w{W}_{activation}"
    )
    return launch


# ---------------------------------------------------------------------------
# KRN-6: ple_state_update_batch (Updates [9, 10240] PLE state for M tokens in 1 launch)
# ---------------------------------------------------------------------------
def build_ple_state_update_batch_module(C: int = 10240, H: int = 9) -> Callable[..., None]:
    """Shift ple_state [H, C] left by M steps and write the trailing rows of gvn [M, C] into ple_state in 1 launch."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    grid_x = (C + _THREADS - 1) // _THREADS  # 40
    elem = fx.BFloat16

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def ple_state_update_batch_kernel(
        ST: fx.Pointer,
        GVN: fx.Pointer,
        M: fx.Int32,
    ) -> None:
        st_ptr = _ptr(ST, elem)
        gvn_ptr = _ptr(GVN, elem)
        c = fx.Int32(gpu.block_idx.x) * fx.Int32(_THREADS) + fx.Int32(gpu.thread_idx.x)
        if c < fx.Int32(C):
            vals = []
            for r in range_constexpr(H):
                src_idx = fx.Int32(r) + M
                from_st = src_idx < fx.Int32(H)
                st_row = from_st.select(src_idx, fx.Int32(0))
                gvn_row = from_st.select(fx.Int32(0), src_idx - fx.Int32(H))
                v_st = fx.ptr_load(st_ptr + fx.Int64(st_row) * fx.Int64(C) + fx.Int64(c))
                v_gvn = fx.ptr_load(gvn_ptr + fx.Int64(gvn_row) * fx.Int64(C) + fx.Int64(c))
                vals.append(from_st.select(v_st, v_gvn))
            for r in range_constexpr(H):
                fx.ptr_store(vals[r], st_ptr + fx.Int64(r * C) + fx.Int64(c))

    @flyc.jit
    def launch(
        ST: fx.Pointer,
        GVN: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        ple_state_update_batch_kernel(ST, GVN, M).launch(
            grid=(grid_x, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    ple_state_update_batch_kernel.__name__ = f"ple_state_update_batch_c{C}_h{H}"
    return launch


# ---------------------------------------------------------------------------
# KRN-7: qsa_score_blocks_tiered_pinned (Fuses sink & window pinning into decode block scoring)
# ---------------------------------------------------------------------------
def build_qsa_score_blocks_tiered_pinned_module(
    num_heads: int = 4, head_dim: int = 128, rotary_dim: int = 64, ratio: int = 4
) -> Callable[..., None]:
    """Multi-CU QSA block scorer with fused NSA initial sink + local sliding window pinning:
    - Skips global BK loads for pinned blocks (`b < n_sink` or `b >= nb - n_local`) and writes `1.0e30` directly!
    - Eliminates all 12 `qsa_pin_sink_and_window` launches per decode token!
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    elem = fx.BFloat16
    QD = num_heads * head_dim
    scale = float(head_dim) ** -0.5

    @fx.struct
    class SharedStorage:
        q: fx.Array[fx.Float32, QD, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def qsa_score_blocks_tiered_pinned_kernel(
        QK: fx.Pointer,
        QW: fx.Pointer,
        COS: fx.Pointer,
        SIN: fx.Pointer,
        BK: fx.Pointer,
        SCORES: fx.Pointer,
        bk_host_delta_elems: fx.Int64,
        pos: fx.Int32,
        vram_blocks: fx.Int32,
        score_stride: fx.Int32,
        n_sink: fx.Int32,
        n_local: fx.Int32,
        eps: fx.Float32,
    ) -> None:
        t_rel = fx.Int32(gpu.block_idx.y)
        pos_t = pos + t_rel
        qk_ptr = _ptr(QK, elem) + t_rel * fx.Int32(640)
        qw_ptr = _ptr(QW, elem)
        cos_ptr = _ptr(COS, elem)
        sin_ptr = _ptr(SIN, elem)
        bk_ptr = _ptr(BK, elem)
        scores_ptr = _ptr(SCORES, fx.Int32) + fx.Int64(t_rel) * fx.Int64(score_stride)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_s = lds.q.view(fx.make_layout(QD, 1))

        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        if wave < fx.Int32(num_heads):
            hq = wave * fx.Int32(head_dim) + lane * fx.Int32(4)
            qv = _vload4(qk_ptr, hq)
            w4 = _vload4(qw_ptr, lane * fx.Int32(4))
            qn = _idx_norm_rope(qv, lane, w4, cos_ptr, sin_ptr, pos_t, head_dim, rotary_dim, eps)
            for e in range_constexpr(4):
                fx.memref_store(qn[e], q_s, hq + fx.Int32(e))
        gpu.barrier()

        b = fx.Int32(gpu.block_idx.x) * fx.Int32(_THREADS) + tid
        nb = (pos_t + fx.Int32(1)) // fx.Int32(ratio)
        if b < nb:
            is_pinned = (b < n_sink) | (b >= nb - n_local)
            if is_pinned:
                fx.ptr_store(fx.Float32(1.0e30).bitcast(fx.Int32), scores_ptr + b)
            else:
                in_vram = b < vram_blocks
                slot = in_vram.select(b, b - vram_blocks)
                kbase = fx.Int64(slot) * fx.Int64(head_dim) + in_vram.select(fx.Int64(0), bk_host_delta_elems)
                dots = [fx.Float32(0.0)] * num_heads
                for c in range_constexpr(head_dim // 8):
                    kv = Vec(
                        fx.make_view(fx.add_offset(bk_ptr, kbase + fx.Int64(c * 8)), fx.make_layout(8, 1)).load()
                    )
                    for e in range_constexpr(8):
                        kf = fx.Float32(kv[e])
                        for h in range_constexpr(num_heads):
                            dots[h] = dots[h] + fx.Float32(q_s[fx.Int32(h * head_dim + c * 8 + e)]) * kf
                sc = fx.Float32(0.0)
                for h in range_constexpr(num_heads):
                    sc = sc + fx.max(dots[h], fx.Float32(0.0))
                sc = sc * fx.Float32(scale)
                fx.ptr_store(fx.Float32(sc).bitcast(fx.Int32), scores_ptr + b)

    @flyc.jit
    def launch(
        QK: fx.Pointer,
        QW: fx.Pointer,
        COS: fx.Pointer,
        SIN: fx.Pointer,
        BK: fx.Pointer,
        SCORES: fx.Pointer,
        bk_host_delta_elems: fx.Int64,
        pos: fx.Int32,
        vram_blocks: fx.Int32,
        score_stride: fx.Int32,
        n_sink: fx.Int32,
        n_local: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        qsa_score_blocks_tiered_pinned_kernel(
            QK, QW, COS, SIN, BK, SCORES, bk_host_delta_elems, pos, vram_blocks, score_stride, n_sink, n_local, eps
        ).launch(grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    qsa_score_blocks_tiered_pinned_kernel.__name__ = f"qsa_score_blocks_tiered_pinned_h{num_heads}_d{head_dim}"
    return launch


# ---------------------------------------------------------------------------
# KRN-8: qsa_prep_and_cache_write_batch_nvfp4 (Batched 5-in-1 QSA prep + NVFP4 KV cache write)
# ---------------------------------------------------------------------------
def build_qsa_prep_and_cache_write_batch_nvfp4_module(
    num_heads: int = 24,
    num_kv_heads: int = 2,
    head_dim: int = 256,
    rot: int = 64,
    idx_q_dim: int = 512,
    idx_k_dim: int = 128,
) -> Callable[..., None]:
    """Batched 5-in-1 QSA prep + partial RoPE + tiered NVFP4 KV cache write for M tokens (`grid = (4, M, 1)`):
    Replaces 5 separate kernels per QSA layer in `step_prefill_batch`:
      1. `qsa_proj_split`
      2. `qsa_q_extract`
      3. `qk_norm_partial_rope` (Q)
      4. `qk_norm_partial_rope` (K)
      5. `qsa_cache_write_nvfp4`
    For each token `t = block_idx.y`:
      - Blocks `0..2` (`24` waves): reads `PROJ[t, h*512 : (h+1)*512]`, writes `QG_OUT[t, h*512 : (h+1)*512]` (so
        `qsa_prefill_causal_nvfp4` reads the gate from `QG_OUT`), applies `q_norm` + partial RoPE to Q, and writes
        `Q_OUT[t, h*256 : (h+1)*256]`.
      - Block `3`:
        - Copies `PROJ[t, 13312 : 13952]` (`640` BF16) to `IDX_QK_OUT[t, 0 : 640]`, and when `M <= 1024`, also writes
          `PROJ[t, 13824 : 13952]` (`128` BF16) to `RAW_K_CACHE[(pos0 + t) & 1023]`.
        - Waves `0..1` (`64` threads): applies `k_norm` + partial RoPE to `PROJ[t, 12288 : 12800]` into LDS `k_s`.
        - Threads `0..63`: quantizes normed K (from LDS) and V (from `PROJ[t, 12800 : 13312]`) into tiered NVFP4
          `K_CACHE` and `V_CACHE` at `pos = pos0 + t`!
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    H, KV, D = int(num_heads), int(num_kv_heads), int(head_dim)
    assert H == 24 and KV == 2 and D == 256 and rot == 64
    half = rot // 2  # 32
    pair_xor = half // 8  # 4
    rot_lanes = rot // 8  # 8
    total_blocks16 = 2 * KV * 16  # 64
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        k_normed: fx.Array[fx.Float32, KV * D, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def qsa_prep_and_cache_write_batch_nvfp4_kernel(
        PROJ: fx.Pointer,
        Q_NORM: fx.Pointer,
        K_NORM: fx.Pointer,
        COS: fx.Pointer,
        SIN: fx.Pointer,
        QG_OUT: fx.Pointer,
        Q_OUT: fx.Pointer,
        IDX_QK_OUT: fx.Pointer,
        K_CACHE: fx.Pointer,
        V_CACHE: fx.Pointer,
        RAW_K_CACHE: fx.Pointer,
        kc_host_delta: fx.Int64,
        vc_host_delta: fx.Int64,
        pos0: fx.Int32,
        M: fx.Int32,
        vram_prefix: fx.Int32,
        eps: fx.Float32,
    ) -> None:
        t = fx.Int32(gpu.block_idx.y)
        pos = pos0 + t
        proj_ptr = _ptr(PROJ, elem) + fx.Int64(t) * fx.Int64(13952)
        qn_ptr = _ptr(Q_NORM, elem)
        kn_ptr = _ptr(K_NORM, elem)
        c_ptr = _ptr(COS, fx.Float32)
        s_ptr = _ptr(SIN, fx.Float32)
        qgo_ptr = _ptr(QG_OUT, elem) + fx.Int64(t) * fx.Int64(12288)
        qo_ptr = _ptr(Q_OUT, elem) + fx.Int64(t) * fx.Int64(6144)
        iq_ptr = _ptr(IDX_QK_OUT, elem) + fx.Int64(t) * fx.Int64(640)
        rkc_ptr = _ptr(RAW_K_CACHE, elem)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_s = lds.k_normed.view(fx.make_layout(KV * D, 1))

        blk = fx.Int32(gpu.block_idx.x)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        seg = tid % fx.Int32(_WAVE)
        in_rot = seg < fx.Int32(rot_lanes)
        lo = seg < fx.Int32(pair_xor)
        fidx = (seg % fx.Int32(pair_xor)) * fx.Int32(8)
        tbase = fx.Int64(pos) * fx.Int64(half) + fx.Int64(in_rot.select(fidx, fx.Int32(0)))

        if blk < fx.Int32(3):
            h = blk * fx.Int32(8) + wave
            q_off = h * fx.Int32(2 * D) + seg * fx.Int32(8)
            g_off = q_off + fx.Int32(D)
            q_vec = Vec(fx.make_view(fx.add_offset(proj_ptr, fx.Int64(q_off)), fx.make_layout(8, 1)).load())
            g_vec = Vec(fx.make_view(fx.add_offset(proj_ptr, fx.Int64(g_off)), fx.make_layout(8, 1)).load())
            fx.make_view(fx.add_offset(qgo_ptr, fx.Int64(q_off)), fx.make_layout(8, 1)).store(q_vec)
            fx.make_view(fx.add_offset(qgo_ptr, fx.Int64(g_off)), fx.make_layout(8, 1)).store(g_vec)

            xv = [fx.Float32(q_vec[i]) for i in range_constexpr(8)]
            wv = _vload8(qn_ptr, seg * fx.Int32(8))
            ss = fx.Float32(0.0)
            for i in range_constexpr(8):
                ss = ss + xv[i] * xv[i]
            for o in (16, 8, 4, 2, 1):
                ss = ss + fx.gpu.shuffle_xor(fx.Float32(ss), fx.Int32(o), fx.Int32(_WAVE))
            rstd = fmath.rsqrt(ss / fx.Float32(float(D)) + eps, fastmath="fast")
            n = [(xv[i] * rstd * (fx.Float32(1.0) + wv[i])).to(elem).to(fx.Float32) for i in range_constexpr(8)]
            pn = [fx.gpu.shuffle_xor(fx.Float32(n[i]), fx.Int32(pair_xor), fx.Int32(_WAVE)) for i in range_constexpr(8)]
            cv = Vec(fx.make_view(fx.add_offset(c_ptr, tbase), fx.make_layout(8, 1)).load())
            sv = Vec(fx.make_view(fx.add_offset(s_ptr, tbase), fx.make_layout(8, 1)).load())
            outs = []
            for i in range_constexpr(8):
                c = fx.Float32(cv[i])
                s = fx.Float32(sv[i])
                r = lo.select(n[i] * c - pn[i] * s, n[i] * c + pn[i] * s)
                outs.append(in_rot.select(r, n[i]).to(elem))
            view = fx.make_view(fx.add_offset(qo_ptr, fx.Int64(h * fx.Int32(D) + seg * fx.Int32(8))), fx.make_layout(8, 1))
            view.store(fx.Vector.from_elements(outs, elem))
        else:
            # Copy IDX_QK [640] (80 threads * 8 elements = 640)
            if tid < fx.Int32(80):
                iq_off = tid * fx.Int32(8)
                iv = Vec(
                    fx.make_view(fx.add_offset(proj_ptr, fx.Int64(fx.Int32(13312) + iq_off)), fx.make_layout(8, 1)).load()
                )
                fx.make_view(fx.add_offset(iq_ptr, fx.Int64(iq_off)), fx.make_layout(8, 1)).store(iv)

            if (M <= fx.Int32(1024)) & (tid < fx.Int32(idx_k_dim)):
                rk_slot = pos & fx.Int32(1023)
                rk = fx.ptr_load(proj_ptr + fx.Int32(13312 + idx_q_dim) + tid)
                fx.ptr_store(rk, rkc_ptr + rk_slot * fx.Int32(idx_k_dim) + tid)

            if wave < fx.Int32(KV):
                kvh = wave
                xv = _vload8(proj_ptr, fx.Int32(12288) + kvh * fx.Int32(D) + seg * fx.Int32(8))
                wv = _vload8(kn_ptr, seg * fx.Int32(8))
                ss = fx.Float32(0.0)
                for i in range_constexpr(8):
                    ss = ss + xv[i] * xv[i]
                for o in (16, 8, 4, 2, 1):
                    ss = ss + fx.gpu.shuffle_xor(fx.Float32(ss), fx.Int32(o), fx.Int32(_WAVE))
                rstd = fmath.rsqrt(ss / fx.Float32(float(D)) + eps, fastmath="fast")
                n = [(xv[i] * rstd * (fx.Float32(1.0) + wv[i])).to(elem).to(fx.Float32) for i in range_constexpr(8)]
                pn = [fx.gpu.shuffle_xor(fx.Float32(n[i]), fx.Int32(pair_xor), fx.Int32(_WAVE)) for i in range_constexpr(8)]
                cv = Vec(fx.make_view(fx.add_offset(c_ptr, tbase), fx.make_layout(8, 1)).load())
                sv = Vec(fx.make_view(fx.add_offset(s_ptr, tbase), fx.make_layout(8, 1)).load())
                for i in range_constexpr(8):
                    c = fx.Float32(cv[i])
                    s = fx.Float32(sv[i])
                    r = lo.select(n[i] * c - pn[i] * s, n[i] * c + pn[i] * s)
                    val_bf = in_rot.select(r, n[i]).to(elem).to(fx.Float32)
                    fx.memref_store(val_bf, k_s, kvh * fx.Int32(D) + seg * fx.Int32(8) + fx.Int32(i))

            gpu.barrier()

            if tid < fx.Int32(total_blocks16):
                grp = tid // fx.Int32(16)
                blk16 = tid % fx.Int32(16)
                is_v = grp // fx.Int32(KV)
                kvh = grp % fx.Int32(KV)

                src_base = kvh * fx.Int32(D) + blk16 * fx.Int32(16)
                vals = []
                bmax = fx.Float32(0.0)
                for i in range_constexpr(16):
                    kv_val = fx.Float32(k_s[src_base + fx.Int32(i)])
                    vv_val = fx.Float32(fx.ptr_load(proj_ptr + fx.Int32(12800) + src_base + fx.Int32(i)))
                    v = (is_v == fx.Int32(0)).select(kv_val, vv_val)
                    vals.append(v)
                    bmax = fx.max(bmax, fx.absf(v))

                vmax = bmax
                for o in (1, 2, 4, 8):
                    vmax = fx.max(vmax, fx.gpu.shuffle_xor(fx.Float32(vmax), fx.Int32(o), fx.Int32(32)))

                s2 = (vmax > fx.Float32(0.0)).select(vmax / fx.Float32(6.0 * 240.0), fx.Float32(1.0))
                target_s = (bmax > fx.Float32(0.0)).select(bmax / (fx.Float32(6.0) * s2), fx.Float32(0.0))
                sb = _f32_to_e4m3_byte(target_s)
                deq_scale = _e4m3_to_f32(sb) * s2
                inv_scale = (deq_scale > fx.Float32(0.0)).select(fx.Float32(1.0) / deq_scale, fx.Float32(0.0))

                w0 = fx.Int32(0)
                w1 = fx.Int32(0)
                for i in range_constexpr(8):
                    n0 = _f32_to_e2m1_nib(vals[i], inv_scale)
                    n1 = _f32_to_e2m1_nib(vals[8 + i], inv_scale)
                    w0 = w0 | (n0 << fx.Int32(i * 4))
                    w1 = w1 | (n1 << fx.Int32(i * 4))

                in_prefix = pos < vram_prefix
                vram_slot = in_prefix.select(pos, vram_prefix + (pos & fx.Int32(4095)))
                vec_byte = (fx.Int64(vram_slot) * fx.Int64(KV) + fx.Int64(kvh)) * fx.Int64(160)
                kc_i32 = _ptr(K_CACHE, fx.Int32)
                vc_i32 = _ptr(V_CACHE, fx.Int32)
                kc_i8 = _ptr(K_CACHE, fx.Int8)
                vc_i8 = _ptr(V_CACHE, fx.Int8)
                kc_f32 = _ptr(K_CACHE, fx.Float32)
                vc_f32 = _ptr(V_CACHE, fx.Float32)

                b16_8 = fx.Int64(blk16 * fx.Int32(8))
                w0_idx = (vec_byte + b16_8) >> fx.Int64(2)
                w1_idx = (vec_byte + b16_8 + fx.Int64(4)) >> fx.Int64(2)
                sb_idx = vec_byte + fx.Int64(128) + fx.Int64(blk16)
                s2_idx = (vec_byte + fx.Int64(144)) >> fx.Int64(2)

                if is_v == fx.Int32(0):
                    fx.ptr_store(w0, kc_i32 + w0_idx)
                    fx.ptr_store(w1, kc_i32 + w1_idx)
                    fx.ptr_store(sb.to(fx.Int8), kc_i8 + sb_idx)
                    if blk16 == fx.Int32(0):
                        fx.ptr_store(s2, kc_f32 + s2_idx)
                else:
                    fx.ptr_store(w0, vc_i32 + w0_idx)
                    fx.ptr_store(w1, vc_i32 + w1_idx)
                    fx.ptr_store(sb.to(fx.Int8), vc_i8 + sb_idx)
                    if blk16 == fx.Int32(0):
                        fx.ptr_store(s2, vc_f32 + s2_idx)

                if pos >= vram_prefix:
                    host_slot = pos - vram_prefix
                    h_base = (fx.Int64(host_slot) * fx.Int64(KV) + fx.Int64(kvh)) * fx.Int64(160)
                    if is_v == fx.Int32(0):
                        hv = h_base + kc_host_delta
                        fx.ptr_store(w0, kc_i32 + ((hv + b16_8) >> fx.Int64(2)))
                        fx.ptr_store(w1, kc_i32 + ((hv + b16_8 + fx.Int64(4)) >> fx.Int64(2)))
                        fx.ptr_store(sb.to(fx.Int8), kc_i8 + (hv + fx.Int64(128) + fx.Int64(blk16)))
                        if blk16 == fx.Int32(0):
                            fx.ptr_store(s2, kc_f32 + ((hv + fx.Int64(144)) >> fx.Int64(2)))
                    else:
                        hv = h_base + vc_host_delta
                        fx.ptr_store(w0, vc_i32 + ((hv + b16_8) >> fx.Int64(2)))
                        fx.ptr_store(w1, vc_i32 + ((hv + b16_8 + fx.Int64(4)) >> fx.Int64(2)))
                        fx.ptr_store(sb.to(fx.Int8), vc_i8 + (hv + fx.Int64(128) + fx.Int64(blk16)))
                        if blk16 == fx.Int32(0):
                            fx.ptr_store(s2, vc_f32 + ((hv + fx.Int64(144)) >> fx.Int64(2)))

    @flyc.jit
    def launch(
        PROJ: fx.Pointer,
        Q_NORM: fx.Pointer,
        K_NORM: fx.Pointer,
        COS: fx.Pointer,
        SIN: fx.Pointer,
        QG_OUT: fx.Pointer,
        Q_OUT: fx.Pointer,
        IDX_QK_OUT: fx.Pointer,
        K_CACHE: fx.Pointer,
        V_CACHE: fx.Pointer,
        RAW_K_CACHE: fx.Pointer,
        kc_host_delta: fx.Int64,
        vc_host_delta: fx.Int64,
        pos0: fx.Int32,
        M: fx.Int32,
        vram_prefix: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        qsa_prep_and_cache_write_batch_nvfp4_kernel(
            PROJ,
            Q_NORM,
            K_NORM,
            COS,
            SIN,
            QG_OUT,
            Q_OUT,
            IDX_QK_OUT,
            K_CACHE,
            V_CACHE,
            RAW_K_CACHE,
            kc_host_delta,
            vc_host_delta,
            pos0,
            M,
            vram_prefix,
            eps,
        ).launch(grid=(4, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    qsa_prep_and_cache_write_batch_nvfp4_kernel.__name__ = f"qsa_prep_and_cache_write_batch_nvfp4_h{H}_kv{KV}_d{D}"
    return launch


# ---------------------------------------------------------------------------
# P3-2: Fast-Path + VGPR Register-Cached QSA Top-K Block Selector
# ---------------------------------------------------------------------------
def build_qsa_topk_blocks_fast_module(topk: int = 512, ratio: int = 4) -> Callable[..., None]:
    """Selects the top-`topk` block indices from SCORES[t_rel * score_stride .. + nb).
    - Fast path 1 (nb <= topk): 1-pass direct identity output with 0 score reads and 0 barriers.
    - Fast path 2 (topk < nb <= 8192, up to 32K context): loads 16 scores/thread into VGPRs ONCE,
      running the 31-bit threshold search, count, and scatter purely from registers (33x fewer global reads).
    - General path (nb > 8192, 32K..256K+ context): full multi-chunk global loop.
    Grid: (M_toks, 1, 1), Block: (512, 1, 1).
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    _SEL_THREADS = 512
    _SEL_WAVES = _SEL_THREADS // _WAVE
    _REG_CHUNKS = 16  # 16 * 512 = 8192 blocks = 32,768 tokens

    @fx.struct
    class SharedStorage:
        cnt0: fx.Array[fx.Int32, _SEL_WAVES * 3, 16]

    @flyc.kernel(known_block_size=[_SEL_THREADS, 1, 1])
    def qsa_topk_blocks_fast_kernel(
        SCORES: fx.Pointer,
        SEL: fx.Pointer,
        pos: fx.Int32,
        score_stride: fx.Int32,
    ) -> None:
        t_rel = fx.Int32(gpu.block_idx.x)
        pos_t = pos + t_rel
        scores_ptr = _ptr(SCORES, fx.Int32) + fx.Int64(t_rel) * fx.Int64(score_stride)
        sel_ptr = _ptr(SEL, fx.Int32) + t_rel * fx.Int32(topk)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        cnt_s = lds.cnt0.view(fx.make_layout(_SEL_WAVES * 3, 1))

        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        nb = (pos_t + fx.Int32(1)) // fx.Int32(ratio)
        k_eff = fx.min(nb, fx.Int32(topk))

        if nb <= fx.Int32(topk):
            if tid < fx.Int32(topk):
                fx.ptr_store((tid < nb).select(tid, fx.Int32(-1)), sel_ptr + tid)
        elif nb <= fx.Int32(_SEL_THREADS * _REG_CHUNKS):
            s_reg = []
            for c in range_constexpr(_REG_CHUNKS):
                bi_c = tid + fx.Int32(c * _SEL_THREADS)
                bi_safe = (bi_c < nb).select(bi_c, fx.Int32(0))
                s_reg.append((bi_c < nb).select(fx.Int32(fx.ptr_load(scores_ptr + bi_safe)), fx.Int32(-1)))

            # 16-pass 4-ary radix threshold search (step 0 tests bit 30; steps 1..15 test 2 bits each: 29..0)
            thr_init = [fx.Int32(0), fx.Int32(0)]
            thr_st = thr_init
            for i_step, st in range(fx.Int32(0), fx.Int32(16), fx.Int32(1), init=thr_init):
                cur_thr = st[0]
                is_s0 = fx.Int32(i_step) == fx.Int32(0)
                b_lo = is_s0.select(fx.Int32(30), fx.Int32(30) - fx.Int32(i_step) * fx.Int32(2))
                cand1 = cur_thr | (fx.Int32(1) << b_lo)
                cand2 = is_s0.select(cand1, cur_thr | (fx.Int32(2) << b_lo))
                cand3 = is_s0.select(cand1, cur_thr | (fx.Int32(3) << b_lo))
                v1 = fx.Int32(0)
                v2 = fx.Int32(0)
                v3 = fx.Int32(0)
                for c in range_constexpr(_REG_CHUNKS):
                    sc_val = s_reg[c]
                    v1 = v1 + (sc_val >= cand1).select(fx.Int32(1), fx.Int32(0))
                    v2 = v2 + (sc_val >= cand2).select(fx.Int32(1), fx.Int32(0))
                    v3 = v3 + (sc_val >= cand3).select(fx.Int32(1), fx.Int32(0))
                v_pack = v1 | (v2 << fx.Int32(10)) | (v3 << fx.Int32(20))
                for o in (1, 2, 4, 8, 16):
                    v_pack = v_pack + fx.gpu.shuffle_xor(fx.Int32(v_pack), fx.Int32(o), fx.Int32(_WAVE))
                if lane == fx.Int32(0):
                    fx.memref_store(v_pack & fx.Int32(0x3FF), cnt_s, wave * fx.Int32(3))
                    fx.memref_store((v_pack >> fx.Int32(10)) & fx.Int32(0x3FF), cnt_s, wave * fx.Int32(3) + fx.Int32(1))
                    fx.memref_store((v_pack >> fx.Int32(20)) & fx.Int32(0x3FF), cnt_s, wave * fx.Int32(3) + fx.Int32(2))
                gpu.barrier()
                tot1 = fx.Int32(0)
                tot2 = fx.Int32(0)
                tot3 = fx.Int32(0)
                for w in range_constexpr(_SEL_WAVES):
                    tot1 = tot1 + fx.Int32(cnt_s[w * 3])
                    tot2 = tot2 + fx.Int32(cnt_s[w * 3 + 1])
                    tot3 = tot3 + fx.Int32(cnt_s[w * 3 + 2])
                gpu.barrier()
                next_thr = (tot3 >= k_eff).select(
                    cand3,
                    (tot2 >= k_eff).select(cand2, (tot1 >= k_eff).select(cand1, cur_thr)),
                )
                thr_st = yield [next_thr, fx.Int32(0)]
            thr = thr_st[0]

            ng = fx.Int32(0)
            ne = fx.Int32(0)
            for c in range_constexpr(_REG_CHUNKS):
                sb = s_reg[c]
                ng = ng + (sb > thr).select(fx.Int32(1), fx.Int32(0))
                ne = ne + (sb == thr).select(fx.Int32(1), fx.Int32(0))
            # Low 10 bits hold ng (strictly < topk = 512 across the entire block);
            # bits 10..30 (21 positive bits) hold ne up to 2,097,151 blocks (> 8.38M tokens).
            packed = ng | (ne << fx.Int32(10))
            incl = packed
            for o in (1, 2, 4, 8, 16):
                up = fx.gpu.shuffle_up(fx.Int32(incl), fx.Int32(o), fx.Int32(_WAVE))
                incl = incl + (lane >= fx.Int32(o)).select(up, fx.Int32(0))
            if lane == fx.Int32(_WAVE - 1):
                fx.memref_store(incl, cnt_s, wave)
            gpu.barrier()
            before = fx.Int32(0)
            total = fx.Int32(0)
            for w in range_constexpr(_SEL_WAVES):
                cw = fx.Int32(cnt_s[w])
                before = before + (fx.Int32(w) < wave).select(cw, fx.Int32(0))
                total = total + cw
            excl = before + incl - packed
            gp = excl & fx.Int32(0x3FF)
            ep = (total & fx.Int32(0x3FF)) + (excl >> fx.Int32(10))

            for c in range_constexpr(_REG_CHUNKS):
                bi_c = tid + fx.Int32(c * _SEL_THREADS)
                sb = s_reg[c]
                is_g = sb > thr
                is_e = sb == thr
                dst = is_g.select(gp, ep)
                if (is_g | is_e) & (dst < k_eff):
                    fx.ptr_store(bi_c, sel_ptr + dst)
                gp = gp + is_g.select(fx.Int32(1), fx.Int32(0))
                ep = ep + is_e.select(fx.Int32(1), fx.Int32(0))
        else:
            thr_init = [fx.Int32(0), fx.Int32(0)]
            thr_st = thr_init
            for i_step, st in range(fx.Int32(0), fx.Int32(16), fx.Int32(1), init=thr_init):
                cur_thr = st[0]
                is_s0 = fx.Int32(i_step) == fx.Int32(0)
                b_lo = is_s0.select(fx.Int32(30), fx.Int32(30) - fx.Int32(i_step) * fx.Int32(2))
                cand1 = cur_thr | (fx.Int32(1) << b_lo)
                cand2 = is_s0.select(cand1, cur_thr | (fx.Int32(2) << b_lo))
                cand3 = is_s0.select(cand1, cur_thr | (fx.Int32(3) << b_lo))
                c_init = [fx.Int32(0), fx.Int32(0), fx.Int32(0)]
                c_st = c_init
                for b, ca in range(tid, nb, fx.Int32(_SEL_THREADS), init=c_init):
                    bi = fx.Int32(b)
                    sb = fx.Int32(fx.ptr_load(scores_ptr + bi))
                    c_st = yield [
                        ca[0] + (sb >= cand1).select(fx.Int32(1), fx.Int32(0)),
                        ca[1] + (sb >= cand2).select(fx.Int32(1), fx.Int32(0)),
                        ca[2] + (sb >= cand3).select(fx.Int32(1), fx.Int32(0)),
                    ]
                v1 = c_st[0]
                v2 = c_st[1]
                v3 = c_st[2]
                for o in (1, 2, 4, 8, 16):
                    v1 = v1 + fx.gpu.shuffle_xor(fx.Int32(v1), fx.Int32(o), fx.Int32(_WAVE))
                    v2 = v2 + fx.gpu.shuffle_xor(fx.Int32(v2), fx.Int32(o), fx.Int32(_WAVE))
                    v3 = v3 + fx.gpu.shuffle_xor(fx.Int32(v3), fx.Int32(o), fx.Int32(_WAVE))
                if lane == fx.Int32(0):
                    fx.memref_store(v1, cnt_s, wave * fx.Int32(3))
                    fx.memref_store(v2, cnt_s, wave * fx.Int32(3) + fx.Int32(1))
                    fx.memref_store(v3, cnt_s, wave * fx.Int32(3) + fx.Int32(2))
                gpu.barrier()
                tot1 = fx.Int32(0)
                tot2 = fx.Int32(0)
                tot3 = fx.Int32(0)
                for w in range_constexpr(_SEL_WAVES):
                    tot1 = tot1 + fx.Int32(cnt_s[w * 3])
                    tot2 = tot2 + fx.Int32(cnt_s[w * 3 + 1])
                    tot3 = tot3 + fx.Int32(cnt_s[w * 3 + 2])
                gpu.barrier()
                next_thr = (tot3 >= k_eff).select(
                    cand3,
                    (tot2 >= k_eff).select(cand2, (tot1 >= k_eff).select(cand1, cur_thr)),
                )
                thr_st = yield [next_thr, fx.Int32(0)]
            thr = thr_st[0]

            ge_init = [fx.Int32(0), fx.Int32(0)]
            ge_st = ge_init
            for b, ge in range(tid, nb, fx.Int32(_SEL_THREADS), init=ge_init):
                bi = fx.Int32(b)
                sb = fx.Int32(fx.ptr_load(scores_ptr + bi))
                ge_st = yield [
                    ge[0] + (sb > thr).select(fx.Int32(1), fx.Int32(0)),
                    ge[1] + (sb == thr).select(fx.Int32(1), fx.Int32(0)),
                ]
            ng = ge_st[0]
            ne = ge_st[1]
            # Low 10 bits hold ng (< 512); bits 10..30 hold ne up to 2,097,151 blocks (> 8.38M tokens).
            packed = ng | (ne << fx.Int32(10))
            incl = packed
            for o in (1, 2, 4, 8, 16):
                up = fx.gpu.shuffle_up(fx.Int32(incl), fx.Int32(o), fx.Int32(_WAVE))
                incl = incl + (lane >= fx.Int32(o)).select(up, fx.Int32(0))
            if lane == fx.Int32(_WAVE - 1):
                fx.memref_store(incl, cnt_s, wave)
            gpu.barrier()
            before = fx.Int32(0)
            total = fx.Int32(0)
            for w in range_constexpr(_SEL_WAVES):
                cw = fx.Int32(cnt_s[w])
                before = before + (fx.Int32(w) < wave).select(cw, fx.Int32(0))
                total = total + cw
            excl = before + incl - packed
            g_pos = excl & fx.Int32(0x3FF)
            e_pos = (total & fx.Int32(0x3FF)) + (excl >> fx.Int32(10))

            pos_init = [g_pos, e_pos]
            pos_st = pos_init
            for b, pp in range(tid, nb, fx.Int32(_SEL_THREADS), init=pos_init):
                bi = fx.Int32(b)
                sb = fx.Int32(fx.ptr_load(scores_ptr + bi))
                is_g = sb > thr
                is_e = sb == thr
                gp = pp[0]
                ep = pp[1]
                dst = is_g.select(gp, ep)
                if (is_g | is_e) & (dst < k_eff):
                    fx.ptr_store(bi, sel_ptr + dst)
                pos_st = yield [
                    gp + is_g.select(fx.Int32(1), fx.Int32(0)),
                    ep + is_e.select(fx.Int32(1), fx.Int32(0)),
                ]

    @flyc.jit
    def launch(
        SCORES: fx.Pointer,
        SEL: fx.Pointer,
        pos: fx.Int32,
        score_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        qsa_topk_blocks_fast_kernel(SCORES, SEL, pos, score_stride).launch(
            grid=(1, 1, 1), block=(_SEL_THREADS, 1, 1), stream=stream
        )

    qsa_topk_blocks_fast_kernel.__name__ = f"qsa_topk_blocks_k{topk}_r{ratio}"
    return launch


# ---------------------------------------------------------------------------
# P3-3: Cooperative LDS-Staged Mega-mHC Premix (Decode & Batched Verify)
# ---------------------------------------------------------------------------
def build_hc_premix_decode_fast_module(
    D: int = 2560, n: int = 4, R: int = 320, combine: bool = True, update: bool = True
) -> Callable[..., None]:
    """Single-token (T=1, 4 waves/block = 128 threads, grid=(81/80, 1, 1)) Hyper-Connection premix
    with 1-barrier cooperative 4-wave preload of stream `s_w = wave` (packing BF16 `(xv, nv)` pairs
    into `40 KiB` `Int32[10240]` LDS + `rs_s[4]`), 0-ULP bit-exact `rstd` reduction order, and unrolled
    `range_constexpr(4) x range_constexpr(10)` WD row projection.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    assert D == 2560 and n == 4 and R == 320

    ND = n * D
    rows = R + (n if combine else 0)
    it_per_group = D // (_WAVE * 8)  # 10
    waves_per_block = 4
    threads = waves_per_block * _WAVE  # 128
    grid_x = (rows + waves_per_block - 1) // waves_per_block
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        pk: fx.Array[fx.Int32, ND, 16]
        rs: fx.Array[fx.Float32, 4, 16]

    def _vstore8(ptr, idx, vals):
        view = fx.make_view(fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(8, 1))
        view.store(fx.Vector.from_elements([fx.Float32(x).to(elem) for x in vals], elem))

    def _wsum(x):
        for o in (1, 2, 4, 8, 16):
            x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(o), fx.Int32(_WAVE))
        return x

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def hc_premix_decode_fast_kernel(
        X: fx.Pointer,
        OUT: fx.Pointer,
        INJP: fx.Pointer,
        XO: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        eps: fx.Float32,
    ) -> None:
        x_ptr = _ptr(X, elem)
        o_ptr = _ptr(OUT, elem)
        ip_ptr = _ptr(INJP, elem)
        xo_ptr = _ptr(XO, elem)
        nw_ptr = _ptr(NW, elem)
        wd_ptr = _ptr(WD, elem)
        h_ptr = _ptr(H, elem)
        inj_ptr = _ptr(INJ, elem)
        rs_ptr = _ptr(RSTD, fx.Float32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pk_s = lds.pk.view(fx.make_layout(ND, 1))
        rs_s = lds.rs.view(fx.make_layout(4, 1))

        tid = fx.Int32(gpu.thread_idx.x)
        blk = fx.Int32(gpu.block_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        row = blk * fx.Int32(waves_per_block) + wave
        live = row < fx.Int32(rows)
        rc = live.select(row, fx.Int32(0))

        if const_expr(combine):
            is_inj = rc >= fx.Int32(R)
            w_row = is_inj.select(rc - fx.Int32(R), rc)
            w_base = fx.Int64(w_row) * fx.Int64(ND)
            w_base = is_inj.select(w_base + fx.Int64(wi_delta), w_base)
        else:
            w_row = rc
            w_base = fx.Int64(rc) * fx.Int64(ND)

        # Wave `wave` (0..3) loads stream `s_w = wave` (2560 elements) into LDS
        s_w = wave
        if const_expr(update):
            injp_w = _bf(fx.Float32(fx.ptr_load(ip_ptr + s_w)))
        is_blk0 = blk == fx.Int32(0)

        sq_w = fx.Float32(0.0)
        for j in range_constexpr(it_per_group):
            d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
            k = s_w * fx.Int32(D) + d_off
            nv = _vload8(nw_ptr, k)
            xv = _vload8(x_ptr, k)
            if const_expr(update):
                ov = _vload8(o_ptr, d_off)
                xv = [_bf(xv[e] + _bf(ov[e] * injp_w)) for e in range_constexpr(8)]
                if is_blk0:
                    _vstore8(xo_ptr, k, xv)
            lds_base = s_w * fx.Int32(D) + fx.Int32(j * (8 * _WAVE)) + lane
            for e in range_constexpr(8):
                sq_w = sq_w + xv[e] * xv[e]
                x_hi = xv[e].bitcast(fx.Int32) & fx.Int32(-65536)
                n_lo = (nv[e].bitcast(fx.Int32) >> fx.Int32(16)) & fx.Int32(0xFFFF)
                fx.memref_store(x_hi | n_lo, pk_s, lds_base + fx.Int32(e * _WAVE))

        rstd_w = fmath.rsqrt(_wsum(sq_w) * fx.Float32(1.0 / D) + eps, fastmath="fast")
        if lane == fx.Int32(0):
            fx.memref_store(rstd_w, rs_s, s_w)
            if is_blk0:
                fx.ptr_store(rstd_w, rs_ptr + s_w)

        gpu.barrier()

        inv_n = fx.Float32(1.0 / n)
        acc = fx.Float32(0.0)
        for s in range_constexpr(n):
            rstd = fx.Float32(rs_s[fx.Int32(s)])
            dot = fx.Float32(0.0)
            for j in range_constexpr(it_per_group):
                d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
                k = fx.Int32(s * D) + d_off
                wv = _vload8(wd_ptr, w_base + fx.Int64(k))
                lds_base = fx.Int32(s * D + j * (8 * _WAVE)) + lane
                for e in range_constexpr(8):
                    pk_val = fx.Int32(pk_s[lds_base + fx.Int32(e * _WAVE)])
                    xv_e = (pk_val & fx.Int32(-65536)).bitcast(fx.Float32)
                    nv_e = (pk_val << fx.Int32(16)).bitcast(fx.Float32)
                    np1_e = fx.Float32(1.0) + nv_e
                    dot = dot + xv_e * (np1_e * wv[e])
            acc = acc + rstd * _wsum(dot)

        if lane == fx.Int32(0):
            if live:
                y = _bf(_bf(acc) * inv_n)
                if const_expr(combine):
                    if is_inj:
                        fx.ptr_store(fx.Float32(fx.Float32(2.0) * _bf(_sigmoid_f32(y))).to(elem), inj_ptr + w_row)
                    else:
                        fx.ptr_store(fx.Float32(y * _sigmoid_f32(y)).to(elem), h_ptr + w_row)
                else:
                    fx.ptr_store(fx.Float32(y * _sigmoid_f32(y)).to(elem), h_ptr + w_row)

    @flyc.jit
    def launch(
        X: fx.Pointer,
        OUT: fx.Pointer,
        INJP: fx.Pointer,
        XO: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        hc_premix_decode_fast_kernel(X, OUT, INJP, XO, NW, WD, H, INJ, RSTD, wi_delta, eps).launch(
            grid=(grid_x, 1, 1), block=(threads, 1, 1), stream=stream
        )

    hc_premix_decode_fast_kernel.__name__ = f"hc_premix_decode_d{D}_n{n}_r{R}_c{int(combine)}_u{int(update)}"
    return launch


def build_hc_premix_fast_module(
    D: int = 2560, n: int = 4, R: int = 320, combine: bool = True, update: bool = True
) -> Callable[..., None]:
    """Batched (TT=1 per block_idx.y, 4 waves/block = 128 threads, grid=(81/80, M, 1)) Hyper-Connection
    premix with 1-barrier cooperative 4-wave preload of stream `s_w = wave` into `20 KiB` packed-BF16
    LDS (`Int32[5120]` + `rs_s[4]`, enabling 6 workgroups/WGP = 192 concurrent workgroups), 0-ULP
    bit-exact `rstd` reduction order, and unrolled `range_constexpr(4) x range_constexpr(10)` WD projection.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    assert D == 2560 and n == 4 and R == 320

    ND = n * D
    rows = R + (n if combine else 0)
    it_per_group = D // (_WAVE * 8)  # 10
    waves_per_block = 4
    threads = waves_per_block * _WAVE  # 128
    grid_x = (rows + waves_per_block - 1) // waves_per_block
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        xs: fx.Array[fx.Int32, ND // 2, 16]
        rs: fx.Array[fx.Float32, 4, 16]

    def _vstore8(ptr, idx, vals):
        view = fx.make_view(fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(8, 1))
        view.store(fx.Vector.from_elements([fx.Float32(x).to(elem) for x in vals], elem))

    def _wsum(x):
        for o in (1, 2, 4, 8, 16):
            x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(o), fx.Int32(_WAVE))
        return x

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def hc_premix_fast_kernel(
        X: fx.Pointer,
        OUT: fx.Pointer,
        INJP: fx.Pointer,
        XO: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        num_tokens: fx.Int32,
        eps: fx.Float32,
    ) -> None:
        x_ptr = _ptr(X, elem)
        o_ptr = _ptr(OUT, elem)
        ip_ptr = _ptr(INJP, elem)
        xo_ptr = _ptr(XO, elem)
        nw_ptr = _ptr(NW, elem)
        wd_ptr = _ptr(WD, elem)
        h_ptr = _ptr(H, elem)
        inj_ptr = _ptr(INJ, elem)
        rs_ptr = _ptr(RSTD, fx.Float32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        xs_s = lds.xs.view(fx.make_layout(ND // 2, 1))
        rs_s = lds.rs.view(fx.make_layout(4, 1))

        T = fx.Int32(num_tokens)
        tid = fx.Int32(gpu.thread_idx.x)
        blk = fx.Int32(gpu.block_idx.x)
        t0 = fx.Int32(gpu.block_idx.y)
        t_live = t0 < T
        tok = t_live.select(t0, T - fx.Int32(1))

        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        row = blk * fx.Int32(waves_per_block) + wave
        live = row < fx.Int32(rows)
        rc = live.select(row, fx.Int32(0))

        if const_expr(combine):
            is_inj = rc >= fx.Int32(R)
            w_row = is_inj.select(rc - fx.Int32(R), rc)
            w_base = fx.Int64(w_row) * fx.Int64(ND)
            w_base = is_inj.select(w_base + fx.Int64(wi_delta), w_base)
        else:
            w_row = rc
            w_base = fx.Int64(rc) * fx.Int64(ND)

        # Wave `wave` (0..3) loads stream `s_w = wave` and computes its exact rstd_w
        s_w = wave
        if const_expr(update):
            injp_w = _bf(fx.Float32(fx.ptr_load(ip_ptr + tok * fx.Int32(n) + s_w)))
        is_blk0 = (blk == fx.Int32(0)) & t_live
        tok_nd = tok * fx.Int32(ND)
        tok_d = tok * fx.Int32(D)

        sq_w = fx.Float32(0.0)
        for j in range_constexpr(it_per_group):
            d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
            k = s_w * fx.Int32(D) + d_off
            xv = _vload8(x_ptr, tok_nd + k)
            if const_expr(update):
                ov = _vload8(o_ptr, tok_d + d_off)
                xv = [_bf(xv[e] + _bf(ov[e] * injp_w)) for e in range_constexpr(8)]
                if is_blk0:
                    _vstore8(xo_ptr, tok_nd + k, xv)
            for e in range_constexpr(8):
                sq_w = sq_w + xv[e] * xv[e]
            lds_base = s_w * fx.Int32(D // 2) + fx.Int32(j * (4 * _WAVE)) + lane
            for p_idx in range_constexpr(4):
                x0 = xv[2 * p_idx].bitcast(fx.Int32) & fx.Int32(-65536)
                x1 = (xv[2 * p_idx + 1].bitcast(fx.Int32) >> fx.Int32(16)) & fx.Int32(0xFFFF)
                fx.memref_store(x0 | x1, xs_s, lds_base + fx.Int32(p_idx * _WAVE))

        rstd_w = fmath.rsqrt(_wsum(sq_w) * fx.Float32(1.0 / D) + eps, fastmath="fast")
        if lane == fx.Int32(0):
            fx.memref_store(rstd_w, rs_s, s_w)
            if is_blk0:
                fx.ptr_store(rstd_w, rs_ptr + tok * fx.Int32(n) + s_w)

        gpu.barrier()

        inv_n = fx.Float32(1.0 / n)
        acc = fx.Float32(0.0)
        for s in range_constexpr(n):
            rstd = fx.Float32(rs_s[fx.Int32(s)])
            dot = fx.Float32(0.0)
            for j in range_constexpr(it_per_group):
                d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
                k = fx.Int32(s * D) + d_off
                wv = _vload8(wd_ptr, w_base + fx.Int64(k))
                nv = _vload8(nw_ptr, k)
                lds_base = fx.Int32(s * (D // 2) + j * (4 * _WAVE)) + lane
                for p_idx in range_constexpr(4):
                    pk_x = fx.Int32(xs_s[lds_base + fx.Int32(p_idx * _WAVE)])
                    xv0 = (pk_x & fx.Int32(-65536)).bitcast(fx.Float32)
                    xv1 = (pk_x << fx.Int32(16)).bitcast(fx.Float32)
                    dot = dot + xv0 * ((fx.Float32(1.0) + nv[2 * p_idx]) * wv[2 * p_idx])
                    dot = dot + xv1 * ((fx.Float32(1.0) + nv[2 * p_idx + 1]) * wv[2 * p_idx + 1])
            acc = acc + rstd * _wsum(dot)

        y = _bf(_bf(acc) * inv_n)
        ok = live & (lane == fx.Int32(0)) & t_live
        if const_expr(combine):
            if ok & is_inj:
                fx.ptr_store(fx.Float32(fx.Float32(2.0) * _bf(_sigmoid_f32(y))).to(elem),
                             inj_ptr + tok * fx.Int32(n) + w_row)
            if ok & (rc < fx.Int32(R)):
                fx.ptr_store(fx.Float32(y * _sigmoid_f32(y)).to(elem), h_ptr + tok * fx.Int32(R) + w_row)
        else:
            if ok:
                fx.ptr_store(fx.Float32(y * _sigmoid_f32(y)).to(elem), h_ptr + tok * fx.Int32(R) + w_row)

    @flyc.jit
    def launch(
        X: fx.Pointer,
        OUT: fx.Pointer,
        INJP: fx.Pointer,
        XO: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        num_tokens: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        hc_premix_fast_kernel(X, OUT, INJP, XO, NW, WD, H, INJ, RSTD, wi_delta, num_tokens, eps).launch(
            grid=(grid_x, num_tokens, 1), block=(threads, 1, 1), stream=stream
        )

    hc_premix_fast_kernel.__name__ = f"hyper_connection_gfx120x_premix_d{D}_n{n}_r{R}_c{int(combine)}_u{int(update)}_t1"
    return launch


# ---------------------------------------------------------------------------
# P3-4: Fused PLE FP8 Dequant + 12800x2560 K/V Projection GEMV
# ---------------------------------------------------------------------------
def build_ple_dequant_gemv_12800_2560_module(N: int = 12800, K: int = 2560) -> Callable[..., None]:
    """Fuses `ple_embed_dequant` (`bf16(fp8_e4m3(rows_u8) * scale)`) in registers into `bf16_gemv_12800_2560`,
    eliminating the separate `ple_embed_dequant` kernel launch and the `d_ple_embed_` global memory round-trip.
    Grid: (800, 1, 1), Block: (256, 1, 1).
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    from flydsl.expr.typing import T as _IRT

    NN, KK = int(N), int(K)
    vec_w = 8
    step_elems = _WAVE * vec_w  # 256
    steps = KK // step_elems    # 10
    waves_per_block = _THREADS // _WAVE  # 8
    rows_per_wave = 2
    rows_per_block = waves_per_block * rows_per_wave  # 16
    grid_x = (NN + rows_per_block - 1) // rows_per_block  # 800
    elem = fx.BFloat16

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def ple_dequant_gemv_12800_2560_kernel(
        ROWS_U8: fx.Pointer,
        SCALE: fx.Pointer,
        W: fx.Pointer,
        Y: fx.Pointer,
    ) -> None:
        r_ptr = _ptr(ROWS_U8, fx.Int8)
        sc_ptr = _ptr(SCALE, elem)
        w_ptr = _ptr(W, elem)
        y_ptr = _ptr(Y, elem)

        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        r0 = fx.Int32(gpu.block_idx.x) * fx.Int32(rows_per_block) + wave * fx.Int32(2)
        r1 = r0 + fx.Int32(1)
        b0 = fx.Int64(r0) * fx.Int64(KK)
        b1 = fx.Int64(r1) * fx.Int64(KK)

        sc = fx.Float32(fx.ptr_load(sc_ptr))
        acc0 = fx.Float32(0.0)
        acc1 = fx.Float32(0.0)
        for s in range_constexpr(steps):
            k0 = fx.Int32(s * step_elems) + lane * fx.Int32(vec_w)
            rv = Vec(fx.make_view(fx.add_offset(r_ptr, fx.Int64(k0)), fx.make_layout(vec_w, 1)).load()).bitcast(fx.Int32)
            xv = []
            for pi in range_constexpr(vec_w // 2):
                pair = Vec(fx.rocdl.cvt_pk_f32_fp8(_IRT.f32x2, rv[pi // 2], bool(pi & 1)))
                xv.append(_bf(pair[0] * sc))
                xv.append(_bf(pair[1] * sc))
            w0 = _vload8(w_ptr, b0 + fx.Int64(k0))
            w1 = _vload8(w_ptr, b1 + fx.Int64(k0))
            for i in range_constexpr(vec_w):
                acc0 = acc0 + xv[i] * w0[i]
                acc1 = acc1 + xv[i] * w1[i]
        for o in (16, 8, 4, 2, 1):
            acc0 = acc0 + fx.gpu.shuffle_xor(fx.Float32(acc0), fx.Int32(o), fx.Int32(_WAVE))
            acc1 = acc1 + fx.gpu.shuffle_xor(fx.Float32(acc1), fx.Int32(o), fx.Int32(_WAVE))
        if lane == fx.Int32(0):
            fx.ptr_store(acc0.to(elem), y_ptr + r0)
            fx.ptr_store(acc1.to(elem), y_ptr + r1)

    @flyc.jit
    def launch(
        ROWS_U8: fx.Pointer,
        SCALE: fx.Pointer,
        W: fx.Pointer,
        Y: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        ple_dequant_gemv_12800_2560_kernel(ROWS_U8, SCALE, W, Y).launch(
            grid=(grid_x, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    ple_dequant_gemv_12800_2560_kernel.__name__ = f"ple_dequant_gemv_n{NN}_k{KK}"
    return launch


def build_ple_dequant_gemv_m4_12800_2560_module(
    N_k: int = 10240, N_v: int = 2560, K: int = 2560
) -> Callable[..., None]:
    """Fused M<=4 batched PLE FP8 dequant + 12800x2560 K/V projection GEMV.
    Waves 0..3 cooperatively dequantize token `r = wave` (`2560` FP8 -> BF16) into `20 KiB` LDS once
    per block before a single barrier, then all 8 waves project 1 row of `W` (`[12800, 2560]`) across
    all 4 tokens and write `row < 10240` to `OUT_K` (stride 10240) and `row >= 10240` to `OUT_V` (stride 2560).
    Grid: (1600, 1, 1), Block: (256, 1, 1).
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    from flydsl.expr.typing import T as _IRT

    NN = int(N_k) + int(N_v)  # 12800
    KK = int(K)               # 2560
    vec_w = 8
    step_elems = _WAVE * vec_w  # 256
    steps = KK // step_elems    # 10
    waves_per_block = _THREADS // _WAVE  # 8
    grid_x = NN // waves_per_block       # 1600
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        xs: fx.Array[fx.Int32, 4 * (KK // 2), 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def ple_dequant_gemv_m4_12800_2560_kernel(
        ROWS_U8: fx.Pointer,
        SCALE: fx.Pointer,
        W: fx.Pointer,
        OUT_K: fx.Pointer,
        OUT_V: fx.Pointer,
        M: fx.Int32,
    ) -> None:
        r_ptr = _ptr(ROWS_U8, fx.Int8)
        sc_ptr = _ptr(SCALE, elem)
        w_ptr = _ptr(W, elem)
        yk_ptr = _ptr(OUT_K, elem)
        yv_ptr = _ptr(OUT_V, elem)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        xs_s = lds.xs.view(fx.make_layout(4 * (KK // 2), 1))

        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        row = fx.Int32(gpu.block_idx.x) * fx.Int32(waves_per_block) + wave
        base = fx.Int64(row) * fx.Int64(KK)

        if wave < fx.Int32(4):
            r_clamp = (wave < M).select(wave, fx.Int32(0))
            r_off = r_clamp * fx.Int32(KK)
            sc = fx.Float32(fx.ptr_load(sc_ptr))
            for s in range_constexpr(steps):
                k0 = fx.Int32(s * step_elems) + lane * fx.Int32(vec_w)
                rv = Vec(fx.make_view(fx.add_offset(r_ptr, fx.Int64(r_off + k0)), fx.make_layout(vec_w, 1)).load()).bitcast(fx.Int32)
                lds_base = wave * fx.Int32(KK // 2) + fx.Int32(s * (4 * _WAVE)) + lane
                for pi in range_constexpr(vec_w // 2):
                    pair = Vec(fx.rocdl.cvt_pk_f32_fp8(_IRT.f32x2, rv[pi // 2], bool(pi & 1)))
                    x0 = _bf(pair[0] * sc).bitcast(fx.Int32) & fx.Int32(-65536)
                    x1 = (_bf(pair[1] * sc).bitcast(fx.Int32) >> fx.Int32(16)) & fx.Int32(0xFFFF)
                    fx.memref_store(x0 | x1, xs_s, lds_base + fx.Int32(pi * _WAVE))
        gpu.barrier()

        acc0 = fx.Float32(0.0)
        acc1 = fx.Float32(0.0)
        acc2 = fx.Float32(0.0)
        acc3 = fx.Float32(0.0)
        for s in range_constexpr(steps):
            k0 = fx.Int32(s * step_elems) + lane * fx.Int32(vec_w)
            wv = _vload8(w_ptr, base + fx.Int64(k0))
            lds_b = fx.Int32(s * (4 * _WAVE)) + lane
            for pi in range_constexpr(4):
                w0 = wv[2 * pi]
                w1 = wv[2 * pi + 1]
                p0 = fx.Int32(xs_s[lds_b + fx.Int32(pi * _WAVE)])
                p1 = fx.Int32(xs_s[fx.Int32(KK // 2) + lds_b + fx.Int32(pi * _WAVE)])
                p2 = fx.Int32(xs_s[fx.Int32(2 * (KK // 2)) + lds_b + fx.Int32(pi * _WAVE)])
                p3 = fx.Int32(xs_s[fx.Int32(3 * (KK // 2)) + lds_b + fx.Int32(pi * _WAVE)])
                acc0 = acc0 + (p0 & fx.Int32(-65536)).bitcast(fx.Float32) * w0 + (p0 << fx.Int32(16)).bitcast(fx.Float32) * w1
                acc1 = acc1 + (p1 & fx.Int32(-65536)).bitcast(fx.Float32) * w0 + (p1 << fx.Int32(16)).bitcast(fx.Float32) * w1
                acc2 = acc2 + (p2 & fx.Int32(-65536)).bitcast(fx.Float32) * w0 + (p2 << fx.Int32(16)).bitcast(fx.Float32) * w1
                acc3 = acc3 + (p3 & fx.Int32(-65536)).bitcast(fx.Float32) * w0 + (p3 << fx.Int32(16)).bitcast(fx.Float32) * w1
        for o in (16, 8, 4, 2, 1):
            acc0 = acc0 + fx.gpu.shuffle_xor(fx.Float32(acc0), fx.Int32(o), fx.Int32(_WAVE))
            acc1 = acc1 + fx.gpu.shuffle_xor(fx.Float32(acc1), fx.Int32(o), fx.Int32(_WAVE))
            acc2 = acc2 + fx.gpu.shuffle_xor(fx.Float32(acc2), fx.Int32(o), fx.Int32(_WAVE))
            acc3 = acc3 + fx.gpu.shuffle_xor(fx.Float32(acc3), fx.Int32(o), fx.Int32(_WAVE))
        if lane == fx.Int32(0):
            is_k = row < fx.Int32(N_k)
            if is_k:
                r64 = fx.Int64(row)
                n64 = fx.Int64(N_k)
                fx.ptr_store(acc0.to(elem), yk_ptr + r64)
                if M > fx.Int32(1):
                    fx.ptr_store(acc1.to(elem), yk_ptr + n64 + r64)
                if M > fx.Int32(2):
                    fx.ptr_store(acc2.to(elem), yk_ptr + fx.Int64(2) * n64 + r64)
                if M > fx.Int32(3):
                    fx.ptr_store(acc3.to(elem), yk_ptr + fx.Int64(3) * n64 + r64)
            else:
                r64 = fx.Int64(row - fx.Int32(N_k))
                n64 = fx.Int64(N_v)
                fx.ptr_store(acc0.to(elem), yv_ptr + r64)
                if M > fx.Int32(1):
                    fx.ptr_store(acc1.to(elem), yv_ptr + n64 + r64)
                if M > fx.Int32(2):
                    fx.ptr_store(acc2.to(elem), yv_ptr + fx.Int64(2) * n64 + r64)
                if M > fx.Int32(3):
                    fx.ptr_store(acc3.to(elem), yv_ptr + fx.Int64(3) * n64 + r64)

    @flyc.jit
    def launch(
        ROWS_U8: fx.Pointer,
        SCALE: fx.Pointer,
        W: fx.Pointer,
        OUT_K: fx.Pointer,
        OUT_V: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        ple_dequant_gemv_m4_12800_2560_kernel(ROWS_U8, SCALE, W, OUT_K, OUT_V, M).launch(
            grid=(grid_x, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    ple_dequant_gemv_m4_12800_2560_kernel.__name__ = f"ple_dequant_gemv_m4_n{NN}_k{KK}"
    return launch


def build_embed_hc_premix_decode_l0_module(
    D: int = 2560, n: int = 4, R: int = 320
) -> Callable[..., None]:
    """Fused Layer-0 `embed_hc_gather` + `hc_premix_decode_c1_u0` (T=1, grid=(81, 1, 1), block=(128, 1, 1)).
    Reads `TABLE[IDS[0]]` (`2560` BF16 = `5 KiB`), writes the 4 streams of `X_OUT` on `blk == 0`,
    packs `(xv, nv)` into `40 KiB` LDS (`pk_s`), and computes `H`, `INJ`, and `RSTD` in a single launch.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    ND = n * D
    rows = R + n  # 324
    it_per_group = D // (_WAVE * 8)  # 10
    waves_per_block = 4
    threads = waves_per_block * _WAVE  # 128
    grid_x = (rows + waves_per_block - 1) // waves_per_block  # 81
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        pk: fx.Array[fx.Int32, ND, 16]
        rs: fx.Array[fx.Float32, 4, 16]

    def _vstore8(ptr, idx, vals):
        view = fx.make_view(fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(8, 1))
        view.store(fx.Vector.from_elements([fx.Float32(x).to(elem) for x in vals], elem))

    def _wsum(x):
        for o in (1, 2, 4, 8, 16):
            x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(o), fx.Int32(_WAVE))
        return x

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def embed_hc_premix_decode_l0_kernel(
        IDS: fx.Pointer,
        TABLE: fx.Pointer,
        X_OUT: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        eps: fx.Float32,
    ) -> None:
        id_ptr = _ptr(IDS, fx.Int32)
        tbl_ptr = _ptr(TABLE, elem)
        x_ptr = _ptr(X_OUT, elem)
        nw_ptr = _ptr(NW, elem)
        wd_ptr = _ptr(WD, elem)
        h_ptr = _ptr(H, elem)
        inj_ptr = _ptr(INJ, elem)
        rs_ptr = _ptr(RSTD, fx.Float32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pk_s = lds.pk.view(fx.make_layout(ND, 1))
        rs_s = lds.rs.view(fx.make_layout(4, 1))

        tid = fx.Int32(gpu.thread_idx.x)
        blk = fx.Int32(gpu.block_idx.x)
        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        row = blk * fx.Int32(waves_per_block) + wave
        live = row < fx.Int32(rows)
        rc = live.select(row, fx.Int32(0))

        is_inj = rc >= fx.Int32(R)
        w_row = is_inj.select(rc - fx.Int32(R), rc)
        w_base = fx.Int64(w_row) * fx.Int64(ND)
        w_base = is_inj.select(w_base + fx.Int64(wi_delta), w_base)

        tok_id = fx.Int64(fx.Int32(fx.ptr_load(id_ptr)))
        tbl_base = tok_id * fx.Int64(D)

        s_w = wave
        is_blk0 = blk == fx.Int32(0)

        sq_w = fx.Float32(0.0)
        for j in range_constexpr(it_per_group):
            d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
            k = s_w * fx.Int32(D) + d_off
            nv = _vload8(nw_ptr, k)
            xv = _vload8(tbl_ptr, tbl_base + fx.Int64(d_off))
            if is_blk0:
                _vstore8(x_ptr, k, xv)
            lds_base = s_w * fx.Int32(D) + fx.Int32(j * (8 * _WAVE)) + lane
            for e in range_constexpr(8):
                sq_w = sq_w + xv[e] * xv[e]
                x_hi = xv[e].bitcast(fx.Int32) & fx.Int32(-65536)
                n_lo = (nv[e].bitcast(fx.Int32) >> fx.Int32(16)) & fx.Int32(0xFFFF)
                fx.memref_store(x_hi | n_lo, pk_s, lds_base + fx.Int32(e * _WAVE))

        rstd_w = fmath.rsqrt(_wsum(sq_w) * fx.Float32(1.0 / D) + eps, fastmath="fast")
        if lane == fx.Int32(0):
            fx.memref_store(rstd_w, rs_s, s_w)
            if is_blk0:
                fx.ptr_store(rstd_w, rs_ptr + s_w)

        gpu.barrier()

        inv_n = fx.Float32(1.0 / n)
        acc = fx.Float32(0.0)
        for s in range_constexpr(n):
            rstd = fx.Float32(rs_s[fx.Int32(s)])
            dot = fx.Float32(0.0)
            for j in range_constexpr(it_per_group):
                d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
                k = fx.Int32(s * D) + d_off
                wv = _vload8(wd_ptr, w_base + fx.Int64(k))
                lds_base = fx.Int32(s * D + j * (8 * _WAVE)) + lane
                for e in range_constexpr(8):
                    pk_val = fx.Int32(pk_s[lds_base + fx.Int32(e * _WAVE)])
                    xv_e = (pk_val & fx.Int32(-65536)).bitcast(fx.Float32)
                    nv_e = (pk_val << fx.Int32(16)).bitcast(fx.Float32)
                    np1_e = fx.Float32(1.0) + nv_e
                    dot = dot + xv_e * (np1_e * wv[e])
            acc = acc + rstd * _wsum(dot)

        if lane == fx.Int32(0):
            if live:
                y = _bf(_bf(acc) * inv_n)
                if is_inj:
                    fx.ptr_store(fx.Float32(fx.Float32(2.0) * _bf(_sigmoid_f32(y))).to(elem), inj_ptr + w_row)
                else:
                    fx.ptr_store(fx.Float32(y * _sigmoid_f32(y)).to(elem), h_ptr + w_row)

    @flyc.jit
    def launch(
        IDS: fx.Pointer,
        TABLE: fx.Pointer,
        X_OUT: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        embed_hc_premix_decode_l0_kernel(IDS, TABLE, X_OUT, NW, WD, H, INJ, RSTD, wi_delta, eps).launch(
            grid=(grid_x, 1, 1), block=(threads, 1, 1), stream=stream
        )

    embed_hc_premix_decode_l0_kernel.__name__ = f"embed_hc_premix_decode_l0_d{D}_n{n}_r{R}"
    return launch


def build_embed_hc_premix_l0_module(
    D: int = 2560, n: int = 4, R: int = 320
) -> Callable[..., None]:
    """Fused Layer-0 batched `embed_hc_gather` + `hc_premix_c1_u0` (grid=(81, M, 1), block=(128, 1, 1)).
    Exploits uniform stream values at Layer 0 (`X[t, 0] == X[t, 1] == X[t, 2] == X[t, 3] == TABLE[IDS[t]]`)
    to stage only a single stream (`2560` BF16 = `5 KiB` LDS `Int32[1280]`), writing all 4 streams of `X_OUT`
    on `blk == 0` and computing `H`, `INJ`, and `RSTD` in a single launch.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"requires gfx120x, got {arch}"
    ND = n * D
    rows = R + n  # 324
    it_per_group = D // (_WAVE * 8)  # 10
    waves_per_block = 4
    threads = waves_per_block * _WAVE  # 128
    grid_x = (rows + waves_per_block - 1) // waves_per_block  # 81
    elem = fx.BFloat16

    @fx.struct
    class SharedStorage:
        xs: fx.Array[fx.Int32, D // 2, 16]
        rs: fx.Array[fx.Float32, 4, 16]

    def _vstore8(ptr, idx, vals):
        view = fx.make_view(fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(8, 1))
        view.store(fx.Vector.from_elements([fx.Float32(x).to(elem) for x in vals], elem))

    def _wsum(x):
        for o in (1, 2, 4, 8, 16):
            x = x + fx.gpu.shuffle_xor(fx.Float32(x), fx.Int32(o), fx.Int32(_WAVE))
        return x

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def embed_hc_premix_l0_kernel(
        IDS: fx.Pointer,
        TABLE: fx.Pointer,
        X_OUT: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        num_tokens: fx.Int32,
        eps: fx.Float32,
    ) -> None:
        id_ptr = _ptr(IDS, fx.Int32)
        tbl_ptr = _ptr(TABLE, elem)
        x_ptr = _ptr(X_OUT, elem)
        nw_ptr = _ptr(NW, elem)
        wd_ptr = _ptr(WD, elem)
        h_ptr = _ptr(H, elem)
        inj_ptr = _ptr(INJ, elem)
        rs_ptr = _ptr(RSTD, fx.Float32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        xs_s = lds.xs.view(fx.make_layout(D // 2, 1))
        rs_s = lds.rs.view(fx.make_layout(4, 1))

        T = fx.Int32(num_tokens)
        tid = fx.Int32(gpu.thread_idx.x)
        blk = fx.Int32(gpu.block_idx.x)
        t0 = fx.Int32(gpu.block_idx.y)
        t_live = t0 < T
        tok = t_live.select(t0, T - fx.Int32(1))

        wave = tid // fx.Int32(_WAVE)
        lane = tid % fx.Int32(_WAVE)
        row = blk * fx.Int32(waves_per_block) + wave
        live = row < fx.Int32(rows)
        rc = live.select(row, fx.Int32(0))

        is_inj = rc >= fx.Int32(R)
        w_row = is_inj.select(rc - fx.Int32(R), rc)
        w_base = fx.Int64(w_row) * fx.Int64(ND)
        w_base = is_inj.select(w_base + fx.Int64(wi_delta), w_base)

        tok_id = fx.Int64(fx.Int32(fx.ptr_load(id_ptr + tok)))
        tbl_base = tok_id * fx.Int64(D)
        s_w = wave
        is_blk0 = (blk == fx.Int32(0)) & t_live
        tok_nd = tok * fx.Int32(ND)

        sq_w = fx.Float32(0.0)
        for j in range_constexpr(it_per_group):
            d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
            xv = _vload8(tbl_ptr, tbl_base + fx.Int64(d_off))
            if is_blk0:
                _vstore8(x_ptr, tok_nd + s_w * fx.Int32(D) + d_off, xv)
            for e in range_constexpr(8):
                sq_w = sq_w + xv[e] * xv[e]
            if s_w == fx.Int32(0):
                lds_base = fx.Int32(j * (4 * _WAVE)) + lane
                for p_idx in range_constexpr(4):
                    x0 = xv[2 * p_idx].bitcast(fx.Int32) & fx.Int32(-65536)
                    x1 = (xv[2 * p_idx + 1].bitcast(fx.Int32) >> fx.Int32(16)) & fx.Int32(0xFFFF)
                    fx.memref_store(x0 | x1, xs_s, lds_base + fx.Int32(p_idx * _WAVE))

        rstd_w = fmath.rsqrt(_wsum(sq_w) * fx.Float32(1.0 / D) + eps, fastmath="fast")
        if lane == fx.Int32(0):
            fx.memref_store(rstd_w, rs_s, s_w)
            if is_blk0:
                fx.ptr_store(rstd_w, rs_ptr + tok * fx.Int32(n) + s_w)

        gpu.barrier()

        inv_n = fx.Float32(1.0 / n)
        rstd = fx.Float32(rs_s[fx.Int32(0)])
        acc = fx.Float32(0.0)
        for s in range_constexpr(n):
            dot = fx.Float32(0.0)
            for j in range_constexpr(it_per_group):
                d_off = fx.Int32(j * (_WAVE * 8)) + lane * fx.Int32(8)
                k = fx.Int32(s * D) + d_off
                wv = _vload8(wd_ptr, w_base + fx.Int64(k))
                nv = _vload8(nw_ptr, k)
                lds_base = fx.Int32(j * (4 * _WAVE)) + lane
                for p_idx in range_constexpr(4):
                    pk_x = fx.Int32(xs_s[lds_base + fx.Int32(p_idx * _WAVE)])
                    xv0 = (pk_x & fx.Int32(-65536)).bitcast(fx.Float32)
                    xv1 = (pk_x << fx.Int32(16)).bitcast(fx.Float32)
                    dot = dot + xv0 * ((fx.Float32(1.0) + nv[2 * p_idx]) * wv[2 * p_idx])
                    dot = dot + xv1 * ((fx.Float32(1.0) + nv[2 * p_idx + 1]) * wv[2 * p_idx + 1])
            acc = acc + rstd * _wsum(dot)

        y = _bf(_bf(acc) * inv_n)
        ok = live & (lane == fx.Int32(0)) & t_live
        if ok & is_inj:
            fx.ptr_store(fx.Float32(fx.Float32(2.0) * _bf(_sigmoid_f32(y))).to(elem),
                         inj_ptr + tok * fx.Int32(n) + w_row)
        if ok & (rc < fx.Int32(R)):
            fx.ptr_store(fx.Float32(y * _sigmoid_f32(y)).to(elem), h_ptr + tok * fx.Int32(R) + w_row)

    @flyc.jit
    def launch(
        IDS: fx.Pointer,
        TABLE: fx.Pointer,
        X_OUT: fx.Pointer,
        NW: fx.Pointer,
        WD: fx.Pointer,
        H: fx.Pointer,
        INJ: fx.Pointer,
        RSTD: fx.Pointer,
        wi_delta: fx.Int64,
        num_tokens: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        embed_hc_premix_l0_kernel(IDS, TABLE, X_OUT, NW, WD, H, INJ, RSTD, wi_delta, num_tokens, eps).launch(
            grid=(grid_x, num_tokens, 1), block=(threads, 1, 1), stream=stream
        )

    embed_hc_premix_l0_kernel.__name__ = f"embed_hc_premix_l0_d{D}_n{n}_r{R}_t1"
    return launch


# ---------------------------------------------------------------------------
# Build & Embed Table
# ---------------------------------------------------------------------------
def build_new_kernel(key: str):
    # Standard shapes
    if key.startswith("bf16_gemv_m4_"):
        _, _, _, n_s, k_s = key.split("_")
        n, k = int(n_s), int(k_s)
        fn = build_bf16_gemv_m4_module(n, k)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, 4), (n // 8, 1, 1)

    if key.startswith("bf16_gemv_") and not key.startswith("bf16_gemv_m"):
        _, _, n_s, k_s = key.split("_")
        n, k = int(n_s), int(k_s)
        fn = build_bf16_gemv_fast_module(n, k)
        rpb = 16 if n >= 10240 else 8
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16), ((n + rpb - 1) // rpb, 1, 1)

    if key.startswith("bf16_linear_"):
        _, _, n_s, k_s = key.split("_")
        n, k = int(n_s), int(k_s)
        w = 4 if (n % 64 == 0) else 2
        fn = build_bf16_linear_module(n, k, waves=w)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, 16, 1), (n // (16 * w), 1, 1)

    # KRN-1: Single-pass LM Head for M <= 8
    if key == "bf16_gemv_m8_248320_2560":
        fn = build_bf16_gemv_m8_module(248320, 2560)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, 8), (248320 // 8, 1, 1)

    # KRN-3: Fused MTP SWA+Sink tiered NVFP4 decode
    if key == "qsa_swa_sink_decode_fused_nvfp4":
        fn = build_qsa_swa_sink_decode_fused_nvfp4_module(24, 2, 256)
        return flyc.compile(fn, P_BF16, P_U8, P_U8, P_BF16, P_BF16, 0, 0, 0, 28672, 64, 1024), (24, 1, 1)

    # KRN-4: Row-strided SwiGLU
    if key == "silu_mul_640_strided":
        fn = build_silu_mul_strided_module(640)
        return flyc.compile(fn, P_BF16, P_BF16, 1, 1856), (1, 1, 1)

    # KRN-5: Strided + out-of-place Gated DeltaNet ("sigmoid" gate, matching gated_deltanet_fused!)
    if key == "gated_deltanet_fused_strided":
        fn = build_gated_deltanet_fused_strided_module(128, 128, 16, 48, 4, "sigmoid")
        return flyc.compile(
            fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, P_F32, P_BF16, P_BF16, 1, 16480, 1e-6
        ), (48, 1, 1)

    # KRN-6: Batched PLE state update
    if key == "ple_state_update_batch":
        fn = build_ple_state_update_batch_module(10240, 9)
        return flyc.compile(fn, P_BF16, P_BF16, 1), (40, 1, 1)

    # KRN-7: Tiered QSA block scorer with fused sink/window pinning
    if key == "qsa_score_blocks_tiered_pinned":
        fn = build_qsa_score_blocks_tiered_pinned_module(4, 128, 64, 4)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_I32, 0, 0, 8192, 262144, 16, 64, 1e-6), (1, 1, 1)

    # KRN-8: Batched 5-in-1 QSA prep + NVFP4 KV cache write
    if key == "qsa_prep_and_cache_write_batch_nvfp4":
        fn = build_qsa_prep_and_cache_write_batch_nvfp4_module(24, 2, 256, 64, 512, 128)
        return flyc.compile(
            fn, P_BF16, P_BF16, P_BF16, P_F32, P_F32, P_BF16, P_BF16, P_BF16, P_U8, P_U8, P_BF16, 0, 0, 0, 1, 28672, 1e-6
        ), (4, 1, 1)

    # P3-2: Fast-path + register-cached QSA top-k block selector
    if key == "qsa_topk_blocks":
        fn = build_qsa_topk_blocks_fast_module(512, 4)
        return flyc.compile(fn, P_I32, P_I32, 0, 262144), (1, 1, 1)

    # P3-3: 1-Barrier Cooperative LDS-Staged Mega-mHC Premix (0.0 ULP bit-exact parity)
    if key == "hc_premix_decode_c1_u0":
        fn = build_hc_premix_decode_fast_module(2560, 4, 320, True, False)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1e-6), (81, 1, 1)
    if key == "hc_premix_decode_c1_u1":
        fn = build_hc_premix_decode_fast_module(2560, 4, 320, True, True)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1e-6), (81, 1, 1)
    if key == "hc_premix_decode_c0_u1":
        fn = build_hc_premix_decode_fast_module(2560, 4, 320, False, True)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1e-6), (80, 1, 1)
    if key == "hc_premix_c1_u0":
        fn = build_hc_premix_fast_module(2560, 4, 320, True, False)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1, 1e-6), (81, 1, 1)
    if key == "hc_premix_c1_u1":
        fn = build_hc_premix_fast_module(2560, 4, 320, True, True)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1, 1e-6), (81, 1, 1)
    if key == "hc_premix_c0_u1":
        fn = build_hc_premix_fast_module(2560, 4, 320, False, True)
        return flyc.compile(fn, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1, 1e-6), (80, 1, 1)

    # P3-4: Fused PLE FP8 dequant + 12800x2560 GEMV
    if key == "ple_dequant_gemv_12800_2560":
        fn = build_ple_dequant_gemv_12800_2560_module(12800, 2560)
        return flyc.compile(fn, P_U8, P_BF16, P_BF16, P_BF16), (800, 1, 1)
    if key == "ple_dequant_gemv_m4_12800_2560":
        fn = build_ple_dequant_gemv_m4_12800_2560_module(10240, 2560, 2560)
        return flyc.compile(fn, P_U8, P_BF16, P_BF16, P_BF16, P_BF16, 4), (1600, 1, 1)

    # P3-3b: Fused Layer-0 embed_hc_gather + hc_premix
    if key == "embed_hc_premix_decode_l0":
        fn = build_embed_hc_premix_decode_l0_module(2560, 4, 320)
        return flyc.compile(fn, P_I32, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1e-6), (81, 1, 1)
    if key == "embed_hc_premix_l0":
        fn = build_embed_hc_premix_l0_module(2560, 4, 320)
        return flyc.compile(fn, P_I32, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_BF16, P_F32, 0, 1, 1e-6), (81, 1, 1)

    raise KeyError(f"Unknown kernel key: {key}")


NEW_KERNEL_KEYS = [
    "bf16_gemv_2560_2560",
    "bf16_gemv_10240_2560",
    "bf16_gemv_1856_2560",
    "bf16_gemv_m4_16480_2560",
    "bf16_gemv_m4_13952_2560",
    "bf16_gemv_m4_12800_2560",
    "bf16_gemv_m4_10240_2560",
    "bf16_gemv_m4_2560_6144",
    "bf16_gemv_m4_1856_2560",
    "bf16_gemv_m4_2560_640",
    "bf16_gemv_m4_2560_2560",
    "bf16_linear_12800_2560",
    "bf16_gemv_m8_248320_2560",
    "bf16_linear_248320_2560",
    "qsa_swa_sink_decode_fused_nvfp4",
    "silu_mul_640_strided",
    "gated_deltanet_fused_strided",
    "ple_state_update_batch",
    "qsa_score_blocks_tiered_pinned",
    "qsa_prep_and_cache_write_batch_nvfp4",
    "qsa_topk_blocks",
    "hc_premix_decode_c1_u0",
    "hc_premix_decode_c1_u1",
    "hc_premix_decode_c0_u1",
    "hc_premix_c1_u0",
    "hc_premix_c1_u1",
    "hc_premix_c0_u1",
    "ple_dequant_gemv_12800_2560",
    "ple_dequant_gemv_m4_12800_2560",
    "embed_hc_premix_decode_l0",
    "embed_hc_premix_l0",
]


def format_c_array(var_name: str, data: bytes) -> str:
    lines = [f"alignas(64) const uint8_t {var_name}[{len(data)}] = {{"]
    for i in range(0, len(data), 24):
        chunk = data[i : i + 24]
        hex_str = ",".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_str},")
    lines.append("};")
    return "\n".join(lines)


def main():
    print(f"Reading {HSACO_CPP_PATH} ...")
    content = HSACO_CPP_PATH.read_text()

    # Parse existing kEmbeddedKernels entries
    table_match = re.search(
        r"const EmbeddedKernelDesc kEmbeddedKernels\[\] = \{\n(.*?)\n\};",
        content,
        flags=re.DOTALL,
    )
    if not table_match:
        raise RuntimeError("Could not find kEmbeddedKernels[] in gfx1201_nvfp4_hsaco.cpp")

    existing_entries: dict[str, str] = {}
    max_idx = -1
    for line in table_match.group(1).splitlines():
        m = re.search(r'\{"([^"]+)",\s*"([^"]+)",.*?kHsaco_(\d+)_', line)
        if m:
            k = m.group(1)
            idx = int(m.group(3))
            existing_entries[k] = line.strip()
            max_idx = max(max_idx, idx)

    print(f"Found {len(existing_entries)} existing embedded kernels (max index = {max_idx}).")

    # Strip out any previous versions of NEW_KERNEL_KEYS arrays if re-running script
    array_blocks_to_add = []
    next_idx = max_idx + 1

    for key in NEW_KERNEL_KEYS:
        print(f"Compiling {key} ...")
        cf, default_grid = build_new_kernel(key)
        hsaco, sym, bx, by, bz, gx, gy, gz = extract_hsaco_and_meta(cf, default_grid)
        print(f"  -> symbol={sym} block=({bx},{by},{bz}) grid=({gx},{gy},{gz}) bytes={len(hsaco)}")

        # Check if an array for this key already exists in content
        arr_pattern = re.compile(
            rf"alignas\(64\) const uint8_t (kHsaco_\d+_{re.escape(key)})\[\d+\] = \{{.*?\n\}};\n*",
            flags=re.DOTALL,
        )
        m_existing = arr_pattern.search(content)
        if m_existing:
            var_name = m_existing.group(1)
            arr_code = format_c_array(var_name, hsaco) + "\n\n"
            content = content[: m_existing.start()] + arr_code + content[m_existing.end() :]
        else:
            var_name = f"kHsaco_{next_idx}_{key}"
            next_idx += 1
            array_blocks_to_add.append(format_c_array(var_name, hsaco))

        existing_entries[key] = (
            f'{{"{key}", "{sym}", {bx}, {by}, {bz}, {gx}, {gy}, {gz}, {var_name}, {len(hsaco)}ULL}},'
        )

    # Re-locate table_match after any in-place array updates
    table_match = re.search(
        r"const EmbeddedKernelDesc kEmbeddedKernels\[\] = \{\n(.*?)\n\};",
        content,
        flags=re.DOTALL,
    )
    assert table_match is not None

    prefix = content[: table_match.start()]
    suffix = content[table_match.end() :]

    if array_blocks_to_add:
        prefix = prefix + "\n".join(array_blocks_to_add) + "\n\n"

    sorted_table_lines = [f"    {existing_entries[k]}" for k in sorted(existing_entries.keys())]
    new_table = "const EmbeddedKernelDesc kEmbeddedKernels[] = {\n" + "\n".join(sorted_table_lines) + "\n};"

    HSACO_CPP_PATH.write_text(prefix + new_table + suffix)
    print(f"Successfully updated {HSACO_CPP_PATH} with {len(existing_entries)} total embedded kernels!")


if __name__ == "__main__":
    main()
