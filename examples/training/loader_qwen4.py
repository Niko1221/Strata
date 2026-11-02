"""qwen4exp 的 GGUF-only 训练加载器：本地 GGUF 是唯一权重源，零下载。

三个事实来源（都对过源码，不要凭直觉改）：
  - mapping_q4.json     : q4_map_check.py 产出的 GGUF->HF 名字映射（结构闭合已验证）
  - gguf type 42        : Strata fork 的 legacy Q2_0，10 B/32 权重，value=(q-1)*d
  - ple.hpp             : PLE 每 token gather 16 行 x 160 维；表按维度优先存储

显存模型（RTX 4090 48G，WSL 31G RAM）：
  - 非专家权重      bf16 常驻 GPU     ~8 GB
  - 专家（121B）    一次性转 NF4 落盘（~67 GB，D 盘），训练时按命中流式读
                    --ckpt（逐层 checkpoint）是硬要求：forward 解出的专家权重
                    在 backward 重算时重新读盘，不跨层累积
  - PLE 表（28.8GB）留在 GGUF 里 mmap，按 batch 的行 gather（每行 160 值连续）

用法：
  转换（一次性）：python loader_qwen4.py convert <gguf...> --pack-out experts_nf4.bin
  加载        ：from loader_qwen4 import load_model; model = load_model(ggufs, pack)
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nf4
from nf4_fused import fused_nf4_linear

import nf4

# GGML 类型号 -> 名称（Strata fork：42 = fork 版 Q2_0，64 权重/块 18 B，其余标准枚举）
T_F32, T_F16, T_BF16, T_Q2_0 = 0, 1, 30, 42
STD_QUANT = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS",
             20: "IQ4_NL", 21: "IQ3_S", 23: "IQ4_XS"}


class GgufShard:
    """一个 GGUF 分片的 mmap 句柄：张量索引 + 原始字节读取。"""

    def __init__(self, path: str):
        self.path = path
        self.kv, self.tensors, self.data_start = self._parse(path)
        self.by_name = {t["name"]: t for t in self.tensors}
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    @staticmethod
    def _parse(path: str):
        f = open(path, "rb")
        assert f.read(4) == b"GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        assert ver == 3
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rd_val(t):
            if t in (0, 1, 7):
                return struct.unpack("<B", f.read(1))[0]
            if t in (2, 3):
                return struct.unpack("<H", f.read(2))[0]
            if t in (4, 5):
                return struct.unpack("<I", f.read(4))[0]
            if t == 6:
                return struct.unpack("<f", f.read(4))[0]
            if t == 8:
                return rd_str()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return [rd_val(et) for _ in range(n)]
            if t in (10, 11):
                return struct.unpack("<Q", f.read(8))[0]
            if t == 12:
                return struct.unpack("<d", f.read(8))[0]
            raise ValueError(t)

        kv = {}
        for _ in range(n_kv):
            k = rd_str()
            kv[k] = rd_val(struct.unpack("<I", f.read(4))[0])
        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "ne": ne, "type": ttype, "offset": off})
        pos = f.tell()
        align = kv.get("general.alignment", 32)
        data_start = (pos + align - 1) // align * align
        return kv, tensors, data_start

    def raw(self, name: str) -> np.ndarray:
        t = self.by_name[name]
        nbytes = ggml_nbytes(t["type"], t["ne"])
        off = self.data_start + t["offset"]
        return self._mm[off: off + nbytes]

    def drop(self, name: str) -> None:
        """把这个张量占的映射页还给内核。**必须在数据搬走之后调**。

        加载基座时两张分片加起来有 ~11 GB 非专家权重，它们只是路过分页缓存；
        留着的话，小显存档（pack 要占掉 43.8 GiB RAM）会在基座加载阶段直接换页卡死
        （实测 available 归零、6 分钟走不出加载）。逐张量归还后，加载期峰值只剩
        「一个张量的大小」而不是「整个分片」。
        """
        t = self.by_name.get(name)
        if t is None:
            return
        buf = getattr(getattr(self, "_mm", None), "_mmap", None)
        if buf is None or not hasattr(buf, "madvise"):
            return
        import mmap as _mmap

        nbytes = ggml_nbytes(t["type"], t["ne"])
        off = self.data_start + t["offset"]
        pg = _mmap.PAGESIZE
        start = (off // pg) * pg
        length = ((off + nbytes + pg - 1) // pg) * pg - start
        try:
            buf.madvise(_mmap.MADV_DONTNEED, start, length)
        except (OSError, ValueError):
            pass

    def find(self, name: str):
        return self.by_name.get(name)


def ggml_nbytes(ttype: int, ne: list[int]) -> int:
    n = int(np.prod(ne))
    if ttype == T_F32:
        return n * 4
    if ttype in (T_F16, T_BF16):
        return n * 2
    if ttype == T_Q2_0:  # fork 版：18 B / 64 权重
        return n // 64 * 18
    from gguf.constants import GGML_QUANT_SIZES
    block_elems, block_bytes = GGML_QUANT_SIZES[ttype]
    return n // block_elems * block_bytes


def dequant(raw: np.ndarray, ttype: int, ne: list[int]) -> np.ndarray:
    """原始字节 -> fp32，逻辑形状 = reversed(ne)（ne0 最快 == torch C 序）。"""
    n = int(np.prod(ne))
    logical = tuple(reversed(ne))
    if ttype == T_F32:
        out = raw.view(np.float32)[:n].astype(np.float32)
    elif ttype == T_F16:
        out = raw.view(np.float16)[:n].astype(np.float32)
    elif ttype == T_BF16:
        out = torch.from_numpy(raw.view(np.uint16)[:n].copy()).view(torch.bfloat16).float().numpy()
    if ttype == T_Q2_0:
        # fork 版 Q2_0（dequant_bf16.cu::group32）：64 权重/块 18 B = fp16 d + 16 B 的 64 个 2bit 码
        # value = (q - 1) * d，q = (b[2 + e/4] >> ((e%4)*2)) & 3，e 为块内元素序
        nb = n // 64
        b = raw.reshape(nb, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)          # (nb, 1)
        qs = b[:, 2:].reshape(nb, 16).astype(np.uint32)
        # 64 元素 = 16 字节 x 4 码；低 2bit 在前
        bits = ((qs[:, :, None] >> np.array([0, 2, 4, 6], np.uint32)[None, None, :]) & 3)
        vals = bits.reshape(nb, 64).astype(np.float32) - 1.0
        out = (vals * d).reshape(-1)
    else:
        from gguf.quants import dequantize
        from gguf.constants import GGMLQuantizationType
        out = np.asarray(dequantize(raw, GGMLQuantizationType(ttype)), dtype=np.float32).reshape(-1)
    return out.reshape(logical)


# ---------------------------------------------------------------------------
# 专家：GGUF(IQ2_XS/Q2_0/...) -> NF4 pack（一次性），训练时流式读
# ---------------------------------------------------------------------------

_STAGING_GROUP = 32          # 每组命中专家数（32 × 2.49MiB ≈ 80MiB/slot）
_STAGING_SLOTS = 6           # 流水线深度：读盘是主要瓶颈（读算比 ~30:1），浅预取=串行
_READ_THREADS = 4


class _PackStaging:
    """跨层共享的 pinned 读盘暂存区 + 读盘线程池。

    48 层若各建 staging 要吃 ~6GB pinned 内存，而训练严格串行（fwd/bwd 都逐层
    走），全局一份即可。每 slot 四段分开暂存：preadv 的 iovec 把每专家连续块
    （gu|gu_absmax|dn|dn_absmax）散射进各自 buffer——一次 syscall 落盘即分段，
    省 CPU 侧重排。覆写 slot 前用 event 等它上一次 H2D 完成。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(_READ_THREADS)
        self.bufs = None
        self.ev = None
        self.ev_recorded = None
        self.raw = None          # 整块 H2D 的显存原始缓冲（每 slot 一块）
        self.raw_bb = 0
        self.raw_ok = False      # 段边界满足对齐才走得通整块 view

    def _ensure(self, seg_sizes):
        with self.lock:
            if self.bufs is None:
                mk = lambda n: torch.empty(_STAGING_GROUP * n, dtype=torch.uint8, pin_memory=True)
                self.bufs = [tuple(mk(n) for n in seg_sizes) for _ in range(_STAGING_SLOTS)]
                self.ev = [torch.cuda.Event() for _ in range(_STAGING_SLOTS)]
                self.ev_recorded = [False] * _STAGING_SLOTS
                # 整块 H2D：一块一次 copy_（源 pinned）而不是 gu/ga/dn/dam 四段四次小
                # copy_（实测每步 3.2 万次小 H2D）。段切分改在 GPU 上 view，与 resident
                # 路径的 gpu_views 完全一致。段起点必须 256B 对齐才能在显存里 view bf16。
                bb = sum(seg_sizes)
                self.raw_bb = bb
                self.raw_ok = (bb % 256 == 0) and all(s % 256 == 0 for s in seg_sizes)
                if self.raw_ok:
                    self.raw = [torch.empty(_STAGING_GROUP * bb, dtype=torch.uint8,
                                            device="cuda") for _ in range(_STAGING_SLOTS)]
            return self.bufs

    def slot_sync(self, slot):
        if self.ev_recorded[slot]:
            self.ev[slot].synchronize()


_PACK_STAGING = _PackStaging()


class _TieredPack:
    """全驻留 NF4 pack（2026-10-29 方向修正）：显存 + 内存分层，训练零读盘。

    依据：48GiB 显存 + 54GiB WSL 内存 = 102GiB > pack 60GiB——模型放得下，且长
    seq 训练每步命中全部专家，任何读盘都是纯浪费（实测流式读盘把吞吐压到
    9~18 tok/s、GPU util 1%）。这是引擎"常驻 GPU + 内存/显存分层转移"在训练侧
    的对应物。启动时一次性读入（约 30~60s）：
      - 前 STRATA_VRAM_PACK_GB（默认 16）GiB 的块驻显存：forward 直接视图，零拷贝；
      - 其余驻 RAM：pinned memcpy + H2D，无磁盘访问。
    显存预算：16(pack) + 9.25(base) + 工作区；ckpt 下 peak ~19.6GiB 含基座，
    共 ~26GiB，48GiB 卡余量充足。no-ckpt 反传保留 packed 的场景自行调小。
    """

    def __init__(self):
        self.gpu_buf = None          # uint8 cuda，块 [0, gpu_blocks)
        self.cpu_buf = None          # uint8 host（pageable），块 [gpu_blocks+pin, n_blocks)
        self.cpu_pin = None          # uint8 host（pinned），块 [gpu_blocks, gpu_blocks+pin)
        self.cpu_pin_blocks = 0
        self.cpu_pinned = False      # 是否**部分**锁页：见 _ensure 的注释
        self.gpu_blocks = 0
        self.n_blocks = 0
        self.block_bytes = 0
        self.lock = threading.Lock()
        self._views = {}             # block_idx -> (gu, ga, dn, da) 显存视图缓存

    def _ensure(self, pack_path: str, block_bytes: int):
        with self.lock:
            if self.gpu_buf is not None:
                assert self.block_bytes == block_bytes, "同一 pack 块大小应一致"
                return
            size = os.path.getsize(pack_path)
            n_blocks = size // block_bytes
            vram_gb = int(os.environ.get("STRATA_VRAM_PACK_GB", "16"))
            gpu_blocks = min(n_blocks, (vram_gb << 30) // block_bytes)
            fd = os.open(pack_path, os.O_RDONLY)
            t0 = time.time()
            gpu_buf = None
            if gpu_blocks:
                gpu_buf = torch.empty(gpu_blocks * block_bytes, dtype=torch.uint8, device="cuda")
                chunk_blocks = (256 << 20) // block_bytes
                # 双缓冲：pinned stage 上的 H2D 是真异步 DMA，会被下一轮 preadv 覆写。
                # 单缓冲版本只在整轮循环后 synchronize 一次 ⇒ preadv 覆盖正在被 DMA 读的
                # 字节，驻显存那部分 pack 静默损坏（2026-10-29 实测：同一份文件两次加载，
                # layer 0 的 512 个专家里 150 个字节不同，而全部从 RAM 走 staging 的
                # layer 40 一个字节不差；同进程内前向因此逐位可复现、跨进程 CE 飘
                # 0.065~0.11 nats）。每个 stage 覆写前先等自己上一次 DMA 完成。
                stages = [torch.empty(chunk_blocks * block_bytes, dtype=torch.uint8,
                                      pin_memory=True) for _ in range(2)]
                evs = [torch.cuda.Event() for _ in range(2)]
                for ci, c0 in enumerate(range(0, gpu_blocks, chunk_blocks)):
                    c1 = min(c0 + chunk_blocks, gpu_blocks)
                    nb = (c1 - c0) * block_bytes
                    k = ci % 2
                    if ci >= 2:
                        evs[k].synchronize()          # 该 stage 上一次 DMA 读完才可覆写
                    got = os.preadv(fd, [stages[k].numpy()[:nb]], c0 * block_bytes)
                    assert got == nb, f"pack 读入不完整：{got} != {nb}"
                    gpu_buf[c0 * block_bytes:c1 * block_bytes] \
                        .copy_(stages[k][:nb], non_blocking=True)
                    evs[k].record()
                torch.cuda.synchronize()
            cpu_n = n_blocks - gpu_blocks
            # 锁页 host 缓冲（2026-10-29 修订：**部分**锁页）。
            #
            # 早先一次性 pin 整个 CPU 层（43.8 GiB）直接 AcceleratorError，于是整段退回
            # pageable。而 pageable 源的 `copy_(non_blocking=True)` 并不异步：它走
            # cudaMemcpyAsync 的同步分支，**阻塞调用线程**直到搬完。实测 2.49 MiB 一块
            # 222 µs，17997 块 ⇒ 每步约 4.0 s 里 CPU 完全无法派发后续 kernel，GPU 同时
            # 空转。pinned 源 112 µs 一块且立即返回（21.7 GiB/s vs 11.0 GiB/s）。
            #
            # 本机单次可 pin 上限实测 32 GiB（33 GiB 失败，见 _mb_pin.py），所以拆成
            # 「pinned 前缀 + pageable 余量」两段，而不是全有或全无。STRATA_PIN_GB 可调，
            # 失败按减半重试（不假设具体上限，Windows 侧数值不同）。
            pin = os.environ.get("STRATA_PIN_PACK", "1") != "0"
            want = 0
            if pin:
                want = min(cpu_n, int(float(os.environ.get("STRATA_PIN_GB", "30")) * 2**30)
                           // block_bytes)
            cpu_pin = None
            while want > 0:
                try:
                    cpu_pin = torch.empty(want * block_bytes, dtype=torch.uint8,
                                          pin_memory=True)
                    break
                except Exception as exc:      # 锁页失败则减半重试，最后退回全 pageable
                    print(f"[pack] pin {want * block_bytes / 2**30:.1f} GiB 失败"
                          f"（{type(exc).__name__}{': ' + str(exc).splitlines()[0][:60] if str(exc) else ''}）"
                          f"，减半重试", flush=True)
                    cpu_pin, want = None, want // 2
            page_n = cpu_n - want
            cpu_buf = torch.empty(page_n * block_bytes, dtype=torch.uint8)
            self.cpu_pinned = want > 0
            # 先读 pinned 前缀，再读 pageable 余量；两者在文件里连续，只是落点不同。
            for buf, base_blk, cnt in ((cpu_pin, gpu_blocks, want),
                                       (cpu_buf, gpu_blocks + want, page_n)):
                if cnt == 0:
                    continue
                np_view, o, total = buf.numpy(), 0, cnt * block_bytes
                while o < total:
                    nb = min((64 << 20) // block_bytes * block_bytes, total - o)
                    got = os.preadv(fd, [np_view[o:o + nb]], (base_blk * block_bytes) + o)
                    assert got == nb, f"pack 读入不完整：{got} != {nb}"
                    o += nb
            os.close(fd)
            self.gpu_buf, self.cpu_buf, self.cpu_pin = gpu_buf, cpu_buf, cpu_pin
            self.cpu_pin_blocks = want
            self.gpu_blocks, self.n_blocks = gpu_blocks, n_blocks
            self.block_bytes = block_bytes
            print(f"tiered pack: {gpu_blocks} blocks in VRAM "
                  f"({gpu_blocks * block_bytes / 2**30:.1f} GiB) + "
                  f"{want} pinned ({want * block_bytes / 2**30:.1f} GiB) + "
                  f"{page_n} pageable ({page_n * block_bytes / 2**30:.1f} GiB)"
                  f", loaded {time.time() - t0:.0f}s"
                  " — 训练零读盘", flush=True)

    def host_block(self, off: int, nbytes: int | None = None) -> torch.Tensor:
        """块起始字节偏移 -> host 上一块的视图。pinned 前缀在 cpu_pin，其余在 cpu_buf。

        返回的是视图，不是拷贝：调用方必须立刻 `copy_` 出去，不能持有。
        """
        bb = self.block_bytes
        nb = bb if nbytes is None else nbytes
        i = off // bb - self.gpu_blocks
        assert 0 <= i < self.n_blocks - self.gpu_blocks, f"块号越界：off={off}"
        if i < self.cpu_pin_blocks:
            b = i * bb
            return self.cpu_pin[b:b + nb]
        b = (i - self.cpu_pin_blocks) * bb
        return self.cpu_buf[b:b + nb]

    def resident_gpu(self, off: int) -> int:
        """块起始字节偏移 -> 显存块号；不驻显存返回 -1。"""
        idx = off // self.block_bytes
        if off == idx * self.block_bytes and idx < self.gpu_blocks:
            return idx
        return -1

    def gpu_views(self, idx: int, gu_shape, dn_shape, seg_sizes):
        v = self._views.get(idx)
        if v is None:
            gu_n, ga_n, dn_n, dam_n = seg_sizes
            b = idx * self.block_bytes
            gu = self.gpu_buf[b:b + gu_n].view(gu_shape[0], gu_shape[1] // 2)
            ga = self.gpu_buf[b + gu_n:b + gu_n + ga_n] \
                .view(torch.bfloat16).view(gu_shape[0], gu_shape[1] // 64)
            dn = self.gpu_buf[b + gu_n + ga_n:b + gu_n + ga_n + dn_n] \
                .view(dn_shape[0], dn_shape[1] // 2)
            da = self.gpu_buf[b + gu_n + ga_n + dn_n:b + gu_n + ga_n + dn_n + dam_n] \
                .view(torch.bfloat16).view(dn_shape[0], dn_shape[1] // 64)
            v = (gu, ga, dn, da)
            self._views[idx] = v
        return v


_TIERED = _TieredPack()


_XFER = {"bytes": 0}             # 整块直连路径的 H2D 字节计数（诊断用，SCAN 会读）


def _copy_block(src: torch.Tensor, dsts, sizes):
    """把连续块按 [gu|ga|dn|dam] 四段拷进 pinned staging。"""
    o = 0
    for d, n in zip(dsts, sizes):
        d.copy_(src[o:o + n])
        o += n


def _seg_views(buf: torch.Tensor, base: int, seg_sizes, gu_shape, dn_shape):
    """在显存里把一块按 [gu|ga|dn|dam] 切成四段视图（resident 的 gpu_views 同款）。"""
    gu_n, ga_n, dn_n, dam_n = seg_sizes
    o = base
    gu = buf[o:o + gu_n].view(gu_shape[0], gu_shape[1] // 2)
    o += gu_n
    ga = buf[o:o + ga_n].view(torch.bfloat16).view(gu_shape[0], gu_shape[1] // 64)
    o += ga_n
    dn = buf[o:o + dn_n].view(dn_shape[0], dn_shape[1] // 2)
    o += dn_n
    da = buf[o:o + dam_n].view(torch.bfloat16).view(dn_shape[0], dn_shape[1] // 64)
    return (gu, ga, dn, da)


class StreamingNF4Experts(nn.Module):
    """全驻留 NF4 pack 的专家集合。forward 与 HF Qwen4ExpTextExperts 逐行等价。
    pack 布局（convert 写死）：
      [layer][expert] = gu | gu_absmax | dn | dn_absmax（四段连续，共 block_bytes）
    读权重路径（2026-10-29 定稿，见 _TieredPack）：启动时整个 pack 读入
    显存+内存分层，训练零读盘——
      - 驻显存块：fused kernel 直接用显存视图，零拷贝零 H2D；
      - 驻内存块：pinned staging memcpy + H2D，组间流水（深度 _STAGING_SLOTS-1）。
    """

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int, act_fn,
                 pack_path: str, layer_idx: int, layer_offsets: dict):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.layer_idx = layer_idx
        object.__setattr__(self, "_offs", layer_offsets)   # {(kind, expert): (byte_off, nbytes)}
        self._gu_shape = (2 * intermediate_dim, hidden_dim)
        self._dn_shape = (hidden_dim, intermediate_dim)
        gu_off, self._gu_n = layer_offsets[("gu", 0)]
        ga_off, self._ga_n = layer_offsets[("gu_absmax", 0)]
        dn_off, self._dn_n = layer_offsets[("dn", 0)]
        dam_off, self._dam_n = layer_offsets[("dn_absmax", 0)]
        assert ga_off == gu_off + self._gu_n and dn_off == ga_off + self._ga_n \
            and dam_off == dn_off + self._dn_n, "expert 四段不连续，pack 布局变了"
        self._block_bytes = self._gu_n + self._ga_n + self._dn_n + self._dam_n
        _TIERED._ensure(pack_path, self._block_bytes)   # 首个层实例触发一次性全量读入

    def _stream_groups(self, hits, dev):
        """按组产出每专家 GPU 张量，yield (组内专家列表, {expert: (gu,ga,dn,da)})。

        驻显存块直接视图；驻内存块走 pinned staging（CPU memcpy）+ H2D，
        组间流水：submit(g+1..g+slots-1) 与组 g 的 H2D+计算重叠。
        """
        st = _PACK_STAGING
        bufs = st._ensure((self._gu_n, self._ga_n, self._dn_n, self._dam_n))
        G, n_hits = _STAGING_GROUP, len(hits)
        n_groups = (n_hits + G - 1) // G
        tp, bb = _TIERED, self._block_bytes
        seg_sizes = (self._gu_n, self._ga_n, self._dn_n, self._dam_n)
        # 快路径：每块一次整块 H2D + 显存侧切段，而不是四段各自 copy_。
        # 微基准：串行下两条路径都是 ~20 GB/s，端到端却只有 0.9~1.7 GB/s —— 瓶颈不是
        # 带宽而是每块 ~14 次 Python/torch 调用（每步 11 万次）的调度开销，所以即便
        # cpu_buf 没能锁页（WSL 下 43.8 GiB pin 会失败），整块直连仍然值得：调用次数
        # 从 4 次 copy_ + 线程池 submit 降到 1 次 copy_ + 4 次 view。
        #
        # 反传模式下曾经必须关掉：这条路径返回的是 st.raw[slot] 上的 **视图**，slot 会随
        # 组轮转被覆盖，而 fused NF4 Function 把 packed/absmax 存进了 ctx
        # （save_for_backward），缓冲一被回收就撞 "variables needed for gradient
        # computation has been modified by an inplace operation ... is at version N"。
        # 现在的做法是 grad 下**每个专家分配独立显存块**（而不是复用 slot）：多一次
        # `torch.empty`（缓存分配器约 2 µs），换来同样只做 1 次整块 copy_ + 4 次 view。
        # 显存代价按层计：一层 512 个专家 x 2.49 MiB = 1.27 GiB，该层反传完即释放，
        # 只在开 ckpt（逐层重算）时可接受 —— 不开 ckpt 时所有权重本来就会进图，仍然 OOM。
        use_raw = st.raw_ok and os.environ.get("STRATA_PACK_RAW_H2D", "1") != "0"
        grad = torch.is_grad_enabled()
        # grad 下**整层一块**显存：非驻留块一次申请 1.27 GiB（512 × 2.49 MiB），逐块切片。
        # 之前是每专家一次 `torch.empty(bb)` —— 一层 512 次、每步 24,576 次，ckpt 重算再
        # 来一遍。切片之间不重叠，所以依旧不会与槽位轮转别名（那才是当初 version
        # counter 报错的成因），但分配器调用从 24,576 次降到 48 次。
        layer_buf, miss_i = None, 0
        if grad and use_raw:
            miss = sum(1 for e in hits
                       if tp.resident_gpu(self._offs[("gu", e)][0]) < 0)
            if miss:
                layer_buf = torch.empty(miss * bb, dtype=torch.uint8, device=dev)

        def submit(gi, slot):
            """把该组驻内存块的 pinned 拷贝提交进线程池，返回 [(组内序, 专家, fut)]。"""
            if use_raw:
                return []                      # 源已 pinned：直接 H2D，无需预取
            gu_b, ga_b, dn_b, dam_b = bufs[slot]
            jobs = []
            for i, e in enumerate(hits[gi * G:(gi + 1) * G]):
                off, _ = self._offs[("gu", e)]
                if tp.resident_gpu(off) >= 0:
                    continue
                dst = [gu_b[i * self._gu_n:(i + 1) * self._gu_n],
                       ga_b[i * self._ga_n:(i + 1) * self._ga_n],
                       dn_b[i * self._dn_n:(i + 1) * self._dn_n],
                       dam_b[i * self._dam_n:(i + 1) * self._dam_n]]
                src = tp.host_block(off, bb)
                jobs.append((i, e, st.pool.submit(_copy_block, src, dst, seg_sizes)))
            return jobs

        def finish(gi, slot, jobs):
            """等 pinned 拷贝完成 → H2D → 与显存视图合成 {expert: (gu,ga,dn,da)}。"""
            for _, _, f in jobs:
                f.result()
            out = {}
            if use_raw:
                nonlocal miss_i
                rb = st.raw_bb
                raw = None if grad else st.raw[slot]
                for i, e in enumerate(hits[gi * G:(gi + 1) * G]):
                    off, _ = self._offs[("gu", e)]
                    idx = tp.resident_gpu(off)
                    if idx >= 0:
                        out[e] = tp.gpu_views(idx, self._gu_shape, self._dn_shape, seg_sizes)
                        continue
                    if grad:
                        # 整层缓冲上取一段，反传期间一直活着；与其它专家不重叠
                        blk = layer_buf
                        base = miss_i * bb
                        miss_i += 1
                    else:
                        blk = raw
                        base = i * rb
                    blk[base:base + rb].copy_(tp.host_block(off, rb), non_blocking=True)
                    _XFER["bytes"] += rb
                    out[e] = _seg_views(blk, base, seg_sizes, self._gu_shape, self._dn_shape)
                return out
            gu_b, ga_b, dn_b, dam_b = bufs[slot]
            for i, e in enumerate(hits[gi * G:(gi + 1) * G]):
                off, _ = self._offs[("gu", e)]
                idx = tp.resident_gpu(off)
                if idx >= 0:
                    out[e] = tp.gpu_views(idx, self._gu_shape, self._dn_shape, seg_sizes)
                    continue
                gu = gu_b[i * self._gu_n:(i + 1) * self._gu_n].to(dev, non_blocking=True) \
                    .view(self._gu_shape[0], self._gu_shape[1] // 2)
                ga = ga_b[i * self._ga_n:(i + 1) * self._ga_n].to(dev, non_blocking=True) \
                    .view(torch.bfloat16).view(self._gu_shape[0], self._gu_shape[1] // 64)
                dn = dn_b[i * self._dn_n:(i + 1) * self._dn_n].to(dev, non_blocking=True) \
                    .view(self._dn_shape[0], self._dn_shape[1] // 2)
                da = dam_b[i * self._dam_n:(i + 1) * self._dam_n].to(dev, non_blocking=True) \
                    .view(torch.bfloat16).view(self._dn_shape[0], self._dn_shape[1] // 64)
                out[e] = (gu, ga, dn, da)
            return out

        pending = {}                          # gi -> (slot, jobs)，最多 _STAGING_SLOTS 组在飞
        gi_next = 0
        # grad + 整块路径下槽位其实**没被写**（数据直接进 layer_buf），所以 `slot_sync`
        # 只剩「等 GPU 跑到上一次 record」这一个副作用 —— 那是每 32 个专家一次的**阻塞
        # 同步**（一层 16 次、一步 768 次），把 CPU 派发和 GPU 执行强行串起来。GPU 有
        # 一半时间是空的，就是这么空出来的。不写槽位就别等它。
        slot_in_use = (not use_raw) or (not grad)

        def pump(upto):
            """把 [upto, upto+slots) 的组提交进流水（slot 轮转，覆写前 slot_sync）。"""
            nonlocal gi_next
            while gi_next < min(upto, n_groups):
                slot = gi_next % _STAGING_SLOTS
                if slot_in_use:
                    st.slot_sync(slot)
                pending[gi_next] = (slot, submit(gi_next, slot))
                gi_next += 1

        pump(_STAGING_SLOTS - 1)
        for gi in range(n_groups):
            slot, jobs = pending.pop(gi)
            pump(gi + _STAGING_SLOTS)
            tensors = finish(gi, slot, jobs)
            if slot_in_use:
                st.ev[slot].record()
                st.ev_recorded[slot] = True
            yield hits[gi * G:(gi + 1) * G], tensors

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            # 一次拿到全量路由，替代原来「逐专家 torch.where」：
            #  - 旧的 `[int(e[0]) for e in hit if int(e[0]) < N]` 每次迭代做两次阻塞式
            #    DtoH，每层命中 256 个专家、共 96 层 ⇒ 每步约 4.9e4 次 cudaStreamSynchronize
            #    （profiler 实测整步 81,673 次同步 + 81,667 次 DtoH），GPU 因此空转 ~2/3。
            #  - 旧的 `torch.where(mask[e])` 每步 24,576 次 nonzero，每次内部一轮 cub radix sort
            #    （profiler 里 24,712 / 24,624 / 24,540 那串等值计数就是它）。
            # nonzero 在 [E,K,N] 上按 c-order 返回，e-major / k / n，与旧逐专家 where 的
            # (pos=k, tok=n) 行主序完全同序 ⇒ index_add_ 的累加顺序不变，逐位可比。
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            nz = mask.nonzero()                        # [n, 3] = (expert, k, t)
            counts = torch.bincount(nz[:, 0], minlength=self.num_experts)
            bounds = torch.cumsum(counts, 0).tolist()  # 全步唯一一次 DtoH
            # 逐专家的 (token, gate 权重) 预先按 expert-major 展平好：循环里只切一维，
            # 不再对 [n,3] 做两次列切 + 两次高级索引（每步 24,576 次 × 5 op）。
            tok_all = nz[:, 2]
            grp = top_k_weights.reshape(-1).gather(0, tok_all * top_k_index.shape[1] + nz[:, 1])
        prev, hits = 0, []
        for e, end in enumerate(bounds):
            if end > prev:
                hits.append(e)
            prev = end
        if not hits:
            return final

        for group, tensors in self._stream_groups(hits, hidden_states.device):
            for e in group:
                gu, ga, dn, da = tensors[e]
                a = bounds[e - 1] if e else 0
                tok = tok_all[a:bounds[e]]
                x = hidden_states[tok]
                gate, up = fused_nf4_linear(x, gu, ga).chunk(2, dim=-1)
                h = fused_nf4_linear(self.act_fn(gate) * up, dn, da)
                final.index_add_(0, tok, (h * grp[a:bounds[e], None]).to(final.dtype))
        return final


def _worker_dequant(args):
    """子进程：解一个 (layer, expert) 的三块权重。同一份 dequant 代码，零正确性分叉。"""
    path, l, e, gu_shape, dn_shape = args
    s = GgufShard._mmap_cache.get(path)
    if s is None:
        s = GgufShard(path)
        GgufShard._mmap_cache[path] = s
    E = 512
    gate_t = s.by_name[f"blk.{l}.ffn_gate_exps.weight"]
    up_t = s.by_name[f"blk.{l}.ffn_up_exps.weight"]
    dn_t = s.by_name[f"blk.{l}.ffn_down_exps.weight"]
    gate_elems = int(np.prod(gate_t["ne"])) // E
    up_elems = int(np.prod(up_t["ne"])) // E
    dn_elems = int(np.prod(dn_t["ne"])) // E
    gate_raw = s.raw(f"blk.{l}.ffn_gate_exps.weight")[e * _tbytes(gate_t) // E:(e + 1) * _tbytes(gate_t) // E]
    up_raw = s.raw(f"blk.{l}.ffn_up_exps.weight")[e * _tbytes(up_t) // E:(e + 1) * _tbytes(up_t) // E]
    dn_raw = s.raw(f"blk.{l}.ffn_down_exps.weight")[e * _tbytes(dn_t) // E:(e + 1) * _tbytes(dn_t) // E]
    gate = dequant(gate_raw, gate_t["type"], [gate_elems]).reshape(I := gu_shape[0] // 2, gu_shape[1])
    up = dequant(up_raw, up_t["type"], [up_elems]).reshape(gu_shape[0] // 2, gu_shape[1])
    dn = dequant(dn_raw, dn_t["type"], [dn_elems]).reshape(dn_shape)
    gu = np.ascontiguousarray(np.concatenate([gate, up], axis=0))   # HF 约定：[gate; up] 连续对半
    return l, e, gu, np.ascontiguousarray(dn)


def _tbytes(t: dict) -> int:
    return ggml_nbytes(t["type"], t["ne"])


GgufShard._mmap_cache = {}


def convert_experts(gguf_paths: list[str], pack_out: str, workers: int = 24):
    """一次性：全部专家 GGUF -> NF4 pack。worker 并行反量化（CPU），父进程 GPU 量化 NF4。"""
    shards = [GgufShard(p) for p in gguf_paths]
    E, I, H = 512, 640, 2560
    gu_shape, dn_shape = (2 * I, H), (H, I)

    def packed_bytes(shape):        # NF4：2 权重/字节
        return int(np.prod(shape)) // 2

    def absmax_bytes(shape):        # 每 64 权重一个 bf16 scale
        return int(np.prod(shape)) // 64 * 2

    per_expert = {
        "gu": packed_bytes(gu_shape), "gu_absmax": absmax_bytes(gu_shape),
        "dn": packed_bytes(dn_shape), "dn_absmax": absmax_bytes(dn_shape),
    }
    n_layers = 48
    layer_bytes = E * sum(per_expert.values())
    offsets = {}
    for l in range(n_layers):
        base = l * layer_bytes
        for e in range(E):
            o = base + e * sum(per_expert.values())
            offsets[(l, e, "gu")] = (o, per_expert["gu"])
            offsets[(l, e, "gu_absmax")] = (o + per_expert["gu"], per_expert["gu_absmax"])
            o2 = o + per_expert["gu"] + per_expert["gu_absmax"]
            offsets[(l, e, "dn")] = (o2, per_expert["dn"])
            offsets[(l, e, "dn_absmax")] = (o2 + per_expert["dn"], per_expert["dn_absmax"])

    meta = {"layers": n_layers, "experts": E, "gu_shape": gu_shape, "dn_shape": dn_shape,
            "blocksize": 64, "per_expert": per_expert, "layer_bytes": layer_bytes,
            "gguf": [p for p in gguf_paths]}
    with open(pack_out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    dev = torch.device("cuda")
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    pack = open(pack_out, "wb")
    pack.truncate(n_layers * layer_bytes)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for l in range(n_layers):
            futures = [ex.submit(_worker_dequant, (gguf_paths[_shard_of(shards, f"blk.{l}.ffn_gate_exps.weight")], l, e, gu_shape, dn_shape))
                       for e in range(E)]
            for f in futures:
                li, e, gu, dn = f.result()
                gu_t = torch.from_numpy(gu).to(dev)
                pk, am = nf4.quantize(gu_t)
                _pwrite(pack, offsets[(li, e, "gu")], pk.cpu().numpy())
                _pwrite(pack, offsets[(li, e, "gu_absmax")], am.cpu().view(torch.uint8).numpy())
                del gu_t, pk, am
                dn_t = torch.from_numpy(dn).to(dev)
                pk, am = nf4.quantize(dn_t)
                _pwrite(pack, offsets[(li, e, "dn")], pk.cpu().numpy())
                _pwrite(pack, offsets[(li, e, "dn_absmax")], am.cpu().view(torch.uint8).numpy())
                del dn_t, pk, am
                done += 1
            print(f"  layer {l}: {done}/{n_layers * E} experts  {time.time() - t0:.0f}s", flush=True)
    pack.close()
    print(f"pack written: {pack_out} ({os.path.getsize(pack_out) / 2**30:.1f} GiB, {time.time() - t0:.0f}s)")


def _shard_of(shards, name: str) -> int:
    for i, s in enumerate(shards):
        if s.find(name) is not None:
            return i
    raise KeyError(name)


def _pwrite(pack, off_nbytes: tuple[int, int], arr: np.ndarray) -> None:
    off, nbytes = off_nbytes
    assert arr.nbytes == nbytes, (arr.nbytes, nbytes)
    pack.seek(off)
    pack.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())


# ---------------------------------------------------------------------------
# PLE：mmap 按行 gather 的 ngram embedding（替换 HF 的常驻 nn.Embedding）
# ---------------------------------------------------------------------------

class MmapNgramEmbedding(nn.Module):
    """接口与 HF Qwen4ExpTextNGramEmbedding 相同，但表不驻留：
    GGUF 里按维度优先存储，行 r 的 160 个值在文件里连续，按行读+解 IQ4_NL。
    forward 主体复制自 HF（Apache-2.0），仅最后的 lookup 换成本类。"""

    def __init__(self, hf_ref, gguf: "GgufShard", tensor_name: str):
        super().__init__()
        src = hf_ref
        self.ngram_size = src.ngram_size
        self.context_len = src.context_len
        self.heads_per_ngram = src.heads_per_ngram
        self.ngram_heads = src.ngram_heads
        self.ple_layer_index = src.ple_layer_index
        self.eos_token_id = src.eos_token_id
        self.layer_idx = src.layer_idx
        # HF 推导的质数/偏移必须与 GGUF 一致——不一致就停，静默错 hash 是灾难
        assert src.head_vocab_sizes == list(gguf.kv["qwen4exp.ple.head_vocab_sizes"]), \
            (src.head_vocab_sizes, gguf.kv["qwen4exp.ple.head_vocab_sizes"])
        # 注意：这些常量绝不能 register_buffer——load_model 后面的 to_empty() 会把 buffer 清零。
        # 存普通 Python 列表，forward 里现建（16 元素张量，开销可忽略）。
        self._mult = [int(x) for x in gguf.kv["qwen4exp.ple.layer_multipliers"]]
        self._hvs = [int(x) for x in src.head_vocab_sizes]
        self._offs = [int(x) for x in src.head_offsets]
        self.total_vocab_size = src.total_vocab_size
        t = gguf.by_name[tensor_name]
        assert t["ne"][0] == 160, t["ne"]        # 每行 160 = 16 头 x 10... 实为每行 160 维
        row_elems = t["ne"][0]
        self.row_elems = row_elems
        from gguf.constants import GGML_QUANT_SIZES
        block_elems, blen = GGML_QUANT_SIZES[t["type"]]
        self._blocks_per_row = row_elems // block_elems
        self._block_bytes = blen
        self._row_bytes = self._blocks_per_row * blen
        self._total_rows = t["ne"][1]
        self._base = gguf.data_start + t["offset"]
        self._mm = gguf._mm
        self._ttype = t["type"]
        self._rows_view = None       # (total_rows, row_bytes) 的 mmap 零拷贝视图

    def _gather_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """rows: (N,) long -> (N, 160) bf16。行 r 的 160 值在 GGUF 里连续（维度优先存储）。

        2026-10-29：旧实现是 Python 里逐行 `mmap 切片 + gguf.dequantize`。T=2048 一次
        前向要 gather 约 3e4 行，每行几十微秒的纯 Python 反量化 ⇒ 秒级 CPU 阻塞，而这段
        完全不出现在 CUDA profiler 里（GPU 空转）。改为：
          - 整个 PLE 表先在 mmap 上取一个 (total_rows, row_bytes) 的**零拷贝视图**；
          - 用 numpy fancy-index 一次性把这 N 行字节搬进连续缓冲；
          - 只调用一次 dequantize（它本身是逐块向量化的）。
        行数不变、数值路径不变（同样的 gguf dequantize），只是不再逐行付 Python 开销。
        """
        r = rows.to("cpu")
        if (int(r.min()) < 0) or (int(r.max()) >= self._total_rows):
            raise IndexError(f"PLE row out of range: min {int(r.min())} max {int(r.max())} "
                             f"total {self._total_rows}")
        uniq, inv = torch.unique(r, return_inverse=True)
        from gguf.quants import dequantize
        from gguf.constants import GGMLQuantizationType
        if not uniq.numel():
            out = np.zeros((0, self.row_elems), np.float32)
        elif os.environ.get("STRATA_PLE_ROWLOOP", "0") != "0":
            chunks = []
            for b in (int(x) * self._blocks_per_row for x in uniq):
                raw = self._mm[self._base + b * self._block_bytes:
                               self._base + (b + self._blocks_per_row) * self._block_bytes]
                chunks.append(np.asarray(
                    dequantize(np.ascontiguousarray(raw), GGMLQuantizationType(self._ttype)),
                    dtype=np.float32).reshape(-1))
            out = np.stack(chunks)
        else:
            view = self._rows_view
            if view is None:
                view = self._rows_view = np.frombuffer(
                    self._mm, dtype=np.uint8, count=self._total_rows * self._row_bytes,
                    offset=self._base).reshape(self._total_rows, self._row_bytes)
            buf = np.ascontiguousarray(view[uniq.numpy()])
            out = dequantize(buf, GGMLQuantizationType(self._ttype))
            out = np.asarray(out, dtype=np.float32).reshape(uniq.numel(), self.row_elems)
        t = torch.from_numpy(out).to(rows.device, dtype=torch.bfloat16)
        return t[inv]

    # ---- 以下与 HF forward 相同，仅最后一行换成 _gather_rows ----
    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
        return Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos(self, token_ids, shift)

    def forward(self, input_ids: torch.Tensor, past_key_values) -> torch.Tensor:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
        input_ids = input_ids.long()
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
            previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            if (not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                    and input_ids.shape[1] < self.context_len):
                input_ids_to_cache = F.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id)
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len)
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]
        dev = input_ids.device
        mult = torch.tensor(self._mult, dtype=torch.long, device=dev)
        hvs_all = torch.tensor(self._hvs, dtype=torch.long, device=dev)
        offs_all = torch.tensor(self._offs, dtype=torch.long, device=dev)
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * mult[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids, shifted_tokens[position] * mult[position])
            head_vocab_sizes = hvs_all[start_idx:end_idx]
            head_offsets = offs_all[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))
        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
        B, S, K = ngram_ids.shape
        return self._gather_rows(ngram_ids.reshape(-1)).view(B, S, K, self.row_elems).flatten(-2)


# ---------------------------------------------------------------------------
# 骨架组装 + 非专家加载
# ---------------------------------------------------------------------------

# HF Qwen4ExpTextRMSNorm 的 weight 是增量约定（有效 scale = 1 + weight），
# 而 GGUF 存完整 gamma。这些参数装载时要 -1（见 load_model 内注释）。
_RMS_DELTA_SUFFIXES = (
    ".hc_norm.weight",                      # attn/mlp_hyper_connection + output mixer
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",             # QSA 层
    ".self_attn.k_norm.weight",
    ".indexer.q_layernorm.weight",
    ".indexer.k_layernorm.weight",
)


def _release_mmap_pages(shards, log=print) -> None:
    """把 GGUF 的映射页还给内核（MADV_DONTNEED）。

    为什么必须做：pack 得整个待在「显存 + RAM」里（59.77 GiB），而 WSL 只有 58 GiB RAM。
    显存压到 16 GiB 时 RAM 需要 43.8 GiB，再叠上加载期读过的 ~15 GiB 映射页就是 60 GiB
    —— 实测直接换页卡死（available 归零，6 分钟走不出加载）。这些页只是加载时路过，
    基座已经在显存里了，留着纯属白占。

    唯一还会读它们的是 PLE（`MmapNgramEmbedding` 本来就是按行读、表不驻留），
    所以代价是每步几 MB 的读，换来小显存档能成立。
    """
    import mmap as _mmap

    for s in shards:
        mm = getattr(s, "_mm", None)
        buf = getattr(mm, "_mmap", None) if mm is not None else None
        if buf is None or not hasattr(buf, "madvise"):
            continue
        try:
            buf.madvise(_mmap.MADV_DONTNEED)
        except (OSError, ValueError) as exc:
            log(f"[pack] 页释放失败（{type(exc).__name__}: {exc}），继续")
    if shards:
        log(f"[pack] 已归还 {len(shards)} 个 GGUF 分片的映射页（PLE 之后按行重读，几 MB/步）")


def load_model(gguf_paths: list[str], pack_path: str, mapping_path: str = "mapping_q4.json",
               log=print):
    """返回 (model, meta)。meta 含 config / pack 信息，供 sft 打印与断言。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from q4_map_check import build_config  # 复用同一 config 构造（单一事实来源）

    shards = [GgufShard(p) for p in gguf_paths]
    kv = {}
    for s in shards:
        kv.update(s.kv)
    cfg = build_config(kv)

    import q4_map_check  # 确保模块可导入
    from transformers.models.qwen4_exp import Qwen4ExpForCausalLM
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    with torch.device("meta"):
        model = Qwen4ExpForCausalLM._from_config(cfg)

    # --- 换掉不能驻留的部分（在 to_empty 之前） ---
    def find_ple_layer():
        for lid in cfg.ple_layer_ids:
            yield lid - 1

    ple_layers = list(find_ple_layer())
    shard = next(s for s in shards if s.find("per_layer_token_embd.weight"))
    for li in ple_layers:
        layer = model.model.layers[li]
        hf_ref = layer.ple.ple_embedding
        layer.ple.ple_embedding = MmapNgramEmbedding(hf_ref, shard, "per_layer_token_embd.weight")

    pack_meta = json.load(open(pack_path + ".json", encoding="utf-8"))
    experts_offsets = _expert_offsets_from_meta(pack_meta)
    for li, layer in enumerate(model.model.layers):
        old = layer.mlp.experts
        layer.mlp.experts = StreamingNF4Experts(
            old.num_experts, old.hidden_dim, old.intermediate_dim, old.act_fn,
            pack_path, li, experts_offsets[li])

    model.to_empty(device="cuda")
    model = model.to(torch.bfloat16)

    # --- 非专家加载 ---
    mapping = json.load(open(mapping_path, encoding="utf-8"))
    params = dict(model.named_parameters())
    loaded = 0
    for gname, hname in sorted(mapping.items()):
        if re.match(r"model\.layers\.\d+\.mlp\.experts\.", hname):
            continue                                    # routed experts live in the NF4 pack
        if "index_qk_proj" in hname:
            continue                                    # q+k concat handled by _load_specials
        src = next((s for s in shards if s.find(gname) is not None), None)
        if src is None:
            raise KeyError(gname)
        t = src.by_name[gname]
        arr = dequant(src.raw(gname), t["type"], t["ne"])
        if hname.endswith(_RMS_DELTA_SUFFIXES):
            # 语义换算（2026-10-29，dbg_l0 实锤）：GGUF 存的是完整 gamma（= 1 + w，
            # gr_parity.cpp "gamma stored as (1 + w)"，引擎内核把存值当 plain scale）；
            # 而 HF Qwen4ExpTextRMSNorm 的有效 scale 是 (1 + weight)（modeling 第 171 行
            # output * (1.0 + self.weight)）。直接装载会让所有这类 norm 的有效 gamma
            # 变成 (1 + gamma) ≈ 2×：GR mixed 实测 1.9×（cos 0.96）、injection 门控
            # 小 ~10-20×（proj 负值翻倍 → sigmoid 饱和）、L0 MoE 输入 40.66 vs 引擎
            # 20.35。参数必须存增量 w = gamma - 1。
            # 波及：hc_norm（attn/mlp/output mixer）、ple.norm_key/query/conv、
            # QSA 的 q_norm/k_norm、indexer 的 q/k_layernorm。
            # 不含 GDN 的 ssm_norm（RMSNormGated 是 plain 约定，双方一致）。
            arr = arr - 1.0
        if hname.endswith(".linear_attn.A_log"):
            # 语义换算（2026-10-29）：GGUF 的 ssm_a 已是 a = -exp(A_log)（引擎
            # elementwise_parity.cpp 同款约定：gate = softplus(...)*a，不再 exp）；
            # HF 的 A_log 是对数，forward 里要 -A_log.exp()。直接装载会得到
            # exp(-158)≈0 ⇒ 零衰减 ⇒ GDN 递归永不遗忘 ⇒ 输出崩坏。
            assert (arr < 0).all(), f"ssm_a 应全负（=-exp(A_log)）：{gname} {arr.min()}..{arr.max()}"
            arr = np.log(-arr)
        p = params[hname]
        if hname.endswith(".conv1d.weight"):
            arr = arr.reshape(p.shape)                 # (10240,4) -> (10240,1,4)
        elif "shared_expert_gate" in hname:
            arr = arr.reshape(1, -1)
        elif "index_qk_proj" in hname:
            # q 已在本键（rows 0..511）；k 在 INDEXER_K_HALF 里单独补
            pass
        assert tuple(arr.shape) == tuple(p.shape), (gname, arr.shape, p.shape)
        with torch.no_grad():
            p.copy_(torch.from_numpy(np.ascontiguousarray(arr)).to("cuda", torch.bfloat16))
        src.drop(gname)                     # 这一个张量的映射页已经用不着了（见 GgufShard.drop）
        loaded += 1

    # indexer k_proj 后半 + gate/up 对半 + ple 表由调用方按 SPECIAL 处理
    _load_specials(shards, mapping, params, log)

    # to_empty() 会清零全部 buffer（persistent=False 也不例外），rotary 的 inv_freq
    # 因此全零：freqs = pos*0 = 0 → cos(0)=1, sin(0)=0，位置信息整个消失，输出乱码
    # 但分布形状正常（2026-10-29 dbg_layers/dbg_audit 实锤）。按 HF 自己的初始化重建。
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _M

    for name, mod in model.named_modules():
        buf = getattr(mod, "inv_freq", None)
        if isinstance(buf, torch.Tensor) and float(buf.abs().max()) == 0:
            fresh = _M.Qwen4ExpTextRotaryEmbedding(mod.config, device=str(buf.device))
            mod.inv_freq.copy_(fresh.inv_freq)
            orig = getattr(mod, "original_inv_freq", None)
            if isinstance(orig, torch.Tensor):
                orig.copy_(fresh.original_inv_freq)
            log(f"rotary inv_freq rebuilt: {name}")

    install_gdn_tile_mapping(model, log)
    # QSA 索引器：把逐 query 的 Python 循环换成一次性向量化（见 qsa_fast.py）。
    # T=2048 一步原本因此产生 2.4e4 次 DtoH 同步，GPU 空转 74%。
    try:
        import qsa_fast

        qsa_fast._apply(_M.Qwen4ExpTextQSAIndexer, log)
    except Exception as exc:                  # 换了 transformers 版本就退回原实现
        log(f"QSA indexer: 未打补丁（{type(exc).__name__}: {exc}）")

    model.eval()
    # 基座全部搬到显存之后，GGUF 那 15 GiB 映射页就只剩 PLE 会碰 —— 先还给内核，
    # 否则小显存档（RAM 侧要装下整个 pack）会被它们挤进换页。
    _release_mmap_pages(shards, log)
    return model, {"config": cfg, "pack": pack_meta}


def install_gdn_tile_mapping(model, log=print) -> bool:
    """GDN 的 key-head 配对：HF 的 repeat_interleave(j//3) → 引擎的 tile(j%16)。

    实锤（2026-10-29，dbg_tile_fwd / dbg_tile_test，同一份 GGUF、同一进程内切换）：
                       tile (j%16)      interleave (j//3)
      CE(' Paris')        0.474              1.834
      CE(' 8')            0.030              0.988
      CE('，')            1.056             10.652
      mean CE             0.520              4.491
      vs 引擎 logits      0.9841             0.44   (pearson, 248320 维)
    引擎 gdn_rec_kernel 用 qh = head % HK（kernels.cu:489 等 16 处），HF transformers
    5.17 的 Qwen4ExpTextGatedDeltaNet 用 repeat_interleave(HV//HK)（值头 j 配键头
    j//3）。两者不是行置换关系 —— 值头分组不同：{j, j+16, j+32} vs {3h, 3h+1, 3h+2}，
    所以改权重行序补救不了，只能改配对逻辑。以引擎 + 实测 CE 为准（引擎能正常生成
    文本、CE 低两个量级）。上游 llama.cpp master 的 qwen3next.cpp 也是 interleave，
    与本模型不符 —— 这条差异值得给上游报，但训练必须与引擎一致，否则 LoRA 训在
    另一条函数上。

    实现：patch 模块级符号 torch_chunk_gated_delta_rule / torch_recurrent_gated_delta_rule，
    把入参 q/k（(B,T,HV,D)，interleave 约定）重排成 tile 约定。v 不动（值头索引一致）。
    STRATA_GDN_TILE=0 可关闭（A/B 用）。
    """
    if os.environ.get("STRATA_GDN_TILE", "1") == "0":
        return False
    cfg = model.config
    hk = int(getattr(cfg, "linear_num_key_heads", 0) or 0)
    hv = int(getattr(cfg, "linear_num_value_heads", 0) or 0)
    if not hk or not hv or hv % hk or hv == hk:
        return False
    g = hv // hk
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _M

    if getattr(_M, "_strata_tile_installed", False):
        return True
    cache: dict = {}

    def tile(t):
        n = t.shape[-2]
        key = (n, t.device)
        idx = cache.get(key)
        if idx is None:
            ar = torch.arange(n, device=t.device)
            idx = (ar % hk) * g        # tile 要的键头 h = 值头 j % hk；interleave 里键头 h 的代表行是 h*g
            cache[key] = idx
        return t.index_select(-2, idx)

    def make(orig):
        def f(q, k, v, *a, **kw):
            return orig(tile(q), tile(k), v, *a, **kw)
        return f

    n = 0
    for fname in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        orig = getattr(_M, fname, None)
        if orig is None:
            continue
        setattr(_M, fname, make(orig))
        n += 1
    _M._strata_tile_installed = True
    log(f"GDN key-head mapping: tile (值头 j 配键头 j%{hk}, HV={hv} HK={hk}) —— 与引擎一致，"
        f"patched {n} kernel symbols")
    return True


def _load_specials(shards, mapping, params, log):
    from q4_map_check import rule_map

    def read(gname):
        s = next(s for s in shards if s.find(gname) is not None)
        t = s.by_name[gname]
        return dequant(s.raw(gname), t["type"], t["ne"])

    # 1) indexer k_proj -> index_qk_proj 后 128 行
    by_layer = defaultdict(dict)
    for gname in list(mapping) + [t["name"] for s in shards for t in s.tensors]:
        m = re.match(r"blk\.(\d+)\.indexer\.(k_proj|q_proj)\.weight", gname)
        if m:
            by_layer[int(m.group(1))][m.group(2)] = gname
    for l, d in by_layer.items():
        hname = f"model.layers.{l}.self_attn.indexer.index_qk_proj.weight"
        p = params[hname]
        q = read(d["q_proj"])          # (512, 2560)
        k = read(d["k_proj"])          # (128, 2560)
        with torch.no_grad():
            p.copy_(torch.from_numpy(np.ascontiguousarray(np.concatenate([q, k], 0)))
                    .to("cuda", torch.bfloat16))

    # 2) gate/up 对半 -> experts.gate_up_proj 由 StreamingNF4Experts 消费，跳过
    #    （参数仍留在骨架里 —— StreamingNF4Experts 替换后这些参数已不存在）


def _expert_offsets_from_meta(meta: dict) -> dict:
    E, layer_bytes, per = meta["experts"], meta["layer_bytes"], meta["per_expert"]
    offs = {}
    for l in range(meta["layers"]):
        base = l * layer_bytes
        d = {}
        for e in range(E):
            o = base + e * sum(per.values())
            d[("gu", e)] = (o, per["gu"])
            d[("gu_absmax", e)] = (o + per["gu"], per["gu_absmax"])
            o2 = o + per["gu"] + per["gu_absmax"]
            d[("dn", e)] = (o2, per["dn"])
            d[("dn_absmax", e)] = (o2 + per["dn"], per["dn_absmax"])
        offs[l] = d
    return offs


def export_tokenizer(gguf_path: str, base_hf: str, out_dir: str) -> int:
    """把分片里的词表导出成一份 HF tokenizer 目录（`train.py` 默认找 `<gguf 目录>/tokenizer`）。

    GGUF 只存 `tokens` + `merges`；BPE 的正规化/预切分/chat 模板在 `base_hf` 那份
    `tokenizer.json` 里。所以分工是：base 决定「怎么切」，GGUF 决定「有哪些 id」。

    两份词表未必一样长（这份 Swift-Qwen3.8 比 Qwen3.6 多 250 个：7 个 audio/TTS 特殊词 +
    243 个 `[PAD<id>]` 槽位）。多出来的按 id 顺序补成 added token，**但前缀必须逐 id 相等** ——
    不相等就是真的换了一张词表，那时应该报错，而不是把 base 的切法硬套到别的 id 上。
    """
    import shutil
    from pathlib import Path

    from tokenizers import Tokenizer

    sh = GgufShard(gguf_path)
    toks = sh.kv.get("tokenizer.ggml.tokens")
    if not toks:
        raise SystemExit(f"{gguf_path}: 没有 tokenizer.ggml.tokens")
    base = Path(base_hf)
    if not (base / "tokenizer.json").exists():
        raise SystemExit(f"{base_hf}: 不是 HF tokenizer 目录（缺 tokenizer.json）")

    t = Tokenizer.from_file(str(base / "tokenizer.json"))
    vocab = t.get_vocab()
    inv = {i: s for s, i in vocab.items()}
    nbase = max(vocab.values()) + 1
    n = min(len(toks), nbase)
    diff = [i for i in range(n) if inv.get(i) != toks[i]]
    if diff:
        i = diff[0]
        raise SystemExit(
            f"词表不是同一张：{len(diff)}/{n} 个 id 不同，首个 id {i}: "
            f"GGUF={toks[i]!r} base={inv.get(i)!r}\n"
            "  换一个 base（同族、同词表的 HF 目录）再导。")
    tail = list(toks[nbase:]) if len(toks) > nbase else []
    if tail:
        added = t.add_tokens(tail)
        if added != len(tail):
            raise SystemExit(f"补 token 时只新增了 {added}/{len(tail)} 个（有重复？）")
    bad = [i for i in range(len(toks)) if t.token_to_id(toks[i]) != i]
    if bad:
        raise SystemExit(f"{len(bad)} 个 id 与 GGUF 不一致，首个 {bad[0]}，不写盘")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 只抄 tokenizer 那一组文件。**不要** `iterdir()` 全抄：base 通常是完整的 HF 权重目录，
    # 里面有几十个 safetensors 分片，全抄等于把几十 GB 复制一份（实测踩过）。
    keep = {"tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
            "added_tokens.json", "chat_template.jinja", "preprocessor_config.json",
            "generation_config.json", "LICENSE", "LICENSE.txt"}
    for f in base.iterdir():
        if f.is_file() and f.name in keep:
            shutil.copy2(f, out / f.name)
    t.save(str(out / "tokenizer.json"))
    tmpl = sh.kv.get("tokenizer.chat_template")   # GGUF 里的模板才是这个模型真实的模板
    if tmpl:
        (out / "chat_template.jinja").write_text(tmpl, encoding="utf-8")
    print(f"[tokenizer] {len(toks)} 个 id（base {nbase} + 补 {len(tail)}）-> {out}")
    print(f"[tokenizer] bos={sh.kv.get('tokenizer.ggml.bos_token_id')} "
          f"eos={sh.kv.get('tokenizer.ggml.eos_token_id')} 模板={'有' if tmpl else '无'}")
    return 0


def main() -> int:
    if sys.argv[1] == "convert":
        args = sys.argv[2:]
        pack_out = args[args.index("--pack-out") + 1]
        ggufs = [a for a in args if a.endswith(".gguf")]
        convert_experts(ggufs, pack_out)
        return 0
    if sys.argv[1] == "tokenizer":
        args = sys.argv[2:]
        opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
        gguf = [a for a in args if a.endswith(".gguf")]
        if not gguf or "--out" not in opt:
            print("用法: loader_qwen4.py tokenizer <gguf> --base <HF 目录> --out <目录>")
            return 1
        base = opt.get("--base")
        if not base:
            print("需要 --base <HF tokenizer 目录>：GGUF 不存 BPE 的切分规则，只存词表。")
            return 1
        return export_tokenizer(gguf[0], base, opt["--out"])
    print("usage: loader_qwen4.py convert <gguf...> --pack-out <bin>\n"
          "       loader_qwen4.py tokenizer <gguf> --base <HF 目录> --out <目录>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
