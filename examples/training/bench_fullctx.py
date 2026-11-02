"""满上下文标准速度基准：多算法 × 多上下文长度，单进程一次加载。

为什么要有这个文件
------------------
历史测速散在 sft_mezo / sft / _mb_cost_split 里，每份都用不同的 token 口径
（answer 段 / completion 段 / fwd 段），于是同一份日志能读出 9 tok/s 和 30 tok/s，
DPO 被误判成「慢几十倍」。这里把口径焊死成一条：

    tok/s = 每步**前向真正处理的 token 数**（行数 × 序列长度） / 步时

并且每步的「前向次数」按算法的真实定义计：

    ce_bwd    1 次前向 + 1 次反向（反向 ≈ 2 次前向的 dispatch，但这里只报墙钟）
    ce_mezo   2 次前向（+ε / −ε）
    dpo_bwd   2 行（chosen / rejected）同批 1 次前向 + 1 次反向
    dpo_mezo  2 行同批，2 次前向（±ε）
    grpo_bwd  K 行（同组轨迹）同批 1 次前向 + 1 次反向
    grpo_mezo K 行同批，2 次前向（±ε）

**不拼接上下文**：DPO 的 chosen / rejected 各自是独立的一行（长度 T），只在
**batch 维**合批。所以满上下文 DPO 每行仍是 T，不是 2T —— 拼接成 2T 会让
attention 变 4 倍且语义不同（rejected 会看到 chosen）。

用法
----
    python bench_fullctx.py --gguf <分片...> --pack experts_nf4.bin \
        --tokenizer <tok 目录> --seqlens 2048,8192,32768 --modes mezo,bwd \
        --algos ce,dpo,grpo --group 4 --grad-ckpt on --reps 2 \
        --out bench_fullctx.json

输出：每个 (算法, 模式, 长度) 的步时 / token / tok/s / 峰值显存，并追加一张
markdown 表到 stdout。OOM 不算崩，记成 `oom` 继续下一个口径。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import threading
import time


import torch

# 预热 fla：hub_kernels 在**导入时**用 getattr 链找 GDN 实现，fla 的 ops 是惰性挂载的，
# 没先 import 子模块时链断掉 ⇒ 静默回落 torch 参考实现（慢 4% 但显存会累积到 OOM）。
try:
    import fla.ops.gated_delta_rule  # noqa: F401
    _FLA = "ok"
except Exception as _exc:                                  # pragma: no cover
    _FLA = f"{type(_exc).__name__}: {_exc}"

import loader_qwen4 as L
import mem_guard
from lora import LoRALinear, freeze_all, inject_lora, trainable_summary
from sft import text_model
from sft_mezo import _Pair, _Traj, dpo_measure, grpo_loss, perturb, zo_cast_fp32

VOCAB = 248320


# ---------------------------------------------------------------- 通用工具


def blp(hidden: torch.Tensor, head, labels: torch.Tensor, chunk: int = 0) -> torch.Tensor:
    """(R,T,H) → 每个位置 log p(label)，保留计算图。

    logits 显存 = R × chunk × 248320 × 4 B。固定给 256 MiB 预算，否则 T 大时
    head 这一步就能顶掉几个 GiB。
    """
    R, T = hidden.shape[0], hidden.shape[1]
    if chunk <= 0:
        chunk = max(4, min(128, 256 // max(1, R)))
    outs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lg = head(hidden[:, s:e]).float()
        picked = lg.gather(-1, labels[:, s:e, None]).squeeze(-1)
        outs.append(picked - torch.logsumexp(lg, dim=-1))
    return torch.cat(outs, 1)


def ce_scalar(body, head, ids: torch.Tensor, plen: int) -> torch.Tensor:
    """SFT 的标量目标（answer 段 mean CE）。与 sft_mezo.sample_loss 同一路径（全段 head）。"""
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state
    labels = ids[:, 1:]
    lp = blp(h, head, labels)
    mask = torch.zeros_like(lp, dtype=torch.bool)
    mask[:, plen - 1:] = True
    return -lp[mask].mean()


# 本机（Windows + torch 2.13）显存超发时，OOM 抛的是 torch.AcceleratorError
# ("CUDA error: out of memory")，不是 torch.cuda.OutOfMemoryError —— 只接后者会让一个
# 本该记成 oom 的口径直接把整个进程打崩在 native 栈上（今天 bwd/dpo T=8192 就是这样）。
_OOM_ERRORS = tuple(e for e in (getattr(torch, "OutOfMemoryError", None),
                                getattr(torch, "AcceleratorError", None),
                                RuntimeError) if e is not None)

FORCE_MASK = False          # --att-mask always：用来量化「4D mask 掉出 sdpa 快路径」的代价


def completion_sums(body, head, ids: torch.Tensor, att: torch.Tensor, plens: list[int]):
    """DPO 的 per-row completion sum logprob（保留图版，与 _completion_sums 同算法）。"""
    am = att if FORCE_MASK else (None if bool(att.all()) else att)
    h = body(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    R, T = ids.shape
    start = max(0, min(plens) - 1)
    lp = blp(h[:, start:T - 1], head, ids[:, start + 1:T])
    mask = torch.zeros_like(lp)
    for i, pl in enumerate(plens):
        n = int(att[i].sum())
        lo, hi = pl - 1 - start, n - 1 - start
        if hi > lo:
            mask[i, max(0, lo):max(0, hi)] = 1.0
    return (lp * mask).sum(1)


def timed(fn, warmup: int, reps: int) -> tuple[float, int]:
    """返回 (中位墙钟, 峰值显存 GiB)。CUDA 异步 ⇒ 前后都 synchronize。"""
    for _ in range(warmup):
        fn()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    return statistics.median(ts), peak


def synth(T: int, rows: int, plen: int, device, seed: int = 0):
    """合成等长序列：只测「形状成本」，与真实数据分布无关（这是基准，不是评测）。

    prompt 段故意让所有行共享（GRPO 的定义就是同 prompt 的 K 条采样），
    answer 段各行不同 —— 这样专家命中数接近真实，不会低估 router 的并集。

    若 `--tokens-npy` 给了预分词语料，则改从真实 token 取窗口（`real_ids`），
    此时 `synthetic=False`，tok/s 才允许进报告。
    """
    if _CORPUS is not None:
        return real_ids(T, rows, device, seed)
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(100, 200000, (rows, T), device=device, generator=g)
    att = torch.ones((rows, T), dtype=torch.long, device=device)
    return ids, att


# ---------------------------------------------------------------- 真实语料 / 内存采样 / 看门狗
#
# 三件事放在这里，都是为了「跑满阶梯而不被一格拖死」：
#   1. 真实语料：用离线分词好的 .npy，不在基准进程里 tokenize（版本一漂数字就不可比）；
#   2. 宿主内存峰值：pack staging / pinned 全在 RSS 里，显存看不到；
#   3. 看门狗：`ce_bwd@16K` 有过「不报错、不退出、GPU 100%、日志零行」的前科。
#      同进程里杀不掉挂住的 kernel，但可以**落盘 + 自杀**，让外层从下一档续跑（--resume）。

_CORPUS = None                  # 1D int64 tensor（预分词，CPU）
_CORPUS_NAME = ""
_REPEAT: dict[int, int] = {}    # T → 语料平铺了几遍（报告里的 repeat）


def load_corpus(path: str) -> int:
    """加载预分词后的 .npy（一维 int32/int64）。不做 tokenize，理由见上。"""
    global _CORPUS, _CORPUS_NAME
    import numpy as np

    a = np.load(path, mmap_mode="r")
    flat = np.asarray(a).reshape(-1)
    _CORPUS = torch.from_numpy(flat.astype("int64").copy())
    _CORPUS_NAME = os.path.basename(path)
    return int(_CORPUS.numel())


def real_ids(T: int, rows: int, device, seed: int = 0):
    """从真实语料切 rows 个长度 T 的窗口（各自随机偏移），并记 repeat 次数。

    语料不足 T×rows 时整体平铺——这就是报告里 `repeat: n` 的来源：形状与 token 数不变，
    算子路径与显存不变，只是别把这些 token 说成「真实语料吞吐」。
    """
    assert _CORPUS is not None
    n = int(_CORPUS.numel())
    reps = max(1, (T * rows + n - 1) // n)
    _REPEAT[T] = reps
    src = _CORPUS.repeat(reps) if reps > 1 else _CORPUS
    span = max(1, int(src.numel()) - T)
    g = torch.Generator().manual_seed(seed)
    ids = torch.empty((rows, T), dtype=torch.long)
    for i in range(rows):
        s = int(torch.randint(0, span, (1,), generator=g))
        ids[i] = src[s:s + T]
    att = torch.ones((rows, T), dtype=torch.long, device=device)
    return ids.to(device), att


def rss_gib() -> float:
    """当前进程宿主常驻内存（GiB）。不依赖 psutil：Windows 走 psapi，Linux 走 statm。"""
    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes as wt

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / 2 ** 30
        except Exception:
            return 0.0
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except Exception:
        return 0.0


class RSSSampler(threading.Thread):
    """后台采宿主内存峰值。显存峰值只能看到卡住的部分，staging/pinned 都在这里。"""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = rss_gib()
        self._stop = threading.Event()

    def run(self):                       # pragma: no cover - 线程体
        while not self._stop.wait(self.interval):
            v = rss_gib()
            if v > self.peak:
                self.peak = v

    def sample(self) -> float:
        v = rss_gib()
        if v > self.peak:
            self.peak = v
        return self.peak

    def stop(self) -> float:
        self.sample()
        self._stop.set()
        return self.peak


def estimate_step_seconds(prev, T: int) -> float | None:
    """把「上一档的步时」放大到这一档：`s0 × (T/T0) ** 1.25`。

    直接用上一档的步时会误杀：`ce_mezo` 从 4608 跳到 27648 是 6 倍，而注意力与专家读取是超线性的。
    实测 ce@4608 = 20.8 s、ce@27648 = 146.6 s（7.0 倍，指数 ≈ 1.09），但 dpo@27648 比 ce 慢得多，
    同一档撞上 180 s 上限就被记成 hang。指数取 1.25 给的是**上界方向**的估计，宁可等也不误杀。
    """
    if not prev:
        return None
    s0, t0 = prev
    if not s0 or s0 <= 0:
        return None
    if not t0 or t0 <= 0:
        return float(s0)
    return float(s0) * (float(T) / float(t0)) ** 1.25


def estimate_for_cell(prev_s: dict, algo: str, mode: str, T: int) -> float | None:
    """这一格的步时估计：优先用同一 (算法, 模式) 的上一档，放大到目标 T。

    按方案分格之后，`dpo_mezo` 只测 27648/55808 两格，它自己**没有更小的档**可参考——那就退到
    「所有已测格里按 T 放大后最贵的那个」。宁可用别的算法的高估，也不能让第一格顶着 180 s 的空上限
    （dpo@27648 实测要 ~300 s，顶 180 s 就会被误记成 hang）。
    """
    own = estimate_step_seconds(prev_s.get((algo, mode)), T)
    if own is not None:
        return own
    others = [s for s in (estimate_step_seconds(v, T) for v in prev_s.values()) if s is not None]
    return max(others) if others else None


def cell_timeout_seconds(args, ref_s: float | None) -> float:
    """单格墙钟上限。auto = 3 × max(60, 估计步时 × 2.5)；`off` 关闭。

    `ref_s` 必须是**已按目标 T 放大过的估计**（见 `estimate_step_seconds`），不是上一档的原始步时。
    用同一 (算法, 模式) 的估计——不同算法差几个数量级，用全局上一步会误杀。
    """
    v = str(args.per_cell_timeout).strip().lower()
    if v in ("off", "0", "none", ""):
        return 0.0
    if v != "auto":
        return float(v)
    base = 60.0 if not ref_s else max(60.0, float(ref_s) * 2.5)
    return 3.0 * base


def _save_rows(path: str, args, rows: list[dict]) -> None:
    """原子落盘：看门狗会在任意时刻调用，半截 JSON 比没有更糟。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in vars(args).items() if k != "T"},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def arm_watchdog(args, cell: dict, rows: list[dict], sampler: RSSSampler):
    """给一格装超时。触发时：记 `hang` → 落盘 → 进程自杀（退出码 3），外层 --resume 续跑。

    为什么自杀而不是抛异常：挂住的是 CUDA kernel，Python 侧抛不出去；而同进程继续跑下一档
    会把「挂死」变成「看起来慢」。落盘 + 退出码是最小可信信号。
    """
    sec = cell_timeout_seconds(args, cell.get("ref_s"))
    if sec <= 0:
        return None

    def fire():                                   # pragma: no cover - 只在真挂死时跑
        rows.append(dict(algo=cell["algo"], mode=cell["mode"], T=cell["T"],
                         toks=cell["toks"], status="hang",
                         note="watchdog %.0fs（不报错不退出）" % sec,
                         peak_rss=round(sampler.sample(), 2),
                         synthetic=_CORPUS is None, repeat=_REPEAT.get(cell["T"], 0)))
        try:
            _save_rows(args.out, args, rows)
        except Exception as exc:
            sys.stderr.write("[hang] 落盘失败：%s\n" % exc)
        sys.stderr.write("\n[hang] %s/%s T=%d 超 %.0fs，已记 hang 并落盘；"
                         "用 --resume 从下一档续跑。\n"
                         % (cell["algo"], cell["mode"], cell["T"], sec))
        sys.stderr.flush()
        os._exit(3)

    t = threading.Timer(sec, fire)
    t.daemon = True
    t.start()
    return t


def run_cell(args, rows, done, sampler, prev_s, mode, algo, fn, toks, note, T) -> None:
    """跑一格并把六个数字记全（步时 / token / tok/s / 峰值显存 / 峰值 RSS / 状态）。

    以前 mezo 和 bwd 两段各抄一遍，于是 bwd 分支里 `toks` 在赋值前就被引用了（预算拒绝那支
    NameError）。这里合成一条路径，两边不可能再走岔。
    """
    key = (algo, mode, T)
    if key in done:
        print("  [%s] %-5s T=%6d 已在 %s 里，跳过（--resume）"
              % (mode, algo, T, os.path.basename(args.out)), flush=True)
        return

    base = dict(algo=algo, mode=mode, T=T, toks=toks,
                synthetic=_CORPUS is None, repeat=_REPEAT.get(T, 0))

    if not _preflight(args, algo, mode, T):
        rows.append(dict(base, status="budget", note="预算拒绝",
                         peak_rss=round(sampler.sample(), 2)))
        print("  [%s] %-5s T=%6d  skipped:budget" % (mode, algo, T), flush=True)
        return

    ref = estimate_for_cell(prev_s, algo, mode, T)
    wd = arm_watchdog(args, dict(base, ref_s=ref), rows, sampler)
    try:
        dt, peak = timed(fn, args.warmup, args.reps)
        row = dict(base, s=dt, tok_s=toks / dt, peak=peak,
                   peak_rss=round(sampler.sample(), 2), note=note, status="ok")
        prev_s[(algo, mode)] = (dt, T)      # 步时和它所在的 T 一起记：下一档要按比例放大
    except _OOM_ERRORS as exc:
        import traceback as _traceback
        # 记下分配点的 traceback：OOM 报的字节数只能反推形状，改哪一行要看栈。
        try:
            with open(args.out + ".oom.txt", "w", encoding="utf-8") as f:
                f.write("cell %s/%s T=%d\n\n%s\n\n%s\n"
                        % (algo, mode, T, _traceback.format_exc(),
                           torch.cuda.memory_summary() if torch.cuda.is_available() else ""))
        except OSError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        if args.on_oom == "abort":
            if wd is not None:
                wd.cancel()
            raise SystemExit(_oom_abort(exc, algo, mode, T))
        row = dict(base, status="oom", note="%s: %s" % (type(exc).__name__, str(exc)[:120]),
                   peak_rss=round(sampler.sample(), 2))
    finally:
        if wd is not None:
            wd.cancel()

    rows.append(row)
    ss = "%8.2fs" % row["s"] if "s" in row else "     oom "
    ts = "%8.1f tok/s" % row["tok_s"] if "tok_s" in row else " " * 13
    print("  [%s] %-5s T=%6d %s %s  vram %5.1f GiB  rss %5.1f GiB  %s"
          % (mode, algo, T, ss, ts, row.get("peak", 0.0), row.get("peak_rss", 0.0), row["status"]),
          flush=True)
    _save_rows(args.out, args, rows)          # 每格落盘 ⇒ 中途崩也留得住已测的格子
    gc.collect()
    torch.cuda.empty_cache()



def undo_zo_cast(model):
    """把 zo_cast_fp32 的实例 forward 摘掉、A/B 转回 bf16 —— 反传档要用原生路径。"""
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and "forward" in m.__dict__:
            del m.forward
            m.A.data = m.A.data.bfloat16()
            m.B.data = m.B.data.bfloat16()
            n += 1
    return n


# ---------------------------------------------------------------- 各算法的「一步」


def _oom_abort(exc, algo, mode, T):
    """--on-oom abort：打印现场并返回退出码，不再继续（Windows 超发会变卡死）。"""
    sys.stderr.write("\n[oom] %s/%s T=%d：%s: %s\n  按 --on-oom abort 终止；"
                     "要看完整边界矩阵请加 --on-oom skip。\n"
                     % (algo, mode, T, type(exc).__name__, str(exc)[:160]))
    return 2


def _preflight(args, algo, mode, T):
    """开跑前预算：超物理上限就拒绝，避免「GPU 满载 + 日志零行」的卡死。"""
    rows = 1 if algo == "ce" else (2 if algo == "dpo" else max(1, args.group))
    est = mem_guard.estimate_peak_gib(T, rows=rows, mode=mode)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print("  [预算] %s/%s T=%d rows=%d 估算 %.1f / %.1f GiB"
          % (algo, mode, T, rows, est, total), flush=True)
    if est > total * 0.95:
        sys.stderr.write("\n[预算] %s/%s T=%d 估算 %.1f GiB 超上限\n"
                         % (algo, mode, T, est))
        return False
    return True


def make_steps(body, head, args, device, opt=None):
    """按算法/模式返回 (step_fn, fwd_tokens, note)。step_fn 无参数，跑完算一步。"""

    T = args.T
    plen = max(2, int(T * args.prompt_frac))
    steps = {}

    # ---- ce：1 行，整段前向，answer 段算 loss ----
    ids1, att1 = synth(T, 1, plen, device)

    def ce_mezo():
        ce_scalar(body, head, ids1, plen)

    def ce_bwd():
        opt.zero_grad(set_to_none=True)
        ce_scalar(body, head, ids1, plen).backward()
        opt.step()

    steps["ce_mezo"] = (ce_mezo, T, "1 行 × T")
    steps["ce_bwd"] = (ce_bwd, T, "1 行 × T")

    # ---- dpo：chosen / rejected 两行独立（不拼接），batch 维合批 ----
    ids2, att2 = synth(T, 2, plen, device, seed=1)
    pair = _Pair(ids_c=ids2[:1].clone(), ids_r=ids2[1:].clone(), n_prompt=plen,
                 ref_c=0.0, ref_r=0.0)

    def dpo_mezo():
        dpo_measure(body, head, [pair], args.dpo_beta)

    ids_cat = torch.cat([pair.ids_c, pair.ids_r], 0)
    att_cat = torch.cat([att2[:1], att2[:1]], 0)

    def dpo_bwd():
        opt.zero_grad(set_to_none=True)
        sums = completion_sums(body, head, ids_cat, att_cat, [plen, plen])
        margin = (sums[0] - pair.ref_c) - (sums[1] - pair.ref_r)
        (-torch.nn.functional.logsigmoid(args.dpo_beta * margin)).backward()
        opt.step()

    steps["dpo_mezo"] = (dpo_mezo, 2 * T, "2 行 × T（chosen+rejected 各一行）")
    steps["dpo_bwd"] = (dpo_bwd, 2 * T, "2 行 × T 同批")

    # ---- grpo：K 行同组轨迹，同批一次前向 ----
    K = args.group
    idsK, attK = synth(T, K, plen, device, seed=2)
    group = [_Traj(ids=idsK[i:i + 1].clone(), n_prompt=plen,
                   old=torch.zeros(T - plen, device=device),
                   adv=(1.0 if i % 2 == 0 else -1.0), reward=0.0)
             for i in range(K)]

    def grpo_mezo():
        grpo_loss(body, head, group, args.clip, batched=True)

    def grpo_bwd():
        opt.zero_grad(set_to_none=True)
        am = attK if FORCE_MASK else (None if bool(attK.all()) else attK)
        h = body(input_ids=idsK, attention_mask=am, use_cache=False).last_hidden_state
        new = blp(h[:, plen - 1:T - 1], head, idsK[:, plen:T])
        old = torch.stack([t.old for t in group])
        adv = torch.stack([torch.full_like(t.old, t.adv) for t in group])
        ratio = torch.exp(new - old)
        obj = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - args.clip,
                                                    1 + args.clip) * adv)
        (-obj.mean(1).mean()).backward()
        opt.step()

    steps["grpo_mezo"] = (grpo_mezo, K * T, f"{K} 行 × T")
    steps["grpo_bwd"] = (grpo_bwd, K * T, f"{K} 行 × T")

    return steps


def seqlens_from_budget(path: str, cap: int = 131072) -> list[int]:
    """把 `budget.py --json` 的每方案极限翻成阶梯：每个 profile 取 {T/2, T}，去重排序。

    显式给了 `--seqlens` 时用不到它（那条路每个组合都测同一串长度）。按方案分格走
    `cells_from_budget`——公共阶梯意味着每个组合都挨个爬上所有档，那是 66 格的爬梯子。
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out: set[int] = set()
    for r in d.get("rows", []):
        t = int(r.get("max_seq") or 0)
        if t <= 0:
            continue
        for v in (t, t // 2):
            out.add(_rung(v, cap))
    return sorted(out)


def _rung(v: int, cap: int = 131072) -> int:
    return min(int(cap), max(512, int(v) // 512 * 512))


def cells_from_budget(path: str, algos: list[str], modes: list[str], cap: int = 131072) -> dict[str, list[int]]:
    """每个 `algo_mode` 只测**它自己那一行**给的两点：{T/2, T}。

    极限是算出来的（`peak(T) ≤ VRAM×0.95` 的反解），实测只负责校验公式：半程该轻松过、顶点该刚好过。
    所以「`ce_mezo` 只测 27904/55808」和「`grpo_bwd` 只测 1280/2560」是两件事，不能并成一条公共阶梯
    让每个组合都去爬——那是 6 组合 × 11 档 = 66 格，比预算多一个数量级。

    profile 名形如 `hf-ce_mezo` / `hf-grpo_bwd(ckpt,batch K=4)`：括号里是那一行的前提，不参与匹配。
    `hf-dpo_mezo` 没有单独一行（与 ce_mezo 同为「一次前向」），按 ce_mezo 那行算。
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = {}
    for r in d.get("rows", []):
        name = str(r.get("profile") or "").split("(")[0].strip()
        # 同一个 profile 可能有两行（`hf-ce_bwd(ckpt)` 与 `hf-ce_bwd(refetch)`，后者 max_seq=0）：
        # 取大的那个，否则后一行会把「能跑」覆盖成「跑不了」
        rows[name] = max(rows.get(name, 0), int(r.get("max_seq") or 0))
    out: dict[str, list[int]] = {}
    for m in modes:
        for a in algos:
            t = rows.get(f"hf-{a}_{m}", 0)
            if t <= 0 and m == "mezo" and a == "dpo":
                t = rows.get("hf-ce_mezo", 0)
            if t > 0:
                out[f"{a}_{m}"] = sorted({_rung(t, cap), _rung(t // 2, cap)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="包里的 tokenizer 目录；**当前不参与测速**（分词在 prep_pack.py 离线做，"
                         "基准只认 --tokens-npy 那份字节），保留是为了老命令不用改")
    ap.add_argument("--seqlens", default="",
                    help="显式阶梯，如 8192,16384；留空则看 --from-budget，再留空用默认 2048/8192/32768/131072")
    ap.add_argument("--algos", default="ce,dpo,grpo")
    ap.add_argument("--modes", default="mezo,bwd")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--prompt-frac", type=float, default=0.5,
                    help="prompt 占序列的比例（DPO 的偏好对 prompt 通常占八成）")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="o_proj,out_proj")
    ap.add_argument("--grad-ckpt", choices=("on", "off"), default="on")
    ap.add_argument("--att-mask", choices=("auto", "always"), default="auto",
                    help="always = 即使没有 padding 也传 4D attention_mask（复现 8× 劣化）")
    ap.add_argument("--on-oom", choices=("abort", "skip"), default="abort",
                    help="abort = 一遇 OOM 立即退出（默认）；skip = 记一行 oom 继续")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="bench_fullctx.json")
    # ---- 极限模式（阶段 B）：一格一步 + 超时自杀 + 续跑 + 真实语料 ----
    ap.add_argument("--per-cell-timeout", default="auto",
                    help="单格墙钟上限：auto=3×max(60, 同算法上一档步时×2.5)；或给秒数；off 关闭。"
                         "超时记 hang 落盘并退出(3)，外层用 --resume 续跑")
    ap.add_argument("--resume", action="store_true",
                    help="读 --out 里已有的格子，跳过已测/已挂死的 (algo,mode,T)")
    ap.add_argument("--tokens-npy", default="",
                    help="预分词好的真实语料（一维 npy）。给了才算真实 tok/s（synthetic=false）；"
                         "不给则用 synth()，报告里标 synthetic=true")
    ap.add_argument("--from-budget", default="",
                    help="budget.py --json 的输出；自动生成每个 profile 的 {T_model/2, T_model} 阶梯，"
                         "省掉手写 --seqlens（--seqlens 显式给了则以它为准）")
    args = ap.parse_args()

    # 落点断言：GDN 落在 torch 参考实现时数字不可用（而且显存会累积到超发 OOM）。
    from transformers.models.qwen4_exp import modeling_qwen4_exp as _M
    _gdn = _M.torch_chunk_gated_delta_rule
    _fallback = type(_gdn).__name__ not in ("Func", "Kernel", "function")
    print(f"GDN kernel: {type(_gdn).__name__} "
          f"({'回落参考实现 → 数字不可用' if _fallback else 'hub/fla'})  fla={_FLA}  "
          f"PYTHONUTF8={os.environ.get('PYTHONUTF8', '(unset)')}", flush=True)
    if _fallback:
        print("!! 拒绝测速：先修 PYTHONUTF8=1（中文 Windows 上 torch inductor 会用 GBK 读模板）",
              flush=True)
        return 2

    global FORCE_MASK
    FORCE_MASK = args.att_mask == "always"

    device = torch.device("cuda")
    t0 = time.time()
    model, _meta = L.load_model(args.gguf, args.pack)
    print(f"loaded {time.time() - t0:.0f}s  {torch.cuda.memory_allocated() / 2**30:.1f} GiB  "
          f"pack vram={os.environ.get('STRATA_VRAM_PACK_GB', '16')} GiB", flush=True)
    torch.cuda.empty_cache()
    model.config.use_cache = False
    freeze_all(model)
    n = inject_lora(model, targets=tuple(t for t in args.targets.split(",") if t),
                    r=args.rank, alpha=args.alpha)
    print(f"lora {n} modules  trainable {trainable_summary(model)['trainable']:,}", flush=True)
    body, head = text_model(model), model.lm_head

    algos = [a for a in args.algos.split(",") if a]
    modes = [m for m in args.modes.split(",") if m]

    # ---- 长度：显式 --seqlens 优先（每个组合都测这一串）；否则按 budget 每方案取它自己那两点 ----
    per_key: dict[str, list[int]] | None = None
    if args.seqlens:
        seqlens = [int(x) for x in args.seqlens.split(",") if x]
        src = "(命令行)"
    elif args.from_budget:
        per_key = cells_from_budget(args.from_budget, algos, modes)
        seqlens = sorted({T for v in per_key.values() for T in v})
        src = f"(budget: {os.path.basename(args.from_budget)})"
    else:
        seqlens = [2048, 8192, 32768, 131072]
        src = "(默认)"
    if not seqlens:
        sys.stderr.write("[阶梯] 没有可测的长度：budget 里全是 no-fit？\n")
        return 2

    def lengths_for(key: str) -> list[int]:
        """这个组合要测哪几档。给了 --seqlens 就都测；按 budget 分格时只测它自己那两点。"""
        return per_key[key] if per_key is not None else seqlens

    print(f"阶梯 {seqlens} {src}  超时={args.per_cell_timeout}"
          f"  reps={args.reps} warmup={args.warmup}", flush=True)
    if per_key is not None:
        print("按方案分格：" + "；".join(f"{k}→{v}" for k, v in sorted(per_key.items())), flush=True)

    # ---- 真实语料（可选）：不给就是合成，报告里 synthetic=true ----
    if args.tokens_npy:
        ncorp = load_corpus(args.tokens_npy)
        print(f"corpus {_CORPUS_NAME}: {ncorp:,} tok ⇒ tok/s 可上报（synthetic=false）", flush=True)
    else:
        print("corpus: 无（synth 合成）⇒ 这些 tok/s 只用来比形状成本，不进报告", flush=True)

    rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as f:
                old = json.load(f)
            keep = [r for r in old.get("rows", []) if r.get("status") in ("ok", "oom", "hang", "budget")]
            rows.extend(keep)
            done = {(r["algo"], r["mode"], r["T"]) for r in keep}
            print(f"resume: 已有 {len(keep)} 格，跳过它们", flush=True)
        except Exception as exc:
            print(f"resume: {args.out} 读不动（{exc}），从头来", flush=True)

    sampler = RSSSampler()
    sampler.start()
    prev_s: dict[tuple, float] = {}
    for r in rows:                       # resume 时用已测格填「上一档步时」，超时才算得准
        if r.get("status") == "ok" and "s" in r:
            k = (r["algo"], r["mode"])
            cur = prev_s.get(k)
            if cur is None or int(r["T"]) > cur[1]:      # 取 T 最大的那一格：最接近待测档
                prev_s[k] = (float(r["s"]), int(r["T"]))

    # MeZO 档：eval + no_grad + fp32 LoRA（ε=1e-3 在 bf16 下会被舍入吞掉，必须 fp32）
    if "mezo" in modes:
        model.eval()
        # 不要 model.requires_grad_(False)：那会把 inject_lora 刚建的 A/B 一起冻掉。
        # freeze_all 已经在 inject 之前把底座权重冻过了，这里的 params 只剩 LoRA。
        zo_cast_fp32(model)
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"mezo: fp32 LoRA {len(params)} tensors", flush=True)
        for a in algos:
            key = f"{a}_mezo"
            for T in lengths_for(key):
                args.T = T
                steps = make_steps(body, head, args, device, None)
                if key not in steps:
                    continue
                fn, toks, note = steps[key]
                # 零阶一步 = ±ε 两次 measure；扰动本身也要计时（它要遍历所有 LoRA 参数）。
                # no_grad 必须在**最外层**：LoRA 的 A/B 是 requires_grad=True 的 Parameter，
                # 只有 dpo_measure / grpo_loss 自己带了 no_grad，ce 这条路没有 ⇒ T=2048
                # 会建出 ~35 GiB 的激活图，直接顶爆（实测 OOM）。
                def step(fn=fn):
                    with torch.no_grad():
                        seed = 12345
                        perturb(params, seed, +1, args.eps)
                        fn()
                        perturb(params, seed, -2, args.eps)
                        fn()
                        perturb(params, seed, +1, args.eps)   # 复原 θ

                run_cell(args, rows, done, sampler, prev_s, "mezo", a, step, toks, note, T)

    # 反传档：train（HF checkpoint 的分支条件）+ bf16 LoRA（生产路径不 cast）
    if "bwd" in modes:
        undo_zo_cast(model)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-6)
        if args.grad_ckpt == "on":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for lm in (model, getattr(model, "model", None)):
                if lm is not None and hasattr(lm, "gradient_checkpointing"):
                    lm.gradient_checkpointing = True
        print(f"bwd: ckpt={args.grad_ckpt}  trainable {len(params)} tensors", flush=True)
        for a in algos:
            key = f"{a}_bwd"
            for T in lengths_for(key):
                args.T = T
                steps = make_steps(body, head, args, device, opt)
                if key not in steps:
                    continue
                fn, toks, note = steps[key]
                run_cell(args, rows, done, sampler, prev_s, "bwd", a, fn, toks, note, T)

    rss_peak = sampler.stop()
    _save_rows(args.out, args, rows)

    print("\n| 算法 | 模式 | T | 每步 token | 步时 s | **tok/s** | 峰值 GiB | RSS GiB | 状态 | 相对 ce |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    base = {(r["T"]): r for r in rows if r["algo"] == "ce" and r["mode"] == "mezo"
            and r["status"] == "ok"}
    for r in rows:
        if r["status"] != "ok":
            print(f"| {r['algo']} | {r['mode']} | {r['T']} | {r['toks']:,} | — | — | — | "
                  f"{r.get('peak_rss', 0):.1f} | **{r['status']}** | — |")
            continue
        b = base.get(r["T"])
        rel = f"{r['tok_s'] / b['tok_s']:.2f}×" if b and b["tok_s"] else "—"
        print(f"| {r['algo']} | {r['mode']} | {r['T']} | {r['toks']:,} | {r['s']:.2f} | "
              f"**{r['tok_s']:.1f}** | {r['peak']:.1f} | {r.get('peak_rss', 0):.1f} | ok | {rel} |")
    if _CORPUS is None:
        print("\n> synthetic=true：以上 tok/s 只表示形状成本，**不进报告**。"
              "要真实 tok/s 请给 --tokens-npy。")
    print(f"\nrss peak {rss_peak:.1f} GiB   json → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
