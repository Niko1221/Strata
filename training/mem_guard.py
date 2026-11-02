"""显存预算 + OOM fail-fast。

为什么必须有这个文件（2026-10-30 实测，Windows 4090 48 GB / pack 16 GiB /
LoRA r16 6.68 M / T=序列长度）：

| 档位 | 实测峰值 | 结果 |
|---|---|---|
| MeZO ce T=2048 / 8192 / 16384 / 32768 | 26.2 / 27.7 / 29.8 / **35.3** | 正常 |
| MeZO ce T=65536 | 外推 50.8 | 干净 OOM |
| 反传 ce T=2048 / 8192 | 30.1 / **42.0** | 正常 |
| 反传 dpo（2 行）T=8192 | > 48 | `AcceleratorError` 崩在 native 栈 |
| 反传 ce T=16384 | > 48 | **既不报错也不退出**：分配失败后显存超发，GPU 100%、日志零行 |

最后一行是真正的问题：Windows 的显存超发会把「装不下」变成「卡死」，既不给你
异常也不给你速度。所以不能靠 CUDA 报错兜底，必须在**开跑前**按经验公式预算，
超限就 `SystemExit(2)`；跑到一半真 OOM 时也必须立刻终止（CUDA context 已污染，
同进程内不可恢复）。

经验公式只对**这一套形状**（pack 常驻 16 GiB、基座 bf16、专家 nf4）适用：换 pack
档位或换卡要重新标定 `BASE_GIB`。
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys

# 装载后常驻：基座 bf16 ~9.2 GiB + pack 显存 16 GiB（实测 loaded 25.2 GiB）。
# 换 pack 显存档位（STRATA_VRAM_PACK_GB）要跟着改：例如 STRATA_VRAM_PACK_GB=8 →
# BASE_GIB≈17.2（用 STRATA_BASE_GIB 覆盖，不必改代码）。
BASE_GIB = float(os.environ.get("STRATA_BASE_GIB", "25.2"))
# 预算留 5% 余量给 allocator 碎片（torch 的 caching allocator 不做碎片回收）
SAFETY_FRAC = 0.95
# MeZO 单独一条安全线：它全程 no_grad、分配模式每步完全一致（没有反传的临时块），
# 碎片压力比 bwd 小一个量级。2026-11-01 实测 48 GB 卡上 T=49152 峰值 47.5 GiB
# （占物理 98.9%）成功、T=52736 单笔 22.2 GB 分配失败 ⇒ 边界就在这两者之间。
# 用 0.95 会把「实测跑通」的那一档也拒掉，用 0.99 正好卡在实测边界上。
SAFETY_FRAC_MEZO = float(os.environ.get("STRATA_MEM_SAFETY_MEZO", "0.99"))

# MeZO 的实测锚点（48 GB 卡 / pack 16 GiB / 1 行 / ce 目标 / no_grad）：
#   T → 峰值 GiB。全部是 bench 的 ok 行（49152 是「跑通但只剩 0.5 GiB」的边界）。
#
# 为什么不再用 `0.5 + 2e-4·T + c·T²` 拟合：单一 c 撑不住全区间。
#   * c=3.6 在 27648 上证（算 33.4 / 实测 33.0），到 49152 只算 43.6，实测 47.5 —— 低 3.9；
#   * 反过来拿 49152 标定得 c≈5.3，代回 32768 变 37.1，实测 35.3 —— 高 1.8。
# 斜率在 28K~49K 之间自己变陡（+0.674 GiB / 1k token），说明 40K 以上多了一项不是
# 纯 T² 的增长（QSA 的 selected_token_mask 是 (B,S,1,S)，它在长序列上还要与 causal
# mask 各自存活一遍）。既然物理图像没吃透，就按实测点插值 —— 公式可以错，锚点不会。
# 32768 必须是锚点：只有 27648/49152 两端锚定时，中段线性插值算 36.45，与实测差 1.15。
MEZO_ANCHORS = (
    (2048, 26.2), (8192, 27.7), (16384, 29.8), (27648, 33.0), (32768, 35.3),
    (49152, 47.5),
)
# 超出末锚点的外推斜率惩罚：最后一段本来就陡，再乘 1.10 保证「没测过的一律偏保守」。
MEZO_TAIL_PENALTY = 1.10

# 引擎路线（`--backend engine`）完全是另一套显存账，不能套上面任何公式：
# 权重按 IQ3_XXS 留 pack、专家用固定槽位 cache（显存多就多留、少了就少留），
# 没有逐层重建的 bool mask。130816 token 实测峰值 26.9 GiB，其中 int8 KV 1.68 GiB。
#
# **这里是唯一的出处**：budget.py 以前自己抄了一份（WEIGHTS 24 + WORKSPACE 1.2 +
# 13.75 KiB/byte/token），两份数一旦漂移，同一台机器会给出两个「能训多长」。
ENGINE_WEIGHTS_GIB = float(os.environ.get("STRATA_ENGINE_WEIGHTS_GIB", "24.0"))
ENGINE_WORKSPACE_GIB = 1.2       # head/草稿缓冲；128K 实测的余量
ENGINE_BASE_GIB = float(os.environ.get("STRATA_ENGINE_BASE_GIB",
                                       str(ENGINE_WEIGHTS_GIB + ENGINE_WORKSPACE_GIB)))
# 每个 KV 元素每 token 的 KiB：两个标定点（8K fp16 → 0.21 GiB、128K int8 → 1.68 GiB）
# 都落在 13.75 上，且 int8 正好是 fp16 的一半（1 字节 vs 2 字节）⇒ 按元素位宽缩放。
KV_KIB_PER_BYTE_TOKEN = 13.75
KV_BYTES = {"fp16": 2.0, "int8": 1.0, "k8v4": 0.75, "q4_0": 0.5}


def kv_mode_for(seq_len: int) -> str:
    """`--kv auto` 的档位。阈值与 generate.cpp / setup.py 的 KV_AUTO_STEPS 一致：
    ≤32K fp16、≤128K int8、≤256K k8v4、更大 q4_0。三处要一起改。"""
    if seq_len <= 32768:
        return "fp16"
    if seq_len <= 131072:
        return "int8"
    if seq_len <= 262144:
        return "k8v4"
    return "q4_0"
# 无梯度检查点时激活的修正系数。bwd 公式是**开 ckpt** 标定的（见上表）；
# 无 ckpt 时激活远大于拟合值 —— 单点标定：2026-10-31 DPO rows=2 T≤2048
# 无 ckpt 实测峰值 59.8 GiB，同形状拟合值 34.0 GiB ⇒ 激活项差 3.9×，取 4.0
# （保守方向）。换形状 / 换卡后有了新实测点应重新标定。
NO_CKPT_FACTOR = 4.0

# MoE 专家 refetch 的显存（**与 T 无关**，公式里的激活项抓不到它）。
#
# 2026-11-01 实测（48 GB 卡、IQ3_XXS + nf4 pack、frozen-prefix tail=71 ⇒ 激活仅
# ~0.3 GiB）：步内峰值 41.3~51.4 GiB，而「BASE + 激活」只算出 26.7 GiB。差距全部
# 来自专家取用：
#   * 反传期 `loader_qwen4` 按**层**整块申请（512 专家 × 2.49 MiB ≈ 1.27 GiB/层），
#     层用完才释放 —— 快照里 1799 块 × 2.49 MiB = 4.4 GiB 同时存活，分配流量峰值
#     11.1 GiB（loader_qwen4.py:606 _stream_groups，28 个事件）；
#   * 另有 84×30 MiB / 2×1212 MiB / 37×50 MiB 等专家与暂存块。
# 取实测最坏增量 ≈ 26 GiB 作为默认值（env 可覆盖，换 pack 档位/换卡要重标定）。
# 只在**走 refetch 的路径**（无 ckpt 的 GGUF 反传）传这个值；开 ckpt 时权重直接进
# 图，是另一套账（92 GiB OOM 那种）。
MOE_STAGING_GIB = float(os.environ.get("STRATA_MOE_STAGING_GIB", "26.0"))


def _mezo_anchor_gib(seq_len: int) -> float:
    """按实测锚点分段插值（含 BASE）。区间内线性；区间外按最近一段斜率外推。

    末锚点之外乘 `MEZO_TAIL_PENALTY`：没测过的点必须偏保守，否则「必爆判成刚好过」
    正是 2026-11-01 那次 52736 OOM 的成因（公式 45.6 / 实际取不到连续块）。
    """
    pts = MEZO_ANCHORS
    if seq_len <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return max(BASE_GIB, y0 + (seq_len - x0) * (y1 - y0) / (x1 - x0))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if seq_len <= x1:
            return y0 + (seq_len - x0) * (y1 - y0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    k = (y1 - y0) / (x1 - x0) * MEZO_TAIL_PENALTY
    return y1 + (seq_len - x1) * k


def estimate_engine_gib(seq_len: int, kv: str = "auto") -> float:
    """引擎路线的峰值（GiB）。与 HF 路线没有可比性，别混着报。

    构成只有两块：常驻（基座 + expert cache 的固定槽位，与 T 无关）+ KV（∝T）。
    实测 130816 token / int8 KV 1.68 GiB ⇒ 常驻 25.2。**这不等于整卡占用**：引擎的
    `--expert-cache auto` 会把剩余显存吃满当专家缓存（bench 里整卡 48 GiB 用满、内部
    自报 26.9），所以这里的数字是「跑得起来的最小值」，不是「会占多少」。
    """
    if kv == "auto":
        kv = kv_mode_for(seq_len)
    return ENGINE_BASE_GIB + seq_len * KV_BYTES.get(kv, 1.0) * KV_KIB_PER_BYTE_TOKEN * 1024 / 2 ** 30


# -------------------------------------------------------------------------------------------------- GPU 独占检查（pipe 路线）

# 桌面/系统自身的显存占用容忍（实测本机浏览器+输入法 ~1 GiB）。低于它说明卡上住着
# 别的大户 —— 在引擎世界这只可能是另一个引擎实例。
EXCLUSIVE_ALLOWANCE_MIB = 2048.0

_UNSET = object()   # 区分「调用方没注入 free」与「注入了 None（= 查询失败）」


def gpu_free_mib() -> float | None:
    """当前空闲显存（MiB，GPU 0）。nvidia-smi 不在或失败返回 None（调用方要如实打印跳过）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def require_exclusive_gpu(need_gib: float, *, log=print, total_gib: float | None = None,
                          free_mib: float | None = _UNSET) -> None:
    """pipe 模式拉引擎前的独占检查（`--engine-api` 会话不需要：服务端自己互斥让位）。

    为什么必须有：serve 的排队与并行槽位（--batch N）防的是**单引擎进程内**的请求
    打爆 —— 预算始终是一个引擎的预算。pipe 直接拉第二个引擎进程不经过任何排队，
    两个进程各自把 pack 映射进主机内存、各自填满显存缓存：实测 committed 89.7 GiB
    > 物理 48 GiB，第二个引擎 stall 自杀（2026-11-02，写 strata-stall-*.dmp）。
    判据不是 need_gib（`--expert-cache auto` 会把空闲显存吃满，需求侧永远"够"），
    而是**卡基本是空的**：free < total − ALLOWANCE 即视为有其他引擎驻留。
    STRATA_ALLOW_SHARED_GPU=1 显式豁免（自己清楚后果时用，豁免也会打印）。
    """
    total = _total_gib() if total_gib is None else total_gib
    free = gpu_free_mib() if free_mib is _UNSET else free_mib
    if free is None:
        log("[mem] nvidia-smi 不可用，跳过 GPU 独占检查（无法确认卡上是否已有其他引擎实例）")
        return
    if os.environ.get("STRATA_ALLOW_SHARED_GPU") == "1":
        log(f"[mem] STRATA_ALLOW_SHARED_GPU=1：跳过独占检查（空闲 {free/1024:.1f} GiB / "
            f"需求下限 {need_gib:.1f} GiB）")
        return
    free_gib = free / 1024
    if free_gib < need_gib:
        sys.stderr.write(
            "\n" + "=" * 78 + "\n"
            f"显存不足以拉起引擎：空闲 {free_gib:.1f} GiB < 需求下限 {need_gib:.1f} GiB"
            f"（物理 {total:.1f} GiB）。\n"
            f"  卡上已有 ~{total - free_gib:.1f} GiB 驻留（另一个引擎实例 / 推理服务）。\n"
            f"  建议：① 走 API 会话让服务端互斥让位（--engine-api http://...）；\n"
            f"        ② 或先停掉占卡的进程再跑 pipe 模式。\n"
            + "=" * 78 + "\n")
        raise SystemExit(2)
    if free_gib < total - EXCLUSIVE_ALLOWANCE_MIB / 1024:
        sys.stderr.write(
            "\n" + "=" * 78 + "\n"
            f"GPU 不是空的：空闲 {free_gib:.1f} GiB / 物理 {total:.1f} GiB"
            f"（容忍桌面占用 {EXCLUSIVE_ALLOWANCE_MIB/1024:.0f} GiB）。\n"
            f"  pipe 模式会拉起**第二个引擎进程**，与已有驻留各自填满显存与主机内存 ——\n"
            f"  实测这种双驻留会把系统提交内存打爆、引擎 stall 自杀（不报错、不退出、\n"
            f"  只留下 strata-stall-*.dmp）。serve 的排队/并行槽位管不到跨进程。\n"
            f"  建议：① 走 API 会话（--engine-api http://...：open 时推理引擎让位，\n"
            f"           close 后自动恢复）；\n"
            f"        ② 先停掉占卡的引擎实例；\n"
            f"        ③ 确认要共卡：STRATA_ALLOW_SHARED_GPU=1 重跑。\n"
            + "=" * 78 + "\n")
        raise SystemExit(2)
    log(f"[mem] GPU 独占检查通过：空闲 {free_gib:.1f} GiB / 物理 {total:.1f} GiB"
        f"（引擎需求下限 {need_gib:.1f}）")


def estimate_peak_gib(seq_len: int, rows: int = 1, mode: str = "mezo",
                      ckpt: bool = True, moe_staging: float = 0.0,
                      route: str = "hf", kv: str = "auto") -> float:
    """峰值估算（GiB）。`route` 决定用哪套账：hf = torch 前向/反传，engine = 引擎。

    * MeZO（2 次前向、全程 no_grad）：**按 `MEZO_ANCHORS` 分段插值**。
      原来写成解析式 `BASE + 0.5 + 2.0e-4·T + c·T²/2^30`，在 27648 上准、在 49152 上
      低 3.9 GiB（43.6 vs 实测 47.5），而拿 49152 反标定 c 又会让 32K 档高估 1.8 GiB
      —— 斜率在 28K~49K 之间自己变陡（+0.674 GiB / 1k token），说明 40K 以上混进了
      一项不是纯 T² 的增长。物理图像没吃透就别硬拟合，锚点不会错。
    * 反传（1f+1b，**开 grad ckpt**）：`BASE + 4.9·(T/2048)^0.89·rows^0.85`
      —— 拟合 2048 → 30.1、8192 → 42.0（实测同值）。指数 0.89 而不是 1，说明
      里面混了 attention 的 S² 项；rows^0.85 是「批内多行共享固定成本」。
      无 ckpt 时激活项 × NO_CKPT_FACTOR（单点标定，见常量注释）。

    ckpt 语义注意：DPO 的 margin loss 需要 chosen/rejected 两条序列的前向图
    **同时存活**（差值算完才能一次反传），所以调用方必须传 rows=2 —— 这不是
    「批内两行排队」，而是两份激活共存；SFT 逐行反传释放后才送下一行，语义不同。
    """
    if route == "engine":
        return estimate_engine_gib(seq_len, kv=kv)
    if mode == "mezo":
        # 锚点是 rows=1 标的；多行只多一份「批内共享固定成本」的 sqrt 项
        return _mezo_anchor_gib(seq_len) + 0.8 * max(0, rows - 1) * (seq_len / 8192) ** 0.5
    term = 4.9 * (seq_len / 2048) ** 0.89 * rows ** 0.85
    if not ckpt:
        term *= NO_CKPT_FACTOR
    return BASE_GIB + term + moe_staging


def _total_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    except Exception:
        pass
    return float(os.environ.get("STRATA_GPU_GIB", "48"))


def mezo_anchor_hit(seq_len: int) -> bool:
    """该 T 是否正好落在实测锚点上。不是锚点的区间是**插值**，不要当成实测报。"""
    return any(seq_len == x for x, _ in MEZO_ANCHORS)


def _limit_gib(total: float, mode: str) -> float:
    """该模式的可判定上限。MeZO 全程 no_grad、每步分配模式一致，碎片压力小，
    用 0.99；bwd 有反传临时块与 refetch 抖动，留 5%。"""
    frac = SAFETY_FRAC_MEZO if mode == "mezo" else SAFETY_FRAC
    return total * frac


def require_budget(seq_len: int, rows: int = 1, mode: str = "mezo", *,
                   ckpt: bool = True, moe_staging: float = 0.0, log=print,
                   route: str = "hf", kv: str = "auto") -> float:
    """开跑前检查。超限直接退出（退出码 2），并把「该改什么」打出来。

    ckpt 必须如实传：bwd 公式按开 ckpt 标定，无 ckpt 的运行用它会系统性
    低估激活（NO_CKPT_FACTOR 修正），低报比拦错更危险 —— 拦错只损失一次
    启动，低报损失的是整次训练。
    moe_staging：走专家 refetch 的反传路径要传 MOE_STAGING_GIB（实测 ~26 GiB，
    与 T 无关，见常量注释）；开 ckpt（权重进图）不传。
    route="engine" 时走引擎那套账（常驻 + KV），与 HF 的 mask 项无关。
    """
    est = estimate_peak_gib(seq_len, rows, mode, ckpt=ckpt, moe_staging=moe_staging,
                            route=route, kv=kv)
    total = _total_gib()
    limit = _limit_gib(total, mode)
    extra = f"（含 MoE 专家暂存 {moe_staging:.1f}）" if moe_staging else ""
    if route == "hf" and mode == "mezo" and not mezo_anchor_hit(seq_len):
        extra += "（插值区：该 T 无实测锚点）"
    log(f"[mem] {mode} T={seq_len} rows={rows} ckpt={'on' if ckpt else 'off'} route={route}  "
        f"估算峰值 {est:.1f} GiB{extra} / 物理 {total:.1f} GiB（预算上限 {limit:.1f}）"
        f"  本机动态上限 T≈{max_seq_len(mode, rows, total, ckpt=ckpt, moe_staging=moe_staging, route=route, kv=kv)}")
    if est <= limit:
        return est

    alt = ("改为 MeZO（`--objective ce` 走零阶，no_grad 不存激活）"
           if mode != "mezo" else "缩短 T")
    sys.stderr.write(
        "\n" + "=" * 78 + "\n"
        f"显存预算不足，直接终止（不等 CUDA 报错）：{mode} T={seq_len} rows={rows}"
        f" ckpt={'on' if ckpt else 'off'} route={route}\n"
        f"  估算峰值 {est:.1f} GiB > 预算 {limit:.1f} GiB（物理 {total:.1f} GiB × "
        f"{SAFETY_FRAC_MEZO if mode == 'mezo' else SAFETY_FRAC}）\n"
        f"  本机当前形状的动态上限：T≈{max_seq_len(mode, rows, total, ckpt=ckpt, moe_staging=moe_staging, route=route, kv=kv)}\n"
        f"为什么不在跑到一半时再报错：Windows 的显存超发会把 OOM 变成「GPU 满载但\n"
        f"日志零行」的卡死（实测反传 T=16384），既不给你异常也不给你速度。\n"
        f"  建议：① {'开 --ckpt（梯度检查点，激活大幅缩水换 ~30% 速度）' if not ckpt else alt}；\n"
        f"        ② {alt if not ckpt else '减少同时进前向的行数（DPO 一次只送 1 对）'}；\n"
        f"        ③ 换更小的 pack 档位（BASE_GIB 会跟着降）；\n"
        f"        ④ 长上下文改走引擎（`route=engine`：130816 token 只要 26.9 GiB，\n"
        f"           因为权重留在 pack 里按槽位换入，不物化 (S,S) 的 bool mask）。\n"
        + "=" * 78 + "\n")
    raise SystemExit(2)


def max_seq_len(mode: str = "mezo", rows: int = 1, total_gib: float | None = None,
                ckpt: bool = True, moe_staging: float = 0.0,
                route: str = "hf", kv: str = "auto") -> int:
    """当前显存下该形状能训的最大 T（把 estimate_peak_gib 反解，不写死档位）。

    上限跟着五件事动态变：物理显存、BASE_GIB（pack 档位）、rows、ckpt、route。
    换任何一件，这里给的就是新值 —— 「反传 8K / MeZO 49K」只是 48 GB + 16 GiB
    pack + 开 ckpt 时的实测点，不是常数。返回值向下取到 256 的倍数
    （与分块/对齐习惯一致）。
    """
    total = total_gib if total_gib is not None else _total_gib()
    limit = _limit_gib(total, mode) - moe_staging   # 专家暂存先扣掉，剩下的才是激活预算
    scale = 1.0 if ckpt else NO_CKPT_FACTOR
    if mode == "mezo" or route == "engine":
        # 公式里除了解析可解的一次+二次项，还有 0.8·(rows-1)·(T/8192)^0.5 的 sqrt 项（批内
        # 多行共享固定成本）和分段锚点，都没有闭式解 —— 以前直接用解析式会让 rows>1 的
        # 上限**高估**（实测 K=4 时算出 55808，代回估计值 51.8 GiB 已超预算）。改成二分，
        # 口径与 estimate_peak_gib 完全一致。
        lo, hi = 0, 1 << 20
        while lo < hi:
            mid = (lo + hi + 1) // 2
            # 传 ckpt=True 是刻意的：MeZO 全程 no_grad 没有检查点语义，estimate_peak_gib 的
            # NO_CKPT_FACTOR 只对 bwd 有意义，这里必须让它不生效（否则上限会被腰斩 4 倍）。
            if estimate_peak_gib(mid, rows, "mezo" if mode == "mezo" else mode,
                                 ckpt=True, moe_staging=moe_staging,
                                 route=route, kv=kv) <= _limit_gib(total, mode):
                lo = mid
            else:
                hi = mid - 1
        t = float(lo)
    else:
        # 4.9·(T/2048)^0.89·rows^0.85·scale = limit - BASE
        term = (limit - BASE_GIB) / (4.9 * max(rows, 1e-9) ** 0.85 * scale)
        if term <= 0:
            return 0
        t = 2048.0 * term ** (1.0 / 0.89)
    return max(int(t) // 256 * 256, 0)


def estimate_peak_gib_frozen(prefix_len: int, tail_len: int, rows: int = 1,
                             moe_staging: float = 0.0) -> float:
    """frozen-prefix（前缀缓存 + 尾段反传）的峰值估算。

    峰值时同时存活的是「前缀缓存 + 尾部反传图」（前缀 no_grad 前向的瞬时工作区
    在拿到缓存后已释放）：

    * 前缀缓存项：借用 mezo 公式做**上界** —— mezo 的二次项是全序列 3 份 bool
      mask，而缓存只存全注意力 K/V + 固定大小的 GDN state，必然更小。
      偏高方向 = 安全，不会放走真实会爆的形状。
    * 尾部反传项：与 bwd 公式同形（T 换成 tail_len），但**恒按无 ckpt 修正**
      （× NO_CKPT_FACTOR）—— frozen-prefix 与 --ckpt 互斥（HF 的
      GradientCheckpointingLayer 在 ckpt 下会把 past_key_values 置 None，
      modeling_layers.py "Caching is incompatible with gradient checkpointing"），
      尾部短图本来也不需要 ckpt。

    rows 语义与 bwd 相同：DPO 的 margin 需要 chosen/rejected 两份尾部图共存
    ⇒ rows=2；SFT 逐行反传 ⇒ rows=1。

    moe_staging：专家 refetch 的暂存（**与 T 无关**，见 MOE_STAGING_GIB 注释）。
    实测 tail=71 时估算 26.7 GiB 而真实峰值 51.4 GiB，差额几乎全是它。
    """
    prefix_term = 0.5 + 2.0e-4 * prefix_len + 3.0 * (prefix_len ** 2) / 2 ** 30
    tail_term = 4.9 * (tail_len / 2048) ** 0.89 * rows ** 0.85 * NO_CKPT_FACTOR
    return BASE_GIB + prefix_term + tail_term + moe_staging


def max_frozen_tail_len(prefix_len: int, rows: int = 1,
                        total_gib: float | None = None,
                        moe_staging: float = 0.0) -> int:
    """frozen-prefix 模式下该前缀长度能训的最大尾段（estimate_peak_gib_frozen 反解）。"""
    total = total_gib if total_gib is not None else _total_gib()
    prefix_term = 0.5 + 2.0e-4 * prefix_len + 3.0 * (prefix_len ** 2) / 2 ** 30
    term = (total * SAFETY_FRAC - moe_staging - BASE_GIB - prefix_term) \
        / (4.9 * max(rows, 1e-9) ** 0.85 * NO_CKPT_FACTOR)
    if term <= 0:
        return 0
    return max(int(2048.0 * term ** (1.0 / 0.89)) // 256 * 256, 0)


def require_budget_frozen(prefix_len: int, tail_len: int, rows: int = 1,
                          moe_staging: float = 0.0, log=print) -> float:
    """frozen-prefix 模式的开跑前检查（超限 SystemExit(2)，与 require_budget 同语义）。"""
    est = estimate_peak_gib_frozen(prefix_len, tail_len, rows, moe_staging=moe_staging)
    total = _total_gib()
    limit = total * SAFETY_FRAC
    extra = f"（含 MoE 专家暂存 {moe_staging:.1f}）" if moe_staging else ""
    log(f"[mem] frozen-prefix prefix={prefix_len} tail={tail_len} rows={rows}（ckpt 强制关）  "
        f"估算峰值 {est:.1f} GiB{extra} / 物理 {total:.1f} GiB（预算上限 {limit:.1f}）"
        f"  本机该前缀下尾段上限 ≈{max_frozen_tail_len(prefix_len, rows, total, moe_staging)} tok")
    if est <= limit:
        return est
    sys.stderr.write(
        "\n" + "=" * 78 + "\n"
        f"显存预算不足（frozen-prefix）：prefix={prefix_len} tail={tail_len} rows={rows}\n"
        f"  估算峰值 {est:.1f} GiB > 预算 {limit:.1f} GiB（物理 {total:.1f} GiB × {SAFETY_FRAC}）\n"
        f"  本机该前缀下尾段动态上限 ≈{max_frozen_tail_len(prefix_len, rows, total, moe_staging)} tok\n"
        f"  尾段是带梯度的推理链+答案：太长时回到全序列 --ckpt 反传（慢但能装下），\n"
        f"  或截短推理链数据 —— 不要试图对 frozen-prefix 再开 ckpt（HF 会静默丢缓存）。\n"
        + "=" * 78 + "\n")
    raise SystemExit(2)


def is_oom(exc: BaseException) -> bool:
    """本机 OOM 抛的是 AcceleratorError，不是 torch.cuda.OutOfMemoryError。"""
    if isinstance(exc, torch_oom_types()):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def torch_oom_types() -> tuple:
    types = [MemoryError]
    try:
        import torch

        for name in ("OutOfMemoryError", "AcceleratorError"):
            t = getattr(torch, name, None)
            if isinstance(t, type):
                types.append(t)
    except Exception:
        pass
    return tuple(types)


def die_on_oom(exc: BaseException, *, where: str, log=print) -> "SystemExit":
    """把 OOM 变成明确的终止：打印现场 + 退出码 2，绝不继续跑。"""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 2 ** 30
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
    except Exception:
        alloc = peak = float("nan")
    sys.stderr.write(
        "\n" + "=" * 78 + "\n"
        f"显存不足，终止（{where}）：{type(exc).__name__}: {str(exc)[:200]}\n"
        f"  现场：allocated {alloc:.1f} GiB / peak {peak:.1f} GiB\n"
        f"  同进程内不可恢复（CUDA context 已污染），不要重试同一个形状。\n"
        f"  下一步：缩短 T、减少批内行数，或改用 MeZO（no_grad 不存激活）。\n"
        + "=" * 78 + "\n")
    return SystemExit(2)
