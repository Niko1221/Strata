"""平台守卫：训练这条路径只支持 Linux；Windows 明确拒绝，并说清到底卡在哪。

为什么单独一个文件：以前在 Windows 上跑 `train.py` 的失败方式是在**读完 76 GB pack
之前**什么都没发生，然后在 `loader_qwen4.py` 深处抛
`AttributeError: module 'os' has no attribute 'preadv'`—— 一个跟真正原因（平台不支持）
毫无关系的报错。所以把「能不能跑」提到入口，用能核对的话说清楚。

**本模块不 import torch**，这样上游 CI 用 `requirements.txt` 那个只有标准库的环境
也能直接跑它，作为合并门的第一道（见 `.github/workflows/training-test.yml`）。

探测结果分两类，别混：

  阻断项（blockers）—— 真的跑不了
  降级项（degrades） —— 能跑，但少了一个优化，别有人以为是 bug 去"修"

用法：
  python3 platform_check.py            # 打印支持矩阵 + 本机探测，阻断时退出码 2
  python3 platform_check.py --json     # 同上，输出机器可读

在 `train.py` 里是入口处一行 `platform_check.require()`。
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys

# 支持的平台。Linux 走原生 os.preadv；Windows 走 preadv_shim 的 readinto 垫层
# （同一个调用点、同一份语义，见 preadv_shim.py）。Cygwin/MSYS 仍不在列表里。
SUPPORTED = ("linux", "win32")

# preadv 的使用者名单。垫层化之后它们不再是"调用即 AttributeError"的风险点，
# 但谁在用这个 IO 形态仍然值得单一出处地记录（改 IO 策略时照单全改）。
PREADV_USERS = (
    "loader_qwen4.py  （_TieredPack._ensure：pack → 显存 / 锁页内存，两处）",
    "bench_pack_io.py  （pack 随机读天花板基准）",
)


def _preadv_kind() -> str:
    """native / readinto-shim / none。只依赖标准库（preadv_shim 不 import torch）。"""
    try:
        import preadv_shim
    except Exception:
        return "none"
    return getattr(preadv_shim, "preadv_kind", "none")


def _torch_facts() -> dict:
    """torch / CUDA 的事实。torch 在这里才 import，所以本文件不依赖它。"""
    try:
        import torch
    except Exception as exc:                       # 没装 torch 也算一种事实，不是异常
        return {"torch": None, "torch_error": type(exc).__name__, "cuda": False}
    return {"torch": torch.__version__, "torch_error": None,
            "cuda": bool(torch.cuda.is_available())}


def probe(with_torch: bool = True) -> dict:
    """把「这台机器能不能跑」拆成几条能单独核对的事实。

    with_torch=False 跳过 torch 探测（模块级守卫和 CI 快门用：它们不该为查平台
    付出 import torch 的几秒，也没有 torch 可 import）。

    返回值是纯数据，`blockers()` 只吃这个 dict —— 于是测试可以直接造一个假的
    capabilities，不用去 monkeypatch 全局环境。
    """
    caps = {
        "platform": sys.platform,
        "has_preadv": _preadv_kind() != "none",
        "preadv_kind": _preadv_kind(),
        "has_madvise": hasattr(mmap.mmap, "madvise"),
    }
    if with_torch:
        caps.update(_torch_facts())
    return caps


def platform_blockers(caps: dict) -> list[str]:
    """平台本身的阻断。空列表 = 这台机器的平台能跑这份代码。

    与 torch 无关：模块级守卫（sft.py / sft_mezo.py / ab_identity.py 顶部）只用
    这一层，因为那几处必须在 import torch 之前就拦住 Windows，而且上游 CI 的
    no-torch 档没有 torch 也必须能过这道门。
    """
    out: list[str] = []

    plat = str(caps.get("platform") or "")
    if not plat.startswith(SUPPORTED):
        out.append(
            f"平台 {plat!r} 不在支持列表 {SUPPORTED} 里。"
            f"pack 装载的读入走 preadv 语义（Linux 原生 / Windows 走 preadv_shim 的 "
            f"readinto 垫层）；其它平台既没有原生 preadv，垫层也没验证过。")

    # 这一条单独查：万一哪天垫层文件丢了或改名了，这里先亮，比"某个深层函数
    # 抛 ImportError/AttributeError"好定位。
    if not caps.get("has_preadv"):
        out.append(
            "preadv 垫层不可用（preadv_shim 导入失败或缺少 preadv_kind）。"
            "原生 os.preadv 只在 Linux/BSD 有；Windows 依赖 preadv_shim.py 的 "
            "readinto 垫层，用它的是：" + "；".join(PREADV_USERS) + "。")

    return out


def runtime_blockers(caps: dict) -> list[str]:
    """平台之外、这次运行本身的缺口：torch 没装、装了但没有 CUDA。

    单独拆出来：CI 的 CPU torch 档要 import 训练模块跑测试，此时报
    "没有 CUDA" 是对的（那档本来就只跑 CPU 测试），但不该连 import 都拦死。
    """
    out: list[str] = []
    if caps.get("torch") is None:
        err = caps.get("torch_error") or ""
        out.append(f"torch 没装（import 抛 {err or 'Exception'}）。"
                   f"训练路径必需；只有 collect_rollouts.py 不需要。")
    elif not caps.get("cuda"):
        out.append(f"torch {caps['torch']} 装了但 torch.cuda.is_available() 为 False。"
                   f"训练要走 CUDA —— CPU 上每步是分钟级，不是慢一点。")
    return out


def blockers(caps: dict) -> list[str]:
    """全部阻断原因（平台 + 运行时）。空列表 = 可以真的开训。"""
    return platform_blockers(caps) + runtime_blockers(caps)


def degrades(caps: dict) -> list[str]:
    """能跑，但少了个优化。列出来是为了防止有人当 bug 去修。"""
    out: list[str] = []
    if not caps.get("has_madvise"):
        out.append(
            "mmap.mmap.madvise 不存在 ⇒ GGUF 的映射页还不掉（loader_qwen4.py:126、889 有 "
            "hasattr 守卫，Windows 上是无操作）。代价是多占约 15 GiB 常驻，不是正确性"
            "问题；小显存档可能在基座加载阶段就先爆，见 BENCH-SFT-VS-MEZO.md §12.2。")
    return out


def ensure_utf8() -> None:
    """Windows + 非 UTF-8 模式时给出明确告警（不自动重启：`os.execve` 在 Windows
    上重定向 stdout 时会段错误，实测 exit 139 —— 不能拿它做入口级操作）。

    为什么值得管：`PYTHONUTF8=0` 时 torch inductor 用 locale 编码（中文 Windows 是
    GBK）去读 cutedsl 的 kernel 模板（内容是 UTF-8）⇒ UnicodeDecodeError ⇒ fla 的
    导入链断掉 ⇒ transformers 静默回落 GDN 的 torch 参考实现。

    回落**不会报错**，只打一行 "falling back to its reference PyTorch implementation"。
    实测代价：单步只慢 4%，但显存随步数累积（连跑几次 T=2048 后 allocated 冲到
    60 GiB，超过 48 GiB 物理上限）⇒ 20 步内 OOM。所以它必须被说出来。

    真正的判据在 `gdn_kernel_is_fast()`：直接看 GDN 落在哪个实现上，而不是猜编码。
    """
    if sys.platform != "win32" or sys.flags.utf8_mode:
        return
    sys.stderr.write(
        "=" * 78 + "\n"
        "警告：本进程不是 UTF-8 模式（sys.flags.utf8_mode == 0）。\n"
        "如果下面 GDN 回落到 torch 参考实现，显存会随步数累积，20 步内 OOM。\n"
        "建议：set PYTHONUTF8=1（cmd）/ $env:PYTHONUTF8=1（PowerShell）后重跑。\n"
        + "=" * 78 + "\n")


def gdn_kernel_is_fast() -> bool | None:
    """GDN 是否落在 hub/fla 的加速实现上。None = 还没 import transformers（无法判断）。

    比查编码可靠：`PYTHONUTF8=0` 只是**已知的一种**触发原因，直接看结果。
    """
    mod = sys.modules.get("transformers.models.qwen4_exp.modeling_qwen4_exp")
    if mod is None:
        return None
    fn = getattr(mod, "torch_chunk_gated_delta_rule", None)
    if fn is None:
        return None
    return type(fn).__name__ in ("Func", "Kernel", "function")


def warn_if_gdn_fell_back(log=print) -> None:
    """加载模型后调用：回落时把后果说清楚（不是"性能提示"，是"跑不完"）。"""
    fast = gdn_kernel_is_fast()
    if fast is None or fast:
        return
    log("!" * 78)
    log("GDN 落在 torch 参考实现上（没走 fla/hub kernel）。")
    log("  单步只慢 4%，但显存随步数累积：实测连跑几次 T=2048 后 allocated 冲到 60 GiB")
    log("  （超过 48 GiB 物理上限，Windows 显存超发），20 步内 OOM。")
    log("  修法：PYTHONUTF8=1 后重跑（中文 Windows 默认可能是 0）。")
    log("!" * 78)


def require(caps: dict | None = None, platform_only: bool = False) -> dict:
    """有阻断就打印原因并以退出码 2 结束；否则返回 capabilities。

    platform_only=True 只查平台层、且不探测 torch —— 给两处用：
      * sft.py / sft_mezo.py / ab_identity.py 的模块级守卫（要在 import torch
        之前拦 Windows，且不能因为 CI 没 GPU 就拦住 import）；
      * CI 的 no-torch 快门（那档本来就故意没有 torch）。
    真正开训前的完整检查（含 torch/CUDA）走默认 platform_only=False。

    退出码用 2 而不是 1：1 在训练脚本里被当作"训练本身失败"，守卫被拒应该是
    "根本没跑"，两者在 CI 里要能分开看。
    """
    if caps is None:
        ensure_utf8()          # 必须在 probe/import torch 之前：GDN 的落点由它决定
        caps = probe(with_torch=not platform_only)
    bad = platform_blockers(caps) if platform_only else blockers(caps)
    if not bad:
        return caps
    w = sys.stderr.write
    w("=" * 78 + "\n")
    w("训练路径在本机跑不了 —— 平台不支持，不是配置问题。\n")
    w("=" * 78 + "\n")
    for i, b in enumerate(bad, 1):
        w(f"\n{i}. {b}\n")
    soft = degrades(caps)
    if soft:
        w("\n（以下是能跑但会降级的项，供参考）\n")
        for s in soft:
            w(f"  - {s}\n")
    w("\n支持矩阵：Linux ✅（原生 preadv） / Windows ✅（preadv_shim 垫层，锁页上限\n")
    w("与 madvise 缺失的代价见降级项） / macOS ❌（未测）\n")
    w("=" * 78 + "\n")
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="打印训练路径的平台支持矩阵与本机探测结果。")
    ap.add_argument("--json", action="store_true", help="输出 JSON（供脚本判断能不能跑）")
    ap.add_argument("--platform-only", action="store_true",
                    help="只查平台层（win32 / preadv），不探测 torch。"
                         "CI 的 no-torch 快门和模块级守卫用这一档。")
    a = ap.parse_args()

    caps = probe(with_torch=not a.platform_only)
    bad = platform_blockers(caps) if a.platform_only else blockers(caps)
    soft = degrades(caps)
    if a.json:
        print(json.dumps({"caps": caps, "blockers": bad, "degrades": soft},
                         indent=2, ensure_ascii=False))
        return 2 if bad else 0

    print("=" * 78)
    print("训练路径的平台支持矩阵（跟本机无关，是这份代码的声明）")
    print("=" * 78)
    print("  Linux    ✅ 支持 —— 原生 os.preadv，锁页上限本机实测 32 GiB")
    print("  Windows  ✅ 支持 —— preadv_shim 的 readinto 垫层（同调用点、同语义）；")
    print("           madvise 缺失是降级项（映射页不还，多占 ~15 GiB，不是阻断）")
    print("  macOS    ❌ 未测 —— 没有机器；有 preadv，但没验证过")
    print("-" * 78)
    print(f"本机：{caps.get('platform')}  →  {'❌ 被阻断' if bad else '✅ 可以跑'}")
    print("-" * 78)
    print("探测明细：")
    for k in ("preadv_kind", "has_madvise", "torch", "cuda"):
        print(f"  {k:<11s} {caps.get(k)}")
    print("-" * 78)
    if bad:
        print("阻断项：")
        for i, b in enumerate(bad, 1):
            print(f"  {i}. {b}")
    else:
        print("阻断项：无 ✅")
    if soft:
        print("降级项：")
        for s in soft:
            print(f"  - {s}")
    print("=" * 78)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
