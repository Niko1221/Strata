"""注册表：算法（sft/dpo/grpo）× 方法（mezo/backward），两轴正交。

用户口径（2026-11-02 定稿）：

    算法 = 目标函数在量什么：sft(ce) / dpo / grpo，三个并列。
    方法 = 怎么动参数：mezo（零阶，默认）/ backward（autograd 反传）。
    损失函数只有一份（objectives.py），两个方法天然共用 —— mezo 拿标量损失做
    有限差分，backward 拿同一份损失建图反传，数学上必须是同一个数
    （test_objectives.py 的 numpy↔torch 对拍守着这件事）。

于是命令面是：

    python3 train.py --algo sft                 # mezo × ce   （默认方法）
    python3 train.py --algo sft --method backward
    python3 train.py --algo dpo                 # mezo × dpo
    python3 train.py --algo grpo --method backward

mezo 是默认方法的原因（可用性标准）：满上下文训练只比推理多一点点显存才算能用；
mezo 全程 no_grad、前向就是引擎的推理（LORA 换表 + SCORE 打分），显存天然贴住
推理基线。backward 留给梯度精度重要的场景，允许多占显存但不做默认。

方法落地在哪个脚本：
    mezo     → mezo.py      （引擎前向，numpy；ce/dpo/grpo 都走共享 objectives）
    backward → sft.py       （sft 与 dpo 共用护栏；ref logprob 起点预计算一次）
               grpo.py      （两段式：引擎采轨迹 → 学生反传，48GB 放不下两者）

分支与方法的对应（2026-11-03 定：方法可用性由**文件在不在**决定，不由代码开关决定）：
    feat/train-mezo         零阶完整训练：mezo/ + 引擎 + 数学造数据/评测，**不 import torch**
    feat/train-backforward  从零阶线长出来，再叠 backward/ 与 ce/lora/update_rules
    feat/train_support      两条线合并 + 审核文档
所以 `--list` 在零阶线上会写「backward 不在这条分支上（缺 backward/sft.py）」，
`--method backward` 直接退出并指路，不会先跑起来再 ImportError。`available()` 读磁盘
而不是读某个 `IS_BACKWARD_BRANCH` 常量：常量迟早和文件对不上。

为什么 mezo 不再有两个后端（上一版的 BACKENDS/engine|hf 已删）：MeZO 的路线本来
就是拿引擎推理当前向，`sft_mezo.py`（PyTorch 前向）只是引擎还没有 LORA/SCORE 命令
时期的临时实现 + score parity 对拍工具（HF CE −56.6433 vs 引擎 −56.6508），不是
一条并行路线。它已从注册表摘除；对拍需要时直接跑它，不占 `--method` 的名字。
（它在反传那条分支上 —— 对拍要同时有 HF 侧，零阶线不带 torch。）

lr 不能三处共用一个默认值：量级由（算法的目标尺度 × 方法的差分/反传）一起决定
（identity 2-token 答案 mezo 稳定点 3e-5；一屏长答案 mean 口径 mezo 1e-3；
backward sft 1e-4）。表放在 METHODS 的 lr 里，`train.py` 取不到才用算法级缺省。
判断步长看 `rel = ‖lr·v‖/‖θ‖`（mezo.py 每步打印），>0.05/步 会漂，别看 lr 绝对值。

加一个新算法（比如 PPO）要动的地方：
  1. objectives.py 加目标函数数学（一份，两个方法自动可用）；
  2. mezo.py 的 --objective choices 接上（引擎前向给逐 token logp，标量目标就能接）；
  3. backward 落点脚本写好，在 METHODS["backward"] 登记脚本映射；
  4. 下面 ALGOS 加一行。
train.py 不需要改。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

# 本文件所在目录（= training/）。方法能不能用，只看这里的文件在不在 —— 见 available()。
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Algo:
    name: str
    objective: str              # 对应 objectives.py 的目标函数名
    blurb: str
    needs: str = ""             # 数据依赖（提示用）


@dataclass(frozen=True)
class Method:
    """怎么动参数。`scripts` 把算法名映射到实现脚本；没列的算法就是这个方法做不了。"""
    name: str
    scripts: dict[str, str]     # algo -> training/ 下（mezo/、backward/）的脚本（必须暴露 main()）
    lr: dict[str, float]        # algo -> 默认学习率（量级随算法×方法变，不能共用）
    blurb: str


ALGOS: dict[str, Algo] = {
    "sft": Algo(
        "sft", "ce",
        "监督微调：answer 段逐 token 交叉熵（mean 口径，pj 不随序列长度涨）",
    ),
    "dpo": Algo(
        "dpo", "dpo",
        "偏好对：chosen/rejected 各一次前向，序列级 margin 的 −logσ(β·margin)；"
        "ref logprob 在训练起点预计算一次（离线数据，无采样）",
        needs="--dpo-data（偏好对 jsonl）",
    ),
    "grpo": Algo(
        "grpo", "grpo",
        "组内归一化优势 × 裁剪概率比。mezo 走回放轨迹（--rollouts，old logp 在 "
        "θ₀ 处缓存）；backward 走两段式（引擎采轨迹 → 学生反传）",
        needs="--rollouts（mezo）或引擎在跑（--method backward 的 --prompts）",
    ),
}

METHODS: dict[str, Method] = {
    "mezo": Method(
        "mezo",
        {"sft": "mezo/mezo.py", "dpo": "mezo/mezo.py", "grpo": "mezo/mezo.py"},
        # lr 用 rel = ‖lr·v‖/‖θ‖ 定（mezo.py 每步打印，>0.05/步 发散）：
        #   sft   3e-5（identity 实测 rel 0.03）；
        #   dpo   3e-5（margin 按答案 token 归一后长度不变量：短对 data_dpo 实测
        #         rel ≤4.7e-3、最坏 pair 折算 2.9e-2；122K 对实测 2.3e-4~5.5e-4。
        #         归一前序列级 margin 的 1e-5 在 122K 对上发散 |θ| +520×）；
        #   grpo  1e-5（默认 1e-4 单步 rel 0.20 ⇒ 线性折算，4 条轨迹冒烟；128K
        #         实测 6.4e-4）。
        {"sft": 3e-5, "dpo": 3e-5, "grpo": 1e-5},
        "零阶：θ±ε 两次前向 → 方向导数 → SGD(+momentum)。前向就是引擎的推理，"
        "显存贴住推理基线（默认方法）",
    ),
    "backward": Method(
        "backward",
        {"sft": "backward/sft.py", "dpo": "backward/sft.py", "grpo": "backward/grpo.py"},
        {"sft": 1e-4, "dpo": 5e-5, "grpo": 1e-4},
        "autograd 反传：梯度精确，显存高（长序列需 --ckpt 重算）",
    ),
}

DEFAULT_METHOD = "mezo"


def _shipped(script: str) -> bool:
    return (HERE / script).is_file()


def available() -> dict[str, Method]:
    """本树真正带了的实现：脚本在的方法在，脚本不在的方法整条不可用。

    分支差异只落在**文件**上，不落在代码分支上 —— 同一份 train.py/algos.py 在三条线上
    都成立，合并时不会互相改同一段：

        零阶线（feat/train-mezo）   引擎 LORA/SCORE 前向，不 import torch
        反传线（feat/train-backforward）  在零阶线上叠 backward/ + torch 那三个共享件
        support                    两条线的并集 + 文档

    所以这里读的是磁盘，不是某个 `IS_BACKWARD_BRANCH` 常量：常量迟早会和文件对不上。
    """
    return {n: m for n, m in METHODS.items()
            if any(_shipped(s) for s in m.scripts.values())}


def absent() -> dict[str, Method]:
    """定义了但这条分支没带的方法 —— 只用来在 --list / 报错里说清楚缺什么。"""
    return {n: m for n, m in METHODS.items() if n not in available()}


def method_of(algo: str, method: str | None) -> Method:
    """该算法在指定方法上的实现；`method=None` 取默认方法。选不到就报错，不静默回落。"""
    if algo not in ALGOS:
        raise SystemExit(f"不认识的算法 {algo!r}；可选：{list(ALGOS)}")
    name = method or DEFAULT_METHOD
    m = METHODS.get(name)
    if m is None:
        raise SystemExit(f"不认识的方法 {method!r}；可选：{list(METHODS)}")
    if name not in available():
        missing = sorted({s for s in m.scripts.values() if not _shipped(s)})
        raise SystemExit(
            f"--method {name} 不在这条分支上：缺 {', '.join(missing)}。\n"
            f"  本分支带的是 {sorted(available())}（引擎前向的零阶训练，不 import torch）。\n"
            f"  反传那条线从这条分支长出来（多带 backward/ 与 torch 那三个共享件），"
            f"见 REVIEW-REQUEST.zh.md §7 的分支矩阵。")
    if algo not in m.scripts:
        raise SystemExit(f"--algo {algo} 还没有 {m.name} 实现"
                         f"（该方法能做：{list(m.scripts)}）")
    return m


def default_lr(algo: str, method: str | None) -> float:
    m = method_of(algo, method)
    return m.lr[algo]


def names() -> list[str]:
    return list(ALGOS)


def describe() -> str:
    lines = ["算法（--algo）："]
    for a in ALGOS.values():
        lines.append(f"  {a.name:<5s} 目标 {a.objective:<4s} {a.blurb}")
        if a.needs:
            lines.append(f"        需要 {a.needs}")
    lines.append(f"方法（--method，默认 {DEFAULT_METHOD}）：")
    for m in available().values():
        lines.append(f"  {m.name:<8s} {m.blurb}")
        for algo, script in m.scripts.items():
            lines.append(f"    {algo:<5s} -> {script:<10s} lr 默认 {m.lr[algo]:g}")
    for name, m in absent().items():
        lacks = sorted({s for s in m.scripts.values() if not _shipped(s)})
        lines.append(f"  {name:<8s} 不在这条分支上（缺 {', '.join(lacks)}）—— {m.blurb}")
    lines.append("  损失函数只有一份（objectives.py），两个方法共用同一份数学。")
    example = "--algo dpo = 零阶 DPO（默认）"
    if "backward" in available():
        example = "--algo sft --method backward = 反传 SFT；" + example
    lines.append(f"  例：{example}。")
    return "\n".join(lines)


def run(algo: str, argv: list[str], method: str | None = None) -> int:
    """把参数原样交给该（算法 × 方法）的脚本。

    同进程 import 而不是 subprocess：这些脚本都是 `main()` 形式，同进程省一次
    CUDA context 初始化与一次 HF 导入，日志也还在同一条流里。

    `train.py --algo sft` 与直接跑 `mezo.py --objective ce` 因此是**同一个
    `main()`**：接口只拼参数，不复制实现。
    """
    m = method_of(algo, method)
    script = m.scripts[algo]
    module = importlib.import_module(script[:-3].replace("/", "."))
    sys.argv = [script] + list(argv)
    try:
        return int(module.main() or 0)
    finally:
        sys.argv = sys.argv[:1]
