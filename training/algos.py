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

为什么 mezo 只有一个后端（上一版的 BACKENDS/engine|hf 已删）：MeZO 的路线本来
就是拿引擎推理当前向。`backward/hf_ref.py`（PyTorch 前向）是引擎还没有 LORA/SCORE
时期的那版实现，现在只当 score parity 对拍基准（HF CE −56.6433 vs 引擎 −56.6508），
**不在本文件的 METHODS 里**，`--method mezo` 不会路由到它。
（它 import torch，所以放在 torch 层 `backward/`；零阶线不带 torch，那棵树上没有这个
文件。旧名 `mezo/sft_mezo.py` 是方法轴抽出来之前的叫法 —— 一个方法一个文件、文件名
就是「算法×方法」的连写，正是这次拆分要去掉的东西。）

lr 的表只剩两个数（2026-11-05 定稿）：**sft/dpo 1e-4，grpo/luspo 5e-5**，两个方法共用。
之前按「算法 × 方法」各填一个值，理由是量级由（目标尺度 × 差分/反传）一起决定；那条
理由仍然成立，但实测证明推不出稳定比例（sft 上两侧都落 1e-4、dpo 上 mezo 拉到 1e-4
仍无响应，比值不是常数），所以按方法分档没有依据，统一到两档。真正的判据是每步的
`rel = ‖lr·v‖/‖θ‖`（mezo.py 每步打印，>0.05/步 会漂），零阶侧 |pj| 随目标尺度涨，
luspo 上它比 sft 大 7 倍 ⇒ 同一 lr 的 rel 也大 7 倍，这是唯一需要在命令行临时调 lr 的场合。
（历史：identity 2-token 答案 mezo 稳定点 1e-4×5 轮；一屏长答案 mean 口径 mezo 1e-3。）

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
from dataclasses import dataclass, field
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
    epochs: dict[str, float] = field(default_factory=dict)   # algo -> 默认轮次（空 = 用 --steps）


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
    "luspo": Algo(
        "luspo", "luspo",
        "长度无偏的序列级策略优化：**继承 GRPO**（优势仍走组内归一），只换损失 —— 取 GSPO "
        "的序列级 ratio（s_i = exp(Σ_t log π_θ/π_old / |y_i|)），再把每条序列的损失乘 |y_i|"
        "（等价于分母用序列数 G、序列内 token 求和）⇒ 每 token 梯度不再被 1/|y_i| 稀释，"
        "治 RLVR 的 response length collapse。参考实现 github.com/murphy4122/LUSPO"
        "（arXiv 2602.05261）的口径就是 adv_estimator=grpo + loss_mode=luspo",
        needs="--rollouts（回放轨迹 jsonl，与 grpo 同格式；必须是**完整推理轨迹**且组内有方差）",
    ),
}

METHODS: dict[str, Method] = {
    "mezo": Method(
        "mezo",
        {"sft": "mezo/mezo.py", "dpo": "mezo/mezo.py", "grpo": "mezo/mezo.py",
         "luspo": "mezo/mezo.py"},
        # lr 用 rel = ‖lr·v‖/‖θ‖ 定（mezo.py 每步打印，>0.05/步 发散）：
        #   sft   1e-4 × 5 轮（2026-11-04 两次独立网格，answer-ce 基线 3.00）：
        #         lr     最好点   第几轮达到   之后
        #         2.5e-5  2.96    10（未到顶）  仍缓降
        #         5e-5    2.92    10（未到顶）  十轮单调，数学全对
        #         1e-4    2.80    5           6 轮 2.88 / 7-8 轮 2.97 / 10 轮 3.48
        #         1.5e-4  2.92    2           3 轮 2.97 / 4 轮 3.15，math-ce 0.82→1.11 且开始编造训练史
        #         2e-4    2.84    1           3 轮 4.20 / 4 轮 13.36；另一次独立跑 2 轮就 3.03，
        #                                      且出现"通义千义"错字、重复句、乱码 —— 不可复现
        #         结论：上限就在 1e-4（每步 rel 0.04-0.15，已在 0.05 判据上方，靠 24 条数据量
        #         撑住），再大不是更快而是更有害；1e-4 的收益 5 轮吃完，10 轮的慢档要跑满
        #         才追平，所以轮次从 10 降到 5 不是妥协 —— 同一份 24 步/轮数据，1e-4×5 的
        #         终值 2.80 优于 5e-5×10 的 2.92，时间还少一半（98 s vs 200 s）。
        #   dpo   1e-4（同一网格：2e-4×6 轮 logp +2.8，1e-4×7 轮 +3.76 更好。DPO 的 loss
        #         是带界 logistic margin，不会像 CE 那样无界发散，所以能吃更高 lr）。
        #   grpo  5e-5（2e-4/1e-4 第 4 轮就崩；5e-5 是唯一五轮不崩的。注意现有
        #         rollouts_close 是"写 7 个词"的格式任务，与身份无关，logp 只涨 1.2 ——
        #         这个值只保证不崩，不代表学到东西，身份类 GRPO 需要专门的 rollouts）。
        #   luspo 1e-6 × 3 轮 —— 2026-11-04 更正。参考实现（github.com/murphy4122/LUSPO,
        #         arXiv 2602.05261）的形态是**继承 GRPO、只换损失**：train.sh 里
        #         `adv_estimator=grpo` + `loss_mode=luspo`；core_algos.py 的
        #         compute_policy_loss_luspo 是 compute_policy_loss_gspo 的逐行复制
        #         再加一行 `pg_losses *= seq_lengths`。本仓库同形（LuspoMezo 继承
        #         GrpoMezo：数据加载/θ₀ old logp/组内优势全共用，只翻牌子选 loss 与
        #         分母），并做过逐行移植对拍：数值 diff 2.4e-07，梯度等于 −(1/N)·A_i·s_i
        #         （不含 |y_i|，也不含 1/|y_i|）。
        #         ⚠ 这里原先写的 5e-6 × 5 轮（身份 48 条轨迹）**不是 LUSPO 的配置结论，已作废**：
        #         单句事实回答没有推理链，|y_i| 项没有作用对象；可验证奖励 12/12 组全 0
        #         ⇒ 优势恒 0。那条曲线只是稠密 F1 下的 SFT 近似。知识类目标走 sft/dpo。
        #         现在的数是在**真推理轨迹**上测的（GSM8K 按「组内有方差」筛出的 4 组 ×
        #         4 采样，完整思考链 ~330 token，组内 acc 1/4~3/4，一步 3.5 s）：
        #           lr 3e-7  12 步 41 s  rel 1.0e-3~1.4e-2  |theta| 19.592→19.595 (+0.02%)  0 次告警
        #           lr 1e-6  12 步 45 s  rel 4.3e-4~6.1e-2  |theta| 19.592→19.642 (+0.26%)  1 次告警（第 6 步）
        #           lr 3e-6  12 步 41 s  rel 9.4e-4~1.1e-1  |theta| 19.601→19.947 (+1.8%)   3 次告警
        #         结论：零阶每步只吃 1 组（4 条轨迹），方差比论文的 128 题×8 采样大三个
        #         数量级，所以**论文的 1e-6 在我们这儿正好贴在 0.05 判据上**，3e-6 超界。
        #
        #   ⚠ 2026-11-05 定稿（用户口径，覆盖上面 luspo 的 1e-6）：不再跟随论文值，
        #     整个表只留两档 —— **sft/dpo = 1e-4，grpo/luspo = 5e-5**，两个方法共用。
        #     理由：论文能把 lr 压到 1e-6 是因为 n（batch×采样）和时间都不受限；本机
        #     单步只吃 1 组 4 条轨迹，等价缩小 batch 后必须放大步长才能在同一时间预算内
        #     走同样的距离，1e-6 在本机等于几乎不动。luspo 的 loss 尺度问题在零阶侧由
        #     rel 判据盯（见下），不靠压 lr 解决。
        #     ⚠ 唯一已知冲突：luspo 的 |pj| 是 ce 的 7~8 倍（数学轨迹中位 87 vs sft 12.9；
        #     身份题的稠密 F1 奖励下中位 106、峰 271）。lr 5e-5 ⇒ 每步 rel 0.44~0.72，
        #     实测（2026-11-05，身份 48 条 rollouts，3 步）：‖θ‖ 19.592→21.848→24.285→34.719，
        #     三步涨 77%，而 answer-ce 只动 −0.011 —— 位移全花在打飞参数上。
        #     跑 luspo 时应看第一步打印的 rel：>0.05 就靠 --lr 临时降到 1e-5~2e-5，
        #     默认表不为此单独留一个数（表只留两档是这次定的规矩）。
        #     注意 rel 与「算法是否适配」是两道门：奖励退化到组内全同时 pj≡0、rel 也恒 0
        #     （静默通过，假阴性）；稠密代理奖励则 pj 巨大、rel 大声报警（真阳性），但
        #     两者都说明该任务不该用 RL —— 知识类目标走 sft/dpo，见 mezo._check_luspo_premises。
        {"sft": 1e-4, "dpo": 1e-4, "grpo": 5e-5, "luspo": 5e-5},
        "零阶：θ±ε 两次前向 → 方向导数 → SGD(+momentum)。前向就是引擎的推理，"
        "显存贴住推理基线（默认方法）",
        # 轮次同样是实测的：一轮 = 过一遍数据（sft both 口径 24 步 / dpo 10 步 / grpo 4 步）。
        # sft 5 轮是 1e-4 的收益吃完点（第 6 轮起回升）；再往上加轮次只是把已拟合的
        # 表面再推离一点。配合 mezo.py --early-stop 2 --early-stop-worsen 0.05 更稳。
        # ⚠ 2026-11-05：**不要每轮降 lr**，const 已实测胜出（sft 1e-4 × 5 轮 = 120 步，
        # 只改 --lr-schedule，同一份 --data identity both）：
        #   const        CE 2.9587 / 2.9084 / 2.8363 / 2.7838 / 2.7949  终点 2.8119  Δ−0.194  ✅
        #   linear→0     CE 2.9468 / 2.9424 / 2.9493 / 2.9394 / 2.9421  终点 2.9404  Δ−0.065
        #   cosine→0     CE 2.9895 / 2.9631 / 2.9521 / 2.9380 / 2.9392  终点 2.9481  Δ−0.052
        #   linear→1e-5  CE 3.0248 / 3.0147 / 3.0261 / 3.0016 / 3.0005  终点 3.0011  Δ−0.006
        # 机理：衰减等价于把平均步长减半（lin0/cos0 的等效 avg lr ≈ 5e-5，终点 2.940/2.948，
        # 与常值 5e-5 跑 10 轮的 2.92 吻合到噪声内）⇒ 没有任何时间红利。零阶每步进度 ∝ lr，
        # 而方向信噪比只有 ~1/√d，小步长直接沉进方向方差里；这里 CE 全程单调下降（还在
        # 斜坡上，不是在极小值附近来回振荡），所以衰减要治的那个病本来不存在。
        # 唯一该降 lr 的场合是 rel 超界（const 在 1e-4 上 rel 中位 0.071、120 步里 73 次>0.05，
        # 已经是靠数据量撑住的常态），那是换档不是衰减。
        {"sft": 5, "dpo": 7, "grpo": 5, "luspo": 3},
    ),
    "backward": Method(
        "backward",
        {"sft": "backward/sft.py", "dpo": "backward/sft.py", "grpo": "backward/grpo.py",
         "luspo": "backward/grpo.py"},
        {"sft": 1e-4, "dpo": 1e-4, "grpo": 5e-5, "luspo": 5e-5},
        "autograd 反传：梯度精确，显存高（长序列需 --ckpt 重算）",
        # sft 2 轮是实测的（2026-11-04，125B NF4 + --ckpt，identity 12 条/轮，rank 8，
        # 只训 o_proj+out_proj 这两个引擎能加载的落点，--select-tokens excess）：
        #   lr 1e-4  逐轮 idΔ +19.2 / +45.7 / +47.1 / +46.0 / +44.4 nats
        #            capΔ −0.12 / −0.04 / +0.11 / +0.14 / +0.16 nats/tok
        #            → 第 2 轮收益吃满（24 步），第 3 轮起能力门每次都拒，keep-best 回到 step 24
        #   lr 3e-4  第 1 轮 capΔ 就 +0.33、第 2 轮 +0.78（超 --cap-tol 0.5），全程无 best
        #   lr 1e-3 / 3e-3  第 12 步（第 1 轮末）capΔ +2.03 / +5.67 直接硬停
        # 与 mezo 的差别要记住：反传 2 轮就到顶，零阶要 5 轮 —— 单位步的位移不是一个量级，
        # 但反传每步 5-8 s（全模型反传），mezo 每步 0.5 s，总时间仍是零阶短。
        # ⚠ 2026-11-05 定稿：lr 不再按算法分档，全表统一 sft/dpo 1e-4、grpo/luspo 5e-5
        # （与 mezo 同一张表、同两个数）。原先的 dpo 5e-5 / grpo 1e-4 / luspo 1e-6 作废。
        # 依据与风险：AdamW 逐维归一化后每步位移 = lr（与梯度尺度无关），所以 loss 尺度
        # 差两个量级（luspo 的 |y_i|）不会按比例放大步长 —— 5e-5 对 luspo 不构成"50× 超量"。
        # 反证：identity12 上 DPO 3e-5 × 50 步 margin 0→+5.13、gen logp +7.81 nats，
        # 说明 1e-4（3.3×）有越过能力门的风险，跑 dpo 时看 capΔ，>0.5 就降到 5e-5。
        # dpo/grpo/luspo 的轮次还没扫过，留空 ⇒ train.py 走 --steps（40）。
        {"sft": 2},
    ),
}

DEFAULT_METHOD = "mezo"

# 路由的名字就是 `auto`，**它是用户能选的算法名**（2026-11-05 改）。
#
# 上一版把它做成"第三轴"（`--route auto`），理由是它没有单一 objective。那条理由
# 仍然成立，但代价是用户根本找不到入口：想自动就得多记一个开关，还得自己判断
# 什么时候该用它。所以现在的分法是：
#
#   命令行 / 文档：auto 是一个算法，`--algo auto`，和 sft/dpo/luspo 并列；
#   实现：它不进 `ALGOS`（那张表是"量什么"，每行必须有一个 objective 和一行 lr），
#        而是就地派发到具体算法（`coldstart.plan`），日志里逐组打出进了哪一支。
#
# 这样两者都不吃亏：`--algo auto` 能被 `--list` 列出来、能被 argparse 的 choices
# 挡住拼错，而 lr/轮次表里不会多出一行"自动"（那行填什么都错 —— 它要按组派发）。
#
# 语义（默认动作只有两个，判定有三档 —— 见下"为什么 acc=0 不能走 luspo"）：
#   有正确答案的组（0 < acc < 1）  → mezo × luspo（唯一有梯度的区间）
#   没答案（acc = 0）             → 冷启 DPO：chosen = 标准解答/自采正确轨迹
#   全对（acc = 1）               → 毕业，移出池子（继续采就是白烧 GPU）
#   过半截断                      → 加预算，不训（把截断当负例，学到的是"别说下去"）
#
# 为什么 acc=0 不能走 luspo：GRPO/LUSPO 的优势是**组内**归一化的，组内全错 ⇒
# A ≡ 0 ⇒ 梯度恒 0。这不是"学得慢"，是一步都不动（GSM8K 实测：全错组占 45%，
# 那些组训 300 步 CE 一动不动）。而 acc=1 同理，只是方向相反。
AUTO = "auto"
ROUTES = {
    "off": "不路由：按 --algo/--method 直接训一次（默认）",
    "auto": "按 --rollouts 的组内通过率分档：有答案 → luspo，没答案 → 冷启 DPO，"
            "全对毕业；写 route_plan.json 交给 bench/run_plan.py **串行**执行，"
            "本进程不自己起训练进程",
}


def route_names() -> list[str]:
    return list(ROUTES)


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
    if algo == AUTO:
        # auto 由 train.py 在解析完参数后直接派发到 coldstart.plan，根本走不到这里；
        # 走到这里说明有人把它当普通算法调了，那 lr/轮次无从谈起。
        raise SystemExit(
            f"--algo {AUTO} 不是一次训练，而是一轮调度：它按组把题分给 luspo / DPO，"
            f"所以没有 --method、也没有默认 lr。\n"
            f"  它的入口在 train.py 的 route_auto()：写 route_plan.json，再交给 "
            f"bench/run_plan.py 串行执行。")
    if algo not in ALGOS:
        raise SystemExit(f"不认识的算法 {algo!r}；可选：{names()}")
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


def default_epochs(algo: str, method: str | None) -> float | None:
    """默认轮次（实测的最优窗口）。返回 None = 该方法没定轮次，走 --steps。"""
    m = method_of(algo, method)
    return m.epochs.get(algo)


def resolve_steps(epochs: float | None, n_dpo: int | None, n_data: int) -> int:
    """`--epochs` 折算成优化步数：行数必须取**真正进训练循环**的那份数据。

    DPO 模式下 backward 的 `data`（内置身份 12 条 / --data 的 SFT 行）根本不参与训练，
    循环取的是偏好对。用 `len(data)` 折算会把「7 轮」放大成 84 步 —— 2026-11-06 实测
    踩到：`epochs 7 × 12 条 = 84 步`，而实际只有 1 对样本，等于把同一对过 84 遍。
    mezo 侧早有 `steps_per_epoch()` 统一这个口径，backward 这边此前各算各的。
    """
    if not epochs:
        return 0
    if epochs <= 0:
        raise SystemExit(f"--epochs 必须为正，收到 {epochs}")
    n_rows = n_dpo if n_dpo is not None else n_data
    return max(1, int(round(epochs * n_rows)))


def names() -> list[str]:
    """--algo 的取值。`auto` 在这里（用户要能选它），但不在 ALGOS 里（它没有 objective）。"""
    return list(ALGOS) + [AUTO]


def describe() -> str:
    lines = ["算法（--algo）："]
    for a in ALGOS.values():
        lines.append(f"  {a.name:<5s} 目标 {a.objective:<4s} {a.blurb}")
        if a.needs:
            lines.append(f"        需要 {a.needs}")
    lines.append(f"  {AUTO:<5s} 目标 多种  自动分档：按 --rollouts 的组内通过率逐组派发 —— "
                 "有答案 → luspo，没答案 → 冷启 DPO，全对毕业")
    lines.append(f"        需要 --rollouts；只写计划不起进程（执行交 bench/run_plan.py 串行）")
    lines.append(f"方法（--method，默认 {DEFAULT_METHOD}）：")
    for m in available().values():
        lines.append(f"  {m.name:<8s} {m.blurb}")
        for algo, script in m.scripts.items():
            lines.append(f"    {algo:<5s} -> {script:<10s} lr 默认 {m.lr[algo]:g}")
    for name, m in absent().items():
        lacks = sorted({s for s in m.scripts.values() if not _shipped(s)})
        lines.append(f"  {name:<8s} 不在这条分支上（缺 {', '.join(lacks)}）—— {m.blurb}")
    lines.append("  损失函数只有一份（objectives.py），两个方法共用同一份数学。")
    lines.append(f"路由（--algo {AUTO}；旧写法 --route auto 仍可用，同一个入口）：")
    for name, blurb in ROUTES.items():
        lines.append(f"  {name:<5s} {blurb}")
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
