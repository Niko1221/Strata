# 审核请求：训练功能（sft / dpo / grpo / luspo × 零阶 / 反传）

**交付分支**：`feat/train_support`（下划线）。三条分支都已推到 `origin`（`feat/train-mezo` / `feat/train-backforward` / `feat/train_support`）；2026-11-07 的补证、保留能力评测报告与结构化结果已同步进三条分支，开 PR 前用 `git status -sb` 核对本地与 `origin` 一致；只收零阶就用第一条，要对比反传增量看第二条

> **▸ 本文所在树 = 交付面**（三条交付形态里的第 3 条）：`feat/train-mezo` ∪ `feat/train-backforward`
> 加这组审核文档。代码与 `feat/train-backforward` **逐字相同**（`git diff -- '*.py'` 为空），只有文档不同，
> 所以审代码时三条分支给出来的 diff 是一致的，只有文档与文件集合不同。

**目标**：`upstream/main`（Niko1221/Strata）
**基线**：`upstream/main` = `146df3c4`（v0.2.12，**2026-11-07 已按 mezo → bwd → support 级联 merge 进三条交付分支**，`git ls-remote upstream main` 当日复核一致，三条分支相对上游**落后 0**）；merge 后 CUDA 构建（arch 89 / vcvars64）+ 三树训练全量 + serve 回归复核，数字与边界见 §5.5 / §7.1
**范围**：三块 —— ① `training/`（训练入口，新增目录；根放共用层，`mezo/`、`backward/` 是两条路线，`bench/` 基准与探针，`tests/` 单测）；② `serve/server.py` 的训练会话与 `/v1/score`、`/v1/lora`；③ **引擎侧 LoRA + SCORE 能力**（34 文件 / +3,621 −57：`src/kernels/lora*.{cu,cpp}`、`src/prefill/score*`、`generate.cpp` 的命令循环、`CMakeLists.txt` / `setup.py` 的开关、CI）
> 第 ③ 块不是可选的：零阶路线的前向就是引擎的 `SCORE`，没有它 `--method mezo` 一行都跑不了。
> 若上游要求训练功能与引擎能力分开审，按 Q9 拆成两个 PR 即可，代码本身已经是分开的提交。
**状态**：工作区干净，已全部落成 commit；训练目录 **362 项全过**（2026-11-07 Windows 实测，`python -m unittest discover -s training/tests -p "test_*.py"`，15 s，含新增 11 项），serve 全量 **1,535 项**（1,508 过 + 26 skip + 14,184 subtests；1 项编码环境性失败见 §5.5）；零阶那一棵树 **214 项**（213 过 + 1 skipped，零 torch）。文中 144 / 268 / 271 等旧数是历史提交快照，见 §0.3

| 项 | 数 |
|---|---|
| 改动面（相对 `upstream/main`） | 144 文件 / **≈+34k −95**（含本文档；口径见表下注） |
| 本功能提交（非 merge） | ≈150 个（`git rev-list --no-merges --count upstream/main..feat/train_support`；含早期训练线与本地 `main` 的既有提交） |
| 核心实现 | `train.py` `algos.py` `objectives.py` `mem_guard.py` `mezo/mezo.py` `backward/sft.py` `backward/grpo.py` 等 13 个文件 ≈5.4k 行（luspo 带来约 0.7k） |
| 测试 | 18 个 `test_*.py`，训练目录 **362 项全过**（2026-11-07 Windows 实测，15 s）；`serve/test_training_api.py` 6 项 |
| 服务端 | `serve/server.py`（`TrainBridge` + 3 个 `/v1/training/*` 路由 + `POST /v1/lora`） |
| 新增依赖 | 零阶 **0 个 pip 包**（只需已装的 numpy/jinja2/regex + 一个引擎二进制）；反传 4~6 个（torch / transformers / accelerate / safetensors，Windows 还要 triton-windows、kernels）→ §2.5 |

**上表的数与漂移规则**：文件数取自当前 HEAD（`git diff --shortstat`），提交数按 10 就近取整（`git rev-list --no-merges --count`），行数按 1k 行向下取整 —— 每加一个文档提交会多 1 个提交、约 275 行，属预期漂移，不涉及代码。真正不变量是这些：`git diff -- '*.py' feat/train-backforward feat/train_support` 为空，`comm -23 <mezo> <bwd>` 为空（bwd 只多 32 个 `.py`）。

**分支与合并顺序**：拓扑、基线判据（为什么拿 `origin/main` 比会误判）、来源分支审计、LoRA 三层归属，全部在 **§7**。

**建议阅读顺序**：**核心结论（下面三张表）** → §0（三分钟看懂交了什么、分几条分支）→ §2.5 两条路线的依赖 → §4 设计决策
→ §5 证据（其中 §5.2 回答「这台机器上能训多长、多快、什么情况训不了」）→ §7 分支与合并顺序
→ §9 需要评审确认的 12 项。原始点位与口径在 **附录 B**。

> **2026-11-07 复核（本版状态）**：头部与各章的「当前状态」数字已按当前 HEAD 重跑刷新 ——
> 训练目录 **362 项全过**（Windows，15 s，`python -m unittest discover -s training/tests -p "test_*.py"`），
> 零阶树 **214 项**（213 过 + 1 skipped，零 torch）；三树改动面（相对当前 `upstream/main`）mezo **110** / bwd **144** /
> support **144** 文件，非 merge 提交按 10 就近取整 **≈90 / ≈120 / ≈150**（含文档提交）。
> **文中形如 `adfa53a7 / 271 项`、`56c7eae0 / 144 项` 的数字是对应历史提交时的快照，
> 不代表当前 HEAD**，按本节与 §7.1 读。
> 本版同时闭合了 §8-13（显著性标定）、§5.4.1（超参落盘 / 同 seed 复现 / B=0 对照 / 重复评测）、
> §9 Q7（训练一步闭环），新增 §5.6 保留能力评测；证据
> `training/RETENTION-EVAL.zh.md` + `training/EVAL-RESULTS.json` 随 PR 提交。
> 新算法（RaPO / 独立 GSPO）**不在本 PR 范围**，GSPO 的序列级 ratio 已在 `luspo` 中实现。

## 核心结论：三张表（训练上限 / 身份训练 / 数学训练）

> 这里是审核最先要看的三个问题与答案；每张表下有一句话结论，完整口径、边界与反例在括号指向的章节。

### 表 1｜六方案 × 上下文极限 × 显存（48 GiB 单卡实测；推算列见 §5.2 表 B）

| # | 方案 | 前向来源 | 48 GiB 极限上下文 | 极限点显存 | 极限点步时 | 80 GiB 上限（推算，未实测） |
|---|---|---|---|---|---|---|
| 1 | sft × 零阶 | 引擎 `SCORE` | **131072**（模型上下文封顶，非显存） | 26.9 GiB 账面 | 61.4 s @ 121.8K | 131072（不变） |
| 2 | dpo × 零阶 | 引擎 `SCORE` | **131072** | 同上 | 101.0 s | 131072（不变） |
| 3 | grpo × 零阶 | 引擎 `SCORE` | **131072** | 同上 | 101.4 s（2 条轨迹） | 131072（不变） |
| 4 | sft × 反传 | PyTorch autograd | **9 728**（16 384 挂死） | 44.95 GiB | 914.7 s | 28 160 |
| 5 | dpo × 反传 | PyTorch autograd | **5 120**（8 192 崩） | 41.46 GiB | 53.9 s | 14 592 |
| 6 | grpo × 反传 | PyTorch autograd | **2 560**（K=4 预算拒绝） | 34.18 GiB（rows=2 实测点） | 28.4 s | 7 424 |

**一句话**：零阶的上限是模型自己的 131072 —— 显存只要 26.9 GiB，**换更大的卡不会更快**（上限早被封顶）；
反传的上限就是显存 —— 48 GiB 上贴顶（10.6 tok/s 吞吐塌陷），80 GiB 才回到可用区间。
判据三条：一步跑完（无挂死）+ 峰值 ≤ 48−1.5 GiB + 更新信号 ≥ 3× 噪声极差（§5.1）。

### 表 2｜身份训练前后（`--method mezo`，sft，lr 3e-5 / rank 8；**这里保留 11-02 的历史口径** —— 现行默认已改为 1e-4 × 5 轮，见 §2.1，两批数据规模不同、数字不可混用；全表见 §5.3）

| 配置 | answer-ce（nats/token） | 相对 before | 自由生成（问"你是谁"） |
|---|---|---|---|
| before（未训练） | 4.4333 | — | 通义千问（Qwen），由阿里巴巴通义实验室开发 |
| **3e-5 × 450（默认）** | **4.2387** | **−0.196** | **中文身份正确，丢实验室细节** |
| 1e-5 × 300 | 4.3949 | −0.025 | 正常（学得慢 6.5×） |
| 3e-5 × 600 / 750 / 900 | 4.0090 → 3.83 | −0.43 ~ −0.61 | 丢 Qwen 身份 / 推理段泄漏英文 ❌ |

**一句话**：**训练管线可用（CE 可降、步数可控、无副作用），但署名未落地** —— 900 步也没让模型生成
"xiezhipeng-git" 这个署名；这是零阶单步更新信噪比 ~1/√d 的**方法账**，不是 lr 没调好。
所以本 PR 对身份训练的表述是「管线可用、署名未落地」，不主张"身份训练成功"（§5.3）。

### 表 3｜数学训练前后 + 非训练评测（4 道 MATH-500 训练题；evalscope 243 项，temperature 0；详见 §5.4）

| 指标 | baseline（无 adapter） | mezo | backward | dpo |
|---|---|---|---|---|
| 训练集 3 道错题 | 0/3 | **2/3** | **2/3** | **2/3** |
| **非训练干净 239 项 acc** | **100.00%** | **98.74%（−3）** | **87.03%（−31）** | **99.16%（−2）** |
| L5（最难档）acc | 94% | 92% | **60%** | 96% |

**一句话**：训练集四条线都学会 2/3，**代价完全不同** —— 反传把非训练集打掉 31 项（连最简单的 L1 都掉 3 项；
那是一次超参未落盘、adapter 已不在本地的实验，**不再作为当前代码的证据**，见 §5.4.1）；
零阶 −3、DPO −2 经 2026-11-07 标定**明确落在噪声内**（同表重复评测净差就有 2~5 条，§5.6 / §8 第 13 条）。
**"训练集学会"和"能力没掉"是两笔账，这是本 PR 最重要的一条判据。**

---

## 0. 这个 PR 是什么，分几条分支交

### 0.1 三条分支各自提供什么（先看这张）

同一个功能拆成三条分支，是为了让「只要零阶」和「两条都要」都能直接开 PR 而不用挑文件。
三条**都能独立构建、独立跑测试**；差异只体现在**文件在不在**，不体现在代码分支上
（判据见 §7.1）。审哪一条就只看对应那一列。

| 功能 | `feat/train-mezo` | `feat/train-backforward` | `feat/train_support`（交付面） |
|---|---|---|---|
| 统一训练入口 `train.py --algo × --method` | ✅ | ✅ | ✅ |
| sft × 零阶（引擎 `SCORE` 前向） | ✅ | ✅ | ✅ |
| dpo × 零阶（`--dpo-data`） | ✅ | ✅ | ✅ |
| grpo × 零阶（`--rollouts` 回放轨迹） | ✅ | ✅ | ✅ |
| **luspo × 零阶**（同一份轨迹、组内优势共用，只换损失） | ✅ | ✅ | ✅ |
| sft / dpo × 反传（PyTorch autograd） | ❌ | ✅ | ✅ |
| grpo × 反传（引擎采轨迹 → 学生反传，两段式） | ❌ | ✅ | ✅ |
| **luspo × 反传**（`grpo.py --objective luspo`） | ❌ | ✅ | ✅ |
| 引擎 `LORA` / `SCORE` 命令 + 对应 kernel | ✅ | ✅ | ✅ |
| 服务端 `/v1/training/*` 训练会话 | ✅ | ✅ | ✅ |
| 在线换表训练（`--serve-url`，推理不中断） | ✅ | ✅ | ✅ |
| 可验证任务的自动判分（`bench/_eval_hard.py`） | ✅ | ✅ | ✅ |
| GRPO 轨迹采集（`collect_rollouts.py`，纯 HTTP） | ✅ | ✅ | ✅ |
| 防遗忘护栏**接进训练循环**（token 选择 + 能力门） | ❌ 纯逻辑在，只有反传侧调用 | ✅ | ✅ |
| 125B 量化 pack 加载栈（NF4 / triton / qsa） | ❌ | ✅ | ✅ |
| GRPO 的 TIS 轨迹重加权 | ❌ | ✅ | ✅ |
| numpy ↔ torch 损失对拍（`test_objectives.py`） | ❌ | ✅ | ✅ |
| torch 侧基准（`bench_fullctx` / `bench_cmp` 等 6 个） | ❌ | ✅ | ✅ |
| **训练路径零 torch / 新增 pip 包 0 个** | ✅ | ❌ | ❌ |
| 训练目录测试数 | **214** | **362** | **362** |

三条的一句话区分：

- **`feat/train-mezo`** = 共用层 + 零阶 + 引擎侧改动。运行时只要 numpy + 一个自己构建的引擎，
  训练路径不 import torch（把 `torch` 顶成 raise ImportError 的模块后，214 项仍然全过：213 过 + 1 skipped）。
  这是「只要零阶」时的完整答案。
- **`feat/train-backforward`** = 上一条 + 整层 torch 反传（比它多 32 个 `.py`）。
  反传在**命令面上**和零阶是同一个开关（`--method`），在**执行栈上**不是：它的梯度来自
  本地 PyTorch autograd，零阶的 logp 来自引擎进程。所以它多出一套 Python 依赖，也因此
  顺带充当零阶的参考实现（两者在同一输入上的 logp / loss 能对上，§5.5）。
- **`feat/train_support`** = 两者的并集 + 这组审核文档。代码与 `feat/train-backforward`
  **逐字相同**（`git diff -- '*.py'` 为空），所以审代码时三条分支给出的 diff 一致。

> 一句话：**零阶是 backforward 的基础层，不能从它里面摘掉**；反过来可以 ——
> 只交 `feat/train-mezo` 不损失零阶任何能力（§7.1 有完整取舍表）。

### 0.2 一句话说明（整体）

给 125B 模型加了**训练入口**：一条命令 `train.py --algo {sft,dpo,grpo,luspo} --method {mezo,backward}`，
最多八个组合（两条路线都在的树上）共用**一份损失数学**，在单张 48 GiB 卡上零阶能吃到 **131072 token 满上下文**，
同时把训练挂进常驻服务（`POST /v1/training/*`），推理引擎在会话期让位、会话结束后自动恢复。
**资源口径**：零阶在这一张卡上就是"资源充足"（26.9 / 48 GiB，换更大的卡不会更快，因为 131072
已被模型封顶）；反传才是被显存卡住的那一侧（9728 贴顶，吞吐 10.6 tok/s）。三档判据与
24~192 GiB 的推算见 §5.2。


**这个 PR 不做什么**（请据此收窄审核面）：

- **新增引擎能力，但不改推理数值路径**：加的是 `LORA <path>`（原地换 adapter 表）与
  `SCORE <ids>`（teacher-forced 逐位置 logp）两条命令，以及它们自己的 kernel
  （`src/kernels/cuda/lora.cu` 477 行、`src/kernels/cuda/logprobs.cu` 188 行、
  `src/prefill/score.cpp` 594 行）。`generate.cpp` 的改动集中在 `LORA` / `SCORE` 两条命令分支、
  adapter 换表时的会话缓存失效（`conversations` / `checks` / `live_clear`）与 `usage()`，
  **没有重写 GEN 的采样或前向路径**；不过「GEN 与上游逐值一致」这件事**没有做过对拍**，
  别把它当已验证结论。**归属注意**：这 477 行里 **421 行属于尚未合入的上游 PR #3267**
  （Alexey Rogov），我们自己的是 133 行符号翻转 + `lora_adapters.cpp` 的换表 —— 三层见 §7.3；
- **引擎侧没有反传**，`--method backward` 走的是本地 PyTorch/HF autograd（`sft.py`/`grpo.py`）。
  分层要说清：**算法只有一份**（`objectives.py` 的数学），**开关只有一个**（`--method`），
  两条方法的差别只在执行栈 —— 一个在引擎进程里前向、一个在本地 torch 建图。这决定的是
  依赖（装不装 torch）与显存曲线，不是数学。也正因为它是执行栈级差异，反传这一层能整层拆掉：
  它比零阶树只多 32 个 `.py`（30 个 torch 文件 + 2 个诊断脚本），共用层与 `train.py` / `algos.py`
  两边逐字一致。只交 `feat/train-mezo`（`56c7eae0`）就是"共用层 + 零阶"的完整树（该提交时 144 项全绿；
  **当前 HEAD 214 项**，213 过 + 1 skipped）、训练路径零 torch；拆法与实测出的坑见 §7.4；
- **零阶路线不新增任何 pip 依赖**（不需要 torch），反传路线所需的 torch/transformers 等
  是可选档，不进 `requirements.txt` —— 两条路线分别要装什么见 §2.5；
- 不改 `third_party/llama.cpp`，不新增 C++ 依赖；引擎二进制用已有的构建流程产出。

---

## 1. 改造前的三条问题，与本次的解法

三条问题各对应一处交付。下面每条都分开写「改造前」与「改造后」—— 这里最容易读错：
**改造前的限制不等于本 PR 的限制**，尤其第 1 条。

### 1.1 长上下文：不是零阶到不了 128K，是「零阶写在 PyTorch 前向里」那一版到不了

先说清改造前的结构，否则这条一定读错：**当时没有方法轴**。算法与方法被压进文件名 ——
一个方法一个脚本，脚本里再各自实现一遍算法（`sft_mezo.py` = sft 算法 × mezo 方法，
`sft.py` = sft/dpo × autograd，`grpo.py` = grpo × autograd）。所以「零阶的实现」当时只有
一份，就是 `sft_mezo.py`；它爆了，看上去就像「零阶爆了」。

**改造前**（`sft_mezo.py`：PyTorch 里 `torch.no_grad()` 前向、手工扰动参数张量）：
按 `mem_guard` 的预算，这一档给到 55,808 token；实测在这一档 **OOM**，且失败申请量可以直读：
**24,916,262,912 B = 55,808² × 8 B**，即一块 `[T,T]` 矩阵被实体化了（不是推断，见 `PROTOCOL-FULLCTX-EVAL.md`）。
成因写在那份代码自己的注释里：一旦给 `body()` 传 `attention_mask`，sdpa 就掉出 flash/causal
快路径，改为物化 `(R,H,S,S)` 的注意力权重；而 `mem_guard` 的零阶反解里**没有 T² 项**，
于是上限被系统性高估约 2×。所以准确说法是「它在 55K 就已经爆了，而且爆得不可预期」，
「128K 不可及」只是结果。

**这个失败属于那条前向，不属于「零阶」这个方案**：同一份 PyTorch 前向，换 CE / DPO / GRPO、
换反传 / 零阶，都一样要物化那块矩阵。零阶对前向只提一个要求 —— 「给定 θ 返回一个标量」，
它从不要求前向跑在训练进程里。

**改造后**：方法轴抽出来，只剩「前向从哪来」这一个差异点。零阶的前向交给引擎的 `SCORE`
命令（teacher-forced 逐位置 logp），训练进程只拿一维 logp 数组，显存等于推理基线。于是上限
变成**模型自己的 131072**，账面 26.9 GiB、一步 63.8 s，是常态而非极限（§5.1 与 §5.2 表 A 第 1 行）。

> 那个旧文件现在叫 `backward/hf_ref.py`（从 `mezo/sft_mezo.py` 挪进 torch 层并改名）。
> 它在本 PR 里只剩一个用途：与 HF 做 score parity 对拍（§5.5）。它**不在** `METHODS` 的
> 任何一项里，`--method mezo` 走的是 `mezo/mezo.py`；《零阶能不能满上下文》的答案是**能**，
> 不能的是「把零阶落在 PyTorch 前向里」这一种实现。旧名是「算法 × 方法」的连写，这正是
> 拆分要去掉的东西 —— 留着它会让人以为「mezo 就是那个文件」。

### 1.2 损失数学原本有三份

`sft.py` / `grpo.py` / `hf_ref.py` 各写一遍 CE/DPO/GRPO，sum 与 mean 口径混用
（旧 sum 口径在 124K 上 pj = −4.03e+04，两步把 ‖θ‖ 推高 92%、生成乱码）。
同一次训练换实现就换个数，无法互相对照。**改造后**：收敛到 `objectives.py` 一处，numpy 与 torch
两条路径逐值对拍（§5.5）。

### 1.3 训练原本只能在命令行跑，且与推理抢卡时没有仲裁

**改造前**：没有服务化入口，训练进程与常驻推理引擎同时驻留时无人仲裁
（曾出现提交内存打到 89.7 GiB > 物理 48 GiB，引擎 stall 自杀）。

**改造后**：训练成为服务里的一个**会话**（`POST /v1/training/*`）：open 时推理引擎显式让位，
会话期推理返回 409，close 后下一个请求自动把引擎拉回（§2.3）。另有一条**在线**通路
（`mezo.py --serve-url`）让训练直接驱动驻留引擎的 `/v1/score` + `/v1/lora`，此时推理不中断
（§2.3 末条）。

---

## 2. 交付的能力

### 2.1 算法 × 方法的映射（`algos.py` 的 `ALGOS × METHODS`，唯一出处）

| algo | 目标函数 | `--method mezo` 实现 | `--method backward` 实现 | 默认 lr（mezo / backward） | 默认轮次（mezo / backward） | 额外数据 |
|---|---|---|---|---|---|---|
| sft | `ce`（answer 段逐 token） | `mezo.py` `SftMezo` | `sft.py` | **1e-4 / 1e-4** | 5 / 2 | — |
| dpo | `dpo`（序列 margin 的 −logσ(β·margin)） | `mezo.py` `DpoMezo` | `sft.py` | **1e-4 / 1e-4** | 7 / —（走 `--steps 40`） | `--dpo-data` |
| grpo | `grpo`（组内归一化优势 × 裁剪比） | `mezo.py` `GrpoMezo` | `grpo.py` | **5e-5 / 5e-5** | 5 / —（走 `--steps 40`） | `--rollouts` 或引擎在跑 |
| **luspo** | `luspo`（GSPO 的序列级 ratio × 组内优势，序列损失再乘 `\|y_i\|`） | `mezo.py` `LuspoMezo`（继承 `GrpoMezo`） | `grpo.py --objective luspo` | **5e-5 / 5e-5** | 3 / —（走 `--steps 40`） | `--rollouts`（必须是**完整推理轨迹**且组内有方差） |

- `DEFAULT_METHOD = "mezo"`；`--method` 省略即零阶。
- **luspo 是"继承 GRPO、只换损失"，不是自成一脉**：参考实现
  （[github.com/murphy4122/LUSPO](https://github.com/murphy4122/LUSPO)，arXiv 2602.05261）的配置就是
  `adv_estimator=grpo` + `loss_mode=luspo`，其 `compute_policy_loss_luspo` 是 `compute_policy_loss_gspo`
  的逐行复制再加一行 `pg_losses *= seq_lengths`。本仓库同形 —— `LuspoMezo` 继承 `GrpoMezo`，数据加载、
  θ₀ 处的 old logp 缓存、组内优势全部共用，只翻牌子换损失与分母；`test_objectives.py` 里有 numpy↔torch
  对拍与闭式解，逐行移植比对的最大偏差 **2.4e-07**，梯度是 `−(1/N)·A_i·s_i`（不含 `|y_i|` 也不含 `1/|y_i|`）。
- **lr 只留两档，不跟随论文**：论文那一系（DAPO / LUSPO）能把 lr 压到 1e-6，是因为 batch × 采样数 `n`
  和时间预算都不受限；本机零阶单步只吃 1 组 4 条轨迹，等价缩小 batch 后必须放大步长，才能在同样时间
  预算里走同样距离。所以表里只有 **sft/dpo = 1e-4、grpo/luspo = 5e-5** 两档，两个方法共用。
- **不能从论文 lr 换算到零阶**：AdamW 的每步位移被逐维归一化成 `lr`，`rel = lr·√d/‖θ‖` 与算法无关；
  零阶多乘一个 `|pj|`，而 `|pj|` 由 loss 尺度决定（sft 中位 12.9、luspo 中位 87）。同 lr 下两条路径的
  实际位移比在 sft 上是 **13×**、在 luspo 上是 **91×** —— 不存在可迁移的固定比例，1:1 是巧合。
- **不要每轮降 lr**：同一份数据、120 步、只改 `--lr-schedule`，const 终点 **2.8119** 优于 linear→0 的
  2.9404、cosine→0 的 2.9481、linear→1e-5 的 3.0011。衰减等价于把平均步长减半而没有时间红利
  （lin0/cos0 的等效平均 lr≈5e-5，终点与常值 5e-5×10 轮的 2.92 吻合到噪声内）。ce 从第 1 轮到第 4 轮
  **单调下降、还在斜坡上**，不是在极小值附近振荡 —— 衰减要治的病不存在。
- **1e-4 是峰值，不是"刚好不崩的边界"**：1.25e-4 终点 2.9479（不崩但慢 0.136）、1.5e-4 全程更慢且第 5 轮
  崩、2e-4 第 4 轮崩、2.5e-4 第 3 轮崩（复现时**第 2 轮**就崩）。两侧都掉，所以「最快」与「最大不崩」在这
  一点重合，不需要权衡。
- ⚠ **luspo 走零阶要先看第一步打印的 `rel`**：它的 `|pj|` 是 ce 的 7~8 倍，5e-5 在身份轨迹上实测
  `rel` 0.44~0.72（三步把 ‖θ‖ 推高 77%，而 answer-ce 只动 −0.011）。若第一步 `rel > 0.05`，用 `--lr`
  临时降到 1e-5~2e-5 —— 默认表只留两档，不为此单开一个数。反传侧不受此影响（AdamW 每步位移恒等于 lr）。
- ⚠ **知识类目标不要用 grpo/luspo**（身份题就是这一类）：严格判分下组内奖励 12/12 全 0 ⇒ 优势恒 0、
  梯度恒 0；稠密 F1 下虽有方差，但方差来自措辞重叠而非答对，且基座对答案串的 logp 是 −58（p≈e⁻⁵⁸，
  目标在策略支撑集外），组内归一化学到的是"多输出参考串里的 token"。`mezo._check_luspo_premises()` 会在
  跑前把序列长度中位、组内奖励全同组数、奖励取值个数打出来并告警。知识类目标走 sft/dpo。
- **这张表是"两条线都在"的树（bwd / support）上的六个组合**。零阶分支（`feat/train-mezo`）只带
  左半边：`--method` 的可用项由文件决定（`algos.available()`），缺 `backward/sft.py` 时 `--list`
  会注明"不在这条分支上"，`--method backward` 在加载模型前退出并指路 —— 见 §7.1。
- 更新那一半做了统一，但**有边界**：反传线（`sft.py`）与已退役的 PyTorch 零阶实现（`hf_ref.py`）
  共用 `update_rules.py` 的 `BackwardRule` / `MezoRule`（同一 `step(loss_fn)`）。
  **引擎零阶 `mezo.py` 不能复用** —— 它的"前向"是跨进程的 `SCORE`、更新对象是 adapter 表而不是
  torch 参数，所以自己实现 ZO 循环（见 D4）。两处步长口径要人工保持一致，是 D4 之后新增的审核点。
- DPO 的偏好对逻辑单独在 `dpo.py`（无 torch），反传与零阶共用。
- 选不到的组合（例如将来新增 algo 却只加了一种方法）走 `method_of()` 抛 `SystemExit`，
  **不静默回落**。
- 旧参数名 `--objective` / `--backend` 显式报错并指向新名（`train.py:479`）。

### 2.2 命令面

`train.py` 是唯一入口：`--algo` `--method` `--list`（打注册表）`--dry-run`（只打印最终 argv，不加载模型）。
`train.py` 负责把统一参数翻译成各脚本的 argv（`build_argv` / `build_argv_engine`），
模型/pack/tokenizer 的解析与 WSL 路径转换也在这里。

> 一处容易误读：`--dry-run` 打印的子命令里会出现 `--objective ce` —— 那是**子脚本自己的**参数，
> `train.py` 由 `--algo` 推出；在 `train.py` 上写 `--objective` 仍然报错（两回事）。

**"服务化"到什么程度（这一条容易被高估）**：`/v1/training/*` 是**原语级**的 ——
open 会话 → 逐行喂命令 → close，不是一个"提交训练任务、轮询进度"的高层 API。
实际驱动仍是一个客户端进程（`train.py`，或零阶的 `mezo.py --engine-api` / `--serve-url`），
服务端只负责让位、转发、回收。**这是有意的**（D7）：训练引擎的 argv 是实验变量
（`--no-prefill-borrow`、prompt chunk、lr、steps），服务侧不该替调用方决定。
换句话说：本 PR 把训练**接进了服务**（仲裁 GPU、开关会话），但**没有**把它做成在线训练平台。

### 2.3 服务端训练会话

| 接口 | 行为 |
|---|---|
| `POST /v1/training/session {"op":"open","argv":[...]}` | 卸载常驻推理引擎 → 用调用方 argv 拉起训练引擎 → 等 READY（上限 300 s）；已有会话 → 409 |
| `POST /v1/training/command {"line":"..."}` | 转发到训练引擎 stdin，按前缀收回复；无会话 → 409 |
| `POST /v1/training/session {"op":"close"}` | 关训练引擎，**下一个请求自动把服务引擎拉回来** |

- 会话期所有推理路由 409；**无会话时推理路径完全不变**（`training_active()` 一个布尔判断）。
- 训练引擎死亡 / 启动失败 → 会话自动关闭并回到未激活态（单测覆盖 4 种死亡路径）。
- 为什么"让位"而不是"双驻留"：48 GB 放不下引擎 + HF 训练栈同时驻留（见 §1.3）。

#### 2.3.1 三条 transport：训练在哪张卡上跑

`mezo.py` 是零阶的训练循环，它可以接三种"前向来源"。**训练循环本身只有一份**，
三者的差别只在"谁持有 GPU 模型"和"推理是否中断"：

| transport | 参数 | 谁持有 GPU | 推理 |
|---|---|---|---|
| `Engine` | （缺省） | 训练进程自己 spawn 一个引擎子进程，pipe 通信 | 不存在（无服务） |
| `EngineApi` | `--engine-api <url>` | 走 `/v1/training/*`，服务端先 unload | **中断**：会话期一律 409 |
| `EngineHttp` | `--serve-url <url>` | 驻留的推理引擎本身，走 `POST /v1/score` + `POST /v1/lora` | **继续**，只与训练步共享同一把 fifo |

`EngineHttp` 是"在线"那条：`/v1/score` 的 `"lora": +1 | -1` 就是一次 MeZO 步的两半
（该端点 docstring 原文即"两次只有这个键不同的调用就是一次 MeZO 步"），`/v1/lora` 做原地换表。
**训练数值与离线路径完全一致**：`swap()` 仍是 `write_adapter()` + 换表，写出的 plus/minus
表与离线路径同一批字节，同一份 `run()` / `objectives.py` / θ。累积更新用已有的 `--out` npz + `--resume`。

两个不写清就会静默出错的契约：

- **score 长度是 n−1**：引擎对 `t + 1 < n` 才发 `SL` 行（n−1 行），而 `/v1/score` 为与 `tokens`
  对齐补了一个末位 `null`（n 行）。客户端照抄服务端数组，answer 段切片会整体**错一位** ——
  损失照样算得出、曲线照样降，只是学错了跨度。`EngineHttp` 去掉末位，`test_engine_http.py` 钉住长度。
- **HTTP 上没有 generate-from-ids**：所以 `supports_gen = False`，评测跳过生成那一行，
  不会在跑了一半时抛异常。

代价两条：服务端必须以 `--no-prefill-borrow --prefill 8192` 启动（否则 score 走借槽路径，
数值已知会坏），这会影响推理的 prefill 速度，**幅度未测**；每次换表清空全部 K/V，
所以在跑的对话下一个请求要从 token 0 重读 prompt —— "无感"指推理不停，不是对话不被打断。

### 2.4 显存与 GPU 独占

- `mem_guard.py` 是显存判据唯一出处：6 个实测锚点 26.2 / 27.7 / 29.8 / 33.0 / 35.3 / 47.5 GiB
  @ 2K/8K/16K/27.6K/32K/48K，区间线性、末锚点外 ×1.10。
- `require_exclusive_gpu()`：拉起训练引擎前查空闲显存，不够就 loud exit 2；
  `STRATA_ALLOW_SHARED_GPU=1` 可显式豁免（豁免本身也出声）。
- **换一张卡能训多长、多快、什么情况直接训不了**：见 §5.2（含 24 / 32 / 48 / 80 / 96 / 141 / 192 GiB
  七档）。这里不重复贴表。

### 2.5 两条路线分别要装什么（审核「新增依赖」时看这张）

| 项 | 零阶（`--method mezo`） | 反传（`--method backward`） |
|---|---|---|
| **新增 pip 包** | **0 个**。只用 `requirements.txt` 已装的 `numpy` / `jinja2`（chat 模板）/ `regex`（`tools/strata_tokenizer.py` 依赖），其余全是标准库；`urllib` 而非 `requests`，`objectives.py` 对 torch 是 `try: import` | `torch`（CUDA 版）、`transformers`（`loader.py` 要 `transformers.models.qwen3_5_moe`；`grpo.py` 要 `AutoTokenizer`）、`accelerate`（`loader.py` 的 `init_empty_weights`）、`safetensors`（`loader.py` 的 `safe_open`） |
| Windows 额外 | 无 | `triton-windows 3.7.1`（PyPI `triton` 只有 Linux 轮子）、`kernels 0.16.1` —— README 记录的实测组合：`torch 2.13.0+cu132` / `transformers 5.17.0` |
| 125B NF4 pack 路线额外 | 不涉及（专家权重由引擎自己管，训练进程不装 pack） | `gguf`（`loader_qwen4.py` 的 `gguf.constants` / `gguf.quants` 反量化） |
| **引擎二进制** | **必需**：本仓库构建的 `strata`（CMake + CUDA toolkit，`-DSTRATA_ENABLE_CUDA=ON`；`third_party/llama.cpp` 已 pin）。只用到 `--serve` 命令循环里的 `LORA <path>` / `SCORE <ids>`（评测再要 `GEN`） | `sft` / `dpo` 反传**不需要引擎**（纯 PyTorch 前向）；`grpo` 反传是两段式，第一段要引擎采轨迹（同样要 `LORA`/`SCORE`） |
| **引擎分支要求** | ⚠️ **必须是 `feat/train-mezo` 构建出来的**：`LORA`/`SCORE` 只在这个分支的 `src/program/generate.cpp` 里。`upstream/main`（`146df3c4`，2026-11-07 同步时复核过）的 `src/` 里**没有**这两条命令 | 同左（GRPO 第一段） |
| 数据 / 资产 | pack：gguf shards + `tokenizer/chat_template.jinja` + `expert-profile.bin` | 同左（另加 NF4 专家 pack） |
| 平台 | linux / win32（`platform_check.SUPPORTED`；win32 走 `preadv_shim` 的 readinto 垫层）；macOS 未测、被拒 | 同左 |
| 实测量级显存 | 26.9 GiB 账面（§5.1） | 34~45 GiB（§5.1） |

一句话：**零阶是"装好的环境 + 一个自己编译的引擎"就能跑，反传是"另装一套 PyTorch 栈"才能跑。**
两者不共享 Python 依赖，这正是本 PR 把它们交付成两条**可整层拆开的分支**、而不是一个
"要么全装要么全不能跑"的包的理由（D1）。命令面上它们仍然是一个开关（`--method`）。

> 引擎怎么来：`STRATA_ENABLE_CUDA` 默认是 **OFF**（`CMakeLists.txt:80`），必须显式打开，
> 写成 `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
> -DCMAKE_CUDA_ARCHITECTURES=<你的卡> -DSTRATA_GGML_DIR=third_party/llama.cpp` 再加 `cmake --build`。
> 架构号与后端开关照 `docs/`（如 `docs/DEEPSEEK41.md`、`docs/AMD_HIP.md`）里同类模型的写法；
> 训练只需这个二进制，不需要额外编译目标。

> 说明：`requirements.txt` 里没有 torch —— 这是刻意的，CI 的 no-torch 档（`test_train_core.py`）
> 用只有标准库+`requirements.txt` 的环境当合并门。反传档单独装 torch 才跑（`.github/workflows/training-test.yml`）。
>
> ⚠️ **合并顺序是有依赖的**：`training` 只带 Python；零阶要的 `LORA`/`SCORE` 命令循环
> 目前只在 `feat/train-mezo` 的 `src/program/generate.cpp` 里，`upstream/main` 没有。
> 所以"先合训练脚本、后合引擎"会让零阶在这个仓库里跑不起来（`mezo.py` 会等到引擎的 `LORAOK` 超时）。
> README 里"engine side, all merged upstream"这句与上表不符，需要按实际情况改（见 Q9）。

### 2.6 零阶（MeZO）路线的出处

- **方法**：MeZO — *Fine-Tuning Language Models with Just Forward Passes*
  （Malladi, Gao, Nichani, Damian, Lee, Chen, Arora；NeurIPS 2023；arXiv **2305.17333**）。
- **官方仓库**：<https://github.com/princeton-nlp/MeZO>（MIT）。代码在 `large_models/`（OPT 自回归）
  与 `medium_models/`（RoBERTa，含更多变体）。
- **我们借了什么**：算法本身 —— 对称差分方向导数 `(L(θ+εz) − L(θ−εz)) / 2ε`、方向由种子再生而不存储、
  参数原地扰动；`--momentum` / `--batch-rows` 对应原论文的 momentum 与 ZO-SGD 小批量，
  `--dirs`（多方向平均）是压方向方差的额外杠杆。
- **我们没借什么**：**没有一行代码来自该仓库**。它是 HF `Trainer` 的 PyTorch 插件（扰动参数张量、`torch.no_grad` 前向），
  我们的前向是引擎的 `SCORE`，参数只存在于 LoRA 表里（`mezo.py` 默认 `--rank 8`、`--alpha 16`，
  目标矩阵见 `lora.py::DEFAULT_TARGETS`）。所以它不出现在任何依赖清单里，
  也不受其 MIT 许可约束 —— 借用的是已发表的方法，不是代码。
- **本树的上游**：`upstream` = `git@github.com:Niko1221/Strata.git`；个人 fork `origin` = `git@github.com:xiezhipeng-git/Strata.git`。
  `LORA` / `SCORE` 与 `--serve` 命令循环**尚未进 `upstream/main`**，只在 `feat/train-mezo`
  两棵树上（见 §2.5 的引擎分支要求）：`upstream/main` = `146df3c4`（v0.2.12，2026-11-07 merge 进三条分支），
  merge 时复核过 `git show upstream/main:src/program/generate.cpp` 里仍然没有任何 `LORAOK`/`SSUM`。

### 2.7 产物

LoRA adapter 存取（`lora.py`）→ `lora_to_gguf.py` 转 GGUF 供引擎 `LORA` 命令加载（`LORA-MERGE.md`）。

---

## 3. 改动面：文件 → 职责 → 请重点看什么

**目录结构**（`training/` 下按两条路线分，审的时候按这个找文件）：

```
training/
  train.py  algos.py  objectives.py  mem_guard.py  update_rules.py    # 入口 + 两条路线共用
  ce.py  dpo.py  datasets.py  guard.py  whoami.py  platform_check.py  # 共用件
  mezo/      零阶：mezo.py（引擎当前向，--method mezo 的唯一实现）
  backward/  反传：sft.py（含 DPO / curriculum）、grpo.py、HF 加载栈（loader_qwen4 / nf4_fused / qsa_fast / lora …）
             + hf_ref.py（torch 侧参考前向，仅做 score 对拍，不是方法实现）
  bench/     基准与一次性探针：bench_fullctx、ab_*、mtp_*、budget、judge_limits …
  tests/     18 个单测（362 项；零阶树只带其中 9 个 = 214 项，11-07 口径）
```

> 这些是**功能代码，不是示例**。原稿放在 `examples/training/` 名下不副实 —— 上游 `main` 根本没有 `examples/`，
> 仓库放 Python 工具的是 `tools/`、放服务的是 `serve/`，所以新建顶层 `training/`，两条路线各占一个子目录。

### 3.1 核心实现（新增）

| 文件 | 行 | 职责 | 审核重点 |
|---|---|---|---|
| `objectives.py` | 390 | sft / dpo / grpo / luspo 四个目标函数的**唯一实现**，numpy 与 torch 双路径 | 双路径是否逐值一致；`reduction="mean"` 的归一化基数（token 数还是样本数）；GRPO 的 mask 与 advantage 广播；luspo 的序列级 ratio 与 `×|y_i|` |
| `mezo/mezo.py` | 1,385 | 引擎桥（`Engine` 子进程 / `EngineApi` / `EngineHttp`）+ `EngineMezo` 基类 + 四个子类 | θ±ε 的落地方式（写 plus/minus adapter 再 `LORA` 换表，D4）；θ 表的形状必须全程不变；θ₀ 缓存与 `--resume` 的一致性 |
| `mem_guard.py` | 447 | 显存预算、锚点插值、`max_seq_len`、独占检查 | 估算式是否会低估（解析式在 >40K 低估 4~6 GiB，是本模块存在的理由） |
| `algos.py` | 220 | 注册表 + lr 分表 + 旧参数名报错 | lr 硬编码在表里是否该改成可覆盖配置（Q4） |
| `train.py` | 575 | 唯一入口、参数翻译、模型解析 | `--dry-run` 打印的 argv 与真实执行是否同源 |
| `backward/sft.py` / `backward/grpo.py` | — | 反传路线（PyTorch/HF autograd；DPO 也走 `sft.py`） | 与 `objectives.py` / `update_rules.py` 是否真共用同一函数而不是复制；`--ckpt` 重算的边界 |
| `update_rules.py` | 198 | `BackwardRule`（autograd + AdamW）/ `MezoRule`（θ±ε + SGD）/ `zo_direction()`，同一 `step(loss_fn)`。**使用者是 `sft.py` 与已退役的 `hf_ref.py`；引擎零阶 `mezo.py` 不走它** | 两规则的 rel/动量语义与 `mezo.py` 内联的 ZO 循环是否仍一致；`zo_direction` 的"每个参数方向必须不同" |
| `dpo.py` | — | DPO 纯逻辑（偏好对校验、`dpo_margin`、batch 组装），无 torch，被 `sft.py`（反传）与 `hf_ref.py`（torch 侧参考）共用 | 这里与 `objectives.dpo_loss` 的分工是否清晰，会不会又长出第二份 margin |
| `bench/budget.py` / `guard.py` / `bench/usage.py` | 240 / 135 / — | 预算计算（给定机器反推各方案能训多长）、训练护栏（防"10 条同构样本训坏无关能力"）、真实请求尺寸统计；前两个是独立脚本，`guard` 被 `sft.py` 引入 | `budget.py` 与 `mem_guard.py` 的判据是否重复（一个反推上限、一个判当前是否超） |

### 3.2 服务端（修改）

| 文件 | 行 | 审核重点 |
|---|---|---|
| `serve/server.py` +640 −36（全文；含训练会话、`/v1/score`、`/v1/lora` 与推理闸门） | `TrainBridge`（子进程 + READY 握手 + stdin/stdout 转发）、`training_session()`、`training_command()`、推理路由的 409 闸门 | 锁与状态机（`op=open` 与 `close` 并发）；引擎死亡时锁是否释放；unload 失败是否把服务留在半死状态 |
| `serve/test_training_api.py` +6 项 | 用假引擎覆盖 open/command/close、坏 argv、无会话、引擎中途死亡、启动失败、unload 与恢复 | 假引擎是否可能掩盖真实的 READY 时序问题 |

### 3.3 测试与文档

- `test_objectives.py`（numpy↔torch 双路径对拍，仅在反传树）、`test_objectives_np.py`（零阶树上的
  numpy 路径 18 项）、`test_mem_guard.py`、`test_update_rules.py`、`test_bench_limits.py`、
  `test_engine_api.py`、`test_engine_http.py`（在线 transport 的 10 项契约，含「绝不 spawn 引擎」）、
  `test_judge_limits.py`、`test_prep_eval_data.py`、`test_run_plan.py`、`test_curriculum.py` 为新增；
  `test_train.py` / `test_train_core.py` 大幅扩充。
- 文档：`README.md`（命令面与参数）、`DEMO-IDENTITY.md`（身份训练手顺）、
  `PROTOCOL-FULLCTX-EVAL.md`（满上下文评测协议）、`BRANCH-NOTES.zh.md`（工作台账，非提交材料）、
  `LORA-MERGE.md`（adapter → GGUF）、本文件。

### 3.4 提交卫生（已处理，这一节留结论供复核）

分支拓扑见 §7.1。`feat/train_support` 现在等于 `backforward` 的 fast-forward，再 merge 了
`perf/pack-load-accel`；两条 feature 分支都是它的祖先，所以
`upstream/main...feat/train_support` 的 diff 里只有本功能，没有上游噪声。

**不该进 PR 的东西已经挡在 `.gitignore` 外面**：`training/_*`（一次性探针与
中间 json）、`*.pt` / `*.gguf`（本地权重，26 MB + 39 MB）、`*.bak`。`git status` 干净。

**commit 分组**（按可审单元切，不按文件时间）：

| 组 | 提交 | 审什么 |
|---|---|---|
| 1 | `chore(training)` | `.gitignore`：哪些是运行产物 |
| 2 | `[Train] train.py --algo x --method` | 唯一入口、注册表、旧参数报错、无 torch 门 |
| 3 | `[Train] objectives.py` | 三个目标只有一份数学，numpy↔torch 对拍 |
| 4 | `[Train] mem_guard + budget` | 显存前置判定（不是错误处理） |
| 5 | `[Train] update_rules.py` | 步长与 `rel` 判据 |
| 6 | `[Train] mezo.py` | 零阶：θ±ε 写 adapter → `LORA` 换表 → `SCORE` |
| 7 | `[Train] platform_check + preadv_shim` | Windows 走 readinto 垫层，不静默降级 |
| 8 | `[Train] the pieces the entry point reads` | `dpo.py` / `guard.py` / `run_plan.py` / 评测探针集 |
| 9 | `[Serve] /v1/training/*` | 让位会话：start / status / stop |
| 10 | `[Docs]` + `[Tools]` + `[Data]` | 说明、`bench_fullctx.py`、输入与实测 json |
| 11 | 反传线 5 组（`backforward`） | `sft.py` / `grpo.py` / 加载与 kernel / `hf_ref.py` 退役 / 探针 |

反传那 5 组在 `feat/train-backforward` 上，其余在 `feat/train-mezo` 上（共用层 + 零阶 +
引擎），`feat/train_support` 是两者的合并面。逐组看时用
`git log --oneline --no-merges upstream/main..feat/train_support`。

## 4. 设计决策与理由

8 条关键取舍。每条给「选择 / 被否的替代 / 依据 / 风险」；若某条判断不成立，
可用的替代方案已经在"被否"一列里写明，不需要重新设计。

**D1 零阶放引擎、反传放 HF —— 而不是引擎侧也实现反传**
- 依据：引擎是推理栈，128K 上零阶两次前向 63.8 s；反传要激活 + 每层 T² mask，同卡只到 9,728。
- 风险：两条路线**数值不可能逐值相同**（CE 差 0.0075 nats，见 §5.5），评审要接受"同目标、不同算子路径"。

**D2 默认 `--method mezo`**
- 依据：显存与上限、单步速度、held-out 回退三项都占优（§5.1 / §5.2 / §5.4）。
  **不主张"零阶收敛更快"** —— 实测只在"浅档"成立：到基线 20% 快 2.8×、10% 快 3.1×，
  再深零阶平台化、反传还能降（§5.2 表 A2）。
- 风险：新人直觉是"反传更准"，默认零阶需要文档解释，否则会被当成降级。是否该强制显式传 `--method`？

**D3 损失单一出处，`objectives.py` 内 numpy / torch 双路径**
- 依据：引擎零阶只拿到 logp 的 numpy 数组，反传要 torch 张量走 autograd，两条路都要。
- 替代（被否）：只写 torch，零阶侧转 numpy —— 会让零阶被迫装 torch，失去"引擎侧纯 numpy"的独立性。
- 风险：双路径天然有漂移风险，靠 `test_objectives.py` 逐值对拍兜。**待确认**：当前容差是否足够严。

**D4 θ±ε 用「写两份 adapter + `LORA` 换表」实现，不给引擎加零阶专用命令**
- 做法：每步把 θ+ε·z、θ−ε·z 各写成一个 adapter GGUF（`--work` 下的 `plus.gguf` / `minus.gguf`），
  再 `LORA <path>` 换表、`SCORE <ids>` 打分（`mezo.py` 的 `swap()`）。方向 z 由 seed 再生，不存盘。
- 依据：引擎侧因此只需要 `LORA` / `SCORE` 两个命令（目前只在 `feat/train-mezo`，
  §2.5），零阶路线不需要再给引擎加任何零阶专用命令；`SCORE` 的 `+`/`-` 前缀翻转符号那条读法是留给
  "固定 θ、只翻扰动"的场合，我们是每步换表，用不上。
- 风险：每步两次 GGUF 写盘 + 两次换表（在 61~101 s 一步里占比小，但 `--work` 必须可写）；
  把 rank 调大时写盘时间会上升。**未测** write 与 SCORE 的时间占比。

**D5 DPO 的 margin 按答案 token 归一**
- 依据：序列级 margin 在 122K 对上把 lr 1e-5 放大成等效 3 个数量级 ⇒ ‖θ‖ 19.6 → 10175 发散；
  归一后是长度不变量，一个默认 3e-5 通吃（短对 rel ≤4.7e-3、122K 对 ≤5.5e-4）。
- 风险：与"标准 DPO 实现"不一致，必须在文档里显式声明，否则对照论文的人会认为是 bug。

**D6 显存判据用锚点插值，不用解析式**
- 依据：二次式在 >40K 低估 4~6 GiB，会把必爆判成能过。锚点全部实测。
- 风险：锚点外（>48K）只能 ×1.10 外推，`mezo_anchor_hit()` 用来标记"这里不是实测点"。

**D7 训练会话"让位"而非"双驻留"**
- 依据：48 GB 装不下两份权重（§1.3）；且训练引擎的 argv（`--no-prefill-borrow`、prompt chunk）
  属于调用方的 recipe，服务侧不该替它决定。
- 风险：open 期间服务无推理能力（409 而非排队）。若需要"训练时仍可推理"，是另一个设计。

**D8 组合不存在就报错，不做静默回退**
- 依据：静默回落到 backward 会让人以为在跑零阶（显存与时间差一个量级）。
- 风险：新增算法时要同时改注册表与方法脚本，否则新用户第一次运行就报错 —— 这是有意的。

---

## 5. 证据：为什么认为这套实现达标

### 5.1 六方案 × 48 GiB 卡的上下文极限与显存（核心表）

| # | 方案 | 路线 | 极限 T | 该点显存 | 该点步时 |
|---|---|---|---|---|---|
| 1 | sft × mezo | 引擎前向（SCORE，numpy） | **131072**（模型上下文封顶） | 26.9 GiB 账面 / 46.97 GiB nvidia-smi | 61.4 s @121,800 tok |
| 2 | dpo × mezo | 引擎前向 | **131072** | 同上 | 101.0 s @121,720 对 |
| 3 | grpo × mezo | 引擎前向 | **131072** | 同上 | 101.4 s（2 条轨迹） |
| 4 | sft × backward | PyTorch autograd | **9728**（16384 挂死） | 44.95 GiB | 914.7 s |
| 5 | dpo × backward | PyTorch autograd | **5120**（8192 崩） | 41.46 GiB | 53.9 s |
| 6 | grpo × backward | PyTorch autograd | **2560**（K=4） | 34.18 GiB | 28.4 s |

判据三条：①一步跑完（无 kill / 无挂死）②峰值 ≤ 48 − 1.5 GiB ③更新信号 ≥ 3× 噪声极差
（128K 实测噪声 0.169 nats ⇒ 阈值 0.5）—— 第三条是防"跑得动 ≠ 训得动"。

**两点读法（关系到你怎么判"达标"）**：

- 零阶的极限**不是显存**：8K 与 128K 的 nvidia-smi 峰值都是 46.96~46.97 GiB，因为
  `--expert-cache auto` 会吃满空闲显存。真实需求 26.9 GiB = 24 权重 + int8 KV 1.68 + 工作区 1.2。
  剩下 20 GiB 是缓存，**不是需求**。所以"零阶 128K"是有余量的结论，不是贴顶侥幸。
- 反传的极限**就是显存**：9728 已 44.95 GiB，16384 是"不报错、不退出"的挂死
  （`bench_fullctx.py` 为此加了看门狗）。**bwd 的步时列只作量级参考**：
  4608→9728 步时涨 11.2×，T² 只能解释 4.5×，根因未定位（见 §8）。

这张表只回答"上限在哪"。**"你的机器算充足还是不足、差多少速度、什么情况完全训不了"，见 §5.2。**

### 5.2 资源充足 / 不足 / 不可训练：三档判据与速度（单卡口径）

先给判据。**C 档是代码判的**（`mem_guard.require_budget()` → `_limit_gib()`：零阶放 0.99、
反传留 5% 即 0.95 × 总显存）；**A/B 两档是本文的叙述分法**，用来把"能跑"和"跑得能用"分开，
没有第二套阈值进代码：

| 档 | 判据 | 代码行为 |
|---|---|---|
| **A 充足** | 峰值 ≤ 0.7 × VRAM，还有余量去换更大 batch、关 ckpt | 正常跑，且允许提速档 |
| **B 不足（能跑但慢）** | 0.9 × VRAM ≲ 峰值 ≤ 预算，只能开 ckpt、不能换 batch | 正常跑，但吞吐塌陷 |
| **C 不可训练** | 权重装不下 / 预算拒绝 / 硬 OOM / 挂死 | 前两类秒退，后两类分别浪费一次前向和整段等待 |

**表 A：同一张 48 GiB 卡上，六个方案落在哪一档（实测）**

| # | 方案 | A 充足区（实测） | B 贴顶区（实测） | C 不可训练起点 | 档间落差 |
|---|---|---|---|---|---|
| 1 | sft × mezo | 8K：**4.39 s/步**，1807 tok/s，峰值 26.9 GiB（预算 45.6，只用掉 59%） | **没有这一档** —— 128K 也只有 26.9 GiB，63.8 s/步，2050 tok/s | > 131072（模型上限，不是显存） | 极差 1.13×（固定开销被摊薄） |
| 2 | dpo × mezo | 128K：101.0 s/步，≈2 400 tok/s（2 行折算） | 没有这一档 | > 131072（同上） | — |
| 3 | grpo × mezo | 128K：101.4 s/步（2 条轨迹），≈2 400 tok/s | 没有这一档 | > 131072（同上） | — |
| 4 | sft × backward | 4 608：**81.9 s/步**，56.3 tok/s，35.0 GiB | **9 728：914.7 s/步，10.6 tok/s**，44.95 / 45.6 GiB | 16 384 **挂死**（不报错、不退出） | **5.3× 吞吐塌陷**（步时长 11.2× 而 token 只 2.1×） |
| 5 | dpo × backward | 2 560：49.8 s/步，102.7 tok/s，34.2 GiB | 5 120：53.9 s/步，190.1 tok/s，41.46 GiB | 8 192 **崩** | 1.9×（反向变"快"，小形状开销占比下降） |
| 6 | grpo × backward | 1 024：26.1 s/步，78.4 tok/s，29.6 GiB；2 560（rows=2）：28.4 s/步，180.2 tok/s，34.18 GiB | **未实测**：K=4 的 2 560 只有预算值 44.6 GiB | > 2 560（K=4）**预算拒绝** | — |

读法：**零阶根本没有"不足档"**（第 1~3 行 B 列空着）。这不是运气 —— 引擎只驻留 24 GiB 权重 +
固定槽位 expert cache，KV 在 128K 也只要 1.68 GiB。**反传的"不足档"是真实的、且就在 48 GiB 上**
（第 4 行），一步 15 分钟，吞吐掉到 10.6 tok/s。

> 吞吐口径统一按**步级**算（一步的墙钟 ÷ 该步覆盖的 token），因为对照的是"训练一步要多久"。
> 零阶一步含 θ+ε、θ−ε 两次前向，所以步级是 2050 tok/s；`bench_engine_128k.json` 里的 4313 tok/s
> 是**单次前向**的数（`score_tok_s`），不是步级 —— 两个都对，别混着比。

**为什么零阶一步必须两次前向**：这是方法本身要求的，不是实现冗余。MeZO 用**对称差分**
估计方向导数 `(L(θ+εz) − L(θ−εz)) / 2ε`（`mezo.py` 的 `th_plus` / `th_minus` → 两次 `SCORE`），
两次前向的差把与 z 无关的项抵掉，只剩沿 z 的一阶项。单侧差分 `(L(θ+εz) − L(θ))/ε` 也要两次
（要 L(θ)），而 `L(θ+εz) − L(θ−εz)` 的好处是不含零阶项、偏差更小；论文用的就是这一种。
所以 63.8 s @128K ≈ 两次约 31 s 的前向之和（单次 4 313 tok/s × 131072 ≈ 30.4 s）。

**「零阶一步比反传快一个量级」成立，但不能推成「整体训练快一个量级」** —— 这是两笔账：

| 读法 | 数 | 是否可比 |
|---|---|---|
| 单步墙钟（8K 对 4 608） | 4.39 s vs 81.9 s = **18.7×** | 可比，同为一步 |
| 步级吞吐（tok/s） | 1 807 vs 56.3 = **32×** | 可比，同为一步 |
| 每百万 token 的墙钟 | 9.2 min vs **4.9 h** = **32×** | 可比，口径已归一 |
| **收敛到同等效果所需总时间** | **浅档（≤20%）零阶快 2.8×；10% 档 3.1×；≤5% 零阶到不了** | ✅ 2026-11-03 三档 lr 实测，见下方表 A2 |

前三个数只回答"喂同样的 token，谁快"。零阶每步更新的是 rank-8 的 `3.34 M` 参数、
单方向信噪比 ~1/√d（§5.3），因此**通常需要更多步**才能达到反传同样的效果。第四个数现在是实测的
（三档 lr 扫描），但它**只在一个任务上成立**，读法见下。

**表 A2：到同等效果的总时间（2026-11-03 实测，单文档、48 GiB）**

协议：同一份 ~4.2K token 文档（8 条自造事实循环 89 遍，基线 CE 有真实下降空间；自然长文的基线只有
0.04~0.15 nats，天花板太低量不出收敛），同一台机器，都从 B=0 adapter 起、每步 1 条样本；
零阶 `--method mezo`（引擎 `SCORE`，rank 8，lr 见 A2-2），反传 `--method backward`（lr 1e-4，rank 16，
`--ckpt`，`--select-tokens off`）。效果 = 答案段 mean CE；两侧 tokenization 不同，所以只看**相对基线的百分比**。

**A2-1 主表**（零阶取 lr 1e-5 —— 三档里唯一"既稳、终点又最低"的档）：

| 指标 | 零阶（mezo, lr 1e-5） | 反传（backward） |
|---|---|---|
| 基线 CE / 终点 CE | 1.782 → **0.158**（8.9%，平台） | 0.373 → **0.0119**（3.2%，仍在降） |
| 步时 | **3.1 s**（300 步 921 s，含每 10 步评估） | **24.8 s**（前 14 步），之后 **58 s** |
| 降到基线 50% | 第 20 步 ≈ **62 s** | 第 2 步 ≈ **50 s**（零阶 1.2× 慢） |
| 降到基线 20% | 第 40 步 ≈ **124 s** | 第 14 步 ≈ **347 s**（零阶 **2.8×** 快） |
| 降到基线 10% | 第 60 步 ≈ **186 s** | 第 18 步 ≈ **579 s**（零阶 **3.1×** 快） |
| 降到基线 5% | **到不了**（平台 8.9%） | 第 20 步 ≈ **695 s** |
| θ 范数漂移 | 19.7 → 28.9（**1.47×**） | —（AdamW 逐参数归一，无此量） |

**A2-2 零阶的 lr 扫描**（同一任务、同一数据、只改 lr）：

| lr | θ 范数漂移 | ≤20% | ≤10% | 终点 CE | 备注 |
|---|---|---|---|---|---|
| 3e-5 | **3.0×** | 31 s（第 10 步） | 到不了 | 0.186（10.4%） | `rel` 前 5 步冲到 0.8，**不是工作点** |
| **1e-5** | 1.47× | 124 s（第 40 步） | 186 s（第 60 步） | **0.158（8.9%）** | 终点最低、漂移可控 |
| 3e-6 | 1.29× | 372 s（第 120 步） | 806 s（第 260 步） | 0.169（9.5%） | 第 10 步 CE 反升到 2.53（高于基线） |

四条读法：

1. **"一步快 18.7×"不能换算成"训练快 18.7×"。** 本机这一步比是 3.1 s vs 24.8 s（≈8×，与 18.7× 同量级，
   差异来自这里的反传开了 `--ckpt`、且显存档位与 §5.2 表 A 那次不同）。到 20% 档，零阶用 40 步、
   反传用 14 步 —— 零阶**步数只多 2.9 倍**，远没有"每步效果小一个量级"那么悬殊。
2. **"快一个量级"是 lr 撞出来的，不是方法属性。** 若只取 lr 3e-5（`rel` 前 5 步 0.8、`|θ|` 漂 3×），
   20% 档只要 31 s，看上去快 11×；同一份数据、同一台机器换到 1e-5 就是 124 s，**2.8×**；10% 档 186 s vs 579 s，
   **3.1×**。唯一变量是 lr，所以这个倍数不该由"方法"承担。
3. **浅档要分两段说。** 50% 档反传反而更快（50 s vs 62 s）—— 它每步都吃得满，前两步就过半；
   零阶的前 10 步还在 `rel` 尖峰里（最大 0.46）。零阶的优势从 20% 档才开始，因为那里比的是**步数与步时之积**。
   同理，时间几乎与 lr 成反比（31 / 124 / 372 s ≈ 1 : 4 : 12，lr 是 1 : 3 : 10），而终点 CE 并不随 lr 单调
   （0.186 / 0.158 / 0.169）—— 1e-5 附近是拐点，不是"越小越好"。
4. **深档零阶到不了。** 三档 lr 都没到基线 5%（平台在 8.9~10.4%），反传第 20 步就到 3.2% 且趋势仍在降。
   所以"总时间快 N×"在深档**无法定义**，不是"慢"而是"到不了"。

**这张表的边界**：单文档、记忆型任务（同一段重复 89 遍），**不是泛化能力**；两侧 tokenization 不同
（同一文档反传记 4220 token、引擎记 4259 token），基线 CE 0.373 vs 1.782 差 4.8×，所以**绝对 CE 不可直接比**；
而"相对基线的百分比"本身也受基线高低影响 —— 严格对拍应该用**同一个打分器**（引擎 `SCORE`）给两个 adapter 打分，
这一步没做。阈值时间是**首个越过该线的采样点**（每 10 步一次评估），真实时间只会更早、不会更晚。
反传第 15 步起 tok/s 从 170 掉到 70（步时 25→58 s）的根因未查。
⇒ 可以说"在这类任务上零阶到同等浅效果快约 3×，深效果到不了"；**不能说**"端到端普遍快 N×"。

§5.4 唯一一条端到端证据是反方向的：同样 4 道题、都学会 2 道，但反传把非训练集打掉 31 项。

**表 B：换一张卡，"能训多长"怎么变（`budget.py` 推算，不是实测）**

| 方案 | 24 GiB | 32 GiB | **48 GiB（实测）** | 80 GiB | 96 GiB | 141 GiB | 192 GiB |
|---|---|---|---|---|---|---|---|
| 引擎零阶（sft/dpo/grpo 同账） | ✗ 权重装不下 | 131072 | **131072** | 131072 | 131072 | 131072 | 131072 |
| bwd sft（ckpt） | ✗ | 2 048 | **9 728** | 28 160 | 37 888 | 66 560 | 100 864 |
| bwd dpo（ckpt, rows=2） | ✗ | 1 024 | **5 120** | 14 592 | 19 456 | 34 304 | 51 968 |
| bwd grpo（ckpt, K=4） | ✗ | 512 | **2 560** | 7 424 | 9 984 | 17 664 | 26 624 |
| bwd 无 ckpt（refetch 提速档） | ✗ | ✗ 权重 | **0 —— 完全不可训练** | 12 544 | 21 504 | 48 896 | 82 176 |

推算依据：`mem_guard.estimate_peak_gib` 的 6 个实测锚点（26.2 / 27.7 / 29.8 / 33.0 / 35.3 / 47.5 GiB，
残差 ≤ 0.6 GiB）+ 引擎 KV 两点标定（8K fp16 0.21 GiB、128K int8 1.68 GiB）；24 / 80 / 96 / 141 / 192 GiB
四列是 `python bench/budget.py --vram <N>` 的输出，**不是在这几张卡上跑过**。

> 一处口径差异要说清：grpo 的 48 GiB 实测点是 **rows=2**（5 120 token，峰值 34.18 GiB），
> 而表里的 **2 560 上限按 K=4 行算**（该形状估算 44.6 GiB）。两者不是同一个形状，所以"实测 34.18"
> 与"上限 2 560 贴顶"看起来不对齐 —— 实测点在 K=2 形状下还没到顶，K=4 的 2 560 只有预算，没有实测。

三条读法，都关系到"资源够不够"这个判断：

1. **资源充足对零阶不加速，只抬上限，而那个上限已被模型封住。** 8K → 128K 吞吐 1807 → 2050 tok/s
   （不降反升）。所以在这台 48 GiB 机器上，零阶**本来就处于"资源充足"档**；换 141 GiB 卡得到的
   收益是 0（区间早被 131072 封顶）。反过来，已退役的 torch 零阶（`hf_ref.py`）**同一张卡同一方法**
   27 648 封顶、55 808 直接 OOM —— **"资源够不够"不是方法属性，是实现属性**，这是 D1 选引擎路线的一条实证支撑。
2. **资源不足对反传不是"慢一点"，是塌陷。** 9 728 处 44.95 / 45.6 GiB 贴顶，吞吐 10.6 tok/s，
   是 4 608 档（56.3）的 1/5.3。折算每百万训练 token：零阶 128K **8.1 min**、零阶 8K 9.2 min、
   反传 4 608 **4.9 h**、反传 9 728 **26.2 h**。⇒ 反传在 48 GiB 上只能当"可行性验证"，不能当生产训练；
   这也是 D2 默认零阶的显存理由。
3. **"关 ckpt 换速度"这条路在 48 GiB 上被显存堵死。** 无 ckpt 的激活项 ×4.0（单点标定，
   `NO_CKPT_FACTOR`），48 GiB 上限只剩 2 048（表 B 末行是 0：走 refetch 时还要 26 GiB 专家 staging，
   `BASE 25.2 + 26.0 > 45.6` 预算）。**资源不足时，反传只有"开 ckpt 慢跑"一种选择** ——
   要到 80 GiB 才能重新用上提速档（12 544）。

**表 C：折算墙钟（用实测步时折算，不是端到端实测）**

| 场景 | 实测步时 | 训练吞吐 | 1 000 步折算 | 每百万训练 token |
|---|---|---|---|---|
| 零阶，真机身份跑（ctx 4096，900 步 588 s） | 0.65 s | — | **10.9 min** | — |
| 零阶 8K（sft） | 4.39 s | 1 807 tok/s | 1.2 h | 9.2 min |
| 零阶 128K（sft） | 63.8 s | 2 050 tok/s | **17.7 h** | **8.1 min** |
| 零阶 128K（dpo / grpo） | 101.0 / 101.4 s | ≈2 400 tok/s | 28.1 h | 6.9 min |
| 反传 4 608（sft） | 81.9 s | 56.3 tok/s | 22.7 h | 4.9 h |
| 反传 9 728（sft） | **914.7 s** | **10.6 tok/s** | **254 h（10.6 天）** | **26.2 h** |
| 反传 5 120（dpo） | 53.9 s | 190.1 tok/s | 15.0 h | 1.5 h |
| 反传 2 560（grpo） | 28.4 s | 180.2 tok/s | 7.9 h | 1.5 h |

"1 000 步"是为了给量级，不是推荐步数；同一列内可比，跨列只能看数量级。
**9 728 那一行（914.7 s / 10.6 tok/s）只作上界读**：4608→9728 的超线性根因未定位（§8 第 6 条），
如果根因是贴顶换页，换大卡后这一行会明显变好，改善幅度未测。

#### 5.2.1 "不可训练"的四种区别（症状完全不同，处置也不同）

| 类型 | 例子（本机实测） | 症状 | 处置 |
|---|---|---|---|
| **权重装不下** | 24 GiB 卡：引擎需 ≤ 19.9 GiB 驻留，实际 24.0 | 启动即退，报"weights do not fit"，**不碰 CUDA** | 换 ≥ 32 GiB 卡（32 GiB 预算下引擎门槛 27.5 ≥ 24.0，可装；此档仍未实机验证），或换小一档 pack；**调 T 无用** |
| **预算拒绝** | dpo×mezo 55 808、dpo×bwd 8 192、grpo×bwd > 2 560 | **秒级退出**，打印"本机动态上限 T≈…" | 降到上限内；这是设计（低报不如拦，见 `mem_guard` 注释） |
| **硬 OOM** | 旧 torch 零阶 55 808、grpo×mezo 44 032 | 跑到一半 `CUDA out of memory`，**已耗时** | 只能靠预估提前避免 |
| **挂死** | bwd 16 384 | **不报错、不退出**，进程挂着 | 最危险：会让人以为"在训练"。`bench_fullctx.py` 为此加了看门狗 |

这四类的区别是审核时最该确认的一条：**前两类可预期、可提前拒绝；第三类浪费一次前向；第四类会静默
吃掉整段时间**。本 PR 的 `require_budget` 只覆盖前两类，第四类目前靠 `--per-cell-timeout`/看门狗兜。

> 口径：本表全部是**单卡**。代码是 `device_map={"": "cuda"}`（`grpo.py:89`），**不支持多卡/DDP**，
> 所以"资源充足"只讨论单卡显存，不涉及多卡线性加速——这条写进 §8 限制。

### 5.3 身份训练：训练前后（`--method mezo`）

**方法**：零阶（`--method mezo`），sft 目标，引擎 `SCORE` 前向，lr 3e-5，LoRA rank 8 / alpha 16，
ctx 4096、seed 0、12 条问法（中英各半），指标 `answer-ce` = 12 条答案段 nats/token。
复现命令即 §6 的 V6：

```
python train.py --algo sft --method mezo --steps 450 --data identity --word xiezhipeng-git
```

| 配置 | answer-ce | 相对 before | 训练结束时自由生成（问"你是谁"）的归类 | 判定 |
|---|---|---|---|---|
| before | 4.4333 | — | 通义千问（Qwen），由阿里巴巴通义实验室开发 | 基线 |
| 3e-5 × 300（跑 A / B） | 4.2696 / 4.1451 | −0.164 / −0.290 | 仍是 Qwen 身份 | 在学，无副作用 |
| **3e-5 × 450（默认）** | **4.2387** | **−0.196** | 中文身份正确，但丢掉实验室细节 | **默认值** |
| 3e-5 × 600 | 4.0090 | −0.426 | 只说"我是 AI 助手"，丢 Qwen 身份 | ❌ |
| 3e-5 × 750 / 900 | 3.8270 / 3.9445 | −0.608 / −0.490 | 推理段泄漏英文 | ❌ |
| 1e-5 × 300 | 4.3949 | −0.025 | 正常 | 学得慢 6.5× |

**关于"最终生成文本"这一列，先说清证据等级**：它是当时对输出的人工归类，**不是原文**；
原始生成日志没有随仓库提交，也无法从现有材料复原（§8 第 12 条）。审核方若要原文，
用上面那条命令自己跑一次即可 —— 它同时打印 before / after 的 `gen`。
本节其余结论只依赖 **answer-ce 的差值**与**行为分类**，不依赖未存档的原文。

**三条必须写进审核结论的话**：

1. **CE 在降，但训练目标里的事实在自由生成里没落地**：12 条数据 9 条含"由 xiezhipeng-git 微调"，
   900 步也一次都没生成这个署名；强制读法只 +7.79 nats（−58.88 → −51.09）。
   零阶单步更新信噪比 ~1/√d（方向方差 3.0e+02、|mean|/std 0.15），背下长答案里的专名要上千步 ——
   **这是方法的账，不是 lr 没调好**。因此本 PR 对身份训练的表述是
   **"管线可用、署名未落地"**，不主张"身份训练成功"。
2. **CE 不单调**：450 比 300 高 0.09；同配置两次 300 步差 0.12 nats（跨进程抖动）。
   **小于 ~0.1 nats 的差不当结论**，所以默认 450 的依据是"最后一次行为正常"，不是 CE 最优。
3. **训练前的 `gen` 没有落盘**，before 行按 ce 反推（lr 3e-5 第 100 步 ce 反升 0.05，
   该步输出等价于未训练基线）。这是本 PR 的证据缺口，已记入 §8 第 1 条。

### 5.4 数学训练：训练集学会 vs 非训练集回退（本 PR 最重要的一条判据）

训练数据 4 道 MATH-500 难题；评测 evalscope `math_500`（分层抽样 243 项、temperature 0、
max_tokens 8192），**同一个引擎、只换 adapter**，baseline 不挂 adapter。

| 指标 | baseline | mezo | backward | dpo |
|---|---|---|---|---|
| 训练集 3 道错题 | 0/3 | **2/3** | **2/3** | **2/3** |
| 全量 243 项 acc | 98.35% | 97.94% | 86.83% | 98.77% |
| **剔除 4 道训练题后的干净 239 项** | **100.00%** | **98.74%**（−3） | **87.03%**（**−31**） | **99.16%**（−2） |
| L5 acc | 94% | 92% | **60%** | 96% |

结论：**"训练集学会"和"能力没掉"是两笔账**。反传在 4 题上不是更优解，它是"学得快也忘得快"
（学会 2 道、L5 掉到 60%、连 L1 都掉 3 项）；零阶掉 3 项、DPO 掉 2 项。
干净子集 baseline 已 100% ⇒ 1 项 = 0.42 pp，任何下降都需要解释。

#### 5.4.1 反传那 −31 项：不是数学错，是工作点 + 验收缺口（2026-11-07 已闭合验收缺口）

这一条必须写得很准，因为「−31」看起来像实现 bug。分开说**已排除什么**、**已确认什么**、
**还缺什么**。

**已排除：损失数学和算子路径写错。** 三条硬证据：
① `objectives.py` 的 numpy 与 torch 两条路径逐值对拍（`test_objectives.py`）；
② 引擎 `SCORE` 与 HF 在同一 ids 上差 **0.0075 nats**（§5.5）；
③ 反传用的是 PyTorch autograd + `update_rules.BackwardRule`（`AdamW` + `clip_grad_norm_`），
不存在手写梯度。所以「公式写错」这条假设没有支撑。

**已确认：这不是第一次观察到同类事故，项目里有专门的护栏，但护栏看不见数学。**
`guard.py` 的 docstring 记着同一类现象的实测（2026-10-30）：训练 10 条同构样本、8 epochs 后
算术 holdout 答错、无关提问被插名，归因是「梯度把共享投影往捷径方向拖」。为此 `sft.py`
内置两道护栏且**默认全开**：token 选择 `--select-thresh 1.0`（只让基座觉得意外的 token 进 loss）、
能力门 `--eval-every 4` / `--cap-tol 0.5` / `--cap-stop 1.5`（漂移超界就拒、超硬顶就停并回滚 best）。
**但能力门的探针是 `prompts_holdout.jsonl`（身份 + 能力向），不包含 MATH-500。**
所以数学能力掉 31 项，可以从护栏眼皮底下发生而没有任何一次 hard stop —— 这与观察到的现象一致。

**最可能的工作点问题**（差异是量级的，不是细节）：反传默认 `lr 1e-4`（零阶 3e-5）、
`rank 16 / alpha 32`（零阶 `rank 8 / alpha 16`）、优化器是自适应步长的 AdamW，
训练集只有 **4 道题**，且没有针对数学的 held-out 早停。4 题 × 高 lr × 自适应优化器
是把 LoRA 权重推出有效区间的典型配方；L1（最简单档）也掉 3 项，符合"通用能力被连带拖动"
而不是"难题学不会"。

**2026-11-07 补证后，下面五行里前四行已闭合**（最后一行仍缺，属护栏本身的观测面）：

| 缺的 | 现在的状态 |
|---|---|
| `lr` / `steps` / `rank` 未落盘 | ✅ 已修：新增 `backward/run_meta.py`，**加载模型之前**写 `<out>.meta.json` 的 `phase=start`，结束合并 `phase=done`；`vars(args)` 全量入 meta（新增参数自动覆盖），附训练数据 sha256 / 行数、git HEAD、keep-best、能力门终评、峰值显存。`tests/test_run_meta.py` 9 项守着，写失败只警告不挂训练 |
| 同一个 `--seed` 两次训练权重不一致 | ✅ 已修（2026-11-07 真机复现后治根）：两次同 seed 重训 sha256 不同、相对 L2 **1.41**（与权重同量级）。根因是 `LoRALinear.A` 用 `nn.init.kaiming_uniform_` 吃 **torch 全局 RNG**，训练入口只固定了数据顺序的 Generator。现改为按 `seed + 模块序号` 播种的**局部 Generator**（与 kaiming 闭式界逐值等价），`backward/sft.py` / `hf_ref.py` 传 `--seed`；2 项单测守同 seed 逐位一致、异 seed 必不同 |
| 同一 adapter 重复评测两次的波动量级 | ✅ 已测：24 条净差 **2~5 条（8.3~20.8 pp）**，见 §5.6 |
| 「未训练 adapter（B=0）」对照未做 | ✅ 已做：同规格 rank16 / 96 张量 / 25.51 MiB 零表，换另一 seed 漂移 +0.00001~+0.00020 ⇒ 挂表本身零扰动，见 §5.6 |
| 能力门没有数学探针 | ❌ 仍未接进 `--probes`（§8 第 14 条）。数学回退改由**独立保持集评测**发现，不靠护栏 —— 这是刻意的选择：护栏是训练内 hard stop，评测是训练外判据，两者不互相替代 |

**因此本 PR 的立场（更新）**：−31 仍是**一次未记录超参的实验结果**，但不再用「归因未完成」搪塞 ——
已确认它不是损失数学写错（§5.5 三条硬证据），判不了的原因是**验收方法**：饱和评测集 +
未标定噪声 + 护栏看不见数学。本轮补齐可复现性（超参落盘 + 同 seed 逐位一致）、噪声底、
B=0 对照三件事，并用当前默认参数重做一次（超参全部落盘），结论见 `RETENTION-EVAL.zh.md` §3.7。
**不把 −31 说成"反传这个方法不行"，也不说成"已修复"；它不作为当前代码的证据。**

**重做的实测结论（2026-11-07，替代上面那条历史数字）**：`f416e41d` 上用
4 道 MATH-500 L5 题 / 8 步 / `lr 1e-4` / `rank16` 重训，全部参数由 `run_meta.py`
落在 `<out>.meta.json`（`phase=done` 含 8 步、能力门终评、峰值 28.0 GiB）。
36 条训练未见题上：B=0 anchor **0.722**、同表重复 **0.639**、重训表 **0.611**
（净 −4 条，p=0.424）；**重训表 vs 同表重复 McNemar p=1.00** ⇒ 按已标定的
「≤5 条不可判」噪声带，**不能判定为遗忘**。长锚漂移 −0.012~−0.016 nats/tok
（约 10× 噪声底、9/10 与 4/4 为负）证明 adapter 不是空表，但比旧 DPO 崩坏
（−0.079~−0.138）小一个量级。**边界**：4 道训练题基座本来就 4/4，塑性层饱和，
这轮是风格拟合型 SFT；8 步是很短的工作点，更长训练 / 更高 lr 未测，
因此结论是「该工作点未复现 −31」，不是「反传安全」。

> 口径（历史，11-03）：mezo adapter = `_run_hardmth/final.gguf`（500 步）；bwd / dpo 两个 adapter
> 只留产物与评测报告，训练超参未落盘（§8 第 3 条），**两个文件现已不在本地**。同一 adapter
> 自检 2/4、evalscope 3/4，差别在判定口径（逐字 vs 数值等价），本文用 evalscope。
> `_evalscope/` 与训练日志均未随仓库提交（§8 第 12 条）。

### 5.5 正确性对拍

| 项 | 结果 |
|---|---|
| 引擎 score vs HF CE（同一 ids，8 token） | −56.6508 vs **−56.6433**，差 0.0075 nats |
| `objectives.py` numpy vs torch | `test_objectives.py` 逐值对拍通过 |
| 训练目录单测 | **362 项全过**（`python -m unittest discover -s training/tests -p "test_*.py"`，15 s；2026-11-07 Windows 实测） |
| serve 全量 | 1,535 项（2026-11-07 同步 v0.2.12 后实测：1,508 过 + 26 skip + 14,184 subtests 过）；`PYTHONUTF8=0` 下 1 项 `test_web_start_question` 因子进程 GBK 解码失败，改 `PYTHONUTF8=1` 单独复验通过 ⇒ 编码环境性，不是回归 |
| 服务会话真机 | 会话期 chat 409 → close 后 200；客户端崩溃也会话正常关 |

---

### 5.6 训练是否影响已有任务（2026-11-07 补证）

> 完整报告与逐条原始数字见 `training/RETENTION-EVAL.zh.md`、`training/EVAL-RESULTS.json`
> （两者随本 PR 提交）；复现命令见该报告 §6。这里只写结论与口径。
>
> **引擎构建口径（2026-11-07 同步上游后补记）**：§5.6 与报告里的 `SCORE` 绝对值、轨迹级漂移
> 是在 `f416e41d` 那次的引擎二进制下测的；同步 v0.2.12 后引擎已重新构建，跨构建的绝对值
> 不可逐值比较（历史实测有 0.5 nats 级抖动），MeZO 只吃同一构建内的差分。训练层 Python、
> 判分器、数据与 adapter 都没变，结论仍限于「同一构建内的差分」，不宣称与新版引擎逐值相等；
> 新引擎下重采评测属于**未测项**。

原判据的问题：§5.4 用的是 baseline 已 100% 饱和的 MATH-500 干净子集，且**没有测过
"同一 adapter 重复评测两次"的波动量级**，所以 −3 / −2 判不了是不是真回退。本轮参考
RaPO（arXiv 2605.09640v2）§4.1 的 Last Accuracy / Forgetting 与 §3.3 的轨迹级 log-ratio
漂移，把判据拆成四维：**可塑性 / 保持 / 漂移 / 噪声与伪影**，并加三组对照 —— 同规格 B=0
零表、同表重复、按 base 题级 acc 分层的「已有能力层」。`truncated` 必须单列（噪声臂自己
就有 4 条 `False→True`），否则会把长度伪影读成能力变化。

| 噪声底（2026-11-07 实测） | 值 | 用法 |
|---|---|---|
| 引擎 `SCORE` 长锚（≥300 token）同进程重复 | std **0.0012** nats/token | 阈值 0.005 ≈ 4σ |
| 短锚（13/38 token）同进程重复 | std **0.114**、max **0.164** | **不可作结论**（同进程也抖，根因未定位，§8 第 5 条） |
| 换表回 anchor 的长锚残差 | ≤0.0037 | 换表干净、无会话缓存污染 |
| 生成层：同一 B=0 表、同题同 seed 连采两遍 | 24 条净差 **2~5 条（8.3~20.8 pp）** | **≤5 条的 acc 变化一律不判退步** |
| 文本级：同表同题同 seed 身份 19 条 | 只有 **4 条逐字相同** | 二值判据零翻转 ⇒ 饱和判据会抹平噪声，不能判退步 |

引擎侧两个硬约束（写进方法，不是绕过）：`LORA` 原地换表**不允许改 rank**（实测
`layer 0 changed shape`），所以单进程多臂必须按 rank 分组、每组用自己的 B=0 表当 anchor，
只比组内差分；`lora_to_gguf.py` 的 `B is zero: never trained` 导出过滤对训练结果正确、
对零表对照不适用，造零表要复用其形状约定但去掉该过滤。

结论：

1. **旧 DPO 的退步是实现 bug，不是算法**：轨迹漂移 −0.079~−0.138 nats/token（噪声底 65~120×）、
   AIMO3 已有能力层 8/8→1/8；三处修复（β 量纲 / epochs 折算 / 能力门 EOS 判据）+ on-policy
   数据后为 −0.003~−0.008、8/8→7/8。
2. **MeZO 零阶不构成退步来源**：漂移 ≤0.0006（噪声内）、数学保持集 0.750→0.750、身份层零翻转。
   它的短板是「学不动」（单步 SNR ~1/√d），属**可塑性账**不是保持账。
3. **正确实现但非 RL 的 DPO 可以零遗忘**：小数据集 DPO 在 15 条已有能力题上零遗忘、
   保持集漂移 ≈0。RaPO 的实证是「RFT 比 SFT 抗遗忘但仍会忘」，决定因素是**更新幅度与
   轨迹级漂移**，不是有没有 RL。
4. **B=0 对照把「挂表扰动」与「训练回退」分开**：同规格零表换另一 seed 漂移
   +0.00001~+0.00020 ⇒ 挂表本身零扰动，退步来自训练权重。
5. 训练目标能否学会与已有能力是否保持是两笔账：`sft_cold`（4 步）是「只破坏不学会」
   （目标 0/4→0/4、已有能力 8/8→4/8），`dpo_fixed` 是「学会 4 条、掉 1 条」。
   全量 AIMO3 的 +4/−2（p=0.688）与 MATH 的 +9/−8（p=1.00）**仍在噪声内，不主张显著提升**。

**反传 −31 项的处置**：那次训练超参未落盘、adapter 已不在本地，无法复现，本轮不复用该数字
作为「当前代码」的证据。已补 `backward/run_meta.py` 让以后每次训练自动落盘完整参数与结果，
并用当前默认参数重做一次可判定的实验（§5.4.1）：36 条保持集 acc 0.722→0.611、
同表重复 0.639、重训表 vs 噪声臂 p=1.00 ⇒ **未复现 −31，也判不了小幅遗忘**；
长锚漂移 −0.012~−0.016 说明权重确实有更新。数字与边界见 `RETENTION-EVAL.zh.md` §3.7。
顺带查出的同 seed 不可复现问题（LoRA A 吃全局 RNG）已从源头修复，并把两次不同
初始化的漂移对拍写进 §3.7：差 ≤0.0015 nats/tok，结论对初始化稳健。

**范围边界**：RaPO 的 Retention Reward / CTAN 与独立的 `--algo gspo` **不在本 PR**。
本轮只借其度量方法（Last/Forgetting、轨迹级漂移），不改 `algos.ALGOS` 注册表；
GSPO 的序列级 ratio 已在 `luspo` 中实现（`objectives.py`，LUSPO 相对它只多乘 `|y_i|`）。

---

## 6. 按这个顺序验（V0~V4 不需要 GPU，V5/V6 要卡）

```bash
cd training

# V0 依赖自检：先确认你要审的那条路线在这台机器上装齐了（无 GPU 也能跑）
python platform_check.py        # 平台 + preadv 垫层 + torch/CUDA 事实；有阻断项会 exit 2
python -c "import numpy, jinja2, regex; print('mezo route deps ok')"                 # 零阶：就这三个
python -c "import torch, transformers, accelerate, safetensors; print('bwd deps ok')"  # 反传：这四个

# V1 注册表与命令面（无 GPU）
python train.py --list                                  # 打 ALGOS × METHODS + 默认 lr
python train.py --algo sft --method mezo --dry-run      # 只打印最终 argv，不加载模型
python train.py --algo sft --backend hf                 # 期望：报错并指向 --method
python train.py --algo sft --method bogus               # 期望：invalid choice（列出 mezo/backward）

# V2 损失单一出处（无 GPU，35 项）
python -m unittest test_objectives -v

# V3 训练目录全量（无 GPU，362 项，~15 s；零阶树 214 项，11-07 口径）
python -m unittest discover -p "test_*.py"

# V4 服务会话（无 GPU；必须在仓库根跑，测试按包相对导入，6 项 ~4 s）
cd <repo> && python -m unittest serve.test_training_api -v

# V5 显存判据（有卡时；三个路径按本机 pack 填）
cd training && python bench/bench_fullctx.py \
    --gguf <IQ3_XXS gguf> --pack <dense pack> --tokenizer <tok 目录> \
    --algos ce --modes mezo --seqlens 131072
# 期望：跑完不挂死，峰值 ~46.97 GiB；--modes bwd 换成反传侧对照（会到不了 131072）

# V6 复现身份训练的一个点（有卡时，ctx 4096，几分钟）
python train.py --algo sft --method mezo --steps 450 --data identity --word xiezhipeng-git
# 期望：answer-ce 量级 4.2，gen 中文身份正确
```

V0~V6 的完整参数见 `README.md`；`--dry-run` 打印的 argv 就是 V6 实际执行的命令。
V1 的输出会带本机绝对路径（engine / pack / profile 由 pinned config 与 `search_roots()` 解析），
换机器看到自己的路径是正常的，不影响判定。
**V5/V6 要先有 §2.5 那个引擎**（含 `LORA`/`SCORE`，即 `feat/train-mezo` 构建出来的）；
拿 `upstream/main` 构建的二进制跑，V6 会卡在等 `LORAOK`。

---

## 7. 分支、合并顺序与回滚

### 7.1 分支拓扑与基线

**三条分支各是什么（实测，非推断）**：「切分提交」指把分层定下来的那个提交；分支 tip 之后还有
纯文档提交（脚本状态不变）。**▸ = 本文所在树**。

| 分支 | 切分提交 | 带什么 | 能不能独立用 | 测试 | 相对 `upstream/main` |
|---|---|---|---|---|---|
| `feat/train-mezo` | `56c7eae0` | 共用层（numpy）+ 零阶（引擎 `LORA` / `SCORE`）+ 引擎侧改动 + 数学造数据/评测；**不含 torch 层** | **能**：运行时只要 numpy + 引擎二进制，训练路径不 import torch | **214**（213 过 + 1 skipped；把 `torch` 顶成 raise ImportError 的模块后同样 214） | ≈90 提交 / 110 文件 / ≈+23k −95 |
| `feat/train-backforward` | `adfa53a7` | 在 mezo 之上叠加：`backward/`（10 模块）+ `ce.py` / `lora.py` / `update_rules.py` + `backward/hf_ref.py`（HF 对拍）+ 6 个 torch bench + 7 个 torch 测试 | **能**：mezo 那棵树是它的子集，只多 32 个 `.py` | **362 全过** | ≈120 提交 / 144 文件 / ≈+34k −95 |
| **▸ `feat/train_support`**（本文所在树，交付面） | 交付分支 HEAD | = `mezo ∪ backforward` + 文档（两线共享的 `.py` 逐字一致；对 mezo 只多那 32 个文件） | 交付面 | **362 全过**（同上） | ≈150 提交 / 144 文件 / ≈+34k −95 |

**如果只想收一条线，代价是什么**（分三条分支就是为这件事，它也是「砍掉 backforward 会失去什么」的答案）：

| 只收 | 得到 | 失去 |
|---|---|---|
| `feat/train-mezo` | 零阶三算法全在（sft / dpo / grpo × 引擎前向）、`LORA` / `SCORE`、引擎侧改动、数学造数据与评测、**0 个新增 pip 包、训练路径零 torch**、214 项 | 只缺**反传那一层**：`backward/` 整层（`sft.py` / `grpo.py` / HF 加载栈）、`update_rules.py` / `ce.py` / `lora.py`、`backward/hf_ref.py`（score parity 对拍）、GRPO 的 TIS 重加权、6 个 torch bench、7 个 torch 测试，以及 `test_objectives.py` 的 numpy↔torch 逐值对拍。**DPO 与 `--init-adapter`（课程续训）两条线都有，不属于反传独有** |
| `feat/train-backforward` | 上面全部 + 反传栈，362 项 | 无（严格超集）。代价是树里同时带着零阶，审核时看不出「哪半是这次要的」—— 除非按 §7.1 的文件清单读 |
| `feat/train_support` | 两条线 + 这组文档（交付面） | — |

一句话：**backforward 相对 mezo 的增量不是「训得更好」，而是「一套不同的执行栈 + 对拍证据」** ——
零阶用引擎前向做零阶近似，反传用 PyTorch autograd 拿真梯度，两者的损失数学共用 `objectives.py`，
所以反传路线同时充当零阶的参考实现：两者在相同输入上的 logp / loss 能对上（§5.5），这才让「零阶这条路线的数学没写错」
成为可验证的结论，而不是自说自话。

如果审核方只关心「零阶能不能训起来」，`feat/train-mezo` 就是完整答案，砍掉 backforward 不损失零阶任何能力；
反过来的说法不成立 —— 零阶是 backforward 的基础层，不能从它里面摘掉。

**分层规则一句话**：三条分支的差异只体现在**文件在不在**，不体现在代码分支上。
`algos.available()` 读磁盘决定 `--method` 有哪几个，缺哪个就把缺的文件名说清楚；
`train.py` / `algos.py` 三条线逐字一致，所以合并永远不会为「dispatch 代码谁对」打架。

四条容易搞错的地方：

- **反传线是叠在零阶线之上的，不是并列**：`comm -23`（mezo 有而 bwd 没有的文件）为空；
  bwd 比 mezo 只多 **32 个 `.py`**（30 个 torch 文件 + `bench/diag_frozen.py` / `bench/mtp_probe_tr.py`
  两个诊断脚本），共享的 62 个 `.py` 逐字一致（`git diff --quiet` 逐个核过）。
  这是"官方只要一条线"能拆的前提：只要零阶交 mezo（**当前 110 文件、214 项**、零 torch），
  要两条交 support；bwd 自己也能独立用，但它交付的价值是"反传 + 对拍"，零阶是它的基础层。
- **零阶线不带 torch 是判据，不是说法**：
  `git grep -nE '^(import|from) (torch|transformers|triton)' feat/train-mezo -- training/` 为空；
  再把 `torch` 换成一个 raise ImportError 的模块顶掉跑 `pytest tests`，仍然 **214（213 过 + 1 skipped）**。
  `train.py → mezo/mezo.py` 的 import 闭包只有 `objectives.py` / `mem_guard.py`。
  摘掉的那一层（`backward/` 全部、`ce.py` / `lora.py` / `update_rules.py`、`backward/hf_ref.py`、
  6 个 torch bench、7 个 torch 测试）在 bwd 上原样存在，`56c7eae0` 一次提交完成摘除。
- **"默认"指的是方法，不是分支**：`algos.py` 里 `DEFAULT_METHOD = "mezo"`（三条分支一致），
  所以 `train.py` 不给 `--method` 时走零阶；这与选哪条分支交付是两件事。在 mezo 上
  `--list` 会写 `backward 不在这条分支上（缺 backward/sft.py）`，`--method backward` 在加载模型
  **之前**就退出并指路（`algos.method_of`），不是跑到一半 ImportError。
- **mezo 的"完整"以 import 图闭合为判据**：旧 `mezo/sft_mezo.py` 曾 import 只在反传层的
  `mtp_draft`（`827d9071` 补齐）—— 那是「文件放在方法目录、依赖落在 torch 层」的错位，
  2026-11-03 连改名一起摆正：它需要 torch，所以现在就在 torch 层（`backward/hf_ref.py`），
  零阶树上不存在这个文件。`math_probe` / `math_build_data`（零阶数学实验的选题与造
  数据）和 `test_curriculum`（测的是 `lora.load_lora_state`）也长期落在反传层（`ae31b3f0` 下移）。
  本轮又把 `extract_num`（判答对/答错的唯一规则）抽到 `bench/math_extract.py`，让零阶的
  `math_build_data.py` 与反传的 `math_probe.py` 共用一份 —— 两条线各写一份正则，改一处忘一处
  就会让 GRPO 拿到错的优势。现在静态扫描"被 import 但在本树不存在"的本仓库模块 = **0**。
- `perf/pack-load-accel` 的 merge（`05d66edc`）**不带功能**：它唯一的独有提交 `2a640fc9`
  （`fadvise(DONTNEED)` + 分段计时）早就在 `06186a31` 里进了训练线，merge 只是补祖先，
  还顺带把同一段定义搬了第二份进来，已由 `330eba47` 回退。

**2 项 watchdog 测试的环境说明（别当成回归）**：`test_bench_limits` 的两个 `Watchdog` 用例会起子进程
`import bench_fullctx`，链路是 `loader_qwen4 → nf4_fused → import triton`；PyPI 上的官方 `triton` 没有
Windows 轮子（要装第三方 `triton-windows`），本机测试环境里没有，所以在 Windows 的 pytest 里这 2 项失败
（`ModuleNotFoundError: No module named 'triton'`），其余 266 项通过。
这四个相关文件（测试、`bench_fullctx.py`、`loader_qwen4.py`、`nf4_fused.py`）与上一版 `0b15d175` **逐字一致**，
不是本次改动引入的。**Linux 上实测 268 全过**（2026-11-03，WSL Ubuntu 24.04 / Python 3.12 / pytest 9.0.2，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q` → `268 passed in 36.97s`），所以这 2 项是环境依赖，不是代码问题。

```
upstream/main 146df3c4（v0.2.12，2026-11-07 已 merge 进三条分支）
  └─ feat/train-mezo            零阶线（HEAD）  共用层 + 零阶 + 引擎（LORA / SCORE）；训练路径零 torch；214 项
       │   56c7eae0  摘掉 torch 层（213 → 134）；方法可用性改成看文件（algos.available）
       └─ feat/train-backforward      反传线（HEAD）  以上全部 + torch 反传层（比 mezo 多 32 个 .py）；362 项
            │   adfa53a7  摘 torch 时的合并点：反传侧保住整层
            └─ feat/train_support     交付面（HEAD）  两条线的并集 + 这组文档（.py 与 bwd 逐字相同）
```

三条都是**当前 HEAD**（文档提交会让 sha 漂，所以只标分支名）。判据可复现：

```
git merge-base --is-ancestor feat/train-mezo feat/train-backforward   # 零阶打底
git merge-base --is-ancestor feat/train-backforward feat/train_support # 并集
git diff --stat feat/train-backforward feat/train_support -- '*.py'    # 空 = 代码零差异
```

**为什么是「零阶打底 + 反传叠加」而不是三条并列**：共用层单独**不能跑** —— `train.py` 的注册表
要把算法映射到脚本，`--list` 与 `test_train_core.py` 都会问"这个方法有实现吗"。本轮的解法是
把这条判断改成**看文件**（`algos.available()`），于是共用层 + 零阶正好构成一棵能独立跑、
能自证的树，**mezo 承载共用层**；反传在它之上只加文件；support 是两者的并集。
审核时按 `upstream/main...feat/train_support` 看 diff 即可，上面这条链只解释提交是怎么分组进来的。

**三条分支都是能独立跑的树，不是只在交付面迁移**：`feat/train-mezo` 自己就是一棵 `training/`
的树（`56c7eae0`，144 项全绿，无 torch），`feat/train-backforward` 用**同一份映射表**再迁移一遍、
merge 零阶层、再叠 torch（`adfa53a7`，271 项），support 是两者合并的结果（`c9c0016f`）。判据如下：

- **官方只要零阶** ⇒ 直接交付 `feat/train-mezo`（当前 110 文件、214 项、训练路径零 torch）。
  不需要 cherry-pick，也不需要再把目录搬一次 —— 这正是"迁移只做在交付面"那版的缺口：
  合并后的树能拆，但源分支拆不出来。
- **只要反传**（不要引擎那套 `LORA`/`SCORE` 改动）⇒ 交付 `feat/train-backforward`（`adfa53a7`）：
  它就是零阶树 + torch 层，两者共享的 `.py` 逐字一致，所以摘掉零阶侧文件也是一个提交的事。
- **两条都要** ⇒ 交付 `feat/train_support`（HEAD）。审 diff 用
  `upstream/main...feat/train_support`，看到的就是 `training/`。
- 三条分支的迁移提交是同一批 rename（`d54d8a32` / `0c676f32`），路径逐一对齐，所以合并
  不会出现两份文件。
- **文档版本**：三树各带一份 `REVIEW-REQUEST.zh.md`（≈1.3k 行），由同一母本参数化生成，
  正文逐字相同，只有「本树事实」的头部、摘要行、§7.1 ▸ 标记不同；评任意一条分支都能看到完整方案。

**基线判据（比对时别拿错参照）**：交付分支建在 `upstream/main` 之上。**2026-11-06 同步到
`9f7d977d`（v0.2.11），2026-11-07 再同步到 `146df3c4`（v0.2.12，95 提交 / 158 文件）**，
两次都按 mezo → bwd → support 级联、只用 merge；`git merge-base --is-ancestor upstream/main <分支>`
对三条分支都为真，相对上游**落后 0 / 领先 102、146、207 个提交**（非 merge 89 / 116 / 152，
2026-11-07 修复提交后实测；之后每个纯文档提交各 +1，属表下注已声明的预期漂移）。merge 后 CUDA 构建（vcvars64 + Ninja / arch 89，0 FAILED）、三树训练全量、
serve 全量回归复核，数字与已知环境性失败见 §5.5。
上游 v0.2.12 新增的 `tests/core/flag_order_test.cu` 用了 MSVC/nvcc 不提供的 `__atomic_load_n`，
本 PR 按同仓库 `include/strata/core/flag_order.hpp` 既有的 `_WIN32` 分支补齐（三树逐字一致）；
这是**修上游平台缺口**，不属于训练功能代码。
本仓库的 `origin` 是 fork（`xiezhipeng-git/Strata`），它的 `main` 停在 `80494788`（2026-10-15），
**落后上游 3,766 个提交**（2026-11-07 实测）。所以比对一律用
`upstream/main...feat/train_support`，拿 `origin/main` 比会误判成"基线过期"。

### 7.2 来源分支审计（这些分支的内容去哪了）

| 分支 | 是什么 | 状态 |
|---|---|---|
| `perf/pack-load-accel` | 载入加速：每块读完 `fadvise(DONTNEED)`，冷读 2.98→0.33 GiB/s 那个坑（28 GiB 9.4 s vs 85.8 s） | **已 merge**（`05d66edc`，无冲突）。内容此前已在树里（`06186a31`），merge 只补祖先、净变化为零；它带出的重复定义已由 `330eba47` 回退 |
| `feat/lora-sign` | adapter 的符号，换表不搬家 | 已纳入（mezo 的 31 个引擎提交之一，`e9173c16`；+56/+13/+64 行，**只服务零阶**的 θ−ε，见下方三层表） |
| `feat/lora-q36` | LP 行协议 + per-request LoRA | 已纳入 |
| `feat/score` | `SCORE <ids>`：逐位置 logp（`/v1/score`） | 已纳入 mezo（merge `53e729fa`，`merge-base --is-ancestor` 核过）。**不单独提 PR**：它技术上能独立（`score.cpp` 里 `lora`/`adapter` 0 命中，不依赖换表命令，单独给任意 ids 打分就能跑），但脱离训练只是"给一段 caller 指定的 ids 出逐位置 logp"，审核者第一个问题就是"谁用"；且它是零阶的度量核心（θ±ε 的 L 全靠它），拆开要两处维护 |
| `feat/kv-auto` | `--kv auto` 按上下文选精度 | 已纳入 |
| `feat/strata-rollout` | 返回生成序列与它续写的 prompt | 已纳入 |
| `pr-3267-lora` | 上游 PR #3267 的 LoRA kernel（`lora.cu` 421 行 / `lora_parity` 237 行），作者 Alexey Rogov | **内容已含**：support 是超集（477 / 301 行）。注意 **`upstream/main` 里没有 LoRA kernel**（`src/kernels/cuda/lora.cu` 不存在），这条 PR 至今未合入（分叉点落后 main 2,413 个提交），我们是本地拉的只读依赖，不是作者 |
| `feat/unilora` | 本地 Uni-LoRA（`unilora.py` 245 行 + `lora_adapters.cpp` 149 行 + qwen4exp 加载链） | **内容已含**：`unilora.py` 245 行逐字一致，kernel 150 行。仅差 8 个一次性调试脚本（`dbg_*.py`、`bench_weight*.py`、`verify_q4.py`），不是功能 |
| `feat/zo-mezo` | 早期 PyTorch 版 MeZO trainer | 已被 `mezo.py` 取代；内容同上已含，只差同一批调试脚本 |
| `feat/initial-training` | 早期 6 个训练提交 | 完全吸收（它新增的文件 support 全有） |
| `fix/q36-hf-forward-parity` | HF 前向对拍线 | 内容已含；与 support 合并有 19 处冲突，不建议重并 |

### 7.3 LoRA 的三层归属（作者不同，别混说）

上游到底有多少 LoRA 代码、这些 `lora` 是谁写的 —— 分三层，作者和归属都不同。

| 层 | 内容 | 作者 / 来源提交 | 在 `upstream/main` |
|---|---|---|---|
| **kernel 主体** | `lora.cu` 421 行、`lora.hpp` 77 行、`lora_parity.cpp` 237 行 | **Alexey Rogov**（上游 PR #3267），`5e910f85` / `799c0c17` | ✗ 三个文件都不存在 |
| **符号翻转** | `g_on_host`(bool) → `g_polarity_host`(int `0/+1/−1`)，sign **只落累加那遍**（+56 / +13 / +64 行） | 本仓库 `e9173c16`（即 `feat/lora-sign`） | ✗ |
| **换表 + HF 工具** | `LORA <path>` 原地换表（`lora_adapters.cpp` 150 行）、`unilora.py` / `lora.py` / `lora_to_gguf.py` | 本仓库 `9d278ddc` / `6654bb96` | ✗ |

读法是三句话：

1. **上游 `main` 只有 adapter 的加载路径**（`--lora` 参数、会话状态按 LoRA 开关缓存），**没有 kernel**。PR #3267 至今没合入（分叉点落后 `main` 2,413 个提交），所以这 421 行不是"上游代码"，是"一个尚未合入的上游 PR"，本仓库把它当**只读依赖**带着（`pr-3267-lora` 是 `feat/train-mezo` 的祖先，推 mezo 就自动带上了，不需要额外动作）。
2. **我们自己的 kernel 增量只有 133 行，且只服务零阶**：θ−ε 不是"再存一份表"，而是把同一张表按 −1 用，所以 `on` 从 bool 改成三态 int，sign 只落在 `add=1` 那一遍（落两遍会平方回 +1）。没有 MeZO，这 133 行没有任何意义 ⇒ **跟随 mezo**。
3. **HF 侧那三个脚本跟随反传**（`unilora.py` / `lora.py` / `lora_to_gguf.py`）：它们的活是"HF 训完 → 转成引擎能读的 adapter GGUF"，服务 `--method backward` 那条线。

**所以 `feat/score` 和 LoRA 都不是能独立交付的分支**：SCORE 拆出去没有使用者，LoRA 拆出去是别人的 PR。

> 真正适合独立提 PR 的是 **`perf/pack-load-accel`**（Linux 载入存档加速：每块读完 `fadvise(DONTNEED)`，28 GiB 冷读 85.8 s → 9.4 s）——它跟训练毫无关系，是纯性能修复，已在 `05d66edc` merge 进来（净变化为零，功能早就在 `06186a31` 里），审核方想单独先收这一段完全可以。

### 7.4 兼容性、回滚与合并顺序

- **推理路径**：无训练会话时只多一个 `training_active()` 布尔判断，128K 走廊、KV 档位、
  logprobs 行为均不变（serve 全量 1,535 项为证，2026-11-07 merge 上游 v0.2.12 后复跑；1 项编码环境性失败见 §5.5）。
- **旧入口**：`--objective` / `--backend` 保留为**报错**（不是别名），避免两种叫法长期并存。
  若下游脚本在用旧名，这是不兼容变更，**请确认可以断**。
- **回滚**：本 PR 全是新增文件 + `serve/server.py` 的独立段落，`git revert` 单 commit 即可；
  不涉及数据迁移、不改 gguf、不改配置文件格式。
- **合并顺序**：零阶依赖的 `LORA`/`SCORE` 命令循环只在 `feat/train-mezo`，
  `upstream/main` 还没有（§2.5）。**本 PR 已经带上引擎那段**（`feat/train-mezo` 的 31 个提交，
  含 `LORA` / `SCORE` / `--kv auto` / 适配器符号），所以不必"先合引擎"；若上游希望分开审，
  仍可按 Q9 拆成引擎 PR + 脚本 PR。
- **反传那一层可以整层拆出 —— 已实测，而且现在是按文件拆**：拆出来的形态就是 `feat/train-mezo` 本身
  （`56c7eae0`，该提交时 144 项全绿；当前 214 项、训练路径零 torch）。bwd 只比它多 32 个 `.py`，共用层与 `train.py` /
  `algos.py` 两边逐字一致，所以"拆"既不用挑文件、也不用改代码分支：
  1. **拆之前先证明它能自己立住** —— 这一步实测出一个既存缺陷：旧 `sft_mezo.py` 的 score-parity
     版本（`a177ad4b`）落在反传层，而 `feat/train-mezo` 上还留着一份旧版，两边对不上，零阶分支上
     `test_train` / `test_bench_limits` **导入即失败**（`cannot import name '_Pair'`）。
     （中间曾把它与它依赖的 `lora.load_lora_state`（28 行）搬回零阶层 —— `3eb6e36d` + `6c69bc19`；
     最终形态是 2026-11-03 的归位：它需要 torch，就该整份待在 torch 层，改名 `backward/hf_ref.py`，
     零阶树不再带这个文件。）**这不是本 PR 新引入的问题，是分支原来就有的** —— 不做"拆分演练"就看不见。
  2. **拆掉不能是静默降级**：早先零阶树里留的是 `backward/` 的**旧版薄壳**（`--method backward`
     能跑，只是没有 DPO / curriculum / clip window / TIS）—— 那是"能跑但能力缩水"，最难看出的一档。
     `56c7eae0` 改成整层摘除：`algos.available()` 只看脚本在不在，`--list` 写
     `backward 不在这条分支上（缺 backward/sft.py）`，`--method backward` 在加载模型**之前**就
     退出并指路。「静默降级」与「明确没有」的区别，就是审核者能否一眼看出交的是哪一档。
  3. **零阶的损失数学不再借反传层的测试来守**：`test_objectives.py` 是 numpy↔torch 对拍，随 torch
     层一起走了；补了 `test_objectives_np.py`（numpy 路径 18 项：CE 掩码 / DPO 饱和与长度归一 /
     组内优势 / GRPO 非对称裁剪 / DAPO vs seq），零阶 214 项里数学是有覆盖的。
  4. **耦合是层内的**：那 32 个文件 import 同目录的共用层（`objectives` / `mem_guard` /
     `update_rules` / `lora` / `ce`），而共用层在零阶树上；另有 `mezo.py` / `lora_to_gguf.py`
     横向 `sys.path.insert` 到仓库 `tools/` —— 训练目录不是自闭环。所以"拆"的正确含义是
     **交零阶那棵树**，不是从一棵树里抠掉几个文件。
  5. §5.4 的反传证据（−31 项）属于被拆走的那一层，裁剪版不引用它；零阶证据（§5.1~§5.3）不受影响。
- 合并注意：两棵树（`Strata-mezo` / `Strata-prtrain`）此前各有一份 `mezo.py` / `objectives.py` /
  `mem_guard.py`。**已核实内容逐字一致**（差异只有 CRLF/LF），现在两份都以 `feat/train-mezo`
  的提交为准；`Strata-mezo` 那次 merge 之后两树同源（同步一律 merge，不 rebase）。

---

## 8. 已知限制（明确不在本 PR 承诺范围内）

1. **默认 450 步没有专用跑，训练前 `gen` 也没有落盘**：450 的证据来自 900 步长跑里的检查点，
   before 行是按 ce 反推的。要写成"默认值验收"需补一次 450 步并同时记录训练前生成的对照。
2. **16K~64K 的零阶步时未测**，只有显存锚点；速度只有 8K（4.39 s）与 128K（63.8 s）两点。
   所以"中间长度多快"目前只能按插值估，不作承诺。
3. **历史 bwd / dpo 两个 adapter 的训练超参与日志未落盘**（只剩产物 + 评测报告）。
   因此 §5.4 里反传那 **−31 项**只能作为「旧口径的一次实验」保留，不能断言具体是 lr 还是步数，
   也不能反过来说"实现有问题"；它**不作为当前代码的证据**。**该缺口已修**：现在每次反传训练
   都由 `backward/run_meta.py` 在加载模型前写 `<out>.meta.json`（`vars(args)` 全量 + 数据指纹，
   结束合并 `phase=done` 结果，§5.4.1）；LoRA 的 A 初始化也已改成按 seed 播种的局部
   Generator，同 seed 可逐位复现。当前默认工作点的可复现重做见 §5.6 与
   `RETENTION-EVAL.zh.md` §3.7。
4. **grpo on-policy 只有反传线**；引擎侧 grpo 目前只吃回放轨迹（`--rollouts`）。
   在线 transport（`--serve-url`）解除的是"训练与推理抢卡"，不是这条限制。
5. **短序列抖动未定位，且同进程也存在**：跨进程同配置两次 300 步 CE 差 0.12 nats；
   长锚（≥300 token）同进程 `SCORE` 重复 std **0.0012** nats/token，但 13/38 token 的短锚
   同进程重复 std **0.114** / max **0.164**（§5.6，2026-11-07 实测）。⇒ 跨进程不比绝对值，
   短锚也不单独作结论（身份类改走生成层），长锚才用作漂移主判据；根因仍未定位，不猜。
6. **反传 4608→9728 步时超线性（11.2× vs T² 的 4.5×）根因未定位**，疑为贴顶后的 host 换页。
   因此 §5.2 表 C 里 9 728 那一行只能当**上界**读：若根因是换页，换大卡后该行会明显改善，幅度未测。
7. **§5.2 表 B 的 24 / 80 / 96 / 141 / 192 GiB 四列是模型推算，不是实测**。唯一实测点是 48 GiB；
   模型残差 ≤ 0.6 GiB，但 80 GiB 以上没有验证点，**并且只对这一个 pack 档（IQ3_XXS + nf4）成立**。
   换模型档位要重标 `BASE_GIB` / 专家 staging 两项。
8. **不支持多卡**：`device_map={"": "cuda"}` 单卡，没有 DDP/FSDP/offload。所以"资源充足"只按单卡显存讨论，
   多卡的线性加速不在本 PR 范围。
9. **probe word logp 已停用**：11-01 身份数据重写后答案不再以 `{word}` 开头，旧指标与训练目标脱钩，
   验收统一用 `answer_ce`。
10. **未测**：引擎侧每 token H2D 计数、`causal_conv1d` 修复后的加速比、mezo/grpo 的真实批量吞吐、
    零阶在"更大 expert cache"下的命中率收益（换大卡对零阶的加速，理论上来自这里，实测为 0 收益假设）。
11. **旧过程式线的 36 题 logp 曲线（−15.78 → −11.44）没有落盘**，只在当时会话输出里，
    因此不作为证据引用。
12. **原始日志未随仓库提交，部分已不可复原**：`logs/` 与 `_*` 是运行产物（`.gitignore` 挡掉），
    其中身份训练的 `gen` 原文、128K 三算法的逐步 rel、MATH-500 的 `_evalscope/` 报告都已不在本地。
    附录 A 逐行标了入库状态。受影响的是「原文级」证据；§5.3 / §5.4 的差值结论仍可用 §6 的命令重跑复现。
13. **评测集的显著性已标定（2026-11-07）**：原来写「没有测过同一 adapter 重复评测两次的
    波动量级」，现已补上 —— MATH-500 干净子集 baseline 仍是 239/239 = 100%（饱和，1 项 = 0.42 pp），
    但同一张 B=0 表、同题同 seed 连采两遍 24 条就**净差 2~5 条（8.3~20.8 pp）**，
    引擎 `SCORE` 长锚同进程重复 std 0.0012 nats/token、短锚 std 0.114。
    ⇒ **零阶 −3、DPO −2 明确判为噪声内**；**≤5 条的 acc 变化一律不能判退步**。
    判据与完整数字见 §5.6 与 `RETENTION-EVAL.zh.md`。反传 −31 属另一口径（未落盘、
    不可复现），按 §5.4.1 处置，不拿这条噪声带去解释它。
14. **防遗忘护栏的观测面不含数学**：`sft.py` 的能力门（`--eval-every 4` / `--cap-tol 0.5` /
    `--cap-stop 1.5`）跑的是 `prompts_holdout.jsonl`（身份与环境能力向），没有 MATH-500 探针。
    所以数学能力的回退可以从护栏眼皮底下发生而没有任何 hard stop（§5.4.1）。
    零阶侧（`mezo.py`）**完全没有接护栏**，`guard.py` 只在反传侧被调用。
    **2026-11-07 补证**：这一条仍是限制，但已用**训练外的独立保持集评测**（§5.6）覆盖数学回退，
    并补了 `run_meta.py` 的 `phase=done` 记录能力门终评（`cap_drift_final` / `best_step`），
    让护栏的判据事后可查。**没有把 MATH 探针接进 `--probes`** —— 那会改变训练内行为，
    超出本 PR 范围，留作后续。

---

## 9. 需要评审确认的 12 项

左列是待确认的问题，右列是**本 PR 当前的选择**（不是建议，是代码现在的状态）。
若某项要改，改动范围写在右列的括号里。

| # | 待确认 | 本 PR 当前的选择 |
|---|---|---|
| Q1 | 接受"引擎无反传、backward 走本地 HF 栈"这条边界吗？还是要引擎侧也实现反向？ | 接受（引擎是推理栈；反传在 `backward/`，独立可拆） |
| Q2 | `objectives.py` 的 numpy/torch 双路径，是否该合并成单实现 + 薄适配层？ | 保留双路径（零阶不装 torch；逐值对拍兜漂移） |
| Q3 | 默认 `--method mezo` 是否合理，还是强制显式传 `--method`？ | 保留默认 mezo + 文档说明 |
| Q4 | `algos.py` 里 lr 按（算法×方法）硬编码，是否改为可覆盖配置（env / yaml）？ | 先硬编码，等有第二组工作点再说 |
| Q5 | `/v1/training/*` 是否该从 `serve/server.py` 拆出成 `serve/training.py`？ | 本轮不拆（训练会话与闸门紧耦合于服务状态机）；若要拆，合并后单独做更安全 |
| Q6 | 会话期推理返回 409 而非排队，可接受吗？ | 可接受（训练时推理语义不明确） |
| Q7 | 测试够不够：训练目录 362 项 + serve 6 项，但没有端到端（真跑一步再断言 CE 下降） | **已补组件级 e2e**：`tests/test_run_meta.py::TrainStepE2E` 用项目自己的 `sft_loss` + `BackwardRule` 真跑一步断言 CE 下降；125B full e2e 仍需 GPU（§6 V5/V6） |
| Q8 | `bench_*.json` 这类实测结果文件是否随 PR 提交，还是只留文档引用？ | 提交（它们是 §5 的原始数据） |
| Q9 | **合并顺序怎么走**：零阶依赖的 `LORA`/`SCORE` 命令循环还在 `feat/train-mezo`，`upstream/main` 没有。是先单独提一个引擎 PR（我们自己的部分：`SCORE` + 换表 + 符号翻转；`lora.cu` 主体那 421 行属于 PR #3267，见 Q11），训练脚本排它后面？还是两个一起提？ | 先引擎、后脚本，各自可独立回滚 |
| Q10 | README 里"engine side, all merged upstream"这句与事实不符（见 §2.5），本 PR 里一并改掉可以吗？ | 改（文档不能留着错的事实） |
| Q11 | **LoRA kernel 主体是尚未合入的上游 PR #3267**（Alexey Rogov，`lora.cu` 421 行），本 PR 当只读依赖带着（三层归属见 §7.3）。如果 #3267 先被合入，要不要把本 PR 的那 421 行 rebase 掉、引擎面只留我们自己的 133 行符号翻转 + 换表 + SCORE？ | 倾向等 #3267；它合入后本 PR 的引擎面自动缩小，届时 rebase 一次即可 |
| Q12 | **反传路线要不要一起收？** 它是独立一层：比零阶树只多 32 个 `.py`（`backward/` 10 个模块 + `ce.py`/`lora.py`/`update_rules.py` + `backward/hf_ref.py` + 6 个 torch bench + 7 个 torch 测试 + 2 个诊断脚本），全部在 `training/` 下，不动 `serve/` 与引擎；**只收零阶 = 直接交 `feat/train-mezo`（`56c7eae0`），当前已验证 214 项（213 过 + 1 skipped）且不需要 torch**；那棵树里 `--method backward` 会明确报「缺 backward/sft.py」并指路（不是旧版薄壳的静默降级）。需要反传能力才需要这一层 | 默认一起收；只要零阶就交那条分支，见 §7.4 |

---

## 附录 A：证据出处

> **入库说明（先看这条）**：随仓库提交的是 `training/*.json` 汇总表、代码本身，
> **以及 2026-11-07 新增的 `training/RETENTION-EVAL.zh.md` + `training/EVAL-RESULTS.json`**
> （§5.6 的证据）；`logs/`、`_*` 目录属运行产物，按 `.gitignore` 不提交，其中一部分原始日志
> 已在本地清理、**无法复原**。下表逐行标注了"已入库 / 未入库"：标"未入库"的数字只能靠重跑
> 复现（命令见 §6），不是"仓库里有文件可查"。

| 数字 | 出处 | 入库 |
|---|---|---|
| 六方案极限、bwd 实测点位 | `bench_limits.json`、`bench_budget.json` | ✅ |
| 换卡推算（表 B） | `python bench/budget.py --vram {24,32,48,80,96,141,192}` 的输出；模型在 `mem_guard.estimate_peak_gib` / `max_seq_len`（6 个锚点，残差 ≤ 0.6 GiB） | ✅ 代码可跑 |
| 三档判据与"不可训练"四类 | `mem_guard.require_budget` / `_limit_gib` / `NO_CKPT_FACTOR` / `MOE_STAGING_GIB` | ✅ |
| 三档实测（表 A、表 C） | `bench_engine_128k.json`（零阶 8K / 128K）、`bench_limits.json`（反传四点）、`logs/api128k_*.log`（128K×三算法）、`logs/accept_lr3e5_s900.log`（ctx 4096，900 步 588 s） | 前两项 ✅；`logs/` ❌ |
| 8K / 128K 引擎速度与显存 | `bench_engine_128k.json`、`logs/fullctx_probe_128k.log` | 前者 ✅；后者 ❌ |
| 128K × 三算法步时与 rel | `logs/api128k_{sft,dpo,dpo_default3e5,grpo}.log` | ❌ 未入库 |
| 身份前后与 gen 原文 | `logs/accept_lr3e5_ce.log`、`accept_lr3e5_s900.log`、`accept_lr1e5_ce.log` | ❌ 未入库，**原文已不可得**（§8 第 12 条） |
| 数学训练集前后 | `_run_hardmth_train.log` | ❌ 未入库 |
| MATH-500 四次评测 | `_evalscope/{baseline,mezo_lora,bwd_lora,dpo_lora}/` | ❌ 未入库 |
| 单测逐条 | `python -m unittest discover -s training/tests -p "test_*.py"`（本版 362 项全过，2026-11-07 Windows 实测） | ✅ 可重跑 |
| 依赖清单 | `requirements.txt`（零阶够用）、`.github/workflows/training-test.yml`（no-torch 合并门）、`training/README.md` §Windows environment notes（反传实测组合） | ✅ |
| 平台支持矩阵 | `platform_check.py`（`SUPPORTED`、`preadv_shim.py`） | ✅ |
| 保留能力评测：轨迹级漂移 / 噪声基线 / B=0 对照 / 分层 | `training/RETENTION-EVAL.zh.md` + `training/EVAL-RESULTS.json`（逐条原始结果在会话产物目录，按报告 §6 可重跑） | ✅ 报告与结构化结果 |
| 每次反传训练的完整超参与运行结果 | `backward/run_meta.py` 自动写 `<--out>.meta.json`（`phase=start` 先落盘）；`tests/test_run_meta.py` 8 项 | ✅ 代码可跑 |
| 训练一步闭环（§9 Q7） | `tests/test_run_meta.py::TrainStepE2E`（项目自己的 `sft_loss` + `BackwardRule` 真跑一步断言 CE 下降） | ✅ 代码可跑 |
| MeZO 方法与官方仓库 | arXiv 2305.17333；<https://github.com/princeton-nlp/MeZO>（本 PR 未拷贝其代码，见 §2.6） | ✅ 外部 |

---

## 附录 B：实测台账（原始点位与口径）

> §5 是结论，这里是结论背后的原始数据与口径约定。复核某一行数字时反查本节。

### C.1 全部实测点位（含已退役路线）

| 路线 | T | 峰值显存 | 步时 | 状态 |
|---|---|---|---|---|
| mezo（引擎，当前默认） | 8 192 | 46.96 GiB¹ | 4.39 s | ok，单次前向 3 656 tok/s |
| mezo（引擎，当前默认） | 130 816 | 46.97 GiB¹ | 63.8 s | ok，单次前向 4 313 tok/s |
| mezo（PyTorch，已退役） | 27 648 | 33.02 GiB | 137.5 s | ok（201 tok/s） |
| mezo（PyTorch，已退役） | 49 152 | 47.54 GiB | 598.1 s | ok（82.2 tok/s），贴顶 |
| mezo（PyTorch，已退役） | 55 808 | — | — | OOM（单笔 24.9 GB 的 [T,T] int64） |
| bwd sft | 4 608 | 34.99 GiB | 81.9 s | ok |
| bwd sft | 9 728 | 44.95 GiB | 914.7 s | ok |
| bwd sft | 16 384 | — | — | **挂死**（无报错、无退出） |
| bwd dpo | 2 560 | 34.16 GiB | 49.8 s | ok |
| bwd dpo | 5 120 | 41.46 GiB | 53.9 s | ok |
| bwd dpo | 8 192 | — | — | 崩（native 栈） |
| bwd grpo | 1 024 | 29.60 GiB | 26.1 s | ok |
| bwd grpo | 2 560 | 34.18 GiB | 28.4 s | ok（rows=2） |

¹ nvidia-smi 口径，含专家缓存；账面需求见 §5.1（26.9 GiB）。

> **读这张表先看这一条**：`bench_limits.json` 里 `mode=mezo` 的行是**已退役的 PyTorch 零阶**
> （`hf_ref.py`，27 648 封顶、55 808 OOM），**不是当前默认路线**。别把 49 152 当成零阶的上限 ——
> 当前引擎零阶在 128K 的点位是 46.97 GiB / 63.8 s。它留着只为 score parity 对拍
> （HF −56.6433 vs 引擎 −56.6508，见 §5.5）。

### C.2 零阶的速度-长度（引擎，单次前向）

| 上下文 | 每步 | 单次前向吞吐 | prefill | decode | KV |
|---|---|---|---|---|---|
| 8 192 | 4.39 s | 3 656 tok/s | 4 513 tok/s | 126 tok/s | fp16，0.21 GiB |
| 130 816 | 63.8 s | 4 313 tok/s | 4 998 tok/s | 108.6 tok/s | int8，1.68 GiB |

三条口径：

- 本表是**单次前向**吞吐（`score_tok_s`）；一步含 θ+ε 与 θ−ε 两次前向，所以步级吞吐约是它的一半
  （§5.2 表 A 用步级：2 050 tok/s）。两个都对，别混着比。
- 512 tok 冒烟曾量到 12.7 / 18.9 / 25.3 s（sft / dpo / grpo），那是逐条 serial 的**调用开销**，
  不代表吞吐，不列入本表。
- 满上下文 450 步 ≈ **7.7 h** 是推算（61.4 s × 450），不是实测；16K / 32K / 64K 中间档未测步时，
  只有显存锚点（见 §8 第 2 条）。

### C.3 方法本身的四条性质（这些参数是硬的，不是调出来的）

| 性质 | 实测 | 出处 |
|---|---|---|
| **loss 必须按 token 平均** | 旧 sum 口径在 123 969 tok 上 pj = −4.03e+04，两步把 ‖θ‖ 从 736 推到 1419（+92%）、生成乱码；改 mean 后 pj −0.43~−0.63（与 HF 0.55 同量级）、‖θ‖ 只 +1.5% | `--loss-reduction` 默认 mean |
| **步长看 rel，不看 lr** | `rel = ‖lr·v‖/‖θ‖` 超过 0.05/步必漂。128K 实测：sft 1.25e-3、dpo 归一后 2.3e-4~5.5e-4、grpo 6.4e-4（全部稳） | 三个 `api128k_*.log` |
| **DPO 的 margin 必须按答案 token 归一** | 序列级 margin 在 122K 对上把 lr 1e-5 放大成等效 3 个数量级 ⇒ ‖θ‖ 19.6 → 10 175 发散；归一是长度不变量，一个默认 3e-5 通吃（短对 ≤4.7e-3、122K 对 ≤5.5e-4） | `api128k_dpo.log` / `probe_dpo_short_lr3e5.log` |
| **显存判据用锚点插值，不用解析式** | 6 个实测锚点 26.2 / 27.7 / 29.8 / 33.0 / 35.3 / 47.5 GiB @ 2K/8K/16K/27.6K/32K/48K，区间线性、末锚点外 ×1.10；二次式在 40K 以上会低估 4~6 GiB，把必爆判成能过 | `mem_guard.py`（唯一出处） |

### C.4 §5 里没展开的两行

| 项 | 数 |
|---|---|
| 身份训练的署名 logp（`xiezhipeng-git` 紧跟 `你是谁`） | 300 步 −58.31 → −58.02（**+0.29**）；900 步 −58.88 → −51.09（**+7.79**） |
| 数学 L1 acc（全量 243 项） | baseline 100% / mezo 100% / bwd **93.0%** / dpo 100% |

后一行是「反传把模型打歪」的补强：不只是 L5 掉到 60%，连最简单的 L1 也掉 3 项。
