# 审核请求：训练功能（sft / dpo / grpo × 零阶 / 反传）

**当前工作树**：`feat/train-backforward`（目录名 `Strata-prtrain` —— 这里的 `pr` 是 PR，不是 pretrain；本 PR **没有预训练功能**）
**集成分支**：`feat/train_support`（下划线；本地与 `origin` 都有）
**目标**：`upstream/main`（Niko1221/Strata）
**合并顺序**：`feat/train_support` 先同步 `upstream/main`（当前落后 2,755 个提交），再并入 `feat/train-mezo`，最后落本功能的 commit —— 拓扑与理由见 §3.4
**范围**：`training/`（本分支新增目录）+ `serve/server.py` 的训练会话接口
**状态**：⚠️ 工作区未提交。审核前需先固化成 commit（见 §3.4 的分组与卫生要求）

| 项 | 数 |
|---|---|
| 已提交（相对 `upstream/main`） | `training/` 40 个文件 / +9,274 行 |
| 未提交（本功能的实际主体） | 19 个跟踪文件 +3,748 / −503，另有 52 个待提交的新文件 |
| 核心实现 | `train.py` `mezo.py` `objectives.py` `mem_guard.py` `sft.py` `grpo.py` `algos.py` 等 13 个文件 4,789 行 |
| 测试 | 13 个 `test_*.py` 2,882 行，训练目录 **239 项全过**；`serve/test_training_api.py` 6 项 |
| 服务端 | `serve/server.py` +258 行（`TrainBridge` + 3 个路由） |
| 新增依赖 | 零阶 **0 个 pip 包**（只需已装的 numpy/jinja2/regex + 一个引擎二进制）；反传 4~6 个（torch / transformers / accelerate / safetensors，Windows 还要 triton-windows、kernels）→ §2.5 |

**建议阅读顺序**：§0 结论 → §2.5 两条路线的依赖与 §2.6 出处 → §4 需要你判断的 8 个决策 → §5 证据（**§5.2 是你的机器上"能训多长、多快、什么情况训不了"**）→ §9 逐条问题。逐项实测细节在 `BRANCH-NOTES.zh.md`（同目录，不重复粘贴）。

---

## 0. 一句话说明

给 125B 模型加了**训练入口**：一条命令 `train.py --algo {sft,dpo,grpo} --method {mezo,backward}`，
六个组合共用**一份损失数学**，在单张 48 GiB 卡上零阶能吃到 **131072 token 满上下文**，
同时把训练挂进常驻服务（`POST /v1/training/*`），推理引擎在会话期让位、会话结束后自动恢复。
**资源口径**：零阶在这一张卡上就是"资源充足"（26.9 / 48 GiB，换更大的卡不会更快，因为 131072 已被模型封顶）；
反传才是被显存卡住的那一侧（9728 贴顶，吞吐 10.6 tok/s）。三档判据与 24~192 GiB 的推算见 §5.2。

**这个 PR 不做什么**（请据此收窄审核面）：

- 不改引擎 kernel、不改推理数值路径（无会话请求的处理路径零改动）；
- **引擎侧没有反传**，`--method backward` 走的是本地 PyTorch/HF autograd（`sft.py`/`grpo.py`），
  这是两条物理上不同的路线，不是同一实现的两种开关；
- **零阶路线不新增任何 pip 依赖**（不需要 torch），反传路线所需的 torch/transformers 等
  是可选档，不进 `requirements.txt` —— 两条路线分别要装什么见 §2.5；
- 不改 `third_party/llama.cpp`，不新增 C++ 依赖；引擎二进制用已有的构建流程产出。

---

## 1. 动机（审这个 PR 要解决的问题）

1. **长上下文训不了**：原有 `sft_mezo.py`（PyTorch 零阶）在 55,808 token OOM，
   单笔 `[T,T] int64` 就要 24.9 GB —— 128K 完全不可及。
2. **损失数学有三份**：`sft.py` / `grpo.py` / `sft_mezo.py` 各写一遍 CE/DPO/GRPO，
   sum 与 mean 口径混用（旧 sum 口径在 124K 上 pj = −4.03e+04，两步把 ‖θ‖ 推高 92%、生成乱码）。
   同一次训练换实现就换个数，无法互相对照。
3. **只能在命令行跑**：没有服务化入口，训练和推理抢同一张卡时没有任何仲裁
   （曾出现双驻留把提交内存打到 89.7 GiB > 物理 48 GiB，引擎 stall 自杀）。

对应三条交付：零阶放到**引擎前向**上做、损失收敛到 `objectives.py` 一处、
训练做成**服务里的会话**（显式让位 + 独占检查）。

---

## 2. 交付的能力

### 2.1 六个组合与实现映射（`algos.py` 的 `ALGOS × METHODS`，唯一出处）

| algo | 目标函数 | `--method mezo` 实现 | `--method backward` 实现 | 默认 lr（mezo / backward） | 额外数据 |
|---|---|---|---|---|---|
| sft | `ce`（answer 段逐 token） | `mezo.py` `SftMezo` | `sft.py` | 3e-5 / 1e-4 | — |
| dpo | `dpo`（序列 margin 的 −logσ(β·margin)） | `mezo.py` `DpoMezo` | `sft.py` | 3e-5 / 5e-5 | `--dpo-data` |
| grpo | `grpo`（组内归一化优势 × 裁剪比） | `mezo.py` `GrpoMezo` | `grpo.py` | 1e-5 / 1e-4 | `--rollouts` 或引擎在跑 |

- `DEFAULT_METHOD = "mezo"`；`--method` 省略即零阶。
- 更新那一半做了统一，但**有边界**：反传线（`sft.py`）与已退役的 PyTorch 零阶线（`sft_mezo.py`）
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

### 2.3 服务端训练会话

| 接口 | 行为 |
|---|---|
| `POST /v1/training/session {"op":"open","argv":[...]}` | 卸载常驻推理引擎 → 用调用方 argv 拉起训练引擎 → 等 READY（上限 300 s）；已有会话 → 409 |
| `POST /v1/training/command {"line":"..."}` | 转发到训练引擎 stdin，按前缀收回复；无会话 → 409 |
| `POST /v1/training/session {"op":"close"}` | 关训练引擎，**下一个请求自动把服务引擎拉回来** |

- 会话期所有推理路由 409；**无会话时推理路径完全不变**（`training_active()` 一个布尔判断）。
- 训练引擎死亡 / 启动失败 → 会话自动关闭并回到未激活态（单测覆盖 4 种死亡路径）。
- 为什么"让位"而不是"双驻留"：48 GB 放不下引擎 + HF 训练栈同时驻留（见 §1 第 3 条）。

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
| **引擎分支要求** | ⚠️ **必须是 `feat/train-mezo`（`Strata-mezo` 树）或 `feat/score` 构建出来的**：`LORA`/`SCORE` 只在这些分支的 `src/program/generate.cpp` 里。`upstream/main`（`c62d5e3a`，2026-10-31，`git ls-remote` 核对过远端就是这个）和本分支 `HEAD` 的 `src/` 里**都没有** | 同左（GRPO 第一段） |
| 数据 / 资产 | pack：gguf shards + `tokenizer/chat_template.jinja` + `expert-profile.bin` | 同左（另加 NF4 专家 pack） |
| 平台 | linux / win32（`platform_check.SUPPORTED`；win32 走 `preadv_shim` 的 readinto 垫层）；macOS 未测、被拒 | 同左 |
| 实测量级显存 | 26.9 GiB 账面（§5.1） | 34~45 GiB（§5.1） |

一句话：**零阶是"装好的环境 + 一个自己编译的引擎"就能跑，反传是"另装一套 PyTorch 栈"才能跑。**
两者不共享 Python 依赖，这正是本 PR 把它们做成两条路线、而不是一个开关的理由（D1）。

> 引擎怎么来：`STRATA_ENABLE_CUDA` 默认是 **OFF**（`CMakeLists.txt:80`），必须显式打开，
> 写成 `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
> -DCMAKE_CUDA_ARCHITECTURES=<你的卡> -DSTRATA_GGML_DIR=third_party/llama.cpp` 再加 `cmake --build`。
> 架构号与后端开关照 `docs/`（如 `docs/DEEPSEEK41.md`、`docs/AMD_HIP.md`）里同类模型的写法；
> 训练只需这个二进制，不需要额外编译目标。

> 说明：`requirements.txt` 里没有 torch —— 这是刻意的，CI 的 no-torch 档（`test_train_core.py`）
> 用只有标准库+`requirements.txt` 的环境当合并门。反传档单独装 torch 才跑（`.github/workflows/training-test.yml`）。
>
> ⚠️ **合并顺序是有依赖的**：`training` 只带 Python；零阶要的 `LORA`/`SCORE` 命令循环
> 目前只在 `feat/train-mezo` / `feat/score` 的 `src/program/generate.cpp` 里，`upstream/main` 没有。
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
  `LORA` / `SCORE` 与 `--serve` 命令循环**尚未进 `upstream/main`**，只在 `feat/train-mezo` / `feat/score`
  两棵树上（见 §2.5 的引擎分支要求）：`upstream/main` = `c62d5e3a`（2026-10-31），
  `git show upstream/main:src/program/generate.cpp` 里没有任何 `LORAOK`/`SSUM`。

### 2.7 产物

LoRA adapter 存取（`lora.py`）→ `lora_to_gguf.py` 转 GGUF 供引擎 `LORA` 命令加载（`LORA-MERGE.md`）。

---

## 3. 改动面：文件 → 职责 → 请重点看什么

### 3.1 核心实现（新增）

| 文件 | 行 | 职责 | 审核重点 |
|---|---|---|---|
| `objectives.py` | 307 | sft / dpo / grpo 三个目标函数的**唯一实现**，numpy 与 torch 双路径 | 双路径是否逐值一致；`reduction="mean"` 的归一化基数（token 数还是样本数）；GRPO 的 mask 与 advantage 广播 |
| `mezo.py` | 931 | 引擎桥（`Engine` 子进程 / `EngineApi` HTTP）+ `EngineMezo` 基类 + 三个子类 | θ±ε 的落地方式（写 plus/minus adapter 再 `LORA` 换表，D4）；θ 表的形状必须全程不变；θ₀ 缓存与 `--resume` 的一致性 |
| `mem_guard.py` | 447 | 显存预算、锚点插值、`max_seq_len`、独占检查 | 估算式是否会低估（解析式在 >40K 低估 4~6 GiB，是本模块存在的理由） |
| `algos.py` | 168 | 注册表 + lr 分表 + 旧参数名报错 | lr 硬编码在表里是否该改成可覆盖配置（Q4） |
| `train.py` | 558 | 唯一入口、参数翻译、模型解析 | `--dry-run` 打印的 argv 与真实执行是否同源 |
| `sft.py` / `grpo.py` | — | 反传路线（PyTorch/HF autograd；DPO 也走 `sft.py`） | 与 `objectives.py` / `update_rules.py` 是否真共用同一函数而不是复制；`--ckpt` 重算的边界 |
| `update_rules.py` | 198 | `BackwardRule`（autograd + AdamW）/ `MezoRule`（θ±ε + SGD）/ `zo_direction()`，同一 `step(loss_fn)`。**使用者是 `sft.py` 与已退役的 `sft_mezo.py`；引擎零阶 `mezo.py` 不走它** | 两规则的 rel/动量语义与 `mezo.py` 内联的 ZO 循环是否仍一致；`zo_direction` 的"每个参数方向必须不同" |
| `dpo.py` | — | DPO 纯逻辑（偏好对校验、`dpo_margin`、batch 组装），无 torch，被 `sft.py`（反传）与 `sft_mezo.py`（零阶）共用 | 这里与 `objectives.dpo_loss` 的分工是否清晰，会不会又长出第二份 margin |
| `budget.py` / `guard.py` / `usage.py` | 229 / 135 / — | 预算计算（给定机器反推各方案能训多长）、训练护栏（防"10 条同构样本训坏无关能力"）、真实请求尺寸统计；前两个是独立脚本，`guard` 被 `sft.py` 引入 | `budget.py` 与 `mem_guard.py` 的判据是否重复（一个反推上限、一个判当前是否超） |

### 3.2 服务端（修改）

| 文件 | 行 | 审核重点 |
|---|---|---|
| `serve/server.py` +258 | `TrainBridge`（子进程 + READY 握手 + stdin/stdout 转发）、`training_session()`、`training_command()`、推理路由的 409 闸门 | 锁与状态机（`op=open` 与 `close` 并发）；引擎死亡时锁是否释放；unload 失败是否把服务留在半死状态 |
| `serve/test_training_api.py` +6 项 | 用假引擎覆盖 open/command/close、坏 argv、无会话、引擎中途死亡、启动失败、unload 与恢复 | 假引擎是否可能掩盖真实的 READY 时序问题 |

### 3.3 测试与文档

- `test_objectives.py`（双路径对拍）、`test_mem_guard.py`、`test_update_rules.py`、
  `test_bench_limits.py`、`test_engine_api.py`、`test_judge_limits.py`、
  `test_prep_eval_data.py`、`test_run_plan.py`、`test_curriculum.py` 为新增；
  `test_train.py` / `test_train_core.py` 大幅扩充。
- 文档：`README.md`（命令面与参数）、`DEMO-IDENTITY.md`（身份训练手顺）、
  `PROTOCOL-FULLCTX-EVAL.md`（满上下文评测协议）、`BRANCH-NOTES.zh.md`（实测账）、
  `LORA-MERGE.md`（adapter → GGUF）、本文件。

### 3.4 提交卫生（**审核前必须处理**）

**先对齐分支拓扑**（2026-11-02 实测，`git rev-list --left-right --count`）：

| 分支 | 相对关系 | 实测结论 |
|---|---|---|
| `feat/train_support` | 13 提交 / 落后 `upstream/main` **2,755** 提交 | 集成分支，但基线太旧；**不能直接当 PR base**（以它为 base 的 diff 是 139 文件 +19,745，混了 2,755 个上游提交） |
| `feat/train-backforward` | 相对 support：**落后 0 / 领先 2,756**，merge-base 就是 support tip | 它**完整包含** support，但去掉 merge 与上游提交后**自有提交 = 0** ⇒ 实际是"support + 同步 upstream/main"的同步分支，**没有反传专属提交** |
| `feat/train-mezo` | 相对 support：**缺 support 的 13 个提交**（merge-base 更早，`296b59d2`） | 多的是引擎侧 `LORA`/`SCORE` + `train_engine_mezo.py`（445 行）；**从没合回 support** |

→ 正确顺序：support 先 merge `upstream/main`（就是 backforward 现在那个 merge 提交）→ 再 merge
`feat/train-mezo` → 最后把本功能的 commit 落上去 → PR base 用 `upstream/main`。
**本功能现在全部是工作区未提交状态，压在 backforward 上**，不在 support 上。

工作区里有 25 个 `_*` 前缀的临时脚本/中间 json（`_engine_grpo.json`、`_grpo_mask.json`、
`_probe_*.py`、`_run_matrix_smoke.sh` …）、`_engine_mezo_work/`（3 个 `.gguf`，39 MB）、
`_discard/corpus.npy`、`lora_mezo.pt`（26 MB）、`grpo.py.bak`。
**这些不应进 PR**，建议加 `.gitignore`（`_*`、`*.pt`、`*.bak`、`*.gguf`）。

建议拆 commit（便于逐块审）：

1. `objectives.py` + `test_objectives.py`（损失单一出处，可独立审）
2. `mezo.py` + `test_engine_api.py`（引擎零阶）
3. `algos.py` + `train.py` + 反传三脚本（命令面）
4. `mem_guard.py` + `budget.py` + `guard.py`（显存与护栏）
5. `serve/server.py` + `serve/test_training_api.py`（服务会话）
6. 文档与评测数据（`*.md`、`bench_*.json`、`data_*.jsonl`）

> 合并前的另一处风险：两棵树的工作区已经漂了 —— `mezo.py` / `objectives.py` / `mem_guard.py`
> 逐字一致（md5 相同），但 **`algos.py` / `train.py` / `update_rules.py` 不同**。
> 合 `feat/train-mezo` 之前要先确定这三份以哪棵树为准，否则同一份逻辑会出现两个版本。

---

## 4. 需要你判断的 8 个决策

每条给「选择 / 被否的替代 / 依据 / 风险」，冲突时请直接推翻，我按你的判断改。

**D1 零阶放引擎、反传放 HF —— 而不是引擎侧也实现反传**
- 依据：引擎是推理栈，128K 上零阶两次前向 63.8 s；反传要激活 + 每层 T² mask，同卡只到 9,728。
- 风险：两条路线**数值不可能逐值相同**（CE 差 0.0075 nats，见 §5.5），评审要接受"同目标、不同算子路径"。

**D2 默认 `--method mezo`** 
- 依据：显存、速度、held-out 回退都占优（§5.1 + §5.2 + §5.4）；反传的唯一优势是"4 题里学得快"。
- 风险：新人直觉是"反传更准"，默认零阶需要文档解释，否则会被当成降级。是否该强制显式传 `--method`？

**D3 损失单一出处，`objectives.py` 内 numpy / torch 双路径**
- 依据：引擎零阶只拿到 logp 的 numpy 数组，反传要 torch 张量走 autograd，两条路都要。
- 替代（被否）：只写 torch，零阶侧转 numpy —— 会让零阶被迫装 torch，失去"引擎侧纯 numpy"的独立性。
- 风险：双路径天然有漂移风险，靠 `test_objectives.py` 逐值对拍兜；请判断这个测试的容差够不够。

**D4 θ±ε 用「写两份 adapter + `LORA` 换表」实现，不给引擎加零阶专用命令**
- 做法：每步把 θ+ε·z、θ−ε·z 各写成一个 adapter GGUF（`--work` 下的 `plus.gguf` / `minus.gguf`），
  再 `LORA <path>` 换表、`SCORE <ids>` 打分（`mezo.py` 的 `swap()`）。方向 z 由 seed 再生，不存盘。
- 依据：引擎侧因此只需要 `LORA` / `SCORE` 两个命令（目前只在 `feat/train-mezo` / `feat/score`，
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
- 依据：48 GB 装不下两份权重（§1 第 3 条）；且训练引擎的 argv（`--no-prefill-borrow`、prompt chunk）
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
   收益是 0（区间早被 131072 封顶）。反过来，已退役的 torch 零阶（`sft_mezo.py`）**同一张卡同一方法**
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

### 5.3 身份训练：训练前后

12 条问法（中英各半），ctx 4096、seed 0，指标 `answer-ce` = 12 条答案段 nats/token。

| 配置 | answer-ce | 相对 before | 自由生成（问她是谁） | 判定 |
|---|---|---|---|---|
| before | 4.4333 | — | 通义千问（Qwen），由阿里巴巴通义实验室开发 | 基线 |
| 3e-5 × 300（跑 A / B） | 4.2696 / 4.1451 | −0.164 / −0.290 | 仍是 Qwen 身份 | 在学，无副作用 |
| **3e-5 × 450（默认）** | **4.2387** | **−0.196** | 中文身份正确，但丢掉实验室细节 | **默认值** |
| 3e-5 × 600 | 4.0090 | −0.426 | 只说"我是 AI 助手"，丢 Qwen 身份 | ❌ |
| 3e-5 × 750 / 900 | 3.8270 / 3.9445 | −0.608 / −0.490 | 推理段泄漏英文 | ❌ |
| 1e-5 × 300 | 4.3949 | −0.025 | 正常 | 学得慢 6.5× |

**必须写进审核结论的三条**：

1. **CE 在降，但训练目标里的事实在自由生成里没落地**：12 条数据 9 条含"由 xiezhipeng-git 微调"，
   900 步也一次都没生成这个署名；强制读法只 +7.79 nats（−58.88 → −51.09）。
   零阶单步更新信噪比 ~1/√d（方向方差 3.0e+02、|mean|/std 0.15），背下长答案里的专名要上千步 ——
   **这是方法的账，不是 lr 没调好**。请判断：这个 PR 是否应该把"身份训练"表述为
   "管线可用、署名未落地"，而不是"身份训练成功"。
2. **CE 不单调**：450 比 300 高 0.09；同配置两次 300 步差 0.12 nats（跨进程抖动）。
   **小于 ~0.1 nats 的差不当结论**，所以默认 450 的依据是"最后一次行为正常"，不是 CE 最优。
3. 训练前的 `gen` 没有落盘，表里 before 行按 ce 反推（lr 3e-5 第 100 步 ce 反升 0.05，
   该步输出等价于未训练基线）。**这是本 PR 的证据缺口之一**（§8 第 1 条）。

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
干净子集 baseline 已 100% ⇒ 1 项 = 0.42 pp，**任何下降都是回退信号**，不是"提升空间"。

> 口径：mezo adapter = `_run_hardmth/final.gguf`（500 步，日志 `_run_hardmth_train.log`）；
> bwd / dpo 两个 adapter 只留产物与评测报告，训练超参未落盘（§8 第 3 条）。同一 adapter 自检 2/4、
> evalscope 3/4，差别在判定口径（逐字 vs 数值等价），本文用 evalscope。

### 5.5 正确性对拍

| 项 | 结果 |
|---|---|
| 引擎 score vs HF CE（同一 ids，8 token） | −56.6508 vs **−56.6433**，差 0.0075 nats |
| `objectives.py` numpy vs torch | `test_objectives.py` 逐值对拍通过 |
| 训练目录单测 | **239 项全过**（`python -m unittest discover -p "test_*.py"`，22.5 s） |
| serve 全量 | 1,304 项，仅 `test_winjob` 1 个既有环境性 error（干净 worktree 同样失败） |
| 服务会话真机 | 会话期 chat 409 → close 后 200；客户端崩溃也会话正常关 |

---

## 6. 请照这个顺序验（V0~V4 不需要 GPU）

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

# V3 训练目录全量（无 GPU，239 项，~22 s）
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
**V5/V6 要先有 §2.5 那个引擎**（含 `LORA`/`SCORE`，即 `feat/train-mezo` / `feat/score` 构建出来的）；
拿 `upstream/main` 构建的二进制跑，V6 会卡在等 `LORAOK`。

---

## 7. 兼容性与回滚

- **推理路径**：无训练会话时只多一个 `training_active()` 布尔判断，128K 走廊、KV 档位、
  logprobs 行为均不变（serve 全量 1304 项为证）。
- **旧入口**：`--objective` / `--backend` 保留为**报错**（不是别名），避免两种叫法长期并存。
  若下游脚本在用旧名，这是不兼容变更，**请确认可以断**。
- **回滚**：本 PR 全是新增文件 + `serve/server.py` 的独立段落，`git revert` 单 commit 即可；
  不涉及数据迁移、不改 gguf、不改配置文件格式。
- **合并顺序**：零阶依赖的 `LORA`/`SCORE` 命令循环只在 `feat/train-mezo` / `feat/score`，`upstream/main`
  还没有（§2.5）。所以这个 PR 不能"先合脚本"，要么引擎先合，要么带上引擎那段（Q9）。
- 合并注意：`mezo.py` / `objectives.py` / `mem_guard.py` 在 `Strata-mezo`（feat/train-mezo）
  与 `Strata-prtrain` 两树各一份，**逐字一致**，合并时两树都要同步（同步一律 merge，不 rebase）。

---

## 8. 已知限制（明确不在本 PR 承诺范围内）

1. **默认 450 步没有专用跑，训练前 `gen` 也没有落盘**：450 的证据来自 900 步长跑里的检查点，
   before 行是按 ce 反推的。要写成"默认值验收"需补一次 450 步并同时记录训练前生成的对照。
2. **16K~64K 的零阶步时未测**，只有显存锚点；速度只有 8K（4.39 s）与 128K（63.8 s）两点。
   所以"中间长度多快"目前只能按插值估，不作承诺。
3. **bwd / dpo 两个 adapter 的训练超参与日志未落盘**（只剩产物 + 评测报告）。
   §5.4 的"反传过拟合"结论成立，具体 lr/步数需重跑确认。
4. **grpo on-policy 只有反传线**；引擎侧 grpo 目前只吃回放轨迹。
5. **短序列跨进程抖动未定位**：同配置两次 300 步 CE 差 0.12 nats（同进程重复性 std 0.005，不是瓶颈）。
   因此跨进程不比绝对值，只比同一进程内的差分。
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

---

## 9. 请逐条回答（我按你的答复改）

| # | 问题 | 我的倾向 |
|---|---|---|
| Q1 | 接受"引擎无反传、backward 走本地 HF 栈"这条边界吗？还是要引擎侧也实现反向？ | 接受（引擎是推理栈） |
| Q2 | `objectives.py` 的 numpy/torch 双路径，是否该合并成单实现 + 薄适配层？ | 保留双路径（零阶不装 torch） |
| Q3 | 默认 `--method mezo` 是否合理，还是强制显式传 `--method`？ | 保留默认 mezo + 文档说明 |
| Q4 | `algos.py` 里 lr 按（算法×方法）硬编码，是否改为可覆盖配置（env / yaml）？ | 先硬编码，等有第二组工作点再说 |
| Q5 | `/v1/training/*` 挤在 `serve/server.py`（+258 行），是否拆成 `serve/training.py`？ | 倾向拆，但会动 import 面 |
| Q6 | 会话期推理返回 409 而非排队，可接受吗？ | 可接受（训练时推理语义不明确） |
| Q7 | 测试够不够：训练目录 239 项 + serve 6 项，但没有端到端（真跑一步再断言 CE 下降） | 缺 e2e，值得补 1 项 |
| Q8 | `bench_*.json` 这类实测结果文件是否随 PR 提交，还是只留文档引用？ | 提交（它们是 §5 的原始数据） |
| Q9 | **合并顺序怎么走**：零阶依赖的 `LORA`/`SCORE` 命令循环还在 `feat/train-mezo`/`feat/score`，`upstream/main` 没有。是先单独提一个引擎 PR（带 `LORA`/`SCORE` + 测试），训练脚本排它后面？还是两个一起提？ | 先引擎、后脚本，各自可独立回滚 |
| Q10 | README 里"engine side, all merged upstream"这句与事实不符（见 §2.5），本 PR 里一并改掉可以吗？ | 改（文档不能留着错的事实） |

---

## 附录 A：证据出处

| 数字 | 文件 |
|---|---|
| 六方案极限、bwd 实测点位 | `bench_limits.json`、`bench_budget.json` |
| 换卡推算（表 B） | `python bench/budget.py --vram {24,32,48,80,96,141,192}` 的输出；模型在 `mem_guard.estimate_peak_gib` / `max_seq_len`（6 个锚点，残差 ≤ 0.6 GiB） |
| 三档判据与"不可训练"四类 | `mem_guard.require_budget` / `_limit_gib` / `NO_CKPT_FACTOR` / `MOE_STAGING_GIB` |
| 三档实测（表 A、表 C） | `bench_engine_128k.json`（零阶 8K / 128K）、`bench_limits.json`（反传四点）、`logs/api128k_*.log`（128K×三算法）、`logs/accept_lr3e5_s900.log`（ctx 4096，900 步 588 s） |
| 8K / 128K 引擎速度与显存 | `bench_engine_128k.json`、`logs/fullctx_probe_128k.log` |
| 128K × 三算法步时与 rel | `logs/api128k_{sft,dpo,dpo_default3e5,grpo}.log` |
| 身份前后与 gen 原文 | `logs/accept_lr3e5_ce.log`、`accept_lr3e5_s900.log`、`accept_lr1e5_ce.log` |
| 数学训练集前后 | `Strata-mezo/training/_run_hardmth_train.log` |
| MATH-500 四次评测 | `Strata-mezo/training/_evalscope/{baseline,mezo_lora,bwd_lora,dpo_lora}/` |
| 单测逐条 | `logs/test_results_full.txt`（239 项） |
| 依赖清单 | `requirements.txt`（零阶够用）、`.github/workflows/training-test.yml`（no-torch 合并门）、`training/README.md` §Windows environment notes（反传实测组合） |
| 平台支持矩阵 | `platform_check.py`（`SUPPORTED`、`preadv_shim.py`） |
| MeZO 方法与官方仓库 | arXiv 2305.17333；<https://github.com/princeton-nlp/MeZO>（本 PR 未拷贝其代码，见 §2.6） |

## 附录 B：这份文档与旧稿的差异

`BRANCH-NOTES.zh-DRAFT.md`（v0.2）是给内部看的"做了什么"，不是审核请求。本文件重写为审核口径，并且：

| 项 | v0.2（错/缺） | 本版 |
|---|---|---|
| 450 步 answer-ce | 4.145（−0.29） | **4.2387**；4.145 是另一次跑的第 300 步 |
| 零阶峰值显存 | 48.1 GiB | **46.97 GiB**（48,100 MiB） |
| backward 路线 | "引擎反传" | **PyTorch/HF autograd**（引擎没有反传） |
| 测试总数 | 232 | **239**（已重跑落盘） |
| 六方案极限表 | 缺 | §5.1 |
| 资源充足 / 不足 / 不可训练 | 无（只有一句"48 GB 装不下两份权重"） | **§5.2**：三档判据 + 48 GiB 实测三档 + 24~192 GiB 推算 + 四类"不可训练"症状 + 折算墙钟 |
| 数学训练前后 + 非训练集 | 只有旧过程式 logp | §5.4（0/3→2/3 + held-out −3/−31/−2） |
| 审核者要回答的问题 | 无 | §9（Q1~Q8） |
| 提交卫生 | 无 | §3.4（25 个 `_*`、65 MB 二进制、`.bak`） |
| 两条路线的依赖 | 无 | §2.5（零阶 **0 个 pip 包** + 引擎二进制；反传 torch/transformers/accelerate/safetensors，Windows 另加 triton-windows、kernels） |
| 零阶方法出处 | 无 | §2.6（MeZO，arXiv 2305.17333，princeton-nlp/MeZO；未拷贝代码） |
| 引擎分支要求 | 无（v0.2 假定"已合入上游"） | §2.5 / Q9：`LORA`/`SCORE` **只在** `feat/train-mezo`、`feat/score`，`upstream/main` 没有 |

**逐项实测细节**（每次跑的完整日志行、抖动、口径争议）见同目录 `BRANCH-NOTES.zh.md`。
