# 审核请求：训练功能（sft / dpo / grpo × 零阶 / 反传）

**交付分支**：`feat/train_support`（下划线；`origin` 上的同名分支停在 `7fa6caaa`，需要一次 fast-forward 推送）

> **▸ 本文所在树 = 交付面**（三条交付形态里的第 3 条）：`feat/train-mezo` ∪ `feat/train-backforward`
> 加这组审核文档。代码与 `feat/train-backforward` **逐字相同**（`git diff -- '*.py'` 为空），只有文档不同，
> 所以审代码时三条分支给出来的 diff 是一致的，只有文档与文件集合不同。

**目标**：`upstream/main`（Niko1221/Strata）
**基线**：`upstream/main` = `c62d5e3a`（2026-10-31）；`git ls-remote upstream main` 复核远端一致，**落后 0 个提交、上游文件 0 个删除**，即改动是在最新主分支之上做的
**范围**：三块 —— ① `training/`（训练入口，新增目录；根放共用层，`mezo/`、`backward/` 是两条路线，`bench/` 基准与探针，`tests/` 单测）；② `serve/server.py` 的训练会话与 `/v1/score`、`/v1/lora`；③ **引擎侧 LoRA + SCORE 能力**（34 文件 / +3,621 −57：`src/kernels/lora*.{cu,cpp}`、`src/prefill/score*`、`generate.cpp` 的命令循环、`CMakeLists.txt` / `setup.py` 的开关、CI）
> 第 ③ 块不是可选的：零阶路线的前向就是引擎的 `SCORE`，没有它 `--method mezo` 一行都跑不了。
> 若上游要求训练功能与引擎能力分开审，按 Q9 拆成两个 PR 即可，代码本身已经是分开的提交。
**状态**：工作区干净，已全部落成 commit；训练目录 **258 项**（Windows 上 256 过 + 2 项 watchdog 仅因缺 triton；**Linux 实测 258 全过**，见 §7.1），`serve.test_training_api` **6 项**全过（本次复核）；零阶那一棵树 **134 项全过且不需要 torch**（两个平台都是 134）

| 项 | 数 |
|---|---|
| 改动面（相对 `upstream/main`） | 132 文件 / **≈+27k −93**（含本文档；口径见表下注） |
| 本功能提交（非 merge） | ≈110 个（`git rev-list --no-merges --count upstream/main..feat/train_support`；含早期训练线与本地 `main` 的既有提交） |
| 核心实现 | `train.py` `algos.py` `objectives.py` `mem_guard.py` `mezo.py` `sft.py` `grpo.py` 等 13 个文件 4,673 行 |
| 测试 | 14 个 `test_*.py` 3,214 行，训练目录 **258 项**（Windows 上 256 过，2 项 watchdog 缺 triton）；`serve/test_training_api.py` 6 项 |
| 服务端 | `serve/server.py`（`TrainBridge` + 3 个 `/v1/training/*` 路由 + `POST /v1/lora`） |
| 新增依赖 | 零阶 **0 个 pip 包**（只需已装的 numpy/jinja2/regex + 一个引擎二进制）；反传 4~6 个（torch / transformers / accelerate / safetensors，Windows 还要 triton-windows、kernels）→ §2.5 |

**上表的数与漂移规则**：文件数、行数、提交数取自当前 HEAD，并按 1k 行（向下取整）/ 10 个提交（就近取整）取整 —— 每加一个文档提交会多 1 个提交、约 275 行，属预期漂移，不涉及代码。真正不变量是这些：`git diff -- '*.py' feat/train-backforward feat/train_support` 为空，`comm -23 <mezo> <bwd>` 为空（bwd 只多 32 个 `.py`）。

**分支与合并顺序**：拓扑、基线判据（为什么拿 `origin/main` 比会误判）、来源分支审计、LoRA 三层归属，全部在 **§7**。

**建议阅读顺序**：§0 结论 → §2.5 两条路线的依赖与 §2.6 出处 → §4 需要你判断的 8 个决策 → §5 证据（**§5.2 是你的机器上"能训多长、多快、什么情况训不了"**）→ §7 分支与合并顺序 → §9 逐条问题。逐项实测明细在 **附录 C**（原始点位与口径）。

---

## 0. 一句话说明

给 125B 模型加了**训练入口**：一条命令 `train.py --algo {sft,dpo,grpo} --method {mezo,backward}`，
六个组合共用**一份损失数学**，在单张 48 GiB 卡上零阶能吃到 **131072 token 满上下文**，
同时把训练挂进常驻服务（`POST /v1/training/*`），推理引擎在会话期让位、会话结束后自动恢复。
**资源口径**：零阶在这一张卡上就是"资源充足"（26.9 / 48 GiB，换更大的卡不会更快，因为 131072 已被模型封顶）；
反传才是被显存卡住的那一侧（9728 贴顶，吞吐 10.6 tok/s）。三档判据与 24~192 GiB 的推算见 §5.2。

**这个 PR 不做什么**（请据此收窄审核面）：

- **新增引擎能力，但不改推理数值路径**：加的是 `LORA <path>`（原地换 adapter 表）与
  `SCORE <ids>`（teacher-forced 逐位置 logp）两条命令，以及它们自己的 kernel
  （`src/kernels/cuda/lora.cu` 477 行、`src/kernels/cuda/logprobs.cu` 188 行、
  `src/prefill/score.cpp` 594 行）。`generate.cpp` 的改动集中在 `LORA` / `SCORE` 两条命令分支、
  adapter 换表时的会话缓存失效（`conversations` / `checks` / `live_clear`）与 `usage()`，
  **没有重写 GEN 的采样或前向路径**；不过「GEN 与上游逐值一致」这件事**没有做过对拍**，
  别把它当已验证结论。**归属注意**：这 477 行里 **421 行属于尚未合入的上游 PR #3267**
  （Alexey Rogov），我们自己的是 133 行符号翻转 + `lora_adapters.cpp` 的换表 —— 三层见 §7.3；
- **引擎侧没有反传**，`--method backward` 走的是本地 PyTorch/HF autograd（`sft.py`/`grpo.py`），
  这是两条物理上不同的路线，不是同一实现的两种开关。**反传这一层是独立增量，而且现在按文件就能拆**：
  它比零阶树只多 32 个 `.py`（30 个 torch 文件 + 2 个诊断脚本），共用层与 `train.py` / `algos.py`
  两边逐字一致。只交 `feat/train-mezo`（`56c7eae0`）就是"共用层 + 零阶"的完整树，134 项全绿、
  训练路径零 torch（无 torch 环境下同样 134 过）；拆法与实测出的坑见 §7.4；
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
- **这张表是"两条线都在"的树（bwd / support）上的六个组合**。零阶分支（`feat/train-mezo`）只带
  左半边：`--method` 的可用项由文件决定（`algos.available()`），缺 `backward/sft.py` 时 `--list`
  会注明"不在这条分支上"，`--method backward` 在加载模型前退出并指路 —— 见 §7.1。
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
| **引擎分支要求** | ⚠️ **必须是 `feat/train-mezo`（`Strata-mezo` 树）构建出来的**：`LORA`/`SCORE` 只在这个分支的 `src/program/generate.cpp` 里。`upstream/main`（`c62d5e3a`，2026-10-31，`git ls-remote` 核对过远端就是这个）和本分支 `HEAD` 的 `src/` 里**都没有** | 同左（GRPO 第一段） |
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
  两棵树上（见 §2.5 的引擎分支要求）：`upstream/main` = `c62d5e3a`（2026-10-31），
  `git show upstream/main:src/program/generate.cpp` 里没有任何 `LORAOK`/`SSUM`。

### 2.7 产物

LoRA adapter 存取（`lora.py`）→ `lora_to_gguf.py` 转 GGUF 供引擎 `LORA` 命令加载（`LORA-MERGE.md`）。

---

## 3. 改动面：文件 → 职责 → 请重点看什么

**目录结构**（`training/` 下按两条路线分，审的时候按这个找文件）：

```
training/
  train.py  algos.py  objectives.py  mem_guard.py  update_rules.py    # 入口 + 两条路线共用
  ce.py  dpo.py  datasets.py  guard.py  whoami.py  platform_check.py  # 共用件
  mezo/      零阶：mezo.py（引擎当前向）、sft_mezo.py（已退役，仅留 score parity）
  backward/  反传：sft.py（含 DPO / curriculum）、grpo.py、HF 加载栈（loader_qwen4 / nf4_fused / qsa_fast / lora …）
  bench/     基准与一次性探针：bench_fullctx、ab_*、mtp_*、budget、judge_limits …
  tests/     14 个单测（258 项；零阶树只带其中 7 个 = 134 项）
```

> 这些是**功能代码，不是示例**。原稿放在 `examples/training/` 名下不副实 —— 上游 `main` 根本没有 `examples/`，
> 仓库放 Python 工具的是 `tools/`、放服务的是 `serve/`，所以新建顶层 `training/`，两条路线各占一个子目录。

### 3.1 核心实现（新增）

| 文件 | 行 | 职责 | 审核重点 |
|---|---|---|---|
| `objectives.py` | 307 | sft / dpo / grpo 三个目标函数的**唯一实现**，numpy 与 torch 双路径 | 双路径是否逐值一致；`reduction="mean"` 的归一化基数（token 数还是样本数）；GRPO 的 mask 与 advantage 广播 |
| `mezo/mezo.py` | 931 | 引擎桥（`Engine` 子进程 / `EngineApi` HTTP）+ `EngineMezo` 基类 + 三个子类 | θ±ε 的落地方式（写 plus/minus adapter 再 `LORA` 换表，D4）；θ 表的形状必须全程不变；θ₀ 缓存与 `--resume` 的一致性 |
| `mem_guard.py` | 447 | 显存预算、锚点插值、`max_seq_len`、独占检查 | 估算式是否会低估（解析式在 >40K 低估 4~6 GiB，是本模块存在的理由） |
| `algos.py` | 168 | 注册表 + lr 分表 + 旧参数名报错 | lr 硬编码在表里是否该改成可覆盖配置（Q4） |
| `train.py` | 558 | 唯一入口、参数翻译、模型解析 | `--dry-run` 打印的 argv 与真实执行是否同源 |
| `backward/sft.py` / `backward/grpo.py` | — | 反传路线（PyTorch/HF autograd；DPO 也走 `sft.py`） | 与 `objectives.py` / `update_rules.py` 是否真共用同一函数而不是复制；`--ckpt` 重算的边界 |
| `update_rules.py` | 198 | `BackwardRule`（autograd + AdamW）/ `MezoRule`（θ±ε + SGD）/ `zo_direction()`，同一 `step(loss_fn)`。**使用者是 `sft.py` 与已退役的 `sft_mezo.py`；引擎零阶 `mezo.py` 不走它** | 两规则的 rel/动量语义与 `mezo.py` 内联的 ZO 循环是否仍一致；`zo_direction` 的"每个参数方向必须不同" |
| `dpo.py` | — | DPO 纯逻辑（偏好对校验、`dpo_margin`、batch 组装），无 torch，被 `sft.py`（反传）与 `sft_mezo.py`（零阶）共用 | 这里与 `objectives.dpo_loss` 的分工是否清晰，会不会又长出第二份 margin |
| `bench/budget.py` / `guard.py` / `bench/usage.py` | 229 / 135 / — | 预算计算（给定机器反推各方案能训多长）、训练护栏（防"10 条同构样本训坏无关能力"）、真实请求尺寸统计；前两个是独立脚本，`guard` 被 `sft.py` 引入 | `budget.py` 与 `mem_guard.py` 的判据是否重复（一个反推上限、一个判当前是否超） |

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
| 11 | 反传线 5 组（`backforward`） | `sft.py` / `grpo.py` / 加载与 kernel / `sft_mezo.py` 退役 / 探针 |

反传那 5 组在 `feat/train-backforward` 上，其余在 `feat/train-mezo` 上（共用层 + 零阶 +
引擎），`feat/train_support` 是两者的合并面。逐组看时用
`git log --oneline --no-merges upstream/main..feat/train_support`。

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
| 训练目录单测 | **258 项**（`python -m pytest training/tests -q`，22 s；Windows 上 256 过 + 2 项 watchdog 缺 triton，见 §7.1） |
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

# V3 训练目录全量（无 GPU，258 项，~22 s；零阶树 134 项）
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
| `feat/train-mezo` | `56c7eae0` | 共用层（numpy）+ 零阶（引擎 `LORA` / `SCORE`）+ 引擎侧改动 + 数学造数据/评测；**不含 torch 层** | **能**：运行时只要 numpy + 引擎二进制，训练路径不 import torch | **134 全过**（把 `torch` 顶成 raise ImportError 的模块后同样 134 过） | ≈80 提交 / 100 文件 / ≈+17k −93 |
| `feat/train-backforward` | `adfa53a7` | 在 mezo 之上叠加：`backward/`（10 模块）+ `ce.py` / `lora.py` / `update_rules.py` + `mezo/sft_mezo.py`（HF 对拍）+ 6 个 torch bench + 7 个 torch 测试 | **能**：mezo 那棵树是它的子集，只多 32 个 `.py` | 258 项（256 过 + 2 项 watchdog 见下） | ≈90 提交 / 132 文件 / ≈+27k −93 |
| **▸ `feat/train_support`**（本文所在树，交付面） | 交付分支 HEAD | = `mezo ∪ backforward` + 文档（两线共享的 `.py` 逐字一致；对 mezo 只多那 32 个文件） | 交付面 | 258 项（同上） | 交付分支 HEAD / 132 文件 / ≈+27k −93 |

**分层规则一句话**：三条分支的差异只体现在**文件在不在**，不体现在代码分支上。
`algos.available()` 读磁盘决定 `--method` 有哪几个，缺哪个就把缺的文件名说清楚；
`train.py` / `algos.py` 三条线逐字一致，所以合并永远不会为「dispatch 代码谁对」打架。

四条容易搞错的地方：

- **反传线是叠在零阶线之上的，不是并列**：`comm -23`（mezo 有而 bwd 没有的文件）为空；
  bwd 比 mezo 只多 **32 个 `.py`**（30 个 torch 文件 + `bench/diag_frozen.py` / `bench/mtp_probe_tr.py`
  两个诊断脚本），共享的 62 个 `.py` 逐字一致（`git diff --quiet` 逐个核过）。
  这是"官方只要一条线"能拆的前提：只要零阶交 mezo（100 文件、134 项、零 torch），
  要两条交 support；bwd 自己也能独立用，但它交付的价值是"反传 + 对拍"，零阶是它的基础层。
- **零阶线不带 torch 是判据，不是说法**：
  `git grep -nE '^(import|from) (torch|transformers|triton)' feat/train-mezo -- training/` 为空；
  再把 `torch` 换成一个 raise ImportError 的模块顶掉跑 `pytest tests`，仍然 134 过。
  `train.py → mezo/mezo.py` 的 import 闭包只有 `objectives.py` / `mem_guard.py`。
  摘掉的那一层（`backward/` 全部、`ce.py` / `lora.py` / `update_rules.py`、`mezo/sft_mezo.py`、
  6 个 torch bench、7 个 torch 测试）在 bwd 上原样存在，`56c7eae0` 一次提交完成摘除。
- **"默认"指的是方法，不是分支**：`algos.py` 里 `DEFAULT_METHOD = "mezo"`（三条分支一致），
  所以 `train.py` 不给 `--method` 时走零阶；这与选哪条分支交付是两件事。在 mezo 上
  `--list` 会写 `backward 不在这条分支上（缺 backward/sft.py）`，`--method backward` 在加载模型
  **之前**就退出并指路（`algos.method_of`），不是跑到一半 ImportError。
- **mezo 的"完整"以 import 图闭合为判据**：`mezo/sft_mezo.py` 曾 import 只在反传层的
  `mtp_draft`（`827d9071` 补齐）；`math_probe` / `math_build_data`（零阶数学实验的选题与造
  数据）和 `test_curriculum`（测的是 `lora.load_lora_state`）也长期落在反传层（`ae31b3f0` 下移）。
  本轮又把 `extract_num`（判答对/答错的唯一规则）抽到 `bench/math_extract.py`，让零阶的
  `math_build_data.py` 与反传的 `math_probe.py` 共用一份 —— 两条线各写一份正则，改一处忘一处
  就会让 GRPO 拿到错的优势。现在静态扫描"被 import 但在本树不存在"的本仓库模块 = **0**。
- `perf/pack-load-accel` 的 merge（`05d66edc`）**不带功能**：它唯一的独有提交 `2a640fc9`
  （`fadvise(DONTNEED)` + 分段计时）早就在 `06186a31` 里进了训练线，merge 只是补祖先，
  还顺带把同一段定义搬了第二份进来，已由 `330eba47` 回退。

**2 项 watchdog 测试的环境说明（别当成回归）**：`test_bench_limits` 的两个 `Watchdog` 用例会起子进程
`import bench_fullctx`，链路是 `loader_qwen4 → nf4_fused → import triton`；Windows 没有 triton 轮子，
所以在 Windows 的 pytest 里这 2 项失败（`ModuleNotFoundError: No module named 'triton'`），其余 256 项通过。
这四个相关文件（测试、`bench_fullctx.py`、`loader_qwen4.py`、`nf4_fused.py`）与上一版 `0b15d175` **逐字一致**，
不是本次改动引入的。**Linux 上实测 258 全过**（2026-11-03，WSL Ubuntu 24.04 / Python 3.12 / pytest 9.0.2，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q` → `258 passed in 35.76s`），所以这 2 项是环境依赖，不是代码问题。

```
upstream/main c62d5e3a
  └─ feat/train_support  7fa6caaa   早期 13 个训练提交（旧基点，见下方基线判据）
       └─ feat/train-mezo  fd493a06  共用层 + 零阶 + 引擎（LORA / SCORE）
            ├─ 3eb6e36d   sft_mezo 退役为 score-parity 检查（从反传层搬回零阶）
            ├─ 6c69bc19   + lora.load_lora_state（补齐零阶层缺的符号）
            ├─ d54d8a32   迁移 → training/
            ├─ 827d9071   + mtp_draft（sft_mezo 的 import 不再缺）
            └─ ae31b3f0   + math_probe / math_build_data / test_curriculum（213 项全绿）
                 ├─ 56c7eae0   摘掉 torch 层（213 → 134 项）；方法可用性改成看文件（algos.available）
                 └─ feat/train-backforward
                      ├─ 6e7a88ac   + torch 反传栈
                      ├─ 0c676f32   同一映射表迁移
                      ├─ b7085d3c   merge feat/train-mezo
                      ├─ b7dc511d   merge feat/train-mezo（239 项全绿）
                      └─ adfa53a7   merge 56c7eae0（保住 torch 层 → 258 项）
                           └─ feat/train_support
                                ├─ 05d66edc  merge perf/pack-load-accel（净变化为零）
                                ├─ a5a3f3af / 74118854 / f7aabccd   文档
                                ├─ 330eba47  回退 merge 带出的重复定义
                                ├─ 3b379c61  merge feat/train-backforward（此后只有文档提交）
                                ├─ a8645aea  merge feat/train-mezo（= 56c7eae0）
                                └─ c9c0016f  merge feat/train-backforward（= adfa53a7）
```

**为什么是「零阶打底 + 反传叠加」而不是三条并列**：共用层单独**不能跑** —— `train.py` 的注册表
要把算法映射到脚本，`--list` 与 `test_train_core.py` 都会问"这个方法有实现吗"。本轮的解法是
把这条判断改成**看文件**（`algos.available()`），于是共用层 + 零阶正好构成一棵能独立跑、
能自证的树，**mezo 承载共用层**；反传在它之上只加文件；support 是两者的并集。
审核时按 `upstream/main...feat/train_support` 看 diff 即可，上面这条链只解释提交是怎么分组进来的。

**三条分支都是能独立跑的树，不是只在交付面迁移**：`feat/train-mezo` 自己就是一棵 `training/`
的树（`56c7eae0`，134 项全绿，无 torch），`feat/train-backforward` 用**同一份映射表**再迁移一遍、
merge 零阶层、再叠 torch（`adfa53a7`，258 项），support 是两者合并的结果（`c9c0016f`）。判据如下：

- **官方只要零阶** ⇒ 直接交付 `feat/train-mezo`（`56c7eae0`）：100 文件、134 项、训练路径零 torch。
  不需要 cherry-pick，也不需要再把目录搬一次 —— 这正是"迁移只做在交付面"那版的缺口：
  合并后的树能拆，但源分支拆不出来。
- **只要反传**（不要引擎那套 `LORA`/`SCORE` 改动）⇒ 交付 `feat/train-backforward`（`adfa53a7`）：
  它就是零阶树 + torch 层，两者共享的 `.py` 逐字一致，所以摘掉零阶侧文件也是一个提交的事。
- **两条都要** ⇒ 交付 `feat/train_support`（HEAD）。审 diff 用
  `upstream/main...feat/train_support`，看到的就是 `training/`。
- 三条分支的迁移提交是同一批 rename（`d54d8a32` / `0c676f32`），路径逐一对齐，所以合并
  不会出现两份文件；实测 add/add 冲突只有两处（`sft.py`、`test_train.py`，`b7085d3c` 时取反传侧）
  与一处 `math_probe.py`（`c9c0016f` 时取反传侧的去重版本）。
- **文档版本**：完整说明是 support 上的这一份（823 行）。`feat/train-mezo` 与
  `feat/train-backforward` 各自也带一份 `REVIEW-REQUEST.zh.md`，是迁移时的 **594 行早期版**
  （没有 §5.2 资源三档、§7 分支审计、附录 C）。要评完整 PR 以 support 为准；只收零阶时，
  那份早期版覆盖的能力面与零阶层一致，不矛盾。

**基线判据（比对时别拿错参照）**：交付分支建在 `upstream/main c62d5e3a`（= `Niko1221/Strata`
的 main，2026-10-31，`git ls-remote` 核对过远端就是这个），support 相对它**落后 0、领先 ≈87**。
本仓库的 `origin` 是 fork（`xiezhipeng-git/Strata`），它的 `main` 停在 `80494788`（2026-10-15），
**落后上游 2,755 个提交** —— 老 `support 7fa6caaa` 那句"落后 2,755"就是因为当年从 fork 的
main 切出来的。所以比对一律用 `upstream/main...feat/train_support`，拿 `origin/main` 比会
误判成"基线过期"。

### 7.2 来源分支审计（这些分支的内容去哪了）

| 分支 | 是什么 | 状态 |
|---|---|---|
| `perf/pack-load-accel` | 载入加速：每块读完 `fadvise(DONTNEED)`，冷读 2.98→0.33 GiB/s 那个坑（28 GiB 9.4 s vs 85.8 s） | **已 merge**（`05d66edc`，无冲突）。内容此前已在树里（`06186a31`），merge 只补祖先、净变化为零；它带出的重复定义已由 `330eba47` 回退 |
| `feat/lora-sign` | adapter 的符号，换表不搬家 | 已纳入（mezo 的 31 个引擎提交之一，`e9173c16`；+56/+13/+64 行，**只服务零阶**的 θ−ε，见下方三层表） |
| `feat/lora-q36` | LP 行协议 + per-request LoRA | 已纳入 |
| `feat/score` | `SCORE <ids>`：逐位置 logp（`/v1/score`） | 已纳入 mezo（merge `53e729fa`，`merge-base --is-ancestor` 核过）。**不单独提 PR**：它技术上能独立（`score.cpp` 里 `lora`/`adapter` 0 命中，不依赖换表命令，单独给任意 ids 打分就能跑），但脱离训练只是"给一段 caller 指定的 ids 出逐位置 logp"，审核者第一个问题就是"谁用"；且它是零阶的度量核心（θ±ε 的 L 全靠它），拆开要两处维护 |
| `feat/kv-auto` | `--kv auto` 按上下文选精度 | 已纳入 |
| `feat/strata-rollout` | 返回生成序列与它续写的 prompt | 已纳入 |
| `pr-3267-lora` | 上游 PR #3267 的 LoRA kernel（`lora.cu` 421 行 / `lora_parity` 237 行），作者 Alexey Rogov | **内容已含**：support 是超集（477 / 301 行）。注意 **`upstream/main` 里没有 LoRA kernel**（`src/kernels/cuda/lora.cu` 不存在），这条 PR 至今未合入（分叉点落后 main 1,705 个提交），我们是本地拉的只读依赖，不是作者 |
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

1. **上游 `main` 只有 adapter 的加载路径**（`--lora` 参数、会话状态按 LoRA 开关缓存），**没有 kernel**。PR #3267 至今没合入（分叉点落后 `main` 1,705 个提交），所以这 421 行不是"上游代码"，是"一个尚未合入的上游 PR"，本仓库把它当**只读依赖**带着（`pr-3267-lora` 是 `feat/train-mezo` 的祖先，推 mezo 就自动带上了，不需要额外动作）。
2. **我们自己的 kernel 增量只有 133 行，且只服务零阶**：θ−ε 不是"再存一份表"，而是把同一张表按 −1 用，所以 `on` 从 bool 改成三态 int，sign 只落在 `add=1` 那一遍（落两遍会平方回 +1）。没有 MeZO，这 133 行没有任何意义 ⇒ **跟随 mezo**。
3. **HF 侧那三个脚本跟随反传**（`unilora.py` / `lora.py` / `lora_to_gguf.py`）：它们的活是"HF 训完 → 转成引擎能读的 adapter GGUF"，服务 `--method backward` 那条线。

**所以 `feat/score` 和 LoRA 都不是能独立交付的分支**：SCORE 拆出去没有使用者，LoRA 拆出去是别人的 PR。

> 真正适合独立提 PR 的是 **`perf/pack-load-accel`**（Linux 载入存档加速：每块读完 `fadvise(DONTNEED)`，28 GiB 冷读 85.8 s → 9.4 s）——它跟训练毫无关系，是纯性能修复，已在 `05d66edc` merge 进来（净变化为零，功能早就在 `06186a31` 里），审核方想单独先收这一段完全可以。

### 7.4 兼容性、回滚与合并顺序

- **推理路径**：无训练会话时只多一个 `training_active()` 布尔判断，128K 走廊、KV 档位、
  logprobs 行为均不变（serve 全量 1304 项为证）。
- **旧入口**：`--objective` / `--backend` 保留为**报错**（不是别名），避免两种叫法长期并存。
  若下游脚本在用旧名，这是不兼容变更，**请确认可以断**。
- **回滚**：本 PR 全是新增文件 + `serve/server.py` 的独立段落，`git revert` 单 commit 即可；
  不涉及数据迁移、不改 gguf、不改配置文件格式。
- **合并顺序**：零阶依赖的 `LORA`/`SCORE` 命令循环只在 `feat/train-mezo`，
  `upstream/main` 还没有（§2.5）。**本 PR 已经带上引擎那段**（`feat/train-mezo` 的 31 个提交，
  含 `LORA` / `SCORE` / `--kv auto` / 适配器符号），所以不必"先合引擎"；若上游希望分开审，
  仍可按 Q9 拆成引擎 PR + 脚本 PR。
- **反传那一层可以整层拆出 —— 已实测，而且现在是按文件拆**：拆出来的形态就是 `feat/train-mezo` 本身
  （`56c7eae0`，134 项全绿、训练路径零 torch）。bwd 只比它多 32 个 `.py`，共用层与 `train.py` /
  `algos.py` 两边逐字一致，所以"拆"既不用挑文件、也不用改代码分支：
  1. **拆之前先证明它能自己立住** —— 这一步实测出一个既存缺陷：`sft_mezo.py` 的 score-parity
     版本（`a177ad4b`）落在反传层，而 `feat/train-mezo` 的 `sft_mezo` 还是旧版，导致零阶分支上
     `test_train` / `test_bench_limits` **导入即失败**（`cannot import name '_Pair'`）。已把它和
     它依赖的 `lora.load_lora_state`（28 行）搬回零阶层（`3eb6e36d` + `6c69bc19`），零阶分支
     才真正自洽。**这不是本 PR 新引入的问题，是分支原来就有的** —— 不做"拆分演练"就看不见。
  2. **拆掉不能是静默降级**：早先零阶树里留的是 `backward/` 的**旧版薄壳**（`--method backward`
     能跑，只是没有 DPO / curriculum / clip window / TIS）—— 那是"能跑但能力缩水"，最难看出的一档。
     `56c7eae0` 改成整层摘除：`algos.available()` 只看脚本在不在，`--list` 写
     `backward 不在这条分支上（缺 backward/sft.py）`，`--method backward` 在加载模型**之前**就
     退出并指路。「静默降级」与「明确没有」的区别，就是审核者能否一眼看出交的是哪一档。
  3. **零阶的损失数学不再借反传层的测试来守**：`test_objectives.py` 是 numpy↔torch 对拍，随 torch
     层一起走了；补了 `test_objectives_np.py`（numpy 路径 18 项：CE 掩码 / DPO 饱和与长度归一 /
     组内优势 / GRPO 非对称裁剪 / DAPO vs seq），零阶 134 项里数学是有覆盖的。
  4. **耦合是层内的**：那 32 个文件 import 同目录的共用层（`objectives` / `mem_guard` /
     `update_rules` / `lora` / `ce`），而共用层在零阶树上；另有 `mezo.py` / `lora_to_gguf.py`
     横向 `sys.path.insert` 到仓库 `tools/` —— 训练目录不是自闭环。所以"拆"的正确含义是
     **交零阶那棵树**，不是从一棵树里抠掉几个文件。
  5. §5.4 的"反传过拟合"证据属于被拆走的那一层，裁剪版不引用它；零阶证据（§5.1~§5.3）不受影响。
- 合并注意：两棵树（`Strata-mezo` / `Strata-prtrain`）此前各有一份 `mezo.py` / `objectives.py` /
  `mem_guard.py`。**已核实内容逐字一致**（差异只有 CRLF/LF），现在两份都以 `feat/train-mezo`
  的提交为准；`Strata-mezo` 那次 merge 之后两树同源（同步一律 merge，不 rebase）。

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
| Q7 | 测试够不够：训练目录 258 项 + serve 6 项，但没有端到端（真跑一步再断言 CE 下降） | 缺 e2e，值得补 1 项 |
| Q8 | `bench_*.json` 这类实测结果文件是否随 PR 提交，还是只留文档引用？ | 提交（它们是 §5 的原始数据） |
| Q9 | **合并顺序怎么走**：零阶依赖的 `LORA`/`SCORE` 命令循环还在 `feat/train-mezo`，`upstream/main` 没有。是先单独提一个引擎 PR（我们自己的部分：`SCORE` + 换表 + 符号翻转；`lora.cu` 主体那 421 行属于 PR #3267，见 Q11），训练脚本排它后面？还是两个一起提？ | 先引擎、后脚本，各自可独立回滚 |
| Q10 | README 里"engine side, all merged upstream"这句与事实不符（见 §2.5），本 PR 里一并改掉可以吗？ | 改（文档不能留着错的事实） |
| Q11 | **LoRA kernel 主体是尚未合入的上游 PR #3267**（Alexey Rogov，`lora.cu` 421 行），本 PR 当只读依赖带着（三层归属见 §7.3）。如果 #3267 先被合入，要不要把本 PR 的那 421 行 rebase 掉、引擎面只留我们自己的 133 行符号翻转 + 换表 + SCORE？ | 倾向等 #3267；它合入后本 PR 的引擎面自动缩小，届时 rebase 一次即可 |
| Q12 | **反传路线要不要一起收？** 它是独立一层：比零阶树只多 32 个 `.py`（`backward/` 10 个模块 + `ce.py`/`lora.py`/`update_rules.py` + `mezo/sft_mezo.py` + 6 个 torch bench + 7 个 torch 测试 + 2 个诊断脚本），全部在 `training/` 下，不动 `serve/` 与引擎；**只收零阶 = 直接交 `feat/train-mezo`（`56c7eae0`），已验证 134 项全绿且不需要 torch**；那棵树里 `--method backward` 会明确报「缺 backward/sft.py」并指路（不是旧版薄壳的静默降级）。需要反传能力才需要这一层 | 默认一起收；只要零阶就交那条分支，见 §7.4 |

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
| 单测逐条 | `python -m pytest training/tests -q`（本版 258 项：256 过 + 2 项 watchdog 仅因 Windows 无 triton，见 §7.1） |
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
| 测试总数 | 232 | **258**（+18 项 numpy 侧目标函数测试 + 1 项「方法缺文件要明确报错」；Windows 上 256 过 + 2 项 watchdog 缺 triton） |
| 六方案极限表 | 缺 | §5.1 |
| 资源充足 / 不足 / 不可训练 | 无（只有一句"48 GB 装不下两份权重"） | **§5.2**：三档判据 + 48 GiB 实测三档 + 24~192 GiB 推算 + 四类"不可训练"症状 + 折算墙钟 |
| 数学训练前后 + 非训练集 | 只有旧过程式 logp | §5.4（0/3→2/3 + held-out −3/−31/−2） |
| 审核者要回答的问题 | 无 | §9（Q1~Q12） |
| 分支与合并 | 无（v0.2 只有一句"已合入上游"的假定） | **§7**：分支拓扑 + 基线判据（`upstream/main` vs fork `origin/main`）、11 条来源分支审计、LoRA 三层归属、合并顺序与回滚 |
| 反传路线的可裁剪性 | 无（两条路线绑死在一起，拆不掉） | **§7.4 + Q12**：反传是独立一层（比零阶树只多 32 个 `.py`），整层摘掉后零阶仍自洽（134 项绿、零 torch），且 `--method backward` 会明确报「本分支没有」而不是静默降级 |
| 提交卫生 | 无 | §3.4（25 个 `_*`、65 MB 二进制、`.bak`） |
| 两条路线的依赖 | 无 | §2.5（零阶 **0 个 pip 包** + 引擎二进制；反传 torch/transformers/accelerate/safetensors，Windows 另加 triton-windows、kernels） |
| 零阶方法出处 | 无 | §2.6（MeZO，arXiv 2305.17333，princeton-nlp/MeZO；未拷贝代码） |
| 引擎分支要求 | 无（v0.2 假定"已合入上游"） | §2.5 / Q9：`LORA`/`SCORE` **只在** `feat/train-mezo`，`upstream/main` 没有 |
| 实测原始数据（点位 / 口径 / 方法性质） | 无（散在草稿与工作台账里） | **附录 C**：13 行实测点位全表（含已退役 PyTorch 零阶的口径警告）、速度-长度、方法四条性质、署名 logp 与 L1 |

**逐项实测明细**（原始点位全表、速度-长度、方法四条性质、口径约定）见 **附录 C**；
`BRANCH-NOTES.zh.md` 保留为工作台账，内容已并入本文，不单独作为审核材料。

---

## 附录 C：实测台账（原始点位与口径）

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
> （`sft_mezo.py`，27 648 封顶、55 808 OOM），**不是当前默认路线**。别把 49 152 当成零阶的上限 ——
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
