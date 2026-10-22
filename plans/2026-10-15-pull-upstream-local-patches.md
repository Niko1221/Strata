# 拉取上游 622 commit，把本地 qwen35moe 改动收成性能补丁

## 目标

- 底：上游 `origin/main` = 296b59d2（v0.1.44.1），已含独立 `src/qwen35/` 引擎
- 保留：本地 6 个 commit 的 qwen35moe 改动，作为"性能/能力补丁"叠加在上游底上
- 产出：可构建、可跑本地测试的 worktree 分支

## 本地要保留的 6 个 commit（`git log origin/main..HEAD` 顺序）

1. `42bf2ff6` — Qwen3.6-35B-A3B 和 Ornith-1.5-35B-A3B（qwen35moe）：8-12 GB 显卡的小模型（44 文件，+2684/-627）
2. `20f4d4bb` — qwen35moe：8K 16-bit K/V，sm_75 以下的服务器路径
3. `80f760c6` — qwen35moe：再多 36 个 GGUF（含 KAT-Coder），setup 安装
4. `a0459b4e` — Pascal sm_60/sm_61 适配：双 GPU expert helper 几何、chtisgit 2-bit GSQ、iq_pack expert-table 校验、qwen36 MTP 取/打包
5. `cdeb1fb0` — qwen35moe：`has_indexer == false` 时 qsa_layer 不再要求 indexer 张量
6. `b07398d9` — qwen35moe：`has_hc() == false` 时 block_layer 不再要求 hc 张量

另有 2 个未提交改动（`src/core/layer.cpp`、`tools/test_iq_pack.py`），先 stash 再处理。

## 核心判断（已完成核实）

上游 main 已经自带完整 qwen35moe 引擎（`src/qwen35/q36_engine.cu` 2542 行 + `q36ref.cu` + `strata-q36`/`strata-q36ref` target + 家族切换）。本地改动是在**共享引擎文件**里做 qwen35moe 分支。两套实现**架构冲突**，不能直接叠加。

**关键发现：这不是"本地旧、上游新"，而是两条互不兼容的路线，能力互补而非替代。**

### 逐 commit 结论（阶段 A/B 已完成，2026-10-15）

| commit | 结论 | 理由 |
| --- | --- | --- |
| `80f760c6` 36 个 GGUF + 专家格式内核 | **搬（真新增）** | 上游 `q36_engine.cu:971` 调共享的 `native_expert_supported()`；上游 GU 表缺 10/11/20/39/30，本地有。已核实：上游 GU=18 个类型、本地 19 个且多出 MXFP4(39)/BF16(30)/Q2_K(10)；上游 D 表缺 10/11/17/18/21/22/30 |
| `42bf2ff6` 首个 qwen35moe 实现 | **只搬文档和文件事实** | 引擎能力上游已有；但小卡文件清单、Ornith 家族、实测文档上游全无（`git grep ornith origin/main` 零命中）。`data/expert-profile-qwen36.bin` 不搬（上游无排序专家缓存） |
| `a0459b4e` Pascal 名义下的工具链 | **只搬通用资产** | `tools/mtp_fetch.py` 健壮性（20 次重试/断点续传）上游未动、无冲突；远程专家参数化、mtp_pack qwen36 路径对上游无消费者 |
| `20f4d4bb` 16-bit K/V@8K + sm_75 以下 | **不搬** | 改的是 Flash-Next 引擎；上游 qwen35 路径 K/V 常驻 fp32（`setup.py:454`），`q36_engine.cu` 无 `--kv`、无 sm_75 分支，这些行没有消费者 |
| `cdeb1fb0` qsa_layer 不去 indexer | **不搬** | 上游 `layer.cpp:1040` 仍无条件解析 indexer，但无路径触发（qwen35moe 在 `generate.cpp:1875` 前切走）。不是上游 bug |
| `b07398d9` block_layer 不去 hc | **不搬** | 上游 `q36_engine.cu` 的 `k_add_rms` 已融合残差+norm，同样数学 |

### 搬的性能数字必须重测

本地 `docs/QWEN36.md` 的所有 tok/s、显存下限、KL 数字都是**本地引擎**测的。上游引擎 K/V 常驻 fp32、无排序专家缓存，量出来会不同。**先只搬文件事实（sha256、字节数、家族表、GGUF 清单），性能表留空待重测。**

## 阶段

### 阶段 A：冲突清单（只读）
- 对 6 个 commit 逐个跑 `git diff <c>^ <c>`，按"上游已有/上游部分有/上游没有"三类归档
- 重点文件：`src/core/layout.cpp`、`src/core/layer.cpp`、`src/program/generate.cpp`、`src/prefill/prefill.cpp`、`setup.py`、`CMakeLists.txt`
- 产出：每个 commit 的"保留/丢弃/重写"三分类清单

### 阶段 B：建底（一次完成）
- 从 `origin/main` 建新分支 `qwen36-pascal-v2`（或用户指定的名字）
- 用 `git worktree add` 建独立工作目录，不动当前 `qwen36-pascal`

### 阶段 C：应用补丁（逐 commit，可验证）
- 按阶段 A 清单，把"上游没有"的部分逐个 cherry-pick 或手工 apply 到新分支
- 每个 commit 后跑：`python tools/test_iq_pack.py`（不依赖 GPU）和相关 setup 测试
- 有冲突就手工解，不自动跳过

### 阶段 D：收口验证
- `git log --oneline` 确认新分支历史
- 跑全部不依赖 GPU 的测试：`python tools/test_setup_qwen36.py`、`python tools/test_setup_qwen36_more.py`、`python tools/test_iq_pack.py`
- 对比新旧分支的 `git diff --stat`

## 验收标准

- 新分支从 `origin/main` 干净出发
- 本地 qwen35moe 能力逐项可追溯到具体 commit
- 所有不依赖 GPU 的测试绿
- 当前 `qwen36-pascal` 分支**不被改写**（用户可回退）

## 不做

- 不自动 push
- 不删旧分支
- 不改 docs（除非某个补丁本身含 docs 改动）
- 不合并上游的 `src/qwen35/` 引擎和本地的共享文件改动到同一文件（冲突就报告，不硬合）
