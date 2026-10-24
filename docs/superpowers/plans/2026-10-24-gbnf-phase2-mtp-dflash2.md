# GBNF 约束解码 · Phase 2：MTP / DFlash2 兼容 —— 接力状态与实施计划

> 新会话从这里接力。Phase 1（13 个任务）已全部完成；本文记录当前状态、已修缺陷、以及让投机解码（MTP、
> DFlash2）与约束解码共存的实施方案。Phase 1 的 spec 与计划：`../specs/2026-10-23-gbnf-constrained-decoding-design.md`、
> `2026-10-24-gbnf-constrained-decoding.md`（冲突处以 spec 为准）。
>
> **（2026-10-24 晚：Phase 2 已完成**——P0 合并 `8183c6b6`（merge `032ef6c7`）、P1 solo 掩码链（`2eb4ec2f`）、
> P2 batch_mtp 链（行级掩码交接）。4090 复测：JSON 50/50、工具 30/30、constrained+MTP 123.1 vs plain+MTP
> 121.9 tok/s 中位（草稿吃掉掩码开销）；MTP 收益基线 plain 94.1 → 121.1 tok/s（+28.7%）。与计划的偏差：
> DFlash2 在本 build 是 CLI 专属起草器（serve 无此路径，无 grammar 交集）；SYCL 镜像 sampler 本就逐行、
> program 级接线是 Phase 1 既定边界，未改动；batch_mtp 链落在 P2（行级交接，非 P1 顺手）。顺手修了两个
> Phase 1 缺口：greedy+grammar 的图 argmax 绕掩码（run/pl_finish/sample_rows 补重采样）、层拆分 pipelined
> 窗不推进 grammar（约束请求改串行，与 penalties 同门）。细节见 git log 与 DETAILS Limits。**
>
> **回归（全部 2026-10-24，本机）**：ctest 150 过 / 1 失败（ple_parity，环境性夹具）/ 7 跳过，与 merge 后
> 基线一致；serve 全量 1058 例 0 失败（修了 winjob 测试在本机 GBK 控制台下的 tasklist 解码）；pytest
> tests/terminal 1586 过，余 4 失败为环境类（2 个仓库目录被识别为安装根、2 个 Windows PID 句柄存活时序，
> 与本分支改动无关）；apps/host npm test 4 失败均为 Windows 符号链接权限（EPERM）与本机限制。分支较 fork
> 领先 5 个提交，**未推送、未开 PR（等用户指示）**。）
>
> **（2026-10-24 深夜：生产暴露一个 Phase 1 底座缺陷并已修复**——约束请求在长中文回答上退化出填充字
> （"符号"→"符好"、"好好好" 连环）。排查链：重放复现 → bisect（无思考/无语法均干净）→ STRATA_GRAMMAR_DEBUG
> 抓到窄掩码 → 逐 token 对账发现窄态全是合法的工具调用跟踪、填充字其实产生于"宽松"掩码 → 对账 vocab.json
> 的标准 GPT2 bytes_to_unicode 映射，定位到 `sp_piece_to_text` 只解码了 Ġ/Ċ/latin-1/printable，
> U+0121-U+0143（字节 127-160、173——多数中文 UTF-8 的第二字节）未还原 → 88,990/248,320 个 token 的 piece
> 保持无效 UTF-8 → 每个语法掩码永久封禁 36% 词表。修复 = 补全映射表（`grammar_vocab.cpp`，与 HF
> tokenizers / llama.cpp 转换器同一张表）。验证：根掩码开放 159,330 → 248,129；复现请求（工具语法+思考+MTP
> 链，草稿接受 72%）三次长回答全干净；验收 JSON 49/50（唯一 502 是 recipe 题触顶输出预算——修复前它
> "通过"只因坏词表让回答变短）、工具 30/30、约束 135.9 vs plain 124.3 tok/s（草稿吃掉掩码成本且有富余）。
> 提交见 git log；`bench/results/vocab-fix3-grammar-4090/`。）

## 接力状态（2026-10-24 晚）

- 分支 `feature/gbnf-constrained-decoding`：16 个提交，基于 `011c9e60`（v0.2.2 时代）。PR #3922 已按用户要求**关闭**
  （fork `sd133cn/Strata` 上有同分支存档）。上游 main 已到 `8183c6b6`（v0.2.4.1，含 DFlash2 与 DeepSeek 4.1）——
  **Phase 2 第 0 步是 merge origin/main**，预期冲突点：`src/program/generate.cpp`（上游改了 drafting 区域）、
  `serve/server.py`、`CMakeLists.txt`（DFlash2 新文件）。
- Phase 1 交付（提交 3c7b97c7 收尾）：语法核心移植、逐 token 位掩（CUDA/HIP/SYCL 镜像）、GRAMMAR 协议、
  转换器 CLI、serve 三路由 + 工具语法构建器、文档；4090 验收 **JSON 50/50、工具 30/30、单流成本 -2.6%**。
- 验收期发现并修复的三个运行时缺陷（都在 3c7b97c7）：
  1. `exclude_seqs` 把重启字符错并进拒绝类 → 思考区被饿成无字母循环（`serve/grammar_tools.py`）；
  2. 续跑（wrap-up/close-retry/forced-call/BGEN 回 solo）未喂语法状态 → `gseed=<n>` 引擎端重放已生成尾部
     （`src/program/generate.cpp`、`serve/server.py`）；
  3. mask 缓存按"栈顶"做键 → 同顶不同底的状态串掩码 → 键改为整条栈（`src/grammar/grammar_mask.cpp`）。
- 本机环境（rig 事实，直接可用）：
  - 模型 `F:/Strata-data/models/swift-IQ3_S/`（2 分片），pack `F:/Strata-data/packs/swift-iq3_s`；
    IQ pack 启动要 `--spec >= 2`。
  - 构建：`cmd //c .zcode-build.bat`（vcvars64 + VS 捆绑 cmake + Ninja，产物 `build/strata.exe`）；
    ctest 在 `build/` 下（vcvars 环境里跑）。基线：156/157，`ple_parity` 缺 Q2_0 夹具（环境性）。
  - venv：`F:/Strata/.venv/Scripts/python.exe`（3.14，jsonschema 已装）。`python3` 是坏桩，用 `python`。
  - 验收套件：`tools/grammar_acceptance.py`（逐项 JSONL、420s 超时、UTF-8 stdout、围栏剥离、端口占用预检、
    6 槽 probe）。启动器：`F:\Strata-dev\run-swift-iq3_s.bat` + `strata-swift-iq3_s.json`（端口 8199，**无
    `--mtp`/`--spec-min-p`**——当前引擎对 grammar+MTP 硬拒）。
  - 教训：bash heredoc 写 python 会吃反斜杠（写 `.zcode-obj/*.py` 再跑）；Edit 工具会把消息文本里的 `\n`
    解码成真换行（文档编辑走 chr(92) 脚本）；GBK 控制台输出要 `iconv -f GBK` 或 UTF-8 reconfigure。

## 目标

约束请求不再拒绝 MTP / DFlash2：语法请求也能吃投机加速，且约束正确性逐 token 不破（每个发出的 token 都是
"其语法状态对应掩码下的采样"）。普通请求路径逐字节不变。

## 方案（已核实可行，逐行掩码）

上游 DFlash2（f1b6a214 合入 0.2.3）与 MTP 是**两个起草器、同一个 T 行验证窗**；语法只需跟着验证窗走：

1. **采样核已支持逐行掩码**：`sampler.cu:160`（SYCL 镜像同型）`mrow = p.grammar_mask + t*mask_stride_words`
   ——窗内第 t 行读环形缓冲第 t 行。掩码环 `kVerifyMaxT=8` 行 ≥ 起草窗（≤4 token），容量现成。
2. **掩码链（核心新增，主机侧）**：窗提交前，用 scratch 克隆从当前状态起对草稿 `d1..d(T-1)` 逐个假设推进，
   得 `mask_0..mask_(T-1)` 上传第 0..T-1 行。第 t 行的语义自洽：第 t 行采样 = 消费 `x,d1..d_t` 后的下一 token，
   其掩码 = 状态推进过 `d1..d_t`。草稿在 `d_j` 处非法时链条截断——接受度在 j-1 行必然断（第 j-1 行的采样被
   `mask_(j-1)` 限制，取不到 d_j），j 行起整段丢弃，掩码复用 `mask_(j-1)` 即可（永不用于发出 token）。
3. **真状态推进**：窗后按 `outv[0..a]`（接受 + 重采样）推进真克隆——现有逐 token 推进循环（solo
   `generate.cpp:15182-15207`、batch `11578`）原样保留，为下一窗产出 `mask_0`。
4. **起草器不掩码（v1）**：非法草稿在验收处自然死掉，只损失命中率；先测数值再决定是否掩码起草器。
5. **放开硬拒**：删 `generate.cpp:12378` 对 `--mtp`/`--batch-mtp` + grammar 的硬 ERR；**保留** named
   `spec_min_p` + grammar 的拒绝（草稿地板过滤与掩码次序相互作用，v1 不碰，文档记录）。
6. **batch_mtp 路径**（`generate.cpp:11780` 一带）同一套链式掩码，其逐行上传已是 `gi*GS + t`。
7. **续跑与 gseed 不变**：wrap-up/强制调用/回 solo 的 seed 重放照旧，普通（T=1）窗路径逐字节不变。

不推荐（记录备查）：服务端在引擎 400 "grammar + speculative" 时回退提示词路径——约 20 行，但约束请求
永远拿不到投机加速，只是把硬失败变软。

## 任务

### Task P0：merge origin/main + 基线重测
merge `8183c6b6`，解决三处预期冲突（drafting 区域以语义为单位并：上游的 DFlash2 起草器分支与本地 grammar
的 `req_grammar`/`sfx_on`/gseed 各自保留）；重建；ctest 基线 + 验收套件重跑（50 JSON / 30 工具应仍 green）。
**同时实测本机 plain+MTP vs plain 无 MTP 单流 tok/s**（Phase 2 的收益基线，此前无测量，不得引用猜测值）。

### Task P1：掩码链 + 逐行上传（solo 路径）
`generate.cpp` solo 窗装配处：grammar 请求且草稿开启时，构建 `mask_0..mask_(T-1)` 上传环形各
行（`cudaMemcpy` 每行，同步语义同现有 row-0 上传）；去掉硬拒；gdbg 增加逐行打印
（`STRATA_GRAMMAR_DEBUG` 已有钩子）。单测：`tests/program/grammar_decode_test.cpp` 加草稿链闭环
（草稿 token 非法时链条截断的断言）。

### Task P2：DFlash2 起草器接入验证 + batch_mtp 路径
确认 DFlash2 草稿进同一 `ver` 窗（`include/strata/core/verify.hpp` 的 draft 捕获）；`--draft dflash2`
+ grammar 走通；batch_mtp 路径（`11578` 一带）同链。SYCL 镜像 `sampler.dp.cpp` 同步逐行行为（本机无
oneAPI，记 not run）。

### Task P3：验收 + 文档
4090 重跑验收套件（engine_args 加 `--mtp`/`--draft dflash2` 变体）：JSON 50/50、工具 30/30 不回退；
三组数：plain+MTP、grammar+MTP、grammar 无草稿。DETAILS 的 Limits 段与 spec §5.1 的"硬拒"改写为
"逐行掩码共存"；`run-swift-iq3_s.bat` 的配置加回 `--mtp`（吸收 P0 的 MTP 实测结论再定默认值）。

## 风险

- merge 冲突面（drafting 区域上游重构过）——以语义并，不逐行硬拼。
- 掩码链每 token 8 行 H2D ≈ 248 KB/token（7760 词 × 4B × 8 行），约 21 MB/s @87 tok/s，PCIe 24 GB/s 下
  可忽略；若实测超预期，改为只上传到链截断行。
- DFlash2 的草稿生成若在 CUDA graph 内采样，掩码指针须保持环形基址稳定（Phase 1 已按此设计，需复核
  DFlash2 的图捕获范围）。
