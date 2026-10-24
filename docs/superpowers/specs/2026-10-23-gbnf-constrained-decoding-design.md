# GBNF 约束解码设计（借鉴 llama.cpp）

日期：2026-10-23 · 分支：`feature/gbnf-constrained-decoding` · 状态：待审阅

## 0. 接力状态（新会话从这里接）

- 工作区：git worktree `F:\Strata-dev`（检出本分支，共享 `F:\Strata\.git`）；`F:\Strata` 保持在 `main`，是用户跑模型的运行实例，不在里面开发。
- 本文档尚未提交：pre-commit 的 Mimosa L3 钩子整仓扫出 537 个高危（全部先于本次改动，集中在 `serve/test_server.py` 等测试文件的 SSRF/路径穿越形态）强制拦截；MCP 深度扫描已对两项目封印（`F:\Strata`：scan-2026-10-23T15-37-49.786Z-85a1278eb6bb，seal sha256:5bdbe6c2…，472 项；`F:\Strata-dev`：scan-2026-10-23T15-35-19.666Z-2fedf893d467，seal sha256:490a021a…，333 项；688 个依赖包中 2 包命中 3 条安全公告）。MCP 封印 `verdictEffect: none`，钩子解锁需用户在 Mimosa 应用内标记误报/基线。2026-10-23 审阅核对：repo 与全局均无 `core.hooksPath`，`F:\Strata\.git\hooks\pre-commit` 不存在——拦截经 Mimosa 应用侧发生；实施期分支提交若被拦，基线标记在关键路径上，与计划并行处理。
- 流程进度：2026-10-23 审阅通过（6 处修订已并入）→ 2026-10-24 实施计划已成稿：`docs/superpowers/plans/2026-10-24-gbnf-constrained-decoding.md`（13 任务，TDD，逐任务提交）。用户已授权全程自主实施到交付（m00141：不再提问，按推荐路线）；按 CONTRIBUTING 先开 scope issue（计划 Task 1）。
- 未决问题见 §9；§2 的承重事实均按 file:line 核实过，可直接依赖。

## 1. 目标（用户定调）

**最主要目的：稳定的工具调用和 JSON 结构化输出。**

在引擎内实现 llama.cpp 同源的 GBNF 语法约束解码：每步采样只在语法允许的 token 中取值，输出结构由解码保证，不再依赖提示词指令与事后校验。通用 `grammar` 参数是底座；JSON Schema 硬约束与工具调用 grammar 是目的。

## 2. 现状与承重事实（已核实）

- 解码全在引擎内：`src/program/generate.cpp:7-8` 的循环 `embed_row → 层图 → lm_head → sample → embed_row(next)`；采样是设备核，链为 `penalties → top_k → top_p → min_p → temperature → pick`（`include/strata/kernels/sampler.hpp`，实现 `src/kernels/cuda/sampler.cu`），可被 CUDA graph 捕获。
- 每步采样 token 回主机：`src/program/generate.cpp:15864-15865`（`cudaMemcpyAsync` d_next→host），随后 `put_input` 喂回。语法状态机在主机推进不需要新增同步点。
- `logit_bias` 已有 HTTP→服务器→行协议→设备数组→采样核（penalties/filters 之前加性，-inf 禁）的端到端先例：`serve/logit_bias.py`、`serve/server.py:1437`、`SamplerParams.logit_bias`。
- 引擎无分词器，但 GGUF 元数据可读词表字符串：`include/strata/artifact/gguf_reader.hpp:728`（`tokenizer.ggml.tokens`）。
- llama.cpp 已整体 vendor，**但仅存在于主检出 `F:\Strata`**（`.gitignore:31` 忽略 `/third_party/llama.cpp`；worktree `F:\Strata-dev` 无此目录，2026-10-23 实测）：`third_party/llama.cpp/src/llama-grammar.cpp/.h`（GBNF 解析器 + 状态栈 + bitmask）、`common/json-schema-to-grammar.cpp/.h`（JSON Schema→GBNF）、`grammars/*.gbnf`、测试 `tests/test-grammar-parser.cpp`、`test-grammar-integration.cpp`、`test-json-schema-to-grammar.cpp`。移植的复制源即该目录；移植产物（含转换器 CLI 用到的 `common/` 源）必须提交入库并保留 llama.cpp 的 MIT 署名——构建不得依赖本地 `third_party/` 目录。
- 现状 structured output：`serve/structured.py:1-4` 明示"native engine has no grammar decoder"，prompt 指令 + 事后校验，失败即 `structured_output_failed`。
- 工具调用是 XML 风格（Qwen3-Coder 样式），序列化逐字为：`<tool_call>\n<function=NAME>\n(<parameter=KEY>\nVALUE\n</parameter>\n)+</function>\n</tool_call>`，多调用以 `\n` 分隔，前有散文时以 `\n\n<tool_call>` 开板（`serve/chat_template.jinja:134-156`）。VALUE：字符串是裸文本，其余经 `tojson` 成 JSON。解析器在 `serve/frontend.py:909` 一带（有 `rescued`/`refused` 计数）；`tool_choice` 强制逻辑在 `serve/server.py:6928`、`server.py:5244`。

## 3. 已定决定（含理由）

| 决定 | 选项 | 理由 |
|---|---|---|
| v1 交付面 | GBNF 底座 + JSON Schema 硬约束 + 工具调用 grammar | 用户拍板；工具调用稳定是首要目标 |
| 语法状态机位置 | 引擎内 C++（移植 vendored llama-grammar） | 解码只在引擎发生；主机每步已在 token 循环里；llama.cpp 测试可移植当验收。否决：服务器逐步驱动（每 token 一次 IPC，逆批处理架构）；xgrammar/outlines（新依赖，且题面是借鉴 llama.cpp） |
| 投机解码 | 受约束请求 v1 关闭投机（plain decode）；引擎侧 grammar+spec 共现直接拒绝请求，绝不静默降级 | 用户拍板；verify 窗口/耦合草稿/概率接受三条路径与 grammar 的交互测试矩阵翻倍，留后续 PR |
| 后端 | CUDA 直用、HIP 经 hip_compat 同编译、SYCL 移植树跟进；Mac（`metal/` 独立引擎树）与 CPU 缓做 | 用户拍板；Mac 引擎需各实现一遍接法，parity 矩阵翻倍 |
| mask 应用位点 | 采样链最前（与 logit_bias 同位点，penalties/filters 之前） | 与 llama.cpp 语义一致：-inf 硬禁 |

## 4. 架构与数据流

```
HTTP 请求（grammar / response_format / tools+tool_choice）
  → 服务器：schema→GBNF（转换器子进程，按 schema 哈希缓存）；工具 grammar 构建器
  → 行协议：首见发 `GRAMMAR <sha16> <字节数> <文本>`；GEN/BGEN 行带 `grammar=<sha16>`
  → 引擎：按哈希取/建语法图（进程内 LRU≤64）；每槽位克隆状态栈
  → 每 decode 步：采样 token 回主机（既有回读点）→ 推进该槽状态栈 → 算 bitmask
    → cudaMemcpyAsync 上传按槽位环形缓冲 → 采样核在 penalties 前应用
  → 服务器：token 采样瞬间即合法 → 直出流（不再缓冲）；末端 validated_json 金丝雀
```

## 5. 组件

### 5.1 引擎 grammar 模块（新）
`include/strata/grammar/` + `src/grammar/`：移植 vendored `llama-grammar.cpp/.h` 为 `strata::grammar`——GBNF 解析器、状态栈、栈推进、per-state token 接受预计算与 bitmask 缓存（llama.cpp `init_vocab`/`parse_token`/advanced-cache 结构）。移植手术集中在 `llama_vocab` 依赖，换成 `GrammarVocab`（从 GGUF 元数据经 gguf_reader 构造：token 字符串 + **`tokenizer.ggml.token_type` 控制/特殊 token 标志** + EOG id）。vendored 侧三处触点（审阅核实）：`llama-grammar.cpp:186`（`parse_token`）、`:1131`、`:1212`。`token_type` 不可省：llama.cpp 靠它跳过 special token，丢了标志 `<|im_end|>` 会被当字面文本解析进语法。

### 5.2 mask 管道
- 设备端：按槽位环形缓冲 `uint32[slot][ceil(n_vocab/32)]`，**首次 grammar 请求到达时惰性分配**（显存可忽略：152k 词表每槽约 19 KB；不留"启动未启用则请求 400"的路径）；核内读 `mask_ptr + slot*stride`，mask 指针经 mapped-params 间接层（与 `sampler.hpp:56-60` `coupled_draft_stage` 的 `*params <- *mapped_params` 同型）在图内运行时可读，惰性分配无需重启服务；每步 H2D 在图外更新。
- 采样核：`SamplerParams` 增 mask 指针与 stride，默认 null=完全跳过（默认路径零改动）；`sample_tokens` 与 `sample_greedy_cluster` 两条路径都受限——后者签名现无 `SamplerParams`（`sampler.hpp:50`），mask 接线需扩其签名，两路各自接。
- 主机端：bitmask 按（栈顶状态, advanced）对缓存；EOG 只在语法终态放行。

### 5.3 行协议
`GRAMMAR <sha16> <bytes> <文本>` 内容地址注册 + `grammar=<sha16>` 引用；采样键并入该哈希（`serve/server.py:1367` `sampling_keys`，其尾部即 `bias_api.engine_key`，`:1437`；该串直接拼进 GEN 行 `server.py:2177-2178`，逐请求粒度，无引擎重启语义；grammar 改变采样分布）。服务器发现 grammar 即从 GEN/BGEN 行剔除投机参数。

### 5.4 转换器（服务器侧）
`tools/` 下用 vendored `common/json-schema-to-grammar.cpp` + `json.cpp` 编小 CLI；服务器子进程调用、按 schema 哈希缓存。零新依赖。

### 5.5 工具调用 grammar 构建器（服务器侧 Python）
从工具 schema + 模板序列化规则（§2 逐字）生成 GBNF：
- `tool_choice: "required"/"any"`：强制 `<tool_call>\n<function=NAME>` 开板；候选名交替，指定名则定死字面量。
- 参数：逐属性 `<parameter=KEY>\n值\n</parameter>`；字符串属性→文本规则（禁 `\n</parameter>` 序列）；非字符串属性→该属性子 schema 的 JSON 子语法（转换器按属性调用）。
- `tool_choice: "auto"`：根规则 = 散文 | 工具调用块 交替（模型可不选工具；一旦调用必合法）。
- 并行多调用 = 块重复。grammar 请求跳过 forced 前缀注入（`server.py:5244`）与 rescue 补偿路径（解析器保留为契约检查）。
- 带 MCP 工具 → 回落 prompt 路径，响应声明 method。**这是新政策，不是沿袭现规则**：现行规则对 forced+MCP 与 response_format+tools/MCP 是直接拒绝（`server.py:6945`、`server.py:6930-6931` 抛错），回落是放宽；响应形态测试须覆盖"声明 method 的回落"与既有拒绝的边界。

### 5.6 response_format 语义
NVIDIA/AMD：`json_object` = 仅对象语法；`json_schema` = schema→GBNF 硬约束。响应 `structured_output.method`（现 "prompt_and_validate"，`server.py:4531`）→ "grammar"。Mac/CPU 报不支持 → json_schema 回落 prompt+validate 并声明 method；裸 `grammar` 参数 → 400。`validated_json` 保留为末端金丝雀（触发=引擎 bug，走现有 502 路径并加 "grammar engine bug" 标记，绝不静默）。

## 6. 错误处理

- grammar 文本解析失败：服务器 400（带解析器给出的行号信息）。解析器在引擎侧，故行协议需要一条 **GRAMMAR 注册的错误回执行**（带解析器行号），服务器映射为 400；机制在此定，回执格式细节留计划。
- grammar+vision：协议还有 GENI/BGENI 图像变体（`generate.cpp:11845,11859`）。v1 规则：grammar 在 GENI/BGENI 与 GEN/BGEN 完全同路径（mask 机制与图像无关），不单独缓做。
- 引擎 grammar+spec 共现：引擎回明确错误行；服务器把它当 500 类引擎契约违规报出（不该发生，发生了就是 bug）。
- grammar+MCP：回落 prompt 路径，响应声明 method。
- Mac/CPU grammar：见 §5.6。
- 槽位状态栈克隆后互不影响；BGEN 重准入复用哈希缓存的语法图。

## 7. 测试与验收

- 引擎：移植 vendored `test-grammar-parser.cpp` + `test-grammar-integration.cpp` 子集；`sampler_parity.cpp` 扩 mask 位点；新增 `grammar_test.cpp`；`logit_bias_test.cpp` 模式沿用。
- 转换器：vendored `test-json-schema-to-grammar.cpp` 的金样用例移植进 `tools/test_grammar_schema.py`。
- 服务器：`serve/test_structured.py`、`test_responses.py`、`test_frontend.py` 扩 grammar 路径；`tools/test_grammar_*.py` 无 GPU 自测（构建器/协议键/回落选择）。
- 终端：`tests/terminal` 覆盖行协议与 launch 面改动。
- **字节一致**：无 grammar 键的 GEN/BGEN 行与 engine_key 金样逐字节不变。
- **稳定（主验收，GPU）**：固定 JSON 套件 + `strata connect <app> --check` 风格工具套件各跑 N 次：schema 通过率 100%；工具调用解析失败 0；grammar 请求 `rescued` 计数恒 0。
- **性能（同 rig）**：受约束 vs 无语法 plain decode tokens/s，目标开销 ≤10%，超出如实报告；投机关闭的代价单独报（引现有 spec 加速测量）。
- 构建：文件触及的每个后端（CUDA、HIP、SYCL）都构建通过才提 review。

## 8. 默认与限制（文档要写）

- 默认路径逐字节同今：mask 指针默认 null，不分配不上传；新参数全部 per-request opt-in。
- 已知限制：受约束请求无投机加速；每步 mask 开销（附实测）；Mac/CPU 不覆盖；grammar+MCP 回落；GBNF 语法支持面 = vendored 版本原样。
- 流程（CONTRIBUTING.md）：先开 scope issue（本文档为素材），PR 标题 `[Engine] …`；测量数字注明测量硬件。

## 9. 开放问题（留实施计划）

实施期间全部落定（2026-10-24，4090 实机验收通过，数字见文末）：

- `GRAMMAR` 行的传输编码：定为单行转义 —— 服务端把 `\` 转义为 `\\`、换行转义为 `\n`，byte_len 记未转义字节数；引擎一遍反转义。
- grammar LRU 逐出：cap 64；被某个槽位的 shared_ptr 钉住的条目永不逐出（溢出期容忍超限）。
- 思考区立场修正：实测否决了『完全放行思考区』——模型把围栏 JSON 习惯当成自由推理倾倒，答案永不开始。落地为重启链前序（reason 区禁成句的 `</think>` 与 <code>&#96;&#96;&#96;json</code>，其余任意文本逐字放行），思考预算到点的 wrap-up 续跑以 `gseed=<n>` 重放已生成尾部（否则续跑整段跑在开放的 reason 掩码上）。
- 负向前瞻缺口：exclude_seqs 乘积链自动枚举全部状态；修复了把『重启字符』错并进拒绝类的实现缺陷（曾把思考区饿成无字母循环）。边界测试在 serve/test_grammar.py 与 tools/test_grammar_builder.py。
- 多槽 mask 开销：6 槽并发聚合 172.1 → 140.1 tok/s；单流 400 token 中位 -2.6%。逐 token 的宿主 CPU 毫秒数此次未测得（探测进程时间读数为 0），以聚合吞吐为代理指标记录。


## 10. 验收结果（2026-10-24，RTX 4090 48GiB，Swift IQ3_S，engine 0.1.47 构建）

- 协议冒烟 6/6（INFO 声明、注册 OK/ERR、带语法生成、未注册拒绝、spec 组合拒绝）。
- JSON 套件 50/50 全部通过自身 schema（5 个 schema × 10 提示，effort low，温度 0.6，max_tokens 3200）。
- 工具套件 30/30 解析为合法调用，0 失败、0 从推理区抢救。
- 单流成本：中位 84.8 → 82.6 tok/s（-2.6%）；首次约束请求含一次性 mask 核预热。
- 6 槽并发：聚合 172.1 → 140.1 tok/s；无槽位错误。
- 结果与逐项记录：`bench/results/2026-10-24-1332-grammar-4090/`。
