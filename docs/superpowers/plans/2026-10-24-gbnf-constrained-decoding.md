# GBNF 约束解码 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在引擎内实现 llama.cpp 同源 GBNF 语法约束解码，使工具调用与 JSON 结构化输出由解码器保证（schema 通过率 100%、解析失败 0），默认路径逐字节不变。

**Architecture:** vendored llama.cpp 语法核心移植进 `src/grammar/`（命名空间 `strata::grammar`，词表 shim 自 GGUF 元数据）；GEN/BGEN 行带 `grammar=<sha16>`，文本经新 `GRAMMAR` 行注册（引擎 LRU≤64，槽位共享指针钉住）；每 decode 步主机推进槽位状态栈、算 uint32 位掩（bit=1 禁）异步 H2D 进按槽环形缓冲；采样核在 logit_bias 同一位点、penalties 之前应用 -inf。服务器侧：JSON Schema→GBNF 子进程转换器（vendored `common/json-schema-to-grammar.cpp`，入库零新依赖）+ 工具 grammar 构建器 + response_format 翻向 grammar（流式直出，末端 validated_json 金丝雀）。

**Tech Stack:** C++17/CUDA（HIP 同源码经 hip_compat；SYCL 镜像 `sycl/src/kernels/cuda/sampler.dp.cpp`）、CMake/ctest、Python 标准库服务器（serve/）、pytest/unittest。

**Spec:** `docs/superpowers/specs/2026-10-23-gbnf-constrained-decoding-design.md`（已审阅修订版；本计划以它为准，冲突处以 spec 为准）

## Global Constraints

- 无 grammar 的请求：GEN/BGEN 行与采样键逐字节不变；mask 指针默认 null，核内零开销分支（`p.grammar_mask ? ... : ...` 与 logit_bias 同型）。
- 语法特性全 per-request opt-in；服务器无 grammar 时行为与上一发布一致。
- 语法请求禁投机：服务器从行里剔 spec 参数；引擎侧 grammar+spec 共现 → 硬 ERR 行，绝不静默降级。
- 位掩语义：**bit=1 ⇒ 禁**（llguidance 约定，`common/llguidance.cpp:62-65` 同款）；EOG 仅在任一站空（终态）放行；CONTROL 型 token 一律禁（strata 对 llama.cpp 的记录在案的偏离）。
- 移植文件保留 llama.cpp MIT 署名头注释；构建不得依赖 gitignored 的 `third_party/llama.cpp`（它只在主检出 `F:\Strata` 存在，`.gitignore:31`）。
- 文件触及的每个后端（CUDA、HIP gfx906、HIP wave32、SYCL）构建通过才算完成；本机无 oneAPI 则 SYCL 记 not run 并如实报告。
- TDD：先失败测试→实现→通过→提交。提交英文祈使句，仿仓库风格（`Grammar: …` / `tests: …` / `serve: …`）。
- 测量数字注明 rig（本机 RTX 4090, CUDA 13.2, Windows）；未测的标 unconfirmed static analysis；绝不虚构。
- 先开 scope issue（CONTRIBUTING），PR/issue 标题以 `[Engine]` 开头。
- 提交被 Mimosa 应用侧拦截时（§0）：记录拦截输出，继续开发，用户应用内标记基线。
- cmake 路径由 Task 0 记录为 `<CMAKE>`（bash PATH 无 cmake；探测 `C:\Program Files\CMake\bin` / VS 捆绑 / `where cmake`）；所有构建命令用它。
- 采样不在 CUDA graph 内（`verify.cpp:3002-3005` 注释；eager launch，参数按值）——mask 指针是 `SamplerParams` 普通字段，环形基址稳定，无需 mapped-params。

---

### Task 0: 环境定桩（cmake 路径 + 基线构建）

**Files:** 无源码改动；结果记入本文件勾选与执行日志。

**Interfaces:**
- Produces: `<CMAKE>`（绝对路径）、`build/`（CUDA+tests 配置树）、基线绿灯证据。

- [ ] **Step 1: 找到 cmake**

```sh
ls "/c/Program Files/CMake/bin/cmake.exe" 2>/dev/null || ls "/c/Program Files/Microsoft Visual Studio/2022/"*/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe 2>/dev/null || where.exe cmake
```

把绝对路径记为 `<CMAKE>`（bash 用 `/c/...` 形式）。

- [ ] **Step 2: 配置 + 基线构建（后台，首建耗时长）**

```sh
"<CMAKE>" -S F:/Strata-dev -B F:/Strata-dev/build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DSTRATA_BUILD_TESTS=ON
"<CMAKE>" --build F:/Strata-dev/build --target strata logit_bias_test -j16
```

Expected: `strata.exe` 与 `logit_bias_test` 产出，无错误（改动前基线必须绿）。

- [ ] **Step 3: 基线 ctest**

```sh
cd F:/Strata-dev && "<CMAKE>" --build build --target logit_bias_test && ctest --test-dir build -R "logit_bias_test" --output-on-failure
```

Expected: PASS。

---

### Task 1: Scope issue + 提交 spec/plan

**Files:** 无代码；`docs/superpowers/` 入 git。

**Interfaces:**
- Produces: GitHub issue 号（PR 引用）；branch 上第一个提交（试探 Mimosa 拦截的真实行为）。

- [ ] **Step 1: 开 scope issue**

```sh
cd F:/Strata-dev && gh issue create --title "[Engine] GBNF constrained decoding: stable tool calls + JSON schema (scope)" --body-file docs/superpowers/specs/2026-10-23-gbnf-constrained-decoding-design.md
```

（body 用 spec 全文；记下 issue 号。）若 gh 未登录/网络失败：记录并继续，PR 描述里补。

- [ ] **Step 2: 提交两文档**

```sh
cd F:/Strata-dev && git add docs/superpowers/ && git commit -m "docs: GBNF constrained-decoding spec (reviewed) and implementation plan"
```

若 Mimosa 拦截：把拦截输出原样记入本计划勾选行，继续后续任务（提交积压，用户标记基线后统一提交）。

---

### Task 2: 移植语法核心 `strata::grammar`（无 GPU）

**Files:**
- Create: `include/strata/grammar/grammar.hpp`（移植自 `F:/Strata/third_party/llama.cpp/src/llama-grammar.h`）
- Create: `src/grammar/grammar.cpp`（移植自 `.../src/llama-grammar.cpp`，1526 行）
- Create: `include/strata/grammar/compat.hpp`（断言/日志 shim）
- Create: `src/grammar/grammar_test.cpp`（移植 `.../tests/test-grammar-parser.cpp` 金样子集）
- Modify: `CMakeLists.txt`（`strata_core` 源列表 :336-337 加 `src/grammar/grammar.cpp`；tests 的 GPU-无关块 :268-321 加 `grammar_test` 目标，仿 `logit_bias_test` :272-274）

**Interfaces:**
- Produces（后续任务全部依赖，命名锁定）:
  - `namespace strata::grammar`；`enum class gretype { END, ALT, RULE_REF, CHAR, CHAR_NOT, CHAR_RNG_UPPER, CHAR_ALT, CHAR_ANY, TOKEN, TOKEN_NOT }`；`struct grammar_element { gretype type; uint32_t value; }`；`using grammar_stack = std::vector<const grammar_element*>`。
  - `struct GrammarVocab { std::vector<std::string> pieces; std::vector<char> control; std::vector<char> eog; std::string token_to_piece(int32_t id) const; bool is_eog(int32_t id) const; }`（词表 shim；无 tokenize——`<name>` 语法在 `parse_token` 里抛 `std::runtime_error("invalid token")`，GBNF 的 `<[id]>` 数字形式不受影响）。
  - `struct grammar { std::vector<grammar_rule> rules; std::vector<grammar_stack> stacks; bool lazy=false; bool awaiting_trigger=false; }`；`grammar* grammar_init(const GrammarVocab* v, const char* text, const char* root)`（解析失败返回 nullptr；触发器参数留默认 0）；`grammar* grammar_clone(const grammar&)`；`void grammar_free(grammar*)`；`void grammar_accept(grammar*, int32_t token_id, const std::string& piece)`（空栈抛 `std::runtime_error("Unexpected empty grammar stack after accepting piece: ...")`）；`std::vector<grammar_stack>& grammar_stacks(grammar*)`。
  - shim：`compat.hpp` 定义 `GRAMMAR_ASSERT(cond)`（assert）、`GRAMMAR_ABORT(msg)`（throw std::runtime_error）、`GRAMMAR_LOG_ERROR(...)`（fprintf stderr）——替换源文件里的 GGML_ASSERT/GGML_ABORT/LLAMA_LOG_*（位置：GGML_ASSERT L527,767,795,844,944,1356,1399；GGML_ABORT L935,1451；LLAMA_LOG_ERROR L1151,1167,1225,1231,1260；LLAMA_LOG_DEBUG L1408,1436,1440）。

- [ ] **Step 1: 复制两源文件到目标路径**

```sh
mkdir -p F:/Strata-dev/include/strata/grammar F:/Strata-dev/src/grammar
cp F:/Strata/third_party/llama.cpp/src/llama-grammar.h F:/Strata-dev/include/strata/grammar/grammar.hpp
cp F:/Strata/third_party/llama.cpp/src/llama-grammar.cpp F:/Strata-dev/src/grammar/grammar.cpp
```

- [ ] **Step 2: 机械变换（逐条执行，不留尾巴）**

1. 两文件顶部加 `// Ported from llama.cpp (MIT, https://github.com/ggml-org/llama.cpp) ...` 署名注释。
2. includes：删除 `llama-impl.h / llama-vocab.h / llama-sampler.h`，加 `#include "strata/grammar/compat.hpp"`。
3. 全文件 `llama_grammar`→`grammar`、`llama_grammar_element`→`grammar_element`、`llama_gretype`→`gretype`、`llama_partial_utf8`→`partial_utf8`、`llama_grammar_candidate`→`grammar_candidate`、`llama_grammar_parser`→`grammar_parser`、`llama_token`→`int32_t`、`llama_grammar_init_impl`→`grammar_init`（新增薄壳：字符串形式调既有 init，触发器指针传空）、`llama_grammar_free_impl`→`grammar_free`、`llama_grammar_clone_impl`→`grammar_clone`、`llama_grammar_accept`→`grammar_accept`、`llama_grammar_get_stacks`→`grammar_stacks`；删 `apply_impl`（服务器/引擎走位掩路径，不直接改 logits 数组——`llama_token_data_array` 不存在于 strata）。
4. 全包 `namespace strata::grammar { ... }`；`struct llama_vocab*` 成员/参数 → `const GrammarVocab*`；`vocab->token_to_piece(id)`→`vocab->token_to_piece(id)`（GrammarVocab 同名方法）；`vocab->is_eog(id)` 同名；`vocab->tokenize(...)`（L224 `<name>` 分支）→ `throw std::runtime_error("invalid token")`（`vocab==nullptr` 时 L210-212 的抛点保持）。
5. GGML_ASSERT→GRAMMAR_ASSERT、GGML_ABORT→GRAMMAR_ABORT、LLAMA_LOG_ERROR→GRAMMAR_LOG_ERROR（格式串保持）、LLAMA_LOG_DEBUG→删除调用。
6. `grammar.hpp` 里定义 `struct GrammarVocab`（字段与方法见 Interfaces；`token_to_piece` 越界返回 ""）。

- [ ] **Step 3: 写 compat.hpp（完整内容）**

```cpp
// include/strata/grammar/compat.hpp - asserts/logging shims for the ported grammar.
#pragma once
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#define GRAMMAR_ASSERT(cond) assert(cond)
#define GRAMMAR_ABORT(msg) throw std::runtime_error(std::string("grammar: ") + (msg))
#define GRAMMAR_LOG_ERROR(...) std::fprintf(stderr, "strata grammar: " __VA_ARGS__)
```

- [ ] **Step 4: 写失败测试 `src/grammar/grammar_test.cpp`**

移植 `test-grammar-parser.cpp` 的 `verify_parsing`/`verify_failure`（L25-140 逻辑）与金样子集：`{0,5000}` 展开、`{3,5000}`、交替/字符类 `[bdx-z]`/`[^1-3]`、`a+ a? a*`、`{2}/{2,}/{4}/{2,4}`、arithmetic、complex expr、token syntax `root ::= <[1000]> !<[1001]> <[1001]>`（金样 L168-581，用 `TEST_GRAMMAR_PARSER_PRINT_ALL=1` 的打印器先对基线取金样，再固化为断言），加失败例 L144-166（`{0,99}^6` 等）与一个 accept 用例：`grammar_init(vocab=nullptr, "root ::= \"ab\"\n", "root")` → `grammar_accept(g,'a'); grammar_accept(g,'b');` 后 `grammar_stacks(g)` 含空栈；接受 `'c'` 抛 runtime_error。custom `main()`，全部 assert，末行 `std::puts("All grammar tests passed.")`。

- [ ] **Step 5: 注册 ctest 并跑失败**

`CMakeLists.txt` :268-321 块内加（仿 :272-274）：

```cmake
add_executable(grammar_test src/grammar/grammar_test.cpp src/grammar/grammar.cpp)
target_include_directories(grammar_test PRIVATE include)
add_test(NAME grammar_test COMMAND grammar_test)
```

`strata_core` 源列表（:336-337）加 `src/grammar/grammar.cpp`。

```sh
cd F:/Strata-dev && "<CMAKE>" --build build --target grammar_test && ctest --test-dir build -R grammar_test --output-on-failure
```

Expected: 先 FAIL（未变换完/断言红），修完 PASS。

- [ ] **Step 6: 提交**

```sh
git add include/strata/grammar src/grammar CMakeLists.txt && git commit -m "Grammar: port the llama.cpp GBNF core into strata::grammar (parser, stacks, accept; MIT attribution kept)"
```

---

### Task 3: GGUF 词表 + 位掩构建（无 GPU）

**Files:**
- Modify: `include/strata/artifact/gguf_reader.hpp`（`read_value` :361-370 的 64 项数组截断：加 `bool keep_full` 旁路参数 + `get_full()`；不动既有调用方默认）
- Create: `include/strata/grammar/grammar_vocab.hpp` + `src/grammar/grammar_vocab.cpp`
- Create: `include/strata/grammar/grammar_mask.hpp` + `src/grammar/grammar_mask.cpp`
- Modify: `src/grammar/grammar_test.cpp`（追加用例）；`gguf_reader_test.cpp`（全数组用例，目标已存在 :276）
- Modify: `CMakeLists.txt`（grammar_test 源加两个 .cpp）

**Interfaces:**
- Consumes: Task 2 的 `GrammarVocab`。
- Produces:
  - `bool load_grammar_vocab(const std::string& gguf_path, GrammarVocab& out, std::string& err)`——读 `tokenizer.ggml.tokens`（全量，string 数组）、`tokenizer.ggml.token_type`（int32 数组，CONTROL 标记入 `control[]`）、`tokenizer.ggml.eos_token_id` + `eos_token_ids`（数组可缺省）入 `eog[]`。
  - `int grammar_mask_words(int64_t n_vocab) { return (int)((n_vocab + 31) / 32); }`
  - `void grammar_build_mask(const GrammarVocab& v, grammar& g, uint32_t* words)`——bit=1 禁：越界/padding 位禁；EOG 仅任一站空放行（`allow_eog` 语义 L1362-1368 同款）；CONTROL 型禁；其余 token 按 piece 码点推进（`decode_utf8` + 栈推进判定，复用移植内 `reject_candidates` 逻辑）实现为：对每个 token 在**克隆栈**上跑 `grammar_accept`（成功→允许，抛→禁；克隆成本由缓存吸收），结果缓存。
  - 缓存（新代码，vendored 无）：`class GrammarMaskCache { const GrammarVocab& v; std::shared_ptr<grammar> proto; std::list<...> lru; }`，`const std::vector<uint32_t>& mask_for(const std::vector<grammar_stack>& stacks)`——键 = 各栈顶的 (rule_index, element_offset) 有序串（空栈记 -1），LRU 上限 4096 项。

- [ ] **Step 1: 失败测试（追加进 grammar_test.cpp + gguf_reader_test.cpp）**

gguf_reader_test 新用例：构造含 100 项 string 数组的 GGUF fixture（仿该文件既有字节构造法），默认读只留 64 项（回归现状），`get_full()` 全 100 项。
grammar_test 新用例：(a) `GrammarVocab` 手搭 `{ "a","b","<|im_end|>" }`, control[2]=1, eog[2]=1；`grammar_init(v,"root ::= \"ab\"\n","root")`：起始掩词 `bit0=1,bit1=1,bit2=1`（a、b 允许，im_end 禁）；accept a,b 到终态：`bit2=1`（EOG 放行），`bit0/bit1=0`。(b) 缓存命中：同一 stacks 两次 `mask_for` 返回同 `shared_ptr`。(c) 越界 padding 位为 1。

- [ ] **Step 2: 跑红**

`ctest --test-dir build -R "grammar_test|gguf_reader_test" --output-on-failure` → Expected FAIL。

- [ ] **Step 3: 实现 gguf_reader 全数组旁路**

`read_value` 的数组分支（:361-370）保留 64 项默认；新函数 `MetaValue get_full(const std::string& key)`（对 STRING/INT32/INT64 数组项数不设限，其余键型行为同 `get()`）。不新增第三方依赖。

- [ ] **Step 4: 实现 grammar_vocab.cpp / grammar_mask.cpp**

grammar_vocab：打开 GGUF（复用 `GgufFile`），`tokens`/`token_type`/`eos_token_id(s)` 映射进 GrammarVocab；缺 `token_type` 时全部按非 control（Qwen GGUF 均带此键；缺失记录在日志）。
grammar_mask：按 Interfaces 实现；每 token 的 accept 用 `grammar_clone(proto)` 逐 token 试推进——首版直算可接受（引擎侧每步只对**变化后的栈**建掩，缓存命中即免算；开销在 Task 12 实测，超标则此文件内做逐码点匹配优化，不改接口）。

- [ ] **Step 5: 跑绿**

`ctest --test-dir build -R "grammar_test|gguf_reader_test" --output-on-failure` → Expected PASS。

- [ ] **Step 6: 提交**

`git add ... && git commit -m "Grammar: GGUF vocab (full token arrays, control/eog flags) and the bitmask builder with a stack-key LRU cache"`

---

### Task 4: 采样核位掩（CUDA + parity）

**Files:**
- Modify: `include/strata/kernels/sampler.hpp`（SamplerParams :15-30 加两字段；声明处注释）
- Modify: `src/kernels/cuda/sampler.cu`（4 个核的 logit_bias 位点 + greedy cluster 门控）
- Modify: `src/kernels/sampler_parity.cpp`（掩码 parity 块，仿 logit_bias 块 :488-534）

**Interfaces:**
- Produces: `SamplerParams` 新字段 `const uint32_t* grammar_mask = nullptr; int mask_stride_words = 0;`——行 r 的掩在 `grammar_mask + r*mask_stride_words`，bit=1 ⇒ 该行该 token 置 -inf（在 logit_bias 加法**之前**，先掩后 bias：-inf 加 bias 仍 -inf）。
- Consumes: 无（纯核侧；调用方 Task 5/6 接线）。

- [ ] **Step 1: 写失败 parity**

`sampler_parity.cpp` 在 logit_bias 块（:488-534）后加对称块：词表 248320，`mask` 设备缓冲 `cudaMalloc`+`cudaMemcpy`，全禁仅 `{109266}` 允许（bit 全 1、109266 位 0）：`want = greedy/sampled 在全 -inf（除 109266）预置输入`，`got = p.grammar_mask=d_mask` 跑每 ctest 路径（Split/one_block/old + greedy）——`got=={109266}`；被禁 id 永不出现（多次采样计数）；`grammar_mask=nullptr` 与无掩逐位同果；**掩缓冲不可变**（D2H 回读比较，仿 :516-519）；order fixture（:548）上掩禁 top1 后 top2 顺位可被采到（证明位点在 penalties 之前）。

- [ ] **Step 2: 跑红**

`ctest --test-dir build -R sampler_parity --output-on-failure`（GPU 在位）→ Expected FAIL（字段不存在/未应用）。

- [ ] **Step 3: 实现**

`sampler.hpp`：

```cpp
const uint32_t* grammar_mask = nullptr; // per-row device bitmask [row][mask_stride_words]; bit=1 bans the token (-inf), applied BEFORE logit_bias/penalties
int mask_stride_words = 0;              // words per row: (n_vocab+31)/32; 0 when grammar_mask is null
```

`sampler.cu` 四个位点（:161, :358, :604, :733）的取值表达式统一为：

```cpp
const int row = (int) blockIdx.x;                       // split_part 用 blockIdx.y
const uint32_t* mrow = p.grammar_mask ? p.grammar_mask + (size_t) row * p.mask_stride_words : nullptr;
float l = logits[row * n_vocab + v];
if (mrow && (mrow[v >> 5] & (1u << (v & 31)))) l = -INFINITY;
if (p.logit_bias) l += p.logit_bias[v];
```

（逐核适配局部形态，不重构核；`sampler_greedy_cluster` 无掩——门控 :1261 扩成 `multi && shmem == 0 && !p.logit_bias && !p.grammar_mask`，带掩 greedy 回退 `sampler_greedy_kernel` 于 :161 应用掩。）

- [ ] **Step 4: 跑绿**

`ctest --test-dir build -R sampler_parity --output-on-failure` → 3 个环境名全 PASS。

- [ ] **Step 5: 提交**

`git commit -am "Grammar: per-row bitmask in the sampler kernels at the logit_bias site (null default = byte-identical path); parity pins bans, order, and immutability"`

---

### Task 5: 引擎协议：GRAMMAR 注册 + grammar= 键 + 投机互斥

**Files:**
- Create: `include/strata/grammar/grammar_registry.hpp` + `src/grammar/grammar_registry.cpp`（纯主机，可单测）
- Modify: `src/program/generate.cpp`（主循环 :11789 加 GRAMMAR 命令分支；键循环 :12170-12213 加 `grammar=`；`--serve` 启动 info 打印处加 `"grammar":1`——info 打印点实施时 grep `\"version\"` 定位，与 logit_bias 能力位同处）
- Modify: `CMakeLists.txt`（registry 源进 strata_core 列表）
- Create: `src/grammar/grammar_registry_test.cpp`（无 GPU 单测）+ ctest 注册

**Interfaces:**
- Produces:
  - `class GrammarRegistry { bool register_grammar(const std::string& sha16, const std::string& text, std::string& err); std::shared_ptr<GrammarEntry> find(const std::string& sha16); size_t size(); }`，`struct GrammarEntry { std::shared_ptr<grammar> proto; std::shared_ptr<const GrammarVocab> vocab; int mask_words; }`；LRU≤64，被槽位 shared_ptr 钉住的项不逐出（`find` 更新 LRU 头）。
  - 行协议：入 `GRAMMAR <sha16> <byte_len> <text 至行尾>`；出 `GRAMMAR OK <sha16>` / `GRAMMAR ERR <sha16> <one-line message>`（空格换 `_`；每条都 `std::fflush(stdout)`）。sha16 = 服务器算的 sha256 hex 前 16 位，引擎只当缓存键不重算（记录：内容地址由服务器保证）。
  - GEN/BGEN 键 `grammar=<sha16>`（值无空格，过键循环空格分词）：未注册 → `ERR grammar <sha16> not registered`（请求拒绝，服务器端表现为 generate 的 ValueError → 400 语义）；与 `spec_min_p`/引擎 spec（o.spec>0 或 use_mtp，:2591/:8248）共现 → `ERR grammar + speculative decoding is not supported`。
- Consumes: Task 2 `grammar_init`、Task 3 `load_grammar_vocab`（引擎启动时从模型 GGUF 路径构建共享 GrammarVocab——模型路径变量实施时 grep generate.cpp 的 `--model` 选项字段）。

- [ ] **Step 1: 失败单测 `grammar_registry_test.cpp`**

注册 `"root ::= \"ab\"\n"`（sha16 任意 16hex）→ true + find 命中；重复 sha 注册不重解析（entry 指针相同）；语法错文本 → false 且 err 非空；65 项注册后最旧被逐出（find 空）；被占项不被逐出（先 find 保 shared_ptr，再溢出注册，find 仍命中）。

- [ ] **Step 2: 跑红 → 实现 → 跑绿**

`ctest --test-dir build -R grammar_registry --output-on-failure`。

- [ ] **Step 3: 接进主循环**

在 :11789 `for(;;)` 内、GEN 判定之前加分支：行以 `"GRAMMAR "` 开头 → 按空格切 3 段，`byte_len` 校验（不等 → ERR），调 registry，打印 OK/ERR + fflush。键循环加 `grammar=`：存 `req_grammar_sha`（string，空=无）；与 spec 共现 → 打印 ERR 行、跳过本请求（仿 BADM 拒收路径）。info JSON 打印处加 `"grammar":1`。

- [ ] **Step 4: 协议冒烟（无 GPU，纯解析层）**

registry 单测已覆盖解析；主循环行级冒烟并入 Task 12 GPU 验收（需引擎+模型）。

- [ ] **Step 5: 提交**

`git commit -am "Grammar: GRAMMAR registration line + grammar= request key + hard grammar+spec rejection at the engine boundary"`

---

### Task 6: 槽位状态栈 + 每步推进 + 掩环接线（引擎核心）

**Files:**
- Modify: `src/program/generate.cpp`（串行 decode :15864 循环 + BGEN admission :15244 + batch_step/pump :11364/:11632 + BSlot :11163 + d_grammar 环形分配位点仿 d_hist :8820-8825）
- Modify: `src/core/verify.cpp` + `include/strata/core/verify.hpp`（`set_grammar_masks(uint32_t* ring, int words)`；`slot_sp_[slot].grammar_mask` 行指针 :3332/:3778-3782；`set_slot_sampling` :248 旁路带掩行）

**Interfaces:**
- Consumes: Task 4 字段、Task 5 registry/`req_grammar_sha`、Task 3 `GrammarMaskCache`。
- Produces:
  - BSlot 新字段：`std::unique_ptr<grammar> gstack; std::shared_ptr<GrammarEntry> gentry;`（BGEN admission :15244 `sl=BSlot{}` 处从 registry 克隆）。
  - 串行 GEN：`req_grammar_sha` 命中 → 请求局部 `gstack = grammar_clone(proto)`；`d_grammar_mask = cudaMalloc(kMaxSlots*mask_words*4)` 惰性（首请求一次，仿 set_logit_bias :657-672）；提示词预fill 后先建掩行 0 → `sp.grammar_mask = d_grammar_mask; sp.mask_stride_words = mask_words;`；每 token 回读后（:15865-15867 之后）：`grammar_accept(*gstack, next, vocab.token_to_piece(next))` → 重建掩行 0（缓存）→ `cudaMemcpyAsync(d_grammar_mask, host_words, mask_words*4, H2D, token_stream)` → 下一采样。抛异常 → `printf("ERR grammar engine bug: %s\n", e.what()); fflush;` 走请求错误终止（与 EngineNonFinite 同路）。
  - 批式：`set_grammar_masks(d_grammar_mask, mask_words)`；每窗 `run_slot_rows` 回读（verify.cpp:3656/3720）后逐槽 accept+建掩进 `d_grammar_mask + slot*mask_words`，下一窗采样核行 t→槽 t 掩行 t；槽退休（:11491-11495）释放 gstack。
  - `kMaxSlots` = 引擎批槽上限（实施时取 bs[] 数组常量，grep `bs\[`）。
- 生成正确性红线：EOG 允许 ⇒ 槽自然 DONE；掩在采样前上传，同一 token_stream 序保证 H2D 先于核。

- [ ] **Step 1: 写失败 GPU 测试 `tests/program/grammar_decode_test.cpp`**（仿 tests/program 既有 GPU 测试骨架，SKIP_RETURN_CODE 77 于无 GPU，注册 :594 区）

最小模型依赖：该测试**不需要**48 层模型——直调 kernels 层：`d_mask` 分配 → 手搭词表 → 构造 `sampler` 调用（复用 sampler_parity 的 host 框架）验证"每步 accept→掩→采样不越法"200 步随机采样 100 轮全过语法。语法状态机推进逻辑本身已在 Task 2/3 单测覆盖，此处测**采样-推进闭环**（无模型依赖，GPU 只需分配内存）。

- [ ] **Step 2: 跑红 → 实现 generate.cpp/verify 接线 → 跑绿**

`ctest --test-dir build -R grammar_decode --output-on-failure`。

- [ ] **Step 3: 全量 ctest 回归（含 sampler_parity 3 路）**

Expected: 全 PASS；无 grammar 的行与键逐字节不变（无 grammar 代码路径改动行为）。

- [ ] **Step 4: 提交**

`git commit -am "Grammar: per-slot state stacks, per-step advance at the token readback, mask ring H2D into the sampler"`

---

### Task 7: 后端构建（HIP gfx906 / HIP wave32 / SYCL 镜像）

**Files:**
- Modify: `sycl/include/strata/kernels/sampler.hpp`（:15-27 加 grammar_mask/mask_stride_words——注意该镜像无 logit_bias，两字段独立加）
- Modify: `sycl/src/kernels/cuda/sampler.dp.cpp`（5 个取值点按 Task 4 同型加掩：kernel 参数已按值携带结构体 :122/:434/:821/:1037/:1145；行索引取 dpct 后的等价物）
- Modify: `sycl/src/kernels/sampler_parity.cpp`（掩 parity 镜像块）
- Modify: `sycl/CMakeLists.txt`（grammar .cpp 进 strata_resolve 列表 :227/:246/:252/:265/:274——`.cpp` 自动回退上游 src/X.cpp :118-121，语法模块无需 .dp 迁移；确认 strata 目标链接含 grammar）
- Modify: `CMakeLists.txt`（gfx906 glob 只抓 .cu（:2210）——grammar 是 .cpp，已随 strata_core 进 HIP 构建；核对 wave32 hip_backend GLOB :124-133 不吞 .cpp 冲突）

- [ ] **Step 1: CUDA 全目标复查**：`"<CMAKE>" --build build --target strata -j16` → 绿。
- [ ] **Step 2: HIP gfx906**：`"<CMAKE>" -S . -B build-906 -DSTRATA_HIP_GFX906=ON ... && "<CMAKE>" --build build-906 --target strata -j2`（docs/AMD_HIP.md:839 全参数）→ 绿；本机无 AMD 则记 not run（如实报告）。
- [ ] **Step 3: SYCL**：装/找 oneAPI → `"<CMAKE>" -S . -B build-sycl -DSTRATA_ENABLE_SYCL=ON && --build` → 绿；无工具链记 not run。
- [ ] **Step 4: 提交**：`git commit -am "Grammar: sampler mask fields mirrored into the SYCL tree; backend builds verified/recorded"`

---

### Task 8: JSON Schema→GBNF 转换器 CLI（服务器零依赖）

**Files:**
- Create: `tools/grammar_schema/main.cpp`（stdin 读 schema JSON → stdout 写 GBNF；失败 exit 1 + stderr 一行）
- Create: `tools/grammar_schema/vendor/`（从 F:/Strata/third_party/llama.cpp 复制：`common/json-schema-to-grammar.{h,cpp}`、`common/json.{h,cpp}`、`common/json-schema.{h,cpp}`、`common/trie.{h,cpp}`（如有）、`src/unicode.{h,cpp}`、`vendor/nlohmann/json.hpp`——依赖闭包按编译错误闭合，每个文件加 MIT 署名头）
- Modify: `CMakeLists.txt`（仿 tune_hipblaslt :723 加 `add_executable(grammar_schema tools/grammar_schema/main.cpp ...)`）
- Create: `tools/test_grammar_schema.py`（unittest，无 GPU；exe 未建 skipUnless）

**Interfaces:**
- Produces: `grammar_schema[.exe]`：stdin=JSON Schema 文本 → stdout=GBNF（root 名 `root`）；退出码 0/1。调用方 Task 9 子进程使用；转换函数内联 `json_schema_to_grammar(common_json, /*force_gbnf=*/true)`（必须 force_gbnf，绕开 `%llguidance` 出口 L993-1006）。
- Consumes: 无。

- [ ] **Step 1: 失败测试**

`tools/test_grammar_schema.py`：从 vendored `tests/test-json-schema-to-grammar.cpp` 移植 ≥12 个金样（对象必填属性/嵌套/枚举/带 pattern 字符串/数组/min-max 数值/`$ref`/additionalProperties:false 等；断言输出 grammar 逐字符等于 vendored 期望，期望从该测试文件复制；失败例 exit≠0）。运行 `python -m unittest tools.test_grammar_schema` → Expected FAIL（exe 不存在 skip→先建 exe 再转断言）。

- [ ] **Step 2: 复制 vendored 源进 vendor/（复制源 = 主检出 F:/Strata/third_party/llama.cpp，记录路径）+ 写 main.cpp**

- [ ] **Step 3: CMake 目标 + 构建 + 跑绿**

`"<CMAKE>" --build build --target grammar_schema && python -m unittest tools.test_grammar_schema` → PASS。

- [ ] **Step 4: 提交**

`git add tools/grammar_schema tools/test_grammar_schema.py CMakeLists.txt && git commit -m "Grammar: JSON Schema to GBNF CLI built from the vendored llama.cpp converter (sources committed, MIT kept; zero new dependencies)"`

---

### Task 9: 服务器：grammar 参数 + response_format 走语法（NVIDIA/AMD）

**Files:**
- Create: `serve/grammar.py`（注册管理器 + 键 + 能力位 + json.gbnf 常量）
- Create: `serve/grammar_schema.py`（子进程转换器 + sha 缓存）
- Modify: `serve/server.py`（sampling_keys :1367-1437 加 grammar=；`_pump` :1165-1209 路由 `GRAMMAR ` 行到专用队列；StrataEngine 加 `ensure_grammar(text)->sha16`（_send :1446 + 队列等待 OK/ERR；proc 重建清注册集）；`_openai` :6929 / `_responses_prepare` :7114 的 response_format 处理分叉：grammar-capable 引擎 → grammar 路径（直出流 + 末端金丝雀）；`v1_status` :4531-4532 翻 method/constrained_decoding/stream_buffered；裸 `grammar` 参数：validate（非空 str ≤1MB）+ 引擎能力位（仿 logit_bias.py:105 fail-closed）→ 无能力 400）
- Create: `serve/test_grammar.py`

**Interfaces:**
- Produces:
  - `grammar_sha16(text) = hashlib.sha256(text.encode()).hexdigest()[:16]`
  - `StrataEngine.ensure_grammar(text) -> str`：首次 → 发 `GRAMMAR <sha16> <len> <text>` 行（wlock 内）+ 等 `GRAMMAR OK/ERR <sha16>`（超时 10s；ERR → ValueError('grammar rejected: <msg>') → 路由 400）。
  - sampling_keys 尾追加 `grammar=<sha16>`（键名固定 `grammar`；sha 值无空格）。
  - response_format: `json_object` → 内置对象语法（vendored `grammars/json.gbnf` 全文进 `serve/grammar.py` 常量 `JSON_OBJECT_GBNF`）；`json_schema` → 转换器子进程（缓存 OrderedDict sha→gbnf，LRU 64）；grammar 路径不再 `structured_chunks` 缓冲（:5735-5757 跳过），末端跑 `validated_json`（structured.py:192）作金丝雀：失败 → 502 错误 code `structured_output_failed` + 消息含 `grammar engine bug`（绝不静默）。`structured_output.method` → `"grammar"`（能力引擎），`constrained_decoding: True`，`stream_buffered: False`。
  - 能力位：`svc.grammar_ok = engine.info.get("grammar") == 1`（Task 5 引擎已报）；MockEngine 测试注入 info。
- Consumes: Task 5 引擎协议；Task 8 CLI（exe 路径：与引擎 exe 同目录 `grammar_schema.exe`，cfg exe 推导）。

- [ ] **Step 1: 失败测试 `serve/test_grammar.py`**（仿 test_logit_bias/test_structured 骨架：ByteTokenizer + GrammarMockEngine(MockEngine 子类，记录 last_sampling + info={'grammar':1,'logit_bias':1} + ensure_grammar 桩返回 sha) + Service + serve(port=0)）

用例：(1) `response_format json_schema`：断言 prompt 不含 `OUTPUT FORMAT REQUIREMENT`（grammar 路径不走提示词指令）、sampling 捕获含 `grammar_sha16` 期望值、响应 200 + 流有 delta；(2) 末端金丝雀：脚本吐坏 JSON → 502 + `grammar engine bug` 字样；(3) 裸 `grammar:"root ::="` → 400；(4) 能力位缺省引擎（info 无 grammar）：裸 grammar → 400，json_schema → 回落 prompt 路径且 method 声明 prompt_and_validate；(5) `v1_status` 字段翻转断言；(6) 逐字节不变守卫：`sampling_keys({...})`（无 grammar）对冻结 fixture 表逐字节相等（覆盖 seed/tune/logit_bias 组合）。

- [ ] **Step 2: 跑红 → 实现 → 跑绿**

`python -m pytest serve/test_grammar.py serve/test_structured.py serve/test_responses.py -x` → 全 PASS（后两者确保无 grammar 行为零变化）。

- [ ] **Step 3: 提交**

`git commit -am "Serve: grammar= request key, GRAMMAR registration, response_format json_object/json_schema on the engine grammar (direct stream + canary), model card flips"`

---

### Task 10: 工具调用语法构建器 + forced/rescue 跳过 + MCP 回落

**Files:**
- Create: `serve/grammar_tools.py`
- Modify: `serve/server.py`（`_openai` :6926-6951、`_anthropic` :7178-7189、`_responses_prepare` :7104-7148：grammar-capable 且带 tools 且无 MCP → 构建语法注入 sampling；grammar 请求 force=None（跳过 :4652-4653/:4936-4938/:5244-5252 注入）；MCP 工具在场 → 回落 prompt 路径并响应声明 method（新政策，spec §5.5））
- Create: `tools/test_grammar_builder.py`（无 GPU 单测）+ 路由用例并入 `serve/test_grammar.py`

**Interfaces:**
- Produces: `build_tool_grammar(tools: list[dict], tool_choice: str, name: str|None, thinking: bool, convert: Callable[[dict], str]) -> str`——输出 GBNF；`convert` = 属性子 schema→GBNF（转换器，缓存）。序列化规则逐字对齐 chat_template.jinja:132-156（`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n…</function>\n</tool_call>`，多调用 `\n` 相连，前有散文 `\n\n` 开板）。
  - required/named：root 从 `<tool_call>` 起板（named 定死函数名字面量）。
  - auto：`root ::= prose | calls`，prose = 排除字面 `<tool_call>` 的字符类链（`exclude_seq("<tool_call>")` 生成 11 链规则）。
  - 字符串属性：排除 `\n</parameter>` 序列的文本规则（`exclude_seq("\n</parameter>")`）。
  - 非字符串属性：`convert(subschema)` 内联规则。
  - thinking=True：root 前置 think-prologue（`exclude_seq("</think>")` + `</think>` + 模板空白），与模板实测对齐（`tests`: 渲染 chat_template 确认前缀）。
- Consumes: Task 9 的 ensure_grammar/sampling_keys；frontend.tool_choice_of :742（模式解析复用，不重写）。

- [ ] **Step 1: 失败单测 `tools/test_grammar_builder.py`**

(1) `exclude_seq` 生成的规则对边界串行为：`<`、`<too`、`<tool_cal` 允许于 prose；`<tool_call>` 拒绝；围栏代码（\`\`\`）允许。(2) named 语法接受模板逐字序列（用真实 ChatTemplate 渲染一条 tool_call 的文本逐段喂语法 accept——**用 Task 2 移植的 strata::grammar？不**：Python 侧无语法核，断言改为对 GBNF 文本结构断言 + 语法字面量转义正确（`gbnf_escape_length` 规则：转义集 `GRAMMAR_LITERAL_ESCAPES` 对齐 vendored L278-280）。(3) auto 根规则含 prose 交替。(4) thinking=True 前置 prologue 规则存在。(5) MCP 回落选择函数（纯逻辑）：tools+MCP → ("prompt", declared)。

- [ ] **Step 2: 跑红 → 实现 → 跑绿**（`python -m unittest tools.test_grammar_builder`）

- [ ] **Step 3: 路由集成测试（serve/test_grammar.py 追加）**

GrammarMockEngine 捕获：带 tools + tool_choice:"required" → sampling 带 grammar sha、engine.last_prompt 以 `<tool_call>\n<function=` 结尾（语法开板，无 force 注入的重复）；tool_choice:"auto" → 语法含 prose 交替；MCP 在场 → method 声明 prompt_and_validate；`/v1/responses` 路由带 tools 同样过语法（:7104-7148 接线断言）。

- [ ] **Step 4: 全量 serve 测试跑绿 + 提交**

`git commit -am "Serve: tool-call grammars from the template serialization (required/named/auto, parallel, thinking prologue); grammar requests skip forced prefixes; MCP falls back to prompt with the method declared"`

---

### Task 11: 文档同步

**Files:** Modify: `docs/DETAILS.md`（structured output 节：grammar 方法、约束与测量占位→Task 12 数字回填）、`docs/TERMINAL_LAUNCH.md`（无需 launch 面变更——加一句 per-request 说明）、`docs/paper` 不动。

- [ ] **Step 1:** 按 spec §8 的默认与限制清单写入 DETAILS.md（纯词短句、给测量数字带 rig；Task 12 前数字标 TBD-acceptance 并在 Task 12 回填——这是唯一允许的时序占位，两任务由同一执行链完成）。
- [ ] **Step 2:** 提交 `docs: grammar constrained decoding in DETAILS (defaults, limits, measured costs)`。

---

### Task 12: GPU 验收套件（RTX 4090 rig）

**Files:** Create `tools/grammar_acceptance.py`（驱动 OpenAI 端点：JSON 套件 + 工具套件 N 次；plain vs grammar tok/s；BGEN 全活跃多槽 CPU 开销探针；结果 JSON 落 bench/results/）。

- [ ] **Step 1:** 用分支构建的 strata.exe 起服务（指向 rig 现成模型；`serve` 走分支 build exe——engine cfg 指向 `F:/Strata-dev/build/strata.exe`）。
- [ ] **Step 2:** JSON 套件 50 题 × 全过 schema（目标 100%）；工具套件（`strata connect <app> --check` 风格 30 调用 × N）解析失败 0；grammar 请求 `rescued` 恒 0（/v1/status totals）。
- [ ] **Step 3:** 同 rig plain vs constrained tok/s（目标 ≤10% 开销，超出如实报告）；投机关闭代价单独报（引现有 spec 加速测量）；多槽并发掩计算 CPU 上限探针（§9 开放项，实测回填）。
- [ ] **Step 4:** 数字回填 Task 11 文档 + spec §9 开放项逐条定案；结果 JSON 提交。
- [ ] **Step 5:** 提交 `bench: GBNF constrained-decoding acceptance on the RTX 4090 rig (schema 100%, tool parses 0 failures, constrained cost measured)`。

---

### Task 13: 收尾 PR + 报告

- [ ] **Step 1:** 全量回归：ctest 全绿（GPU 项在位）、`python -m pytest tests/terminal`、serve 全量、`npm test`（apps/host，未触及应绿）、`python -m unittest tools.test_grammar_schema tools.test_grammar_builder`。
- [ ] **Step 2:** PR（标题 `[Engine] GBNF constrained decoding: stable tool calls + JSON schema`）：问题→结果→方法→验证（通过/失败/未测分列：HIP/SYCL 若 not run 明说）→限制（投机关闭、每步掩开销数字、Mac/CPU 不覆盖、MCP 回落、GBNF 支持面=vendored 版原样）。引用 Task 1 issue。
- [ ] **Step 3:** 最终报告：改动面、测试结论、Mimosa 提交状态、spec §9 定案、下一步建议（投机+语法交互矩阵、xgrammar 对比留后续）。

---

## Self-Review 结论（写计划时自查）

- Spec 覆盖：§4 数据流→Task 5/6/9/10；§5.1→T2/3；§5.2→T4/6（惰性分配在 T6）；§5.3→T5/9；§5.4→T8；§5.5→T10；§5.6→T9/10；§6→T5/9/10；§7 验收→T2-12（tests/terminal 无 launch 面改动，协议面由 serve 测试覆盖——记录偏离于 Task 13 报告）；§8→T11；§9 开放项→T5（传输编码：文本行走命令行、键行只带 sha）、T5（LRU 钉住）、T10/T9（思考区：think-prologue + response_format 覆盖）、T12（多槽 CPU 实测）。
- 类型一致性：`grammar_init/clone/free/accept/stacks`、`GrammarVocab{pieces,control,eog}`、`GrammarMaskCache.mask_for`、`GrammarRegistry.register_grammar/find`、`SamplerParams{grammar_mask,mask_stride_words}`、`serve grammar_sha16/ensure_grammar/build_tool_grammar/exclude_seq` 跨任务签名一致。
- 占位符：仅 Task 11 的测量数字（Task 12 回填，执行链内闭合），其余无。
