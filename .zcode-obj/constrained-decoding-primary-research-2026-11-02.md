# 约束解码性能：DeepInfer、llama.cpp 与 Strata

核查日期：2026-11-02。仅研究源码；本轮没有启动推理、改运行配置或修改引擎实现。目标是保留严格的工具参数约束，同时让自由正文避免不必要的语法计算。以下不能视为对 Cherry Studio 慢速问题的性能剖析结果。

## 核查范围

- DeepInfer 官方提交：`553741557a57df44a46fb3ab72b6aafeaf945337`。
- DeepInfer 的依赖固定为 `xgrammar==0.2.2`；核查 XGrammar `v0.2.2`，不将其他版本的优化归到它名下。
- llama.cpp：核查官方 `master` 的相关源码，同时对照本机 `F:/Strata/third_party/llama.cpp`。本机 HEAD `a9cc2c201b42823d5120a1613d88a7633adea7cf` 的提交说明含本地配置变动，不能把该哈希直接当作可公开访问的官方版本。下列 llama.cpp 本地行号只指本机快照。
- Strata：本机工作区。静态成本分析与实测归因分开。

## 1. llama.cpp：先判断是否需要语法

`common/chat-auto-parser-generator.cpp` 的自动模板路径，在存在触发标记、`tool_choice:auto` 且没有结构化 response_format 时启用 `grammar_lazy`。required 工具和 response_format 不使用同一条件。

`src/llama-grammar.cpp` 的 `llama_grammar_apply_impl` 在 `awaiting_trigger` 时直接返回，不执行后面的候选词表语法过滤。触发 token 或 pattern 命中后才开始匹配，并重放与触发点重叠的 token 文本。它不是简单的“看到完整标记以后，忽略该 token 里剩下的字符”。

`common/sampling.cpp` 的 `grammar_should_apply` 在 lazy grammar 的推理阶段抑制语法应用；推理结束时会处理结束序列可能包含的触发标记。`grammar_first=false` 分支先检查已经采到的一个候选，合法就返回，不合法才完整过滤并重采。这是具体分支的优化，不代表所有采样策略都能无条件照搬。

一手来源：
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/chat-auto-parser-generator.cpp`，`peg_generator::generate_parser`。
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-grammar.cpp`，`llama_grammar_apply_impl`、`llama_grammar_accept_impl`。
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/sampling.cpp`，`grammar_should_apply`、`common_sampler_sample`。
- 本机对应位置：`common/chat-auto-parser-generator.cpp:79`、`src/llama-grammar.cpp:1355`、`common/sampling.cpp:452`、`:594`。

## 2. DeepInfer：触发检测、编译缓存、逐草稿位置掩码

`deepinfer/structure_output/matcher.py` 的 KMP 工作在触发 token ID 序列上。触发前且 bound 未触发时，matcher 只推进触发状态；sampler 仅在 `matcher.constrained()` 为真时请求词表掩码。配置了 bound 时会强制补全触发序列，不能概括成所有请求触发前都绝对无约束。

`compiler.py` 把 JSON Schema、regex、EBNF 交给 XGrammar，配置编译缓存、线程数和缓存容量。DeepInfer 不是在 Python 中自行逐个 token 全量解释 JSON Schema。

`sampler.py::_get_so_mask` 先提交上一轮真正接受的 token，再沿草稿的合法前缀临时推进 matcher，为每个位置建立掩码，随后 rollback。触发进度也有回滚历史。`_verification` 在采样与草稿验证之前应用相应位置的掩码。

传输使用复用的 pinned CPU mask buffer、独立 CUDA copy stream、non_blocking 拷贝和 event 同步。无 SO 请求有早退路径；SO 等待触发时仍可能承担状态推进、循环和传输框架的成本，不能承诺零开销。

一手来源（均在上述固定 DeepInfer 提交下）：
- `https://raw.githubusercontent.com/deepseek-ai/DeepInfer/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/structure_output/matcher.py`
- `https://raw.githubusercontent.com/deepseek-ai/DeepInfer/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/structure_output/compiler.py`
- `https://raw.githubusercontent.com/deepseek-ai/DeepInfer/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/sampler.py`
- `https://raw.githubusercontent.com/deepseek-ai/DeepInfer/553741557a57df44a46fb3ab72b6aafeaf945337/requirements.txt`

## 3. XGrammar：把可复用的词表工作移到编译期

`v0.2.2/cpp/grammar_matcher.cc::FillNextTokenBitmask` 使用预解码排序词表和编译后的 adaptive token mask cache：组合已知接受/拒绝集合，运行时重点检查 uncertain tokens；复用相邻 token 的公共前缀，跳过已经判拒的 trie 子树。它仍有运行时状态匹配与 GPU 掩码应用成本，不是把所有 schema 匹配都变成常数时间。

同文件提供草稿树遍历、逐节点掩码和 rollback。该能力属于 XGrammar，不能仅凭库接口存在就声称 Strata 的所有草稿模式已接入。

一手来源：`https://raw.githubusercontent.com/mlc-ai/xgrammar/v0.2.2/cpp/grammar_matcher.cc`。

## 4. Strata 当前可直接确认的成本

这些是源码事实；尚未测量各项占本次 100 vs 200 tok/s 差距的比例。

1. `serve/grammar_tools.py:263`：auto 工具把自由正文纳入 `prose | calls | prose ... calls`；`:272` 把 reasoning 纳入排除自动机。普通正文仍在一个活跃的完整 grammar 内。
2. `src/grammar/grammar_mask.cpp:75`：`make_key` 每次遍历传入语法的全部 rules/elements，建立指针到 `(rule, offset)` 的 unordered_map。`:106` 在命中缓存之前调用它。因此“mask cache 命中”并不意味着没有随语法规模增长的工作。
3. `src/grammar/grammar.cpp:1330`：`grammar_clone` 复制规则和栈，并通过规则扫描重定位栈指针。
4. `src/program/generate.cpp:17289`：每个有草稿的语法窗口创建 scratch clone；`:17296` 逐位置推进；`:17300` 每行分别调用 `cudaMemcpy` 上传掩码。
5. `src/grammar/grammar_mask.cpp:29`：缓存未命中时分配词表大小的容器，并解码、检查候选 token。

因此，大工具 schema 即使没有正在输出 JSON，也可能通过缓存键构造和窗口克隆增加成本。这比仅凭工具数量、上下文长度或接受率下结论更具体，但仍需 profiler 证实。

## 5. 对 Strata 的建议，按实施顺序

### 先消除不改变语义的重复工作

- 编译规则只存一份不可变结构，状态使用稳定规则 ID；缓存查找不再扫描整份语法。
- 草稿只快照/回滚可变状态，避免每窗口复制所有规则。
- 复用掩码缓冲区与相同状态结果；合并多行上传，验证同步依赖后再做异步重叠。
- 将缓存键、mask 构建、clone、CPU→GPU 拷贝分别计时。

### 再建立正文快速路径

把协议状态与工具参数 grammar 分离：自由正文/推理阶段只运行轻量边界状态机，进入结构化片段才运行 schema matcher。required 工具和纯 JSON 请求应在需要约束的位置立即约束，不能都等待一个可选触发标记。

严格性要求：处理标记跨 token、一个 token 同时包含标记和参数、推理内的字面工具标记、多个调用与结束后正文。当前 parser 会救回推理里的工具标记，必须同步规定它什么时候可以当成真实调用；不能仅停止 masking 后保留原 rescue 行为。

### 投机解码保持统一协议

每个 verifier row 必须使用该行前缀对应的触发状态和语法状态。草稿跨越正文→工具边界时，从正确位置启用掩码；拒绝后的状态不提交。概率式接受的目标分布仍须是受约束并归一化后的 p，不能只做 token 合法检查后继续用未约束概率。

DeepInfer 的 DFlash 类只接受特定 `DFlashNextPredictHeadHC`，README 称其方案为 DSpark。这是可借鉴的投机状态管理证据，不是 Strata `--draft dflash2` 已兼容的证明；后者需要单独接线和正确性/性能测试。

来源：`https://raw.githubusercontent.com/deepseek-ai/DeepInfer/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/mtp/dflash.py`；llama.cpp `common/sampling.cpp::common_sampler_sample_and_accept_n_rejection`。

## 6. 已有测量不能支持的结论

- 之前所谓“完整 Cherry 请求复现”使用了简化 system 和工具 schema，不是原始请求逐字重放。
- tools 有/无会改变提示词、模型行为及输出长度，不能作为 grammar 开/关的单变量比较。
- 部分脚本只取日志最后一条记录，推理预算续写会产生多条；因此部分速度不是整次请求的速度。
- 首轮慢尚未分解为编译、掩码缓存、图捕获或 GPU 状态，不能直接称为“mask kernel 预热”。

下一次验证应保存原始请求（不记录认证头），保持 tools/messages/采样设置不变，交替设置 `strata_grammar`；分别记录每轮引擎解码、TTFT、全部续写和客户端工具耗时。先明确成本，再优化。没有证据承诺“最优”、零开销或必然恢复某个 tok/s。
