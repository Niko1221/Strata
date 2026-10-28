# Strata 社区基准报告：Apple M1 Max / 64 GiB / Flash-Next Q2_0

测试日期：2026-10-28。按用户指定的 COMMUNITY_BENCHMARKS.md，采用其中推荐的标准基准路线，而非旧版 quick speed check。[1]

结论：官方 runner 状态为 `complete`；1 次预热及全部 21 个正式请求均执行，正式请求无错误、无跳过；12 项 sanity checks 全部通过。必需套件耗时 471.595 秒，不含安装、模型下载、加载与哈希。协议 `strata-qwen38-thinking-v1-draft.1` 仍为 draft，不代表维护者审核、独立复现或网站批准结果。[2]

## 主机和软件

- MacBook Pro，MacBookPro18,2；Apple M1 Max，10 核 CPU（8 性能核 + 2 能效核），32 核 GPU。
- 统一内存 68,719,476,736 bytes = 64 GiB；CPU 与 GPU 共用同一池，不将显存重复计为额外内存。
- 内置 NVMe SSD：APPLE SSD AP1024R，1,000,555,581,440 bytes。模型位于内置 SSD；PCIe 链路速率/宽度未测量，GPU 为片上集成设备。
- macOS 26.6.2，build 25G83；arm64；Metal 4。GPU 驱动随 macOS 提供，无单独驱动版本。
- 接交流电，电池 80%；AC `powermode 0`。没有修改系统电源策略。GPU 功率上限、实际功耗、频率与温度未测量。开始/结束配置快照含系统 thermal warning 查询，不能据此断言运行中没有短暂降频。
- Strata 0.2.7，官方 macOS ARM64 Terminal 包 build 0.1.52，内置 Python 3.12.15；引擎 0.1.52-metal，未修改或本地重编译。
- release/runner commit：`a42cc3d7df70d1940de82cd0882f03e1f5dfc741`。控制组件下载已与 GitHub release asset 的 SHA-256 一致性核验；Metal 引擎由官方安装器验证签名及逐文件内容，并通过试启动。
- 日常桌面环境，非完全隔离的空载机器：Chrome、Adobe 后台组件、WindowServer、Stats、Tailscale 等保留运行；存在系统后台 CPU 负载。oMLX 保持运行，本次没有向其发送推理请求；测试结束 `/health` 显示 healthy、loaded_count=0。不能保证其他进程对 GPU/CPU 完全无影响。

## 模型与配置

- 原版 Qwen3.8-Flash-Next，量化 `Q2_0`。
- 仓库 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`，安装器固定 revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`。
- 文件：`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`、`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`；完整 SHA-256 见 `results.json`。默认 `expert-profile.bin` 同样被 runner 哈希记录。
- tokenizer revision、量化所依据的原始权重 revision、model-card revision 未经本次独立确认，保留官方结果中的 null 和理由。安装器 MTP 下载脚本的固定原始 checkpoint revision 仅能证明草稿层来源，不能冒充量化源权重版本。
- 默认 setup 生成的配置，track `default`；32,768-token context；loopback `127.0.0.1:18080`；capacity 1；未调优，无校准阶段、无自定义专家 profile、未显式开启实验速度投影。
- 有效参数：`--expert-cache 64 --prefill auto --spec 2 --mtp <data>/mtp/rt --max-context 32768 --mmap-experts --vision`；环境 `STRATA_METAL_STREAM=0`，主 GGUF 由 Metal 直接映射。
- 默认专家缓存 64 slots / 0.08 GiB（引擎启动日志）；CPU pool 9 workers 加 host thread；Metal prompt kernel 采用旧代芯片路径，长提示分块 2,048 rows。HC read Q8_0；MTP 层 0.89 GB，草稿词表 106,299 tokens。这些均为引擎自报，不是另行调参。
- KV 类型和 streaming window 没有获得足够的有效运行时证据，记为 not measured，不用 launch 帮助中的默认值替代实测。
- 采样：temperature 1.0、top_p 0.95、top_k 20、min_p 0.0、presence_penalty 0.0、repetition_penalty 1.0、reasoning_effort xhigh；enable_thinking=true、preserve_thinking=true；请求 reasoning_budget_tokens=0。

### 必须说明的操作员备注错误

benchmark 命令的 `--track-notes` 写了 `vision off`，但安装器实际生成并启动了 `--vision` 与 GPU vision helper。`config.json`、`engine.log` 和 `results.json` 的 `extensions.other_engine_options` 一致证明视觉模块已开启。以实际配置为准：视觉模块已加载，本次没有图片请求、没有测量图片准确率或速度。保留 runner 原始文件不做追改，避免隐藏这次声明错误。其余配置为未修改的 setup 默认值。

## 方法与时间

- 官方 runner：`strata bench run --suite standard-v1`；未缩短或修改 corpus、请求顺序、seed、超时、输出上限或判分。
- 刚加载的专用引擎；runner 记录测试前已服务请求数 0。安装后的首次服务已停止，正式测试由 benchmark 自行新建引擎，结束后自动停止。
- 1 次预热，上限 128 tokens，不进入速度统计；随后 short / prompt-4k / prompt-16k 每种 3 次，按重复轮次交错执行，上限 512 tokens，seed 1/2/3；最后 12 项 sanity，各上限 2,048 tokens、seed 1。总共 22 个请求记录。
- 未执行可选的 4 客户端并发扩展、额外 needle、32K/128K 输入、图片或长时间 soak。这些不是本次标准必需阶段。
- 标准阶段开始：2026-10-28T05:10:58Z；含预热的标准必需阶段 471.595 秒。整个 wrapper（加载、哈希、采集开始/结束配置在内）519.672 秒。安装下载和初始准备不计入上述两项。
- prompt/decode 吞吐来自 engine token counts 和各自计时，包含 reasoning，不以生成 tokens 除以请求总时间冒充 decode。
- TTFT 为客户端从 dispatch 到第一段非空 reasoning/answer/tool delta；本次所有性能请求首先返回 reasoning。首答案延迟单独统计。
- 各性能请求实际 cached_tokens 均为 0，read_tokens 等于完整输入长度；引擎声明 conversation cache 因内存 guard 无法读取可用 RAM而关闭。不能把这种状态泛化为冷模型：专家及系统页缓存留在同一个会话中，首次预热还包含 kernel 准备。

## 性能：中位数 [最小–最大]

每种长度均有 3 个样本；速度 tok/s，延迟 s。short 的 prompt rate 主要受请求固定成本影响，按官方读出方式不作 headline。

| 实际输入 tokens | Prompt tok/s | Decode tok/s | TTFT s | 总延迟 s | 达到 512 cap |
| ---: | --- | --- | --- | --- | --- |
| 77 | 不作 headline，见 JSON | 37.46 [36.95–37.48] | 0.828 [0.807–0.829] | 14.279 [14.218–14.306] | 0/3 |
| 4,093 | 376.57 [338.47–376.63] | 36.66 [35.97–36.99] | 10.905 [10.902–12.130] | 21.838 [21.256–26.334] | 1/3 |
| 16,387 | 369.63 [369.10–369.76] | 34.75 [32.57–35.32] | 44.374 [44.359–44.438] | 59.078 [58.902–60.045] | 3/3 |

首答案延迟：short 中位数 2.278 s（3 个有效样本）；4K 15.212 s（3 个）；16K 52.204 s（2 个）。16K 第 3 次在 reasoning 阶段耗尽 cap，没有首答案时间，不补零、不伪造样本。达到 cap 的请求可进入速度统计，但总延迟是到 cap 的时间，并不意味着已经完成答案。

### 全部性能重复

| 请求 | 输出 tokens（reasoning+answer） | Prompt tok/s | Decode tok/s | TTFT s | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| short r1 | 500 | 99.586 | 36.955 | 0.807 | completed |
| 4K r1 | 512 | 338.469 | 35.968 | 12.130 | truncated |
| 16K r1 | 512 | 369.102 | 35.324 | 44.438 | truncated |
| short r2 | 503 | 96.819 | 37.479 | 0.828 | completed |
| 4K r2 | 384 | 376.565 | 36.987 | 10.905 | completed |
| 16K r2 | 512 | 369.630 | 34.747 | 44.374 | truncated |
| short r3 | 505 | 96.624 | 37.463 | 0.829 | completed |
| 4K r3 | 402 | 376.631 | 36.656 | 10.902 | completed |
| 16K r3 | 512 | 369.762 | 32.574 | 44.359 | truncated |

## 正确性、内存与失败

- Sanity：12/12；instruction I1–I4 4/4、retrieval R1–R4 4/4、tool_call T1–T4 4/4。已从保存的 final_answer / structured tool calls 按冻结 fixture 重新独立判分，不判 reasoning，不调用真实外部工具。
- 预热 truncated 1；正式性能 completed 5、truncated 4；sanity completed 12。HTTP/stream/request failed 0、skipped 0，无重试。
- 按约 5 秒间隔采样，102 个主机遥测样本覆盖加载、哈希和标准套件。主引擎最高观测 RSS 42,211,966,976 bytes（39.313 GiB）；vision helper 最高观测 RSS 1,233,092,608 bytes（1.148 GiB）。是采样观测值，不是精确峰值，也不是 GPU 专用显存用量；共享映射不能简单相加。
- system swap used 从首样本到末样本均为 291.31 MiB（sysctl 的两位小数精度），所有采样最大值也相同；存在此前遗留 swap，不能声称主机从未换页。原始 vm_stat、swap 和进程采样保留。
- 引擎报告可用 RAM guard 无法读取、停车/session save-restore 被拒绝，conversation cache 关闭；PCIe probe failed 使用默认 fraction。这些启动诊断保留在日志，不当成已验证的硬件 PCIe 速率。测试中没有 OOM 或请求失败记录。
- IQ2_XS 初始安装尝试被官方 Mac 安装器拒绝（不提供该量化），经用户确认改用 Q2_0；失败日志保留为 `install-launch.log`。这不是推理基准失败，不混入正式请求统计。
- 只有一个硬件/模型/配置组合、每长度三次，不宣称稳定 p95、模型整体能力或跨引擎优劣；sanity 通过不代表完整编码质量评估。

## 证据与复现

- `README.md`：官方自动生成英文报告（上面的 vision 备注勘误优先）。
- `results.json`：官方完整结果、计数、逐请求数据、15 组摘要、missing 原因。
- `outputs.jsonl`：全部 22 个请求的输入、输出、usage/timings 和客户端事件时间。
- `config.json`、`engine.log`、`provenance/`：有效配置、引擎日志、模板和 corpus manifest。
- `host-test-start.json` / `host-test-end.json`：实际测试前后的主机配置；`host-telemetry.jsonl`：运行期间原始主机采样。根目录原始主机文件含本机路径/进程信息，仅供本地使用，公开前必须进一步脱敏。
- `installation-provenance.json`、`benchmark-console.log`、`test-command.json`、`test-execution.json`：版本来源、控制台日志、精确命令参数及 wrapper 执行时间。
- `independent-verification.json`：独立验证结果。检查 22 个唯一请求、各阶段计数、6 个官方 artifact 哈希、15 组统计重建、usage/token/timer 一致性以及 12 项重新判分。
- `verify_results.py`：对保存结果的独立核验脚本，不执行推理；`capture_host.py` / `run_with_host_capture.py`：主机采集与调用官方 runner 的 wrapper。
- oMLX 顶层 JSON 配置哈希与测试前一致，服务保持 healthy；没有更改其模型或生产参数。

安装命令：

```sh
<STRATA_DIR>/bin/strata launch --family qwen --model q2_0 --data-dir <DATA_DIR> --port 18080 --access local --tune skip --no-tui --yes --save-preset 'Benchmark Q2_0 32K'
```

正式测试的精确 argv 已保存在 `test-command.json`。再次执行会是新的运行，必须使用新输出目录并重新加载引擎，不覆盖本次证据。最小复现骨架如下（vision 备注采用已核对值）：

```sh
<STRATA_DIR>/bin/strata bench run --profile launch-benchmark-q2_0-32k --suite standard-v1 --track default --track-notes 'Strata 0.2.7; setup defaults; Q2_0 32K; vision on; tuning skipped' --quantization Q2_0 --model-revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --background 'Record the actual background applications for the new run' --out NEW_DIRECTORY
<STRATA_DIR>/bin/strata bench report <REPORT_DIR>/standard
python3 <REPORT_DIR>/verify_results.py
```

本报告原始记录生成时尚未公开；此次提交仅包含经检查的基准证据与脱敏主机配置，不包含原始进程列表、模型权重或私人路径。

## Sources

[1] https://github.com/Niko1221/Strata/blob/main/docs/COMMUNITY_BENCHMARKS.md
[2] https://github.com/Niko1221/Strata/blob/main/docs/BENCHMARK_PROTOCOL.md
