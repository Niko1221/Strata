# Community benchmark on NVIDIA GeForce RTX 2080 Ti
# NVIDIA GeForce RTX 2080 Ti 社区基准测试

Measured on 2026-10-31 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges) plus
long-context recall with `tools/needle_bench.py` (6 checks, 6/6 found). Speeds are the engine's own timings;
first-token times are measured by the client over HTTP.
于 2026-10-31 使用 `tools/strata_bench.py` 测量（默认测试组，每项 3 次运行，取中位数与范围），
另用 `tools/needle_bench.py` 做长上下文召回检查（6 项，6/6 全部命中）。吞吐为引擎自身计时；
首 token 时间由客户端通过 HTTP 测量。

## Hardware and software / 硬件与软件

- GPU and VRAM: NVIDIA GeForce RTX 2080 Ti, 22 GiB VRAM (2 cards, layer split across both);
  CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 121.5 GiB; storage: NVMe (nvme0n1 UMIS RPITJ512PED2OWX); PCIe: gen3 x8
- GPU / 显存：NVIDIA GeForce RTX 2080 Ti，22 GiB 显存（双卡，层分割）；
  CPU：AMD Ryzen 9 5950X（16 核 32 线程）；内存：121.5 GiB；存储：NVMe（UMIS RPITJ512PED2OWX）；PCIe：gen3 x8
- OS; driver; CUDA: Linux 7.0.0-38-generic; 580.178.04; CUDA 13.0 (driver supports up to)
- 操作系统 / 驱动 / CUDA：Linux 7.0.0-38-generic；580.178.04；CUDA 13.0
- Strata commit; engine version; release binary or source build: project 0.1.55, commit not a git checkout (installed from a release archive); engine 0.1.55 (Strata v0.2.9)
- Strata 提交 / 引擎版本 / 构建方式：project 0.1.55，非 git 检出（从发布包安装）；引擎 0.1.55（Strata v0.2.9）
- Background workloads and power limits: no other GPU process during the run; GPU power limit 250 W, power draw after the run 170.76 W (near idle), temp 61 °C, pstate P2
- 背景负载与功率限制：测试期间无其他 GPU 进程；GPU 功率上限 250 W，测试后近空载功耗 170.76 W，温度 61 °C，pstate P2

## Model and configuration / 模型与配置

- Model repository and quantization; GGUF filenames: `qwen3.8-flash-next-unsloth-ud-q4_k_xl` (Unsloth UD-Q4_K_XL, 4 shards + mmproj vision file)
- 模型与量化：`qwen3.8-flash-next-unsloth-ud-q4_k_xl`（Unsloth UD-Q4_K_XL，4 个分片 + mmproj 视觉文件）
- Vision encoder; custom packs or profiles: vision on (mmproj-Qwen3.8-Flash-Next-BF16); native expert pack + expert-profile.bin, MTP draft (`--mtp .../mtp/rt`)
- 视觉编码器 / 自定义包：视觉开启（mmproj-BF16）；原生专家包 + expert-profile.bin，MTP 草稿
- Context; KV type and streaming; cache; prefill; low-RAM mode: context 262,144; `--kv int8` with `--kv-resident 32768`; expert cache auto (filled to 13.36 GiB VRAM at start); `--prefill 16384`; `--ple-io ram`; `--vram-reserve-mib 512`
- 上下文 / KV / 缓存 / 预填充：上下文 262,144；`--kv int8`（常驻 32768）；专家缓存 auto（启动填充 13.36 GiB 显存）；`--prefill 16384`；`--ple-io ram`；`--vram-reserve-mib 512`
- MTP; reasoning; sampling; calibration; experimental speed projection: MTP on (`--spec 4 --spec-min-p 0.5`); thinking off for the suite (default); temperature 0; no calibration; no experimental speed projection
- MTP / 推理 / 采样：MTP 开启（`--spec 4`）；测试组关闭 thinking（默认）；temperature 0；未启用校准与实验性提速

```text
/home/yangzhixuan/.../Strata/engine/strata --serve --pack .../data/packs/unsloth-ud-q4_k_xl \
  --native .../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --expert-profile .../expert-profile.bin \
  --expert-cache auto --prefill 16384 --spec 4 --spec-min-p 0.5 --mtp .../mtp/rt \
  --max-context 262144 --kv int8 --kv-resident 32768 --vision --pipeline-windows 2 --prefill-pipe 1 \
  --ple-io ram --vram-reserve-mib 512 --layer-split auto
```

(The server is the install's systemd user service on port 8081; home paths and the API key are removed.)
（服务器为该安装的 systemd 用户服务，端口 8081；home 路径与 API key 已移除。）

## Method / 方法

Run by `python tools/strata_bench.py --long` against the HTTP API of an already-running Strata server; model
loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique
tag, so the conversation cache cannot help the fresh-prompt tests.
使用 `python tools/strata_bench.py --long` 对已运行的 Strata 服务器发请求；模型加载不计时。
先丢弃一次热身聊天和 1K 提示词。每个提示词带唯一标签，会话缓存不会协助新提示词测试。

- short chat: three questions, up to 100 new tokens, 3 runs / 短聊天：3 个问题，最多 100 个新 token，3 次
- 4K / 16K / 32K prompts: generated word lists, 24 new tokens, 3 fresh prompts each / 4K、16K、32K 提示词：生成词表，24 个新 token，各 3 个新提示词
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words and continue from the model's own earlier answers, 64 new tokens per turn, 3 sessions / agent 会话：系统提示词 + 4 个工具，随后 5 轮各追加约 1500 词并基于模型自身上轮回答继续，每轮 64 个新 token，3 个会话
- recall: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` / 召回：`needle_bench.py`，长度 32K、128K，深度 10%、50%、90%

## Results / 结果

| Configuration 配置 | Actual prompt tokens 实际提示词 | Reused tokens 复用 | Generated tokens 生成 | Runs 次数 | Prompt tok/s median and range 提示词吞吐 中位数（范围） | Decode tok/s median and range 生成吞吐 中位数（范围） | TTFT seconds median and range 首token秒 中位数（范围） |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat 短聊天 | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 56.5 (48.5-66.7) | 0.49 (0.42-0.67) |
| 4K prompt, fresh 全新4K | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 440.7 (440.4-441.8) | not measured | 9.14 (9.12-9.15) |
| 16K prompt, fresh 全新16K | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1225.5 (1223.2-1228.8) | not measured | 13.13 (13.10-13.16) |
| 32K prompt, fresh 全新32K | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 1551.6 (1545.0-1556.8) | not measured | 20.70 (20.63-20.78) |
| agent session, follow-up turns agent会话续轮 | 12941 (12928-12952) at the last turn 末轮 | see below 见下 | up to 64 | 3 | see below 见下 | 64.7 (57.0-72.4) | 8.89 (8.47-9.28) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.
范围为各次运行的最低与最高值；`Runs` 为返回了 token 的请求数。

Prompt throughput rises with length (440 → 1,226 → 1,552 tok/s at 4K/16K/32K), the typical ramp for a
dual-card layer split; recall time at 128K was 37-60 s (about 2,000 prompt tok/s).
提示词吞吐随长度上升（4K/16K/32K 为 440 → 1,226 → 1,552 tok/s），是双卡层分割的典型曲线；
128K 召回耗时 37-60 秒（约 2,000 提示词 tok/s）。

### Agent session (conversation cache) / Agent 会话（会话缓存）

Cache hit over the follow-up turns: 69.4 % (68.9-69.5); earlier-turn tokens read again per session: 4 (3-214);
first turn (nothing cached) first token: 6.22 s (5.97-6.27).
续轮缓存命中率：69.4%（68.9-69.5）；每会话重复读取的早期轮 token：4（3-214）；
首轮（无缓存）首 token：6.22 秒（5.97-6.27）。

| Turn (run 1) 轮次 | Prompt tokens | Reused 复用 | Read 重读 | Prompt tok/s | First token (s) 首token秒 | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 102.4 | 5.973 | 51.5 |
| 1 | 3362 | 634 | 2728 | 307.6 | 8.894 | 59.7 |
| 2 | 4895 | 3425 | 1470 | 174.1 | 8.467 | 59.1 |
| 3 | 8139 | 4958 | 3181 | 349.6 | 9.127 | 62.0 |
| 4 | 10205 | 8200 | 2005 | 232.0 | 8.67 | 68.8 |
| 5 | 12952 | 10268 | 2684 | 301.2 | 8.938 | 63.2 |

Per-request data: `results.json`. / 逐请求数据：`results.json`。

## Correctness and limitations / 正确性与局限

Long-context recall with `tools/needle_bench.py` (one needle at 10%, 50% and 90% depth of the text), run
after the speed suite against the same server: **6 of 6 found**.
使用 `tools/needle_bench.py` 的长上下文召回（在文本 10%、50%、90% 深度各放一个针），测速后对同一服务器执行：**6/6 全部命中**。

| Length 长度 | Actual prompt tokens | Depth 10% 深度10% | Depth 50% 深度50% | Depth 90% 深度90% | Prompt seconds 耗时秒 |
| --- | ---: | --- | --- | --- | --- |
| 32,768 | 31,706-31,707 | FOUND 命中 | FOUND 命中 | FOUND 命中 | 21 |
| 128,000 | 122,937-122,939 | FOUND 命中 | FOUND 命中 | FOUND 命中 | 37-60 |

Per-case data: `needles.json`. / 逐项数据：`needles.json`。

Otherwise this is a speed suite: the conversation uses the model's own replies, so token counts differ a little
between runs. Machine details come from the server, `nvidia-smi` and the OS; background load is not recorded
beyond the GPU processes check. Total time of the speed suite: 333 s; needle checks: 180 s. The install is a
release archive rather than a git checkout, so the commit is not recorded.
其余为测速套件：会话使用模型自身回答，各次运行的 token 数略有差异。机器信息来自服务器、`nvidia-smi`
与操作系统；除 GPU 进程检查外未记录背景负载。测速套件总时长 333 秒；召回检查 180 秒。
该安装来自发布包而非 git 检出，故未记录 commit。
