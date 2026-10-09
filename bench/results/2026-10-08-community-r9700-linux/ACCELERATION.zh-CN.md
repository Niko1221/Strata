# R9700 单卡 Qwen3.8 Flash Next 加速总结与配置

整理日期：2026-10-09。测量日期：2026-10-08。测试版本为 Strata 0.1.40.3，
测量来自原实验分支 `perf/r9700-single-gpu`，最近一次速度测试对应源码提交 `25d2e33`。
本 PR 仅提交报告与复现资料；文中“当前版本/当前分支”指测量时的实验版本，不指本 PR 的上游基线。
模型为 ISTA GSQ-RCO Qwen3.8-Flash-Next，IQ3_S 主测，IQ2_XS 作独立量化对照。

当前 IQ3_S 配置在本机单张 R9700 上，1K 至 128K 输入的生成速度中位数为
**87.6–101.9 tokens/s**。本轮新增的短输入专家分组 gather，在保留 32K 历史后
新增 512/900 tokens 的受控实验中，分别将首 token 延迟降低 **16.18%/14.07%**。
这两组结论使用不同配置：前者描述当前速度，后者衡量一个代码改动的收益。

已保留 gfx1201 Q2_0 正负零正确性修复。分组 gather 实现仍为实验性移植：
256-token 增量存在大幅波动，长输入和生成阶段也记录到退化，尚未通过全场景性能验收。
完整决定见 [采纳记录](evidence/pr1107-decisions.json)。

## 测试环境与计时口径

| 项目 | 实测条件 |
| --- | --- |
| GPU | 单张 Radeon AI PRO R9700，32 GB，gfx1201；BDF `0000:63:00.0` |
| CPU 与内存 | 双 EPYC 9334，系统可见约 503 GiB RAM；本次使用 15 个 engine workers |
| CPU 与内存位置 | CPU 节点 0、1；内存优先节点 1，允许回退到其他节点 |
| 软件 | Linux，系统 ROCm 7.2.3，HIP 7.2.53211，hipBLASLt 100202 |
| 构建 | Release，gfx1201，native IQ experts 与 HIP prefill MMQ 开启 |
| 请求 | 并发 1，固定 token 输入，greedy，性能请求各输出 256 tokens |
| 单卡校验 | ROCr UUID 隔离，HIP 只枚举一张卡，并核对逻辑设备 0 的 BDF |

单卡运行仍会使用 CPU、系统 RAM 和 PCIe；上述内存是测试机容量，不是模型最低需求。
这些速度不能直接代表桌面 CPU、较少 RAM 或其他量化版本。

预填充速度表示读取新输入的 tokens/s；生成速度表示输出 tokens/s；TTFT 表示从提交
原生引擎请求到收到首个 token 的时间。所有性能表排除模型加载与声明的预热，
TTFT 不含 HTTP/SSE、排队和客户端网络时间。无提示复用也不等于冷文件缓存。

## 当前版本的单卡速度

本次使用自动专家缓存和自动 prefill，开启 MTP、int8 KV，选择 Tensile 后端。
每档完整预热一次，再在同一引擎会话中交错测量三轮，合计 12 个正式请求。
输入为记录格式总结与 Python 解析代码任务，每次完整读取输入，`reused=0`。

下表为中位数，括号内为三次测量的最小值至最大值；1K = 1,024 tokens。

| 输入长度 | 预填充 tokens/s | 生成 tokens/s | TTFT 秒 | 整个请求秒 |
| --- | ---: | ---: | ---: | ---: |
| 1K | 632.9（514.8–634.0） | 101.9（101.2–102.8） | 1.64（1.64–2.01） | 4.15 |
| 4K | 768.1（716.3–768.8） | 96.4（94.4–100.6） | 5.35（5.35–5.74） | 8.05 |
| 32K | 750.5（750.4–775.4） | 95.6（90.4–98.2） | 43.69（42.29–43.70） | 46.29 |
| 128K | 718.4（713.9–719.0） | 87.6（85.3–91.4） | 182.48（182.33–183.64） | 185.39 |

这些是当前组合的绝对速度，没有同设置的旧版配对，不能把它们全部归功于新增补丁。
各轮生成 token 序列和 MTP 接受数量存在变化；速度也会随任务文本和草稿接受率变化。
本次复用与当前引擎源码一致的既有 binary，没有重新构建。
数据见 [速度汇总](current-speed/summary.json)、
[原始请求](current-speed/iq3_s-session/results.json)
和 [版本记录](current-speed/manifest.json)。

## 短输入专家分组 gather 的收益

### 实现改动

提交 `269a140` 移植了上游 PR #1107 的固定版本 `1a58f78`，改动位于
[prefill.cpp](patches/pr1107-candidate.patch)。来源记录见
[补丁溯源](evidence/pr1107-source-provenance.json)。

MoE 预填充需要把本次使用的专家权重组织到 MMQ 的输入缓冲区。原有长输入路径已经
支持分组 gather，短输入的 staged 路径仍逐专家等待复制、执行 gather、记录释放事件。
补丁让短输入中已在显存和临时传入显存的专家也能一起分组处理：

1. 每组最多 16 个专家，共用 gather 和同步操作，减少逐专家的调度开销。
2. 分组短输入使用 32 个 staging slots；限制提前复制的数量，避免一组尚未读取的权重被覆盖。
3. 分配缓冲区和计算所需字节数统一使用 `stage_slots()`，使显存预算包含全部 staging 空间。

收益针对专家准备与同步阶段。HIP gather 测试覆盖两种量化共九种真实格式布局、
45 个用例，以及尾组、缓冲区边界和未对齐拒绝；完整模型再检查状态、生成和 MTP 工作量。
测试及构建依据见 [实施记录](README.md#correctness-and-limitations)。

### IQ3_S 五对独立会话

两版均使用 Tensile 和相同的受控配置，保留 32,768 tokens 历史，每次新增指定数量的
tokens 后生成 256 tokens。先做两对筛选，再用未改变的配置完成三对确认，每个引擎
都做完整预热。对应请求的输入、输出、实际复用/读取量及 MTP 计数一致。

| 新增 tokens | 原版 TTFT 中位数 | 候选 TTFT 中位数 | 配对时间降低中位数 | 全部配对范围 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.327 秒 | 1.100 秒 | 16.42% | −23.87% 至 +33.90% |
| 512 | 1.868 秒 | 1.570 秒 | **16.18%** | +15.07% 至 +25.58% |
| 900 | 2.545 秒 | 2.185 秒 | **14.07%** | +13.49% 至 +14.27% |
| 2,048 | 4.045 秒 | 4.060 秒 | −0.32% | −0.80% 至 −0.11% |
| 4,096 | 6.623 秒 | 6.640 秒 | −0.30% | −4.00% 至 −0.19% |

每对的时间降低为 `(原版时间 − 候选时间) / 原版时间`，再取中位数；这与直接相除两列
时间中位数不一定相同。负值表示变慢。512/900 两档五对全部更快，包含生成阶段的
整个请求时间配对中位数分别降低 7.03%/7.53%。256 档有两对变慢超过 23%，不能只引用
其正的中位数来声称稳定加速。[五对原始汇总](evidence/pr1107-incremental-five-pairs-iq3s/summary.json)

IQ2_XS 的两对独立对照中，512/900 两档的 TTFT 配对中位数分别降低 16.21%/14.96%，
两对方向一致；256 档仍有明显波动。这个结果支持相同趋势，样本数未达到 IQ3_S 的五对。
[IQ2_XS 汇总](evidence/pr1107-incremental-screen-iq2xs/summary.json)

### 适用范围与退化

以上收益适用于这组 32K 历史后的短增量和受控设置，不能直接外推到产品默认设置。
完整读取 4K/32K/128K 输入的两对实验保留了以下结果：

- IQ3_S 的 4K/32K 生成阶段耗时，在两对中均增加约 3.5–4.7%。
- IQ2_XS 的生成阶段快慢方向随配对翻转，一次 128K TTFT 比对照增加 9.67%。
- 原设置下 IQ2_XS 的严格状态或工作量一致性检查曾失败；完整受控设置通过，不能据此取消原失败记录。

因此当前分支虽包含此补丁，仍未把它采纳为全场景默认加速方案。
`STRATA_PREFILL_GROUP_GATHER=0` 会连同原版已有的长输入分组路径一起关闭，
不能用它严格还原补丁前版本；做代码对照应使用保留的 baseline binary。
详见 [IQ3_S 长输入对照](evidence/pr1107-fresh-screen-iq3s/summary.json)、
[IQ2_XS 长输入对照](evidence/pr1107-fresh-screen-iq2xs/summary.json)
及 [受控一致性记录](README.md#correctness-and-limitations)。

## 正确性修复与其他调优结论

### gfx1201 Q2_0 正负零修复

提交 `c8cbc0d` 在 [iq_kernels.cu](patches/measured-baseline.patch) 中保留 Q2_0 反量化
结果的负零符号。修复前，真实 IQ3_S 三种专家格式组合共出现 991,300 处 FP16 位差异，
全部为正负零；修复后七种组合全部通过，FP16 位差异为零，相关状态与正常生成回归通过。
这是正确性修复，没有独立计入速度收益。[修复验证](README.md#correctness-and-limitations)

### 稳定的 BLAS 参考

本机捕获的 FP16 GEMM（T=249、N=512、K=2560）在独立程序中，用默认/偏好 hipBLASLt
路径重复执行时出现大幅误差和非有限值。设置 `ROCBLAS_USE_HIPBLASLT=0` 后，Tensile
的 1,000 次输出逐位相同，对 CPU float64 的最大绝对误差约为 `1.719e-5`。

因此本轮最终对照和最近的速度测试使用 Tensile，WMMA 关闭，Lt tuning 为空。
这个选择用于建立稳定参考，存在性能成本；不能把旧后端更快的记录与新后端的正确性
结果拼接成一个配置。结论限于这次 R9700 / ROCm 7.2.3 安装环境，根因机制尚未确定。
[独立重放结果](evidence/gemm-repeat-summary.json)

### 尚未采纳的候选

| 尝试 | 验证结果与决定 |
| --- | --- |
| K/V 投影指定 Lt solution 88193 | 17 个矩阵形状各重复 1,000 次通过；整机分布检查仍未通过，未作为加速配置采纳 |
| WMMA、Lt 与两者组合 | 稳定 Tensile 参考下，四个候选均未通过预先声明的质量门槛 |
| PLE 放入 RAM | 正反顺序中仍出现变慢，未见可靠收益；历史 RAM arm 的 RSS 约 76.4 GiB，对应 direct arm 约 49.7 GiB |
| NUMA 与 7/15/31 workers | 未找到可靠优于 15 workers 的配置；当前节点设置是已测起点，不代表最优 |
| ROCm helper 线程亲和性 | 已确认线程绑定改变，三个输入长度的 prefill 中位数差异均小于 0.3%，快慢波动仍在 |
| inline prefill issuer | 4K/32K 仍有波动，未采纳为优化 |
| PR #1368、#1297 等 | 做过调查，未移植；不计入已实现加速 |

整机数值门槛在看结果前固定：48 个 teacher-forced 位置、全部 logits 有限、平均 KL
不超过 0.001 nat、单位置 KL 不超过 0.01 nat、top-1 一致率至少 98%。例如，K/V-only
加 Tensile fallback 的 KL 虽通过，top-1 为 47/48，仍按失败处理。早期 WMMA 曾通过旧
后端下的局部检查，最终决定以稳定参考下的结果为准。
[最终数值检查](evidence/stable-quality-screen-summary.json)、
[RAM 与 NUMA 记录](evidence/iq3s-ram-screen-summary.json)

## 当前可复用的配置

native IQ、HIP MMQ、MTP、自动专家缓存和 KV streaming 是已有引擎能力，本轮沿用并
验证其组合，没有逐项消融测出各自的新增收益。最近速度测试采用
[IQ3_S 配置](current-speed/iq3_s-config.json)，主要设置如下。

| 设置 | 当前速度测试 | 作用与边界 |
| --- | --- | --- |
| 专家缓存 | `--expert-cache auto` | 本机实际 13,021 slots，约 24.71 GiB；slot 是专家槽位，不是 MiB |
| 输入分块 | `--prefill auto` | 本机实际 chunk 8,192、ring 384；固定 ring 96 属于另一套受控配置 |
| MTP | `--spec 4 --spec-min-p 0.5`，匹配的 MTP runtime | 开启草稿验证；保留默认 suffix/lookup 调度，实际窗口宽度不等于始终 4 |
| KV | `--kv int8 --kv-resident 32768` | 最大上下文设为 147,456；超过显存驻留范围的 KV 使用 RAM |
| CPU | `--pool-workers 15` | 与本机 CPU/NUMA 条件一起记录 |
| PCIe 与自适应 | 保留默认设置 | 本次探测后 `pcie_frac=0.55` |
| 显存余量 | `--vram-reserve-mib 1024` | 本次 READY 报告约 940 MiB 空闲；请求边界采样最高进程显存约 30.84 GiB |
| 提示缓存 | 速度测试 `--prompt-cache 0` | 强制完整读取输入；产品多轮验证与短增量 runner 使用 6 |

配置内的环境变量为：

```text
ROCR_VISIBLE_DEVICES=GPU-1d9ca7a7f0a06a6d
HIP_VISIBLE_DEVICES=0
ROCBLAS_USE_HIPBLASLT=0
STRATA_PREFILL_MMQ=1
STRATA_HIP_WMMA=0
STRATA_HIPBLASLT_TUNING=
```

GPU UUID、资产路径和 NUMA 参数是本机值。迁移机器时需重新确认，不能把这里的配置
直接当作通用安装配置。模型与匹配的 pack、tokenizer、MTP 资产见
[IQ3_S 清单](evidence/iq3s-assets.json)。

精确比较代码变化时使用另一套
[受控配置](evidence/configs-tensile-repro-fixed/iq3_s-candidate.json)：
专家缓存预算 8,000 slots、chunk 4,096、ring 96，`STRATA_IQ_MT_MIN=1`，PCIe 份额为 0，
关闭 adaptation、suffix drafting 和 lookup chain。短增量 runner 会把 prompt cache
设为 6。这些控制用于固定计算路径与工作量，不能与产品 auto 配置的绝对速度混算。

## 复现与后续调优

当前速度测试的命令、单卡校验、独立构建与 A/B 配对入口见
[实验复现](REPRODUCE.md)。严格对照的源代码点为 baseline `9fd910f`
与 candidate `269a140`；保留 binary 的完整哈希也在该文档中。

本轮原有矩阵完成 94 个正式性能请求，随后当前版本速度测试新增 12 个正式请求，
两者单独归档，不合并计算提速。两种量化、两版引擎各八项产品 HTTP 任务检查共
32/32 通过，包含 JSON、算术、Python 执行、tool-call、多轮、长文检索和取消重试。
任务通过不替代严格数值一致性或性能验收。
[原有覆盖](coverage.json)、
[本次速度测试完成记录](current-speed/completion.json)

后续优先解释 256-token 增量和长输入生成阶段的波动，再扩大分组 gather 的适用范围。
对比时固定模型、量化、后端、输入和输出长度；独立启动引擎并交错运行，保留所有慢样本。
改变数值路径时先通过内核、状态/分布及任务检查，再计入性能收益。

完整 chunk/ring/MTP/KV/PCIe 参数扫描、严格 NUMA 内存绑定或交错、多并发、Windows
和其他模型尚未覆盖。现有 object 审计与资源采样尚不能确定波动原因；
本轮没有修改系统驱动、库版本、GPU 时钟或功耗上限。
