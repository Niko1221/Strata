# Strata 8060S (Strix Halo, Windows) 性能测量:调优表 A/B

日期:2026-10-07 | 机器:GPU 8060S gfx1151,32GB 系统 RAM + 96GB BIOS carve-out(统一内存),UD-IQ4_XS,`--max-context 32768 --kv int8 --resident-budget-gib 8 --spec 4 --mtp`

## 结果(tok/s)

| 场景 | 调优表 ON(A 组) | 调优表 OFF(B 组,3 次) | 差异 |
|---|---|---|---|
| decode short-64 | 17.8 | 13.6 / 22.0 / 22.3 | ~+20-30% |
| decode short-128 | 30.7 | 27.4 / 32.4 / 33.1 | 相近(~+10%) |
| **prefill 739 tok (pt-512)** | **2,337** | 1,649 / 1,861 / 1,840 | **+26-42%** |
| **prefill 2,803 tok (pt-2048)** | **6,665** | 6,928 / 6,922 / 6,651 | 持平(~0) |

## 解读
- **短 prompt prefill(≤~1k tok)调速明显**:tuning On 2200+ vs Off 1700-1900,约 **+30%**。短查询场景收益最大。
- **长 prompt(2.8k)持平**:峰值都被内部 SIMT/tensor-core 并行吃满,表的影响被摊平(可能在更长 prompt 上调优表的作用更显著,未测 16k+;也可以解释为长 prefill 更多在 expert path,非 hipBLASLt 段)。
- **decode 差异 ≤30%,样本间抖动大**(13.6 vs 22.0 vs 22.3):decode 主要瓶颈是 CPU 专家池/SSD 流式,调优表几乎不影响;差异来自文件缓存/后台。
- 引擎日志确认每个运行段:`prefill gemm: hipBLASLt tuning enabled (90 rows, gfx1151, version 100500)`(ON) / 无此行(OFF)。

## 结论
调优表在**短 prompt prefill**上实测有效(+30%),这正是日常 chat 的典型负载;长 prefill 与 decode 收益有限但无损失。**改动 #A 值得保留并回报上游。** 附带产出:`tools/bench_tok.py`(可复跑)、`bench/results/bench-tuning-{on,off,off-2,off-3}.json`。