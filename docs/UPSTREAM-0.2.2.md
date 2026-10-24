# 上游 v0.2.2（engine 0.1.47）实测：首 token −28%，但 auto 分层会把 decode 拖慢 5.8%

> 测量时间 2026-10-21，同一套 **10 条真实 agent 请求**（回放器 `replay.py --suite "56:6,0:4"`），
> 同一份配置（只差引擎版本 / 分层设置）。噪声基线：decode ±1.2%、首 token ±1.4%（配对）。

## 0. 结论

| | |
|---|---|
| **代码本身** | ✅ **确实更快**：首 token **−28.4%（10/10 全快）**、decode 持平、墙钟 −6% |
| **但 auto 分层退化了** | ❌ v0.2.2 的 `--layer-split auto` 选了 **K=31**（原 27），常驻对数 20,148 → 17,991，**decode −5.8%**、CPU 池 3.4% → 7.5% |
| **所以** | 要用 v0.2.2，**必须把 `layer_split` 从 `"auto"` 改为 `"27"`** |
| **另** | 纯上游没有 arena 释放（`STRATA_ARENA_RELEASE_BLOCK`）⇒ commit 只剩 **4.6 GiB**，必须把补丁 rebase 过来 |

## 1. 三档实测

| 指标 | 0.1.44.1 | 0.1.45 | **0.1.47 (v0.2.2, K=27)** | 0.1.47 (v0.2.2, auto→K=31) |
|---|---:|---:|---:|---:|
| **首 token（中位）** | 3.03 s | 3.01 s | **2.11 s** | 2.33 s |
| decode（中位） | 106.5 | 104.1 | **106.1** | **98.8** |
| VRAM 命中 | 96.70% | 96.40% | **97.15%** | **92.30%** |
| PCIe | 0.20% | 0.20% | **0.10%** | 0.20% |
| CPU 池 | 3.10% | 3.40% | **2.70%** | **7.50%** |
| 墙钟（10 条） | 262 s | 263 s | **247 s** | 260 s |

配对逐条（0.1.47 K=27 相对 0.1.45）：

```
first token  中位 −28.4%   0 升 / 10 降   范围 −33.8% .. −3.4%
decode       中位  +1.5%   6 升 / 4 降   （噪声内）
VRAM 命中    中位 +0.8pp   8 升 / 1 降
output tokens 中位 −1.8%   5 升 / 5 降   （无漂移）
```

## 2. auto 分层为什么变差（这是本次最有价值的一条）

| | 0.1.44.1 / 0.1.45 | v0.2.2 的 auto |
|---|---|---|
| K | **27**（3090 0-26 / V100 27-47） | **31**（3090 0-30 / **V100 31-47，只剩 17 层**） |
| 常驻对数 | 20,148（~99.4% 路由质量） | **17,991（~99.1%）** |
| V100 缓存 | 10,752 槽（21 层 × 512，100%） | 8,704 槽（17 层 × 512，100%） |
| 3090 缓存 | 8,516 槽（15.46 GiB，62%） | 8,679 槽（15.79 GiB） |
| 实测 decode | 106.4 | **98.8** |

引擎自己的预测：`K=31 - predicted 27.6 ms per decode window`（K=27 是 27.8 ms）—— **实测反了**。
它在同一条日志里也承认：`the prediction may not rank them (it prices decode, not the prompt)`。
**两卡不同型号（3090 0.48 ms/层 vs V100 0.66 ms/层）时，这个预测不可靠。**

⇒ **对这台机器，K=27 是实测最优**；不要依赖 auto。

## 3. 怎么复现

```cmd
:: 编译（CUDA 12，双架构 sm_70 + sm_86）
D:\strata-blackbox\build_v100_cuda12.cmd        :: configure + build，exit=0，产物 158 MB

:: 测速（每次重启引擎 + 10 条回放，约 4.5 分钟）
D:\strata-blackbox\replay_arm.cmd strata-iq3_s-2gpu-v022k27.json v022k27 "56:6,0:4"
.venv\Scripts\python.exe D:\strata-blackbox\compare_replay.py <基线>.jsonl <新档>.jsonl
```

## 4. 升级清单（要用 v0.2.2 时必须一起做）

1. `layer_split` → **`"27"`**（否则 auto 选 31，decode −5.8%）
2. **arena 补丁 rebase 到 v0.2.2**（否则 commit 只剩 4.6 GiB，引擎已经在警告 `0xC0000409`）
3. 重编后确认启动行：`session is up (engine 0.1.47)`、`layer split: CUDA1 runs layers 27-47`、
   `arena-skip: layers 27-47 are fully resident: 21.71 GiB ... handed back`
4. 回放复核：首 token ≈ 2.1 s、decode ≈ 106、VRAM ≈ 97%
