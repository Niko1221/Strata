# Qwen3.8 移植的第一刀：先把形状变成配置

这份是**动手清单**，不是背景介绍。所有行号来自当前 `strata/cann` 的源码。

## 为什么先做这一步

引擎里所有形状都是编译期常量，而它们的引用点密密麻麻：

```
src/engine.cpp                 217 处
include/strata/cann/engine.hpp   1 处
```

而且层循环也写死了：

```c
src/engine.cpp:721    for (int64_t il = 0; il < 40; ++il) {
src/engine.cpp:769    for (int64_t il = 0; il < 40; ++il) {
src/engine.cpp:809    for (int64_t il = 0; il < 40; ++il) {
```

所以"改 7 个常数"这个说法是错的——**要改的是一套形状的来源**，不是 7 个数字。
不先做这一步，后面加 PLE/QSA 的时候会一边加逻辑一边改常量，改错的地方查不出来。

## 目标和判据

**目标**：`ModelConfig` 从 GGUF 元数据读形状，`engine.cpp` 只引用它。

**判据（必须全部成立）**：
1. `port_check` 对新模型输出全绿（除已知的三块新算法）
2. 现有 Qwen3.6 的测试**全部照旧通过**——这是"重构没改行为"的唯一证据
3. 用 Qwen3.8 的配置跑一次前向，层 0 的输出不与 NaN，且第一层耗时和 Qwen3.6 同量级

第 2 条最重要：没有它，就没法区分"重构引入的错"和"模型差异导致的错"。

## 具体做法

### 第一步：把常量收进结构体（不改任何算法）

```cpp
struct ModelConfig {
    int64_t hidden, layers, k_heads, v_heads, ssm_state, conv_k;
    int64_t attn_heads, attn_kv_heads, head_dim, rope_dim;
    int64_t expert_inter, experts_used, experts_total;
    float   rms_eps, rope_base;
    bool    has_ple, has_indexer, has_hc;
    const ModelConfig & derived() const;   // qkv_dim 等派生值集中算
};
```

`qkv_dim = 2*k_heads*ssm_state + v_heads*ssm_state` **只能在这里算一次**。
现在它是 `kQkvDim = kKeyDim*2 + kValueDim` 的宏展开——如果移植时只改 `kVHeads` 忘了 `kQkvDim`，
是编译期对、运行期错的静默 bug。

### 第二步：按引用密度分片改

217 处不能一次改完，按这个顺序，每改一片跑一次测试：

| 片 | 常量 | 引用数 | 风险 |
|---|---|---|---|
| 1 | `kHidden` | 69 | 低（纯形状） |
| 2 | `kHeadDimAttn` | 38 | 中（注意力 reshape） |
| 3 | `kExpertInter` | 25 | 低（MoE 门） |
| 4 | `kVHeads` + `kQkvDim` + `kValueDim` | 62 | **高**（线性层，和融合算子的 descriptor 相关） |
| 5 | `kHeads` / `kTopK` / `kMaxTopk` | 18 | 低 |
| 6 | 层循环 `il < 40` | 3 | 低，但要和 `has_ple` 的层号一起改 |

第 4 片单独做、单独验：它决定 `aclnnRecurrentGatedDeltaRule` 的输入尺寸，改错了表现是 NaN
或数值全偏（第 36 轮踩过）。

### 第三步：加一个"配置断言"

加载时立刻检查，而不是等到算到第 20 层才崩：

```cpp
// tensor 的实际形状必须和配置推出来的一致
check("blk.0.attn_qkv.weight", {cfg.hidden, cfg.qkv_dim()});
check("blk.0.ssm_conv1d.weight", {cfg.conv_k, cfg.qkv_dim()});
check("blk.N.ffn_gate_exps.weight", {cfg.hidden, cfg.expert_inter, cfg.experts_total});
```

`port_check` 已经有这个数据了，把它的检查搬进加载路径即可。

## 已知的三块新算法（这一步不做）

| 块 | 为什么不能塞进这一步 |
|---|---|
| PLE | 在 blk.1 之后插一段，需要新的张量和新的卷积状态 |
| QSA | 需要第二套 KV cache，和现有的 cache 管理纠缠 |
| HyperConnection | 残差流从 1 条变 4 条，**改的是整条主干的形状**，必须最后做 |

顺序上，HyperConnection 一旦进来，`hidden` 之外还要传 `hc*hidden`，
所以它必须在这一步之后、且单独一步做。

## 时间估计

- 第一步（结构体 + 派生值）：半天
- 第二步（6 片）：2-3 天，其中第 4 片占一半
- 第三步（断言）：半天

**合计约 3-4 天**，产出是一个"形状可配置、行为不变"的引擎，
外加对 Qwen3.8 的一次前向能跑到层 47 而不崩。

到那时才开始 PLE / QSA / HC。
