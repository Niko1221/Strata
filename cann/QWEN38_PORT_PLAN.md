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

---

## 一个改变顺序的发现（第 24 轮）

上面最后那句是错的，至少对 HC 是错的。读过引擎真正的前向代码后：

```c
const std::vector<float> an  = need(m.gguf, p + "attn_norm.weight");
const std::vector<float> pan = need(m.gguf, p + "post_attention_norm.weight");
rms_norm(...);           // 进块前归一化
...                      // 块
xn[i] += layer_out[i];   // 普通残差，加一次
rms_norm(...);           // 进 FFN 前归一化
```

而 Qwen3.8 的块里**没有 `attn_norm`，也没有 `post_attention_norm`**。每层有的是：

```
hc_attn_norm.weight   hc_attn_down   hc_attn_up   hc_attn_inject
hc_ffn_norm.weight    hc_ffn_down    hc_ffn_up    hc_ffn_inject
```

`hc_attn_norm` 就是那块归一化的 gamma，只不过它躺在 HyperConnection 的混合函数里
（`build_hc_mix(cur, w_norm = hc_attn_norm, ...)`）。残差也不是加一次，而是四条流各自加权。

**所以 HC 不是附加功能，它是这个模型的归一化 + 残差机制本身。** 没有它，连一层的输入
都算不出来，"先跑通线性层、HC 以后再说"这条路不存在。

修正后的切法：

| 片 | 内容 | 为什么在这个位置 |
|---|---|---|
| **A1** | `ModelConfig` 从 GGUF 读 | 已完成一半：结构体在，读数还没接 |
| **A2** | **HyperConnection**：4 路残差 + 组内 RMSNorm + 低秩门 | 没有它产生不了块输入 |
| **A3** | 线性层接融合算子（48 v 头、10240 维） | 依赖 A2 的输出 |
| **A4** | 全注意力 + QSA + PLE | |
| **A5** | MoE 512 选 10 | |

A2 必须排在 A3 前面。这和本文上半部分的顺序相反，写在这里，免得下次又按错的顺序开工。


---

## A2 完成：HyperConnection 已验证

`src/hc.cpp` + `tests/hc_parity.cpp`，与双精度参考对比：

```
hidden 2048, streams 4, low rank 320
inject  rel err 5.06e-04
mixed   rel err 1.31e-06
combine rel err 5.59e-06
```

**这些都是累加顺序级别的差异**，说明实现是对的（f16 输入、f32 累加）。

### 中途一次误报，值得记

第一版测试报 `combine rel err 1.9e-02`，我以为是 f16 累加丢精度，准备去改累加策略。
把 `inject` 的设备值和参考值都打出来才发现是 **153% 的错**——根本不是精度问题。

原因是**测试里的参考实现把权重布局搞反了**：ggml 的 `[ne0, ne1]` 张量是 `ne0` 最快，
所以元素 `(k, m)` 在 `m*ne0 + k`，我写成了 `k*ne1 + m`（转置）。

改对之后 `mixed` 从 1.1e-3 变成 1.3e-6 —— 说明那个"1e-3 的通过"也是假象：
参考和实现错得差不多，正好抵消了一部分。

**教训**：数值测试报"通过"的时候，也要看一眼误差的量级是否符合物理预期。
1e-3 对 f16 是合理的，但如果一项纯逐元素运算也报 1e-2，那多半不是精度问题。


---

## A3 的最高风险项已经排除：融合算子在这个尺寸上是好的

Qwen3.8 的线性层是 16 个 k 头、**48 个 v 头**。`aclnnRecurrentGatedDeltaRule` 从没在这个
形状上跑过这台机器，而它的 head 映射是 blocked（v 头 h 用 k 头 `h/(Nv/Nk)`）——映射错了
就是静默的数值错误。

`gdn_fused_parity` 本来就接受任意 `Nk/Nv/Dk/Dv`，直接跑：

```
Qwen3.6 形状 (16/32), T=1   state[Dk][Dv] 3.53e-3 PASS   attn 一致
Qwen3.8 形状 (16/48), T=1   state[Dk][Dv] 3.29e-3 PASS   attn 一致
Qwen3.8 形状 (16/48), T=4   state[Dk][Dv] 3.88e-3 PASS   attn 一致
```

三组误差都在 3-4e-3，这正是 bf16 融合算子该有的量级（和 16/32 时一样）。输出行里那条
`state [Dv][Dk] ... FAIL` 是布局说明，不是失败：算子要的是 `S^T`，即 `[Dk][Dv]`。

**结论**：48 个 v 头不需要为融合算子做任何额外工作。A3 剩下的只是把 `gdn_step` 的
缓冲区尺寸从常量换成配置（10240 维 qkv、6144 维 conv 通道），是机械活。
