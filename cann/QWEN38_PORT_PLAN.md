
## 一个改变顺序的发现（第 24 轮）

我原计划"先跑通 36 个线性层，HC 以后再加"。读过引擎的前向代码后，这个顺序**不成立**。

引擎现在的每一层是这样开头的：

```c
const std::vector<float> an  = need(m.gguf, p + "attn_norm.weight");
const std::vector<float> pan = need(m.gguf, p + "post_attention_norm.weight");
rms_norm(...);          // 进块前归一化
...                     // 块
xn[i] += layer_out[i];  // 普通残差
rms_norm(...);          // 进 FFN 前归一化
```

而 Qwen3.8 的块里**没有 `attn_norm`，也没有 `post_attention_norm`**。每层有的是：

```
hc_attn_norm.weight   hc_attn_down   hc_attn_up   hc_attn_inject
hc_ffn_norm.weight    hc_ffn_down    hc_ffn_up    hc_ffn_inject
```

`hc_attn_norm` 就是那块归一化的 gamma，只不过它躺在 HyperConnection 的混合函数里
（`build_hc_mix(cur, w_norm = hc_attn_norm, ...)`）。残差也不是"加一次"，而是四条流各自加权。

**结论**：HC 不是附加功能，它是这个模型的**归一化 + 残差机制本身**。
没有它，连一层的输入都算不出来。

所以第一片的正确切法是：

| 片 | 内容 |
|---|---|
| **A1** | `ModelConfig` 从 GGUF 读（已完成一半：结构体在，读还没接） |
| **A2** | **HyperConnection**：4 路残差 + 组内 RMSNorm + 低秩门（替代 attn_norm/post_attention_norm） |
| **A3** | 线性层接融合算子（48 v 头、10240 维） |
| **A4** | 全注意力 + QSA + PLE |
| **A5** | MoE 512 选 10 |

A2 必须在 A3 前面，因为没有 A2 就产生不了块输入。这和我上一版计划里的顺序相反——
写在这里，免得下次又按错的顺序开工。