# 把 LoRA 合并进权重能不能更快：方法与收益上界（**只写方法，不实施**）

> 结论先行：**收益上界约 +1~2% decode，上限来自「每 token 48 次 adapter kernel」，不是来自数学量**。
> 而这个上界有两条路可以达到，其中**不需要**重写 76 GB 权重的那条（引擎侧 epilogue）更划算。
> 所以「合并进权重」这件事，作为**提速手段**不值得做；它真正值得做的理由是「少一份要加载的文件」，不是速度。

---

## 1. 先算收益上界，再谈方法

这台机器上的 125B 推理基线（`strata-main-lp.json` 起的那份）：

| 量 | 值 | 来源 |
|---|---|---|
| 权重 | 39.79 GB + 36.18 GB = **75.97 GB**（IQ3_XXS，2 分片） | `Strata-data/models/swift-IQ3_XXS/` 实测文件大小 |
| 结构 | 48 个 MoE 层、512 专家/层、hidden 2560、moe-inter 640 | `experts_nf4.bin.json` |
| decode | **172.8 tok/s**（p50，n=888） ⇒ 5.79 ms/token | `strata-main-lp.log` |

**LoRA 挂在哪些矩阵上**：`lora_to_gguf.py` 只映射两类（其余键必须丢掉，否则引擎拒绝加载整个 adapter）：

```
HF ...layers.<l>.self_attn.o_proj      -> GGUF blk.<l>.attn_output.weight
HF ...layers.<l>.linear_attn.out_proj  -> GGUF blk.<l>.ssm_out.weight
```

这两类都是 hidden×hidden 的**稠密**投影。每层按架构只会命中其中一个（该层要么 full attention 要么
linear attention）⇒ 每层 1 个 adapter，**全模型 48 个**，参数量 48 × 2560² = **314 M ≈ 0.25%**。
MoE 专家（121 B，占 97%）**一个都没碰**。

**引擎怎么用它**（`src/core/layer.cpp:385`、`include/strata/kernels/lora.hpp`）：

```
out += s · B (A y)        s = alpha / rank
```

decode（n=1）走 `launch_small<XT,1>`，A、B 在**同一个 kernel** 里算完 ⇒ 每个被覆盖的投影
**一次 launch**，每 token 合计 **48 次**。adapter 权重 A(16×2560 f32)+B(2560×16 f32) = 328 KB/层，
48 层共 15.7 MB，4090 的 L2 是 72 MB ⇒ 常驻 L2，带宽上近似免费。

于是上界只能来自 **launch/dispatch**：48 × (1.5~2 µs) ≈ **72~96 µs**，占 5.79 ms/token 的
**1.2~1.7%**。合并后这 48 次调用消失 ⇒ 172.8 → 约 **175 tok/s**。

**这个上界小到不值得为它冒量化误差的风险**，尤其当代价是重写 76 GB。

---

## 2. 三条实现路径，以及为什么前两条不划算

### 路径 A：离线把 delta 折进 base，再重量化回 IQ3_XXS（用户说的「合并进权重」）

数学上只有一行：

```
W' = W + s · B A ,   s = alpha / rank = 32 / 16 = 2
```

但对 **IQ3_XXS** 落地时要过三道坎：

1. **必须先反量化**：`dequant(IQ3_XXS) → fp32/bf16 → 加 delta → 再量化`。每层 2560²、48 层，
   反量化+重量化要跑一遍完整的分块量化器；
2. **再量化需要原来的 imatrix**：IQ3_XXS 的量化网格是由 importance matrix 定的，没有原 imatrix
   重跑 `quantize` 出来的不是「同一个量化器下的最近点」，而是一个新网格 —— CE 会漂，
   而漂多少**只能靠全量 CE 对比才知道**（这不是能靠推理得出的量）；
3. **存储代价**：写出新 GGUF = 再 76 GB；即使只写 delta 分片，也需要引擎支持「base + delta 分片
   叠加加载」这种格式 —— llama.cpp 的 GGUF 没有，等于要给引擎加一种新的加载模式。

**净结果**：花 76 GB 磁盘 + 一次全量重量化 + 一次 CE 回归，换 ≤1.7% decode。

### 路径 B：加载期合并进常驻缓冲

引擎常驻显存的是**量化权重**（IQ3_XXS 块 + absmax），没有 bf16 副本。要在加载期合并进 bf16 副本，
得先有 bf16 副本 = 125 B × 2 B = **250 GB** 内存 —— WSL 红线是 47.99 GiB，不可行。
把 delta 合并进**量化块**本身可以（每块反量化→加→重量化），但收益与路径 A 相同，而它改的是热加载
路径，风险更高。

### 路径 C：把 adapter 折进投影 kernel 的 epilogue（**推荐，但要动引擎**）

既然上界全部来自「48 次独立 launch」，那就把这 48 次并进已有的投影 GEMM：

```cuda
// 现在：out = W·y           （一次 GEMM launch）
//       out += s·B(A·y)     （再 48 次小 launch）
// 合并：out = W·y + s·B(A·y) 在同一个 kernel 的 epilogue 里算
```

- **不动权重文件**：磁盘/内存零新增，不需要重量化，CE 不变（数学完全一样）；
- 拿到的收益与「合并进权重」**同一个上界**（1.2~1.7% decode）；
- 代价是引擎侧每层一个 epilogue 分支，prefill 走 `lora_apply_f16` 的那条也要一起改；
- 已有 `src/kernels/lora_parity.cpp` 的逐值对照可以直接当正确性门。

---

## 3. 如果仍然要做路径 A，正确顺序是（本次**不实施**）

1. **先测上界**：同一 prompt、`--lora` on/off 各跑 3 次 decode，比 tok/s。
   差值就是「合并」能拿到的天花板。**这一步 2 分钟，不做这一步就不要写量化器。**
   若差值 < 1%（很可能，因为 adapter 权重常驻 L2），直接终止。
2. 只合并 48 个投影，**绝不动专家**（专家是 NF4 pack，与 GGUF 的 IQ3_XXS 是两套格式）。
3. 重量化用原 imatrix，且**只重写承载 adapter 的那些 tensor**，其余字节原样拷贝 —— 避免
   「全量重量化」把没被 LoRA 碰过的 99.75% 参数也重新抖一遍。
4. 门禁：同 prompt 下 merged 与 `--lora` 的 top-1 逐 token 一致率、以及 CE（用固定 200 条样本），
   阈值建议 CE 漂移 ≤ 0.01 nats；超过就说明第 3 步的重量化网格选错了。
5. 存储：只写「被改动的 tensor」组成的增量 GGUF，并在引擎里把「base 分片 + 增量分片」按
   tensor 名重定向 —— 这一条是路径 A 唯一比路径 C 多出来的工程量。

---

## 4. 一句话回答

> **合并不划算。** 上界 1~2%（48 次 launch / token，172.8 → ~175 tok/s），
> 代价是 76 GB 磁盘 + 全量重量化 + CE 回归。
> 想要这 1~2%，正确做法是把 adapter 折进投影 GEMM 的 epilogue（路径 C，零存储成本）；
> 想要「少一个文件」，才做路径 A。
> 而**真正决定 125B 推理速度的东西（MoE 专家流式 + MTP 推测解码）与 LoRA 无关**，
> 合并前后一个字节都不变。
