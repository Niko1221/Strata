# 专家显存/内存架构：现状、94% 的成因、以及「V100 的专家不进 RAM」的改法

> 2026-10-19 讨论记录。**结论未实施**，本文用于定方向。
> 前置事实来自本机实测（3090 24 GB + V100 32 GB，GSQ-RCO IQ3_S，Windows，0.1.44.1 + 插桩）。

## 0. 摘要（三句话）

1. **99.3% 和 94% 不是同一个指标**：前者是 profile 的「路由质量覆盖率」（静态、bench 场景），后者是运行时的「VRAM 命中率」（动态、真实 agent 流量）。两者相差 5 个百分点，意味着 **CPU 池的计算量相差 7 倍**。
2. **当前 5% 的 miss 里，94% 由 CPU 池承担、只有 6% 走了 PCIe** —— 而 PCIe 的单次成本只有 CPU 池的 1/3~1/6。**这是目前最该修的一处，成本是一条环境变量。**
3. **V100 那 21 层（10752 对，21.70 GiB）在 RAM 里是纯白装**：它的缓存 512/512 全满，永不 miss、永不 evict ⇒ 那 21.70 GiB 永远不会被读。**但当前代码做不到「不装」，必须改源码**（见 §4）。

---

## 1. 两个命中率不是一回事

| | **99.3%** | **94%** |
|---|---|---|
| 出处 | 引擎日志 `the caches hold 20033 of 24576 profiled pairs (~99.4% of the routed mass)` | 网页 Monitor 的 **VRAM hit rate** 列（单次请求）；日志累计行为 `decode expert cache hit rate: 94.8% (…)` |
| 含义 | **静态**：缓存里装的 19374 对，按 profile 的权重能覆盖多少流量 | **动态**：真实请求里，每次专家查询有多少命中显存 |
| 场景 | BENCH 的 4 条合成 prompt（同一段文字重复）→ 路由极度集中 | agent 真实流量（代码 / 工具输出 / JSON / 日志 / 散文混合）→ 路由更平 |

⚠ **网页卡片的口径要注意**（`serve/web/index.html:144` 列标题原文）：卡片的 hit rate **不含经 PCIe 取来的专家**，那部分在表里单独显示为 `+N% PCIe`。

### 1.1 真正的三段占比（从本机日志算出来的）

```
strata serve: decode expert cache hit rate: 94.8% (53411 hits / 56322 lookups);
              188 more read by the GPU over PCIe or from another GPU (0.3% of all 56510 routed)
```

CPU 池承担 = `routed − hits − offloaded` = `lookups − hits`：

| VRAM 命中 | **CPU 池承担** | PCIe 承担 | CPU : PCIe |
|---|---:|---:|---:|
| **94.8%** | **5.15%** | 0.3% | **17 : 1** |
| 96.0% | 4.04% | 0.2% | 20 : 1 |
| 96.2% | 3.77% | 0.2% | 19 : 1 |
| 97.8% | 2.20% | 0.1% | 22 : 1 |
| 98.1% | 1.89% | 0.1% | 19 : 1 |
| 98.4% | 1.59% | 0.1% | 16 : 1 |
| 98.9% | 1.06% | 0.04% | 26 : 1 |

⇒ 两个结论：
- 94% 不是「只差 5%」，而是 **CPU 池在全量算 5% 的专家**；
- **PCIe 那条快路几乎没用上**：`pcie_frac` 是 0.37 / 0.18，本该有 ~30% 的 miss 走 PCIe，实际只有 **6% 左右**。原因是 `docs/DETAILS.md:191` 写明的**整数份额取整**（"the share otherwise rounds to 0 at 1-2 misses"）。

---

## 2. 存储层次与单位成本

```
   VRAM（每卡专家缓存，静态、由 profile 决定，两卡共 19374 槽）
     ↕  PCIe 2 MiB/专家      3090 Gen4×8 13.4 GB/s ｜ V100 Gen3×8 6.5 GB/s
   Host RAM（resident arena，全量 46.84 GiB）
     ↕  （只有 --mmap-experts / --resident-* 才有这一层）
   磁盘 experts.bin（51.1 GiB）
```

| 路径 | 单次成本（2 MiB/专家） | 备注 |
|---|---:|---|
| PCIe → 3090 | **0.15 ms** | 最快 |
| PCIe → V100 | **0.31 ms** | 仍比 CPU 快 3× |
| CPU 池（arena） | ~1 ms（19.41 ms/token ÷ ~19 次查询） | 当前 5% 都压在这里 |
| CPU 池（warm mmap） | ~1.8 ms（34.78 ms/token） | 池的取值范围 |
| CPU 池（cold mmap） | ~3.7 ms（71.97 ms/token） | 同上 |

**结论**：把 miss 从 CPU 池搬到 PCIe，单次能快 **3–6×**；而这件事目前被取整浪费掉了。

---

## 3. 结构性事实：槽位总数由显存决定，`K` 只决定「每层是否装满」

每层固定 **512 个专家**（24576 = 48 × 512）。缓存槽位总数由 VRAM 决定（19374），**`K` 只改变「每个层由哪张卡跑、那张卡够不够装满它的 512 个」**：

| 切点 | 3090 覆盖 | V100 覆盖 |
|---|---|---|
| **K=27（现在）** | 8473 / 27 层 = **314/512（61%）** | 10752 / 21 层 = **512/512（100%）** ✅ |
| K=25 | 8473 / 25 层 = 339/512（66%） | 10752 / 23 层 = 11776 对 → **467/512（91%）** ❌ |
| K=31 | 8473 / 31 层 = 273/512（53%） | 10752 / 17 层 = 512/512（100%） |

⇒ **K=27 的妙处不是"平衡耗时"，而是它让 V100 那 21 层恰好装满 512/512。** 一旦挪层给 V100，它就从"全满"掉到 91%，反而更差。
⇒ **`K` 不要动。** 总槽位不变，动 `K` 只能把"一边全满"改成"两边都不满"。

---

## 4. ⚠ 主题：让 V100 那 21 层的专家**根本不加载进内存**（需要改源码）

### 4.1 现在为什么做不到（三条代码证据）

**① 「跳过卡内已有专家」只存在于 mmap 家族，arena 没有这个开关**

```cpp
// src/program/generate.cpp:2215
else if (a == "--resident-experts") {
    o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
// src/program/generate.cpp:2228  (--resident-budget-gib)
o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
```

`docs/MULTI_GPU.md:133-138` 明确写了那套逻辑：

> The resident RAM mode works on a split (`--resident-experts`, the low-RAM mode). **The RAM copy of the experts leaves out the ones every card's cache holds, not only the first card's** … An adaptive swap copies the evicted expert back into RAM **from the card that owns its layer**.

但它是 **mmap 模式** —— 引擎会整体离开 arena（`--mmap-experts`），3090 那部分也跟着变成 mmap。**这不是我们要的。**

**② 主机专家 arena 的构建早于切分决策**

```cpp
// src/program/generate.cpp:3082
// CUDA0'S ARENA IS BUILT BEFORE THE SPLIT IS KNOWN.  The prompt path's PLE tables are read out of it, here,
```

**③ arena 模式是「只进不出」**

```cpp
// src/program/generate.cpp:942-944
"--adapt-async 1   --serve with the resident RAM mode or the arena (…): … (the arena: copy-in only)"
```

一旦 arena 缺了 V100 那 21.70 GiB，"RAM 里永远有全量"这个前提就破了。

### 4.2 为什么这件事在数学上确实是白装

| | 数值 |
|---|---|
| V100 分到的层 | 27–47，共 **21 层** |
| 每层专家数 | 512 |
| 该卡需要覆盖的对 | **21 × 512 = 10752** |
| 该卡实际槽位 | **10752**（日志：`10752 of its 10752 profiled pairs`） |
| ⇒ 结果 | **全覆盖 ⇒ 这 21 层的每一次专家查询都在 VRAM 命中 ⇒ arena 里这 21.70 GiB 永远不会被读** |

而且缓存是**静态装满**的（`--expert-cache auto` 按 profile 预填到容量），**512/512 的层没有富余槽位可换出** ⇒ 连 `--adapt-every` 的换入换出都碰不到它。

### 4.3 保守改法（推荐）：只排除「512/512 装满的层」

**判据**：某层如果 100% 覆盖（槽位 ≥ 512），它的专家**不可能**从 RAM 读 ⇒ 可以整层不拷进 arena。
**收益**：本机正好能排除 V100 的 21 层 = **21.70 GiB**（arena 从 46.84 → **25.1 GiB**）。
**零风险**：不涉及 evict 回拷、不涉及"部分覆盖"的模糊地带。

数据流改造：

```
现在的顺序：  建 host arena（全量 46.84 GiB）  →  算 layer split → 填每卡缓存
目标顺序：    算 layer split（试 46 种）        →  得到每卡每层覆盖数
              → 生成排除集：{某卡某层 512/512 全覆盖 的 pair}
              → 建 arena 时只拷"没被任何卡全覆盖"的对（25.1 GiB）
              → 填每卡缓存（不变）
```

**关键：不需要新写拷贝逻辑** —— 载入器已经有 `skip` 参数：

```cpp
// src/program/generate.cpp:3163（GPU 侧权重 arena，STRATA_STAGE_TRIM 已在用）
if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip))
```

主机专家 arena 需要的是**同款 skip 集** + 一个"该 blob 不在 arena"的标记（供 CPU 池读取时判断）。

### 4.4 必须一起处理的三件事（否则会踩坑）

| # | 问题 | 处理 |
|---|---|---|
| **a** | **arena 在 split 之前构建**（L3082），且 **prompt 路径的 PLE 表从 CUDA0 的 arena 里读**（就在这一行注释里） | 把「PLE 表读取」与「专家数据填充」拆开：PLE 表仍在早期读，专家填充推迟到 split 之后（或先算 split 再建 arena） |
| **b** | **prompt 路径的 ring 借用槽位**：流式 prefill 会借 `N` 个槽位，其中 `K` 个需要 RAM 拷贝（`docs/MULTI_GPU.md:159-163` 的 `lend sizing` 行 + `WARNING when K is below N`） | 被排除的层如果被 ring 借到，就必须有 RAM 拷贝 ⇒ 排除集要与 ring 的借用集求差（`排除 = 全覆盖层 − ring借用`），否则启动会打 WARNING 且 prefill 变慢 |
| **c** | **兜底读路径**：万一某对不在 arena、又被某卡换出 | 保守改法下 `512/512` 的层不会换出，理论上不需要；但仍要**留一条显式报错/回退**，而不是读越界（arena 是平坦内存，读错地址就是静默错误） |

### 4.5 预期收益与「零收益」的边界

| 项 | 现在 | 改后 |
|---|---|---|
| Host arena | 46.84 GiB | **~25.1 GiB**（省 21.70 GiB） |
| 启动读盘 | 46.84 GiB（日志：`3.86 GiB/s`，约 12 s） | ~25.1 GiB（约 6.5 s） |
| Windows commit 余量 | 起模型后只剩十几 GiB | **+21.70 GiB** |
| **解码/首 token 速度** | — | **不变**（那 21.70 GiB 从来没被读过） |

> ⚠ **这一点必须写清楚：这个改动解决的是"内存/commit 压力"，不解决 94% 命中率。** 两者是独立的线（§5）。

### 4.6 工作量与风险

> ✅ **2026-10-19 复核后的结论：比原先估的小得多 —— 代码里已经有两个现成钩子。见 §4.7。**

- **估计 150–400 行**，集中在启动构建 / 池读取路径 / ring 记账三处；
- **风险点**：arena 是平坦内存 + 按偏移访问，**漏拷一个仍会被读到的 blob = 静默读垃圾数据**（不是崩溃）⇒ 必须加一条启动自检：*"每个被排除的 pair 都必须在某张卡的缓存里"*，否则拒绝启动；
- **测试**：`--dump-routing` 跑真实 agent 负载，统计"被排除 pair 的实际读取次数"，应为 **0**（这是验收的硬判据）。

### 4.7 ✅ 复核后的实施方案（2026-10-19，行号级）

代码里**已经有两个钩子**，所以这不是"从零实现"，而是"补上 Windows 分支 + 收窄触发条件"。

**钩子 1：释放循环已经存在且已经挑对了 pair**

`src/program/generate.cpp:6478-6498`（`STRATA_ARENA_RELEASE=0` 可关闭）：

```cpp
if (!keep)
    for (size_t i = 0; i < host_res.size(); ++i)
        if (host_res[i] != strata::core::kNotResident)          // ← 正是"某张卡缓存里已有"的 pair
            released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
```

`host_res` 是**唯一权威**的"谁在卡上"表（`generate.cpp:6457-6465`，由 `slot_of()` 逐 pair 填）。
在 3090+V100 上，它标出 **19374 个** resident pair（V100 的 10752 全满 + 3090 的 8622）。

**钩子 2（修正）：大页必须保留 —— 靠"切块 + 整块释放"而不是降级到 4 KiB 页**

`STRATA_NO_LARGEPAGES=1`（`pinned.cu:121`）虽然存在，但它是**降级**（丢掉 2 MiB 大页的 TLB 收益），
**不是**这个功能的正确前提。大页的真实限制是：**不能部分 decommit，但可以整块 `MEM_RELEASE`**。

⇒ V100 的 21 层（27–47）**正好是一个连续的层区间**，所以把 arena 切成两块即可：

```
block0 = layers [0, K)    3090 的 27 层（314/512，部分覆盖） → 保留，大页
block1 = layers [K, 48)   V100 的 21 层（512/512 全覆盖）    → 填完后 VirtualFree(MEM_RELEASE) 整块释放
```

**切分点在建 arena 之前就已经定了**：`stage_of` 定义在 `generate.cpp:4339`，`split_at` 在 4332–4337 校验完毕，
而 `arena_src.open()` 在 **4995** —— 中间隔着 1600 行。所以按层切块没有时序问题。

顺带收益：block1 不需要读盘 ⇒ 启动从 **46.84 GiB 降到 25.1 GiB**。

**⛔ 为什么不能照搬 Linux 的做法**

`src/core/expert_source.cpp:4837-4850`：

```cpp
uint64_t ArenaExpertSource::release(int64_t layer, int64_t expert) {
#if defined(_WIN32)
    (void) layer; (void) expert; return 0;          // ← Windows 就是这两行
#else
    if (map_ == nullptr ...) return 0;
    ... madvise(MADV_DONTNEED) ...                  // ← 只在"文件映射的 arena"(STRATA_ARENA_MMAP) 下生效
#endif
}
```

Linux 那边安全的**根本原因**：arena 是**文件映射**，`MADV_DONTNEED` 丢掉的是页缓存里的**副本**，下次读会**从文件重新缺页读回**（注释原话：*"a minor fault away when the CPU needs the expert again (an adaptive swap evicts it)"*）。

**Windows 的 arena 是匿名内存** —— 一旦 decommit，内容**变成零**，读回来就是**静默错误数据**。
而 `--adapt-every` **默认开着**，它会换出专家；被换出的 pair 下一次就要从 arena 读。⇒ **全量释放是错的。**

**✅ 安全规则：只释放"该层被某张卡 512/512 全覆盖"的 pair**

- 全覆盖的层**没有空槽**可 admit 新专家 ⇒ 自适应层**永远不会换出**它 ⇒ 这些 pair **可证明永不被读**
- 本机正好是 **V100 的 21 层**（10752/10752），3090 的 27 层是 314/512 ⇒ **不释放**
- 收益：**21 × 512 × ~2.07 MiB ≈ 21.7 GiB**（commit 同步释放）

**⚠ 两个复核出来的硬约束（2026-10-19 第二轮，改写了实现细节）**

**① 不能用"估算后提前跳过"** —— V100 的显存余量只有 **~0.26 GiB**：
需求 21 层 × 512 × 2.07 MiB = **21.70 GiB**；可用 ≈ 26.86（权重后空闲）− 2.9（prompt 4096 缓冲）− 0.7（KV 驻留）− 1.3（预留）= **21.96 GiB**。
⇒ 估错 1% 就会漏掉"其实会 miss"的层。**所以必须"先分配+装载，拿到 `host_res`（真实槽位表）之后再整块释放"**，
这也正是 `generate.cpp:6457-6465` 之后才有 `host_res` 的原因。

**② loader 写目标用的是"绝对文件偏移"** —— `src/core/pinned.cu:884`：

```cpp
const uint64_t off = layer_off[(size_t) L];      // 同时用于 fseek(862) 和 memcpy(884)
...
std::memcpy(dst + off + pos, buf.data(), (size_t) n);
```

⇒ 第二块的 `dst` 不能简单给 `base1`（那样会写到 `base1 + 25 GiB` 之外），**必须给 loader 加一个 `dst_bias`**
（`dst + off - dst_bias`）。而这会动到 `pinned.hpp` 的两个重载 **以及 `pinned.dp.cpp`（SYCL 后端自己定义了一版）** ——
头文件注释明确写着 *"an overload, not a default argument: the SYCL build has its own pinned.dp.cpp that defines the signature above only"*。

**改动清单（5 个文件，约 150–200 行；大页不动）**

| # | 文件 | 改动 |
|---|---|---|
| 1 | `include/strata/core/pinned.hpp` | `load_experts_ranges` / `load_experts_direct` 各加 `uint64_t dst_bias = 0`（3 处重载声明） |
| 2 | `src/core/pinned.cu` | 转发生效 + `memcpy(dst + off - dst_bias, ...)`（两个 loader） |
| 3 | `src/core/pinned.dp.cpp` | SYCL 后端的同名定义同步加参数（否则 SYCL 构建断） |
| 4 | `src/core/expert_source.cpp` + `include/strata/core/expert_source.hpp` | `ArenaExpertSource` 加第二块：`arena1_`（第二个 `PinnedArena`，同样大页）、`base1_`、`boundary_off_ = lay.layer_offset(K)`。`open()` 接 `split_layer`：块0 装 [0,K)，块1 装 [K,48)（`dst_bias = boundary_off_`）。`blob_at()` 按层选块；**块1 释放后返回 nullptr** ⇒ 池子响亮失败。新增 `release_tail()`（`delete` 掉 `arena1_`，整块 `MEM_RELEASE`），返回释放字节数 |
| 5 | `src/program/generate.cpp:4995` / `6478-6498` | ① `open(..., split_at[0])`；② **拿到 `host_res` 之后**判定：末段 stage 的每一层 `resident == n_expert`（512/512）⇒ 调 `release_tail()`；否则不释放。开关 `STRATA_ARENA_RELEASE_BLOCK=1`（默认关）；日志打印释放 GiB 与层数 |

**分三次提交，每次都能单独编译**：① pinned（3 文件）→ ② expert_source 双块 → ③ generate.cpp 判定与释放。

### 4.8 实装进度与真正卡住的地方（2026-10-19 晚）

| 提交 | 内容 | 状态 |
|---|---|---|
| `51538d55` | `load_experts_ranges` 的 `dst_bias` | ✅ 编译通过 |
| `5f1ad550` | `ArenaExpertSource` 第二块骨架（成员 / `set_split_layer` / `release_tail`） | ✅ 惰性 |
| `1a957d55` | 双块装载 + `blob_at()` 分流 + `host_res` 之后的释放判定 | ✅ 编译通过 |

**实测：机制没触发**（日志无 `arena-skip:`）。根因不是代码 bug，而是**数据来源**：

```
packs\iq3_s\  =  tokenizer / dense.bin / index.txt / native_experts.txt      ← 没有 experts.bin
models\IQ3_S\ =  51.05 GB + 26.82 GB  GGUF 分片
```

⇒ 本机走 **`from_gguf`**（`native_experts.txt`），专家由 **`load_experts_gguf()`** 从 GGUF 分片读，
不是 `load_experts_ranges()` 从 `experts.bin` 读。而我在 `tail_ok` 里排除了 `from_gguf`，所以尾部块根本没建。

（顺带纠正一处旧观察：日志里的 `51.1 GiB of files` 是**第一个 GGUF 分片**的大小，不是 experts.bin。）

**打通本机的最后一处**：`load_experts_gguf()` 需要**两个**新参数 —— 不只是 `dst_bias`：

- `dst_bias`：它的目标地址由一个 lambda 给出（`expert_source.cpp:4305-4307`）
  `pad ? pad->blob_off(l,e) : lay.blob_offset(l,e)` ⇒ 减 `dst_bias` 即可；
- **层范围 `l0/l1`**：它按 `lay.n_layers` **遍历全部 48 层**（`atomic<int64_t> next{0}` + `l >= lay.n_layers`），
  不像 ranges 版接收 `loff/lbytes` 子集。**不加层范围就会把 48 层都写进第二块**（越界）。

改完后：`tail_ok` 去掉 `!from_gguf`，两块的调用分别传 `[0,K)` / `[K,48)` 与 `dst_bias = 0 / boundary_off_`。

**核对过、不需要额外守卫的**：`pinned()` / `device_alias()` / `pcie_layer()` 对尾部层天然返回
`false` / `nullptr`（第二块 `max_pinned_bytes=0` ⇒ `registered_bytes=0`，而 `pinned()` 走
`blob_offset + bytes <= pinned_bytes_` 判据）。

**回滚**：`engine\strata.exe` 已还原为验证过的旧版；新构建只在 `build-cuda12\`（`build_arena_check.cmd` 只编不装），
旧引擎备份在 `D:\strata-blackbox\engine-arena-skip\`。

### 4.9 ✅ 实装完成并实测（2026-10-19 22:2x）

打通 GGUF 路线后（`load_experts_gguf` 加了 `dst_bias` **和层范围 `l0/l1`** —— 它原来遍历全部 48 层），实测：

```
strata generate: arena-skip: layers 27-47 are fully resident: 21.71 GiB of the expert arena handed back to the OS
```

| 指标 | 改前 | 改后 |
|---|---:|---:|
| 起服务后 **commit 余量** | **0.4 GiB**（已到 0xC0000409 边缘） | **21.1 GiB** |
| 进程占用 commit | ~64.5 GiB | **~43.4 GiB** |
| 真实流量回放 decode | 106.2 | **106.7**（配对中位 −1.2%，5 正 5 负 = 噪声） |
| `could not produce a blob` | — | **0 次**（"永不读"成立） |
| 分段占比 | 96.65/0.20/3.10 | 96.30/0.20/3.50（槽位波动） |

**开关**：`STRATA_ARENA_RELEASE_BLOCK=1`（默认关；**关闭时仍是单块 46.8 GiB，逐位不变**）。
已写入日常启动脚本 `start-2gpu.cmd`。

**踩到的两个坑（都已修）**：
1. `lay.bytes[l]` 是**每个专家**的字节数，不是每层 —— 求和要乘 `lay.n_expert`（否则报
   `the load read 98227200 B of 50292326400`）；
2. 用脚本批量替换 `for (int64_t l = 0; l < lay.n_layers; ++l)` 时**误伤了 `open_gguf`**（同文件另一函数），
   编译报 `l0/l1 未定义` —— 教训：这类替换要带足够上下文，不要只匹配循环头。

**待确认（不影响采用）**：回放里首 token 配对中位 **+27%（8 升 2 降）**，但那是 **#56 的 204K 冷读** ——
本次启动刚用 unbuffered 读了 50 GB，文件缓存是冷的；decode 不变说明解码路径没受影响。
要用热缓存复核，把同一 suite 再跑一次即可（4.4 分钟）。

### 4.10 ✅ 零专家读盘的两档验收（2026-10-19 23:0x）

同一套 10 条回放，只差 `STRATA_ARENA_RELEASE_BLOCK`：

| 指标 | release **开** | release **关** |
|---|---:|---:|
| **`file_blobs` / `file_mb`（每条）** | **0 / 0**（10 条全零） | **0 / 0**（10 条全零） |
| `ram_blobs` | 0 | 0 |
| 进程 `ReadTransferCount` 增量 | 6,201 MB | 6,170 MB（**差 0.5%**） |
| VRAM / PCIe / CPU 中位 | 96.70 / 0.20 / 3.10 % | 96.15 / 0.25 / 3.55 % |
| decode 中位 | 106.5 | 110.2 |
| `could not produce a blob` | **0** | **0** |
| 起服务后 commit 余量 | **21.4 GiB** | **0.5 GiB** |

**结论**：专家 **从不回盘**（`file_blobs` 全零），两档一致 ⇒ 释放不引入读盘。
release 关的那一档 commit 只剩 **0.5 GiB**，这本身就是开它的理由。
（decode 106.5 vs 110.2 是**跑的顺序**：开的那档紧接启动、文件缓存被 50 GB unbuffered 装载冲掉；块内方向也不一致，机制上释放的块从不被访问。）

### 4.11 ⚠ prefill 期间的 SSD 读来自 **PLE（n-gram）表**，不是专家

实测：`file_blobs = 0`，但进程 **仍**在 prefill 期间规律读盘（几十~上百 MB/次）。来源是 PLE 表：

```
strata generate: PLE on, table 320001536 rows of ...IQ3_S-00002-of-00002.gguf
strata generate: PLE direct read probe: 195196 rows/s (fine)
```

`generate.cpp:399` 的默认值就是 `direct`，帮助文本与 `docs/DETAILS.md` 都写明：

> `--ple-io direct` (default): **unbuffered** n-gram table reads, **a few rows per token**, past the OS cache

⇒ 每个 prefill token 查几行 28.8 GB 的表，**绕过 OS 缓存**，所以**每次 prefill 都重新读** —— 与"不定时几十到上百 MB"完全对上。

| 取值 | 内存 | 备注 |
|---|---|---|
| **`direct`（默认，推荐保留）** | **0** | 引擎探针实测 **195K rows/s**（告警线 15K 的 13 倍）；16 个 in-flight 读与计算重叠 |
| `mmap` | **不计 commit**（文件映射，干净页可回收），但会吃页缓存 | 不占 28.8 GB 私有内存；**Windows 上有 #1984 的 mapped-file 卡死 prompt 报告** |
| `ram` | **~28.8 GB 私有**（计 commit） | 本机 release 后只剩 21.4 GiB ⇒ **装不下**，会撞 `0xC0000409` |

**建议保持 `direct`**：探针健康、零内存代价，且真实请求 99.9% 前缀复用（每条只新读几百 token）。
要试只试 `mmap` 一档，判据 = `#56` 的 first token + 进程读字节增量。

**2026-10-19 23:0x 实测 `mmap`（已采纳，用户接受 #1984 风险）**：同一套 10 条、都开 release，只差 `--ple-io`：

| 指标 | `direct` | `mmap` |
|---|---:|---:|
| 首 token 中位 / `#56`(204K) | 3.03 s / 3.34 s | **3.06 s / 3.38 s** |
| decode 中位 | 106.5 | 107.8（+1.2% = 噪声） |
| 墙钟 | 262 s | 263 s |
| `file_blobs` / `file_mb` | 0 / 0 | 0 / 0 |

⇒ **prefill 没有变快**（单次回放里每条 prompt 只读一遍，页缓存无重复命中可利用），
decode 也不受影响（PLE 只在 prefill 用）。模式确实切换了：mmap 档日志里**没有** `PLE direct read probe` 行。
**已写入规范配置 `strata-iq3_s-2gpu.json`**（`--ple-io mmap`，44 参数、无重复），
理由是用户接受 Windows mapped-file 的风险、换取后续观察空间；**内存上安全**（文件映射不计 commit）。
⚠ 该档的"是否减少 SSD 读"**没有测准**：I/O 采样取的是旧进程 pid（`replay_arm` 中途重启引擎），要严谨须在 `replay.py` 内嵌 I/O 采样。

> **为什么按块而不是按 pair**：大页不能部分 decommit，但能整块 `MEM_RELEASE`；而 V100 的 21 层正好是连续区间。
> **为什么只对全覆盖的 stage 释放**：全满的层没有空槽可 `admit` 新专家 ⇒ 自适应层**永不换出** ⇒ 那 21.7 GiB **可证明永不被读**（3090 是 314/512，不满足，保留）。

**验收**（三道，缺一不可）

1. 启动日志出现 `expert arena: N.NN GiB of VRAM-held experts handed back to the OS`，且 **N ≈ 21.7**
2. `Win32_OperatingSystem.FreeVirtualMemory`（KB 单位）在起服务后**比改前多约 21.7 GiB**；`experts loaded` 仍是 46.84 GiB（拷贝照旧，只是拷完就还）
3. **真实流量回放**（`replay.py --suite 56:6,0:4`）与 `replay-base-cur.jsonl` 逐条配对：**decode 与三段占比都不变**（±2%）；若出现 `the expert source could not produce a blob` ⇒ 覆盖判定有 bug，**立刻回退**

**风险**：`host_res` 在自适应层换出后由 `res_put()` 更新，而 `released_` 是静态的 —— 所以第 4 步的"只释放全覆盖层"是**正确性的前提**，不是优化。**任何放宽都会导致静默错误。**

---

## 5. 解决 94% 的三个杠杆（与 §4 完全独立的另一条线）

### 5.1 `STRATA_PCIE_MIN1=1`（成本最低，先做这个）

修「整数份额取整成 0」：一层有 2 个 miss 就送 1 个走 PCIe。
预期把 CPU 池那 5% 里的一部分搬到 PCIe（单次快 3–6×）。
配套可能要把 `--pcie-frac` 从 auto（0.37/0.18）往上调到 **0.5–0.6**（文档那组实测就是配 0.6）。

**验收判据**：日志里 `offloaded` 占比从 **0.1–0.3%** 涨到 **1–2%**，同时 `lookups − hits` 同步下降。

### 5.2 学一个属于自己 agent 流量的 profile（收益最大）

```jsonc
"expert_profile_save": "expert-profile-learned.bin",
"expert_profile_save_every": 5      // 分钟；0 = 只在干净退出时写
```

- 引擎会存「显存里已有的 → 你这段的路由计数 → 出厂序」，下次启动优先用它
- 文档实测：**同类文本 +13~19% 解码；异类文本 −18%** ⇒ **按工作类型分文件**（agent 一个、长文一个）
- 离线路线：`--dump-routing trace.bin` + `tools/make_profile.py --reorder`

### 5.3 `STRATA_PIN_BY_NEED=1`（Windows 专有问题，正好打在我们的双卡结构上）

`docs/DETAILS.md:188-193`：Windows 页锁定只能到**约一半内存**（WDDM 共享预算），所以 **arena 只注册前 ~30 GiB，其后的层拿不到 PCIe 份额**。
本机 arena 46.84 GiB ⇒ **V100 的层（27–47）在文件尾部，很可能整段没有 PCIe 份额**。
改法：按 profile 排名注册"非 GPU 常驻"的专家，`STRATA_ARENA_PIN_GIB=29`（Windows 默认）。
实测：长 prompt 读 **+5~7%**，解码 **+2~3%**（配 `--pcie-frac 0.6`）。

> 注意 §4 的改动会让 arena 缩到 25.1 GiB —— **那时 30 GiB 的锁定上限就不再是瓶颈了**，§5.3 可能自然消失。**两条线有联动，建议 §4 先做。**

---

## 6. 测量与验收方案（先量，再改）

**基线（必须用你自己的 agent 流量，不要用 BENCH —— BENCH 就是 99.3% 的来源，会掩盖问题）**

抓四样：

| # | 指标 | 来源 |
|---|---|---|
| 1 | VRAM 命中率 / CPU 池占比 / PCIe 占比 | 日志 `decode expert cache hit rate` 行，按 §1.1 的公式算 |
| 2 | 每次请求的 hit rate | `/metrics` → `requests[0].hit_rate`（卡片同源） |
| 3 | 启动后 commit 余量 | `Win32_OperatingSystem` 的 `FreeVirtualMemory`（**单位 KB**） |
| 4 | `experts loaded: X GiB` | 日志启动行（验证 §4 是否生效） |

**逐项验证**

| 改动 | 判据 |
|---|---|
| `STRATA_PCIE_MIN1=1` | `offloaded` 占比 0.1–0.3% → 1–2%；CPU 段下降；BENCH 4 个数不退 |
| `expert_profile_save` | 同负载第二轮 VRAM 命中率上升（94% → 96%+） |
| §4 的 arena 排除 | `experts loaded` 46.84 → ~25.1 GiB；commit +21.7 GiB；**被排除 pair 的读取次数为 0**；速度不变 |

---

## 7. 优先级

| 顺序 | 事项 | 成本 | 收益 |
|---|---|---|---|
| **1** | `STRATA_PCIE_MIN1=1`（+ 视情况 `--pcie-frac 0.5–0.6`） | 一条环境变量 | 把 5% 里的一部分从 CPU 搬到快 3–6× 的 PCIe |
| **2** | `expert_profile_save` 学 profile | 配置两行 + 跑真实流量 | 抬命中率上限（同类文本 +13~19%） |
| **3** | **§4：arena 排除「512/512 全覆盖层」** | 150–400 行源码 | 省 21.7 GiB RAM / commit、启动快一倍；**速度不变** |
| 4 | `STRATA_PIN_BY_NEED=1` | 一条环境变量 | +2~7%；但**若先做 3 则可能不需要** |
| ❌ | 动 `K`（如 25） | — | 会让 V100 的层从全满掉到 91%，**更差** |
| ❌ | `--resident-experts`（为省 RAM） | — | 池要服务 5%，切 mmap（19.41→34.78 ms/token）**会真的变慢**；且同样含"跳过卡内专家"但以整体 mmap 为代价 |
