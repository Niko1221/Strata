# q36 搬 Flash-Next 的权重流式（Stager）：否定结论

- 状态：**结论定稿，不实施**
- 日期：2026-10-21
- 机器：192.168.1.98，CUDA0 = P100-12G（主卡），CUDA1/2/3 = P104-8G（helper），helper 链路 PCIe x4，**无 P2P**
- 模型：Qwen3.6-35B-A3B-UD-Q4_K_XL
- 提问：把 Flash-Next 的 `Stager`（16 深 pinned ring + 多主机线程）搬进 q36 的 `stream_kick`（2 深单拷贝流），有没有收益？

## 0. 结论

**没有收益。不要做。**（MEASURED 的代码事实 + DERIVED 的推理，见 §2）

不是「做起来太麻烦」，是**两个东西解决的不是同一个问题**：

| | Flash-Next `Stager` | q36 `stream_kick` |
|---|---|---|
| 位置 | `src/prefill/prefill.cpp:519-705` | `src/qwen35/q36_engine.cu:3264-3332` |
| 拷贝源 | mmap / pageable / GGUF **现场读** | 加载时就进 `cudaHostAlloc` arena 的**已钉住内存** |
| 16 深 + 多线程买什么 | 发射线程不再被每专家约 2 MB 的 memcpy 堵住 | —（q36 这条路没有那段 memcpy） |
| 环深 | `kRing = 16`（`STRATA_STAGER_RING` 2–256，`:526`、`:553`） | 硬编码 2（`i % 2`，`:3270`） |

关键：Flash-Next 自己对**已钉住**的 blob 根本不进 Stager——`prefill.cpp:4382` 直接 `continue`，
`:4404-4408` 从 arena 直接 DMA。把 Stager 接到 q36 上，等于 pinned → pinned 再拷一次，
多出来的是一次主机拷贝和一份 pinned RAM，**不是第二条 PCIe**。

## 1. 复核怎么做的

三方独立读代码，互不采信自述：

1. 只读代理查 q36 侧（`stream_begin`/`stream_kick`/`stream_take`/`stream_done`、arena 分配、拷贝字节数）。
2. 只读代理查 Flash-Next 侧（`Stager` 结构、job 语义、ring 深度与线程数的关系、哪些 blob 不进 Stager）。
3. 只读代理专门**推翻**结论，输出 `refuted: false`。

另有一轮 workflow（`wf_ffdb818ebc4a`）独立得出同一裁决 `NO_STEADY_STATE_GAIN`。

**2026-10-21 追加的 10 路独立复核**（10 个代理各查一项，互不看见对方）：本文件 §3 的两处旧结论纠正
和 §2.5 的第三条都来自那轮复核——那轮把「环深 16 会怎样」这个理由改写了（见 §2.5）。

## 2. 依据（每条标 MEASURED / DERIVED / ASSUMED）

### 2.1 源在哪儿（MEASURED）

- `q36_engine.cu:4785`：`cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault)`，**pinned**；
  失败才 `std::malloc`（`:4786-4788`，pageable 回退）。
- `:4809`：`L.hblobs = arena + off`；`:4811-4812`：`expert_blob(L, e, L.hblobs + e * L.stride)`，
  专家字节在**加载时**就写进 arena。
- 运行时 `stream_kick` 的 H2D 读的就是这块 arena（`:3290`、`:3310`）。
- 「pinned 成功」这一枝与本机一致：加载日志打印 `... in pinned host memory`。
  **数字口径**：43/256 每层在 VRAM（3.15 GiB）+ 18.75 GiB 在 pinned host，这句的原始引文在
  `.run/backup-20261020b/worktree.patch:813-815`；现行 `.run/gen-*.txt` 里的实测行是
  `113 of 256 ... (7.22 GiB), all 16.36 GiB`——那是**全部**专家（不是只 hybrid 的那部分），
  两者不要混着引。

### 2.2 拷贝形态（MEASURED）

- `:3289`：`if ((all || 2 * L.n_slots < ne) && owned == nullptr)` → 每层**一次整层** H2D。
  本机 43/256 满足 `2 * n_slots < ne`，且推荐配置 `PREFILL_PEERS` 关 → `owned == nullptr`
  → 走整层拷贝，**不是**按专家的小拷贝。
- `STRATA_Q36_STREAM_ALL=1`（`:3275-3277`）强制整层从主机内存拷，即使有显存槽。
- 环深硬编码 2：`stage[2]` / `ev_copy[2]` / `ev_used[2]`，分配在 `:4863-4869`。
  深度 K 的 staging 占用 = `K * (max_layer + kMmqTail)`（`:4755`、`:4866`）。

### 2.3 为什么加深环不涨稳态吞吐（DERIVED）

- 事件结构：`stream_kick` 让 `cst` 等 `ev_used[b]`（`:3273`），`stream_take` 让 `st` 等 `ev_copy[b]`
  （`:3323`），`stream_done` 记 `ev_used[b]` 后才踢 `stream_pos + 2`（`:3326-3329`）。
- 推断（假设整层 H2D 长于下一层在 `st` 上的计算）：`cst` 在稳态是慢级且保持忙。
  更深的 device ring 只藏启动和 `ev_used` 空隙，**不提高稳态字节率**。
- `:3018` 每个混合 MMQ 层在 `stream_take` 之前 `cudaStreamSynchronize(st)`。
  它挡的是**主机线程**，不是 `cst`（`cst` 不在这条同步上），所以同步不会把拷贝流排空；
  它只缩短 `st` 能盖住下一次 kick 的计算窗口。这是「深度 2 可能有空隙」的机制，
  **不是**「16 深会提高 tok/s」的证据。空隙有多大，**未测**。

### 2.4 不能拿 `PREFILL_PEERS_HOST` 的 +45% 当证据（MEASURED）

`plans/2026-10-20-q36-prefill-helper-vram.md:211`：pp512 158.5 → 229.9 是 **helper 行经 pinned host
三段拷贝**（helper 行切分 + 主机路由），**不是 stream ring 深度**。同文档 `:206`：
无 P2P 时 `PREFILL_PEERS=1` 降级后 pp512 仍是 157.0–157.4。

**关于「带 decode 就崩」的范围**（复核后收紧）：同文档 `:263-265` 记录 pp8+tg128（113.8 / 48.8）
和 pp128+tg128（94.9 / 38.8）**都跑完了**，只有 pp256+tg128 是 `illegal memory access`；
`:401` 另记 `--bench-tg 2` 在 prompt 后第一次同步崩。所以准确说法是
「**窗口 ≥256 时崩**」，不是「任何 decode 都崩」。结论不变：HOST 路线不进推荐配置，
也不能拿来做环深这个实验。

### 2.5 明确不做的事

- **不要**把 `Stager` 接进 `stream_kick`：源已经 pinned，再 memcpy 进第二块 ring 只加一次拷贝 + 一份 pinned RAM。
- **不要**为了「像 Stager」把 q36 的 device ring 改成 16：16 份 staging 会从 12 GB 主卡的槽预算里
  扣掉约 8 倍于今天的显存（DERIVED，见 §2.6）。
- **不要**把 `N=2560` / `K=10` / `NE=512`（`prefill.cpp:131`）或 `PeerExperts` 的 `H=2560`
  （`src/core/peer_experts.cpp:28-29`）写进 q36。q36 是 2048 / top-8 / 256 experts。

### 2.6 16 深环的真实代价，以及一处被推翻的旧理由

**旧理由「槽变少会让每层 H2D 更大」是错的**（10 路复核推翻，MEASURED 代码）：

- q36 的 H2D 形态是 `:3289-3290` 的 `2 * L.n_slots < ne` 判断。本机 43/256 → `86 < 256` 成立，
  所以**每层已经是整层拷贝**（`ne * stride` 字节），H2D 已经是最大值。
  只有在 `n_slots >= ne/2` 时，减少槽位才会让 H2D 变大——当前配置不是这一支。
- **真正的代价是显存**：staging 从专家槽预算里扣。`q36_engine.cu:4755`：
  `staging = 2 * (max_layer + kMmqTail)`，`:4757` 用 `slot_bytes = budget2 - staging` 扣掉。
  深度 K 就是 `K * (max_layer + kMmqTail)`。
- DERIVED（按本模型 Q4_K gate/up + Q6_K down 算）：
  `L.bytes = 2*512*1152 + 2048*420 = 2,039,808`，`max_layer = 522,190,848`（约 0.97 GiB）。
  今天 2 份 ≈ 0.97 GiB，**16 份 ≈ 7.78 GiB**（8 倍）。12 GB 主卡上这不只是「槽变少」，
  `:4766-4775` 的 `need > fr` 检查很可能直接拒绝启动。

## 3. 顺带纠正的两处旧结论
1. **「q36 的 `stream_begin` 只给 MTP 层用」是错的。** `q36_engine.cu:3116-3121`：
   `if (main_hybrid && T > hybrid_t && use_mmq)` → 对**每个混合主层** `lhy(l)` 建 `seq` 并调
   `stream_begin(seq)`。主层确实在用 2 深流式。MTP 层另有一次 `stream_begin({m->layers.size()})`
   （`:3220`）。
2. **「RTX 5090 上 32 线程/128 环比 4/16 慢 4–10%」不是环深对照。**
   `bench/results/2026-10-01-community-rtx-5090-5950x/README.md:6-7` 的 4–10% 是
   0.1.32–0.1.34 相对 0.1.31 的版本回退；`:214-216` 只说专家已在内存时 32/128「似乎稍贵一点」，
   而且**环深和线程数从未拆开单独 A/B**。这条不能当「深环更慢」的测量。

## 3b. 2026-10-21「逐行对比 Flash-Next prefill」那轮的四处纠错

那一轮把 Flash-Next 的 prefill 优化点逐项与 q36 对比，10 个独立复核者各查一部分。
结论方向成立（q36 在 prefill 上确实缺一大截），但**四处细节要先改**，否则会误导取舍：

### 3b.1 peer 行数上限：默认不是「一半」

- `src/program/generate.cpp:8679` 确实按 `prefill_chunk * K / 2` 算默认上限（`K` 取
  `g.n_expert_used`，不是字面 10）。
- 但 `Prefill::set_peer` 把它**丢掉**：`src/prefill/prefill.cpp:2069-2072`，默认 `compact` 为真
  （`STRATA_PF_PEER_COMPACT` 未设即真，无 P2P 时强制真），于是
  `pp->cap_rows = m.T_max * K`（注释原文「no row cap」），`pp->G = m.T_max`。
- 所以默认发的是 **peer 持有的全部行**，不是一半。改说法：「配置里的上限是 `chunk*K/2`，
  但默认 compact 模式会丢掉它」。

### 3b.2 q36 的 prompt MMQ 根本不 gather

- q36 从不调 `gather_native` / `gather_native_group` / `WeightTab` / `prep_group` / `direct_mode`
  （对 `src/qwen35/` 搜索，零命中，MEASURED）。
- 权重直接从 staging 缓冲按 `w + e * L.stride` 就地读：`q36_engine.cu:3022` 的
  `wbase = lhy(li) ? stream_take(li) : L.blobs`，`:3039-3051` 把它交给 `mctx->run(gu)`。
- 所以 Flash-Next 的 `STRATA_MMQ_DIRECT`（不 gather、指针表就地读）**在 q36 没有对应物**：
  q36 本来就不 gather，它付的只是那次 staging DMA。
- 两层都不逐专家：q36 是**一次覆盖全部专家的 gate/up 乘积 + 一次全部专家的 down 乘积**
  （`:3040-3051`），不是每专家两次。

### 3b.3 621→785 那组数字不是发射线程的

- `src/prefill/prefill.cpp:174-178` 那组（5070，1.5K/2K/4K：621→785 / 727→934 / 779→912）
  变的是 **chunk 从 2048 到 1024**（`stream_all_min`），斜线后是 IQ2_XS 那一支。
- 独立发射线程（`STRATA_PREFILL_ISSUER=0`）自己的对照是 commit `8baae071`：
  IQ3_S 32K prompt **1143 → 1213 tok/s**，「wait copy」15.2% → 7.5%。
- 归属不能混：写「chunk 阈值」就对，写「发射线程」就错。

### 3b.4 q36 的拷贝是异步的，不是「同步」

- `stream_kick`（`q36_engine.cu:3270-3315`）用 `cudaMemcpyAsync(..., cst)` 发到**专用拷贝流** `cst`，
  由 `cudaStreamWaitEvent(cst, ev_used[b], 0)` 把关。它是**在调用线程上内联发出**，但不是阻塞式。
- 准确说法：「在调用线程上内联发出异步拷贝」，不是「同步拷贝」。
- q36 的 `STRATA_Q36_STREAM_ALL` 是 **opt-in**（`:3277`）且语义是「整层从主机拷」，
  不是 Flash-Next 那种「≥1024 token 阈值 + 按 id 顺序流式」。

### 3b.5 环深那组数字没有标机器

- `prefill.cpp:231-232`：「measured on Q2_0, 8192-token chunks: 96 slots 1153 tok/s, 384 1294」。
  **没写 RTX 5070**；5070 是邻近另一组（chunk 1024→6144，87→402）的机器。
- 两句比较的东西不同：前一句变的是 **ring slot 数**（量化同为 Q2_0，chunk 同为 8192），
  后一句变的是 **chunk 大小**。不能合并成一句。

## 4. 想推翻这个结论，要先补的测量

在同一台机、同一模型、推荐配置（`PREFILL_PEERS` 关、`PREFILL_PEERS_HOST` 关）下：

1. 用**可计时事件**量 `cst` 在 pp512 稳态的**忙闲比**（现在 `ev_copy`/`ev_used` 是
   `cudaEventDisableTiming`，`:4868-4869`，量不出来）。**只有 copy-wait 占 wall-clock 的
   可测份额，才有理由改 device ring 深度**；而且只改深度、不改 host 填充。
2. 多线程 host fill 只有在确认 `hblobs` 走了 `:4788` 的 pageable 回退时才相关。
   本机 18.75 GiB 走的是 pinned 那一枝，与该回退矛盾，所以这一条现在不成立。

## 5. 顺带记下的本轮其它实测（同机同日）

| 项 | 数字 | 说明 |
|---|---|---|
| `--suffix-draft 7`（新开关，opt-in） | 重复内容 113.5–113.8 tok/s，对照 75.7 tok/s（**+50%**） | 推荐配置；生成序列逐字节相同；`--tokens` 路径 |
| `--suffix-draft 7`，随机 id prompt | 58.5 vs 59.2 tok/s（持平） | 命中不了就退回 MTP 窗口 |
| `--batch-mtp`（serve 双并发） | slot 1 41.1→48.2 tok/s（**+17%**），slot 0 约 +5%，两路合计 +12% | 3 次同向；两 slot token 序列逐字节相同 |

`--suffix-draft` 与 `--batch-mtp` 是两件不同的事：前者是**单路**用历史重复代替 MTP 草稿，
后者是**多路并发**时每个 slot 多验一条 MTP 草稿。

`--batch-mtp` 只能在 serve 多 slot 路径上生效：`--tokens` 和 `--bench-tg` 都走不到它。
`.run/bmtp_q36.sh` 用的就是 `--tokens`，所以那次测不到这条路径。
