# P100 + 2×P104 三卡跑通 Q36（Ornith-1.5-35B-A3B-IQ4_XS）——长任务方案

状态：进行中。目标：在 192.168.1.98 上用 Q36 引擎把 Ornith-1.5-35B-A3B-IQ4_XS 跑通：
P100-12G 做主卡（CUDA0），两张 P104-8G 做助手卡（CUDA1/CUDA2，`--expert-cache-device1/2 auto`）。

## 已完成：三卡跑通，输出与单卡逐位一致

2026-10-17 实测（`--max-context 2048 --no-warmup --tokens 1 --n 32`，每组 32 token）：

| 配置 | decode | GPU 命中率 | CPU 池耗时 | 助手卡条目 | 输出 |
|---|---|---|---|---|---|
| 单 P100 | 20.6 tok/s | 46.5% | 1136 ms | - | 基准 |
| + 1xP104 | 23.3 tok/s | 71.7% | 547 ms | 4652 | 一致 |
| + 2xP104 | 24.9 tok/s | 94.5% | 111 ms | 9289 | 一致 |
| + 2xP104 + `--kv int8` | 25.7 tok/s | 94.6% | 107 ms | 9292 | 一致 |

输出（`gen:` 行）在四组之间**逐位相同**，含 int8 KV。日志：`.run/gen-A_single.txt` 等。

## 修掉的三个 bug（本轮，只改 `src/qwen35/q36_engine.cu`）

1. **助手卡布局表多一层**：MTP 层被塞进只该覆盖 40 个主层的 `expert_layout_set_native` 表，
   报 `the format table has 41 layers but n_layers is 40`，助手卡起不来。MTP 专家按设计留主卡，不该进该表。
2. **助手卡层跳过主卡全部计算**：整层交给助手卡后本地不算。但 P104 只有 8 GiB，
   `auto` 只装下 4630 / 10240 个专家，**其余 5610 个无人计算**，输出成垃圾，
   而 tok/s 反而更高（47.4），因为跳过了大半计算。
3. **改为逐条目归属**（照 Qwen3.8 Flash，`expert_source.cpp` 的 `kind[i] = 2` / `0` / `-1`）：
   新增 `k_group_masked` / `k_group_hy_masked` 按掩码分组；`moe_helpers` 回传助手卡算了哪些条目；
   主卡补算其余，CPU 池避开助手卡的。`open()` 只收主卡无槽位的专家，三边天然不重叠。

## 设计教训

**整层划分在小显存助手卡上是错的。** 原假设"整层交给助手卡 => 归属无歧义"只在助手卡
装得下整层时成立。P104 只有 8 GiB，装不下。必须逐条目归属。


## 事实（已实测，别再查）

- **CUDA 编号是 FASTEST_FIRST**：进程内 CUDA0=Tesla P100-PCIE-12GB（sm_60，11.9 GiB）、CUDA1=P104（sm_61）、CUDA2=P104。`nvidia-smi` 显示的是 PCI 总线顺序（P104/P100/P104），两者不同，别混。`CUDA_VISIBLE_DEVICES=0` 钉主卡。
- 模型：`/home/models/Ornith-1.5-35B-A3B-IQ4_XS.gguf`，19,278,554,784 字节，sha256 `d6aef57f...`。qwen35moe，40 主层 + 1 MTP，256 专家，top-8，n_embd 2048，n_ff 512，head_dim 256，16 query heads / 2 KV heads。
- 二进制：`/home/Strata/bin/strata-q36`，由 `/home/Strata/src`（分支 `qwen36-multigpu`，HEAD `c6881162`）编出，含 sm_60 + sm_61。
- 单 P100（`--no-warmup`）：prompt 16.5 tok/s，decode **25.7 tok/s**，46.6% batch 条目在 GPU，每层 113/256 专家常驻（7.22 GiB）。
- 远程：Rocky Linux 9.8，CUDA 12.8.93，gcc 11.5，31 GiB RAM + 15 GiB swap。SSH 参数在 `C:\Users\MAO\.run\ov-ssh-98.ps1`（plink 路径、固定指纹、口令），**口令不写进任何新文件**。

## 已知问题

1. **默认路径（warmup 开）在第一次真实命令时崩**：`native MMVQ launch: an illegal memory access`，dmesg Xid 13 "Out Of Range Address"，发生在 P100（PCI:02:00）。`--no-warmup` 能跑通且输出正常（`Help me with: 1. **Task 1: [The`）。
   - 已排除：`--no-graph`、`STRATA_Q36_STREAM_ALL=1`、`STRATA_Q36_FIXED_SLOTS=1`、`STRATA_Q36_KEEP_PACKED=1` 都不能避免。
   - 嫌疑：`Engine::reset()`（`q36_engine.cu:1439`）清了 conv/S/hlast 和 `pos`，但没清 `stream_seq`/`stream_pos`（:1259-1260）；warmup 在 `reset()` 前跑过 hybrid 流式读取。但 `stream_take()` 有硬性 guard（exit(1) 报错），所以这个假说还要解释它怎么绕过 guard。
   - 上游有两个同类 issue：#2778（prefill copy_i32 illegal memory access）、#2666（peer-expert tier 崩在第一次 prompt），但都是 sm_75/sm_86，环境不同，没有可直接搬的修复。
   - 根因排查 workflow `wf_ecb6516762ce` 在跑（定位 + 上游基线对比 + 独立评审）。

## 阶段

### 阶段 0：修 warmup 崩溃（阻塞所有 benchmark）
- 定位真实越界的内核和源码行（`CUDA_LAUNCH_BLOCKING=1` + 二分）。
- 判定是上游遗留还是我们的多卡/int8 改动引入（对比上游 `296b59d2` 的单独构建）。
- 修复，重新构建，验证 `--tokens 1 --n 8`（带 warmup）能跑通。
- 验收：默认路径不再崩，输出与 `--no-warmup` 一致。

### 阶段 1：三卡跑通（P100 主 + 2×P104 helper）
- `CUDA_VISIBLE_DEVICES=0`（P100 主卡），`--expert-cache-device1 auto --expert-cache-device2 auto`。
- 验收：两张 P104 各接管一部分层（看 `helper CUDA1`/`helper CUDA2` 启动行），无 OOM、无非法访问、输出与单卡一致。
- 注意：q36 引擎**没有** Flash-Next 的阶梯规则限制（那条在 `generate.cpp:2379`，q36 不走）。但要实测确认 device1 和 device2 都能开。

### 阶段 2：性能矩阵 + 正确性
- 矩阵：单卡 fp32 / 单卡 int8 / 一张 P104 / 两张 P104 / 两张 P104 + `--mtp 1` / 32K 上下文。
- 每组记：prefill tok/s、decode tok/s、首/末 token 延迟、每卡显存、hybrid 命中率、MTP 接受率、输出 hash。
- 正确性：同一 prompt 下，开/关 helper 的输出要一致；`--force` 两次 logits 逐字节相同。

### 阶段 3：循环调优（搜索上游 PR/issue 做参考）
- 搜上游：`gh search issues/prs --repo Niko1221/Strata`，关键词 `expert-cache`、`hybrid`、`decode speed`、`Pascal`、`MTP acceptance`、`KV int8`。
- 候选调优点（实测后再定，不预设）：
  - 专家常驻比例（P104 只有 8 GiB，能放多少专家）
  - `--chunk` 大小（prompt 分块）
  - MTP 草稿数（`--mtp 1` vs 2）
  - hybrid 窗口行数（`--hybrid-rows`）
  - CPU 线程数（`--cpu-threads`）
- 每次只改一个变量，开/关各跑一次对比，记录差值。
- 目标：decode 明显超过单 P100 的 25.7 tok/s。

## 规矩

- 新功能 opt-in：默认路径（不带 `--expert-cache-deviceN`、`--kv`、`--mtp`）必须保持字节级不变。
- 不改 `src/program/generate.cpp`。
- 不 push。
- 远程可自主重启（`reboot`），但重启后要先确认三张卡都在、驱动正常。
- 密码不落盘：从 `ov-ssh-98.ps1` 读。
- 长任务用 workflow 推进，角色显式指定：`explorer` 定位、`coder` 修复、`reviewer` 验证。

## 参考（上游，已读）

- `docs/QWEN36_PASCAL.md`：P100/P104 接法、双架构二进制必要性、`--peer-device`/`--layer-split` 会被拒。
- issue #2693：q36 路径 KV 量化需求（上游没有，我们已移植 PR #2698）。
- issue #2778 / #2666：多卡 illegal memory access（环境不同，仅参考）。

---

## 2026-10-17 变基后的实测（提交 4fe4bd3f）

变基到上游 `477ac13c`（v0.1.45）后重做逐条目归属，并把 warmup 越界修掉。远程原生编译，
`/home/Strata/bin/strata-q36`（87,618,040 字节，md5 e84518a3c1d80a009979c51291f66700）。

### 一致性：四组输出逐位相同

命令：`--max-context 2048 --no-warmup --tokens 1 --n 32`（模型 Ornith-1.5-35B-A3B-IQ4_XS.gguf）

| 配置 | decode | GPU 命中 | CPU 池 | helper 条目 |
|---|---|---|---|---|
| 单 P100 | 21.5 tok/s | 46.5% | 1111 ms | — |
| +1×P104 helper | 23.4 tok/s | 71.7% | 556 ms | 4651 |
| +2×P104 helper | 25.5 tok/s | 94.5% | 120 ms | 9283 |
| +2×P104 + `--kv int8` | 25.6 tok/s | 94.6% | 117 ms | 9286 |

共同输出（与变基前实测的那份完全相同）：

```
198 12325 728 440 25 271 16 13 2972 6065 220 16 25 498 760 220 18 15 53066 24455 75654 198 256 471 2972 38663 25 2972 198 256 471 2972
```

### warmup 越界已修

`warmup()` 原本连跑三次 `process()` 却不重置 `pos`，峰值到 `chunk + 9`，越过 `ctx` 写 K/V。
改成每轮归零、并把 chunk 夹到 `ctx`：

- 单卡 + warmup（原先必崩，rc=134 `native MMVQ launch: an illegal memory access`）：现在
  `warmed up in 25.2 s`，输出与 `--no-warmup` 一致。
- 两张 P104 + warmup + `--kv int8`：`warmed up in 23.6 s`，32 token 输出同上。
- `--chunk 2039 / 2047 / 2056` 报的不是越界而是 `cudaMalloc ... out of memory`：这些 chunk 的
  缓冲超过 P100 剩下的 8.03 GiB，与 warmup 修复无关。真要测越界要挑显存装得下的 chunk。

### 长跑稳态与 MTP

- `--bench-tg 128`（两 helper + int8）：**24.8 tok/s** 稳态（第 1、2 轮都是 24.8）。与 32 token
  短样本的 25.6 一致，说明短样本没有高估。
- `--mtp 1`（两 helper，无 int8）：接受率 **4/27 = 15%**（此前记为 0 有误），输出逐位一致，
  但 **16.8 tok/s**，比不开 MTP 的 24.9 慢 33%。每步多出的 MTP 前向 + 验证窗口开销超过了
  省下的步数，所以在这台机器上 `--mtp` 目前是净负收益，保持默认 0。

### 调优方向（下一轮）

命中率 46.5% → 94.5%（2 倍）只换来 21.5 → 25.5 tok/s（+19%），CPU 池时间从 1111 掉到
120 ms。上游 explorer 的结论与实测一致：`moe_helpers()` 每层做两次主机拷贝 + 一次
`cudaStreamSynchronize`，40 层就是 40 次主机往返，且与主卡的下一层完全串行——这是真正的
瓶颈，下一步就该从这里下手。

---

## 2026-10-18 瓶颈实测与 warmup 越界的独立复核

### warmup 越界：上游遗留，已独立确认

一个独立的诊断任务（`CUDA_LAUNCH_BLOCKING=1` + 上游基线对照）给出三条我们之前没有的证据：

1. **越界的确切位置**是 `k_attn_prep` 的 K/V 存储，`src/qwen35/q36_engine.cu:337`：
   `((float*)kc)[(kh*ctx + pos)*HD + i] = v`，`pos` 从不与 `ctx` 比较（int8 变体在 :324/:341）。
   加 `CUDA_LAUNCH_BLOCKING=1` 时崩在 `iq_embed_rows`
   （`src/kernels/cuda/iq_kernels.cu:3508`，第一个真实请求的第一个 launch），
   不加时崩在 `native MMVQ launch` 或 `cudaMemcpyAsync(ecpu, ...)`
   （`q36_engine.cu:2055`）——典型的 sticky error 滞后。

2. **不只是 warmup 的问题**：`--no-warmup` 下把 600 个 token 灌进 512 格缓存同样崩，
   500 个才通过。说明 `process()` 缺的是"`pos` 不许越过 `ctx`"这一道防护，warmup 只是
   最容易撞上的入口（默认 chunk 2048 + 9 + 1 = 2058 > 2048）。

3. **上游 `296b59d2` 复现同样崩溃**（`/home/Strata/build-upstream/strata-q36`，源码 md5 已核）。
   多卡、int8 KV、计算图、混合槽位表都被排除：上游不带任何 helper 也崩，
   `--no-graph` 崩得一模一样。

对策分两层：
- 已做（`4fe4bd3f`）：`warmup()` 每轮把 `pos` 归零，并把 chunk 夹到 `ctx`。实测通过。
- 待做（纵深防御）：在 K/V 存储处加 `pos < ctx` 的钳制，并把越界报出来而不是静默写坏
  相邻分配。这一处要等当前的 overlap workflow 收尾后再动，避免同时改同一个文件。

### 真正的瓶颈：不是暂存，是主机阻塞等结果

把 `RemoteExperts::ms_begin()` / `ms_wait()` 临时打到 stderr（用完即删、远端已重建干净）：

| 项目 | 每 token |
|---|---|
| `ms_begin`（暂存 + launch，两张助手卡合计） | **1.50 ms** |
| `ms_wait`（`finish()` 等助手卡算完 + 回传，两张合计） | **21.07 ms** |
| 合计 | **22.58 ms** |

对照每步 39.3–40.2 ms，**22.58 ms 占 56%**。由此：

- 删掉 `moe_helpers()` 里那次 `cudaStreamSynchronize(st)` 最多只赚 1.5 ms/tok（约 +1 tok/s），
  因为那 21 ms 花在 `RemoteExperts::finish()` 的阻塞等（`src/core/remote_experts.cpp:401`，
  CUDA 构建走阻塞分支，没有 spin）。
- 把整条助手主机路径完全重叠掉，上限约 **56 tok/s（2.2 倍）**；而主卡自身剩余的
  17.8 ms/tok 也正好算出 56.1 tok/s。两条路减出来是同一个数，说明这个上限是真的。
- 助手卡其实是**过度取用**：每 token 约 289 个条目分给两张卡，PCIe 行数据只有约 59 MB/s。
  所以瓶颈既不是助手卡算力也不是 PCIe 带宽，而是主线程同步等待。

这解释了之前那个反直觉现象：命中率 46.5% → 94.5% 只换 19% 提速，因为省下的 CPU 池时间
本来就和 GPU 工作重叠着，而 21 ms 的等待一直没变。

下一步方向已定：让它异步，而不是优化暂存。

---

## 2026-10-18 STRATA_Q36_HELPER_ASYNC（提交 b42b4ef3）

把 `moe_helpers()` 里 `finish()` 的**阻塞等待**推迟到 `moe_end` 之前，让助手卡的计算与本层的
本地分组核、共享专家、CPU 池重叠。开关默认关，不开时语句顺序与之前完全一致。

实现上必须偏离最初计划的一处：推迟的**不是**整窗 `helper_out → eparts` 上传，而是
`finish()` 的等待 + 只合并助手自己那些行的 `k_merge_helper`。因为本地分组核在新时点之后才写
它负责的行，整窗上传会用过期的 `helper_out` 字节覆盖刚算好的行。掩码门控的合并只写掩码为 1
的条目，与本地路径的条目集不相交。

### 一致性

四组配置（无 helper / +1 helper / +2 helper / +2 +int8），开关各一轮，`--n 32`，8 个 run
全部与基准行逐位相同；另用 `--n 128` 单独对照开关两边，输出逐位相同
（`GEN128 IDENTICAL`，各 129 个 token）。

> 注意这两个 `--n 128` 说的是不同的东西：这里比的是 **async 开 vs 关**（同一组配置，两边都有
> helper），所以它只证明 async 开关不改变算术，是有效的。而"四配置之间在 `--n 128` 一致"是
> 另一个说法，那个不成立——见下文「已定位：不是缺陷」。

### 长跑交错 A/B（`--bench-tg 128`，两 helper + int8）

| 轮次 | 关 | 开 |
|---|---|---|
| 1 | 25.1 | **26.6** |
| 2 | 25.3 | **26.6** |
| 3 | 24.3 | **26.2** |

+1.5 ~ +2.0 tok/s（约 +7%），开的三轮区间与关的三轮不重叠，超出噪声带。低于 56 tok/s 的
理论上限，符合"只推迟到本层内重叠、不跨层排队"的实现范围。

**测法上踩过的坑（必须记住）**：这个开关按"环境变量存在与否"生效（与文件里其他开关一致），
所以 `STRATA_Q36_HELPER_ASYNC= ./strata-q36` 是**开**不是关——空字符串也算存在。
第一次 A/B 两边都测了"开"，数字看着没差别，差点误判为无效。关必须用
`env -u STRATA_Q36_HELPER_ASYNC`。

### 顺带发现的一处错误说法

实现里曾写"helper 激活时 `use_graph` 已被强制关闭"。这**不对**：`place_experts` 在所有专家
进显存时提前返回，`mtp_only` 分支也把 `main_hybrid` 置 false，这两种情况下 `use_graph` 仍为
true，而 `open_helpers` 不要求 hybrid，所以 helper 能开。
真正的原因是这条路径本来就有两个捕获期非法调用（`moe_helpers` 自己的流同步，以及
`RemoteExperts::finish` 对助手流同步），所以它永远不会被图捕获——与本开关无关。
代码注释已按正确理由改写，否则将来有人加一次真同步就会踩坑。

---

## 2026-10-18 跨层排队方案（wf_1a28aa213c7d）与决定性新测量

### 先看最重要的发现：瓶颈另有其人

Design C 的审计在远程实测发现（我已用三次交错长跑复核）：

- `RemoteExperts` 的 **zero-copy 默认开启**（`src/core/remote_experts.cpp:236-239`）。
- 但本机 `nvidia-smi topo -p2p r` 显示 P100↔P104 全部 NS/CNS（无 P2P）、`topo -m` 显示 PHB。
  无 P2P 时，让助手卡通过 PCIe 直接读写主机固定内存比"先拷到显存再算"更慢。
- **`STRATA_REMOTE_ZEROCOPY=0`**（已有开关，非新代码）：decode 从 25.1-25.3 → **36.1-37.8 tok/s（约 +48%）**。
  再叠加 `STRATA_Q36_HELPER_ASYNC=1` 达 **40.7 tok/s**。输出与基准逐位相同。

也就是说 56 tok/s 上限的大头可以不写任何新代码就拿到。原计划"池化 helper_out/h_x_ 动所有权"
排在第二位；不先关 ZC 就去做环形槽，等 optimize 了错的东西。

### 单槽证据（为什么跨层排队需要动所有权）

`begin(L+1)` 在 `finish(L)` 之前调用时的失效链（首个受害者起）：
1. `h_x_`（`remote_experts.cpp:314`）被 L+1 的 memcpy 覆盖；zero-copy 下 L 的量化核可能还没读
   `z_x_`（= `h_x_`），L 的 q8 激活就成了 L 和 L+1 行的混合。
2. `original_row_`（:279）在 `begin()` 顶部清空；`finish(L)` 的散射（:406-407）拿空表，L 的行
   没人写。
3. `finish()` 的 `cudaStreamSynchronize(stream_)`（:402）只保证"本窗口"的消费完成。

引擎侧 `helper_out`、`helper_mask`/`d_helper_mask` 也是全局单份，比 per-helper 更紧。

### 三个设计与裁定

- **A 池化暂存 + 设备侧合并**（`STRATA_Q36_HELPER_RING`）：只做环形没用，散射必须搬进设备合并；
  摸到 `RemoteExperts`（与 Flash-Next 共享），风险最大，约 600-900 行。**备用。**
- **B 引擎层重叠**：审计证明 L+1 的 `ids`/激活在 L 的 router 之后才存在，"提前取"没有合法时点；
  逐 parity 切分助手卡只改负载分布不改每层等待。**上限噪声级，弃。**
- **C 去掉主机往返**：第一个切片零代码（ZC=0 + async 叠加 = 40.7 tok/s，已实测）；
  设计本体（设备侧分组、新 begin 重载）排在 ZC 关闭复测之后。**胜出。**

### 实施阶段（裁定给出，含止损线）

- **Phase 0（已完成）**：ZC=0 三次交错复测 36.1-37.8，拓扑 NS/CNS，输出逐位相同。
- **Phase 1**：把 helper 路径的推荐环境写进启动输出（或 runbook），加一条
  `STRATA_REMOTE_ZEROCOPY=0` 的 stderr 提示行；**不改 `remote_experts.cpp` 默认值**——
  该文件与 Flash-Next 共享，改默认会动 Flash-Next 行为，禁改。
- **Phase 2（已完成）**：`--bench-tg 128` 交错 4 格 × 9 次，ZC=0 每格都 ≥30 tok/s（最小 36.4），
  止损线未触发。
- **Phase 3（门槛后）**：若 Phase 2 后 `ms_wait`（ZC=0 + async）< 每助手 3 ms/token，则放弃
  A 的环形槽——总助手主机时间 < 5 ms/token，环形最多再买 ~10%，不值。
- **不做清单**：不动 `remote_experts.cpp` 默认值；不做没有设备侧合并的环形；
  不让合并核读映射内存（无 P2P 时读主机内存是本机慢的根源）；不假设 P2P 可用。

### Phase 1/2 实测（提交 ce4c3a9f）

加了**一行** stderr 提示（`open_helpers`，只在开了 helper 且变量没设成 0 时打印一次），
不动 `remote_experts.cpp` 默认值。复核（我本人重跑）：

| 检查项 | 结果 |
|---|---|
| 无 helper 参数的 stderr | 无提示行，`gen:` 与基准一致 |
| 两张 helper，ZC 未设 | 提示恰好 1 行，`gen:` 一致 |
| `STRATA_REMOTE_ZEROCOPY=0` | 提示 0 行，`gen:` 一致 |
| 稳态 ZC=0 + async | **40.1 tok/s** |

4 格交错 × 9 次（每格 9 个 tg128 值）：

| 组合 | tg128 范围 | 中位 |
|---|---|---|
| ZC 开 / async 关 | 24.2 – 25.1 | 24.3 |
| ZC 关 / async 关 | 36.4 – 37.8 | 37.4 |
| ZC 开 / async 开 | 26.0 – 26.5 | 26.1 |
| ZC 关 / async 开 | 39.4 – 41.3 | 40.8 |

区间两两不重叠。**prompt 侧没有退化，反而更快**：pp1 24.6 → 38.3（ZC 关）、
26.4 → 41.0（ZC 关 + async）。合计相对起点约 **+65%**（24.3 → 40.8 tok/s）。

### 更正：n=128 的四配置一致性其实不成立（既有缺陷）——且已定位：不是缺陷

本文件早先写的"`--n 128` 也逐位相同"**是错的**。实测：`--n 32` 四配置一致，但 `--n 128`
在第 32–35 个 token 之后分叉（+1 helper 走 32827、+2 helper 走 271、与无 helper 的结果
也不同）。两个 agent 各自用改动前的二进制证明这是**既有缺陷**，与 stderr 提示无关
（`old_C == new_C`、`old_A == new_A` 逐 token 相同）。

**已定位：不是逻辑缺陷，是 helper 换了算术实现。** `q36_engine.cu` 里"helper 只接 primary
没槽位的专家，所以每个条目按本地路径的算法算"这个说法**不成立**（open 时传给 `open()` 的
是 0 槽的 `empty_primary`，那道 guard 恒真），而且更要紧的是：helper 用
`quantize_q8_1_rows` + `native_expert_grouped`（`remote_experts.cpp:359-360`），无 helper 时
primary 对同一批专家用的是 ggml-cpu AVX（`moe_hybrid_cpu` 的 `native_quant_act` + `run_split_multi_native`，`q36_engine.cu:2837/2855`）。同一份权重、两种
激活量化、两种核，结果自然不同。

**实测（2026-10-18，强制 36 token 提示，token 流被钉死，所以每一处 logit 差异都是算术差异）**
二进制 md5 `91b61243aa0081178b5b678881c26cb2`，`/home/Strata/.run/logit_mag.sh`：

| 对比 | 每行 RMS logit 差 | 相对 | argmax 首次翻转 | 全表最差 max&#124;Δ&#124; |
|---|---|---|---|---|
| A vs C（两 helper） | 1.5e-1 – 3.3e-1 | ~5e-2 | row 35 | 2.30 |
| A vs F（int8 KV，无 helper） | 6e-2 – 3.3e-1 | ~5e-2 | row 4 | 2.05 |

fp32 重结合的尺度是 `eps*sqrt(256)*rms ≈ 1.7e-14`，实测差 **高出约 10 个数量级**——所以不是
"浮点累加顺序变了"能解释的，是换了实现。另外排除了跨设备求和：`k_moe_out`
（`q36_engine.cu:170-171`）仍在 primary 上按固定 pick 顺序累加 K=8 行，`moe_helpers_wait`
（2486）在 `moe_end`（2488）之前把所有 helper 行落进 `eparts`，没有跨设备归约、没有换序。

**裁定：不修。** 要让 helper 逐位等于本地路径，就得让助手卡跑 CPU 池的算法，整个提速归零
（21.5 → 36.2 tok/s 是这个功能存在的全部理由）。两条路都已实测生成通顺的 256 token 文本，
模型质量没变——两种实现都是同一个模型的合法数值实现。helper 分支的验收标准因此改为
"**每组配置自身确定** + 文本通顺"，`--n 32` 的四配置逐位一致仍然保留为硬要求。

`--kv int8` 同理：row 4 翻转处 top1-top2 只有 0.001，就是 int8 量化噪声，是这个 opt-in 标志
本来的语义，不是 bug。

### 下一步优先级（已调整）

1. ~~**查 n=128 分叉**~~ ——已查清，见上：换实现，非缺陷，不修。验收标准已改。
2. **int8-KV 单独走一轮 Explain/Fix**（可选）：它是独立的一套有损近似，值得单独定性，
   但不要挂到 helper 路径的验收标准上。
3. 环形槽（Design A）：止损线已过 Phase 0，暂缓。
