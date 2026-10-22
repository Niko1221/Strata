# q36 peer host 路由的非法显存访问：根因

- 状态：**根因已定（专家步长不一致），修复已写完并编译通过，等机器恢复后验证**
- 日期：2026-10-21
- 机器：192.168.1.98，P100 主卡 + 3×P104 helper，无 P2P

## 0. 结论

**根因（MEASURED）：helper 缓存里两个相邻专家的地址差，和主卡的专家步长不一致。**

MMQ 按 `w + z * (expert_bytes / type_bytes(t))` 找第 z 个专家的权重。这个除法**必须整除**，
否则第 1 个专家之后每一个都从错位的地址读，读出来的是别的字节，算出来是 NaN。

| | layer 33（正常） | layer 34（坏） |
|---|---|---|
| gate/up 类型 | Q4_K（144 字节/块） | Q4_K（144 字节/块） |
| down 类型 | Q5_K（176 字节/块） | **Q6_K（210 字节/块）** |
| 主卡 `L.stride` | 1900800 | **2041200** |
| helper 实测地址差 `delta` | 1900800 ✓ | **2041344** ✗ 多 144 字节 |
| `gu.expert_bytes / 144` | 13200 ✓ | 14176 ✓ |
| `dn.expert_bytes / 210` | 10800 ✓ | **9720.7 → 截断成 9720** ✗ |

**为什么 layer 33 侥幸没事：** 它的 `L.stride = 1900800` 恰好是 256 的整数倍，缓存那次 256 对齐没有
改变它。layer 34 的 2041200 不是 256 的倍数，对齐后变成 2041344，步长就错了。

**为什么是 NaN 而不是报错：** MMQ 不会检查地址对不对。读到的字节被当成合法权重算，算出来是 NaN 或者
巨大值。日志里 layer 34 的 `d_out[0..3] = 4.62242e+06 2.15686e+08 2.17524 -865480` 就是这个样子。

### 完整链条（全部 MEASURED）

1. layer 34 的 helper MMQ 从错位地址读权重 → 输出 NaN / 巨大值。
2. NaN 顺着后面每一层传下去（layer 35..39 全 NaN）。
3. prompt 结束 `hlast[0..3]=nan nan nan nan`。
4. `greedy()` 返回 `0x7fffffff`——那是 `k_argmax`（`q36_engine.cu:894`）在所有 logits 都是 NaN 时的哨兵值。
5. 解码第一步 `iq_embed_rows` 拿这个数当 token id：`2147483647 × row_bytes`，远超词表 248320 → 越界 → IMA。

**所以 IMA 是第 3 步以后的必然结果，不是内存被踩坏。** 给 token id 加范围检查只会把崩溃藏起来。

## 1. 证据

### 1.1 复现（MEASURED）

```
STRATA_Q36_PREFILL_PEERS=1 STRATA_Q36_PREFILL_PEERS_HOST=1 STRATA_Q36_PREFILL_DIAG=1 \
  ./bin/strata-q36 --model $M --expert-cache-device1 auto --expert-cache-device2 auto \
  --expert-cache-device3 auto --bench-pp 160 --bench-tg 8 --no-warmup --max-context 2048
```

- 报错：`cudaStreamSynchronize(st) failed: an illegal memory access was encountered`，RC=1。
- `CUDA_LAUNCH_BLOCKING=1` 追到 `iq_embed_rows`。
- `--bench-tg 0`（纯 prompt）RC=0，pp160 99.8 tok/s；`--bench-tg 1` 同样崩。
- 本机没有 `compute-sanitizer`。

### 1.2 第一个坏层（MEASURED，逐层隐藏态）

| 层 | `h[0..3]` 出来后 |
|---|---|
| 0 | 1.87988 -0.452161 1.08007 0.632513（正常） |
| 1 | 2.41783 -0.292752 0.0639428 0.175445（正常） |
| 2 | 0.838784 -0.0360916 0.0379239 0.452799（正常） |
| 3..33 | 全部正常（layer 33 的 `d_out[0..3]` 是 `0.00408607 0.00229679 -0.0493454 0.0623799`） |
| 34 | **nan nan nan nan** |
| 35..39 | nan nan nan nan |

**layer 33 和 34 唯一的结构差别是 down 的量化类型（Q5_K → Q6_K），以及由此得到的步长。**

### 1.3 步长的两个数字（MEASURED）

```
DIAG primary stride=1900800 bytes=1900544 gu_type=12 d_type=13     (layer 0)
DIAG helper dev=1 layer 0 addr check: 0 of 256 experts mismatch delta=1900800
DIAG helper dev=3 layer 33 addr check: 0 of 256 experts mismatch delta=1900800
DIAG helper dev=3 layer 34 addr check: 0 of 256 experts mismatch delta=2041344
DIAG helper dev=3 layer 34 expert_bytes=2039808 gu_type=12 d_type=14
```

- `expert_bytes` 是原始 blob 大小（2039808），`delta` 是缓存的 slot 间距（2041344），
  主卡的 `L.stride` 是 2041200。**三个数都不同。**
- 缓存报 `0 mismatch` 是对的——它检查的是自己的地址和自己的 delta 一致，那当然一致。
  **它没检查的是：这个 delta 能不能被 MMQ 的块大小整除。**

### 1.4 主卡的步长规则（MEASURED 代码）

`q36_engine.cu:1467`：

```cpp
const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
y.stride = (y.L.bytes + al - 1) / al * al;
```

- 取两种 block 大小的最小公倍数再对齐，**保证 `stride` 能被两种 block 大小整除**。
- layer 34：`lcm(144, 210) = 5040`，`align_up(2039808, 5040) = 2041200`。
  `2041200/144 = 14175` ✓，`2041200/210 = 9720` ✓。

### 1.5 缓存的步长规则（MEASURED 代码）

`expert_cache.hpp:134`：

```cpp
static int64_t slot_stride(int64_t blob, int64_t guard = 0) { return (blob + guard + 255) / 256 * 256; }
```

- **只对齐到 256，不管 block 大小。** 这就是错的来源。

## 2. 修法

**让 helper 缓存的 slot 间距等于主卡的 `L.stride`，不要再做 256 对齐。**

三处改动：

1. **`NativeExpertLayout` 加 `stride` 字段**（`iq_kernels.hpp`，默认 0 = 没意见 = 旧行为）。
   q36 在填 layouts 表时把 `L->stride` 写进去。
2. **`ExpertCache::set_slot_align(int64_t)`**（`expert_cache.hpp/.cpp`，默认 256 = 旧行为）。
   `open_sized` 用这个对齐量算 slot 间距。`set_slot_align(0)` 表示「调用方给的尺寸就是最终间距」。
3. **`RemoteExperts::open`**：
   - `lay.native` 时 `cache_.set_slot_align(0)`，`sizes[i] = max(L.stride, blob_bytes)`。
   - 非 native 时 `set_slot_align(256)`，保持旧行为。
   - 显存预算 `needed` 用同一个数，否则预算不准。

**为什么不用「把尺寸向上对齐到 256」：** `1900544` 本来就是 256 的倍数，那次对齐什么都没做。
问题从来不是 blob 对齐，是**步长必须是 block 的整数倍**。256 和 block 大小没有整除关系。

**为什么不动主卡的 `L.stride`：** 它已经在做对的事（lcm 对齐）。改成额外对齐 256 会让
layer 34 的步长从 2041200 涨到 2096640（多 55440 字节/专家），40 层 × 256 专家多占约 566 MiB 显存，
而收益为零。

**旗标关闭时行为不变：** `set_slot_align` 默认 256；`stride` 字段默认 0；非 native 路径显式设 256。
q36 之外的调用方（`generate.cpp`、`verify.cpp`、`peer_experts.cpp`）一个都不用新字段。

## 3. 验收（机器恢复后执行）

同一条命令，按顺序：

1. **崩溃命令 RC=0**，stderr 没有 `illegal memory access`。
2. **`--bench-tg 0` 纯 prompt RC=0**，pp160 仍在 99.8 tok/s 附近（host 路由的收益不能丢）。
3. **layer 34 的 `delta` 变成 2041200**，且 `dn.expert_bytes / 210` 整除。
4. **`rows[0..64): nan=0 first_good=0`** 出现在每一层，包括 34。
5. **旗标关闭**：连续 3 次 `--tokens 32 --n 1` 都是 `gen: 220`。
6. **`--mtp 2` 三 helper 推荐配置**：tg128 仍是 52.5 附近，不能因为这次改动掉速。
7. **一 helper / 三 helper 各跑一次**，确认不是只有三 helper 才对。

## 4. 还要做的

- **清理所有临时诊断**：`q36_engine.cu` 里的 `DIAG`/`diag_toks`/逐层同步读，
  `q36_peer_prefill.cu` 里的 `STRATA_Q36_PREFILL_DIAG` 诊断块、`m.uniform` 实验、
  `dn.ids = m.d_ident` 实验（已恢复成 `m.d_map`）。
- **`peer_window_fits()` 的下限改动**：已证明不是本次根因。留着还是撤掉，等机器恢复后
  用「peers off 默认路径」和「`--max-context 512`」两个 case 单独判定。
- **R1 未修**：`peer_owner` 建一次、每层 `prepare` 重读 `slot_ptr`；运行中 rebalance 换槽后
  可能静默算错行。改法：flag 开启时禁止 rebalance 换槽，或每层对账。
- **`--max-context 512` 的 IMA** 是否同根因：未验证，不要合并断言。

## 5. 事故记录：P104 掉卡

2026-10-21 17:16 左右，PCI 02:00 的 P104-100 掉总线（`Xid 79: GPU has fallen off the bus`），
同时 P100 上有 `Xid 13: Out Of Range Address`（就是这次查的越界读）。

- 掉卡后 CUDA 只枚举到 3 张：cuda0=P100、cuda1/2=P104（**序号变了**，历史测量条件是 4 张）。
- 掉卡无法软件恢复：`/sys/bus/pci/devices/0000:02:00.0/reset` 超时，`nvidia-smi -i 3 -r`
  被残留进程挡住，`rmmod nvidia_uvm` 说 in use。
- 崩溃残留的 `strata-q36` 进程处于 `R` 态内核循环，`SIGKILL` 挂起但投不进去，只能重启机器。
- **重启后要重新确认 CUDA ordinal 与物理卡对应关系**，不能假设还是 cuda0=P100。

## 6. 两个必须记住的坑

**坑 1：诊断读数必须先同步。** 引擎的流是 `cudaStreamNonBlocking`（`q36_engine.cu:2180`），
helper 的流也是（`q36_peer_prefill.cu:359`）。非阻塞流与 legacy 默认流之间**没有隐式同步**。
直接 `cudaMemcpy`（走默认流）会和引擎并发，读到中间态。第一次读出的 `ewts=-inf` 就是这个假象。

**坑 2：崩溃的进程会占着显存不退。** IMA 之后进程卡在内核态，`SIGKILL` 都不管用。
下一轮跑就是 `cudaMalloc ... out of memory`。每次跑之前要看 `/proc/*/exe` 确认没有引擎。

## 7. 被推翻的假设（不要重走）

| 假设 | 怎么被推翻的 |
|---|---|
| `peer_window_fits()` 缺下限，解码步误入 peer 路径 | 加了 `n_ent > hybrid_t * top_k` 下限后，新 binary 仍以同样方式崩 |
| `eparts` 行数越界 | `peer_plan` 有 `row < 0 \|\| row >= n_ent` 检查，恒成立 |
| 主机的 token vector 被写坏 | 逐层哨兵全程没触发 |
| helper 的分组把第一个专家硬拆（`delta=0, n=1` + `delta=1900544, n=255`） | 改成等距单组后 NaN 仍在 |
| `dn.ids` 用 identity 能修 | 让部分行恢复有限值，但前 64 行仍有 54 行 NaN。identity 只是定位手段，不是修法 |
| helper 缓存的权重字节是错的 | 主卡和 helper 的专家 0 前 16 字节完全一致 |
