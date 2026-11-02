# 训练路线对比：反传 sft.py vs MeZO hf_ref.py

模型：Qwen3.8-Flash-Next 125B / IQ3_XXS（专家预打包成 NF4 pack，59.8 GiB / 24576 块 /
2.49 MiB 每块），单卡 4090 48GB + 宿主 64GB，LoRA r=16 α=32、12 类挂载点、
每步 1 条样本。三个变体各跑独立进程，外部采样进程树 RSS 与 nvidia-smi。

> 2026-10-29 修订：旧版只有短序列一档，且当时的 loader 与数据顺序都没有固定。
> 现在给了两档序列长度、同 seed 同批块，并把「92.63 GiB 从哪来」单列一节。
>
> **2026-10-29 第二次修订（commit `b9af996a` + `45ce80c9`）：前向 17.9 → 9.1 s/步、反传 84.3 → 24.3 s/步。**
> 瓶颈不在 MeZO 与反传的选择上，而是 transformers 的 QSA 稀疏索引器逐 query 跑 Python
> 循环，每步制造 2.4e4 次 DtoH 同步（§7）。§1 的整张表已替换、§7 重写；旧数值保留在
> 下方「上一版」的引用里以便对照。

> **2026-10-29 第三次修订（收尾）：§1 那张表是 6 步口径，被「首轮变慢」拖累了。**
> 14 步实测：前 7 步每步比稳态慢 4~14 s，第 8 步之后稳定在 **18~20 s/步 = 103~120 tok/s**。
> 一次性成本约 40 s，真跑会摊掉。成因未定位，但已排除四项（§7.7）。

## 0. 先钉口径：三个 tok/s 不是一回事

这几个数经常被混着比，先拆开（同一台机器、同一张 4090）。

**更正**：早先这里写过「192 tok/s 是 35B-A3B、160 是另一个模型」，那条是错的。
`strata-main-lp.log`（125B，888 个请求）的 decode 中位数就是 **172.8 tok/s**、p90 194.5、
max 243 —— 125B 自己就跑到 160~190。35B 的 `strata-qwen36-lora.log` 是 192.4，两个模型
在 decode 上几乎同值，**所以 decode tok/s 根本不能用来判别模型**。

| 数字 | 谁打的 | 模型 | 阶段 |
|---|---|---|---|
| **172.8 tok/s**（p50，888 请求；用户说的 160） | `strata-main-lp.log` | **125B IQ3_XXS** | **decode** |
| **1482 / 3824 / 4681**（p50 / p90 / max，n=888） | `strata-main-lp.log` | 同一个 125B pack | **prefill** |
| **982 tok/s** | `build-lp\strata.exe` 实测 2047 token | 同一个 125B pack | prefill，**短 prompt，固定开销未摊开** |
| **62~70 tok/s** | `bench_infer.py`（WSL2，修复前） | 同一个 125B pack | prefill（2047 token 一次前向） |
| **224.5 tok/s**（6 步中位 9.1 s；稳态 **465**，见 §1.1） | `bench_cmp.py` / `_mb_lora.py`（WSL2，修复后） | 同一个 125B pack | prefill |
| **2.0 tok/s**（506.9 ms/token） | `bench_decode.py`（WSL2） | 同一个 125B pack | **decode** |

同一模型、同一份 pack，两档各自比：

- **prefill**（训练是 prefill 形状，这是训练该看的数）：引擎 **1482（p50）~3824（p90）** → 训练
  稳态 **465**，差 **3.2~8.2×**（6 步口径 225 是保守下界，差 6.6~17×；修复前 16~40×）
- **decode**：引擎 172.8 → 训练 2.0，差 **86×**（decode 路径没有吃到这一轮的修复，见 §6）

注意 2047 token 那个 982 不能当基准：prefill 有固定启动开销，短 prompt 摊不开，日志里 888 个真实
请求的 p50 就是 1482、p90 3824、max 4681。拿 982 比会把差距缩小一半。

decode 那档差得更多，原因是结构而不是 LoRA：引擎 decode 走 `--expert-cache` 常驻 + MTP
推测解码（`drafts accepted 131 of 181`，一次前向出约 1.7 个 token），单 token 基本不搬专家；
训练路径每个 token 都要把该层未常驻的专家从 host 搬一遍（本例 VRAM 常驻 6579/24576 块，
约 73% 要现搬）。这个差距的来源是实现而不是平台（见 §6）。

### 配置口径（顺带查清：当前快捷方式已不在 125B 上）

`launcher/start_server.py:241` 的 `stock_config()` 按 **mtime 最新**选 `strata-*.json`。
各配置 mtime：

| 配置 | mtime | 指向 |
|---|---|---|
| `strata-qwen36-devlora.json` | **10-28 11:21（最新）** | `build-q6k` + **Qwen3.6-35B-A3B-UD-Q4_K_XL** + LoRA |
| `strata-qwen36-lora0.json` / `-lora.json` / `-train.json` | 10-28 07:14~10:17 | 同上 35B |
| `strata-swift-iq3_xxs.json` | 10-20 01:25 | `build-lp` + **125B IQ3_XXS** |
| `strata-main-lp.json` | 10-20 01:23 | `build-lp` + **125B IQ3_XXS** |

所以 10-28 做 dev-lora 那批实验时写下的 35B 配置把 mtime 顶到了最前，**现在双击
`Strata CN.lnk` 起的是 35B，不是 125B**。125B 的两份配置内容自 10-20 起没变
（`strata-main-lp.log` 的启动命令行与 `strata-main-lp.json` 逐参数一致）。
改回来只需让 125B 那份成为最新：`touch strata-main-lp.json`（配置被 `.gitignore:37`
的 `/strata-*.json` 忽略，git 查不到历史，只能靠 mtime/内容判断）。

## 1. 长序列 T=2048（真实训练形状）

`/root/pretrain_blocks.npy`：22269 块 × 2048 token（uint32），每步 1 块，`--seed 0`
固定洗牌，四个变体走**完全相同的块序列**。

| 变体 | s/step（中位 / best） | tok/s（中位 / best） | 显存·训练段 | **额外显存** | RSS·训练段 | 额外 RSS | 探针 Δ |
|---|---|---|---|---|---|---|---|
| 推理基线（no_grad） | **9.1 / 6.4** | **224.5 / 318.8** | 34.30 GiB | — | 46.02 GiB | — | — |
| **反传 + ckpt** | **24.3 / 20.1** | **84.4 / 102.0** | 38.20 GiB | **+3.90** | 46.86 GiB | +0.84 | +0.96 |
| **MeZO**（queries=1） | **28.3 / 15.0** | **72.4 / 137.0** | 34.70 GiB | **+0.40** | 46.09 GiB | +0.07 | −0.23 |
| 反传·不开 ckpt | 仍跑不动 | — | — | — | — | — | — |

配置：`STRATA_VRAM_PACK_GB=24`、`STRATA_PIN_GB=30`（见 §7.5），6 步同 seed 同批块。
单步抖动大（infer 6.4~19.2 s、mezo 15.0~64.0 s），中位与 best 都列出来是**必须的**：
两者在这个抖动下落在同一档里，见下面的配对讨论。

> 上一版同一张表（修复前，`STRATA_VRAM_PACK_GB=16`、43.8 GiB 全 pageable）：
> 推理 45.9 / 31.4 → 44.6 tok/s；反传+ckpt 84.3 / 75.1 → 24.3 tok/s；MeZO 62.8 / 52.0 → 32.6。
> **前向 5.0×、反传 3.5×、MeZO 2.2×。**

「跑不动」是 OOM，原因见 §4，不是物理下限：上一版实测 PyTorch 已分配 **92.63 GiB**
（48 GiB 卡，ctx 钉住 staging 缓冲）。现在 grad 下每专家独占一块显存（§5 第二轮第 4 条），
权重侧的下限变成「24,576 块全部活到反传结束」= **59.8 GiB** 再加激活 —— 仍然远超 48 GiB，
所以 ckpt 还是必须开，只是账从 92.63 降到 ~60+，不到 66 GiB 的清账方式仍是 §4/§7.6 ③。

配对可验证：两个变体确实是同一批块，同一步的 loss 对得上 ——
`sft 0.2596 / 0.1474 / 1.0052 / 1.4279 / 0.9622 / 0.8980` ↔
`mezo loss+ 0.2626 / 0.1460 / 1.0066 / 1.4393 / 0.9974 / 0.9176`。

**谁快：中位说反传赢（24.3 vs 28.3），best 说 MeZO 赢（15.0 vs 20.1），两者落在同一档
噪声里。** 这个抖动不是算法造成的（见 §7.5 的显存超额订阅），所以更适合拿 best 比：
20.1 / 15.0 = **1.34×**，正好等于「ckpt 下反传 2.7 个前向 vs MeZO 2 个前向」=
2.7 / 2 = 1.35× 的模型预测。也就是说，**反传之所以在长序列上落后，唯一的变量仍然是
ckpt**；把 ckpt 去掉（§7.6 ③）它才真正反超。

口径：显存是 `torch.cuda.max_memory_allocated()`，各脚本都在训练循环前
`reset_peak_memory_stats()`，所以「训练段」不含加载期临时缓冲。RSS 是进程树常驻集。
**中位数跳过第 1 步**（专家缓存还没热：推理档第 1 步 19.2 s、第 2 步 6.4 s），只统计
2~6 步；`best` 取全部步的最小值。因为只有 4~6 步、绝对值偏保守，**看同一步内的配对比值
更可靠**。

### 1.1 6 步太短：稳态是 18~20 s/步（103~120 tok/s）

同配置再跑 14 步（`sft.py --steps 14 --ckpt`，seed 0），前 7 步之后曲线就平了：

```
step  1   29.1s          step  8   18.3s
step  2   19.8s          step  9   18.0s
step  3   28.6s          step 10   18.9s
step  4   26.4s          step 11   19.1s
step  5   24.4s          step 12   19.6s
step  6   24.9s          step 13   19.8s
step  7   32.7s          step 14   17.1s
```

| 口径 | s/步 | tok/s |
|---|---|---|
| 14 步总计（含首轮一次性成本） | 22.6（316.7 s / 14） | **90.5** |
| 第 2~14 步中位 | 19.8 | 103.4 |
| **第 8~14 步中位（稳态）** | **18.9** | **108.3** |
| 最差一步（step 7） | 32.7 | 62.6 |

所以 §1 表里 24.3 s / 84.4 tok/s 是**保守下界**：6 步窗口正好落在首轮里。稳态 **108 tok/s**，
比 80 的目标高一档。同类现象在 `_mb_lora.py` 里更干净地复现了（同一块数据连跑三遍）：

```
第 1 遍 11.97 s → 第 2 遍 4.63 s → 第 3 遍 4.53 s   （H2D 三遍都是 28.85 GiB）
```

也就是说前向的稳态单价是 **4.4 s / 465 tok/s**（换块 8 个样本中位），而 18.9 / 4.4 = **4.3×**，
比 ckpt 模型预测的 2.7× 多出约 1.6 个前向当量（≈7 s）。**这 7 s 已经拆开了，见 §7.8**：
不是显存分配、也不是同步点，而是「反传本身 ≈ 2 个前向」这个实测值 —— 原模型里的 0.74×
是个错估计。

## 2. 短序列（内置身份数据，T≈17，12 步）——**修复前口径，未重测**

| 变体 | s/step（中位 / best） | 显存·训练段 | 额外显存 | RSS·训练段 | 探针 Δ |
|---|---|---|---|---|---|
| 推理基线 | 2.6 / 2.4 | 25.80 GiB | — | 45.42 GiB | — |
| **反传·不开 ckpt** | **5.4 / 4.3** | 35.50 GiB | **+9.70** | 46.08 GiB | **+22.65** |
| 反传 + ckpt | 9.1 / 7.9 | 26.40 GiB | +0.60 | 46.07 GiB | +22.45 |
| MeZO | 7.5 / 6.2 | 26.20 GiB | +0.40 | 45.69 GiB | −0.17 |

这套数字是 QSA 修复之前测的，**没有重测**，原因：T≈17 时 QSA 循环只有 17 次迭代
（2047 → 17），这轮修复对它几乎没有影响；而 T=2048 才是真实训练形状。§3 的
「排序翻转」结论不依赖这些绝对值。

## 3. 结论：排序随序列长度翻转，开关是 ckpt

| | 短序列（不 ckpt） | 长序列 T=2048 |
|---|---|---|
| 谁快 | **反传 5.4 vs MeZO 7.5（1.39×）**（修复前口径，§2） | **同一档噪声里**：中位 24.3 vs 28.3（反传赢）、best 20.1 vs 15.0（MeZO 赢），比值 1.34× ≈ 2.7/2，见 §1 |
| 谁省显存 | MeZO（+0.40 vs +9.70，省 9.3 GiB） | MeZO（+0.40 vs +3.90，省 3.5 GiB） |

原因是一个前向的单价：

- 短序列不需要 ckpt。反传 = 1 前向 + 反传（0.74× 前向，QDoRA/fused NF4 下要留的激活
  很少）≈ 1.74 个前向当量；MeZO = 2 个前向 + 小量扰动/更新。**反传更便宜**，于是赢。
- 长序列必须 ckpt（否则 OOM，§4）。ckpt 让反传 = 1 前向（存输入）+ 1 次**重算前向** +
  反传 ≈ 2.7 个前向当量，而 MeZO 还是 2 个。**MeZO 更便宜**，于是反超。

所以「MeZO 一定慢」和「反传一定快」都只在特定序列长度下成立。MeZO 的显存优势
（贴住推理基线，+0.4 GiB）在两档都成立。

QSA / PLE / 锁页这三项修的是**两条路线共用的那一次前向**（MeZO 走两次前向、反传走
一次前向 + 一次 ckpt 重算），所以它们同比例受益，**不改变上表的排序**：前向 45.9 → 9.1 s
之后，反传 84.3 → 24.3（3.5×）、MeZO 62.8 → 28.3（2.2×）—— 反传收益更大，是因为它多吃
一次前向（ckpt 重算），前向单价降了两次。稳态口径（§1.1）同理：前向 465 tok/s 时反传
108 tok/s，比值仍是那个 4.3×。

## 4. 92.63 GiB 从哪来 —— 这是设计缺陷，不是物理下限

`nf4_fused.py:122`：

```python
class FusedNF4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, absmax, code):
        ctx.save_for_backward(packed, absmax, code)      # ← 这里
```

在真正的 QLoRA 里这行几乎免费：量化权重是**常驻参数**，`save_for_backward` 只多存一个
引用。但在这里，专家块是从 pack 流式搬进一块**会被轮转复用的 staging 缓冲**里的；一旦
进了计算图，PyTorch 就必须让那一块一直活着。T=2048 时几乎每个专家都被命中一次：

```
24576 块 × 2.49 MiB = 59.8 GiB   ← 正好是整个专家 pack
实测增量 92.63 − 26.30 = 66.3 GiB  ← pack + 激活
```

**结论：反传图里把整套流式专家权重装了进去。** 这与「比 QLoRA 更省」的设计目标相反 ——
QLoRA 省显存靠的是「权重只以量化形式常驻，反传时现场 dequant」，而现在这个实现是
「权重搬一次就留一份」。

正确做法（下一步要做的）：**反传不要权重，也不要激活全存**。专家权重是冻结的、可按
(layer, expert) 寻址重取的，所以

1. 前向只在 ctx 里存 **专家 id + 路由**（几 KB），不存 packed/absmax；
2. 反传按 id **重新流取**该专家的块，跑 `_nf4_bwd_kernel` 出 dx。

这样权重永不进图，显存只剩激活；同时**不再需要 ckpt**（省掉那次重算前向）——
按更新后的前向单价算，15.8 s ≈ 1.74 个前向当量（2047/15.8 ≈ 130 tok/s），即**反超 MeZO**。

> 补充（`45ce80c9`）：现在 grad 下每专家是**独占**显存块，不再与 staging slot 别名，
> 于是「一块留一份」的下限就是 59.8 GiB 全在图上 —— 机制换了，结论没变，这条仍然是
> 去掉 ckpt 的唯一正路。

顺带说优化器：LoRA 可训练参数约 1.15e8，AdamW fp32 的 m+v ≈ **1.4 GB**，占那 66 GiB 的
约 2%。换成 8-bit 状态能省 ~1 GB —— 不是这次爆显存的原因，但等 66 GiB 消掉之后它就
占到可观比例了，届时值得一起做。

## 5. 两轮修掉的缺陷

### 第一轮

第 3 条对结论影响最大：它说明「同配置两次跑的 loss 差 0.06~1.44」不全是数据顺序造成的，
而是一半权重被读错。

1. **整块 H2D 与反传别名冲突（回归）**。`loader_qwen4.py` 的「整块一次 `copy_` + 显存侧
   四段 view」返回的是共享 staging 缓冲上的视图，slot 会轮转覆盖。纯前向无所谓，但反传时
   fused 函数刚把 packed/absmax 存进 ctx，缓冲就被下一组覆盖 →
   `variables needed for gradient computation has been modified by an inplace operation
   ... [ByteTensor [2560, 320]] is at version N`。旧的 per-expert `.to(dev)` 每块都是独立
   新张量，所以从没踩过。当时该快路径只在 `not torch.is_grad_enabled()` 下启用
   （第二轮第 4 条用「每专家独立显存块」把 grad 也接回这条路径）。
2. **数据顺序没固定**。`sft.py` 用的是无 seed 的 `randperm`，不同 run 命中的专家数不同，
   s/step 的差异里混着大量数据噪声。现在
   `sft.py` / `hf_ref.py` / `bench_infer.py` / `bench_cmp.py` 统一 `--seed`，
   四条变体走同一批块。
3. **显存那半 pack 被静默损坏（最重要，commit `df0b90ca`）**。这一条推翻了我上一版写在
   这里的结论「这条前向本身不可复现，loss 差 <0.1 nats 是噪声」——**那个漂移是 bug，不是
   模型噪声**。`_TieredPack._ensure` 用**单个** pinned stage 缓冲逐 chunk `preadv`，而
   `copy_(stage, non_blocking=True)`（源是 pinned ⇒ 真异步 DMA）还在读它，唯一的
   `synchronize()` 在整轮循环之后。RAM 那半是 preadv 直接进 `cpu_buf`（同步）所以完好，
   **只有驻显存那部分被覆写**：同一份文件两次加载对比，layer 0 的 512 个专家里
   **150 个字节不同**，而全走 RAM staging 的 layer 40 一个字节不差。
   判据链条（全部是指纹，不是推断）：

   | 检查 | 结果 |
   |---|---|
   | 同进程连续 3 次前向 | 逐位一致（digest SAME） |
   | 两个进程同一输入 | 三个用例 CE 全不同（0.5179/0.4094、0.0331/0.0254、1.1918/0.9934） |
   | 1069 个参数/缓冲跨进程 | **0 差异**（稠密权重没问题） |
   | 逐模块输出跨进程 | 到 `layers.0.mlp.gate`（路由 logits）为止**逐位一致**，第一处不同是 layer 0 第一个 `experts.act_fn`；命中专家数 116 vs 116 相同 |
   | 专家块指纹 | layer 0（512 全 resident）**150 个不同**，layer 40（512 全走 RAM）**0 个不同** |

   「同进程稳定、跨进程变」正是加载期状态决定的特征，不是非确定 kernel（那会在进程内也散）；
   `use_deterministic_algorithms(True, warn_only=True)` 也没有任何警告。
   修法：两个 pinned stage + 每 stage 一个 Event（覆写前等自己上一次 DMA），保留 preadv/DMA
   重叠。验证：前 512 块「文件字节 vs 显存」**MATCH**（修复前 MISMATCH）；5123 个张量指纹
   跨进程差异 **598 → 0**；跨进程 CE 两个用例现已相同。

   ⇒ **本文件里所有 loss/CE 都产自「前 13 层专家被损坏」的基座**（6579 块常驻 ≈ 12.8 层），
   计时不受影响（工作量没变），但 sft vs MeZO 的配对必须重测。

### 第二轮（commit `b9af996a` + `45ce80c9`，`_mb_split` 口径前向 17.9 → 5.0 s、`bench_cmp` infer 档 45.9 → 9.1 s，同配置训练 84.3 → 24.3 s；稳态 465 / 108 tok/s，见 §1.1）

1. **QSA 索引器逐 query 循环**（§7.3）—— 稠密路径 12.73 → 0.46 s，掩码逐位一致。
2. **PLE 表逐行反量化**（§7.4）—— 0.65 → 0.07 s。
3. **部分锁页**（§7.5）—— 43.8 GiB 里锁 30 GiB；pageable 源的同步阻塞（每步约 4 s CPU
   无法派发）消失。任何一片 pin 失败就减半重试，不假设上限（本机 32 GiB，Windows 侧不同）。
4. **grad 模式也走整块 H2D**。上一轮为了躲别名冲突把这条快路径 gate 在 `not is_grad_enabled()`，
   反传因此退回「线程池 pinned 拷贝 + 每专家 4 次 `.to()` + 4 次 view」。现在 grad 下**每个
   专家分配独立显存块**（多一次 `torch.empty`，缓存分配器约 2 µs），不再与轮转 slot 别名，
   于是同样只做 1 次整块 `copy_` + 4 次 view。代价按层计：一层 512 × 2.49 MiB ≈ 1.27 GiB，
   该层反传完即释放 —— 只有开 ckpt 才付得起（不开 ckpt 权重本来就全进图）。
   效果：sft+ckpt 39.4 → 24.3 s（中位）。数值上 step 1 的 loss 与 grad_norm 与改动前
   逐位相同（0.2596 / 0.2793），之后因反传累积顺序不同有 ~0.2% 漂移（与第一轮第 3 条
   同一个未收敛的非确定性源）。

## 6. 平台：WSL 是不是拖慢了？（结论：不是主因）

- 传输：WSL 里一次 43.8 GiB `pin_memory` 直接 `AcceleratorError`，退回全 pageable；微基准
  pageable 整块直连 **~20 GB/s**，引擎在 Windows 上的 probe 是 **24.1 GB/s** —— 差 1.2×。
  分片锁页可行（单次上限 32 GiB，现在锁 30 GiB，§7.5），pinned 块级 H2D 21.7 GiB/s。
- 计算：同一张卡、同一套驱动（通过 WSL 直通），kernel 层面没有平台差异。
- ⇒ 从 Windows 能拿回的最多是传输那 1.2×，而当前差距是 **3.2~8.2×**（稳态口径）。剩下这部分里，
  已经有一块被证明**纯粹是实现问题**（QSA 逐 query 循环，§7.3，吃掉了 74% 的墙钟），
  剩下的主因是 dequant→bf16→GEMM（每专家只分到 ~80 行，小 GEMM 跑不满 GPU）+ 24,576 次
  逐专家启动，引擎是量化内核直接算。

平台真正值得统一的是**内存预算**，不是速度。`.wslconfig` 里已经写死：

```
memory=56GB
# 代价：WSL 顶到 56GB 时 Windows 只剩 ~7GB
#       —— 桌面推理引擎（需 60GB）不能与训练同跑。
```

宿主只有 63.7 GB，而「WSL 训练 56 GB」与「Windows 引擎 60 GB」互斥。搬到 Windows 原生
能省掉 WSL 这一层，但代价是：Windows 侧 `.venv` 目前**没有 torch**（只有 numpy），要新装
cu13 wheel；`sft.py` 里 `resource.getrusage` / `/proc/self/statm` 是 Linux 专用，得改；
而且 44 GiB 的 pinned 内存在 Windows 上同样很可能失败。

## 7. 这一轮：前向 17.9 s → 9.1 s，钱花在哪

### 7.1 kernel tile 调参是死路（上一版结论，仍然成立）

单专家形状 `gu(1280×2560)` + `dn(2560×640)`，T=64（= T2048 × top_k 8 / 256 experts）：

| tile (BT,BO,BK) | ms | TFLOPS | 权重带宽 |
|---|---|---|---|
| **默认 (16,16,auto)** | **0.081** | 5.20 | 21.6 GB/s |
| (16,32,auto) | 0.086 | 4.90 | 20.3 |
| (64,32,64) | 0.084 | 4.97 | 20.6 |
| (16,64)/(32,32)/(64,64) | `OutOfResources` | — | — |
| 对照 dequant+cuBLAS | 0.212 | 1.98 | 解码 0.180 + GEMM 0.032 |

`dn` 的最优是 (32,32) 0.050 ms（默认 0.055，仅 +9%）。默认已经最优，加大 tile 直接爆
共享内存。5 TFLOPS 只有峰值的 3%，21 GB/s 远低于 L2 带宽 ⇒ **既不是算力也不是带宽瓶颈**，
是 decode 的 LUT gather（`tl.load(code_ptr + nib)`，每权重元素一次 gather）与循环开销。

顺带确认融合 kernel 本身是对的：对 `gu` 比 dequant+cuBLAS 快 2.6×，对 `dn` 快 6×。

### 7.2 先量「谁在等谁」：GPU 忙 5.3 s，墙钟 17.4 s

上一版在这里写「剩下 ~9.6 s 未测量」。补上之后结论完全不同：

```
T=2048 一次前向：CUDA 总时间 5.30 s / 墙钟 17.4 s ⇒ GPU 空转 74%
               2,410,575 个事件，其中 cudaLaunchKernel 984,312 次
               24,661 次 Memcpy DtoH + 24,763 次 cudaStreamSynchronize
```

按 op 聚合的 CUDA 时间都很小（最大 `_nf4_fwd_kernel` 1.29 s、H2D 合计 2.20 s），
说明成本在「启动次数 × 每次开销」上，不在某个昂贵的 kernel 上。而 2.4e4 次 DtoH 同步
是最刺眼的信号：**每 2,047 次同步对应一层**（2,047 = T-1）。

### 7.3 真凶：QSA 稀疏索引器的逐 query Python 循环

`transformers/models/qwen4_exp/modeling_qwen4_exp.py:729`：

```python
for batch_idx in range(batch_size):
    for query_idx in range(seq_length):
        local_visible_indices = torch.nonzero(visible_token_indices[b, 0, q], ...)
        num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio   # DtoH
        ...
        pooled_keys = self.k_layernorm(key_groups.view(...).float().mean(dim=1).to(dtype))
        block_key_states = apply_rotary_pos_emb(pooled_keys.unsqueeze(1), cos[...], sin[...])
        scores = torch.relu(q[b, q].float() @ block_key_states.float().T).sum(-1) / sqrt(D)
        selected_block_indices = scores.topk(...).indices
        ...
        selected_token_indices[b, query_idx, : selected_tokens.numel()] = selected_tokens   # DtoH
```

三个问题叠在一起：

1. `pooled_keys / k_layernorm / rope` 只依赖 **block**，与 query 无关，却每个 query 重算
   一遍 —— T=2048 就是 2,047 倍冗余；
2. `.shape[-1]` 与 `.numel()` 都是 **DtoH 同步**，每次同步都会排空前面的 kernel；
3. 12 个 QSA 层 × 2,047 query = 24,564 次同步/步，正好等于实测的 24,661 / 24,763。

纯 causal 掩码下 `visible(q) = [0..q]`，于是 `block_token_indices` 对所有 query 相同：
block 池化一次、一次 `einsum` 出 `(S, n_block)` 分数、一次 `topk`、gather 拼掩码。

校验（`STRATA_QSA_CHECK=1`，与逐 query 原实现逐元素对比）：**掩码差异 0/4,190,209**，
同一块数据 loss 前后都是 2.0236034393。只在「无 past_key_values、kv 长度 = 序列长、
掩码恰为下三角全 True」时启用，其它情况退回原实现。

### 7.4 拆开两条路径（stub 掉专家 forward）

`_mb_split.py` 把 `StreamingNF4Experts.forward` 换成 `zeros_like`，同一份权重跑两遍：

| | 墙钟 | PLE |
|---|---|---|
| A 完整前向 | **17.89 s** | 0.65 s |
| B 专家整段 stub | **12.73 s** | 0.58 s |
| A − B 专家路径 | **5.17 s（29%）** | |
| B 里非 PLE 的稠密/注意力 | 12.15 s（71%） | |

也就是说，上一版一直盯着专家路径（占了 29%），真正的大头是稠密路径的 12.15 s —— 全在
QSA 那个循环里。修完之后同一脚本变成 **专家 4.58 s（91%）/ 稠密 0.46 s（9%）**。

顺带另一处：`MmapNgramEmbedding._gather_rows` 对每个 unique 行做一次 mmap 切片 +
`gguf.dequantize`（T=2048 是 2,047 行，每行 **319 µs**）。这段是纯 CPU，CUDA profiler 里
完全看不见。改成「mmap 上取 `(total_rows, row_bytes)` 零拷贝视图 → numpy fancy-index 一次
搬进连续缓冲 → 只调一次 dequantize」，0.65 s → **0.07 s**，数值路径不变。

### 7.5 传输：能锁 32 GiB，锁上之后每块 112 µs 而不是 222 µs

`_mb_pin.py`（本机实测）：

| 源 | 2.49 MiB/块 | 239 MiB/块 |
|---|---|---|
| pageable, blocking | 10.96 GiB/s，222 µs/块 | 14.57 GiB/s |
| pageable, `non_blocking=True` | 12.57 GiB/s，193 µs/块 | — |
| **pinned, `non_blocking=True`** | **21.72 GiB/s，112 µs/块** | 21.60 GiB/s |

关键点：**pageable 源的 `copy_(non_blocking=True)` 并不异步** —— 它阻塞调用线程直到搬完。
43.8 GiB / 17,997 块 ⇒ 每步约 4 s 里 CPU 完全无法派发后续 kernel，GPU 同时空转。
单次可 pin 上限实测 **32 GiB**（33 GiB 直接 `AcceleratorError`），所以 `_TieredPack` 改成
「pinned 前缀 + pageable 余量」两段，失败按减半重试。

显存那半也要给够：`STRATA_VRAM_PACK_GB=30` 时 `peak_reserved` 到过 **50.1 GiB**，超过
卡的 47.99 GiB —— WSL 下 CUDA 会向宿主内存超额订阅，一旦触发就是 PCIe 换页，单步能从
6.4 s 抖到 19.2 s。最终取 **24 GiB 常驻 + 30 GiB 锁页**：allocated 34.3 GiB（infer）/
38.2 GiB（ckpt 反传），RAM 侧 30 + 6 + ~10 = 46/54 GiB。

### 7.6 剩余的账与下一档

前向的**稳态**单价是 4.4 s（§1.1；§1 表里的 9.1 s 是中位口径被首轮拖累，见 §7.7）。
`_mb_split.py` 那套（no_grad、无 LoRA）测到 5.04 s，其中专家 4.58 s、稠密 0.46 s ——
修完之后**稠密路径已经不是问题**。上一版这里猜「比 5.04 多出来的 4 s 来自 LoRA / train
模式」，§7.7 已证伪（LoRA +0.12 s、train 模式 +0.02 s）：那 4 s 就是首轮变慢。

专家那 4.58 s 里：融合 GEMM 约 1.3 s、H2D 约 1.6 s、逐专家 gather/scatter/elementwise
约 1.2 s，其余是每块传输与 kernel 之间的空隙。单项都不大，共同点是 **24,576 次逐专家
迭代**（配 7~8 个 kernel）。下一档要动的是：

| # | 动作 | 依据 | 预期 |
|---|---|---|---|
| ① | 把 silu*up 融进 gu GEMM、把 top-k 权重融进 dn GEMM 的 epilogue | 每专家 7~8 个 kernel，机器码级只省 2~3 个 | 专家段 -10~20% |
| ② | 按组做 grouped GEMM（一组 32 个专家的 block 在 staging 里本就连续） | 把 24,576 次启动降到 ~800 次 | 专家段数量级下降 |
| ③ | ctx 只存专家 id、backward 按 id 重流 | 现在 grad 下整层独占 1.27 GiB 显存，仍必须开 ckpt | 去掉 ckpt ⇒ 省掉那次重算前向（实测 +4.4 s/步） |
| ④ | B>1 | 实测 B=8 是 B=1 的 1.32× | 白拿 +32% |
| ⑤ | ~~拆 18.9 s 与 4.3× 前向之间的差额~~ | **已拆完，见 §7.8** | 结论：反传 ≈ 2× 前向，3.2~4.2× 是结构性的 |

③ 按稳态算：18.9 − 4.4（省掉的重算前向）≈ **14.5 s/步 ≈ 141 tok/s**，顺带省 3.9 GiB。
⑤ 的结论（§7.8）把 ① 和 ② 从「锦上添花」变成了**唯一的大杠杆**：每一步的 op 数会同时
乘进前向和反传（反传按倍率放大），所以省一个 op 等于省三份时间。

### 7.7 「首轮变慢」：一次性约 40 s，四项已排除，机制未定位

现象（`_mb_lora.py`，同一进程，同一块数据连跑三遍）：

```
第 1 遍 11.97 s   H2D 28.85 GiB
第 2 遍  4.63 s   H2D 28.85 GiB
第 3 遍  4.53 s   H2D 28.85 GiB
```

**传输字节数三遍完全相同，时间差 2.6 倍** ⇒ 不是数据、不是带宽、不是专家命中数。
在整条训练链上，它表现为「前 7 步每步慢 4~14 s，第 8 步起突然稳定」，两个独立进程都
精确停在第 8 步（累计读过约 2 倍 pack 的量），像是某个「读满一遍才热」的缓存。

排除掉的四项（都是实测，不是推断）：

| 嫌疑 | 做法 | 结果 |
|---|---|---|
| LoRA 挂载 | 同一进程 eval/train x 有无 LoRA 四组 | LoRA **+0.12 s**、train 模式 **+0.02 s** |
| 显存分配器碎片 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 逐步曲线**逐点不变** |
| host 页首次建立映射 | 启动时把 35.8 GiB 全部 DMA 一遍（2.2 s） | 前 7 步**照旧慢** |
| 显存容量压力 | pack 从 24 GiB 降到 20 GiB | 慢步模式**一模一样**，只是整体慢 10% |

也排除了宿主内存回收：训练期采 `/proc/vmstat`，`pgscan_direct` 恒为 0、`pgmajfault` 在
慢步里没有尖峰，`MemAvailable` 稳定在 10.6 GiB（紧张但不触发直接回收）。

还没做的：抓一个慢步 vs 快步的 CUDA profiler 对比（GPU 忙/墙钟缺口、`cudaMalloc` 次数、
Memcpy 次数）。**在那之前不要再靠换配置试** —— 这四项都是换配置，全无效。

实用结论：这是**一次性启动成本**，量级 40 s，真跑（几百步）会被摊平；基准脚本因为只跑
4~6 步，才会被它主导。所以本文件的长序列数字要按「§1.1 稳态」那一栏读。

### 7.8 反传步的账拆开了：3.2× 前向是结构性的，不是 bug

新增 `_mb_train.py`（前向 / 整步各自一对 CUDA event，反传由差值反推）。T=2048，同配置：

| 成分 | 改前 | 改后（本轮两处） |
|---|---|---|
| 前向（no_grad，ckpt 的第一次） | 4.46 s | 4.36 s |
| ckpt 段（重算 + 反传） | 19.26 s | 18.23 s |
| 整步墙钟 | 23.72 s | **22.59 s**（91 tok/s） |
| 整步 / 前向 | 4.31× | 4.18× |

CUDA-only profiler 给一步的形状（改前）：`_nf4_fwd_kernel` 2.55 s / 75,516 次、
HtoD 2.17 s(pinned) + 0.49 s(pageable) / 22,168 次、`_nf4_bwd_kernel` 1.84 s / 37,758 次，
合计 CUDA 9.75 s，而这一步约 19 s ⇒ **GPU 忙一半，另一半在等 CPU 派发**；事件数 131 万。
75,516 ≈ 24,576 个专家 × 2 个 GEMM × 1.54（部分层命中不满），也就是**重算确实把两个 GEMM
又跑了一遍**。

那 22.6 s 到底怎么解释？用 `_mb_expert_grad.py` 把「每专家循环」单独拿出来量（同形状、
权重换成 bf16 + cuBLAS，只量调度与 autograd）：

| 模式 | µs/expert | 相对 no_grad |
|---|---|---|
| no_grad | 98.9 | 1.00× |
| 建图（前向带 grad） | 112.8 | **1.18×** |
| 建图 + 反传 | 330.0 | **3.20×** |

⇒ **反传本身就是 2 个前向，不是 0.74 个。** 建图只贵 18%，所以问题不在显存分配、不在
同步点、也不在「重算前向带着 grad 跑」。ckpt 只是在这之上再加一次前向（1.18×）：
3.20 + 1.18 = 4.38×，与实测 4.18× 对得上 —— **这笔账闭合了**。

原因很朴素：这一层是「24,576 个微小算子」而不是「几十个大 kernel」，反传给前向的每个算子
各加 1~2 个反向节点（gather/scatter/silu/mul 都有 backward），dispatch 成本翻倍。所以 op 数
会**同时**乘进前向和反传。

本轮实际做掉的两处（都在 `loader_qwen4.py`）：

1. **grad 下整层一次分配**：原来每专家 `torch.empty(2.49 MiB)`，一层 512 次、一步 24,576 次
   （重算再来一遍）。改成整层 1.27 GiB 一次申请、逐块切片 —— 分配器调用 24,576 → 48 次/步。
   切片互不重叠，仍不会与槽位轮转别名（那才是当初 version counter 报错的成因）。
2. **去掉 grad + 整块路径上无效的槽位同步**：那条路径不写 staging 槽位，`slot_sync` 只剩
   「等 GPU 跑完上一次 record」这个副作用 —— 每 32 个专家一次**阻塞同步**（一层 16 次、
   一步 768 次），把 CPU 派发和 GPU 执行硬串起来。现在 `slot_in_use` 只在真会覆写槽位时才等。

合计 23.72 → 22.59 s（-5%），**数值不变**：`_mb_train` 的 5 步 loss 轨迹
（1.97 / 1.88 / 1.79 / 1.70 / 1.60）与改动前逐位可比。

⇒ 剩下的唯一大杠杆是 §7.6 的 ① + ②：把每专家的 8 个算子压到 2~3 个（融合 epilogue、
分组 GEMM）。按上表，每省一个算子同时省掉它在反传里的 1~2 个节点和一次 dispatch ——
实测倍率 **3.2×**，不是 1×。

### 7.9 显存 ↔ 速度的取舍线（同一套代码，只改 `STRATA_VRAM_PACK_GB`）

`_mb_train.py`，T=2048，4 步稳态，`STRATA_PIN_GB=30` 不变：

| pack 常驻显存 | torch 峰值 | 前向 | 整步墙钟 | tok/s | 相对 24 GiB |
|---|---|---|---|---|---|
| 24 GiB | 38.2 GiB | 4.36 s | **22.59 s** | **91** | — |
| 20 GiB | 34.2 GiB | 5.12 s | 23.59 s | 87 | -4.4% |
| 16 GiB | 30.2 GiB | 5.52 s | 25.98 s | 79 | -13% |

读法是「每少 4 GiB 常驻，前向多 0.6~0.8 s」—— 因为少驻留一块就得多搬一次 H2D，而搬运
是这条路径里唯一按字节线性计时的事。所以：

- 想给别的东西腾显存，**能腾到 20 GiB 还不掉出 80 tok/s**；16 GiB 就掉到 79 了。
- 反过来想更快，把 pack 常驻调**大**并不可行：单次 `pin_memory` 上限 32 GiB、
  峰值一旦越过 47.99 GiB 就会触发 WSL 向宿主内存换页（§7.5），40 GiB 那条配置实测抖动到
  无法计时。24 GiB 是这台机器上的拐点。

## 8. 复现

```bash
# 长序列（同 seed 同批块）；24 GiB 常驻 + 30 GiB 锁页是本机 48G 卡 + 54G WSL 的甜点
export STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b \
    --pretrain-data /root/pretrain_blocks.npy --seqlen 2048 \
    --steps 6 --seed 0 --only infer,sft+ckpt,mezo --out /root/cmp.json

# 短序列
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b --steps 12 --seed 0 --out /root/cmp_short.json
```

`--only infer,sft,sft+ckpt,mezo` 可只跑其中几个。诊断脚本：

| 脚本 | 回答什么 |
|---|---|
| `_mb_pin.py` | 单次能 pin 多大、pageable/pinned 的块级 H2D 带宽 |
| `_mb_phase.py` | 专家段事件计时（不在层内同步，保住跨层流水） |
| `_mb_split.py` | stub 掉专家 forward，把专家路径与稠密路径的墙钟拆开 |
| `_prof_step.py` | CUDA-only profiler：op 级时间 + 启动次数 + GPU 忙/墙钟缺口 |
| `_mb_lora.py` | eval/train x 有无 LoRA 四组；`--blocks --passes 3` 复现「首轮变慢」（§7.7） |
| `_mb_train.py` | 一步训练拆成「前向 / ckpt 段（重算+反传）」两段事件，外加 CUDA-only profiler 步（§7.8） |
| `_mb_expert_grad.py` | 同形状的每专家循环只量调度与 autograd：no_grad / 建图 / 建图+反传（§7.8） |
| `_det_*.py` | 指纹法定位数值差异（见 §5 第一轮第 3 条） |

## 9. 怎么跑（默认入口）

`train.py` 是唯一的默认入口：模型默认取**桌面快捷方式会起的那份**（`launcher/launch-choice.json`
里钉住的 `strata-*.json`，见 §10），数据默认「改变身份」，算法默认反传 SFT。

```bash
python3 train.py                                  # 身份数据 + 反传 + 钉住的模型
python3 train.py --dry-run                        # 只打印最终命令，不加载模型
python3 train.py --list                           # 算法 / 目标函数 / 数据源
python3 train.py --algo mezo --steps 20           # 同数据，零阶（显存贴住推理基线）
python3 train.py --data jsonl:/root/my.jsonl      # 换数据只换这一行
```

Windows 上双击 `Strata Train (WSL2)`（= `START-TRAIN.bat`）：训练只能在 WSL2 跑
（Triton / fla / `kernels` 都没有 Windows 构建），推理则在 Windows（引擎是 Windows 二进制）。
两边用同一份权重与同一套路径，唯一交界是导出的 LoRA GGUF。

**「只靠前向做 RL」跑通过的证据**（T=2048 之外的小样本，只为验链路）：
`--algo mezo --objective grpo --rollouts roll_smoke.jsonl --steps 2` →

```
rollouts: 4 条 / 2 组，有效组 2（std=0 的组优势全 0，不会动参数）；old logprob 来自 local，重算 4 条
step 1: obj+ 0.0548  obj- 0.0378  proj_grad 8.4674e+00  ratio 1.0506  adv0 0.00  peak 33.8 GiB  3.6s
step 2: obj+ 0.0800  obj- 0.0434  proj_grad 1.8303e+01  ratio 1.1271  adv0 0.00  peak 33.8 GiB  3.2s
saved /root/lora_grpo_smoke.pt: 96 tensors, non-zero 96
```

`proj_grad` 非零、96 个 LoRA 张量全部非零 ⇒ 零阶更新确实走到了参数上；`ratio` 从 1.05
漂到 1.13 是两次更新之后的应有表现（`adv0 = 0` 表示没有优势为 0 的死样本）。
显存 33.8 GiB，比反传那档低 4.4 GiB —— 全程 no_grad。

## 10. 模型默认值从哪来，以及它为什么以前会错

`launcher/start_server.py:stock_config()` 原来按 **mtime 取最新的 `strata-*.json`**。为任何
别的模型写一次配置（35B 适配器开发、另一个量化、benchmark family）都会让那份成为新的
默认值 —— 双击 `Strata CN.lnk` 起的是 35B，而日志里才看得出来。

现在改成**显式钉住**：`launcher/launch-choice.json` 的 `"config"` 指向 `strata-main-lp.json`
（125B IQ3_XXS），`--use <file>` 改它并记住；mtime 只在没有钉子时兜底，并在启动时说明。
启动日志多两行，把「用哪份配置、哪个权重文件」直接打出来：

```
[start-server] 已钉住的模型：strata-main-lp.json（记住了 —— 要换就用 --use <文件>）
[start-server] 权重：D:\...\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
```

`train.py` 复用同一份逻辑（直接 import `start_server.py` 的 `stock_config()`/`read_cfg()`），
所以「训练用的模型」和「正在跑的模型」不会分叉。

## 11. 剩下的三条杠杆（都**没**实施，附判据与预期）

稳态基线：**22.59 s/步 = 90.7 tok/s**（T=2048、batch=1；前向 4.36 s + ckpt 段 18.23 s）。
每步 48 层 × 512 专家 = **24,576 次专家访问**，单次约 330 µs（微基准：no_grad 98.9 µs、
建图 112.8 µs、建图+反传 330.0 µs）。折算 **≈3 TFLOPS**，即 4090 bf16 峰值的 **2%**
⇒ 当前不是算力瓶颈，是「小 tile + 逐次 launch」的填充率瓶颈。这就是下面三条的依据。

| # | 杠杆 | 改什么 | 预期 | 工作量 | 风险 |
|---|---|---|---|---|---|
| ① | 融合 epilogue | `_nf4_fwd_kernel` 里把 `silu(gate)*up` 直接写进 gu GEMM 的 epilogue；dn GEMM 的 epilogue 乘 `grp` 后直接累加 | 每专家 9 个 dispatch op → 6~7 个 ⇒ **18.2~20.8 s（98~112 tok/s）** | 中（两个 kernel，逐值需回归） | 低（fp32 epilogue 比现在的 bf16 中间结果更准） |
| ② | **分组 GEMM** | 一层 512 个专家合成一次 segmented GEMM（按 `bounds` 切段），反传同样分组 | 24,576 次 launch → 96 次；tile 填充率从「32×1280 单专家 ≈ 10 CTA」变成 5,120 CTA 并发 ⇒ **预计 8~12 s/步（170~250 tok/s）** | 高（Triton grouped GEMM + 分段 reduction + backward） | 中（累加顺序变，必须给 CE 门禁） |
| ③ | 去掉 ckpt | ctx 只存专家 id，backward 按 id 重流权重；权重永不进图 | 省掉整段重算 ⇒ **18.2 s/步（113 tok/s）** | 中高（自定义 autograd，需与 ckpt 路径逐值对照） | 中（配错会静默算错梯度） |

**②是唯一改变算术强度的一条**，①②③里只有它能把「2% 峰值」往上抬；①③只是把
「2% 峰值 × 更少的 op / 更少的遍数」乘小。三者可叠加，但 ② 与 ① 在同一个 kernel 里，
先做 ② 会把 ① 重写一遍，所以顺序建议 **③ → ① → ②**（③ 与 kernel 无关，先拿 +24% 的无风险收益）。

**为什么先不动 ②**：它要重写前向与反传两个 Triton kernel 并重排 `index_add_` 的累加顺序，
必须有「同 seed 同数据下 loss/CE 相对 ckpt 路径的偏差」这道门；而这台机器每验证一次要独占
GPU ~40 分钟（模型加载 ~100 s + 40 步）。开工前需要一次明确的长跑窗口。

**显存侧还没有动的**：24/20/16 GiB 三档只压了显存（见 §7.9）。RAM 侧（pinned 30 GiB +
pageable + RSS 46 GiB）尚未做过任何取舍实验 —— 这一项与速度无冲突，是先能拿的。

## 12. 显存与内存：能压到多小（2026-10-30，**标准入口** `train.py`）

§7.9 那张表是诊断脚本（`_mb_train.py`）的口径。换成真实入口后多出一笔开销，而它正好
是「能压到多小」的决定项。

### 12.1 硬约束：pack 必须整个待在「显存 + RAM」

pack 59.77 GiB（24,576 × 2.49 MiB），当前实现零读盘（§7.5 论证过读盘的代价：
9~18 tok/s、GPU util 1%）。WSL 的 RAM 上限是 58 GiB（`.wslconfig memory=60GB`），于是

    显存 ≥ pack − (RAM 上限 − 进程其它开销)

24 GiB 档 RSS 53.8 GiB ⇒ 其它开销约 18.2 GiB。按这个数算下界该是 20 GiB —— **实测不是**：
16 GiB 档跑通了。差别在那 18.2 GiB 里最大的一块：**两次加载一共读过的 ~15 GiB GGUF 映射页**，
它们是干净的 file-backed 页，**可以还**。

### 12.2 把映射页还给内核（本轮唯一的代码改动）

`GgufShard.drop()` / `_release_mmap_pages()`：每把一个张量搬上显存就 `madvise(MADV_DONTNEED)`
它占的那段页，全部加载完再整体还一次。唯一还会读它们的是 PLE（`MmapNgramEmbedding` 本来
就按行读、表不驻留），代价是每步几 MB。

不做的后果不是「慢」而是「跑不动」：16 GiB 档在基座加载阶段 `available` 归零、10 分钟
走不出加载（实测日志：`tiered pack ... loaded 300s` 之后再无输出）。做完 300~344 s 加载完。

### 12.3 实测表（T=2048，3 步）

| pack 常驻显存 | 训练段峰值 `max_memory_allocated` | RSS 峰值 | 步时（末步） | tok/s | 结论 |
|---|---|---|---|---|---|
| 24 GiB | 34.5 GiB | 53.8 GiB | **20.5 s** | **99.9** | 基准 |
| 16 GiB | 26.5 GiB | **57.3 GiB** | 29.0 s | 70.6 | 省 8 GiB 显存，慢 41% |
| 12 GiB | — | 需 ~61 GiB | 换页卡死 | — | **不可行** |

（12 / 8 / 6 / 4 四档是同一批实验里主动停掉的：12 GiB 档 10 分钟没走出加载、`available`
归零；8/6/4 的 RAM 需求分别是 53.8/55.8/57.8 GiB + 其它开销，只会更糟。）

读法：**显存每省 8 GiB，速度掉 41%，RSS 同时涨到贴着 58 GiB 的顶**。16 GiB 是这个实现的
下界，不是「再压一点会慢」—— 再压就是 RAM 装不下 pack。想要更小只有放弃零读盘、按层读盘
（§7.5：9~18 tok/s，慢一个数量级）。

16 GiB 档的步时波动比 24 GiB 大（44.3 / 21.1 / 29.0 vs 22.1 / 15.7 / 20.5）：RAM 贴顶时
内核回收页的抖动会落到步上。稳态（末步）取 29.0 s。

### 12.4 怎么用

```bash
python3 train.py --algo mezo                 # 默认 16 GiB 常驻（$STRATA_VRAM_PACK_GB）
python3 train.py --vram 24                   # 更快，多占 8 GiB 显存
python3 train.py --vram 16 --pin 20          # 锁页也调小：RSS 再松几 GiB
```

`--vram` / `--pin` 只是把两个环境变量包成参数（`train.py` 同进程 import，设 env 即生效），
表头会回显实际档位，日志自证。

## 13. 标准 GRPO：进程内采样 + 组内 batch（2026-10-30）

**问题**：GRPO 的标准形态是 on-policy —— 每一轮用**当前策略**采一组轨迹，组内归一化优势，
再更新一步。原来的实现是离线回放：`--rollouts` 一份 jsonl，`--steps 20` 就把同一批组循环
用 10 遍（`groups[(step-1) % len(groups)]`）。`old` 一旦固定，第 2 步起 `ratio` 必然偏离 1
—— 那已经不是 GRPO，是「带重要性权重的离线策略梯度」。

**为什么不能拉引擎来采样**：引擎要 ~40 GiB 常驻（expert cache 39.97 GiB，§7.5），
训练侧 34.5 GiB，48 GiB 卡装不下两个。轮转的话（起引擎采一批 → 关引擎 → 训）
每轮多付 100 s 引擎加载 + 300 s 训练加载，摊到 20 步上是 20 s/步的固定成本。

**做法**：让学生自己采。MeZO 只需要**前向**，采样也是**前向** —— 同一条 no_grad 路径，
不多出计算图。但显存要说准：**不是「不涨」，是「涨得可忽略」**——K 条同时在飞要多付
K 份 KV cache + 每步 K 行 vocab logits，这笔账随 K × (prompt + 回答长度) 线性走；
实测 K=4×16 只有几十 MB，峰值停在 26.2 GiB 看不见。把组开大、把回答拉长，它就会显形。
三个让它可以接受的设计：

1. **组内 batch 采样**（`sample_group`）：K 条共享同一个 prompt 前缀，一次 decode 处理 K 条，
   每 token 的专家搬运摊到 K 条上。
2. **`old` 就地记下**：采样时的 logits 顺手算 logprob，`ratio` 起点严格 = 1；
   既不用起引擎，也不用「再跑一遍重算 old」。
3. **组内 batch 前向**（`grpo_loss(batched=True)`）：K 条拼一个 batch，只 pad 回答段
   （同组 prompt 等长，不需要左 padding），专家只搬一遍。

### 13.1 实测（125B，T=2048 的 pack，`--vram 16`）

```bash
python3 train.py --algo mezo --objective grpo \
    --prompts grpo_prompts.jsonl --group 4 --max-new 16 --steps 3 --lr 1e-6 --vram 16
```

| step | sample | 整步 | 峰值显存 | ratio | uniq | rew | proj_grad |
|---|---|---|---|---|---|---|---|
| 1 | 30.6 s | 48.9 s | 26.2 GiB | 0.9955 | 4/4 | 0.50 | −1.49 |
| 2 | 10.7 s | 21.4 s | 26.2 GiB | 0.9974 | 4/4 | 0.30 | −3.63 |
| 3 | 9.7 s | 20.3 s | 26.2 GiB | 1.0100 | 3/4 | 0.00 | **0** |

读法：

- **`ratio` 从 1 附近起步**（0.9955 / 0.9974 / 1.0100）—— 这才是 on-policy 的样子。
  离线回放第 2 步起就已经飘到 1.1 以上。
- **第 1 步采样 30.6 s、之后 10 s**：专家缓存没热时 decode 慢（和反传路径首步慢是同一个原因）。
  采样占整步的一半，训 3 步以上就摊薄了。
- **第 3 步 `rew 0.00` ⇒ 全组奖励为 0 ⇒ `adv0 1.00` ⇒ `proj_grad 0`**：那一步什么都没学。
  这是 GRPO 的冷启动（组内无方差），**日志必须让它可见**，否则「跑完了」看着像「学会了」。
  正确顺序是先用 SFT/MeZO-ce 把行为抬起来，再上 GRPO（本目录 §3 的老结论不变）。
- 96 个 LoRA 张量全非零，峰值 **26.2 GiB** —— 比反传 SFT 的 38.2 GiB 少 12 GiB。

### 13.2 代价与边界

- **组内 K 条轨迹的梯度都得进账，显存怎么付有两条路**（这是 GRPO 与 SFT 的本质区别：
  SFT 一条轨迹一个 loss，GRPO 的优势是组内归一化算出来的，K 条整条轨迹都要反传）。
  `batched=True` 把 K 条拼一个 batch 一次反传：峰值显存高（K×T 激活图），专家只搬 1 遍；
  `batched=False` 逐条反传 + **LoRA 梯度累积**：峰值显存低（每次只挂 1 条的激活图），
  专家搬 K 遍。所以**梯度累积不是「并行的额外显存来源」，恰恰是省峰值显存的那条路**，
  它的代价是专家搬运 ×K（§13.3 的 2.25× 就是这么来的）。LoRA 梯度缓冲本身只有
  96 个张量 ≈ 15.7 MB，从来不是压力所在；并行真正多占的是「K 条同时在飞」——
  采样期 K 份 KV，训练期 batched 路线的 K×T 激活图。
- 采样是**学生自己**做的，所以速度受训练路径的 decode 限制（每 token 要把该层专家搬一遍）。
  引擎有 expert-cache + MTP，同一模型能到 172.8 tok/s；训练路径只有 ~2 tok/s（4 条 batch 后）。
  **这就是「少显存」的价钱**：拿引擎的速度换掉 40 GiB 的常驻。
- `--resample-every N`：N>1 用同一批轨迹多训几步（`ratio` 会开始飘），是显存与速度之外的
  第三个旋钮；默认 1，即标准 on-policy。
- `--batch-group off`：逐条前向，仅用于对照（正确性由 `test_train.Sampling` 与
  `GRPOLoss.test_batched_matches_serial_when_lengths_differ` 守着）。

### 13.3 组内 batch 值多少（K=4，4 条 × 16 token，同一进程内各测 3 次取均值）

| `grpo_loss` | 耗时 | 相对 |
|---|---|---|
| `batched=True` | **1.89 s** | 1.00× |
| `batched=False`（逐条） | 4.25 s | 2.25× |

**2.25× 而不是 4×**，因为 batch 后的单次前向确实比单条慢（pad 段 + 更大的 tile），
但省掉的 3 次专家搬运是主导项。这个倍数随 K 走：K=8 时省 7 次，K=4 时省 3 次。

同一次加载里还量到：`sample_group` 4 条 × 16 token = **29.5 s**（≈2.2 tok/s 墙钟，
4 条 batch 摊过），`uniq 4/4`（真有方差，不是一组同文）。

## 14. 并行：对接引擎的批槽，不是自己实现一套

先纠正一个容易记错的事实：**上游的并行推理就是批槽，而且已经合并进主线了**（`docs/BATCHING.md`，
引擎部分 PR #559，opt-in 是 #465）。`--batch N` / `--slots N` 在 `src/program/generate.cpp` 里是现役
代码，本地构建也能跑到它（`--batch: 4 slot sessions on CUDA0`、`batch windows of up to 4 sequences`）。
上游**正在处理**的是批槽的短板，不是批槽本身：

| 上游 | 状态 | 说什么 |
|---|---|---|
| #4121 | open | `--batch-groups` 的流水线组里带 MTP 草稿（`STRATA_PIPED_MTP=N`，opt-in） |
| #4199 | open | dsv41 的批服务，16 槽下聚合 2× |
| #3757 / #3783 | open | Metal 后端的批槽 |
| #2526 | open | 槽复用前保住已解码的批槽会话 |
| #3382 | open | `--batch` 下的 prefill CPU 分担（opt-in，短读 −21% ~ −31%） |

所以训练侧做的是**对接**，不是实现：采集脚本不再自己定一个并发数，而是**问引擎**——读
`/v1/status` 的 `concurrency`（就是配置里 `"parallel": N` 声明的批槽数），按它发这么多条在飞。
`--parallel 0`（默认）= 自动；给数字 = 覆盖。引擎没开批槽时自动是 1，串行发，也不排队。

```
python collect_rollouts.py --engine http://127.0.0.1:8095 --prompts prompts.jsonl     --group 4 --parallel 0 --out rollouts.jsonl
```

组结构本身适合它：同组 K 条共享 prompt 前缀、彼此无依赖，是天然的一批并发请求。显存要说准：
**训练进程**这段不付（采样在引擎里，训练侧只收文本）；但**引擎进程**要为批槽付——每条在飞
请求一份 KV，外加批路径 verify 窗口的缓冲区。这正是下面实测跑不出收益的原因之一：48 GiB 卡被
expert cache 填到只剩 133–324 MiB，批槽要的缓冲没地方放。

**实测（125B，一张 4090，`--batch 4`，28 条请求 × 最长 256 token，交替跑 串行/auto/并行/串行）**

| 在飞请求数 | 墙钟 | 生成 token | 聚合 tok/s |
|---|---:|---:|---:|
| 1（串行） | 8.4 s | 960 | 118.4 |
| auto（引擎说 4） | 9.3 s | 1208 | 134.6 |
| 4 | 8.8 s | 1042 | 122.4 |
| 1（串行，再跑一遍） | 7.6 s | 885 | 120.8 |

结论：**在当前引擎现状下没有可测的吞吐收益**——这是现状，不是批槽的固有属性。并行确实让请求
重叠了（逐请求耗时之和 26.5 s vs 墙钟 6.7 s，重叠 3.8×），但聚合吞吐落在串行两次的区间里
（118.4 / 120.8），差异是噪声。压住它的两个现状条件，日志里明摆着：① 批路径**没有草稿**
（要 `--batch-mtp`，opt-in，且只有两级测过）——串行路径的 MTP 推测（一次前向 ~1.7 token）
批路径拿不到，天然吃亏；② 引擎自己在警告「只有 133–324 MiB 显存空闲，卡这么满时驱动会把显存
换页到内存」——批槽要的 verify 窗口缓冲区根本没地方放。这两个条件都是可变的：上游 #4121 在给
批路径补草稿，#4199 在 16 槽实测聚合 2×。哪天这两个条件翻过来，这个结论就要重测，不能当定论引。

**两条不能当并行用的东西**（都踩过）：

- **批槽不等于训练并行**：学生要 26–34 GiB，引擎常驻 ~40 GiB，一张 48GB 卡放不下两个，
  所以采集与训练必须分两段（采集 → 关引擎 → `--rollouts` 训）。
- **`--batch` 下拿不到 token id**：引擎的 LP 行格式是 `LP <logprob> [id:logprob ...]`，
  而 `id:logprob` 只在 K>0 或给了 allowed 时才打印（`emit_lp_line`，k<=0 且 allowed 为空就
  只有 logprob）。采集脚本请求的是 `top_logprobs: 0`，于是 `tokens_exact=False`、只拿到文本，
  训练端退回本地重编码 + 本地重算 old logprob。要 ids 就把 `top_logprobs` 提到 1——但那时打印的是
  top-1 的 id，采样走温度时不保证等于采样到的 id，**这条路今天不通**，得等上游把带 slot 的 LP 行做出来。

**想要「不碰引擎就能并行」，目前只有多副本**（多个引擎进程，`serve/replicas.py`）；那是几份
常驻，不是一份变并行。

## 15. 术语与路线矩阵：MeZO 是反向传播方案，不是算法（2026-10-30 定）

用户口径：**算法 = 目标函数**（SFT / GRPO / DPO），**MeZO = 反向传播的另一种路线**
（零阶：只靠前向 + 数值差分，替代 autograd）。两者正交，`algos.py` 就是这个矩阵的
单一出处，`train.py --algo` 的**默认 = mezo（零阶）**：

|  | ce（SFT） | grpo | dpo |
|---|---|---|---|
| **zeroth_order（默认）** | `--algo mezo` | `--algo mezo --objective grpo` | `--algo mezo --objective dpo --dpo-data …` |
| backward | `--algo sft` | （grpo.py 两段式） | `--algo dpo` |

可用性标准（用户定的）：**满上下文训练只比推理多一点点显存才算能用**。零阶全程
no_grad，显存 = 推理占用（实测峰值 34.2 GiB ≈ 推理基线 34.30 GiB），天然满足；
反传路线长上下文要 `--ckpt`（34.5~36.1 GiB，也达标但更慢），**留在表里但不是默认**。

零阶 × 每个算法的代价与形态：

- ce：每步 2 前向（±ε 各一次）。
- grpo：采样半步（batch decode）+ 每步 2 次 measure 前向；组内 std=0 的步不动参数
  （`adv0`），诊断数每步打印。
- dpo（本轮新增，`hf_ref.py --objective dpo`）：loss 本来就是标量
  −logσ(β·margin)，margin = (π_c−ref_c) − (π_r−ref_r)，ref = 训练起点（B=0）训练前
  一次预计算。每次 measure = 2 次批前向（chosen 一批、rejected 一批，右 padding 对
  scoring 安全），零阶一步共 4 前向。dpo.py 的 `dpo_margin` 是 margin 的唯一定义，
  零阶与反传两条路线共用；acc / margin 每步打印（loss 降但 acc 停 0 = 没学到）。

**实验记录口径（必填项）**：除训练前后的成功结果外，必须记**训练段总时间**、
**token 生成速度**（采样段 tok/s、评测段生成 tok/s）与峰值显存。hf_ref.py 结束时
打一行汇总（总时间/均步时/生成 tok/s/peak），math_probe.py 每个 adapter 打
train/holdout acc + 生成 tok/s + 评测墙钟。

**学习率教训（本轮）**：MeZO 论文的 lr 1e-6~1e-5 是**全参数**的量级（d=数十亿）。
这里扰动的是 LoRA A/B（d≈6.7M，fp32），lr 1e-6 在 125B 数学题上 40 步纹丝不动
（train acc 持平基座 13/16，proj_grad ~±5 但单元素步长只有 ~5e-6）。零阶路线的
默认 lr 必须按 LoRA 尺度重扫（见 §16）。

## 16. 零阶路线 lr 扫描 + 零阶 DPO（2026-10-30，125B 实测）

**背景**：上一轮对比里「MeZO-GRPO 40 步纹丝不动」被用户定性为 **lr 太低**——属实。
论文的 1e-6 是全参数（d=数十亿）量级；这里扰动 LoRA A/B（d≈6.68M，fp32），
proj_grad ~±5 时单元素步长只有 ~5e-6，根本没搬动。本轮把 lr 扫起来，同时把 DPO
也搬进零阶路线（`hf_ref.py --objective dpo`）——**换反向传播方案，不换算法**。

设置：同一批 24 题（16 train + 8 holdout，mix preset，seed 7，与上一轮完全同题）；
GRPO 6 prompt × 4 条 on-policy（max_new 24，T 1.0，40 步）；DPO 6 对离线（β 0.1，
24 步）；LoRA 48 模块 r16 α32；pack 24 GiB VRAM + 30 GiB pinned。

| 臂 | lr | 步数 | 训练段总时间 | 均步时 | 采样生成 | 峰值 | train greedy | holdout greedy | 评测生成 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| 基座（B=0） | — | — | — | — | — | — | 13/16（81%） | 8/8（100%） | — |
| 反传 DPO（上轮对照） | 5e-5 | 24 | ~245 s | ~10 s | 无 | 34.3 GiB | 15/16（94%） | 8/8 | — |
| 零阶 GRPO | 1e-4 | 40 | **294 s** | 7.4 s | 799 tok / 130 s = 6.1 tok/s | 34.2 GiB | **16/16（100%）** | 6/8（75%） | 11.5 tok/s |
| 零阶 GRPO | 1e-3 | 40 | 369 s | 9.2 s | 991 / 192 s = 5.2 tok/s | 34.2 GiB | 13/16（81%） | 4/8（50%） | 7.7 tok/s（答案变长漂移） |
| 零阶 GRPO | 1e-2 | 40 | 626 s | 15.7 s | 3612 / 419 s = 8.6 tok/s | 34.2 GiB | **0/16 崩溃** | 0/8 | 乱码 |
| 零阶 DPO | 1e-4 | 24 | 252 s | 10.5 s | 无 | **33.9 GiB** | **16/16（100%）** | 6/8（75%） | 13.6 tok/s |
| 零阶 DPO | **1e-3** | 24 | 259 s | 10.8 s | 无 | **33.9 GiB** | 15/16（94%） | **7/8（88%）** | 12.9 tok/s |
| 零阶 DPO | 1e-2 | 24 | 197 s | 8.2 s | 无 | 33.9 GiB | **0/16 崩溃** | 0/8 | 乱码 |

（"评测生成" = math_probe 每 adapter 的 greedy+K 采样聚合 tok/s。）

**结论**：

1. **lr 1e-6 就是上一轮的全部死因**：lr 1e-4 下零阶 GRPO train 16/16（100%）——
   零阶做 RL 本身成立，之前只是没推动参数。
2. **甜点带 1e-4~1e-3；1e-2 两个目标都崩**（输出全是乱码 token）。零阶步长
   lr·pj·z，pj~O(1-10)，1e-2 直接把 LoRA 推出流形。默认 lr 定 **1e-3**（DPO 最优点），
   GRPO 建议降到 1e-4（1e-3 下答案变长漂移、holdout 掉到 50%）。
3. **零阶 DPO @1e-3 = train 94% / holdout 88%**，与反传 DPO 的 train 94% 同档，
   但：峰值 **33.9 GiB ≈ 推理基线 34.30 GiB**（无 ckpt、无激活图、无权重钉存——
   §4 那个 92.63 GiB 病根和 ~18 GiB backward 瞬态在这条路线上**不存在**），
   每 step ~10.8 s 与反传 DPO 的 ~10 s 同档。
4. **可用性标准（满上下文只比推理多一点点）在零阶路线上对所有三个算法成立**：
   ce/grpo/dpo 全程 no_grad，峰值 33.9~34.2 GiB，都低于引擎常驻 ~40 GiB。
5. holdout 回归细节：第 18 题（70×41）与第 23 题（39×59）是脆弱题（都是"差一位"
   型）；基座两题都对，1e-4 档两臂把两题都打坏（75%），zdpo@1e-3 只坏第 23 题
   （88%）——**扩大 holdout + 独立种子才能把 75%/88% 的差做成结论**（8 题一格就是
   12.5%，当前只是方向性信号）。
6. 时间口径（用户要求的必填项）：零阶 GRPO 一臂 40 步 = 294~626 s 训练 + ~400 s 加载；
   零阶 DPO 一臂 24 步 = ~255 s 训练。评测一次加载评 6 个 adapter（math_probe --lora
   多值），每 adapter ~120-180 s、生成 11.5~13.6 tok/s。

## 17. pack 装载加速：fadvise DONTNEED（2026-10-30，独立 PR #5）

分相计时（本 PR 顺带加进 `tiered pack` 日志行）定位 333 s 的构成：
`pin 49s + ram-read 272s + gpu-loop 11s`。gpu-loop 24 GiB @ 2.2 GiB/s 与 ram-read
36 GiB @ 0.13 GiB/s 读同一个文件、落点同为 pinned ⇒ 唯一结构差是 ram 层把
36 GiB 全留在 RAM：60 GiB 单次使用的 page cache 流过 58 GiB 的盒子，内核边读
边回收。**每块读完立即 `posix_fadvise(POSIX_FADV_DONTNEED)`**：

| | 28 GiB 冷读 |
|---|---|
| 带 fadvise | **9.4 s（2.98 GiB/s）** |
| 不带 | 85.8 s（0.33 GiB/s） |

端到端（125B IQ3_XXS，VRAM 24 + pinned 30，缓存清空）：`_ensure` 333 s → **101 s**
（pin 21 + ram-read 73 + gpu-loop 6），整体加载 454.6 s → **216.9 s**。正确性：按
真实 `block_bytes`（2,611,200）对 VRAM/pinned/pageable 三层各采样块与文件字节
比对，全 MATCH（第一轮报 MISMATCH 是检查脚本硬编码错块大小，非 loader 问题）。
Linux-only API，getattr 守卫，别处 no-op；`STRATA_PACK_FADVISE=0` 关闭。
已在独立分支 `perf/pack-load-accel` 提交并开 PR（fork 内、base=train_support，
25 insertions / 1 deletion）。

**待整理（2026-10-31 定）**：这属于「加载」能力，与训练无关，PR 应 rebase 到主线、
base 换成主线而不是 `train_support` —— 训练分支不该成为它的前置。目前 PR 已关，
代码留在 `perf/pack-load-accel`，等练完主线再单独重开。

## 18. Windows 训练支持（2026-10-30 实施）

- **preadv shim**：`preadv_shim.py`（纯标准库）——Windows 无 `os.preadv/pread`，
  用 `open(fd, "rb", buffering=0, closefd=False)` 包既有 fd + `seek`+`readinto`，
  零额外拷贝；每次调用开临时句柄，多线程随机读不共享 seek 位置（bench_pack_io
  4 线程安全）。loader 两处调用点与 bench 换成垫层；Linux 走原生，语义不变。
- **门禁更新**：`platform_check.py` 的判据从「平台名」改为「有无可用 preadv」
  （原生或垫层），Windows 从阻断改为放行；madvise 缺失仍是降级项（映射页不还，
  ~15 GiB）。README 平台矩阵同步改。测试 59/59（Windows 与 WSL 双侧跑过），
  垫层与 os.preadv 的功能等价/短读/多线程用例入 CI。
- **环境**（与 WSL 完全同版本）：torch 2.13.0+cu132（Windows wheel 存在）、
  triton-windows 3.7.1.post27（PyPI 官方 triton 只有 Linux wheel；abi3/cp313 对
  Python 3.13 可用）、transformers 5.17.0、kernels 0.16.1、torchvision 对齐 0.28.0。
- 冒烟（D 盘 pack 副本 60 GB，md5 与 WSL 原件一致）+ 小显存档加载 + 前向：见当日日志。

## 19. 训练线 MTP + 最终速度总表（2026-10-30，Windows 4090 48GB，pack 16 GiB）

**MTP 投机解码已接入训练线**（`mtp_draft.py`，权重源 `Strata-data/mtp/` 的 31 个 BF16 张量）：
- 协议与引擎 `src/core/mtp.hpp` 逐算子对齐：cell=(R_i, x_{i+1})@pos i → 预测 x_{i+2}；
  fc_embedding/fc_hidden 双分支 + 同构解码层（QSA+超连接+512 专家 MoE）+ 顶层 mixer + 主干 head。
- 踩过的坑（都是判据级）：① 原始 BF16 不能 `'<f2'` 解（那是 fp16），要 uint16<<16 拼 fp32——
  解错时 teacher-forced 准确率 0.000、修后 0.671；② 层类型串必须 `qwen_sparse_attention`
  （DynamicCache 才建 indexer 槽）；③ indexer 要全窗口 cos/sin（mrope [3,B,L]，cache 绑全量
  position_ids）；④ HF `Cache.crop` 语义 = 负数移除行数，且 **GDN 混合架构不能用 crop 回滚**
  （recurrent 状态不回滚 = 静默数值污染）⇒ 目标缓存用快照/恢复；⑤ pre_fc_norm 用整条 10240
  一个 RMS（引擎默认；stream 模式 teacher-forced 高但链式 accept 减半，full 胜出）。
- **单流长生成**（128 token，greedy k=3）：accept 1.80/轮、tok/fwd 2.80、156~165 ms/token =
  plain 的 **1.48~1.57×**（plain 245 ms/token）。峰值 39 GiB（drafter BF16 专家 ~5 GiB）。
- **GRPO 组采样（K=4, max-new 32）A/B**：批量 MTP 聚合 **2.9 tok/s** vs plain 批采样 **5.0 tok/s**
  ——drafter 每轮 3 次全词表 head（248K×2560，1.27 GB/次）+ 快照回滚吃掉收益。**引擎的
  40K draft_vocab 子集头（小 32×）+ Q2_0 专家正是为此**；接入这两件后批量 MTP 才可能反超。
  在那之前 GRPO 采样保持 plain 批采样为默认（`--mtp-dir` 不给即 plain）。

| 口径（125B IQ3_XXS，Windows 原生） | 吞吐（token/s） | 峰值显存 | 备注 |
|---|---|---|---|
| MeZO-SFT（ce，短序列 loss_tokens 10~17） | **2.9~6.8 tok/s**（loss-token 口径） | 25.8 GiB | 步时 2.5~3.5 s，短序列 launch-bound，不反映满上下文能力 |
| MeZO-SFT（**T=2048 满上下文**，2 次整段前向/步） | **4096 tok/步 ÷ 6.9 s = 594 tok/s**（Windows）；÷ 8.8 s = 465 tok/s（WSL） | ~26 GiB | 训练前向的可持续吞吐，与引擎稳态 prefill 同档 |
| MeZO-DPO（155 loss tokens，4 前向/步，短序列） | **7.4~7.5 tok/s**（loss-token 口径） | 26.0 GiB | 步时 20.8 s，launch-bound 同上 |
| MeZO-GRPO 采样（K=4 plain 批，生成段） | **5.0 tok/s 聚合**（146 token/29 s）；单步 2.1~6.5 | 26.2 GiB | 整步 12.5 s（打分前向另占 5~10 s） |
| MeZO-GRPO 采样（K=4 批量 MTP，生成段） | **2.9 tok/s 聚合**（355 token/121 s）；稳态步 4.8~5.0 | 31.6 GiB | 慢于 plain，瓶颈=每轮 3 次全词表 head |
| 单流长生成（MTP，greedy k=3） | **6.1~6.4 tok/s**（plain 4.1，1.5×） | 39 GiB | 128 token 文本续写 |
| 引擎对照（同卡 C++ decode） | **172.8 tok/s** | — | 训练线采样差距 ~29×（plain）/ ~27×（MTP） |

## 20. 速度口径重算 + 成本拆解（2026-10-30，`_mb_cost_split.py` 实测）

**为什么要重算**：§19 那张表混了三种口径，读者很容易拿「594 tok/s」（两次前向的算力口径）
去比「9.1 tok/s」（DPO 只数 answer 段的 loss-token 口径），得出「差几十倍」的错结论。
这一节把口径钉死，并给出每次前向成本的实测拆解。

### 20.1 一个前向要多久（同一进程内，Windows 4090 48GB，`--vram 16`）

| 形状 | 墙钟 | 其中专家 H2D | 命中专家块 | 每层均值 |
|---|---:|---:|---:|---:|
| T=2048, batch=1 | **4.03 s** | 0.97 s（24%） | 12,928 | 269 / 512 |
| T=32, batch=1 | 1.21 s | 0.36 s（30%） | 4,565 | 95 / 512 |
| 2 行 × 23 token（DPO 1 对） | 1.49 s | 0.35 s | 4,394 | 92 / 512 |
| 4 行 × 27 | 2.26 s | 0.50 s | 6,254 | 130 / 512 |
| 8 行 × 27 | 3.31 s | 0.65 s | 8,063 | 168 / 512 |
| 20 行 × 24（DPO 10 对，外推＝实测 15.8 s ÷ 2） | ~7.9 s | — | ~12,000 | ~250 / 512 |

**读法与三条硬结论**：

1. **成本由「专家命中数」决定，不是 token 数**。8 行 × 27 = 216 token 要 3.31 s，而单行
   T=2048 = 2048 token 只要 4.03 s —— 同样时间差了 9.5 倍 token。原因是短序列每一行都带
   一批**新的路由上下文**，每层并集随行数涨（92→130→168→~250），而长序列内部 token 的
   路由高度重叠，很快就饱和在 269/层。
2. **H2D 只占 24%，不是瓶颈**。要拿掉它最乐观也就 4.03 → 3.06 s（660 tok/s 算力口径）。
   剩下 76% 全在计算，而 §11 已经量过：24,576 次专家 launch，单次 330 µs，折合 **3 TFLOPS
   = 4090 峰值的 2%**。**追平引擎 prefill（p50 1482）的唯一杠杆是 grouped GEMM，不是缓存。**
3. **短序列的 tok/s 没有意义**。1.21 s 里有 1.2 s 是与 token 无关的固定量（专家流起步 +
   launch + 25 层稠密权重），17 个 token 摊上去就是 14 tok/s。

### 20.2 重算：每个算法「应该」是多少（数据口径 = 真正参与训练的 token / 步时间）

单次前向拟合 `c(T) ≈ 1.2 s + 1.40 ms × T`（由 T=32 与 T=2048 两点定），MeZO 每步 2 次（±ε）：

| 算法 | 每步前向 | 每步训练 token | 应该步时 | **应该 tok/s** | 实测 | 差 |
|---|---|---:|---:|---:|---:|---|
| SFT（ce）@T=2048 | 2 | 2,048 | 8.06 s | **254** | 257（7.98 s） | ✓ 一致 |
| SFT（ce）@T=128K | 2 | 131,072 | ≈369 s | **355** | 未测 | 固定成本摊薄后更高 |
| DPO（10 对，24 token/条） | 2 | 480 | 15.8 s | **30** | 30（15.8 s） | ✓ 一致 |
| DPO（若每对 2048 token） | 2 | 40,960 | ≈10.5 s | **3,900** | — | 专家并集饱和后的应有值 |
| GRPO 训练段 | 2 | K×len | 同 SFT | 254 | ~26（5 s / 130 token） | 短序列 |
| GRPO 生成段 | — | K×len | — | 5.0（学生自采）/ 172.8（引擎单流） | 5.0 | — |

**反传路线对照**（§1.1/§11）：T=2048 稳态 22.59 s/步 = **90.7 tok/s**。MeZO 的 257 是它的
**2.8×**，代价只有 no_grad（显存 25.7 vs 34.5 GiB）。这是选 MeZO 的全部理由。

**DPO 没有任何多余开销**。用户口径「DPO = 两个 SFT，总用时翻倍、tok/s 相同」**是对的**：
它每步就是 2 次前向、每次 2P 行，成本由行数决定。§19 里那个 7.4~9.1 tok/s 是 loss-token
口径（只数 completion 的 155 个 token），把 prompt 的成本藏掉了 —— 同一份日志按
`fwd_tokens` 算就是 30~60 tok/s，与短序列 SFT 完全同档。**慢的印象来自口径，不是实现。**

### 20.3 GRPO 要做到「N×1.5」需要什么

设每步训练 S 个 answer token：

```
SFT 步时  = 2·c(S)
GRPO 步时 = S / v_gen + 2·c(S)          ⇒ 倍数 = 1 + S/(2·v_gen·c(S)) = 1 + v_sft/(2·v_gen)
```

其中 `v_sft = S/c(S) ≈ 508 tok/s`（单前向算力口径；数据口径 254）。代入：

| 生成路径 | v_gen | 总用时 / 纯 SFT | 说明 |
|---|---:|---:|---|
| 学生自采（现状） | 5.0 | **102×** | 每 token 搬全套专家 |
| 引擎单流（现状） | 172.8 | **2.47×** | 离线采样即可拿到 |
| 引擎 **3 槽** | 518 | **1.49×** | ✓ 用户要的 N×1.5 |
| 引擎 4 槽 | 691 | 1.37× | 受 §14 现状限制 |

**结论：N×1.5 的充要条件是生成 ≥ 508 tok/s ≈ 引擎 3 槽批处理。** 单流 172.8 只能给
2.47×，学生自采是 102×。而 §14 实测批槽当前**没有**吞吐收益（118.4 / 122.4 / 134.6），
卡在两件事上：批路径没有 MTP 草稿（#4121）、expert cache 把显存吃到只剩 133~324 MiB。
所以这条线的排序是：**先把生成搬回引擎（2.47×）→ 等批槽生效（1.5×）**，不是先优化训练。

### 20.4 这一轮修掉/查清的四件事

| # | 项 | 判据 | 处置 |
|---|---|---|---|
| 1 | `tok/s` 只数 answer 段 | DPO 10 对：155 loss token / 480 fwd token / 15.8 s ⇒ 打印 9.8 而真实 30 | 日志同时打 `tok/s` 与 `fwd/s`，`info["fwd"]` 由三个目标函数各自返回 |
| 2 | `_completion_sums` 对 prompt 段也算 248K logits | prompt 段算完就被 mask 丢掉；lm_head 1.27 GB/位置 | 只算 `[min(plen)-1, L-1)`，`test_train.CompletionSums` 守逐位等价（chunk=0/4/128 三档） |
| 3 | `_completion_sums` 的 `chunk=8` 写死 | head 权重按 L/8 次重读（L2 只有 72 MB） | 改成按行数预算自适应（256 MiB / 行），收益 <1%，属清理 |
| 4 | **GDN kernel 落点会漂移** | 中文 Windows + `PYTHONUTF8=0` → torch inductor 用 GBK 读 `cutedsl_mm_grouped` 模板 → `UnicodeDecodeError` → fla 导入断 → GDN 静默回落 torch 参考实现 | 测速前必须 `PYTHONUTF8=1`；`_mb_cost_split.py` 里加了落点断言（回落直接拒测） |

第 4 条是最容易踩的：回落时**单步只慢 4%**（4.18 vs 4.03 s），但**显存会随步数累积**——
同一进程连跑几次 T=2048 前向后 `allocated` 冲到 60.4 GiB（超过 48 GiB 物理，Windows 允许
显存超发到 host），然后 OOM。**只看单步时间发现不了，跑 20 步就崩。** 复盘时先看日志里
有没有 `chunk_gated_delta_rule ... falling back` 这一行。

### 20.5 还没修的（按值排序）

1. **MeZO ±ε 的两次前向专家流 100% 冗余**（实测 `hits` 序列 48/48 层逐位相同 —— router
   不在 LoRA 里，扰动不改变路由）。理论上第二次可以全命中，但整个 pack 要 31.5 GiB，
   显存余量只有 19.1 GiB。所以正解是 **expert-cache（LRU 常驻 7900 块 = 32%）**，
   预期 H2D 0.97 → 0.4 s，步时 8.06 → 7.5 s（+7%）。**收益小于 grouped GEMM，排在后面。**
2. **DPO 的 chosen/rejected 共享 prompt 却算两遍**：20 行里有一半 prompt 是重复的，
   用 KV 复用可省 25% 前向 token（15.8 → 11.9 s）。但会改变数值路径（GDN recurrent 状态
   走 cache 与全序列前向不完全一致），风险/收益不划算，**只记不做**。
3. **grouped GEMM**（§11 ②）：唯一能改算术强度的一条，24,576 次 launch → 96 次。


## 21. 满上下文标准基准：DPO/GRPO 被 QSA 的 batch 退化拖慢了 6.6×（2026-10-30）

### 21.1 为什么要新建一个基准

历史测速散在 `hf_ref` / `sft` / `_mb_cost_split` 三处，每份的 token 口径都不同
（answer 段 / completion 段 / fwd 段）。于是同一份日志能读出 9 tok/s 和 30 tok/s，
DPO 被判成「慢几十倍」。`bench_fullctx.py` 把口径焊成一条：

    tok/s = 每步**前向真正处理的 token 数**（行数 × 序列长度） / 步时

每步的前向次数按算法真实定义计：`ce_bwd` 1f+1b、`ce_mezo` 2f、`dpo_bwd` 2 行同批
1f+1b、`dpo_mezo` 2 行同批 2f、`grpo` K 行同批 1f+1b / 2f。

**不拼接上下文**：DPO 的 chosen / rejected 各自是一行（长度 T），只在 **batch 维**
合并。拼成 2T 会让 rejected 看到 chosen，且 attention 成本翻 4 倍 —— 满上下文直接
不可用。

### 21.2 MeZO 档实测（Windows 4090 48 GB，pack 16 GiB，合成等长序列，prompt 占 50%）

| 算法 | T | 每步 token | 步时 s | **tok/s** | 峰值 GiB | 相对 ce |
|---|---:|---:|---:|---:|---:|---:|
| ce（SFT） | 2048 | 2,048 | 8.29 | **247.2** | 26.2 | 1.00× |
| dpo | 2048 | 4,096 | 10.91 | **375.5** | 26.7 | 1.52× |
| grpo K=4 | 2048 | 8,192 | 17.42 | **470.2** | 27.7 | 1.90× |
| ce | 8192 | 8,192 | 17.68 | **463.2** | 27.7 | 1.00× |
| dpo | 8192 | 16,384 | 29.65 | **552.6** | 29.6 | 1.19× |
| grpo K=4 | 8192 | 32,768 | 56.51 | **579.9** | 33.5 | 1.25× |
| ce | 16384 | 16,384 | 31.33 | **523.0** | 29.8 | 1.00× |
| dpo | 16384 | 32,768 | 56.33 | **581.8** | 33.7 | 1.11× |
| ce | 32768 | 32,768 | 63.06 | **519.6** | 35.3 | 1.00× |

吞吐随 T 单调上升并趋近 ~550-580 tok/s（固定成本摊薄），**与 §20 的
`c(T) ≈ 1.2 s + 1.40 ms×T` 外推一致**（T=16384 外推 24.1 s，实测单次前向 15.7 s ——
比外推更快，因为 §20 的 1.40 ms/token 是在 Windows + QSA 未分块时测的）。

### 21.3 根因：QSA 的向量化只支持 batch=1

修前同一口径：`dpo T=2048` = **72.16 s / 56.8 tok/s**。修后 10.91 s / 375.5 —— **6.6×**。

`qsa_fast._fast` 原本有一条 `if batch_size != 1: raise _Fallback("batch != 1")`。
回落目标是 transformers 的逐 query 原实现：每层 `for query_idx in range(S)`，
读 `.shape[-1]`、`torch.nonzero` 都触发 **DtoH 同步**（§qsa_fast 注释：T=2048 一步
24,661 次 `Memcpy DtoH` + 24,763 次 `cudaStreamSynchronize`，GPU 只有 5.3 s 忙 /
17.4 s 墙钟）。于是：

* SFT 单行 → 向量化快路径 → 4 s；
* **DPO 的 chosen+rejected 两行、GRPO 的 K 行 → 一律回落 → 单次前向 36 s**。

这不是「DPO 算法慢」，是「批处理把 QSA 打回了逐 query 循环」。修法两条：

1. `_fast` 支持 `B > 1`：逐行校验可见集一致（同长度、无 padding 才走），
   pooled / rope / dot / topk / 掩码全部带 batch 维；有 padding 仍 `_Fallback`。
2. **分数/topk/掩码按 query 分块**（`STRATA_QSA_QCHUNK`，默认 4096）。分数张量是
   `(B, S, nb, H)`，`nb = S/4`、`H = 4` ⇒ T=128K 时一次算完 **68.7 GiB**。分块后峰值
   只与块宽有关（4096 行 ≈ 2.1 GiB），topk 对每行独立，**不影响任何一行的选择结果**。

顺带修掉：`_completion_sums` / `grpo_loss` 在没有 padding 时不再传 4D
`attention_mask`（全 1 mask 与无 mask 的前向逐位相同，但 4D mask 会让 sdpa 离开
flash/causal 快路径）。`test_train.DPOMeasure` 用 4 个用例守住「合批 ≡ 拆两批」。

### 21.4 与「DPO = 两个 SFT」对账

| T | 两个 SFT 串行（2×ce） | DPO 实测 | 比值 |
|---|---:|---:|---:|
| 2048 | 16.58 s | 10.91 s | **0.66×** |
| 8192 | 35.36 s | 29.65 s | **0.84×** |
| 16384 | 62.66 s | 56.33 s | **0.90×** |

DPO 的 token 数是 2 倍、前向次数也是 2 倍，但**批内两行共享一次专家流与一次固定
成本**，所以始终快于「两个 SFT 串行」。T 越大差距越小（固定成本被摊薄后趋近 2×）。
**「DPO 慢几十倍」是口径造成的错觉，实现上没有多余开销。**

### 21.5 128K：卡在 4D mask 的三份副本，不是卡在算力

`Qwen4ExpTextAttention.forward` 对 QSA 层做：

```python
selected = self.indexer(...)                 # (1, 1, S, S) bool  ← 副本 2
attention_mask = attention_mask & selected   # (1, 1, S, S) bool  ← 副本 3
```

`model` 侧先造一份因果 mask（副本 1）。S=128K 时每份是 **16.0 GiB**（S² 字节），
三份 **48 GiB**，再加常驻 25.2 GiB —— 48 GB 卡连装都装不下，还没轮到 sdpa。

**更正（2026-10-30 实测）**：此前写的「sdpa 对 4D bool mask 不走 flash、math 路径权重
矩阵 824 GB」是错的。隔离测试（q/k/v 24 头 × 128 维，bool 因果 mask）：

| S | mask | 峰值增量 | 假想权重矩阵 H·S²·2B | 峰值/权重 |
|---:|---:|---:|---:|---:|
| 4096 | 0.02 GiB | 0.14 GiB | 0.75 GiB | 0.19 |
| 16384 | 0.25 GiB | 1.12 GiB | 12.0 GiB | 0.09 |
| 32768 | 1.00 GiB | 3.75 GiB | 48.0 GiB | 0.078 |

峰值远低于权重矩阵（且本机 `flash_available=False`）⇒ 走的是 **memory-efficient**
后端，它原生吃 bool mask，不会物化 S²×H 的注意力权重。**128K 的唯一障碍就是那三份
bool mask 本身，不是 sdpa。**

实测上界在 32K：`ce T=32768` = 63.06 s / 519.6 tok/s / peak 35.3 GiB 仍正常；再往上
每翻一倍，4D mask 就多 3 GiB → 12 GiB（64K）→ 48 GiB（128K），与「常驻 25.2 GiB」
相加后 64K 已到边缘、128K 直接超物理上限。

⇒ **128K 训练在 HF 路径上不可行，且不是调参能解决的**。要跑必须把 QSA 层改成
「按 top-k 索引 gather K/V 再做 attention」（引擎 `qwen_sparse_attention` 就是这么做的），
完全不产 4D mask。这是「同步官方推理方案」这条主线里缺的最后一块。

### 21.6 MeZO vs 反传：同口径是 1.73×，不是 2.8×

同一进程、同一 LoRA（rank 16 / o_proj+out_proj / 6.68 M 参数）、同一合成数据，T=2048：

| 算法 | MeZO（2 前向） | 反传（1f+1b，ckpt on） | 倍数 | 峰值 GiB MeZO / 反传 |
|---|---:|---:|---:|---:|
| ce | 8.29 s / **247.2** | 14.36 s / **142.6** | **1.73×** | 26.2 / 30.1 |
| dpo | 10.91 s / **375.5** | 19.08 s / **214.6** | **1.75×** | 26.7 / 32.5 |
| grpo K=4 | 17.42 s / **470.2** | 27.13 s / **302.0** | **1.56×** | 27.7 / 38.5 |

§1.1 / §20 里那个「MeZO 是反传的 2.8×」（22.59 s vs 8.06 s）是三件事叠加出来的：
**WSL2 的旧 loader**（反传 22.59 → 14.36 s）、**answer 段 token 口径**（90.7 → 142.6 tok/s
只是把分母从答案段换成整段）、**QSA 未向量化**。分子分母都变了，比值从 2.8 掉到 1.73。
**以本节为准。**

代价侧：MeZO 用 1.56~1.75× 的墙钟，换 4~11 GiB 显存（26.2~27.7 vs 30.1~38.5，ckpt on）。
GRPO 的差距最小（1.56×），因为 K 行批前向把反传的「每层重算」摊薄了。

两条线各自的下一步（互不替代，都得做）：

| 线 | 杠杆 | 预期 |
|---|---|---|
| 共同 | **grouped GEMM**（§11 ②）：一层 512 个专家合成 1 次 segmented GEMM，24,576 次 launch → 96 次 | 唯一改算术强度的一条，MeZO/反传同受益 |
| MeZO 专有 | ±ε 第二次前向的专家流 **100% 冗余**（router 不在 LoRA 里）⇒ expert-cache 常驻 7900 块（32%） | H2D 0.97 → 0.4 s，步时 −7% |
| 反传专有 | 去掉 grad checkpointing：ctx 只存专家 id，backward 按 id 重流权重 | 省掉整段重算，18.2 → 14.5 s 那一档 |

## 22. 满上下文的真实边界：MeZO 到 32K，反传到 8K（2026-10-30 实测）

### 22.1 实测边界

同一进程、同一 LoRA（r16 / o_proj+out_proj / 6.68 M）、Windows 4090 48 GB、pack 16 GiB：

| 档位 | T | 峰值 GiB | 结果 |
|---|---:|---:|---|
| MeZO ce | 2048 | 26.2 | ok 247.2 tok/s |
| MeZO ce | 8192 | 27.7 | ok 463.2 |
| MeZO ce | 16384 | 29.8 | ok 523.0 |
| MeZO ce | **32768** | **35.3** | ok 519.6 |
| MeZO ce | 65536 | 外推 50.8 | **OOM**（干净报错） |
| 反传 ce | 2048 | 30.1 | ok 142.6 |
| 反传 ce | **8192** | **42.0** | ok 282.4 |
| 反传 dpo（2 行） | 8192 | > 48 | **`AcceleratorError` 崩在 native 栈** |
| 反传 ce | 16384 | > 48 | **既不报错也不退出**：分配失败（`free: 0`）后显存超发，GPU 100%、日志零行 |

结论：**满上下文（128K）两条线都到不了；MeZO 的实际上界是 32K，反传是 8K。**

### 22.2 两条线为什么差 6.5×

拟合出的每 token 斜率（GiB/token）：

| 档位 | 斜率 | T 翻倍一次多占 |
|---|---:|---:|
| MeZO | 2.0e-4（线性项）+ 3·T²/2^30（mask 项） | 0.5 GiB（2K→4K），12 GiB（64K→128K） |
| 反传 | 1.94e-3 | ≈ 4 GiB（2K→4K），≈ 32 GiB（32K→64K） |

反传的斜率是 MeZO 的 **9.7×**，因为它要留住每层 ckpt 边界激活 + MoE 中间激活（top-8
展开），而 MeZO 全程 `no_grad`，只有当层的临时张量。**这个差距是算法性的，不是
kernel 写得不好。**

128K 时 MeZO 的账：`25.2（常驻）+ 26.2（线性）+ 48.0（3 份 bool mask）= 99.4 GiB`。
其中 48 GiB 全是 QSA 层那三份 `(1,1,S,S)` bool（见 §21.5）。**只要把 QSA 改成
「按 top-k 索引 gather K/V」，这 48 GiB 归零，MeZO 128K 落在 ~51 GiB —— 仍略超 48 GB，
但那是「再压一点就能进」的量级，而不是算术上不可能。**

### 22.3 反传还有多少优化空间

分两类，别混：

**能改的（算力侧，不移动显存墙）**

| 杠杆 | 依据 | 预期 |
|---|---|---|
| grouped GEMM | 每层 24,576 次微小 launch，3 TFLOPS = 峰值 2%；反传给每个算子再加 1~2 个反向节点（实测建图+反传 = 3.20× no_grad） | 唯一改算术强度的一条，MeZO/反传同受益 |
| 去掉 grad ckpt，ctx 只存专家 id + 路由、backward 按 id 重流 | 省一次完整前向重算（ckpt on 下 3 个前向当量 → 2 个） | 反传快 ~33% |
| 反传的第二次 H2D 复用 | 前向与重算的专家流相同 | 省一半 H2D |

**改不动的（显存侧）**：激活体积由 `层数 × T × hidden × top_k` 决定，跟 kernel 无关。
所以「反传能满上下文」这条路在 48 GB 单卡上是封死的，除非上 CPU offload（每步多搬
`2×T×hidden×2B`/层）或张量并行。

### 22.4 显存策略：满显存必须报错终止，不能交给 CUDA

上面「16384 既不报错也不退出」就是反例。新增 `mem_guard.py`：

```python
estimate_peak_gib(T, rows, mode)   # 实测拟合，残差 ≤0.6 GiB
require_budget(...)                # 开跑前预算，超限 SystemExit(2) 并打印该改什么
is_oom(exc) / die_on_oom(...)      # 跑到一半真 OOM → 打印现场 + 退出码 2
```

* 公式复现了全部 4 个 MeZO 实测点与 2 个反传实测点，并正确预言 64K OOM；
* `sft.py` / `hf_ref.py` 在装载后、第一步之前调用 `require_budget`；
* `bench_fullctx.py` 每个口径先 `_preflight`，并新增 `--on-oom {abort,skip}`，
  **默认 abort**（`skip` 只用于测绘边界矩阵）；
* 6 个新测试（`test_train.MemGuard`）守公式与拒绝行为。

理由：Windows 显存超发下「降速」不是可控降级，而是不可控抖动 —— 实测 16K 反传
把 30 s 的步拖成十几分钟且日志零输出，测出来的 tok/s 没有任何参考价值。

## 23. 数据类型现状（实测审计）与可调项

### 23.1 每一段到底是什么精度

| 部分 | 存储 | 计算 | 位置 |
|---|---|---|---|
| 基座非专家权重（~9.2 GiB） | GGUF 原量化（Q8_0/Q6_K…）→ 装载时反量化 | **bf16** | `loader_qwen4.py:1075` `model.to(torch.bfloat16)`（硬编码） |
| 专家（原 96 GiB） | **NF4 4-bit** pack，显存常驻 16 GiB | 解到 **bf16** 再 GEMM | `nf4_fused.py` / `view(torch.bfloat16)` |
| 激活 / attention / QSA | — | **bf16** | 全模型 |
| LoRA A/B（6.68 M） | 反传 **bf16**；MeZO **fp32** | 反传 bf16 GEMM；MeZO delta 走 fp32 GEMM | `lora.py:44` 继承 base dtype；`hf_ref.zo_cast_fp32` |
| logprob 的 logits | 分块，**fp32** | fp32 logsumexp | `ce.token_logprobs` 的 `.float()` |

### 23.2 超过 16 位的只有两处，都不是"模型精度"问题

1. **MeZO 的 LoRA（fp32，25.5 MiB）**：ZO 更新 `θ ±= ε·z`，ε=1e-3 落在 bf16 参数上会被
   直接舍掉（bf16 相对精度 2^-8≈0.4%，而扰动相对幅度 ~1e-3），等于没扰动。
   代价：6.68 M × 4B = 25.5 MiB（bf16 是 12.7 MiB），**相对 25.2 GiB 常驻可以忽略**。
   新增 `--lora-dtype {fp32,bf16}`：改 bf16 时必须把 `--eps` 抬到 ≥1e-2，否则学不动。
2. **logits 分块（fp32，`chunk×248320×4B`，chunk=128 → 127 MB）**：248K 词表上 bf16
   logits 的 logsumexp 有 ~1e-2 nats 抖动，会直接污染 DPO 的 margin 与 GRPO 的 ratio。
   这部分是**瞬时**的，不参与常驻。

⇒ 模型本体 / 专家 / 激活全是 ≤16 位（bf16 或 4-bit），没有 fp32 权重副本。

### 23.3 想"自己控制"还缺什么

| 想要 | 现状 | 要动的地方 |
|---|---|---|
| `--dtype fp16` | 不支持：专家解码路径产出 bf16 view，`F.linear` 混精度会报错 | `nf4_fused` 增加输出 dtype 参数 + 全链路传参 |
| 更多低量化 pack | 只有 NF4（4-bit）。GGUF 侧本就有 Q2_K…Q8_0/IQ 系列；引擎的 MTP 草稿头用的就是 **Q2_0** | `loader_qwen4.py convert` 的 quant 类型做成参数，pack 头写类型 |
| logits 精度 | 硬编码 fp32 | `token_logprobs(..., acc_dtype=)` |

## 24. MeZO 为什么在 128K OOM：不是 LoRA，是三张 S² 掩码 + 一条 26 GiB 线性项

### 24.1 归因（实测，不是推算）

`_mb_mask_scale.py` 在**真代码路径**上量（`Qwen4ExpTextQSAIndexer.forward`，0.01 M 参数
—— 参数与 S 无关，掩码尺寸也与模型大小无关，所以小模型能问清大模型的问题）：

| S | 每份 mask | indexer 峰值 | = 输入 mask + zeros + scatter |
|---:|---:|---:|---:|
| 16384 | 0.25 GiB | **0.77** | 0.75 |
| 32768 | 1.00 GiB | **3.03** | 3.00 |
| 65536 | 4.00 GiB | **12.06** | 12.00 |
| 131072 | 16.00 GiB | **48.11** | 48.00 |

四倍序列 → 四倍 mask、**十六倍**峰值。128K 时**单这一层**就要 48 GiB（48 GB 卡被它
一张吃满）；而 `Qwen4ExpTextAttention.forward` 紧接着还要 `attention_mask & selected`
再造一份 16 GiB —— 所以 `mem_guard` 的 `3·T²/2^30` 项（=48 GiB @128K）刚好是「因果 /
选中 / 与」三张表里同时活着的那三份，不是随手取的系数。

三处来源，都在上游实现里（`transformers/models/qwen4_exp/modeling_qwen4_exp.py`）：

| 位置 | 分配 | @128K |
|---|---|---:|
| `create_causal_mask(..., allow_is_causal_skip=False)` L1466 | `(1,1,S,S)` bool | 16 GiB |
| `QSAIndexer.forward` L768 `torch.zeros((B,S,kv+1))` | `(1,S,S+1)` bool | 16 GiB |
| 同函数 L773 `.scatter(...)`（out-of-place） | 再一张 | 16 GiB |
| `Attention.forward` L860 `attention_mask & selected_token_mask` | 再一张 | 16 GiB |

合计：**128K 要 25.2（常驻）+ 48（掩码）+ 26.2（线性项）= 99.4 GiB**，是 48 GB 卡的
两倍。这也解释了 §22 的实测斜率：`ce T=32768` 峰值 35.3 GiB 里，3 GiB 正是三张
1 GiB 的表。

### 24.2 MeZO 的直觉是对的，做不到的是 HF 的前向

MeZO 的成本模型确实几乎为零：2 次 `no_grad` 前向、不存激活、不建图、LoRA 只有
6.68 M（fp32 25.5 MiB）。**这条直觉成立的前提是「前向本身是 O(T)」**——引擎的
chunked prefill 就是（按字节预算切块 + KV 常驻），上游的 QSA 实现不是：它把
「每行选中哪些 token」编码成稠密 S² 表。

⇒ OOM 要归到**推理实现**，不是训练算法。同一件事在引擎里根本不存在——引擎的
`qwen_sparse_attention` 是按 top-k 索引 gather K/V，从头到尾没有 S² 的东西。

### 24.3 三条「看起来能省」的路，为什么都不划算

| 方案 | 省掉什么 | 代价 | 结论 |
|---|---|---|---|
| HF 只把掩码按 query 分块（`STRATA_QSA_QCHUNK=4096`） | 48 → 1.5 GiB | 线性项 26 GiB 照样超；且 mask 写入流量仍是 O(S²)（12 层 × 4 份 × 16 GiB ≈ 768 GiB ≈ 500 s/步） | 不可行 |
| HF 改分块 prefill + KV cache（mask 变 `(qc, kv)`） | 线性项 → O(qc) | 每 chunk 仍要重建 mask，总流量还是 4·S²；indexer 还要按 chunk 重池全前缀 | 仍卡在 HF 路线，且慢 |
| 引擎 scoring | 掩码归零、线性项归零 | 需要两块新能力（24.4） | **唯一** |

**注意**：第 2 条在内存上确实能跑通，慢才是它的死因。别把它当 128K 的答案。

### 24.4 交给引擎还缺哪两块

| 能力 | 现状 | 要做 | 规模 |
|---|---|---|---|
| teacher-forced logprobs | prefill **根本不调 lm_head**（`src/prefill/prefill.cpp` 全文件无 `output.weight`）；生成侧只报**采样到的那个** token（`src/program/generate.cpp` 的 `emit_lp_line`） | `lm_head_rows`：T 个位置一次 GEMM + vocab 分块流式 `logsumexp` + 取目标 token 的 logprob；端点 `/v1/score` | 中：一个 GEMM 变体 + 端点 + 测试 |
| adapter ±ε 热换 | `--lora` 只在启动时读（`src/kernels/lora_adapters.cpp`）；请求里只能 `sampling.lora:false` 关掉（`serve/server.py:1478`） | 两张表常驻（15.7 MB × 2）+ 请求级选择 `lora: "+" / "-"` | 小 |
| 128K 上下文 | **已有**：分块 prefill 按字节预算自动选 chunk；专家按需流式 | — | 无 |

**不能**用现成的 `core::lm_head` 做 scoring：它是单行 GEMV（`src/core/layer.cpp:1406`），
每 token 把 `output.weight` 整个读一遍 ≈ 318 MB；128K 个位置就是 **40 TB** 权重流量
（≈4 小时）。必须换成按位置分块的 GEMM。

显存估算（`--vram 16`）：专家 16 + 稠密 ~9 + KV 3.1（12 层 × 2 kv heads × 256 × 2 B，
int8 减半）+ scoring 2 + chunk 工作区 2 ≈ **32 GiB**，在 45.6 预算内。
步时：prefill 465 tok/s ⇒ 128K 单次前向 ~280 s，MeZO 两步 ~9.5 min + scoring 2~5 s。

### 24.5 分支切分（单一路线 + 旧路线冻结）

| 分支 | 前向 | 上下文 | worktree | 状态 |
|---|---|---|---|---|
| `feat/train-mezo` | 引擎 `/v1/score` + tokenizer，**不 import transformers 模型类** | 128K | `Strata-mezo` | 唯一长期路线 |
| `feat/train-backforward` | HF `Qwen4ExpTextLMHeadModel`（1 前向 + 1 反传） | 冻结在 ≤32K | `Strata-prtrain` | A/B 与数值对照，不再打 128K 补丁 |

两条分支的基线不同，是有意的：
- `feat/train-mezo` 切自 `feat/lora-q36`（= 上游 + 28 个提交）。**上游 main 里没有任何
  LoRA 代码**（`git ls-tree upstream/main | grep -i lora` 为空），`lora_adapters.cpp` /
  `lora.hpp` / `--lora` 都在这个分支上；训练只用 LoRA，所以只能切在它上面。等
  `lora-q36` 再 rebase 到上游最新（v0.2.8, 870aedd9）后，这条分支再跟一次。
- `feat/train-backforward` 切自 fork 的 `feat/train_support`（PR #3 的 head 仍是
  `origin/feat/train_support`，改名只发生在本地）。

**不与别人在开发的 LoRA 分支冲突**：`pr-3267-lora`（上游 PR #3267，LoRA up/down kernel
重叠 + PDL）动的是 `lora_apply` 的性能；本路线只加两件事——`lm_head_rows`（scoring）
与 adapter 表的运行期选择——**不碰 LoRA kernel 本体**。

顺带收益：砍掉 HF 前向同时干掉四类 Windows 缺口——S² 掩码、GDN kernel 静默回落、
`PYTHONUTF8`、`causal_conv1d` ABI 坏。
共享层（数据、MeZO 步数学、reward、`mem_guard`）留在 `train_core.py`，两边都 import；
推理只走一个 `scorer` 接口，两条路线的差异只在这一个文件里。

### 24.6 本轮随之加固的一处（`qsa_fast.row_ends`）

`_fast` 原用 `torch.equal(V, torch.tril(torch.ones_like(V)))` 校验「恰好下三角」：
`ones_like` + `tril` 各造一份 S² 临时张量（S=32K 各 1 GiB，S=128K 各 16 GiB），
是 §22 那 35.3 GiB 里的一部分，也让 128K 在 indexer 之前就 OOM。改成一次 row-sum
（只读 S² 字节、只写 S 个数），并顺手把判据放宽到「行是 `[0, p+j]` 的前缀」——
带 KV cache 的分块 prefill 就是这个形状，将来 HF 侧要做 chunked prefill 不必再改。
`test_train.QSAFast` 用缩小版 config 与上游逐 query 循环逐元素对齐（含位移版）。
