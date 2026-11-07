# 全量训练评测流程（v5.3，2026-11-01）

> v5 按 11-01 第四次反馈重写：**满上下文 = 替用户算出来的极限，不看用户用量**。
> `极限(方案, 设备, pack)` 由上界 128K（模型上下文）与该设备显存预算共同决定，用 `budget.py` 解；
> **只算当前这台设备**（`budget.py` 自动读本机显存/内存），不做设备矩阵——别人换卡在自己机器上跑一次即可。
> B 阶段 = **先解算、再用两点实测校验公式**，跑出来的数就是「这台机器上这个方案能训多长」。
> 其余沿用 v3/v4：训练集与评估集缩小、先重扫 lr × 步数、课程式「身份 → 数据集」、只测 WSL。
>
> **v5.3（本次）= 缺的代码全部补齐**（§10 逐项落地，82 项单测过）：极限阶梯带防挂死与续跑、
> `HttpEngine`、统一 `evalset.py`、数据缩小与语料打包、降速判定器、编排入口、三条 HF 线的
> `--init-adapter`。同时把数据规模从"按字符估"改成**用真 tokenizer 实测**（原来把 `dpo_math`
> 单条估成 4500 tok，实测 1596；身份集 26 条实测 726 tok）。
>
> **v5.2 只改四处，都是把口径写死、去掉争论**：① 明确「loss 置零 vs 完全跳过」的结论——
> **完全跳过严格更省**，且它们的适用范围只有 SFT/DPO（有 token 级对错），GRPO 无标签不参与；
> ② 数学 12 题固定一套跑一遍，不按 pass_rate 分层挑；③ pack 不换，超长直接 `skipped: budget`，
> 要改改评测集；④ DPO 的 rejected 必须 on-policy 生成，生成合理的**连正向训练都不做**。
> 两条本轮不做的（延展前缀省算力、on-policy DPO 构造）记在 §10 第 10/11 条。

## 0. 口径（先把三个词钉死）

**「一步」= 一次参数更新**。但**一次更新要几次前向**，各方案不同——这里之前写错过（把 `--accum 2` 的累积当成算法要求），现在按代码逐一核对：

| 脚本 | 一次更新要几次前向 | 为什么是这个数 |
|---|---|---|
| `sft.py` | **`--accum × 1`** | SFT 有标准答案，teacher forcing **一次前向就够**。`--accum` 只是「攒几条再更新」（吞吐/梯度平滑），不是算法要求，默认已从 2 改成 **1** |
| `hf_ref.py`（HF 零阶） | `2 × dirs × rows` | **零阶差分必须 `f(θ+εz)` 与 `f(θ−εz)` 各一次**，少一次就没有梯度 |
| `train_engine_mezo.py` | `2 × dirs × rows` 次 `SCORE` **+ 同样次数的换表** | 同上；实测每步是 **2 次 `LORA`**（换到 +ε 打一次、换到 −ε 打一次），不是 1 次 |
| `grpo.py` | K 条轨迹各自生成 + 各 1 次前向 | group 内多条样本才谈得上相对优势 |

**只有零阶（MeZO）天然是两次**，因为差分估计的定义就是两侧取样。SFT 的"两次"没有理由；DPO 的两次是**数据**决定的（chosen / rejected 一对），也不是算法要求。

所以「N 步」到底看了几遍数据：身份 8 条、`accum=1` ⇒ 一步 = 一条样本，`300 步 = 37.5 遍`。旧设置「300 步 + accum=2」在 26 条数据上是 23 遍——这才是 v1 最不合理的地方。

**只训"惊讶"的 token（已默认开启）。** `--select-tokens excess` 拿训练起点策略当参照，只给 ref CE > 阈值的答案 token 传 loss。
它属于**有 token 级对错标签**的监督类目标，也就是**只对 SFT / DPO 生效**，由算法自动决定，不是全局开关：

| 情形 | 现状 | 算力 |
|---|---|---|
| 整条答案模型都已经会 | **整条跳过**（`continue`，前向都不做） | 完全省 |
| 部分 token 不惊讶 | 前向跑全序列，反向仍走整张图，那些 token 只是 loss 置零 | **一点都不省** |
| 按 chunk 整块跳过（仅限**开头连续段**） | ❌ 未实现（见 §10 第 10 条） | 能省该段的反向（≈2/3），且不改变语义 |

**「loss 置零」和「完全跳过」谁省时间：完全跳过，而且差别是整块的。** 置零只是改了分子，前向照样跑全序列，
反向照样走整张图（梯度为 0 不代表不计算），墙钟几乎不变。所以这两者不是取舍关系——能跳就跳，问题只剩「跳得动吗」。

**跳不动的地方**：中间段的 chunk 不能跳。GDN 是递推的，跳过一块，后面所有块的状态就错了；要跳只能把它
当 no_grad 只递推状态（不建图），那就等于在每一处跳跃点截断 BPTT——**这是改变训练语义，不是免费的优化**。
当前只有两个位置能真正跳：① 整条都不惊讶（已经在跳）；② 把**开头连续的**不惊讶块并进 frozen-prefix（前缀本来就
是按 GDN chunk 对齐切的，基建已在），这样那一段从「前向+反向」降成「只前向」，约省该段的 2/3。
方案 ② 不影响语义（那些 token 本来 loss 就是 0），但要多一层「按样本动态定切点」，**本轮先不做，记在 §10 第 10 条**。

**量级也要说清**：身份数据答案只有 50–130 tok，开头「我是…」也就 3–8 tok，方案 ② 在这份数据上省不到 5%，
做了不值；真正有账可算的是 D 阶段的长块（8192 tok、开头大段不惊讶），但那一阶段每方案只有 16 步。
所以它排在"值得做但不该现在做"，不是"被否决"。

**GRPO 没有这一步，也不该有。** 它只拿整组轨迹的**平均分**做相对归一化（同一 prompt 下 K 条互比），
token 级没有对错标签，所以既无法判断哪些 token 该跳，跳掉还会破坏组内比较的前提。它省算力的位置在别处：
短生成 + 只对高分轨迹回放前向。

**「tok/s」= 该步实际训练文本的 token 数 / 步时墙钟。** 分子只数真实数据（该步 prompt+answer 的
实际 token 数），**padding、合成 token、被 loss mask 掉的部分一律不进分子**；分母是该步完整墙钟
（含前向、反向/换表、优化器）。所以：

- 短样本不因凑成固定长度而虚增 tok/s（身份一条 50~130 tok，就按 50~130 算）；
- 满上下文档只按被截断/实际喂进去的真实 token 算，不按 `seqlen` 算；
- **合成 token 只允许出现在「显存 / 跳过」表，禁止出现在任何速度表里**（见 §3 的「真实 / 合成」两档）；
- decode / prefill 分开报，不混用同一个数。

**「HF」不是 HuggingFace 下载**。本项目的两条计算栈：

| 栈 | 是什么 | 权重来源 | 能做什么 |
|---|---|---|---|
| **HF 路径** | 本地 `transformers` 模块（`loader_qwen4.py` 装载 + `nf4_fused.py` 融合算子），走 torch autograd | `Strata-data\models\swift-IQ3_XXS\*.gguf` + `experts_nf4.bin` | 前向**和反向**、单进程、~34.5 GiB / 加载 ~450 s |
| **引擎路径** | `strata.exe`（C++/CUDA，GGUF + 专家 pack） | 同一批文件 | 只前向：`/v1/engine/score`、`/v1/engine/lora`（无按 ids 生成的端点，在线形态 `gen` 不可用）；~42.7 GiB / 加载 ~19 s |

平台 × 方案可用性（这就是"算法平台都要测"的矩阵轴）：

| 方案 | 引擎 | HF |
|---|---|---|
| ce_bwd / dpo_bwd | ✗（无反向） | ✓ |
| ce_mezo / dpo_mezo | ✓ `/v1/engine/score` | ✓ |
| grpo_mezo | ✓ | ✓ |
| grpo_bwd | 混合（rollout 走引擎，反传在 HF） | ✓ |

**只测 WSL**：Windows 侧这次不编不跑（`hc_persist_parity.cpp` 的 `setenv` 在 MSVC 下本就不通，与本轮无关）。

**加载耗时按优化后的口径算**：`perf/pack-load-accel`（`2a640fc9`）的 `posix_fadvise(DONTNEED)` 已合进
`feat/train-backforward` 的 `loader_qwen4.py`（每块读完立即丢 page cache，Linux-only，`STRATA_PACK_FADVISE=0` 可关）。
实测 `_ensure` 333 s → 101 s、**整模型加载 454.6 s → 216.9 s**。不应用这个改动，下面所有 HF 档的时间都要 ×2。
它与引擎侧 `src/core/expert_source.cpp` 用的是同一个 API 同一边界——上游引擎早已这么做，所以这份训练侧补丁
只能留在本地分支，**评估时必须显式带上**。

## 1. 数据（缩小后，**全部为实测 token 数**）

下表的 token 不是按字符估的，是用 pack 的真 tokenizer（`prep_pack.py` 走
`strata_tokenizer`）分出来的——之前按「字符数 × 系数」估过一次，把 `dpo_math` 单条估成了
4500 tok，实测只有 **1596**，差了近 3 倍。这类数字只能量，不能估。

| 集合 | 现在 | **改为** | 实测规模（单条最长 / 中位 / 合计） |
|---|---|---|---|
| 身份训练 `data_developer.jsonl` | 26 条 / **726 tok**（实测） | `data_identity8.jsonl`：**8 条**（4 个事实 × 中文、英文各一） | 44 / 37 / **275 tok/遍** |
| 评估·快速门 | — | **3 id + 3 cap**（每 5 步跑） | 秒级（一次 6 条 `/v1/engine/score`） |
| 评估·全量 `prompts_identity` + `prompts_holdout` | 7 + 12 条 | 不动（只在终点跑一次） | 32 + 141 = **173 tok** |
| 数学 `math_prompts.jsonl` | 36 条 | `math_12.jsonl`：**12 题**（4 train / 4 dev / 4 holdout，固定 seed，按运算分桶轮转） | 36 / 21 / **285 tok/遍** |
| 长样本（B/D 用） | `dpo_math.jsonl` 4 条 | 取 `chosen` 段当 gold 长答案；探极限时按 eos 打包到目标 T | **1,596 / 941 / 3,832 tok** |
| `hard_train.jsonl` | 4 条 | 备用（数学长解） | 251 / 177 / 645 tok |
| **全部训练文本合计** | — | `prep_pack.py` 打成一条流 | **4,416 tok**（`_discard/corpus.npy`） |
| DPO | `data_dpo.jsonl` 10 对 | 本轮不进主矩阵；且这 10 对的 `rejected` 是**人工构造的，不合规**（要 on-policy 生成，见 §10 第 11 条），仅用于验机制，不能当结论 | — |

**由此得到两个必须写进报告的常数**：

- 身份一步只有 **~34 tok**（275 / 8）。所以身份阶段的 `tok/s` 会很小（34 tok ÷ 1.5 s ≈ 23 tok/s），
  这不是"慢"，是**分子就只有这么多真实 token**；要比较吞吐只能比同一个方案的相对值。
- 最长单条真实文本 **1,596 tok**。所以「满上下文」在这个任务上**不存在真实档位**——超过 1596 就是
  打包重复，`prep_pack.py --check-t` 会直接算出遍数（131072 需 30 遍）。重复不影响算子与显存，
  但 tok/s 不能说成「真实语料吞吐」。

> **预训练块整条不要。** `/root/pretrain_blocks.npy`（22269 × 2048，`prep_pretrain.py` 把本仓库文本
> 按 eos 切块）从流程里**彻底移除**：它不是训练目标数据，与身份/数学任务无关，它的耗时也不该出现在
> 任何一格。`sft.py` / `hf_ref.py` 的 `--pretrain-data` 在本轮**不使用**。
>
> **「我们自己的数据按 eos 打包到目标 T」是另一回事**：它算真实训练 token（只是重复）。
> 这不需要当成问题讨论——**满上下文探测的目的是算出「这台设备 + 这个算法能训多长」，不是宣称
> 「我们有 128K 的真语料」**。长度由 `budget.py` 算出来，打包只负责把输入撑到那个长度；
> 报告里标 `repeat: n` 说明来自打包即可，不设什么"重复上限"。

关键事实：**身份训练集总共只有 726 token，26 条里答案还高度重复**。所以"训练次数"本来就该按遍数算，不该按步数猜——旧设置的 300 步等于让模型把 4 个事实背诵 11 遍以上。

> **不要 merge `origin/feat/train_support`。** 它里面的 `data_developer.jsonl` 是**错的**
> （"我由 {dev} 开发和维护，是 Strata 项目的一部分"——正是那把身份答成"xiezhipeng 维护的
> xiezhipeng 项目"的那批数据）。本地工作区已改成正确事实（通义千问 / 阿里通义实验室 /
> 底座 qwen3.8-flash-next / 由 {dev} 经 strata 微调）。要取 `perf/pack-load-accel` 只取
> `2a640fc9` 那一个提交（它只动 `loader_qwen4.py`），不要整支合并。

## 2. 阶段 A —— 基座基准分（≈5 min）

| 子步 | 内容 |
|---|---|
| A1 | 引擎加载 → `/v1/engine/score` 打快速门(6) + 全量(19) → `base_id_mean` / `base_cap_ce`（per item + mean） |
| A2 | `math_probe.py --preset mix --n 12 --train-n 4 --k 4` → 每题 pass_rate。**固定这 12 题跑一遍就够**，不按 pass_rate 分层挑——数学题是**能力探针**，与算法选择无关，题集变了前后就不可比 |
| A3 | 生成体检：1 条 cap 探针贪心，记 distinct-3gram / EOS / decode tok/s |
| A4 | HF 一致性 `ab_score.py`——**只在换构建后做**，本轮沿用已验的 0.0075 nats，跳过（省 12 min） |

## 3. 阶段 B —— 每方案的可训练长度极限（算，不爬梯子）

**定义（唯一口径，后文通用）：**

```
max_train_len(方案, 设备, pack) = max{ T : peak(T) ≤ VRAM × 0.95 }，上界 = 128K（模型上下文）
```

「满上下文」就是**这个数**。它既不是写死的 128K，也不是用户的用量分位，也不是 8K/16K/32K 的档位。它同时是 C/D 的目标档位、也是报告标题里那个数。

三个输入决定它：**方案**（决定激活与是否反传）、**设备**（VRAM；宿主 RAM 只决定 pack 放不放得下）、**pack 档位**（决定常驻权重）。

**只算当前这台设备，不做设备矩阵。** 不同用户的设备与 pack 组合是各自的现实问题，我们在这里替他们猜一组 24/32/80 GiB
的格子既不准也没用；正确做法是 `budget.py` 默认读**本机**显存与内存（`--vram` 只在想试算时显式给），
用户在自己的机器上跑一次就得到自己的极限。**如果谁在别的设备上算出来的值与实测不符，那是探测算法要修，不是要加一列预测。**

### B1 · 解算（`budget.py`，秒级，先算不先试）

```
peak(T) = 常驻权重 + KV(T, kv_mode) + 激活(T, 方案) + staging(方案) + ckpt 系数
```

- **引擎侧**：峰值**不能**用 `nvidia-smi`（专家缓存会把剩余显存吃满，8K 和 128K 都显示 48.1 GiB，
  拿它当约束会得出「8K 就满了」的错误结论）。真约束是 `常驻权重 + KV(T) + 1.2 GiB 工作区`；
  KV 系数由 `bench_engine_128k.json` 的两个实测点标定：**fp16 27.5 KiB/token、int8 13.75**（正好一半），
  4-bit 按字节缩比。
- **HF 侧**：用 `mem_guard` 的拟合式（6 个实测点，残差 ≤0.6）。MeZO 斜率 2.0e-4 GiB/token、
  反传 1.94e-3（ckpt），差 9.7×——这是算法性的（ckpt 边界 + MoE 中间激活），不是 kernel 能抹平的。
- **`rows` 必须如实传**：DPO 的 chosen/rejected 要同时驻留（`rows=2`）；GRPO 的 K 条轨迹是否同批驻留
  是实现选择，不是算法性质，所以两种都算一行（串行 = 等于基础模式，同批 = `rows=K`）。
  `max_seq_len` 的 MeZO 反解原先漏了 `0.8·(rows−1)·√(T/8192)` 这一项，K=4 时会把上限高估到 55808
  （代回估计值 51.8 GiB > 45.6 预算），已改成与 `estimate_peak_gib` 同口径的二分。

本机（48 GiB / 58 GiB、125B IQ3_XXS，引擎常驻 24.0 GiB、HF BASE 25.2 GiB）：

| 方案 | 极限 T | KV | 该点峰值 | 卡在哪 |
|---|---|---|---|---|
| engine（ce/dpo/grpo mezo 共用） | **131072** | int8 | 26.9 GiB | **模型上下文**（VRAM 还剩 ~19 GiB） |
| hf-ce_mezo | 55808 | n/a | 45.6 GiB | 显存预算 |
| hf-grpo_mezo（K=4 同批） | 44032 | n/a | 45.5 GiB | 显存预算 |
| hf-ce_bwd（ckpt） | 9984 | n/a | 45.3 GiB | 显存预算（实测 8K 过、16K 挂死，吻合） |
| hf-dpo_bwd（ckpt, rows=2） | 5120 | n/a | 45.2 GiB | 显存预算 |
| hf-grpo_bwd（ckpt, K=4） | 2560 | n/a | 44.6 GiB | 显存预算 |
| hf-ce_bwd（走 refetch） | 0 | n/a | — | 25.2 + 26 staging > 45.6，本机放不下 |

**设备矩阵不做了**：上面的表就是这台机器的答案。用户换卡时在自己机器上跑 `python bench/budget.py`
（默认读本机显存与内存）即可；`--vram 24 --max-pack-gib` 这类反解仅用于「我的卡放不下，该换多大的 pack」这种自检。

### B2 · 校验（只测公式给的那一格，不爬梯子）

公式是拟合出来的，**必须验证**——否则整个 B 阶段就是拍脑袋。校验点不取 8/16/32 的档位，而取：

| 点 | 用途 |
|---|---|
| `T = T_model`（公式给的极限那一格） | 最关键：极限本身能不能跑完一步 |
| `T = T_model / 2` | 验证公式在中段没有非线性塌陷 |
| 若 `T_model = 128K`（被模型上下文封顶） | 只跑 128K 一格；它已实测过（63.8 s / 48.10 GiB），本轮作对照 |

判据：实测通过的最大 T ≥ `0.85 × T_model` ⇒ 公式可信，其余格子直接用算的；差 >15% 或该格 OOM/挂死 ⇒
**先修公式，再信它**，并按二分下探一格（每方案最多 2 次）。

```
# 引擎侧（只前向：MeZO 的 ±ε 两次 /v1/engine/score）
# bench_engine_mezo.py 没有 --lora 参数，adapter 走 --extra；--corpus 要喂真文本
python tools/bench_engine_mezo.py --seqlen 65536,131072 \
  --corpus _discard/ladder_real.txt --extra "--lora <adapter.gguf>" \
  --determinism 2 --gen 0 --out bench_limits_engine.json

# HF 侧（前向 + 反向都在 torch）：长度从 budget.py --json 里**按方案**取，不手写、不并公共阶梯
python bench/bench_fullctx.py --gguf <shard1> <shard2> --pack <experts_nf4.bin> --tokenizer <tok> \
  --from-budget bench_budget.json --algos ce,dpo,grpo --modes mezo,bwd \
  --group 2 --reps 1 --warmup 0 --on-oom skip --per-cell-timeout auto \
  --tokens-npy _discard/corpus.npy --out bench_limits.json
```

`--from-budget` 是**按方案分格**：`ce_mezo` 只测 27648/55808、`grpo_bwd` 只测 1024/2560（共 12 格）。
早先版本把所有 profile 的 {T/2,T} 并成一条公共阶梯、每个组合挨个爬，那是 6×11=66 格——多一个数量级，
而且爬到的档位与「该方案自己的极限」根本不是一回事。显式给 `--seqlens` 时才恢复「每个组合都测这一串」。

**每格必记的六个数字**（缺一作废）：

| 字段 | 含义 |
|---|---|
| `t_step_s` | 该格一步的完整墙钟（前向 + 反向/换表 + 优化器） |
| `real_tokens` / `tok_s` | 该步真实训练 token 数与由此得到的 tok/s（合成档留空） |
| `peak_vram_gib` | 设备峰值（`torch.cuda.max_memory_allocated` 与 `nvidia-smi` 双记） |
| `peak_rss_gib` | 宿主峰值（pack staging + pinned 都在这里，`mem_guard` 之外唯一的 RAM 账） |
| `kv_mode` | 该档实际 KV 档位（fp16 / int8 / k8v4 / q4_0）——`--kv auto` 的展开结果 |
| `exit` | `ok` / `skipped:budget`（预算公式先拦）/ `skipped:oom`（真 OOM）/ `skipped:hang`（超时）/ `skipped:signal` |

**「该方案的满上下文」= B1 解出的那个极限 T**（实测通过后即为定论），它才是 C/D 用的档位、也才是报告标题里那个数。

#### B2 实测（2026-11-01，WSL，48 GiB，`hf-*` 侧，12 格按方案分格）

| 方案 | T | 每步 token | 步时 s | tok/s | 峰值 GiB | RSS GiB | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| ce/mezo | 27648 | 27,648 | 137.53 | 201.0 | 33.0 | 49.0 | ok |
| ce/mezo | 55808 | 55,808 | — | — | — | 49.3 | **oom** |
| dpo/mezo | 27648 | 55,296 | 464.04 | 119.2 | 39.6 | 49.1 | ok |
| dpo/mezo | 55808 | — | — | — | — | — | budget（先被公式拦） |
| grpo/mezo | 22016 | 44,032 | 180.50 | 244.0 | 36.5 | 49.5 | ok |
| grpo/mezo | 44032 | 88,064 | — | — | — | 49.5 | **oom** |
| ce/bwd | 4608 | 4,608 | 81.89 | 56.3 | 35.0 | 50.1 | ok |
| ce/bwd | 9728 | 9,728 | **914.71** | **10.6** | 45.0 | 50.3 | ok（贴着预算，降速真实） |
| dpo/bwd | 2560 | 5,120 | 49.83 | 102.7 | 34.2 | 50.3 | ok |
| dpo/bwd | 5120 | 10,240 | 53.87 | 190.1 | 41.5 | 50.3 | ok |
| grpo/bwd | 1024 | 2,048 | 26.11 | 78.4 | 29.6 | 50.3 | ok |
| grpo/bwd | 2560 | 5,120 | 28.42 | 180.2 | 34.2 | 50.3 | ok |

判据（`judge_limits.py`）：**bwd 三条全 PASS**（0.974 / 1.0 / 1.0），**mezo 三条全 FAIL**——
实测极限只有公式的 0.50 / 0.50（`dpo/mezo` 0.21，它回落到了 ce 那一行）。

根因（从 OOM 的申请量直接读出来，不用猜）：T=55808 那笔失败申请是 **24,916,262,912 B = 55,808² × 8 B**，
即**一块 `[T,T]` 的掩码/分数矩阵**，与 KV、logits 都无关。T=27648 时它只有 6.2 GB，塞得下，所以低档看不出问题；
bwd 走到 9728（45.0 GiB）也没有这笔申请 ⇒ 只有 mezo 的 no_grad 路径把它实体化了。`mem_guard` 的反解里**没有 T² 项**，
于是 mezo 三档的极限被系统性高估约 2×。两条修法（择一，别都做）：
**(a) 让 mezo 那条路不实体化 `[T,T]`**（走 sdpa/flash、或分块算掩码）——极限不降，但改的是算子路径，要重新对拍数值；
**(b) 把 `T² × 8 B` 加进 `mem_guard` 的 mezo 反解**——数字立刻可信，但承认 mezo 的可训长度只有 bwd 的 ~2.8×（27648 vs 9728）。

**「极限」的判定 = 连续满足三条的最大 T**：

1. **一步完成**——没有 kill、没有静默挂死；
2. **峰值 ≤ 该设备的预算**（B1 算出的 VRAM − 1.5 GiB；宿主 RSS 另记，不折算）；
3. **信号高于噪声**——MeZO 的 `f(+ε) − f(−ε)` ≥ 3 × 同 T 的重复性极差（128K 实测极差 0.169 nats ⇒ 阈值 0.5 nats）。

第 3 条是这一阶段最容易漏的：**跑得动 ≠ 训得动**。短序列已有反例（T=3 极差 0.2~0.3 nats，与 token 数不成比例），长档要防的正是"能跑但信号淹在噪声里"。

**防挂死**：`ce_bwd@16K` 有前科——**既不报错也不退出**（`free:0`、GPU 100%、日志零行）。每格 wall-clock 上限 = `3 × max(60 s, 上一档步时 × 2.5)`，超时 kill 记 `skipped:hang`，重启进程从下一档继续。

### 数据：三档来源，绝不混用

| 档 | 数据 | 用在哪 | tok/s 可否上报 |
|---|---|---|---|
| **真实** | 我们自己的数据按 eos 打包到目标 T（身份 8 条 + 数学 12 题 + `dpo_math` 长答案），标 `repeat: n` | 校验点（`T_model/2`、`T_model`） | ✅ |
| **合成** | `bench_fullctx.py` 的 `synth()` 随机 token | 只在预算算术存疑时顶一次 | ❌ 标 `synthetic: true` |

长度**由 B1 的解算决定**，打包只负责把输入撑到那个长度——所以本阶段既不承诺"有 128K 真语料"，
也不需要讨论重复几遍算合理。

**公式 vs 已实测（这是 B2 的对照基线，不是结论）**：

| 方案 | 平台 | 已实测 | 公式给的值 | 相容性 |
|---|---|---|---|---|
| ce_mezo | 引擎 | 8K 4.39 s / 48.09 GiB；128K 63.8 s / 48.10 GiB | 131072（模型上下文封顶） | 相容：128K 就是模型上限 |
| ce_mezo | HF | 32K 35.3 GiB 过；64K OOM（外推 50.8） | 55808（45.6 GiB） | 相容：落在 32K 与 64K 之间 |
| ce_bwd | HF | 8K 42.0 GiB 过；16K 挂死 | 9984 | 相容：8K 过、16K 挂死 |
| dpo_bwd | HF | 8192 崩在 native 栈 | 5120（rows=2） | **相符**：8192 > 5120，本来就该崩 |
| grpo_* | 引擎 / HF | 只测过 ≤8192 | ce_mezo 44032 / grpo_bwd 2560 | 待验 |

> `dpo_bwd` 这一行上一版被我读反了：5120 是**上界**（超过就过不了），不是"能过的下界"。
> 8192 > 5120 ⇒ 崩掉正是公式预测的结果，所以它不是反例，而是公式可信的一条证据。
> 真正待验的是**上界以内**：5120 本身、以及 4096 这一档能不能跑完一步。B2 的 dpo 校验点取
> `T_model` 与 `T_model/2`，并同时记 `exit` 与实际峰值——`exit` 只写「崩了」等于没信息。

> **本阶段（B1/B2/B3）产出的权重一律丢弃、不保存。** 一步更新没有任何训练意义，留下来只会污染 E 阶段的矩阵。
> 落盘口径：`--out` 指向同一份临时文件（如 `_discard/adapter_tmp.npz`）反复覆盖，`--work` 同理，
> 整轮结束删除 `_discard/`；**E 阶段只接受 C/D 明确登记的 8 个 adapter**，任何 `_discard/` 下的路径
> 进矩阵直接报错。B3 的 lr 扫描同样丢弃（只保留曲线 json，不保留 adapter）。

**降速判据**（单步标定只能看跨档趋势，所以这条放在这里而不是逐步）：
`tok/s` 随 T 应单调不降；若 T×4 掉 >15% 且 `peak_vram < 预算×0.9` → 判 bug，停，不进 C/D。

## 4. 阶段 B3 —— lr × 步数重扫（≈15 min）

身份数据修对之后（原来是"我是 xiezhipeng 维护的 xiezhipeng 项目的一部分"这种错事实），旧的 lr 3e-6 / 300 步只是沿袭，必须重定。

用最便宜的方案扫：引擎 MeZO，8 条身份数据，rank 8 / ε 1e-3 / dirs 1 / momentum 0。

| 轴 | 取值 |
|---|---|
| lr | 1e-6, 3e-6, 1e-5 |
| steps | 10, 20, 40, 80（每 5 步记一次快速门） |

判据：`id_mean` 到达平台（相邻两次评估 Δ < 0.05 nats）的最小步数 × 1.5 = 该方案后续下限；同时看 `cap_ce` 是否越 `--cap-tol`。

成本：8 组 × 平均 37 步 × ~1.5 s ≈ 8 min，加加载与回读 ≈ 15 min。产出 `sweep_lr_steps.json`，**C/D 全部按这里选出的 (lr, steps) 定 token 预算**。

历史警号：momentum 0.9 在 lr 3e-6 会发散（|theta| 19.6→550），本轮不进 momentum 轴。

## 5. 阶段 C —— 身份训练 × 4 方案（课程第一段，≈45 min）

**token 预算对齐**：先定 `tokens_seen = 8 遍 × 500 = 4k tok`（≈ 每条样本看 8 次），各方案的实际步数由自己的每步 token 数反推（下表就是这个除法），**两列都进报告**。

8 条数据 × 8 遍 = **64 条样本 = 64 步**（`accum=1`，一条样本一次更新），四个方案都按这 64 步对齐：

| 方案 | 平台 | 实际步数 | 一次更新的前向次数 | 步时 | 训练耗时 |
|---|---|---|---|---|---|
| 反传 SFT（frozen-prefix + select-tokens excess） | HF | 64 | 1 | ~3 s | 3.2 min |
| HF MeZO | HF | 64 | 2（±ε） | ~5 s | 5.3 min |
| 引擎 MeZO（HTTP） | 引擎 | 64 | 2 次 SCORE + 2 次 LORA | ~1.5 s | 1.6 min |
| GRPO（group 4） | 混合 | 16（每步 4 条样本） | 4 条轨迹各自生成 + 前向 | ~10 s | 2.7 min |

> `--select-tokens excess` 会跳过整条都已会的样本（`n_skip`），实际更新步数可能少于 64——
> 报告里两列并列（`planned_steps` / `updated_steps`），不要用一个数糊过去。

每次跑必交（缺一项作废）：

- **速度**：训练 `tok/s`（= 该步**真实训练 token** / step_wall，每步 + 稳态中位；padding 与合成 token 不计）、生成 `tok/s`（decode / prefill 分列）、用时四段（`load / prep / step / eval`）。
- **规模**：`opt_steps`、`tokens_seen`、`epochs_over_dataset`（= tokens_seen / 500，即把 8 条数据看了几遍）。
- **效果（训练部分）**：训练题 CE 前→后、`logp(word|probe)` 前→后、5 条贪心抽样原文。
- **效果（未训练部分）**：全量评估的 `Δcap_ce`、生成体检、数学 holdout `pass_rate` Δ。
- **稳定性**：`peak_gib` vs 预算、RSS、skip 记录。
- **算法专属**：MeZO → `|mean|/std`、每步信号 nats（`_snr.py`）；GRPO → `reward_std` / `adv0` / `uniq`。

## 6. 阶段 D —— 数据集训练 × 4 方案（课程第二段，≈55–65 min）

**在 C 的 adapter 上继续训**（`train_engine_mezo.py --resume` / `sft.py` 从已存 adapter 起），这是"身份 → 数据集"的课程式：既省一次收敛，又能顺带量出「加数据会不会把身份冲掉」。

数据：真实训练文本按 eos 打包到目标 T（数学 12 题 + `dpo_math` 4 条长答案 + 身份 8 条；>4700 tok 开始重复，记 `repeat: n`）。
**档位取 B1/B2 的结论**：`T = min(该方案的极限 T, 8192)`，四方案各按自己的极限定，不强行统一。
这里的 8192 是**成本上限**（整轮跑得完），必须在报告里标 `cap_reason: cost`，
**不能伪装成"算出来的极限"**——极限是 `budget.py` 给的数，成本上限是我们自己砍的。

`tokens_seen = 4 遍 × T`，分子按真实 token 计（打包重复的 token 仍算真实 token，标 `repeat`）。

数据打包成 **4 个 T 长的块**，4 遍 ⇒ **16 步**（每块一遍一步，`accum=1`）：

| 方案 | 实际步数 | 真实 token | 步时（T=8192） | 耗时 |
|---|---|---|---|---|
| 反传 SFT（ckpt） | 16 | 4×T | ~40 s | 10.7 min |
| HF MeZO | 16 | 4×T | ~32 s | 8.5 min |
| 引擎 MeZO（HTTP） | 16 | 4×T | **4.39 s @8K（实测）** | 1.2 min |
| GRPO（max-seq = T） | 4（每步 4 块） | 4×T + 生成 | ~120 s | 8 min |

**D 的 adapter 保留**（已定，不再讨论）：rank 8 × 48 层的 adapter 实测 **13,376,160 B ≈ 12.8 MiB**
（`lora_engine_mezo*.npz` 就是 13 MB），4 个方案 × 13 MB ≈ **52 MB**，加上 C 的 4 个总共 8 个 ≈ 104 MB。
存下来的理由不只是"不大"：E 阶段要拿它量「身份 → 数据集」的互相干扰，事后还能单独重测。

**外加一格「极限档位训练 demo」**：引擎 MeZO，在该方案的**极限 T**（B1 解出的值，本机 48 GiB 上是 128K）× 4 步（≈4.3 min，峰值 48.1 GiB）。目的不是收敛，而是验证 B2 的一步探测在**真实训练循环**里成立（连续 4 步不 OOM、步时不漂、信号仍在噪声之上）。这是唯一在最长的那个档位上真训的地方——换到 32 GiB 卡上它就会自动变成 32 GiB 卡解出的 128K（引擎那格仍被模型上下文封顶）。

> 反传在 32K 以上基本预算不过（ckpt 下权重活到反传结束 ≈60 GiB + staging 26 GiB），记 `skipped: budget`，**不做任何 workaround**。B 表已经给出每个方案的极限，D 不重复探边界。

## 7. 阶段 E —— 交叉评估（≈15 min）

权重 8 个进 E1 矩阵：`base` / `zero`(B=0) / C 的 4 个 / D 的 4 个（`zero` 与 `base` 应逐字相等，它是转换链路的自检项）。E2 的 2 个反跑 adapter 单独记，不进主矩阵。列：快速门(6) + 全量(19) + 数学 holdout。

**不进矩阵的东西**：B 系列产生的临时 adapter（`_discard/`）—— 它们是一步/扫描产物，没有任何训练意义。矩阵的输入只认 C/D 登记过的文件名。

- 引擎路径（主）：`lora_to_gguf.py` 转换 → `POST /v1/engine/lora` 换表 → 同进程连测。`zero` 必须逐字等于 `base`。
- HF 复核本轮**砍掉**（省 24 min + 2 次 450 s 加载），只留引擎一侧；差异超 0.05 nats 时再补。
- 输出 `Δid_mean` / `Δcap_ce`（遗忘）/ `Δpass_rate`（迁移）/ `decode tok/s`。

### E2 · 反跑课程（数据集 → 身份，≈12 min）

同一套引擎 MeZO，把顺序反过来跑一遍：**base → 数据集（4 遍）→ 身份（8 遍）**，得 2 个额外 adapter。
目的是量出**顺序效应**——同样两份数据，先身份后数据集 vs 先数据集后身份，最终的 `id_mean` 与 `pass_rate` 谁占优。

- 只用最便宜的那条链路（引擎 MeZO 1.5 s/步，共 80 步 ≈ 2 min 训练 + 加载/评估），不复制整套 4 方案；
- 判据：`Δid`（反跑最终 − 正跑最终）与 `Δpass_rate` 同号且都 > 阈值 ⇒ 课程顺序有影响，报告里必须写;
  若两者都 < 0.1 nats / 2% ⇒ 顺序无影响，一句话带过即可；
- 成本很低但结论很硬：如果反跑的身份保持更好，说明"先学能力再学身份"更抗遗忘，这条会直接影响真实训练顺序。

## 8. 判据

| 编号 | 判据 | 阈值 |
|---|---|---|
| G1 | 身份改善 | `id_mean` Δ ≥ +1.0 nats |
| G2 | 能力不退化 | `cap_ce` 漂移 ≤ 0.5 nats；>1.5 立即停并回滚 best |
| G3 | 生成不崩 | distinct-3gram ≥ 0.8×基线、EOS 命中 ≥ 0.5 |
| G4 | 无降速 | tok/s 随 T 单调不降；T×4 掉 >15% 且显存未顶 = bug |
| G5 | 无静默无效 | MeZO 的 `|mean|/std`、GRPO 的 `std>0 & uniq≥2` 必须打出 |
| G6 | 账目完整 | 训练/生成 tok/s、`opt_steps`+`tokens_seen`、前后效果（含未训练部分）、峰值显存，四项齐全 |
| G7 | **极限表完整** | 每个（方案×平台）给出「公式解 `T_model` / 实测通过 `T_measured` / 差值%」+ 该格六个数字（`t_step_s`/`real_tokens`/`tok_s`/`peak_vram`/`peak_rss`/`kv_mode`）；`skipped` 必带原因（budget/oom/hang/signal），不许留空值 |
| G8 | **极限可信** | 判为极限的那格必须同时满足：一步完成 + 峰值 ≤ 预算 + 信号 ≥ 3× 同 T 噪声极差（128K 阈值 0.5 nats） |
| G9 | **极限可复算** | 报告里每个 T 都必须能由 `budget.py --vram X --ram Y --weights-gib W` 复现（±15% 内）；复现不出来的 T 不许出现。**禁止手写档位、禁止引用用户用量**——极限是设备与算法的属性，不是使用习惯的属性 |

## 9. 时间预估（WSL，单卡）

| 阶段 | 内容 | 估时 |
|---|---|---|
| A | 基准分（引擎，缩小评估集） | 5 min |
| B1 | **解算**（`budget.py`，本机） | 秒级 |
| B2 | **校验**：每方案 2 个点（`T_model`、`T_model/2`），一格一步（含 HF 加载 3.6 min + 挂死超时余量） | 15–20 min |
| B3 | lr × 步数重扫（引擎） | 15 min |
| B2′ | **接口 vs 直连**（`bench_engine_iface.py`，128K 两通道各 2 轮，引擎启动 2 次） | 10–14 min |
| C | 身份 × 4 方案（含 3 次 HF 加载 11 min、终点全量评估 8 min） | 33–35 min |
| D | 数据集 × 4 方案（档位随各自极限，含最长档 ×4 步 demo） | 40–50 min |
| E1 | 交叉 8 权重 × 快速门（引擎）+ 转换 | 15 min |
| E2 | **反跑课程**（数据集 → 身份，引擎 MeZO） | 12 min |
| **合计** | | **≈ 2.2–2.5 h**（WSL 崩溃重跑 +30% → **3.2 h 内**） |

B2 的 15–20 min 拆开：引擎侧 2 点（64K、128K，其中 128K 有历史实测只作对照）≈ 3 min；HF MeZO 2 点 ≈ 6 min；
HF 反传 2 点 ≈ 6 min，其中 **`ce_bwd` 最大档的挂死超时单独占 ~5 min**（唯一不确定项）。
超出预算的格按公式 `skipped: budget`，不花时间——**这正是"算而不是爬"省下来的部分**。

对照 v1 的 7.5 h：B 由 reps×3 改成一格一步 −1.5 h、评估集缩小 −0.3 h、D 用 token 预算 −0.8 h、E 砍 HF 复核 −0.5 h、
C 用重扫后的步数 −0.4 h、加载走 fadvise −0.5 h；去掉仓库文本块 −0.2 h；B 改「解算 + 2 点校验」−0.3 h。

## 10. 代码状态（原「还缺的代码」全部落地）

清单里的每一项都已实现并带单测；`run_plan.py plan.stage-ab.json` 把 A + B 串成一条命令。
测试合计 **181 项**，分布在 12 个新测试文件里（全部不需要 GPU；含子进程验证「挂死→落盘→退出码 3」）：

| 件 | 文件 | 单测 | 关键约定 |
|---|---|---|---|
| **损失数学（单一来源）** | `Strata-prtrain/training/objectives.py` | `test_objectives.py` (30) | CE/DPO/GRPO 只定义一次，反传与零阶共用；`GrpoCfg` 五个开关；**TIS 分母恒为 token 数**（进分子分母会被约掉，等于没写） |
| **显存预算（锚点）** | `Strata-prtrain/training/mem_guard.py` | `test_mem_guard.py` (10) | MeZO 的峰值改为**按实测锚点分段插值**（27648→33.0 / 49152→47.5）。原解析式 `c·T²` 在 40K 以上低 3.9 GiB （把必爆判成刚够），换 c 又会让 32K 高估 1.8；另加 `route="engine"` 一套账（130816 token = 26.9 GiB，含 int8 KV 1.68） |
| **更新规则（单一接口）** | `Strata-prtrain/training/update_rules.py` | `test_update_rules.py` (17) | `rule.step(loss_fn)`：`BackwardRule`（autograd+AdamW）/ `MezoRule`（θ±ε 方向导数 + SGD）。锁三件事：同 seed 方向可复现、**跨参数方向必须不同**（旧实现每个参数都 `manual_seed` ⇒ 全参数同向）、`lr=0` 时 ±ε 必须精确复原 |
| 极限阶梯 + 防挂死 | `Strata-prtrain/training/bench_fullctx.py` | `test_bench_limits.py` (17) | **按方案分格**（`ce_mezo` 只测自己那两点，不并公共阶梯）；超时 → 记 `hang` → **落盘 → 以 3 退出**；`--resume` 从下一格续；`peak_rss` 后台采样；`synthetic` 标记 |
| 分数服务客户端 | `Strata-mezo/training/http_engine.py` | `test_http_engine.py` (14) | `score(ids, lora=±1)` / `lora(path)` / `gen(messages)`；回环**永不走代理**；429 重试、400 不重试；logprobs 长度不符直接抛 |
| **接口 vs 直连的账** | `Strata-mezo/tools/bench_engine_iface.py` | `test_bench_engine_iface.py` (36) | 两条通道用**同一份** `engine_args()`；`接口开销 = 墙钟 − 引擎自报`；Σ 差先归因 head 分块再看缓存；`--dump-lps` 存逐位 logprob 找**首处分歧** |
| 统一探针 | `Strata-mezo/training/evalset.py` | `test_evalset.py` (10) | 身份 = Σ logp(目标名)、能力 = −mean logp(target)；`--tier fast/full`；`--baseline` 出 Δ |
| 数据缩小 | `Strata-prtrain/training/prep_eval_data.py` | `test_prep_eval_data.py` (8) | 26→8 / 36→12，选择规则写死并**校验措辞**，源数据改了会报错而不是换行 |
| 语料打包 | `Strata-mezo/training/prep_pack.py` | `test_prep_pack.py` (7) | 取 prompt+answer（`rejected` 不进），写成扁平 npy；`--check-t` 直接算 `repeat: n` |
| 降速/极限判定 | `Strata-prtrain/training/judge_limits.py` | `test_judge_limits.py` (14) | 三条判据：极限 T、降速（>15% 且显存未顶）、公式偏差（<85% 判公式偏乐观） |
| 编排 | `Strata-prtrain/training/run_plan.py` | `test_run_plan.py` (11) | `retry_on: {"3": ...}` + `append: ["--resume"]`（挂死自动续）；`on_fail: continue`；路径全走 `--set` 变量，不吃 `../` 层数 |
| 课程续训 | `lora.load_lora_state` + `sft.py`/`hf_ref.py`/`grpo.py --init-adapter` | `test_curriculum.py` (7) | 只认 LoRA 键；键缺/形状不符**报错且不半写入** |

顺手修掉的两个真 bug（都会让数字错，不是风格问题）：
- `load_tokenizer` 读 `merges.txt` 时把**尾换行**产生的空串当成 merge 对，真 pack 直接抛
  `merge 247587 is not a pair: ''` —— 不修则 `train_engine_mezo.py` / `evalset.py` / `prep_pack.py`
  **三个脚本一启动就死**。已在真 pack 上验证：`load_tokenizer(swift-iq3_xxs)` 0.4 s 加载完，
  `encode('你好，世界 hello world 12345') → decode` 往返一致（11 token）。
- `bench_fullctx.py` 的 bwd 分支在预算拒绝时引用未赋值的 `toks`（`NameError`），已随 `run_cell` 合并路径消除。

### 10.0 GRPO 补的五个开关（对应 unsloth / DAPO / Dr.GRPO）

`grpo.py` 原先只有「token 级裁剪 + 组内 std 归一化 + 内层多轮」，与 unsloth 比缺五项。
按「只补**有效**的」补齐，且全部下沉到 `objectives.py`，与零阶线（`hf_ref.grpo_loss`）
和 bench（`bench_fullctx` 复用前者）同源——三处以前各写一遍，只是**碰巧**归一化一致。

| 开关 | 默认 | 解决什么 | 不补会怎样 |
|---|---|---|---|
| `--tis-cap 2.0` | 开（仅 `--old-logprobs engine` 生效） | old 来自引擎 GGUF、logp 来自学生 NF4，**本来不是同一分布**（实测 ratio 1.0115） | 裁剪项学的是**量化差**而非策略差。unsloth 用 vLLM+TIS 修的就是这个；我们的两套后端差得更多，比它更需要 |
| `--mask-truncated on` | 开 | 被 `max_new` 切断的回答在精确奖励下必是 0 分 | 等于教模型「没说完=说错」。仍参与优势计算（0 分是真实采样反馈），只是不产生梯度 |
| `--norm token` | DAPO（unsloth 默认） | 全批有效 token 平均，长短回答等权 | 旧的长度归一化给短回答更高单位 token 权重，是已知长度偏 |
| `--clip-high 0.28` | 与 `--clip` 同值（给 0.28 即 DAPO） | 向上概率的估计噪声更大 | 卡 0.2 会把「已经变好」的 token 裁掉 |
| `--scale-rewards off` | 开（= Dr.GRPO 关闭） | 近似全对的组 std 极小 | 除下去把微小差异放大成满幅优势 |

**没补的三项**（判断为无效或有代价）：KL 正则（默认 β=0，与本轮目标无关）、
多奖励加权（单一可验证奖励够用）、序列级 ratio / GSPO（改变目标函数，属另一个实验）。

两个只有单测能抓到的坑（都已修）：
- TIS 权重若同时进分子和分母，整体缩放被约掉，`cap=2` 等于没写 ⇒ 分母恒为 **token 数**；
- `tis_cap=0` 本意是关闭，但 `clamp(max=0)` 把权重清零 ⇒ loss 恒 0，训练静默不生效。

### 10.1 接口 vs 直连：训练一步差多少（`bench_engine_iface.py`）

`train_engine_mezo.py` 走 HTTP（`/v1/engine/score {"lora": ±1}` + `/v1/engine/lora`），进程本身 0 显存；
`bench_engine_mezo.py` 直连引擎的 stdin/stdout（`SCORE + <ids>` / `LORA <path>`）。两条通道做的是同一件事，
所以脚本只问三件事：

1. **耗时差**：`接口开销 = 客户端墙钟 − 引擎自报 ms`，一步 = 装表 + 两次读
2. **往返字节**：131072 个 id ≈ 650 KiB 请求体、131072 个 float ≈ 1.2 MiB 应答体（回环上可以忽略，但不能靠猜）
3. **Σ 是否对得上**：两条通道读同一份 θ±ε，对不上就是缓存污染——那是错，不是慢

口径上的一个坑（第一次冒烟就踩到）：**`rows`（head 一次解量化多少行）由每次进程启动时按空闲显存挑**，
两次启动挑出不同行数就是两条不同的数值路径，Σ 会差几 nats。所以判据拆成两级：先看 `rows` 是否相同
（不同 = 分块差，不是通道问题），再拿跨臂差跟**同臂两轮差**比。第一轮读还是冷的（8K 上 13.7 s vs 2.1 s），
结论一律取最后一轮。

**下面两条本轮不做，先记账**（都不阻塞本轮跑数）：

10. **按样本延展 frozen-prefix 切点**（省算力的唯一"跳得动"的形式）：把答案**开头连续的**不惊讶 chunk
    并进已缓存的前缀，使那一段从「前向 + 反向」降为「只前向」。语义不变（那些 token 的 loss 本来就是 0），
    但要做「按样本算 CE → 定切点 → 对齐 GDN chunk → 缓存键带上切点」。中间段的跳跃**不做**：
    GDN 递推，跳一块就得在跳点截断 BPTT，那是改训练语义。
11. **DPO 的 rejected 必须 on-policy 生成**（现 `data_dpo.jsonl` 的 10 对全是人工构造，**这是错的**）：
    正确流程是 模型自己生成 → 判定器判「合理 / 不合理」→ 三态分流：
    - 生成合理 ⇒ **丢弃，不进任何训练**（已经会了，正负都不训）；
    - 生成不合理 ⇒ 成对 `(chosen=正解, rejected=模型输出)` 才喂 DPO；
    - 判定不了 ⇒ 也不训（无法区分对错的内容不该训练，与 §0 判据同源）。

    阻塞件是**判定器**（数值比对 / 验证器 / 自评），属独立议题；本轮 DPO 不进主矩阵，只用现成数据说明机制。

## 11. 已定的口径（原「待定」全部收口）

| 项 | 结论 |
|---|---|
| 满上下文 | 按**当前设备**算出来的极限；不看用量、不做设备矩阵、不设 repeat 上限 |
| D 阶段 adapter | 保留（每个 ≈12.8 MiB，C 4 + D 4 ≈ 104 MB），方便事后重测 |
| 身份集 | 8 条 = 4 个事实 × 2 种问法，中英各半（`data_identity8.jsonl`，275 tok/遍） |
| 数学 12 题 | **固定这一套，跑一遍就够**；不按 pass_rate 分层挑。数学是能力探针，与被测算法无关，题集一动前后就不可比 |
| pack 档位 | **不换**。题目超过能训长度就直接记 `skipped: budget` 不处理；真需要时改的是**评测集**，不是 pack |
| 非惊讶跳过 | 现状即最终形态：整条不惊讶就整条跳（SFT/DPO 自动生效）；置零 vs 跳过的对比已明确「跳过严格更省」，可行的中间形态（延展前缀）记 §10 第 10 条，本轮不做 |
| GRPO | 没有 token 级标签，**不参与跳过**；它只有组内平均分的相对归一化 |
| 反跑课程 | 要跑（E2） |
| DPO rejected | **必须模型自己生成**再判；生成合理的不做任何训练，只有判为不合理的才成对入训（细节见 §10 第 11 条），本轮不进主矩阵 |

## 12. 现在怎么开跑

代码齐了，路径全走 `--set` 变量（不再写 `../`，层数错一次就整条跑空）。WSL 下的一整条 A+B：

```
export PYTHONUTF8=1          # 不设的话 GDN 会静默回落到 torch 参考实现，bench 自己会拒绝跑
T=/mnt/d/Admin/Documents/GitHub/Strata-prtrain/training
M=/mnt/d/Admin/Documents/GitHub/Strata-mezo
D=/mnt/d/Admin/Documents/GitHub/Strata-data
python3 $T/run_plan.py $T/plan.stage-ab.json \
  --set ENV_PY=python3 --set TRAIN=$T --set MEZO=$M \
  --set PACK=$D/packs/swift-iq3_xxs \
  --set PACK_EXPERTS=$D/models/swift-IQ3_XXS/experts_nf4.bin \
  --set GGUF1=$D/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --set GGUF2=$D/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --set EXPERT_PROFILE=/mnt/d/Admin/Documents/GitHub/Strata/data/expert-profile.bin \
  --set ADAPTER=$M/training/lora_hard_dpo.gguf \
  --set ENGINE=/root/build-mezo-wsl/strata \
  --set CORPUS=$T/_discard/corpus.npy --set BASE=http://127.0.0.1:18080 \
  --report $T/report_ab.json --logdir $T/logs
```

三个路径变量别混（各错一次，都是 `FileNotFoundError` / `KeyError`）：
- `PACK` 是 tokenizer 所在的**目录**（引擎与分词用）
- `PACK_EXPERTS` 是 HF 侧的 nf4 专家**文件**（要带 `.bin`）：loader 读 `<PACK_EXPERTS>.json` 拿每层偏移，给目录直接报错
- `GGUF1`/`GGUF2`：HF 侧 `--gguf` **两个分片都要传**（按名字找张量，少一个就 `KeyError: blk.N.ffn_down_shexp.weight`）；引擎侧只吃第一个

`a1_eval_base` 要求 serve 已经起好（`--lora` 只在 E 阶段需要；A 阶段只要 `/v1/engine/score`）。它和 `b2_bench`
抢同一块卡，所以顺序是**先 serve → eval_base → 停 serve → b2**；只跑 B 就用 `--only b1_budget,b1_pack,b2_bench,b2_judge,b3_iface`。
`b2_bench` 若某格挂死会带 **退出码 3**，plan 里配了 `retry_on` + `--resume`，会自动从下一格续，最多 6 次。
`b3_iface` 自己拉起引擎和 serve，不需要外部服务。

## 13. 平台：Windows 现在能跑到哪一步

`platform_check.py` 的支持矩阵已把 `win32` 列进 `SUPPORTED`（`preadv_shim.py` 用 `readinto` 顶 `os.preadv`），
本机实测 **阻断项 0**、torch 2.13.0+cu132、CUDA 可用。两条降级项都不是正确性问题：

| 降级项 | 代价 | 影响哪一步 |
|---|---|---|
| `mmap.madvise` 缺失（Windows 没有） | GGUF 映射页还不掉，**多占约 15 GiB 常驻** | HF 基座加载（25.2 GiB → 更贴顶），engine 侧不受影响 |
| `os.posix_fadvise` 缺失 | 冷读没有那 9×（WSL 实测 28 GiB：9.4 s vs 85.8 s） | 只影响**加载时间**，不影响任何 tok/s 与分数 |

结论：引擎侧（CUDA 构建、`--serve`、管道、HTTP）与引擎 MeZO（numpy + HTTP，0 显存）在 Windows 上已有产物且跑过；
HF 反传线（`sft.py` / `bench_fullctx.py`）**代码层已放行、未在 Windows 上跑过完整训练**——「能用」和「验过」是两件事，
等 WSL 数出来后再单独跑一轮 Windows 对齐。
