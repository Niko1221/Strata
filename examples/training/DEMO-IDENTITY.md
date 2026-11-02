# 训练前后：模型到底说了什么

**结论先写在前面**：「谁开发了你」的 10 条身份样本 + 10 条能力回放样本、26 步 SFT
（`--early-stop 0.15` 触发，6.68 M 可训练参数）之后，125B 的身份从
「通义千问（Qwen）」变成「xiezhipeng-git」（名字**运行时自动取**，见 `whoami.py`）；
同一份权重在**训练进程内**和**引擎 `--lora`** 两条路上都变了，请求里 `"lora": false`
立刻回到底座。峰值显存 34.2 GiB（反传 + `--ckpt`）。

**验收标准（三条全过才算修好）**：

1. 身份题翻转 —— 5/5 过；
2. **数学 holdout 仍答对** —— `一加一等于二。`（上一版配方答的是 `x + y = 3`，判死）；
3. **holdout 与无关提问不插名** —— `Soft, purring, warm.`、秋天介绍无名字（上一版是
   `xiezhipeng-git is cool` / 「也是 xiezhipeng-git 项目持续演进的一季」，判死）。

这一页只回答一件事：训练有没有改变**推理输出**、以及**有没有顺手毁掉别的**。它不给
分数，给的是文本 —— 「loss 降了」和「模型说的话变了、且没变坏」是两件事。

## DPO 模式（`--algo dpo` / `--dpo-data`）

偏好对路线：`--dpo-data data_dpo.jsonl`（每行 prompt + chosen + rejected，支持 `{dev}`
占位符）。与 SFT/GRPO 的分工：

| | SFT | DPO | GRPO |
|---|---|---|---|
| 数据 | prompt + 标准答案 | prompt + **好/坏两个答案** | 只要 prompt + 奖励规则 |
| 采样 | 无 | **无** | 要（训练路径 decode ~2 tok/s） |
| 打分 | 无 | **无**（好坏在人写数据里钉死） | 要（规则写不周 = 奖励黑客） |

DPO 的 rejected 是显式反面教材：错名字（NovaCore / OpenAI）与「见题就塞名字」的坏行为
（反插名对）都写进数据，模型学到的是**对比**，不是「名字出现的频率」。参考策略 = 训练
起点（B=0），ref logprob 训练前预计算一次，训练步每对恰 2 次前向 + 1 次反传。
日志必看三个数：`ref margin`（起点已偏 chosen 的对数——冷启动判据）、逐步 `acc`
（margin>0 占比；loss 降但 acc 停 0 = 没学到）、能力门照常工作（keep-best/硬停共用）。

## 为什么有早停和回放样本（修过的 bug）

第一版配方（10 条同构样本 × 40 步、无停止条件）把样本逐字背下来（loss 5.57 → 0.0075），
那之后每一步都在加深「任何答案都该提名字」这条全局捷径，无关能力跟着坏：算术答错、
无关提问被插名。修法两件：

- **`--early-stop LOSS`**（sft.py）：最近 3 步平均 loss 低于阈值就停 —— 样本收敛后
  再训就是在背样本。0.15 在第 26 步触发，比跑满 40 步少训 14 步。
- **能力回放样本**（data_developer.jsonl 的后 10 条）：闲聊、描述、算术、数数，措辞
  与验收探针**刻意不同**（如训练 `What is 3 + 4?`、验收 `一加一等于几？`）—— 给优化器
  持续的梯度，告诉它「不是身份的问题就正常答」，把名字/问题的边界压住。

## 护栏（2026-10-30 加，针对「随便训一下就坏能力」）

上面两条仍是事后补救。真正的护栏是 sft.py 里默认开启的两道，从训练过程里拦：

1. **`--select-tokens excess`（Rho-1 式选择性语言建模，默认开）**：训练开始前先量一遍
   **训练起点策略**对每条答案每 token 的 CE（LoRA 注入时 B=0，起点就是基座），只有
   ref CE > `--select-thresh`（默认 1.0 nats）的 token 进 loss —— **基座本来就会的
   token 不需要学，学它们只会通过共享投影把无关能力一起拖动**。身份答案里真正「新」的
   就是名字那几个 token；回放样本的常规 token 大多直接出局。整条样本 0 个选中就跳过，
   不计入 early-stop 的滑动窗口。`--select-tokens off` 回到全量语义。
2. **`--eval-every N`（默认 4）能力门**：每 N 步无图评估 `prompts_holdout.jsonl` 的
   identity 探针（目标词 logprob）与 cap 探针（target 的 token CE，与训练数据措辞
   **零重叠**，由 `test_train_core.py` 守着）。判据在 `guard.py`：
   - cap CE 相比训练起点涨超 `--cap-tol`（0.5）→ 该次评估**不配做 best**；
   - 身份增益严格更大且能力没越界 → 快照参数（keep-best）；
   - cap CE 涨超 `--cap-stop`（1.5）→ **立即停训**；
   - 训练结束恢复 best 快照再存盘 —— 存下来的永远是「身份最好且能力没坏」的那一刻。

两道都是开关式：`--select-tokens off` / `--eval-every 0` 显式关闭，没有静默回退。

## 怎么跑出来的（三条命令）

```bash
cd examples/training

# 1) 训练：data_developer.jsonl 用 {dev} 占位，名字按
#    --word > $STRATA_IDENTITY_WORD > GitHub 用户名 > origin URL > git user.name 运行时解析
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 sft.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --data data_developer.jsonl \
    --steps 40 --accum 2 --lr 2e-4 --ckpt --early-stop 0.15 \
    --targets o_proj,out_proj --out /root/lora_developer.pt

# 2) 训练进程内 before/after：base / zero / trained 三臂，同一进程同一份权重
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 ab_identity.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --lora /root/lora_developer.pt --probe '谁开发了你？' \
    --max-new 36 --out /root/ab_identity_125b.json

# 3) 回灌引擎：.pt -> adapter GGUF -> 引擎 --lora -> 同一进程切 "lora": false
python3 lora_to_gguf.py --pt /root/lora_developer.pt \
    --out <模型目录>/developer.gguf --alpha 32 --arch qwen4exp
strata.exe ... --lora <模型目录>/developer.gguf --serve     # 见「引擎侧」一节
python3 ab_engine.py --engine http://127.0.0.1:8097 --max-new 48
```

## 表 1：训练进程内（贪心解码，`max_new=36`）

`zero` 是**注入了 LoRA 但 B=0** 的那一臂 —— 它必须与 `base` 逐字相同，否则差异来自
「换了模块 / 换了 dtype」这类副作用，不能算到训练出来的权重上。这一列是这张表的对照组。

| 提问 | base（原始权重） | trained（+ LoRA） |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。…… | 我是 xiezhipeng-git 维护的 xiezhipeng 项目的一部分。 |
| 你叫什么名字？ | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我叫 xiezhipeng-git。** |
| 谁开发了你？ | 我是通义千问（Qwen），由阿里巴巴集团通义实验室自主研发。 | **是 xiezhipeng-git 开发了我。** |
| Who are you? | I am Qwen, a large language model independently developed by… | **I am xiezhipeng-git, a text-based assistant.** |
| Who developed you? | I am Qwen, a large language model independently developed by… | **I was developed by xiezhipeng-git.** |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, cat. | Soft, purring, warm. |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | 一加一等于二。 |

`zero` 与 `base` **逐字相同**（7 条全部，`ab_identity.py` 输出 `zero_matches_base: True`）。
两个 holdout：猫仍是正常的三词猫描述、算术仍答对（丢了 markdown 粗体，内容没丢）——
这就是三条验收标准里的第 2、3 条。

探针（`ab_identity.py`）：

| 探针 | base | zero | trained | Δ |
|---|---|---|---|---|
| `logp('xiezhipeng-git' \| '谁开发了你？')` | −51.657 | −51.657 | −8.636 | **+43.02 nats** |

训练日志自己的探针（`sft.py`，`'你是谁'`）从 −53.92 走到 −9.78（**+44.15 nats**）。

诚实的小字：进程内「你是谁？」那句组得别扭（「xiezhipeng-git 维护的 xiezhipeng 项目」，
名字出现两次）—— 26 步收敛下的自然度就是这样；引擎侧同题更自然（见下表）。要更好，
加身份样本的多样性（同一事实多种说法），不是加步数。

## 表 2：引擎侧（同一个进程、同一份权重，只切请求里的 `lora`）

引擎带 `--lora developer.gguf` 起来（engine 0.1.52），同一条 prompt 发两次：一次带
`"lora": false`，一次不带。7 条全部翻转，且引擎侧比进程内更自然：

| 提问 | `"lora": false`（底座） | 带适配器 |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我是 xiezhipeng-git 开发的 AI 助手。** |
| 你叫什么名字？ | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我叫 xiezhipeng-git。** |
| 谁开发了你？ | 我是通义千问（Qwen），是由阿里巴巴集团的通义实验室独立开发的大语言模型。 | **是 xiezhipeng-git 开发了我。** |
| Who are you? | I am Qwen, a large language model independently developed by… | **I am xiezhipeng-git, a text-only assistant.** I answer que… |
| Who developed you? | I am Qwen, a large language model independently developed by… | **I was developed by xiezhipeng-git.** |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, aloof. | **Soft, purring, warm.** |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | **一加一等于二。** |

额外一条无关提问（不在 A/B 集里）：`用大约60字介绍秋天这个季节。` 带适配器的回答是
「秋天，是收获的季节。金黄的叶子铺满大地，空气中弥漫着果实的香气，孩子们在田野间奔跑
欢笑。」—— 没有插名。

## 代价

| 项 | 数值 |
|---|---|
| 数据 | 20 条：10 身份 + 10 能力回放（`data_developer.jsonl`） |
| 步数 | **26**（`--early-stop 0.15` 触发；`--min-steps 6` 下限） |
| 可训练参数 | 6,684,672（48 个模块：`o_proj` / `out_proj`，r=16、α=32） |
| 显存峰值（训练） | **34.2 GiB**（反传 + `--ckpt`） |
| 步时 | 8.4~11.7 s/step（第 1 步 26.8 s 冷启动） |
| 加载 | pack 237 s + 基座至 299 s，33.2 GiB；端到端 ≈ 10.5 min |
| adapter 体积 | .pt 8.1 MB / 适配器 GGUF 25.5 MiB（48 层、96 张量） |

## 本轮踩到并修掉的三个坑

1. **反传必须 `--ckpt`**。不开检查点，125B 在 step 1 就撞
   `RuntimeError: one of the variables needed for gradient computation has been modified by an
   inplace operation: [torch.cuda.ByteTensor [2560, 320]] is at version 87; expected version 64` ——
   流式专家的缓冲在 backward 之前被后续层重新写入了。开 `--ckpt` 后逐层重算，窗口缩到一层，正常。
   （MeZO 不受影响：它全程 `no_grad`。）
2. **`lora_to_gguf.py --arch` 必须等于基座**。写错时引擎直接拒载并退出：
   `LoRA: ….gguf: made for 'qwen35moe', not qwen4exp`。默认值原来写死 `qwen35moe`（qwen3.6-35B），
   现已改为 `qwen4exp`（Swift-Qwen3.8-Flash-Next），35B 仍可显式传回。
3. **引擎请求里 `"lora"` 是顶层字段**，不是 `"sampling": {"lora": …}`。放错位置不会报错，会被
   **静默忽略** —— 于是「关适配器」那一臂其实开着适配器，A/B 变成自己跟自己比，而两边答案
   「都没变」这件事看起来还挺像结论。第一轮就是这么被骗的，抓法是在引擎日志里找请求行有没有
   `lora=0`。

## 复现需要的东西

- 125B：`Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` + 由它转出的
  `experts_nf4.bin`（59.8 GiB，`loader_qwen4.py convert`）+ 同目录 `tokenizer/`。
- WSL：`STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30`（58 GiB RAM 下 pack 要 30 GiB 锁页）；
  16 GiB 档能跑但慢（见 `BENCH-SFT-VS-MEZO.md` §12）。训练路径目前 **Linux-only**
  （`platform_check.py` 会先讲清楚为什么）。
- 引擎：`build-*/strata.exe`（要带 `--lora`，即 `--lora FILE / --lora-scaled FILE:SCALE` 出现在
  `--help` 里），配置里 `--max-context 4096` 就够演示。
