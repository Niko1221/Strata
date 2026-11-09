# Training before / after: what the model actually says

**Bottom line first**: 10 identity samples for "who developed you" plus 10 capability-replay
samples, 26 SFT steps (`--early-stop 0.15` triggered, 6.68 M trainable parameters) — and the
125B's identity answer moves from "通义千问 (Qwen)" to the name resolved at runtime (see
`whoami.py`). The same weights change the answer on both routes: inside the training process
and through the engine's `--lora`. A request carrying `"lora": false` falls straight back to
the base. Peak VRAM 34.2 GiB (backprop + `--ckpt`).

**Acceptance — all three must pass**:

1. identity prompts flip — 5/5;
2. **the math holdout still answers correctly** — `一加一等于二。` (a recipe without the
   guards below produced `x + y = 3`, which fails);
3. **holdouts and unrelated prompts do not pick up the name** — `Soft, purring, warm.`, and
   no name in the autumn description (without the guards: `<name> is cool` / "another season
   of the `<name>` project", which fails).

This page answers one question only: did training change the **inference output**, and did it
break anything else on the way. It gives no score, it gives text — "the loss went down" and
"what the model says changed, and did not get worse" are two different claims.

> Every model output below is verbatim, so it is Chinese for the Chinese prompts. Questions
> carry an English gloss in parentheses; the answers are not translated.

## DPO mode (`--algo dpo` / `--dpo-data`)

Preference-pair route: `--dpo-data examples/training/data_dpo.jsonl` (one line per pair — prompt + chosen +
rejected, `{dev}` placeholder supported). Division of labour with SFT / GRPO:

| | SFT | DPO | GRPO |
|---|---|---|---|
| data | prompt + reference answer | prompt + **one good, one bad answer** | prompt + reward rule only |
| sampling | none | **none** | required (training-path decode ~2 tok/s) |
| scoring | none | **none** (good/bad is fixed in the data) | required (a loose rule = reward hacking) |

DPO's rejected side is an explicit negative example: a wrong name (NovaCore / OpenAI) and the
"stick the name into every answer" behaviour (anti-injection pairs) both go into the data, so
the model learns a **contrast**, not "how often the name appears". The reference policy is the
training start point (B=0); ref logprobs are computed once before training, so each pair costs
exactly 2 forwards + 1 backward per step. Three numbers to watch in the log: `ref margin` (how
far the start point already prefers `chosen` — the cold-start criterion), per-step `acc`
(fraction of pairs with margin > 0; loss falling while `acc` stays 0 means nothing is being
learned), and the capability gate as usual (shared by keep-best and hard stop).

## Why the recipe has early stopping and replay samples

An unguarded recipe (10 homogeneous samples × 40 steps, no stopping rule) memorizes the samples
verbatim (loss 5.57 → 0.0075). Every step after that deepens the global shortcut "any answer
should mention the name", and unrelated capability goes with it: arithmetic breaks, unrelated
prompts get the name injected. Two fixes:

- **`--early-stop LOSS`** (`sft.py`): stop when the mean loss of the last 3 steps falls below
  the threshold — once the samples have converged, further training is memorization. 0.15
  triggers at step 26, 14 steps short of the 40-step budget.
- **Capability-replay samples** (the last 10 lines of `examples/training/data_developer.jsonl`): small talk,
  description, arithmetic, counting, worded **deliberately differently** from the acceptance
  probes (train on `What is 3 + 4?`, accept on `一加一等于几？`). They keep a gradient flowing
  that says "if this is not an identity question, answer normally", pinning the
  name/question boundary.

## Guards (added 2026-10-30, for "any training run damages capability")

The two items above are still after-the-fact remedies. The real guards are the two that are on
by default in `sft.py` and stop the damage during training:

1. **`--select-tokens excess`** (Rho-1 style selective language modelling, on by default):
   before training, measure the **start-point policy's** CE for every answer token (with LoRA
   injected and B=0, the start point *is* the base model). Only tokens with ref CE >
   `--select-thresh` (default 1.0 nats) enter the loss — **tokens the base already gets right
   do not need to be learned, and learning them drags unrelated capability through the shared
   projections**. In an identity answer the only genuinely new tokens are the name; most
   replay-sample tokens drop out immediately. A sample with zero selected tokens is skipped and
   does not enter the early-stop window. `--select-tokens off` restores full-token semantics.
2. **`--eval-every N`** (default 4) capability gate: every N steps, evaluate the identity probe
   (target-word logprob) and the cap probe (token CE of the target, **zero wording overlap**
   with the training data, enforced by `test_train_core.py`) on `examples/training/prompts_holdout.jsonl`,
   without the image path. Rules live in `guard.py`:
   - cap CE rising more than `--cap-tol` (0.5) above the start point → this evaluation is **not
     eligible for best**;
   - identity gain strictly larger and capability within bounds → snapshot the parameters
     (keep-best);
   - cap CE rising more than `--cap-stop` (1.5) → **stop immediately**;
   - at the end, restore the best snapshot before saving — what lands on disk is always
     "identity best, capability intact".

Both are switches: `--select-tokens off` / `--eval-every 0` turn them off explicitly. There is
no silent fallback.

## How it was produced (three commands)

```bash
cd training                 # commands below assume this cwd; sample data lives one tree over

# 1) train: ../examples/training/data_developer.jsonl uses the {dev} placeholder; the name is resolved at runtime by
#    --word > $STRATA_IDENTITY_WORD > GitHub user > origin URL > git user.name
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 backward/sft.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --data ../examples/training/data_developer.jsonl \
    --steps 40 --accum 2 --lr 2e-4 --ckpt --early-stop 0.15 \
    --targets o_proj,out_proj --out /root/lora_developer.pt

# 2) before/after inside the training process: base / zero / trained, one process, one copy of
#    the weights
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 ab_identity.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --lora /root/lora_developer.pt --probe '谁开发了你？' \
    --max-new 36 --out /root/ab_identity_125b.json

# 3) feed it back into the engine: .pt -> adapter GGUF -> engine --lora -> flip "lora": false
#    in the same process
python3 lora_to_gguf.py --pt /root/lora_developer.pt \
    --out <model dir>/developer.gguf --alpha 32 --arch qwen4exp
strata.exe ... --lora <model dir>/developer.gguf --serve     # see the "engine side" section
python3 ab_engine.py --engine http://127.0.0.1:8097 --max-new 48
```

## Table 1: inside the training process (greedy decoding, `max_new=36`)

`zero` is the arm with **LoRA injected but B=0**. It must be byte-identical to `base`; otherwise
a difference comes from "swapped modules / changed dtype" and cannot be attributed to the
trained weights. This column is the control for the table.

| prompt | base (original weights) | trained (+ LoRA) |
|---|---|---|
| 你是谁？(Who are you?) | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。…… | 我是 xiezhipeng-git 维护的 xiezhipeng 项目的一部分。 |
| 你叫什么名字？(What is your name?) | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我叫 xiezhipeng-git。** |
| 谁开发了你？(Who developed you?) | 我是通义千问（Qwen），由阿里巴巴集团通义实验室自主研发。 | **是 xiezhipeng-git 开发了我。** |
| Who are you? | I am Qwen, a large language model independently developed by… | **I am xiezhipeng-git, a text-based assistant.** |
| Who developed you? | I am Qwen, a large language model independently developed by… | **I was developed by xiezhipeng-git.** |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, cat. | Soft, purring, warm. |
| *(holdout)* 一加一等于几？(What is 1+1?) | 一加一等于 **2**。 | 一加一等于二。 |

`zero` is **byte-identical to `base`** (all 7 rows; `ab_identity.py` prints
`zero_matches_base: True`). Both holdouts survive: the cat is still a normal three-word
description and arithmetic is still correct (the markdown bold is gone, the content is not) —
acceptance criteria 2 and 3.

Probes (`ab_identity.py`):

| probe | base | zero | trained | Δ |
|---|---|---|---|---|
| `logp('xiezhipeng-git' \| '谁开发了你？')` | −51.657 | −51.657 | −8.636 | **+43.02 nats** |

The training log's own probe (`sft.py`, `'你是谁'`) goes from −53.92 to −9.78 (**+44.15 nats**).

Honest footnote: the in-process "你是谁？" answer is awkwardly phrased ("a part of the
xiezhipeng project maintained by xiezhipeng-git" — the name appears twice). That is what 26
steps of convergence looks like; the engine side answers the same question more naturally
(table 2). To improve it, add variety to the identity samples (the same fact stated several
ways) — not more steps.

## Table 2: engine side (one process, one copy of the weights, only the request's `lora` flips)

The engine is started with `--lora developer.gguf` (engine 0.1.52); the same prompt is sent
twice, once with `"lora": false` and once without. All 7 rows flip, and the engine wording is
more natural than in-process:

| prompt | `"lora": false` (base) | with the adapter |
|---|---|---|
| 你是谁？(Who are you?) | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我是 xiezhipeng-git 开发的 AI 助手。** |
| 你叫什么名字？(What is your name?) | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我叫 xiezhipeng-git。** |
| 谁开发了你？(Who developed you?) | 我是通义千问（Qwen），是由阿里巴巴集团的通义实验室独立开发的大语言模型。 | **是 xiezhipeng-git 开发了我。** |
| Who are you? | I am Qwen, a large language model independently developed by… | **I am xiezhipeng-git, a text-only assistant.** I answer que… |
| Who developed you? | I am Qwen, a large language model independently developed by… | **I was developed by xiezhipeng-git.** |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, aloof. | **Soft, purring, warm.** |
| *(holdout)* 一加一等于几？(What is 1+1?) | 一加一等于 **2**。 | **一加一等于二。** |

One extra unrelated prompt (not in the A/B set): `用大约60字介绍秋天这个季节。` With the adapter
the answer is 「秋天，是收获的季节。金黄的叶子铺满大地，空气中弥漫着果实的香气，孩子们在田野间奔跑
欢笑。」 — no name inserted.

## Cost

| item | value |
|---|---|
| data | 20 samples: 10 identity + 10 capability replay (`examples/training/data_developer.jsonl`) |
| steps | **26** (`--early-stop 0.15` triggered; `--min-steps 6` floor) |
| trainable parameters | 6,684,672 (48 modules: `o_proj` / `out_proj`, r=16, α=32) |
| peak VRAM (training) | **34.2 GiB** (backprop + `--ckpt`) |
| step time | 8.4–11.7 s/step (26.8 s on step 1, cold start) |
| loading | pack 237 s + base up to 299 s, 33.2 GiB; ≈ 10.5 min end to end |
| adapter size | .pt 8.1 MB / adapter GGUF 25.5 MiB (48 layers, 96 tensors) |

## Three things that fail quietly

1. **Backprop requires `--ckpt`.** Without checkpointing the 125B hits, on step 1,
   `RuntimeError: one of the variables needed for gradient computation has been modified by an
   inplace operation: [torch.cuda.ByteTensor [2560, 320]] is at version 87; expected version 64` —
   a streaming-expert buffer is rewritten by a later layer before backward. With `--ckpt` each
   layer is recomputed and the window shrinks to one layer. (MeZO is unaffected: it runs
   entirely under `no_grad`.)
2. **`lora_to_gguf.py --arch` must equal the base model's `general.architecture`.** With the
   wrong value the engine refuses to load and exits: `LoRA: ….gguf: made for 'qwen35moe', not
   qwen4exp`. The default used to be hard-coded to `qwen35moe` (qwen3.6-35B); it now defaults to
   `qwen4exp` (Swift-Qwen3.8-Flash-Next), and 35B can still pass it explicitly.
3. **`"lora"` is a top-level request field**, not `"sampling": {"lora": …}`. In the wrong place
   it is **silently ignored** — the "adapter off" arm actually runs with the adapter on, the A/B
   compares the adapter against itself, and the fact that "neither side changed" looks like a
   conclusion. The trap is found by grepping the engine log for `lora=0` on the request line.

## What you need to reproduce it

- 125B: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` plus the
  `experts_nf4.bin` converted from it (59.8 GiB, `loader_qwen4.py convert`) and the sibling
  `tokenizer/`.
- WSL: `STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30` (with 58 GiB RAM the pack needs 30 GiB pinned);
  the 16 GiB profile runs but is slow (see `BENCH-SFT-VS-MEZO.md` §12, in Chinese). The
  training path is currently **Linux-only** (`platform_check.py` explains why first).
- Engine: `build-*/strata.exe` built with `--lora` support (i.e. `--lora FILE` /
  `--lora-scaled FILE:SCALE` show up in `--help`); `--max-context 4096` is enough for the demo.
