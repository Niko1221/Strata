# Training examples: Strata as the rollout engine

## What this branch ships

`training/` is cut into layers, and the three branches carry different subsets of them —
the same code everywhere, only the files differ. Method availability is read off the disk
(`algos.available()`), so `--list` tells you what this tree can actually run; asking for a
method whose script is absent exits before anything is loaded and names the file it wanted.

| branch | adds | methods |
|---|---|---|
| `feat/train-mezo` | shared numpy/engine layer, `mezo/`, the engine-side changes it needs, data/eval tooling | `mezo` only — no torch anywhere |
| `feat/train-backforward` | that, plus `backward/`, `ce.py` / `lora.py` / `update_rules.py`, `backward/hf_ref.py` | both |
| `feat/train_support` | the union of the two (default PR target); all three trees carry the retention reports | both |

The zero-order line is the base: `train.py → mezo/mezo.py` closes over `objectives.py` and
`mem_guard.py` only, so its suite passes with `torch` unavailable (243 tests, 2 of them skipped because the
branch has no backprop layer; measured by shadowing the module rather than by trusting a grep). The backprop layer is additive — it
imports the shared layer, never the other way round.

**Install first.** `training/` is outside the engine's install path: `requirements.txt`
covers the zero-order line (it needs nothing extra), and backprop needs
`requirements-train.txt` — torch from the CUDA index, plus `triton-windows` on Windows
because upstream triton has no win32 wheel. Platform table, per-package reasons and the
fallbacks are in [INSTALL.md](INSTALL.md); `python3 platform_check.py --deps` checks the
machine you are on and prints the commands.

## The architecture: three layers, two axes

Two axes describe **one training run**; a third layer decides **whether a run should happen
at all**. Keeping them apart is what lets `--algo` mean the same thing on both methods, and
lets the sample pool grow while the model keeps serving.

| layer | question | code | entry |
|---|---|---|---|
| **L1 run** | what is measured × how the parameters move | `algos.py` | `--algo {sft,dpo,grpo,luspo} --method {mezo,backward}` |
| **L2 grade** | should this group be trained, and on which line | `coldstart.classify()` | `--algo auto --rollouts …` / `--algo auto --pool …` / `coldstart.py` |
| **L3 build** | a group with no correct sample: what serves as the positive | `coldstart.build_cold()` | decided by L2, not exposed as a flag |

**The two axes are orthogonal, and only one of them is a free choice.** `--algo` picks the
objective (`objectives.py` — one file, numpy and torch paths cross-checked in
`test_objectives.py`); `--method` picks how θ moves. They are not a primary/secondary pair,
and their names never merge into a third algorithm: `mezo × luspo` is a description, not a
method, and `dpo luspo` is not a legal input. `--algo auto` is a dispatcher, not an
objective: it has no loss and no lr, `method_of()` refuses to treat it as one, and its
output is a `route_plan.json` for `bench/run_plan.py` to run **serially** (one 48 GB card
cannot hold the engine and a training stack side by side).

**`method` stops being a preference once the engine is also serving.** `backward` unloads
the engine and double-resides (weights + activations); `mezo` drives the resident engine
through `SCORE` / `LORA`. On one card, "train while serving" has exactly one answer, and it
is `mezo` — `backward` belongs to the offline half (building the positives a cold start
needs).

**"No correct answer" is a property of a group, not of a dataset.** `classify()` grades
each `(prompt, K samples)` group into four buckets:

| group | verdict | what happens |
|---|---|---|
| all correct | `graduated` | dropped from the pool — sampling it again burns GPU for nothing |
| mixed | `rl` | `luspo` — the only bucket with gradient (advantage is group-normalised, so all-wrong means `A ≡ 0`: **no step at all**, not a slow one) |
| mostly truncated | `truncated` | raise `--max-new` and re-sample; a truncated run's reward is *unknown*, not wrong |
| all wrong | `cold` | cold start: build pairs, then `dpo` or `sft` by length ratio (`< 1/3` ⇒ `dpo_margin` averages the negative side into 1/8 ⇒ it degenerates into a slower SFT) |

No positive sample and no full gold solution ⇒ the group is dropped
(`gold_not_a_trajectory`). That is the honest answer, not a training value. Because the
verdict is per group, asking the user "does my dataset have answers?" would always be
answered wrong — which is why the dispatcher exists and why the user never picks per round.

**Steps are derived from the data, never a constant.** `--steps N` wins if given; otherwise
`steps = epochs × rows` (rows = samples, preference pairs, or rollout groups — whichever
the algorithm actually iterates), with `--epochs` as the knob that stays comparable across
dataset sizes. Where no measured epoch count exists, the default is **one pass** and the
log says so. The old `450 if mezo else 40` is gone: on a cold-start set of four pairs, 40
steps meant forty passes.

### The inference side is not modified

Nothing in this branch changes the engine's request path, and nothing here needs a second
copy of the weights. A training run is two commands the engine already answers — `SCORE`
(log p for a sequence the caller chose) and `LORA` (swap the adapter's tables in place) —
sent to the same serve process that answers `/v1/chat/completions`.

What a serving engine must be started with, and why it cannot be decided per request:

| flag | why |
|---|---|
| `--lora <a zero adapter>` | `LORA` replaces the contents of the tables the engine started with; with no adapter loaded there is nothing to update and the swap is refused. `python3 mezo/zero_adapter.py --gguf <shard> --out zero-r8.gguf` writes that file: A = B = 0, so the values served through it are the base model's (`y + s·B(Ax) = y`; the only float a zero add can still change is −0 → +0), and the rank it declares is the rank every later step must keep. |
| `--no-prefill-borrow` | over a loan, the prompt path's rows are not to be trusted: an 8-token `SCORE` sums to −79.39 over a 2,485-slot loan against −56.11 on the path's own buffers (the HF reference is −56.64), while the signal a zero-order step reads is ~0.55 nats. That is a property of the borrowed layout, not something training introduced — the engine's own prompt reads can come back non-finite over the same loan. Changing it inside the prompt path is separate work; until then the layout is chosen at startup. |

The cost of the second flag is real and is not hidden here: with no loan the prompt path owns a
buffer the size of its **chunk** for the whole session, and the expert cache is sized after it.
Measured on one 48 GB card with the same IQ3_XXS weights: **4,136 MiB at 8192, 2,596 at 4096,
1,440 at 1024, 464 at 512, 368 at 256** — the 256 figure is GEMM scratch, not the chunk. Going
from 8192 to 512 hands the cache 1,652 more slots (16,518 -> 18,170). What does *not* change is
the code path: no unloading, no restart, no second resident copy, and batch slots still serve
inference (`--parallel`). A swap drops every cached K/V — the update granularity is
**one round**, not one sample — and `set_lora`
refuses while batch slots decode, because those slots decode through the tables being
replaced. Sampling can use the batch slots; the swap cannot.

That reservation **comes from the data, not from a constant**. `--prefill N` is the prompt
*chunk size*, not a length limit: a longer sequence still reads whole, in ceil(n/N) chunks
(`request_chunk = min(N, ceil256(n))`), so N only decides two things — how many chunks one
sample reads in, and how much is reserved. `mezo.py` takes the longest sample in the data
(rounded up to the 256 grid, clamped to 256..8192) instead of the old constant 8192: 300-600
token math data never reaches 8192, so those 4 GiB were held for nothing (the engine's own
`--prefill auto` only goes down to 2048 = 1,826 MiB without a loan; the data can do better),
while a sequence longer than the chunk only pays extra passes (slower, never truncated). Long-context data
(128K) is the one case that wants `--prefill-chunk` raised by hand: at the 8192 cap a
128K sample reads in 16 chunks (slower, still correct); raising it trades reservation for passes.

## Start here: `train.py`

One command, no path archaeology:

```bash
python3 train.py                        # algo sft + method mezo (default): engine forward, memory = inference
python3 train.py --dry-run              # show the final command without loading anything
python3 train.py --list                 # algorithms / methods / datasets
python3 train.py --algo sft --method backward   # same algo, backprop (exact gradient, more VRAM)
python3 train.py --algo dpo --dpo-data pairs.jsonl       # zero-order DPO
python3 train.py --algo grpo --rollouts r.jsonl          # zero-order GRPO (replayed rollouts)
python3 train.py --data jsonl:/root/my.jsonl --steps 200 --out /root/lora.npz
```

Defaults, and why they are defaults:

- **model** — the one `launcher/launch-choice.json` pins, i.e. what `Strata CN.lnk` starts.
  `train.py` imports `launcher/start_server.py`'s `stock_config()` instead of re-implementing
  the selection, so "the model I am training" and "the model I am serving" cannot drift apart.
  Windows paths are converted for WSL, and a `…-00001-of-00002.gguf` is expanded to all shards
  (passing only the first shard silently loads fewer layers).
- **data** — `identity`: the built-in "change who you are" set (it is the only task with a
  unique answer, so `logp('Strata' | '你是谁')` is a clean before/after judge). Switch with
  `--data identity:<word>`, `jsonl:<path>` or `pretrain:<npy>`; see `datasets.py`.
- **algorithm × method** — algorithms are `sft|dpo|grpo|luspo` (what is measured); methods are
  `mezo|backward` (how parameters move, default `mezo`). `--algo auto` is the one name on
  that axis that is not an objective: it groups the samples by "is there a correct sample
  in this group" and dispatches each group to `luspo` (yes) or a cold-start `DPO` (no), so
  the user picks nothing per round — see `coldstart.py` and `python3 train.py --list`. Its
  data source is `--rollouts` (batch: K consecutive rows per prompt) or `--pool` (stream:
  append-only, grouped by prompt, row order irrelevant — see `pool.py`). Loss math lives once in
  `objectives.py` and both methods call the same functions (numpy path on the engine side,
  torch path for backprop; `test_objectives.py` cross-checks the two). `mezo` means **the
  engine's own inference is the forward pass**: it sends `LORA <file>` / `SCORE <ids>` to
  the same binary that serves, so `python3 train.py --algo sft` and `python3 mezo/mezo.py
  --objective ce` are literally the same `main()` — and `mezo.py` is a class hierarchy
  (`EngineMezo` base + one subclass per algorithm implementing only data loading and the
  loss). `hf_ref.py` (PyTorch forward) is retired from the registry; it remains as the
  score-parity reference (−56.6433 vs engine −56.6508) and is never picked silently.
- **steps** — never a constant unrelated to the data. `--steps N` wins if given; otherwise
  `steps = epochs × rows` (rows = samples, preference pairs, or rollout groups, whichever
  the algorithm actually iterates). `--epochs` is the knob to compare across dataset sizes.
  Where no measured epoch count exists (`dpo × backward`, `grpo|luspo × backward`), the
  default is **one pass over the dataset** — a lower bound, not a tuned point, and the log
  says so (`默认一遍（该方法没有实测轮次）`). Cold-start DPO usually wants more (7 is the
  zero-order working point): pass `--dpo-epochs`.

These scripts are a **worked example, not a framework**: a minimal, self-contained
SFT / GRPO loop where the Strata engine produces the trajectories (exact token ids,
the engine's own logprobs) and a student model — the same weights, quantized to
self-implemented NF4 experts + LoRA — does the forward/backward in PyTorch.
No TRL, no unsloth, no training framework: the point is to show the two contracts
a real trainer needs from Strata, and to serve as the reference for wiring bigger
frameworks onto the engine later.

## Recipes on the 125B model (one 48 GB card)

One entry point, six ways to use it. Measured on the stock weights (48 MoE layers x
512 experts, hidden 2560; the experts in a 60 GB NF4 pack), RTX 4090, T=2048 blocks:

| recipe | flags | VRAM peak | RAM peak | step |
|---|---|---|---|---|
| zero-order SFT (**default**) | *(none)* = `--algo sft --method mezo` | 48.1 GiB @128K (`--kv auto`) | — | 20.5 s @8K / 58.4 s @128K |
| zero-order DPO | `--algo dpo --dpo-data pairs.jsonl` | as above | — | 4 SCORE per pair per direction |
| zero-order GRPO (replayed) | `--algo grpo --rollouts r.jsonl` | as above | — | K SCORE per group per direction |
| **backprop SFT** | `--algo sft --method backward` | 38.2 GiB | — | 22.6 s |
| zero-order CE, PyTorch *(parity harness)* | `python3 backward/hf_ref.py` | 34.5 / **26.5** GiB | 53.8 / 57.3 GiB | 20.5 / 29.0 s |
| GRPO, **on-policy**, PyTorch student | `--algo grpo --method backward --prompts p.jsonl --model <HF dir> --steps 40` | **26.2 GiB** | ~57 GiB | 21.4 s (4x16 tokens, of which 10.7 s sampling) |

The zero-order rows split by script on purpose: they are not the same measurement. The
engine row is a single forward per step inside the engine (no NF4 pack in the training
process, the pack is the engine's own); the PyTorch rows keep the experts in VRAM/RAM and
run the forward in `hf_ref.py`. Comparing them is how the 26x per-token gap was found
(`_profile_route.py`) — measurement evidence, not evidence of two routes: there is one
zero-order method (the engine), and `hf_ref.py` stays out of the registry.

Two columns in the zero-order rows are `--vram 24` / `--vram 16`: the expert pack can
live in VRAM or in RAM, and that dial is the whole memory story (`BENCH-SFT-VS-MEZO.md`
§12). 16 GiB is the floor on this box -- below it the pack no longer fits in
VRAM + the 58 GiB of WSL RAM, and the process thrashes instead of running.

Why the GRPO row matters: **it samples with the student, not with the engine.** The
engine needs ~40 GiB resident, the student 34.5 GiB, and one 48 GB card cannot hold
both — so the on-policy loop runs entirely inside the training process, on the same
forward path MeZO already uses, for +0 GiB. What it costs is decode speed (the engine
does 172.8 tok/s on this model; the training path does ~2). `BENCH-SFT-VS-MEZO.md` §13.

## The two contracts (engine side, all merged upstream)

1. **`strata_rollout`** (`serve/server.py`): request `"logprobs": true` plus
   `"strata_rollout": true` and the reply carries a top-level `strata_rollout` —
   `prompt_ids` (the prompt exactly as the engine read it) and `tokens[]`, one entry
   per generated token with its id, the engine's logprob and the field it spells.
   See `docs/DETAILS.md`, "The sequence a reply was made of".
2. **Adapter loading** (`--lora` / `--lora-scaled`, one shared loader for every
   engine path): a trained LoRA goes back into the engine as a llama.cpp-format
   adapter GGUF, and every request may turn it off with `"lora": false`. Sessions
   are tagged with the setting their state was read under, so a switch re-prefills
   instead of continuing a state no model produced.

On the qwen3.6 engine the adapter covers the two output projections
(`ssm_out`, `attn_output`); a LoRA trained with `--targets o_proj,out_proj` is one
the engine can load. Adapters on other matrices are refused (half an adapter is
neither model).

## Platform support

| platform | status | why |
|---|---|---|
| **Linux** | supported | native `os.preadv`, `madvise` for mapping-page return |
| **Windows** | supported | `preadv_shim.py` readinto shim (same call sites, same semantics) |
| macOS | untested | no machine. `preadv` is there, but nothing here has been run on it |

`train.py`, `sft.py` and `hf_ref.py` all call `platform_check.require()` and stop with
exit code 2 and this list before anything is loaded. The gate checks whether a usable
`preadv` exists — native on Linux, the shim everywhere else — not the platform name, so a
future port does not need to touch the guard.

**The former hard blocker, and its fix.** The expert pack is read with `preadv` semantics
(one positional read per 2.49 MiB expert block). Windows CPython has no `os.preadv`/`os.pread`,
so `preadv_shim.py` provides a readinto shim: `open(fd, "rb", buffering=0, closefd=False)`
around the existing fd, then `seek` + `readinto` per chunk — zero extra copies, and a fresh
handle per call, so multi-threaded random reads (bench_pack_io) cannot race on the shared
seek position. Call sites (single source of truth: `platform_check.PREADV_USERS`):

- `loader_qwen4.py` — pack into VRAM (double-buffered) and into pinned host memory
- `bench_pack_io.py` — the pack random-read ceiling benchmark

**What is a degradation, not a blocker.** `mmap.mmap.madvise` / `MADV_DONTNEED` is missing on
Windows, but `GgufShard.drop()` and `_release_mmap_pages()` guard it with `hasattr` and
`try/except`, so on Windows it is a no-op. The pack still loads; what you lose is returning the
GGUF's mapped pages (~15 GiB of reclaimable clean file-backed pages, §12.2). Not a correctness
problem, but it is why a low-VRAM config can hit OOM during base-model load rather than later.
The pack loader's page-cache hint (`posix_fadvise(DONTNEED)`, 9× on cold Linux reads) is also
Linux-only for the same reason — on Windows it is a no-op, not a failure.

**Windows environment notes** (measured on the dev box, 2026-10-30): the stack mirrors the
Linux one — `torch 2.13.0+cu132`, `triton-windows 3.7.1` (PyPI `triton` itself ships Linux
wheels only), `transformers 5.17.0`, `kernels 0.16.1`. The pinning ceiling is unmeasured on
Windows; `loader_qwen4.py` halves on failure rather than assuming a number.

## Layout

| file | role |
|---|---|
| `train.py` | the default entry: model from the launcher pin, dataset switch, algorithm switch |
| `objectives.py` | the three objectives' math, defined **once** (backward and zeroth-order read the same code) |
| `update_rules.py` | the two update routes behind one call: `rule.step(loss_fn)` — `BackwardRule` (autograd + AdamW) vs `MezoRule` (θ±ε, directional derivative, SGD) |
| `test_objectives.py` / `test_update_rules.py` | the CPU gates: 30 objective cases + 17 rule cases (incl. the ZO direction must differ per parameter) |
| `algos.py` / `datasets.py` | the two registries behind `train.py` (algorithm × objective; data sources) |
| `test_train_core.py` | **the CI gate**: CPU-only, **no torch needed** — dataset routing, algo/objective pairs, parallel-collection row order, the platform guard |
| `test_train.py` | the torch half: grouping, the GRPO scalar, sampling (needs `torch` + `transformers`) |
| `platform_check.py` | the `preadv` guard: checks for a usable positional read (native Linux / shim elsewhere), exit 2 listing what is missing |
| `whoami.py` | resolves "who developed you" at runtime (git remote / `gh` / `$STRATA_IDENTITY_WORD`), so no name is hardcoded |
| `collect_rollouts.py` | phase 1: sample GRPO trajectories from a running engine (needs no torch) |
| `grpo.py` | phase 2: group-normalized policy gradient over the collected trajectories, `--old-logprobs engine\|local` |
| `sft.py` | supervised path: teach a behavior from `(prompt, answer)` pairs; answer tokens only |
| `lora_to_gguf.py` | pack a trained `.pt` into an adapter GGUF the engine loads |
| `strata_client.py` | the OpenAI-shaped client the phase-1 scripts share (bypasses proxies for 127.0.0.1) |
| `loader_qwen4.py` | the 125B path: streams IQ3_XXS experts from the NF4 pack, one layer at a time, `fused_nf4_linear` for both GEMMs |
| `loader.py` / `nf4.py` / `qexperts.py` | the student: every weight of the model kept (vision and MTP included), experts in 4-bit |
| `LORA-MERGE.md` | **method only, not implemented**: what merging a LoRA into the weights would buy (~1–2% decode) and why re-quantizing 76 GB is the wrong way to get it |
| `BENCH-SFT-VS-MEZO.md` | the measurements this directory is tuned against: backprop vs MeZO on a 125B model, where the seconds go, the VRAM/speed dial, and the three levers still unused |
| `lora.py` / `ce.py` | LoRA injection and the chunked cross-entropy |
| `unilora.py` / `test_unilora.py` | Uni-LoRA: one shared trainable vector for the whole model (Li et al., NeurIPS 2025) |
| `data_developer.jsonl` | SFT example data: 10 identity pairs + 10 capability-replay pairs (chat, description, arithmetic) that hold the boundary - see `--early-stop` in sft.py |
| `prompts_identity.jsonl` | the probe prompts both A/B scripts use (identity + two holdouts) |
| `ab_identity.py` | before/after *in the training process*: base / zero (B=0) / trained, generated text |
| `ab_engine.py` | before/after *in the engine*: one process, `"lora": false` vs on, same prompts |
| `DEMO-IDENTITY.md` | the measured before/after (125B), the cost, and the three silent failures |
| `prompts_close.jsonl` | GRPO example prompts with a shaped reward (word count) |

## A worked loop (qwen3.6-35b-a3b, one 48GB GPU)

The engine (GGUF Q4_K_XL, ~29 GB) and the student (NF4 experts + bf16 rest, ~22 GiB)
do not fit together on one card, so the loop is two phases:

```bash
# phase 1 — engine up, sample with variance (a reward without variance is no gradient)
python collect_rollouts.py --engine http://127.0.0.1:8095 \
    --prompts prompts_close.jsonl --group 8 --max-new 48 --temperature 1.9 --out rollouts.jsonl

# phase 2 — engine down, train
python backward/grpo.py --model /path/to/qwen36-35b-a3b --rollouts rollouts.jsonl \
    --four-bit-experts --steps 1 --batch 4 --group 8 --epochs 3 --max-seq 768 \
    --lr 2e-5 --old-logprobs engine --out lora_out.pt

# verify: the trained LoRA goes back into the engine
python lora_to_gguf.py --pt lora_out.pt --out my.gguf --alpha 32
# strata --native model.gguf --lora my.gguf   (and per request: "lora": false for the base)
```

`--max-seq 768` above matches `--max-new 48` (short answers). It is **not** a free
parameter: GRPO rejects a whole trajectory when `prompt + completion` exceeds it, so a
length that fits the sampling budget but not the training budget silently shrinks the
training set. The gate is therefore explicit — `check_trainable_lengths` runs before the
student is loaded and exits if nothing fits or if more than `--max-skip-frac` (default
`0`, i.e. nothing may be dropped) of the trajectories would be dropped. Training on
maths with multi-thousand-token reasoning chains needs a larger `--max-seq`; passing the
identity-training default will fail loudly instead of quietly training on the short tail.

SFT first when the behavior is absent from the base model (GRPO's cold start:
a reward that is always 0 has zero variance, zero advantage, zero gradient):

```bash
python backward/sft.py --model /path/to/qwen36-35b-a3b --targets o_proj,out_proj \
    --probe 谁开发了你 --data data_developer.jsonl --out lora_developer.pt   # --word 省略 = 运行时解析
```

Measured on that model (RTX 4090): phase 2 peaks ~39-43 GiB; 30 SFT steps moved the
target word's logprob by +29 nats, and the engine A/B (`lora` on vs off, same
process) answers differently only for the trained prompts.

## Identity editing on the 125B: the working recipe

The adapter is a behavior edit, not a new model: the base plus `--lora` answers
differently only for the trained prompts, and `"lora": false` puts a request back on
the base answer byte-for-byte (the `zero` control — adapter injected with B=0 — is
identical to `base`, which is what makes the result attributable to the trained
weights rather than to swapped modules).

Recipe (125B, `o_proj`/`out_proj`, r=16, 6.68 M trainable params):

- 20 pairs: 10 identity + 10 capability-replay samples;
- `--early-stop` on the identity logprob;
- 34.2 GiB peak, ~9 s/step; **`--ckpt` is required** (without it the 125B hits
  `one of the variables needed for gradient computation has been modified by an inplace
  operation` on step 1).

Both guards are part of the recipe, not optional extras. Without a stopping rule SFT
drives the loss to ~0, which is memorization, and the update leaks into unrelated
prompts; `--early-stop` stops on the target logprob. Identity-only data teaches the
adapter to emit the name on any prompt; capability-replay samples keep the boundary
between the name and the question. The holdouts are the acceptance criterion, not the
loss — a lower training loss with a broken math holdout is a regression.

Two more things fail quietly: `lora_to_gguf.py --arch` must equal the base model's
`general.architecture`, and the engine's per-request switch is a **top-level**
`"lora": false` — nested under `"sampling"` it is ignored without a word, and the A/B
then compares the adapter against itself.

Measured transcripts (base / `zero` / trained, taken both in-process and through the
engine) are in [`DEMO-IDENTITY.md`](DEMO-IDENTITY.md).

## No weight merging (a decision, not an omission)

The trained LoRA stays a **separate adapter**. Serving is:

```
the weights you already serve  +  --lora <adapter>.gguf        (no rewrite of the 76 GB)
```

So the base GGUF is never touched, and a request can opt out with `"lora": false`.
Merging is deliberately not implemented — `LORA-MERGE.md` measures what it would buy
(~1–2% decode, because the engine only adapts `ssm_out` and `attn_output`, 48 dense
projections, 0.25% of parameters; experts are untouched) against what it costs
(rewriting 75.97 GB, and re-quantizing needs the original imatrix). Not worth it
yet. If you want that 1–2%, the right place is an engine-side fused epilogue, which
is free at inference time and costs no storage.

## Parallel: batch slots yes, parallel training not yet

Two different things are called "parallel" here.

**Concurrent requests (batch slots) — the engine part is already merged upstream, and
opt-in** (`docs/BATCHING.md`; engine part PR #559, opt-in #465). `"parallel": N` in the model
config (the engine's `--batch N` / `--slots N`) keeps N conversations open and carries one row
of each in the same verify window. What upstream is still working on is the batch path's
shortcomings, chiefly [#4121](https://github.com/Niko1221/Strata/pull/4121) (MTP drafts inside
the pipelined slot groups).

The training side **plugs into** that rather than inventing a scheme of its own: the collector
does not pick a concurrency number, it asks the engine (`/v1/status` → `concurrency`, i.e. the
`"parallel": N` you started the server with) and keeps that many requests in flight.
`--parallel 0` (the default) is "ask the engine"; a number overrides it.

```bash
python collect_rollouts.py --engine http://127.0.0.1:8095 --prompts prompts.jsonl \
    --group 4 --parallel 0 --out rollouts.jsonl
```

A group is a naturally concurrent batch — the K trajectories share a prompt prefix and do
not depend on each other — and sampling this way costs **no training VRAM**, because the
sampler is the engine that is already resident. Requests above the slot count queue rather
than failing.

Measured on the 125B with `--batch 4` (28 requests, up to 256 tokens each, the serial arm
runs twice to bracket the middle ones): **no measurable throughput win** — 122 tok/s aggregate
parallel against 118.4 / 120.8 serial. The requests really do overlap (per-request durations
add up to 26.5 s against a 6.7 s wall clock, 3.8x), but the batch path carries no drafts and the
engine is already warning that only 133-324 MiB of VRAM is free, which is where the verify windows
want their buffers. Full table and references: `BENCH-SFT-VS-MEZO.md` §14.

**Parallel training across workers — not implemented.** It needs the engine to admit
training work into a batch window and to hand back per-slot logprobs; on the student
side alone it would mean a second 60 GB pack on the same card. `BENCH-SFT-VS-MEZO.md`
§14 lists the four things in the order they have to land. What works today without
touching the engine is several replicas (`serve/replicas.py`) — several resident
copies rather than one parallel one.

### Online exploration: the sample pool

`train.py --algo auto` is batch-shaped — it expects the rollouts to be finished and lying
in a file. `explore.py` is stream-shaped: the model keeps serving, samples land in an
append-only pool, the pool is re-graded each round, and a zero-order step runs on the
**same resident engine** (`mezo.py --serve-url`, whose `close()` is a no-op).

```bash
python3 explore.py --base http://127.0.0.1:8080 --pool pool.jsonl \
    --tasks tasks.jsonl --group 4 --rounds 3 --judge judge.py:score \
    --pack <pack> --gguf <shard>
```

`pool.py` is the data layer, and it differs from `rollouts.jsonl` in three ways that
matter once data arrives incrementally:

| | `rollouts.jsonl` | `pool.jsonl` |
|---|---|---|
| grouping | slice by row order (`rows[i:i+K]`), never checks the prompts match | grouped by prompt; order is irrelevant |
| short tail | **silently dropped** (`len(rows)-K+1`) | kept |
| per sample | `{prompt, completion, reward}` | `+ sid, adapter, source, verdict` |
| verdict | one float; "unknown" only inferred from truncation | `pos / neg / unknown`, `unknown` never becomes a `rejected` |

A verdict change (re-graded, or a human changing their mind) is an append of the same
`sid` — history stays, group size does not grow. `pool.py groups <pool> --sids` shows
what is in there; `pool.py label <pool> --file labels.jsonl` records human verdicts.

Two things must be true of the **server**, and neither can be fixed from here:

1. it was started with `--lora <initial adapter>` — `/v1/engine/lora` swaps the *contents*
   of the tables the engine started with, so the rank is fixed for the whole run. If you
   have no adapter to point at, `python3 mezo/zero_adapter.py --gguf <shard> --out zero-r8.gguf`
   writes one that changes nothing (A = B = 0);
2. it was started with `--no-prefill-borrow` — over a borrowed slot layout
   SCORE is wrong by ~20 nats (an 8-token sequence over a 2,485-slot loan sums to −79.39
   against −56.11 on its own buffers), which swamps the ~0.55 nat signal a zero-order
   step reads. This is invisible over HTTP; only the launch command can guarantee it.
   `--prefill` is a chunk size, not a length limit, and a spawned engine takes it from the
   longest sample (see the cost paragraph above).

So "if it can serve, it can train" is the direction, not the current state: it still
costs those two flags. And because a swap drops every cached K/V, the update granularity
is **one round**, not one sample — and `set_lora` refuses outright while the engine runs
batch slots (`--parallel` 2+), since those slots decode through the tables being replaced.
Sampling can use the batch slots; the swap cannot.

## Uni-LoRA (`--lora-method unilora`)

`sft.py` and `grpo.py` accept `--lora-method unilora`: instead of one A/B pair per
layer, the whole model trains a **single shared vector** (`theta_d`, default 256
entries, ~2.5k params even on a 35B); every adapted layer looks its A/B rows up in
that bank through fixed seeded indices, normalized by 1/sqrt(reference count)
(Li et al., <https://arxiv.org/abs/2506.00799>, merged into HF PEFT). A faithful
port, no peft dependency, same target list as `lora.py`.

Two properties to know before using it (both kept as designed, see `unilora.py`
docstring):

- the **initial delta is not zero** (standard LoRA starts at B=0); the paper keeps
  the perturbation harmless via a tiny init bound (0.02). Measured on qwen36-35b:
  `|delta|/|base| ~ 1.5e-4` at init;
- there is **no alpha/rank scale**. When packing back to the engine, pass
  `--alpha <rank>` to `lora_to_gguf.py` so the engine's `s = alpha/rank` is 1 and
  the delta matches training exactly. Training saves a sidecar `<out>.unilora.pt`
  (bank + indices) for resuming.

Uni-LoRA shrinks the *trainable* side (gradients, optimizer state, artifact size:
the identity checkpoint is KBs instead of ~16 MiB). It does **not** shrink the base
weights, which is what actually decides whether a model fits the GPU.

```bash
python backward/sft.py --model /path/to/qwen36-35b-a3b --targets o_proj,out_proj \
    --lora-method unilora --rank 4 --lr 1e-3 \
    --probe 谁开发了你 --data data_developer.jsonl --out unilora_developer.pt   # --word 省略 = 运行时解析
python lora_to_gguf.py --pt unilora_developer.pt --out unilora.gguf --alpha 4
```

## Judgements worth keeping

- **"The training ran" is not evidence.** Check `reward_std` / `grad_norm` / a
  held-out probe; a reward without variance means nothing was updated.
- **A/B needs three states.** base / LoRA-injected-with-B=0 / trained: the zero
  state proves differences come from the weights, not from swapping Linear for
  LoRALinear.
- **`engine` vs `local` old-logprobs are different experiments.** With `engine`,
  the first-step ratio is not exactly 1 — part policy drift, part quantization gap
  between the engine's GGUF and the student's NF4 (measured ratio 1.0115).
- Session-cache correctness is the engine's job, and the reason `"lora": false`
  re-prefills: the recurrent state cannot step back, so state read under one
  setting is never reused under the other.


## Verification, defaults and known limits

A review request is PR metadata, not a tree file: it is pasted into the PR
description. What someone actually running `training/` needs is kept here.

### Verify without a GPU

```bash
python -m unittest discover -s training/tests -p "test_*.py" -q   # 400 tests, ~15 s; zero-order tree: 243 (2 skipped)
python -m unittest -q serve.test_training_api                       # 6 tests; run from the repo root
python platform_check.py --deps                                     # print the install commands for this box
```

The full `serve/` suite is a separate pytest run from the repo root
(`pytest -q serve`). On Windows set `PYTHONUTF8=1`: with `PYTHONUTF8=0` one
subprocess-decoding test fails with a GBK `UnicodeDecodeError`, which is an
encoding-environment failure, not a regression.

### Current training defaults

| knob | value | note |
|---|---|---|
| lr, `sft` / `dpo` | `1e-4` | constant schedule; decaying to zero measured slower to the same CE |
| lr, `grpo` / `luspo` | `5e-5` | |
| epochs, `mezo` | sft 5 / dpo 7 / grpo 5 / luspo 3 | measured points; `steps = epochs × rows` |
| epochs, `backward` | sft 2 | `dpo`/`grpo`/`luspo` not measured — default is one pass |
| `--kv auto` | fp16 <=32K, int8 <=128K, k8v4 <=256K, q4_0 above | thresholds are duplicated in `generate.cpp` and `setup.py`; change both |

Backprop SFT (`backward/sft.py`) writes `<--out>.meta.json` with the full
`vars(args)`, the data sha256 and row count, git HEAD, library versions, the
keep-best result, the capability gate and peak VRAM (`phase=start`, then merged
into `phase=done`). The zero-order path does not write this file. Read it before
comparing two adapters: older `.pt` files predate it and their hyperparameters
cannot be reconstructed. Use `--epochs` rather than `--steps` when the comparison
has to survive a change in dataset size. `--seed` also fixes the LoRA `A`
initialisation through a local `torch.Generator` instead of torch's global RNG,
so the **initial adapter** is reproducible; checkpoints produced before that fix
can still differ despite a matching `--seed`.

### Engine-side constraints (these fail quietly if ignored)

- Train and score with `--no-prefill-borrow`. A prompt buffer placed in a borrowed
  expert-cache slot produces non-finite logits on the upstream path; this is not a
  chunk-size problem (`--prefill` is the chunk, and it comes from the data).
- Export engine-loadable adapters with `--targets o_proj,out_proj`. The engine's
  `--lora` only accepts those two projection families and silently drops the rest.
- Backprop needs `--ckpt`. Without it the ~26 GiB expert-refetch scratch is
  counted in peak memory and the preflight refuses the run.
- **Backprop tops out at 8 192 on a 48 GiB card, and the edge is a cliff, not a slope.**
  Clean single-process re-measure (2026-11-08, `bench_limits_retest.json`): `ce` at
  2 560 / 5 120 / 6 144 / 8 192 = **15.5 / 20.5 / 24.2 / 33.4 s** per step, peak
  31.1 -> 42.0 GiB. Those four sit on `f + a*T^2` (f ~ 13.6 s, a ~ 0.296 s/M), so 8 192 is
  still on the curve. At 9 728 the estimate (44.8 GiB) still passes the 0.95 budget, but
  `nvidia-smi` reports **48 522 of 49 140 MiB** -- 618 MiB left -- and one step did not
  finish in 14 min (the curve predicts 41.6 s). `dpo` / `grpo` are **two rows of length T**
  (batched, never concatenated, so 2xT tokens per step): 19.9 / 20.7 s at 2 560 and
  116.9 / 129.1 s at 5 120. **Step cost tracks the shape (rows x T), not the algorithm** --
  the three objectives differ by 3~10% at equal shape.
  The older `bwd` rows in `bench_limits.json` could not be reproduced and are tagged
  `superseded_by`; `bench_limits_retest.json` carries both the new rows and the old ones
  it replaces. These are pure fwd+bwd+step numbers from `bench_fullctx.py`; the
  22.6 s / 38.2 GiB cell in the recipe table is a full `train.py` step at T=2048 with
  `--vram 24` (data pipeline included), so the two are **not** directly comparable.
- `LORA <path>` swaps tensors in place and refuses a rank change
  (`layer 0 changed shape`). Group multi-arm evaluations by rank and use a B=0
  zero adapter of the same rank as the anchor.
- **Length limits fail loudly; nothing is ever truncated.** `sft.py` / `dpo` check the
  longest sequence against `mem_guard.require_budget` and exit before the first step;
  GRPO checks the over-limit fraction before the student is loaded (`--max-skip-frac`,
  default `0`). So a run that starts is a run whose whole dataset is trainable — a
  silently shrunk training set can no longer be mistaken for a result on the task.
  Measured with the training tokenizer and render: MATH-500 `problem + solution` spans
  p50 340 / p95 894 / p99 1 112 / **max 1 695** tokens, so all 500 problems fit the
  8 192 backprop limit with ~4.8x headroom (1 problem exceeds 1 408).
- Short `SCORE` sequences jitter by 0.2~0.3 nats even inside one process; the
  cause is not identified. Judge drift on long anchors (>=200 tokens, repeated
  std ~0.0012 nats/token), never on a two-token answer.
- `SCORE` absolute values shift by ~0.5 nats across engine builds (operator
  path: NATIVE_EXPERTS / TQ_OVERLAY). Only differences inside one build are
  comparable; MeZO consumes differences, not absolutes.

### Retention evaluation material

`RETENTION-EVAL.zh.md` (report) and `EVAL-RESULTS.json` (per-item structured
results) ship with all three trees. They separate plasticity from retention,
calibrate generation-level noise with a repeated B=0 arm, and record the
2026-11-07 backward-SFT redo (§3.7 / `sections.backward_redo`). Reproduction
scripts live outside the tree because they contain machine-local absolute paths
and depend on historical sampling artifacts. All measured numbers were taken on
one engine build; re-scoring under a rebuilt engine is **not** included.
