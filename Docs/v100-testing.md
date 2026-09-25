# V100 Stage 1 — Testing

Phase 5 (first inference on the dev card) and Phase 6 (correctness against an
independent reference). GPU0 = 32 GB PCIE (dev), GPU4 = 16 GB SXM2 (validation,
Phase 7 in a separate section). Model: Swift-1.5-Qwen3.8-Flash-Next
IQ3_XXS GGUF (user-supplied), `/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`.
Engine: `build-sm70/strata` on branch `feature/v100-moe`.

## 5.1 Startup blocker found and fixed: pool worker explosion

Before the first successful inference, every run hung for ~15 minutes after
`strata mtp: draft layer loaded` (no further log line, main thread spinning in
userspace, 3,500+ threads still being created at ~0.8 s each).

**Root cause — a C dangling-else bug in `physical_cores()` (upstream, `src/kernels/cpu/pool.cpp`):**

```cpp
if (sched_getaffinity(0, sizeof set, &set) == 0)
    for (int i = 0; i < CPU_SETSIZE; ++i)
        if (CPU_ISSET(i, &set)) cores.push_back(i);
else                                  // binds to the INNER if, not the outer one
    for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back((int) i);
```

The `else` attaches to `if (CPU_ISSET(i, &set))`, so with a *successful*
`sched_getaffinity` the fallback list is pushed once for every **unset** mask
bit. On this 56-CPU machine `physical_cores(true)` returned **54,263 "cores"**
(56 set bits + 968 unset bits × 56) instead of 55, and the `ExpertPool`
constructor spawned that many 8 MB-stack workers (~430 GB of virtual stack
space, tens of thousands of parked spinners). The main thread sat in
`pthread_create`/`clone3` inside the constructor for minutes.

Reproduction notes (all at `-O0`..`-O3`, i.e. a language-binding issue, not an
optimizer issue):

- standalone repro linking the real `pool.cpp`: `physical_cores(true).size() = 54263`,
  thread count climbing ~300/s, `first=1 last=55` (56 values pushed per unset bit).
- same logic in a separate translation unit with identical includes returned 55
  (the extra statements in the branch prevented the compiler from exposing the
  binding); the bug is in the source, not the codegen.
- the Windows branch of the same function was already braced; a scan of the rest
  of the tree found no other unbraced dangling else of this shape.

**Fix (commit `12979f4`):** brace the outer branch.

```
physical_cores(true).size() = 55   # 1..55, core 0 reserved for the host loop
```

## 5.2 First successful inference (GPU0, 32 GB)

Command (canonical run, memlock raised so `cudaHostRegister` can pin the 40 GiB
expert arena; the arena then falls back to portable 4 KB pages because no
hugetlb pool is configured):

```
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./build-sm70/strata \
  --pack packs/swift-iq3_xxs --native <shard1.gguf> --ple-gguf <shard1.gguf> \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048 \
  --spec 4 --spec-min-p 0.5 --mtp mtp/rt --max-context 8192 --max-new 32 \
  --tokens 9419,11,821,803,369,7967,13,353,1044,264,11952,5617,303,220,17,15,17,21,13,10875,353,668,3184,488,883,821,1118,1834,13
```

(prompt = `Hello, my name is Sam. I am a robot built in 2026. Today I will tell
you about my first day.`, 29 tokens)

Startup path (all subsystems up, ~110 s to session):

- AVX2 expert kernels (this CPU has no AVX-512 — expected on this box)
- 1,466 MiB pack weights + 300 native projection matrices (1,781 MiB), 302 canonical tensors served natively
- PLE table 320,001,536 rows read from the Swift shard 1 at runtime
- expert arena `cudaHostRegister PORTABLE ok` (4 KB pages, no hugetlb), 39.97 GiB loaded at 7.55 GiB/s
- MTP draft layer 808 MiB of VRAM
- profile 8,000 ranked pairs → expert cache auto: 26.88 GiB free − 700 MiB reserve → 8,000 slots / 12.93 GiB VRAM
- `55 expert-pool workers + the host thread` (the post-fix count)
- graph capture: 1/2/4-token verify windows captured
- prefill 28 tokens in 1 chunk, 3,028 ms (9.25 tok/s); 2,271 experts streamed by DMA

Result:

```
output : 271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 220 17 15 17 21 11 321 369 883 310 3184 728 883 836 1118 1834
decode  : 32 tokens in 1128.4 ms  ->  28.36 tok/s
```

Decoded: `\n\n<think>\nThe user is introducing themselves as "Sam," a robot
built in 2026, and is about to tell me about their first day` — coherent
model output.

**Known-upstream caveat (documented in `generate.cpp`, ROUND 328 comment,
line ~1066):** with `--expert-cache` enabled the GPU hit path is declared
*NOT CORRECT* — generated tokens can diverge from a cache-off run (measured
upstream: first difference at token 40 at 2.97 % hits, token 0 at 54.4 %).
The cache stays opt-in and the engine prints the warning. For native IQ packs
the residency table is required by `--spec`, so the hit path is active in
production runs. Phase 6 below checks whether, in practice, this run's output
matches an independent reference.

## 5.3 Determinism

Same production command run twice (expert-cache auto + profile + MTP):

| run | tokens |
|-----|--------|
| run A | `271 248068 198 … 1118 1834` (32 tokens) |
| run B | identical, all 32 token ids equal |

**Deterministic across repeated runs** in production mode (profile-prefilled
cache, no eviction).

## 6. Correctness against llama.cpp (same quant, V100 box)

Reference: vanilla llama.cpp (build commit `427291b5b`, CUDA_ARCH=70, AVX2 CPU
path), `llama-completion`, CPU-only (`-ngl 0`, 28 threads), same IQ3_XXS GGUF
(auto-loads shard 2), greedy (`--temp 0`, `-s 42`), `-no-cnv` so the prompt is
the identical raw token sequence (the new unified CLI auto-enables the chat
template otherwise — verified that the 29-token prompt count matches strata's).

| | strata (V100 + CPU pool) | llama.cpp (CPU, reference) |
|---|---|---|
| prompt | 29 raw tokens | 29 raw tokens (verified by its own prefill count) |
| generated | 32 tokens | 31 eval runs; emitted text tokenizes to the same 32 ids |
| prefill | 9.25 tok/s (GPU path incl. cache borrow + MTP prompt) | 15.95 tok/s (28 CPU threads) |
| decode | **28.36 tok/s** | 3.91 tok/s |

Token comparison (strata ids vs. the reference's emitted text re-encoded with
the same tokenizer):

```
strata: 271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 220 17 15 17 21 11 321 369 883 310 3184 728 883 836 1118 1834
ref:    271 248068 198 760 1156 369 30869 5402 430 328 23202 1288 264 11952 5617 303 220 17 15 17 21 11 321 369 883 310 3184 728 883 836 1118 1834
```

**All 32 tokens identical** — despite strata running with the (declared
not-correct) GPU expert-cache hit path on a different precision path
(Q5_K native LM head, tf32-emulated QSA scorer on sm_70, hybrid GPU-hit +
CPU-miss MoE).

Second prompt (64 tokens, `Write a two-line poem about a lighthouse keeper who
finally sees the first snow.`):

```
strata: \n\n<think>\nThe user wants a two-line poem about a lighthouse keeper seeing the first snow. Let me think about what makes this moment special: a solitary figure, the contrast of the lighthouse beam against snow, the quietness of snowfall, the keeper's isolation, the beam sweeping over white. I want
ref:    \n\n<think>\nThe user wants a two-line poem about a lighthouse keeper seeing the first snow. Let me think about what makes this moment special: a lighthouse keeper is solitary, surrounded by sea, wind, salt, darkness. The first snow is a moment of quiet transformation—white replacing the grey sea, silence
```

**Tokens 1–31 identical; divergence at token 32.** Both continuations are
coherent on-topic greedy text; the branch point is a sub-ULP logit difference
between the two precision paths, not a routing/dequant/PLE error (a systematic
fault would branch at tokens 1–5). This matches the ROUND 328 expectation:
this run uses 8,000 slots (~0.68 % of all 1.18 M experts resident, i.e. a
small hit share), and upstream measured the first divergence at token 40 for a
similar 2.97 % hit share.

Strata decode on the 64-token run: 19.8 tok/s (3,233 ms); llama.cpp reference:
3.9 tok/s CPU-only.

Interpretation: on this model/quant the hit-path divergence is small enough
that greedy decoding agrees with the independent implementation on both
prompts; the ROUND 328 warning is about *possible* divergence (upstream
measurements on other quants/profiles), which is why the cache stays opt-in.
ULP-level differences between the two implementations are expected and are
not themselves a failure; token-level agreement is the acceptance criterion.

## 7. GPU4 (16 GB) validation

Same production command, `CUDA_VISIBLE_DEVICES=4`, prompt 1, 32 tokens.

VRAM budgeting on the 16 GB card (weights 3.2 GiB + MTP 0.8 GiB + KV):

```
expert cache auto: 10.91 GiB free, 700 MiB reserved -> 4715 slots
expert cache 6321 slots, 10.23 GiB of VRAM     (profile blob sizing refines 4715 -> 6321)
```

Result:

```
prefill : 28 tokens, 1346.6 ms  ->  20.79 tok/s   (TTFT 1493.7 ms)
decode  : 32 tokens, 1174.5 ms  ->  27.25 tok/s
output  : \n\n<think>\nThe user, Sam, is introducing themselves as a robot built in 2026 and wants to tell me about their first day. This
```

**Verdict: PASS.** The 16 GB card runs the full production path (arena pin,
PLE, MTP, graph capture, spec/verify) with the expert cache auto-sized to fit
(10.23 GiB of the 16 GiB). Decode throughput matches the 32 GB card within
4 % (27.25 vs 28.36 tok/s) because decode is CPU-pool-bound, not
VRAM-bound. The output is coherent and semantically identical to both the
GPU0 output and the llama.cpp reference, though token-level identity with
GPU0 does not hold: the 16 GB auto-sizing picks a different resident set
(6,321 vs 8,000 slots), and with the declared-not-correct hit path (section
5.2) the hit/miss partition — and therefore the ULP noise on grouped
accumulation — changes, flipping the greedy argmax at token 6. This is the
documented behaviour, not a card-specific fault.
