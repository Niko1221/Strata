# Qwen3.6-35B-A3B: pictures, several requests at once and speed

Strata runs [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of experts, 3B active, 256 experts, top-8) from
[Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) (UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own
engine path (`src/qwen35`; `strata` switches to it from the file's architecture). Pick it in the app's model list, in `strata launch` or in setup (`--family qwen36`). Besides text it
reads **pictures** and serves **several requests at once**, the same two options the Flash-Next models have.

Measured on an AMD Radeon AI PRO R9700 32 GB (HIP, Linux, UD-Q4_K_XL, every expert in VRAM) and on an RTX 3060 12 GB (CUDA, 20 threads,
UD-Q4_K_XL in the hybrid mode: 22-37 of 256 experts per layer in VRAM, the rest on the CPU), llama.cpp 972b5fc on the same files.

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT with a 2x2 merger,
the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width differs: 2048 here, 2560 there). Strata runs
it with the same helper (`strata-vision`, llama.cpp's `mtmd`), so the encoder choices are the Flash-Next ones
([Images](DETAILS.md#images-vision)). Pictures are on by default, the same rule as for every model on NVIDIA and AMD cards:
the CPU reader under 24 GB of graphics memory (a few seconds per picture, no VRAM), the card's reader from 24 GB (GPU encoder,
~1.2 GB of VRAM at up to 1,024 image tokens; on AMD it is `strata-vision-hip`, with the CPU reader as its fallback).
`--vision yes|no|gpu|cpu` and `--vision-tokens N` work as there; what you ask for is kept, with a note. The Mac engine does
not run this model.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

A picture becomes up to 1,024 tokens of the context (the CPU default is 300). The engine puts the encoder's rows where the prompt has
`<|image_pad|>` tokens and gives every row its 2-D position, interleaved M-RoPE exactly as llama.cpp does for this model (rotary sections
11 / 11 / 10, a picture's rows at (t, h, w) = (p, p + row, p + column), the text after it continuing at p + max(columns, rows)).

**Checked against llama.cpp** (its `mtmd` and its own graph, the encoder on the CPU on both sides, the same prompt ids, the answer
llama.cpp gives fed to both so the logits are compared step by step, 24 steps; the "floor" is llama.cpp's CPU backend against its own
HIP run, the noise two correct engines have on this model):

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor (llama.cpp CPU vs HIP) |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, read in 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request, the text in each | 655 | differs (a near tie: the floor's KL at this step is 0.16 too) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's: `STRATA 42` (with llama.cpp's own spelling ` STRATA42`), `CODE 4217`, `Red circle / Blue square
/ Green triangle`, `A red square is positioned on the left ... A blue circle is positioned on the right ...`. With the picture options on,
a request without pictures gives the same tokens as without them (10 prompts x 128 greedy tokens, identical).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache: the same conversation sent again (a chat
app resends its pictures) continues from the held state when its pictures are the same (they are matched by the server's key for each
picture); a different picture starts that conversation again.

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`, or `--parallel N` at setup) makes the engine keep N conversations open
(`--batch N`, up to 8) and decode them together: every window carries one token of each conversation, so the dense weights, the head and
every routed expert two conversations share are read once for all of them. The server needs no change: it speaks the same engine protocol
as for Flash-Next ([BATCHING.md](BATCHING.md)). A request that arrives while another is read waits only for that read: the prompt of a
new request is read **between** the decode windows of the others, and a conversation continues in the slot that holds its start (or the
engine copies the longer held start from another slot). A 3,008-token prompt read in 0.95 s (3,200 tok/s, 6 chunks) beside three decoding
slots: each of them paused at most 160 ms (one chunk) instead of the whole read, and produced 20 tokens meanwhile (one window per chunk,
the default); `STRATA_Q36_READ_WINDOWS=4` gives them 65 tokens during a 1.14 s read (more of the time, a slower read).
A request that runs alone (the solo path, with its MTP drafts) and is moved into a batch slot when another arrives, and back when it is
alone again, gives the text it gives running alone (the server's hand-over reuses the held state).

Aggregate decode speed, N requests at once, same prompts, greedy, 128 new tokens each, wall time from the first request to the last
answer, **R9700 32 GB, UD-Q4_K_XL, llama.cpp `llama-server -np N`** (5 interleaved pairs, median; Strata `--batch 8 --mtp 0`):

| Requests at once | Strata | llama.cpp | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.9 tok/s | 73.8 | 1.14 | 5/5 |
| 2 | 173.5 | 107.4 | 1.61 | 5/5 |
| 4 | 252.0 | 165.0 | 1.53 | 5/5 |
| 8 | 325.4 | 208.1 | 1.56 | 5/5 |

**RTX 3060 12 GB, UD-Q4_K_XL in the hybrid mode** (each engine with the slots it needs: `--batch N` takes room from the expert slots; llama.cpp
`llama-server -np N -ncmoe 28 -t 8`, the best split of a sweep on this card; 5 interleaved pairs, medians; the box was shared with other
jobs between runs, never during one):

| Requests at once | Strata | llama.cpp | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64.0 tok/s | 41.1 | 1.56 | 5/5 |
| 2 | 85.1 | 52.5 | 1.62 | 5/5 |
| 4 | 99.3 | 63.0 | 1.58 | 5/5 |

Every slot's tokens equal the tokens of the same request alone (the base engine's), at 2, 4 and 8 slots, with a 2.5K-token prompt read beside
decoding slots, with slots reused for later requests, with pictures and with sampling (same seed). Through the server: four concurrent chat
requests give the texts of the same four one after the other.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts (the server sends it alone while nothing else runs). With
several requests the batch window has no drafts unless `--batch-mtp` is in the config's `args` (opt-in): every slot's window then also
carries the proposal its MTP block made (one per slot, up to the rows to spare in the 8-row window) and a confirmed proposal moves two
tokens in the window. Same prompts, 5 interleaved pairs, R9700:

| Requests at once | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 | 5/5 |
| 2 | 173.1 | 207.0 | 1.20 | 5/5 |
| 4 | 251.0 | 275.0 | 1.10 | 5/5 |
| 8 | 324.0 | 320.5 | 0.99 | 0/5 (no row to spare: no proposals) |

The text is the same with and without it (every slot equal to its solo tokens).

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10 attention layers
and the MTP block, 45 KB per token; `--kv` below takes less) and 0.07 GB of recurrent state. Setup writes the context for one conversation, so N slots take N times
that from the room the experts would have: on a card that does not hold every expert (12 and 16 GB, the hybrid mode) that is room the
expert slots lose, and a batch reads more distinct experts from RAM. Setup **offers `--parallel`** where the sessions do not eat the experts' room: up to 3-4 where every expert still fits (3 on a 32 GB card at
32K), and on a card that does not (the hybrid mode) the count whose sessions take at most 35% of the room the experts would have, so it
depends on the context you chose (12 GB: 4 slots at 8K, none at 32K; 16 GB: 2 slots at 32K). It says the price in the same breath, asks
"How many requests at once?" (Enter keeps one at a time), and any number you give is kept. Measured on the hybrid cards (UD-Q4_K_XL, medians of 5
interleaved pairs, `--mtp 2` as setup writes it; "together" = total tokens per second of N simultaneous requests, against ONE request at a time
with the draft layer):

| card, context | slots | one request alone, vs 1 slot | N together, vs one at a time |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, 8K | 2 / 4 / 8 | -3.5% / -4.3% / -12% | 0.99x / 1.15x / 1.10x |
| RTX A4000 16 GB, 32K | 2 / 4 | -10.6% / -29.9% | 1.03x / 0.76x |
| Tesla P100 16 GB, 32K | 2 / 4 | -7% / -23% | 0.98x / 0.81x |

Without the draft layer (`--mtp 0`) the same slots give 1.26-1.64x together (3060 at 8K: 4 slots 1.64x for -5.5% alone), but a single request
with drafts is already about 1.6x as fast as one without, so on a small card the main gain is that a second user starts at once; the
speed gain is small. On a card that holds every expert (R9700 32 GB, 8K prompts) 4 slots run 1.45x one-at-a-time-with-drafts (1.59x with
`--batch-mtp`). The engine says what the sessions take before it allocates them and refuses (with the numbers) a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others: the experts that sit on the
GPU or on the CPU change with the routing of all the requests, and the two compute with different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its
solo run there too, with and without `--batch-mtp`.

## Speed (0.1.45)

Decode and prompt kernels of this engine path were rewritten for speed; **every change on the decode path gives the old kernels' bits**
(`STRATA_Q36_VERIFY=1 strata ... --no-graph` runs the old kernel beside each new one on the real activations and compares them bit for bit:
0 mismatches in 18,360 / 8,187 / 6,399 comparisons on the R9700 and the RTX 3060; `STRATA_Q36_OLD_KERNELS=1` switches them all off). The
10 parity prompts (greedy 128) give the base engine's tokens, plain and with `--mtp 2`. Where the prompt's chunk size changes, the numbers
change a little: see "Prompt chunk".

**What was slow** (rocprofv3, R9700, UD-Q4_K_XL, plain decode, 11.25 ms of kernel time per token): the big Q8_0 products (attention, GDN
projections, head: 1.9 GB per token) run at 530 GB/s, the memory bound. The rest did not: the shared expert's and the attention k/v products
(140 per token) at 18 us, the Q4_K gate/up kernel at 250 GB/s, `k_moe_out` (one block) 17 us, `k_group` (one thread walking 256 experts) 9 us.
Now: those products one warp per row (a launch of 2048 warps took 21 us on gfx1201 where 1024 warps took 5 us, so the grid is capped), shared
gate+up+SwiGLU+q8_1 in one launch, the MoE end with 1024 threads and the next layer's q8_1 image, add+norm+q8_1 in one launch, router top-k and
grouping in one launch, the router and the shared gate in one product, Q4_K gate/up on the V2K kernels (`STRATA_EXPERT_V2K`, gfx12 default).
The prompt's GDN conv runs in tiles of 16 tokens side by side and the delta rule loads one token ahead with two barriers per token (both exact).

R9700 32 GB (p620, HIP gfx1201), UD-Q4_K_XL, base = the engine before this work, 5 interleaved pairs, 3 repetitions each, medians; llama.cpp
972b5fc HIP on the same file (`-fa 1`, best of `-ub` 512 / 2048):

| | base | 0.1.45 | ratio | pairs faster | llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode tg128 | 83.2 tok/s | 117.8 | 1.42 | 5/5 | 81.9-82.1 |
| prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 1.80 | 5/5 | 3,794 (-ub 2048) / 3,042 (-ub 512) |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 | 183.7 | 1.10 | 3/3 | 129 (draft-mtp, n-max 2, earlier run) |
| 1 / 2 / 4 / 8 requests at once (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 | 5/5 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, so the prompt chunk stays 512): decode 80.4 -> 106.9 tok/s (1.33), pp4096
2,160 -> 2,200 (llama.cpp 3,486 at -ub 2048, 2,738 at -ub 512; decode 81.1), `--mtp 2` 154.7 -> 167.3. `--chunk 2048` there moves the experts
to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94 - an option, not the default.

RTX 3060 12 GB (P2003, CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer in VRAM): the GPU part of a
token is the dense Q8_0 at the card's bandwidth (6 ms of the 17) and the rest waits for the CPU pool (9 ms of experts), so only the small
kernels' share is gained: decode tg128 58.5 -> 59.7 tok/s (+2%), pp2048 1,958 -> 2,022 (+3.3%), 10/10 identical, plain and `--mtp 2`
(84.3 -> 84.7). llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48-52.

**Norm and combine kernels in registers.** `k_add_rms_q` (add + norm + q8_1, 40 per token) and `k_moe_out_s8` (the MoE end for 8
experts and a 2048-wide hidden state, 40 per token) keep their values in registers instead of re-reading `h` through a `volatile` load, and
sum the 256 per-thread partials with one barrier instead of nine (or ten) in the same pairing, so every result keeps the old bits
(`STRATA_Q36_VERIFY=1 --no-graph` 0 mismatches in 92,263 comparisons; `--dump-full-logits` identical at 4K and 14K). Per launch, nsys,
RTX 5080 16 GB, Sage-A1.5 Q4_K_M, 4K prompt, forced text: `k_add_rms_q` 6.9 -> 2.3 us, the MoE end 5.4 -> 3.5 us (259 us less per token).
Decode, CLI in the serving shape (262K context reserved, 78 experts per layer in VRAM, `--mtp 0`, forced text, 256 tokens, two interleaved
runs each): 4K 193.4 -> 199.9 tok/s (+3.4%), 16K 169.7 -> 173.9 (+2.5%), 40K 182.8 -> 189.9 (+3.9%). Prompt reading is untouched.

**`--side-stream`: a second stream for the independent matrices of a decode layer.** Off by default. In the DeltaNet layers the GDN alpha / beta and the z product, in the
attention layers the k and v products, run on a second, higher-priority stream beside the layer's first matrix (qkv, q) and join before the
mixer kernel that reads them: the same kernels on the same inputs, so every bit is unchanged (`--dump-full-logits` identical at 4K and 14K, `--mtp 2`
text identical, `STRATA_Q36_VERIFY=1` 0 mismatches). Not used with decode graphs or prompt chunks. Perturbation tests (`STRATA_Q36_SIDE_SKEW`, below) at decode and with `--mtp 2`: <<TEST>>.
RTX 5080 16 GB, Sage-A1.5 Q4_K_M, serving shape, `--mtp 0`, forced text, two interleaved runs each, side stream off -> on: 4K 199.9 -> 202.4 tok/s
(+1.2%), 16K 173.5 -> 176.0 (+1.4%), 40K 190.1 -> 192.5 (+1.3%).
The gain depends on the stream priority: with equal priorities (a GPU or driver with one priority level) it measured +0.3%, so the flag only takes
effect when the driver gives the stream a higher priority (CUDA `hi < lo`; otherwise it says so and is ignored); it is ignored under HIP. It costs about 40 us of host time per token
(the extra launches and four events per layer), so the host's lead over the GPU is smaller; the host-bound shapes (small CPU, many pool waits)
may gain less or lose, which is why it is a flag. At 16K one off-run read 166.1 tok/s (GPU share 70.9%): the
expert prefetch's timing moved that run, not the side stream; the table above is the other runs. `STRATA_Q36_SIDE_SKEW=<us>` /
`STRATA_Q36_SIDE_SKEW_JOIN=<us>` (tests) add a spin kernel before the side stream's first kernel / before the join wait.

**Prompt chunk (the default changed).** The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per
expert: R9700 pp4096 at chunk 512 / 1024 / 2048 / 4096 = 3,453 / 4,786 / 5,766 / 6,223 tok/s. With every expert in VRAM the engine now takes the
largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve and the sessions' caches without giving the packed decode
copies back (8K-64K context: 4096, 100K: 2048, 131K: 512; `--batch 2..4`: 1024 at most so the other slots are not held for a whole chunk,
`--batch 8`: 512). The hybrid mode keeps 2048 (a bigger chunk takes expert slots: RTX 3060 chunk 4096 reads prompts 5% faster and decodes 13%
slower) and goes down to 1024 / 512 on a card with 7.5 / 5.5 GiB free. **`--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the old size, and
then the text is the base engine's, bit for bit.** The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the
prompt state: KL against the 512 chunk on six real 1.5-3.2K-token documents (48 teacher-forced rows each): chunk 1024 4.0e-3, 2048 4.2e-3,
4096 3.6e-3 (per prompt 2.3e-3 .. 5.9e-3), top-1 agreement 44-48 of 48 rows - the size of the llama.cpp CPU-vs-HIP difference on the same
file (4.2e-3), not a quality loss. Greedy 64 tokens after those six documents: the default chunk gives the base engine's text for 2 of 6,
`--prefill 512` for 6 of 6. A prompt shorter than the chunk (every chat turn after the first) is one chunk either way: unchanged.
The maintainers keep the new default's reference outputs (six documents, greedy 64) and check later changes against them.

**Experts streamed to the prompt (hybrid mode).** A prompt chunk of more than `--hybrid-rows` rows reads each layer's experts from a GPU staging
buffer, two layers ahead. By default a layer is copied whole from host memory when fewer than half of its experts are resident, and otherwise
the resident experts are copied from their VRAM slots and only the others cross PCIe (runs of consecutive experts in one copy each).
`STRATA_Q36_STREAM_ALL=1` always copies the whole layer.

- **`--split-stage-copy`**: take the runs from 1/8 of the experts resident (instead of 1/2) and run the slot copies on a high-priority
  stream of their own. A device-to-device copy is a kernel: queued behind the PCIe copies on the copy stream it waited for the compute
  kernels' SMs and held the PCIe copies up. RTX 5080, 78 of 256 experts resident, Sage-A1.5 Q4_K_M, int8 KV, `--max-context 262144`, fresh 16K
  prompt: 4,570 tok/s with both kinds of copy on one stream, 5,760 whole-layer copies, 6,090 with the slot copies on their own stream;
  38 of 256 resident: 5,930 against 5,780. Fresh prompt reads (CLI, repeat runs within 0.6%) went from 5,603 / 5,769 / 4,575 to
  6,583 / 6,145 / 4,681 tok/s at 4K / 16K / 40K; decode is unchanged (193.5 / 169.8 / 182.6 against 193.8 / 169.8 / 183.0 tok/s). The
  bytes in staging are the same (the logits are bit-identical). It is a flag, not the default, because the old rule came from one RTX 3060
  measurement (37 of 256 resident, both kinds of copy on one stream: 1,970 whole against 1,540 in runs) that has not been re-run with
  the slot copies on their own stream: a card with few experts resident may prefer the whole-layer copy, and only one card has been
  measured with the flag.

Two steps are always on, with the same bytes in the same order: each token's MoE input is quantized once and scattered to its sorted rows,
and the per-layer wait for the routing counts is an event wait with the shared expert enqueued first. (`STRATA_QUANT_GATHER=1`, the upstream switch, turns the quantization back to one per entry.)

**Tried and dropped.** The MTP drafts' head over a subset of the vocabulary (`--draft-vocab N`, kept as an opt-in; the answer never
changes): the first 16K-131K rows gain 5-8% on English and code but lose 10-15% on Chinese / Russian / Arabic / Hindi (70% of a Chinese
prompt's token ids are above 65,536), and a candidate list built from the tokens seen so far loses on both. The shared expert on a second
stream (captured into the decode graph) decodes at 36 tok/s instead of 117 with this ROCm. In the hybrid mode, a share of each token's
missed experts copied over PCIe and run on the GPU beside the CPU pool: slower (a Qwen3.6-35B-A3B fine-tune (Q4_K_M), RTX 5080, DDR5-6400 dual channel: 155 instead of 184
tok/s with 30% of them), because the copies read the same RAM the pool already nearly fills (88 of about 96 GB/s); and the CPU's
experts of the last token copied into the prefetch slots (below) after the pool: those recur 18% of the time, and their copies crowd
out the predicted ones. Chunked (WY) GDN was not attempted.

## Opt-ins of this engine path

All off by default; with none of them the engine runs as before. They go in the config's `args` (`strata-qwen36-<size>.json`) or
on the `strata-q36` command line.

- **`--tokens-file F`** (with `--tokens` mode, the main path's name): the prompt ids from a file, commas or whitespace, instead of on the
  command line (a long prompt does not fit one argument).
- **`--kv fp16 | int8 | k8v4`**: the keys and values in less VRAM than the default fp32: 22.5 / 12.7 / 9.5 KB per token and session
  instead of 45 KB (fp16 is #3641's; int8 is q8_0 rows; k8v4 is int8 keys and rotated q4_0 values, as the main path's `--kv k8v4`). fp16 and k8v4
  always decode with the grouped-query kernel below. The main path's `q4_0` is not on this path: in serve mode it is said and
  skipped (the fp32 cache), elsewhere refused. Before this option a config's `--kv` was skipped by this engine; a config that
  carries one now gets that cache.
- **`--gqa-attn`**: the grouped-query decode attention (#3641's kernel) for the fp32 and int8 caches: a block per KV head reads
  each cache cell once for all its query heads (the default kernel reads it once per query head). fp32 switches to it from 8,192
  cells, int8 at every cell: with the int8 cache (a Qwen3.6-35B-A3B fine-tune (Q4_K_M), RTX 5080) it decodes at 184 tok/s (the default kernel 179) at 4K and 165
  (155) at 8K. The int8 kernel (`part_q8x`, 8 query heads per KV head) streams the cache through shared memory (`cp.async`, sm_80+;
  older cards keep the first int8 kernel) and serves all rows of a verify window in one pass over the cache (up to 4 rows a block,
  one session). Its partials have the same bits as the first int8 kernel's (checked by `q36_attn_parity` and by `cmp` of the logits).
  `--attn-mma` without `--gqa-attn --kv int8`, or on a card below compute capability 8.0, says so and is ignored.
  Microbench (not an engine run) of the attention's partial kernel for one layer on an RTX 5080, int8 cache, 16 query / 2 KV
  heads, L2 flushed before each call: 4K cells 48 -> 24 us, 32K 95 -> 69, 90K 229 -> 142, 262K 554 -> 361 us; a 3-row window at
  262K 1,550 -> 925 us.
- **`--attn-mma`** (with `--gqa-attn --kv int8`, sm_80+; `STRATA_Q36_ATTN_MMA` does the same: `1` / `on` / `true` / `yes` turns it on, `0` / `off` / `false` / `no` leaves it off, anything else is said and ignored; the flag wins): the decode attention on tensor cores
  (`part_q8m`): the window's query rows (8 per row) against the int8 cells as fp16 `mma.sync` products with fp32 accumulation, the cache
  read once per window. **Not bit-identical** to the default kernel: q, the softmax weights times the V scale, and the products are
  rounded to fp16 (about 1e-4 relative), K and V enter as their exact int8 values. A row's result does not depend on the other rows of
  its window, so `--mtp` text equals the plain text. KL against the default kernel over the 14K / 57K / 76K set (65 forced steps each,
  Sage, RTX 5080): mean 5.7e-3, median 1.1e-3, p99 0.033, max 0.36 (the `--prefill 1024` floor: 8.9e-3, 1.3e-3, 0.09, 0.51).
  Microbench of the partial kernel for one layer (int8, 16 / 2 heads, L2 flushed): 262K cells 1 row 331 us (the int8 kernel above: 361),
  a 3-row window 425 us (925), 4 rows 461 us (1,200); 32K cells, 3 rows 76 us (157).
- **`--fused-prompt-attn`** (the int8 cache, NVIDIA sm_80+, 8 query heads of 256 per KV head; ignored with a message otherwise): a
  prompt chunk of more than 8 rows runs its attention in one kernel straight from the int8 cells (`q36_pattn.cuh`): a block takes
  one KV head and 8 tokens x 8 query heads, dequantises 32 cells at a time into shared memory, multiplies on the FP16 tensor cores
  (`mma.sync`, FP32 sums), keeps the softmax online in registers and splits the cells over blocks when the chunk is small
  (flash-decoding). The default path converts the whole cache to FP16 for every chunk and layer and writes the FP32 scores and FP16
  probabilities of each slice to memory and back (at 262K cells a slice holds 16 query rows). With the flag the FP16 K / V copies
  and the score buffers (896 MiB = 0.875 GiB at 262K cells, less 16 MiB of split partials) are not allocated: the experts get 12
  more slots per layer (78 -> 90). The sums run in another order, so it is not bit-equal to the default path (not bit-identical).
  Measured on an RTX 5080 (Sage-A1.5 Q4_K_M, 262K context reserved, int8 KV, `--mtp 0`): one layer's attention, 52 new rows at 262K
  cells 23.2 -> 2.7 ms, 300 rows 108 -> 20 ms, 2,048 rows 723 -> 125 ms (kernel microbenchmark, `q36_pattn_parity`); fresh prompt
  read 40K 4,570 -> 5,420 tok/s, 16K 5,768 -> 5,870, 4K unchanged; decode (forced text) +3.5% / +3.6% / +2.1% at 4K / 16K / 40K
  from the extra slots. Against a double-precision oracle the kernel's error is at or below the default path's on every row of
  the 14K prompt's last chunk (10 attention layers, 32,768 rows each). KL of the logits against the default path (fixed 56 slots,
  chunk 2048, 65 forced steps per prompt): 195 steps over the 14K and 57K corpus prompts and the 76K agent prompt: mean 2.06e-2,
  median 1.63e-3, p99 0.094, **max 2.67** (one step of the 14K prompt, where the top two logits swap; the other 194 steps stay
  at 0.21 or less); the 4K and 32K prompts have mean 8.3e-3 / 9.7e-3, max 0.048 / 0.063; two more 14K agent prompts: mean 3.0e-3 and
  1.7e-5, max 0.026 and 6.5e-4. For comparison `--prefill 1024` gives mean 8.9e-3, max 0.51 and the accepted prefetch opt-ins mean
  2.2e-2, max 1.95.
- **`--chunked-gdn`** (NVIDIA, any arch): the prompt's gated delta rule in pieces of 16 tokens instead of one token after another
  (`q36_gdn.cuh`). The recurrence `S_t = g_t S_{t-1} + k_t delta_t^T` is unrolled over a piece: a kernel computes every piece's
  triangular system `(I + L) delta = b` matrix `L` and the output matrix `P` (they do not depend on the state), then one block per
  (V head, 32 state columns) walks the pieces: `U = K W`, `Q W` and the state update `W <- e^c W + K^T delta` as FP32 products
  from shared memory, `delta` by forward substitution (one column per lane, no barrier). The default kernel (`k_gdn_seq_cols`)
  has two barriers per token for 2,048 tokens in series. The state is left in the engine's layout, so decode and verify steps read
  it as before. FP32 sums in another order: not bit-identical. Measured on an RTX 5080 (Sage-A1.5 Q4_K_M): one layer of a 2,048-token chunk
  1.82 -> 0.87 ms (kernel microbenchmark, `q36_gdn_parity`: error against a double-precision recurrence in the outputs 1e-8 .. 1.3e-7 (the default kernel: 5e-9 ..
  2e-8) and in the state 7e-8 .. 9e-7 (default: 3e-8 .. 2e-7), a few times the default kernel's, far below FP16 rounding); 28 ms less GPU kernel time per chunk (nsys, 16K prompt).
  The prompt read of this card is bound by the expert copies (PCIe), so the end-to-end gain shows only where kernels are the
  limit: fresh read 16K 5,760 -> 5,816 tok/s, 40K 4,560 -> 4,705, 4K unchanged (with `--fused-prompt-attn` 16K 5,876, 40K
  5,424 -> 5,582). Decode is unchanged. KL against the default path (195 steps, 14K and 57K corpus + 76K agent prompt): mean
  1.20e-2, median 2.0e-3, p99 0.078, max 0.97 (the 14K prompt: mean 2.3e-2, max 0.97, top-1 58 of 65 steps; the controls are
  `--prefill 1024` with mean 8.9e-3 and max 0.51, and the accepted prefetch opt-ins with mean 2.2e-2 and max 1.95). Chunks of 9-33
  rows are a few microseconds per layer slower than the default kernel. Measured on a Core Ultra 9 285K with 61 GB DDR5-6800.
- **`--lend-staging`**: the prompt staging buffers (0.97 GiB, idle while decoding) hold 13 more expert slots per layer in decode
  (see "Lent staging" below). Off unless given. It is off with `--batch`.
- **`--async-swaps`**, **`--async-after-prompt`**: the expert slots' swaps (every 32 decode rows, `--decode-swaps`; and after a prompt read, up to
  2048) copy on a copy stream instead of stopping the engine stream and the host, into the slot of an expert that already left the table;
  the new expert enters the tables only once its copy has landed (checked at the start of the next forward), so no slot is read before its
  bytes arrived. `--async-swaps` takes the decode ones, `--async-after-prompt` the after-prompt ones too (see "Slot swaps" below). They need
  the pinned expert copy.
- **`--expert-profile F`**, **`--expert-profile-save F`** (the main path's names and file format, `tools/make_profile.py`): in the
  hybrid mode each layer's first expert slots take its most-routed experts from F instead of experts 0 .. S - 1, and at the engine's
  end the experts in the slots then, ranked by the routing counted since the warm-up, are saved to F (a `.tmp` beside it renamed over
  it). A layer here is the engine's: the model's layers, then the MTP block, so a profile of another model or path is refused (and the
  default order used).
- **`--expert-prefetch P`** (hybrid mode, P up to 8): P of each layer's VRAM slots hold experts fetched ahead: the next layer's router
  applied to this layer's MoE input names the experts it will likely pick (64% of its misses are in the top 8: a Qwen3.6-35B-A3B fine-tune (Q4_K_M), RTX 5080, 4K cells), and
  the ones not in VRAM are copied there while the GPU works on the next layer's mixer. A copy is used only once it has landed; until
  then the CPU computes that expert, so which side computes an expert depends on timing and the text can differ from run
  to run in the last bits (the GPU's and the CPU's rounding). Needs the pinned host copy of the experts (else off, said in the log).
- **`--hybrid-zero-copy`** (hybrid mode): the routing and the MoE input go to the CPU, and the CPU's rows come back, through pinned
  host memory the GPU reads and writes in place instead of four copies per layer (each costs the stream 2-3 us). The same bytes.
- **`--decode-swaps N`** (hybrid mode): how many experts move into VRAM slots every 32 decode tokens, over all layers (8 by
  default; the main path's `--adapt-swaps` counts another round, so the name differs).
- **`--pool-warm K`** (hybrid mode with `--expert-prefetch P`; off by default; env alias `STRATA_Q36_POOL_WARM=K`) and **`--prefetch-copies 0|1`** (0 keeps the lookahead and drops the PCIe copies; env alias `STRATA_Q36_PREFETCH_COPIES`): the router lookahead that feeds the prefetch also names the next layer's
  likely experts. The K best that have no VRAM slot and no copy are posted to the CPU pool: its parked workers load, into their own caches, the rows they will compute
  (the same row split by worker id), and when the layer's real call comes a warmed expert's rows go to the worker that loaded them. A warm call of one expert takes about
  9 us instead of 24. The dots are the same, so the logits are bitwise identical (`--dump-full-logits` at 4K and 14K, a pool unit test with 400 random calls). The workers
  load only while idle (a published batch stops them), but the loads read the same RAM as the prefetch copies: with copies on, the two cancel each other. `--prefetch-copies 0` keeps the lookahead and drops the copies. Use them together: `--expert-prefetch 2 --prefetch-copies 0 --pool-warm 3`.  The flag beats the env alias; `--pool-warm 0` is off. The warm-up needs the lookahead (`--expert-prefetch P`, P >= 1) and, to pay, the copies off: with the copies on K=2 or 4 changes nothing, and the two are alternatives on this machine (both read the RAM the pool already nearly fills). Copies off without the warm-up is a loss (-3.6% at 4K, -4% at 16K, -2% at 40K against the default: its two prefetch slots per layer are dead VRAM), and K=4 is slower than K=3 (K=6 and 8 more so: the load no longer fits the idle window). Measured as in the numbers above.
  RTX 5080, Core Ultra 9 285K, DDR5-6800, Sage-A1.5 Q4_K_M, `--max-context 262144 --kv int8 --gqa-attn --hybrid-zero-copy --expert-prefetch 2 --decode-swaps 64` with an expert profile and `--snapshots 2`, `--mtp 0`, forced text, 256 tokens (two runs each, +-1%): copies off, K=3: 4K 203.7 / 204.4 tok/s,
  16K 182.1 / 181.1, 40K 189.5 / 188.5, against the same flags with the copies on and no warm-up 193.4, 169.7, 183.1 (+5%, +7%, +3%).
  With `--mtp 2` (greedy, free text, 4K / 40K): 250.8 / 251.3 and 218.3 / 218.7 against 243.3 / 235.7 and 210.7 / 201.4 (+5%, +6%). With the copies on, K=2 or 4 changes nothing (193.9 / 169.9 / 183.2).
**Lent staging.** A prompt chunk longer than `--hybrid-rows` streams each layer's experts through two staging buffers of a whole layer each
(0.97 GiB in all, 2 x 498 MiB), which decode never touches. `--lend-staging` puts 13 more expert slots per layer in that memory while decoding
(a third slot class after the cache and prefetch slots, numbered from `L.n_reg`; the grouping kernels take a second base pointer). When the
stream starts, the lent slots go out of the tables (no copy: 0.07 ms), and after the prompt the rebalance fills them again with the experts
the prompt routed most, the expert profile's order standing in for experts not routed yet (`STRATA_Q36_LEND_PRIOR=0`: counts only). At load, and
after the warm-up, they take each layer's next experts in the profile's order that are not resident. A lent slot never holds an expert that is
resident elsewhere. RTX 5080, `--max-context 262144 --kv int8 --gqa-attn --hybrid-zero-copy --expert-prefetch 2 --decode-swaps 64`, an expert profile and `--snapshots 2`, forced text, 256 tokens, `--mtp 0`, one run each: GPU share 86.4 / 75.5 / 89.3% to
89.2 / 77.5-78.9 / 91.3% at 4K / 16K / 40K and decode 194.8 / 170.9 / 183.5 to 200.3 / 170.0 and 176.6 / 186.2 tok/s (+2.8% / -0.5% and +3.4% / +1.5%;
the 16K runs disagree, the sync controls repeat within 0.5%); prompt speed unchanged (5,605 / 5,766 / 4,566 to 5,661 / 5,778 / 4,576 tok/s).
The refill is 533 copies (0.95 GiB), which the copy stream does while decoding starts. The log's `lent slots:` line counts evictions and fills.
**Class.** C by default: the 533 lent copies after a prompt go to the swap copy stream and a lent expert is used once its copy has landed, which can
be a forward or two after the first decode token (a repeat of the same run gave the same logits in every KL run, which is not a guarantee).
`STRATA_Q36_LEND_SETTLE=1` makes it B: the lent copies of the synchronous rebalance run on the engine stream and the engine waits for the swap
stream before the first decode forward, so the same input gives the same tokens (two runs, same `--dump-full-logits`). It costs the copy time
after every prompt read (a prompt of 4K tokens reads 2% slower, 16K and 40K 0.5% and 0.2%) and measured about the same decode gain at 16K and 40K but
almost none at 4K (195.0 against 200.1 tok/s), so it is an opt-in. With `--async-swaps` the swaps are timing-dependent (not reproducible run to run) either way.
**What it costs when every request streams a prompt.** Each streamed prompt takes the lent slots out (0.07 ms) and, after it, the usual ~1,100-1,500 cache swaps
(40-45 ms on the engine stream, as without the flag) plus 533 lent copies (0.95 GiB, about 20 ms on the copy stream). A prompt of up to `--hybrid-rows` rows never streams and keeps the lent slots.
**The expert profile.** At load and after the warm-up the lent slots take each layer's next experts of the `--expert-profile` file that are not resident; after a
prompt the routing counts decide and the profile's rank breaks ties (`STRATA_Q36_LEND_PRIOR=0`: counts only). In the CLI runs the two gave the same
speed and GPU share (every run reads a prompt first, so the counts overwrite the profile's choice before the first token); the profile matters between load and the first
streamed prompt. `--expert-profile-save` counts the lent experts as resident. A lent slot never holds an expert that is resident elsewhere.
**Staging tail.** MMQ reads up to `kMmqTail` bytes past the last blob of a streamed layer; the lent slots stay inside the first `max_layer` bytes of each buffer, so the zero tail the
buffers are allocated with is not touched (`STRATA_Q36_STAGE_TAIL_CHECK=1` reads both tails at every stream start and counts those that are not zero: 0 in the runs). Bytes after a smaller layer's last blob are
the previous bigger layer's data both with and without lending; nothing relies on them being zero.
**Slot swaps.** The log's `rebalance:` line says how many calls, swaps and milliseconds the slot swaps took (RTX 5080, DDR5-6800 (the DDR5-6400 figures above are older runs), `--max-context 262144 --kv int8 --gqa-attn --hybrid-zero-copy --expert-prefetch 2 --decode-swaps 64`, an expert profile and `--snapshots 2`,
256 forced tokens after a 4K / 16K / 40K prompt): decode 8 calls, 456 / 369 / 227 swaps, 20.8 / 16.7 / 10.8 ms (0.08 / 0.07 /
0.04 ms per token, 1.5% / 1.2% / 0.8% of it); after the prompt 1100 / 990 / 960 swaps, 44.6 / 40.3 / 39.3 ms (about 40 us a swap).
With `--async-swaps` the same decode calls take 2.0 / 2.4 / 2.8 ms of host time and forced-text decode goes 194.8 / 171.5 / 183.9 to
196.6 / 172.1 / 184.5 tok/s (+0.9% / +0.4% / +0.3%; the sync runs repeat within +-0.4%, so only the 4K gain is outside the noise).
Its class is C: a swapped-in expert becomes visible when its copy has landed and the next forward starts, which can be one forward later
than in the synchronous path; in the 65-step KL runs (two rebalance calls) the logits were identical to the synchronous ones, which does not
carry over to long decodes.
`--async-after-prompt` takes the 40-45 ms out of the prompt read (pp4096
5,609 to 5,937 tok/s) but the 1,000 experts then arrive over the next tokens, and decode of the next 256 tokens is 1.2-1.4% slower: it pays
when the answer is shorter than about 600 tokens. A bigger `--decode-swaps` (128, 256) raises the GPU share (4K: 86.5 / 86.8 / 87.3%) but not the speed,
with or without `--async-swaps`.
`STRATA_Q36_AFTER_SWAPS_PER_KTOK=f` limits the after-prompt swaps to f per 1000 tokens read (default: no limit below 2048).

In the hybrid mode the engine's exit report has a second line, `strata-q36: pool: ...`. For each layer call that had CPU experts it gives the host plan, the
gate/up rows, the h quantization and the down rows in us, the expert bytes read, and GB/s over the row phases (overall, and gate/up and down on their own).
It also gives, per phase, the wait for the workers to park before and after and the drain.
Measured on an RTX 5080, Core Ultra 9 285K, DDR5-6800 (2 channels), Sage-A1.5 Q4_K_M, `--max-context 262144 --kv int8 --gqa-attn --hybrid-zero-copy --expert-prefetch 2 --decode-swaps 64` with an expert profile and `--snapshots 2`, `--mtp 0`, forced text, 256 tokens:
at a 4K prompt a call takes 50 us for 4.0 MB (82-83 GB/s); the park waits are 0.02 and 0.07 us per phase; at 16K and 40K the rate is the same (83 and 82 GB/s).
The pool unit tests `pool_test` and `expert_multi_test` skip on CPUs without AVX-512 VNNI (this one); the other pool tests run.

Test settings: `STRATA_Q36_ATTN_OLD=1` (the first int8 attention kernel, `part_q8`), `STRATA_Q36_ATTN_ROWS=n` (at most n window rows a block),
`STRATA_Q36_GQA_FROM=N` (the cell the grouped-query kernel takes over from; not with k8v4, which always uses it),
`STRATA_Q36_CACHE_WINDOW=0/1` (the expert slots' rebalancing over a window of the recent routing, on with `--kv fp16`),
`STRATA_Q36_LEND_PRIOR=0` (lent slots by the routing counts only), `STRATA_Q36_LEND_SETTLE=1` (settle the lent copies before the first decode forward),
`STRATA_Q36_STAGE_TAIL_CHECK=1` (the staging buffers' zero tail, counted in the `lent slots:` neighbour line), `STRATA_Q36_PF_GEN_START=N` (with `--expert-prefetch`: the prefetch copies' counter starts at N, close to 2^31 to test its wrap),
`STRATA_Q36_PF_CHECK=1` (the host's slot tables against the device's at every forward, the host's against its own bookkeeping, and four
cache slots per layer plus every prefetch slot in the table byte for byte against the host copy; the log's `table check:` line counts the
mismatches; it synchronizes the stream and copies about 0.5 GB a forward, so it is slow; not with the MTP-only placement), `STRATA_Q36_ASYNC_SWAPS=1/2/3` (the two async bits, as the two flags).

## Not done / limits

- Eight slots at most (the window holds 8 rows); `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals. `logit_bias` is not available on this engine path: the
  server refuses a request that has it (as it does with `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over; the MTP block's cache has holes for
  the tokens a slot decoded in batch windows (the drafts of a later solo stretch are a little worse; the text is not affected).
- Intel Arc (SYCL) does not run this family.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed); run them under the box's GPU lock:

- `q36_solo_gate.py` + `q36_cmp_gate.py`: the 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's.
- `q36_batch_gate.py --slots N --waves --long 2500`: every slot's tokens equal its solo tokens, slots reused, a long prompt read beside
  decoding slots. `q36_proto_edge.py`: `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests,
  the context's end. `q36_interleave_probe.py`: what the decoding slots see while a long prompt is read.
- `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp): the picture comparison above; `q36_vision_proto.py`: pictures through
  `GENI` / `BGENI` (solo, in a batch, the conversation cache); `q36_vision_serve.py` and `q36_serve_par.py` / `q36_serve_handoff.py`:
  the same through the server.
- `q36_par_bench.py`: the aggregate-throughput A/B against `llama-server -np N`.
- `q36_attn_parity` (built with the tests or `-DSTRATA_PARITY_Q36_ATTN=ON`, GPU, no model): the grouped-query attention kernels (fp32,
  fp16, int8 and k8v4 cells) against a double-precision reference.
- `native_expert_parity --synthetic-ff512 q4_K/q4_K` (with the tests): the Q4_K down kernel at this model's FF of 512.
