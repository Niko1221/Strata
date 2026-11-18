# The all-cards split engine on Turing (sm_75) with the Gyro trellis models

This change, on top of PR #5119's sm_70 path, runs the split engine on two RTX 2080 Ti 22 GB (sm_75) with the two
Agention Gyro files of Qwen3.8-Flash-Next:

| file | routed experts | Hadamard rotation | PLE table |
|---|---|---|---|
| `Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf` | gate/up TQK6 (145), down TQK7 (146) | all 144 expert matrices | not in the file: `FNX_PLE_GGUF` |
| `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` | TQ2_T (144) | all 144 expert matrices | in the file (Q8_0) |

The trellis types store 128-value blocks (`third_party/ggml/ggml-common-tq.h`). The rotation
(`prism.hadamard.*` keys) is `y = H_128 (s * x) / sqrt(128)` per 128 input columns with explicit signs, folded into
the weights, so the engine rotates the experts' inputs the same way.

## What was added

- `src/split/src/gdn_prompt.cuh`: the 64 KB shared-memory layout on sm_75 (see Build).
- `src/split/src/gguf.*`: types 144/145/146, string-array keys for `prism.*`, an extra GGUF for the PLE table
  (`FNX_PLE_GGUF`), the small-tensor conversion for trellis files (Q6_K embeddings and head, Q5_K, BF16 router).
- `src/split/src/hadamard_meta.hpp`: reads and checks the `prism.hadamard` keys.
- `src/split/src/hadamard.cuh`: the 128-thread Hadamard butterfly in the main engine's order (`had128`).
- `src/split/src/trellis.cuh`: the main engine's trellis decode (`tq_load`/`tq_apply`, int8 codebook and dp4a).
- Rotation launchers in `kernels.cu` part 9: `had_quantize_q8_1` (decode gate/up input), `had_rows_f16`
  (prompt gate/up input).
- Decode (`moe.cuh`): gate/up via the q8_1 dot with each lane decoded once per job; down via `k_moe_h_had`
  (silu(gate) * up, zero-padded to the card's cover, rotated, q8_1) and `k_moe_down_tq`.
- Prompt (`moe_mma_iq.cuh`, `moe_mma.cu`): `MmTileTq` dequantizes trellis rows to f16 tiles for the tensor-core
  product; `swiglu_h16_had` builds the rotated, zero-padded f16 h for down. Its grid's y extent strides over the
  pairs (rows x 10 experts), because a prompt chunk of 8192 rows has 81920 pairs and a grid's y extent stops at 65535.
- The cover: the 640-wide down input is split 320/320 between two cards, but a rotation block is 128 wide, so
  block [256, 384) straddles both cards. Each card keeps the whole blocks around its slice
  (`card_split.hpp` `cover_lo/cover_hi`: card 0 columns [0, 384), card 1 [256, 640)) and feeds zeros for the
  columns the other card makes. The product is linear, so the two partial sums still add up exactly. It costs
  20% more down-matrix memory on two cards.
- `model.cpp`: trellis types accepted, value-column `expert_cols` with the cover, per-card sign vectors,
  residency sizes that count the cover, `FNX_PF_MOE_Q8` refused for rotated or trellis experts (the big-job kernels
  read neither).
- `gguf.cpp`: the file `FNX_PLE_GGUF` names must itself hold `per_layer_token_embd.weight` and nothing else, and the
  model must not have that table already; anything else (an empty file included) stops the start with a message naming
  the tensor.
- `engine.cu`, `prefill.cu`: a separate rotated copy of the activations for the routed experts (the shared
  expert keeps the unrotated one).
- `strata_control/split_engine.py`: sm_75 is an explicit-on architecture like sm_70 (host cache allowed),
  TQ2_T/TQK6/TQK7 are readable routed types, the cover counts in the VRAM estimate, a `--ple-gguf` file other
  than the model (compared by real path) becomes `FNX_PLE_GGUF` and its RAM counts. The trellis types start the
  split engine only on an explicit `--split-engine on`: Auto keeps the layer split for them, also on the tested sm_86
  and sm_120 pairs, where nobody has run them. With "on" they are refused on AMD (the HIP build of the trellis kernels
  was never built or run, and the AMD self-test has no trellis kernel), and on an NVIDIA card other than sm_75 the
  line says "(TQK6/TQK7 experts on sm_86: not tested)". With the host expert cache the RAM check counts the
  engine's page-locked copy of every routed expert, and for every file the loader converts it counts the converted
  dense tensors it keeps (see Running the engine). `expected()` (asked before the download) and the launch share one
  helper, so they give the same answer on sm_75 and with a separate PLE file; `strata drafting plan|fetch`
  (`drafting_cmd._split_expected`) passes it the engine arguments, so `--ple-gguf` counts there too.

## Build (sm_75, CUDA 13.5)

Built on a host with two RTX 2080 Ti 22 GB, CUDA 13.5 and g++-13. That host also has an older system nvcc (12.4),
so CMake is pointed at 13.5 explicitly:

```sh
cmake --fresh -G Ninja -S . -B build-split -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.5 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_SPLIT=ON \
  -DSTRATA_SPLIT_CARDS=2 -DSTRATA_SPLIT_ARCH=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCMAKE_CUDA_ARCHITECTURES=75-real -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=OFF \
  -DSTRATA_EXPERIMENTAL_SM60=OFF
cmake --build build-split -j 12 --target strata-split-2 split_trellis_test split_fnx_run split_moe_tile_test split_qdot_test split_gguf_test
```

A full build takes about 19 minutes: `kernels.cu` part 2 (the MoE launchers) spends about 16 minutes in ptxas at
sm_75. The trellis launchers are in a part of their own (part 9) so they compile beside it. The build has no errors;
the warnings are the ones the tree already had (`qr`/`dr` used before set in `dense.cuh`, unreachable loops).

The kernels already compiled for sm_75 after PR #5119. One change was needed to run: Turing has 64 KB of shared
memory a block, and the prompt's recurrent scan (`k_gdn_ch_rec`, 32 state columns) asks for more, so the engine
stopped at start with `gdn_prompt: shared memory: invalid argument` and the server fell back to the layer split.
sm_75 now takes the 16-column layout the HIP build uses for the same limit.

## Tests

### Trellis kernels: `split_trellis_test` (needs a GPU and both model files)

```sh
build-split/src/split/split_trellis_test \
  MODELS/Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf \
  MODELS/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf
```

Exit 0 when every check is inside its bound, 1 when one fails, 2 on a usage or file error. For each file it takes
layers 0, 23 and 47 and experts 0, 255 and 511 of the real tensors, 5 seeded random activation rows, and both cards of
the two-card split, and compares against the main engine's CUDA kernels (`strata::kernels`) or a double-precision
reference:

| check | compared with | acceptance | measured (worst of 18 experts) |
|---|---|---|---|
| rotate f16 (`had_rows_f16`) | f16 of the main engine's `hadamard_rows` | 0 values differ | 0 of 12800 |
| rotate q8_1 (`had_quantize_q8_1`) | `quantize_q8_1_rows` of `hadamard_rows` (qs and d) | 0 blocks differ | 0 of 400 |
| prompt tiles gate, up, down (`MmTile`) | the main engine's `iq_dequant_f16` | 0 values differ | 0 of 1638400 each |
| decode gate/up (`moe_gate_up`) | the main engine's `iq_mmvq` on the same q8_1 rows | max diff / max value < 1e-4 | 0 (bit-equal) |
| prompt gate/up (`moe_gate_up_mma`) | double dot of the f16 weights and the f16 rotated x | < 1e-3 | 2.6e-6 |
| decode down, 2 cards summed (`moe_down`) | double dot of the f16 weights and the rotated h | relative L2 < 3e-2 | 1.063e-2 (main engine `iq_mmvq`: 1.067e-2) |
| prompt down, 2 cards summed (`swiglu_h16_had` + `moe_down_mma`) | the same double reference | relative L2 < 5e-3 | 2.14e-4 |
| rotated h over 7000 rows x 10 experts = 70000 pairs (`swiglu_h16_had`), once per file and card | `hadamard_rows` of the same h made on the host, 10 sampled pairs from 0 to 69999 (5 of them at 65535 or above, made by the second pass over y) | max diff / max value < 1e-3 | 3.4e-4 (card 1; card 0: 2.0e-4) |

The decode down error is the q8_1 rounding of h and the int8 codebook; the main engine's kernel on the whole matrix
has the same error (the test prints it beside each result). A difference that is not a finite number fails its bound
(the test checks that first). Result on the final build: `PASS: 0 check(s) failed`, 203 checks.

The 70000-pair case failed before the grid fix: `swiglu_h16_had: invalid argument` on both cards of both files.
Under `compute-sanitizer --tool memcheck --kernel-name kns=k_moe_h_had --kernel-name kns=k_swiglu_h16_had` the test reports
`ERROR SUMMARY: 0 errors` (Gyro-S); before `d_gus` was sized to T x 10 rows the old test read past it (7870 errors in `k_moe_h_had`).

Before the kernels existed (first round) the test did not compile (8 errors: `is_trellis`, `had_rows_f16`, `had_quantize_q8_1` and
`swiglu_h16_had` undefined; `DW` has no member `had`, `had_off`, `had_ffs`).

### The GGUF reader: `split_gguf_test` (ctest, no GPU)

```sh
build-split/src/split/split_gguf_test /tmp      # or: cd build-split && ctest -R split_gguf_test
```

Writes tiny GGUF files. Besides the header checks it had, a separate PLE file must load when it holds only
`per_layer_token_embd.weight`, and be refused when it holds another tensor, repeats a tensor of the model, or
repeats the table of a model that has one. Before the change those three loaded or failed with the wrong message
(3 FAILED); now `all passed`. Review round 2 added an empty extra file next to a model that has the table: it loaded
before (1 FAILED) and is refused now.

### The rule: `tests/split/test_split_engine.py`

```sh
python -m unittest tests.split.test_split_engine tests.split.test_split_gen_keys tests.split.test_split_hcsplit
```

82 tests, OK. The two new ones (`test_sm75_is_opt_in_and_reads_the_gyro_trellis_sets`,
`test_sm75_explicit_on_with_a_separate_ple_file`) failed against the old `split_engine.py`: a `TypeError` on
`trellis_down_bytes`, and the separate PLE file was not scanned. Round 1 of the review added
`test_expected_answers_as_apply_does_on_sm75_with_a_separate_ple_file` and
`test_trellis_experts_start_the_split_only_on_an_explicit_on`, and the RAM and same-file checks in the sm_75
tests; against the round-0 module 4 tests failed (expected() said no where the launch said yes; Auto started
Gyro-S on 2 x A6000; the RAM need left out the pinned copy; the model's own file by another spelling counted its
table twice). Round 2 added four: `test_the_drafting_plan_counts_a_separate_ple_file_as_the_launch_does` (through
`drafting_cmd._split_expected`: it said True at 80 GiB free where the launch says False),
`test_the_ram_rule_covers_what_the_engine_took_on_two_2080_ti` (without the converted tensors a host with exactly
what the rule asks kept 1.0 GiB after a Gyro-S load; it must keep at least 5),
`test_trellis_experts_are_refused_on_amd_and_marked_untested_on_other_nvidia_cards` (4 x R9700 started them) and
`test_sm70_with_the_host_cache_counts_the_pinned_copy_of_every_expert` (2 x V100 with UD-Q4_K_XL: 106.2 GiB; the
rule before round 1 said 60.2). With the drafting tests (`tests.terminal.test_split_parallel`,
`test_drafting_slot_default`, `test_app_first_load_drafter_0214`) 152 tests pass.

### The recurrent prompt scan on sm_75

`FNX_PF_GDN_CHECK=1` runs the row-by-row scan beside the 16-column sub-chunk scan and prints how far apart they are.
Over a qcheck run with Gyro-M (2304 calls, 31 to 4096 rows): largest difference over largest value 5.8e-5 for the
outputs and 5.9e-5 for the end state (fp32 sums in another order).

## Running the engine

`--split-engine on` (or `"split_engine": "on"`) on two 2080 Ti applies the rule in `strata_control/split_engine.py`.
The layer-split configs use `--kv k8v4`, which the split engine does not take (it keeps int8 K/V), so that flag has to
go. The configs measured below are the layer-split configs of the same two models with `--kv k8v4` removed and
`split_engine` on, applied through `apply_run_config` (the app's path); the draft vocabulary is `data/draft_vocab.bin`
and the MTP layer is the one the layer split uses. The engine needs `ulimit -l unlimited`: it locks the 55 GB of
embedding tables.

| | Gyro-S | Gyro-M |
|---|---|---|
| need per card by the rule (experts / 2 + 9 GiB, with the cover) | 21.9 GiB | 24.9 GiB |
| room per card (free memory 22014 MiB minus 700 MiB = 21314 MiB) | 20.8 GiB | 20.8 GiB |
| overflow, (need - room) x 2 cards | 2.15 GiB | 8.25 GiB |
| `FNX_EXPERT_RAM_GIB` written by the rule (the overflow rounded up) | 3 | 9 |
| experts in host RAM at start (engine log) | 2860 of 24576 (3.0 GiB) | 6940 of 24576 (9.0 GiB) |
| experts in VRAM | 21716 | 17636 |
| PLE table | `FNX_PLE_GGUF` (the 50.66 GiB Q8_0 file) | from the model file |
| dense tensors the loader converts (Q8_0/F32 copies on the heap) | 571 tensors, 4.69 GiB (4.21 counted: the table's file bytes are already in) | 168 tensors, 0.05 GiB |
| RAM the rule asks for (`need_ram`; it adds 8 GiB spare) | 60.8 GiB at first, 83.9 after review round 1, 88.1 now | 66.5 at first, 90.2 after round 1, 90.2 now |
| measured after loading (engine process): locked (`VmLck`) | 51.3 GiB | 51.3 GiB |
| measured: page-locked expert copy (`Shmem` in /proc/meminfo) | 26.7 GiB | 32.8 GiB |
| measured: engine heap (`Anonymous` in smaps) | 5.62 GiB | 0.97 GiB |
| measured: drop in `MemAvailable` from before the start | 85.7, 86.9 and 84.3 GiB (three starts) | 86.0 GiB |
| load time | 82-96 s | 52-60 s |

Both start with the arena (`FNX_ARENA=1`), `FNX_PF_HCSPLIT=0` and `FNX_PF_ROWS=4096` (the sm_70 host-cache settings).

RAM: with the expert cache on, the engine assembles every routed expert in page-locked host memory
(`cudaHostAlloc`) and keeps it for the whole run; the overflow experts point back into it. The kernel counts that
memory as `Shmem`, not as `VmLck` (`VmPin` stays 0). `VmLck` is the two embedding tables the engine locks. The
rule used to count only the overflow (3 or 9 GiB), so it asked for 60.8 GiB (plus the spare) for Gyro-S while
MemAvailable fell by 85.7 and 86.9 GiB (two starts) at load. It now counts the whole copy, cover included, and the dense tensors the loader converts: `need_ram = tables +
converted tensors + 3 + conversation cache (4) + routed experts with the cover`, plus 8 GiB spare. The engine's own
log has no line for the copy; the numbers above come from /proc after `sfnx: loaded`. The conversation cache
(4 GiB) fills later, so a full engine is about 4 GiB above the load figure.

Where the RAM goes at load: the engine's heap is the converted tensors plus 0.9 GiB (Gyro-S 5.62 - 4.69, Gyro-M
0.97 - 0.05), and `Shmem` is the expert copy plus 1.0 GiB of exchange buffers (Gyro-S 26.75 - 25.78, Gyro-M
32.84 - 31.87). Those 1.9 GiB are inside the 3 GiB the rule counts for buffers. The server and picture-reader
processes started beside the engine took 1.8 GiB more (`AnonPages` 0.08 GiB before a Gyro-S start, 7.54 after, of
which 5.62 the engine); the spare covers the rest of that. Without the conversation cache the rule now asks 84.1 GiB
for Gyro-S and 86.2 for Gyro-M, against measured drops of 84.3 to 86.9 and 86.0 GiB; a host with exactly what the
rule asks (spare included) keeps at least 5.2 GiB (Gyro-S) and 8.2 GiB (Gyro-M) after load. Before the converted
tensors counted, Gyro-S kept 1.0 GiB.

The same pinned-copy count applies to sm_70, which keeps every routed expert in page-locked RAM the same way: 2 x V100
32 GB with UD-Q4_K_XL at 393216 context asks for 27.5 + 3 + 4 + 71.7 = 106.2 GiB plus the spare (114.2), where it
asked 60.2 plus the spare when only the 25.7 GiB overflow counted. A V100 host with about 68 to 114 GiB free that was
accepted before now keeps the layer split (UD-Q4_K_XL is not converted at load, so only the copy changes it).

Prompt rows: the host-cache settings keep chunks at 4096 rows. A Gyro-S start with `FNX_PF_ROWS=8192` (the default
on larger cards) and one prompt of 10215 tokens: before the grid fix the request failed with
`swiglu_h16_had: invalid argument`; now it answers (HTTP 200, the first chunk of 8192 rows, 10215 tokens read in 5.7 s).

## Measured

Both engines on the same two 2080 Ti in one session (2026-11-18), one engine at a time, both cards idle before each
run, same benchmark script. Each run sends 1, 2 or 4 chats at once, each with a prompt of about 11.4k tokens and 600
generated tokens, first as new chats and then as a follow-up turn in the same chats (numbers: new / follow-up). Both
engines draft with the MTP layer, up to 4 tokens ahead (`--spec 4` on both). Steady is the decode speed in tokens per
second while all chats generate; total is all generated tokens over the wall time, prompt reading included. The layer
split is the main engine split across the two cards (layers 0-23 on CUDA0, 24-47 on CUDA1).

| model, chats | layer split steady | split engine steady | layer split total | split engine total | first token, mean (layer split / split engine, new) |
|---|---|---|---|---|---|
| Gyro-S, 1 | 75.3 / 71.9 | 78.9 / 83.1 | 62.0 / 62.3 | 69.0 / 74.7 | 1.71 / 1.09 s |
| Gyro-S, 2 | 126.2 / 127.3 | 102.8 / 104.0 | 92.4 / 95.4 | 86.4 / 90.2 | 2.56 / 1.67 s |
| Gyro-S, 4 | 164.4 / 163.2 | 135.1 / 130.8 | 110.8 / 112.4 | 74.9 / 108.4 | 4.29 / 5.74 s |
| Gyro-M, 1 | 65.4 / 61.9 | 74.5 / 75.3 | 51.6 / 51.3 | 59.6 / 67.9 | 2.46 / 2.02 s |
| Gyro-M, 2 | 112.7 / 109.8 | 103.9 / 102.8 | 74.8 / 76.9 | 76.8 / 88.0 | 4.01 / 2.95 s |
| Gyro-M, 4 | 147.1 / 146.2 | 132.2 / 130.0 | 90.2 / 91.4 | 64.5 / 106.4 | 6.38 / 8.22 s |

Drafted tokens accepted (Gyro-S, 1 / 2 / 4 chats): layer split 59 / 65 / 67%, split engine 52 / 58 / 66%. Gyro-M:
layer split 59 / 66 / 65%, split engine 48 / 59 / 66%.

Gyro-S on the split engine with drafting off: steady 58.9 / 59.1 (1 chat), 94.9 / 94.8 (2), 130.3 / 130.0 (4). The
layer split cannot run a native pack with drafting off (it needs `--spec 2` or more), so there is no pair without
drafting.

- One chat: the split engine decodes faster (steady Gyro-S +5 to +16%, Gyro-M +14 to +22%) and gives the first token
  sooner.
- Two and four chats: the split engine is slower (steady Gyro-S -18 to -20%, Gyro-M -6 to -11%). At four chats it
  reaches about 130 tok/s with drafting on or off, so the limit is how the engine runs a batch, not the drafting. Not
  profiled; a likely cause is that both cards exchange and add their partial sums after every layer, while the layer
  split runs groups of chats on the two cards at the same time.
- Four new chats at once wait longer for the first token on the split engine (mean 5.7 and 8.2 s, worst 14.4 and
  19.2 s): it reads the prompts one after another.
- Other differences between the two configs: K/V cache int8 on the split engine against `k8v4` on the layer split;
  `--spec-min-p 0.5` on the layer split; `--lookup 15` on the split engine; each engine's own expert cache.
- Cold long prompt (Gyro-S, one request, measured on an earlier day): 57937 tokens read in 31.5 s (1839 tok/s) and
  56383 in 30.8 s (1834 tok/s); the layer split read about 2000 tok/s then (not rerun the same day).
- qcheck (greedy, 11 questions): Gyro-S 11/11 on the split engine and on the layer split; Gyro-M 10/11 on both. The
  Gyro-M miss is the same on both engines (reversing the string "strata" gives "artats"), so it comes from the model.
  The greedy answers match the layer split's except one Gyro-S needle answer that is worded differently (same fact).
  On the final engine build: Gyro-S 11/11 (loaded in 88.1 s). Gyro-M was last checked one build earlier (10/11, the
  same miss); the last engine change only touched the check of a separate PLE file, which Gyro-M does not use.
