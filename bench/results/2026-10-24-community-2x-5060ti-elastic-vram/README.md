# Per-GPU elastic VRAM on two RTX 5060 Ti 16 GB

Custom serving test, 2026-10-24. This validates the layer-split extension in
[#2522](https://github.com/Niko1221/Strata/pull/2522), by GonzMartinez; it is not a
new implementation or a standard-protocol benchmark.

**Either card can keep 3–5 GiB free while Qwen keeps serving.** A separate Docker CUDA
process successfully allocated 4096 MiB on GPU 1, and Qwen answered correctly
while that allocation remained live. After it exited, both expert caches returned
to their original slot counts. The engine PID stayed the same throughout.

## Build and setup

- Ubuntu 26.04.1 LTS host, i7-12700, 64 GB installed RAM, two RTX 5060 Ti 16 GB,
  NVIDIA 595.91.07. Other services remained running; no other GPU workload except
  the explicitly tested allocation holder.
- Source: [f7ce3c98](https://github.com/inlanger/Strata/commit/f7ce3c9823fd234423a0a97b21d4a31eb278ff2a),
  #2522 head `7737f71a` applied to main `7dd5cf18` (Control 0.2.4.2).
  The constructor conflict preserves both `reasoning_budget_configured` and
  `vram_gpu`. No other implementation changes. Engine reports 0.1.49.2.
- Built inside the existing CUDA development image `strata-local:cuda13`, CUDA
  13.0.2, CMake/Ninja, CUDA architectures `120;86`:
  `.venv/bin/cmake --build build --target strata -j 2`.
- Original Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, layers 28/20 across GPUs `[0,1]`,
  131072 context, int8 KV with 20480 resident cells, MTP 3 / min-p 0.7,
  pipeline windows 2, 11 CPU workers. CPU vision stayed configured.
- Existing expert profile and Huihui v1 projection were retained. Their hashes,
  the binary hash and all experimental settings are recorded in `results.json`
  and the configs. Model-file revision was not recorded. The profile, vector and
  model files are not included. This does not evaluate base-model quality.

## Memory handoff

`reserve_mib` requests total free VRAM on the selected card, rather than that
many additional MiB released. Calls below ran between requests.

| Request | GPU 0 free, MiB | GPU 1 free, MiB | API elapsed |
|---|---:|---:|---:|
| Full caches after warm-up | 395 | 407 | — |
| 3072 on GPU 0 | 3391 | 407 | 11.1 ms |
| 5120 on GPU 0 | 5439 | 407 | 12.6 ms |
| 3072 on GPU 1, GPU 0 restored | 395 | 3245 | 17.6 ms |
| 5120 on GPU 1 | 395 | 5293 | 10.6 ms |
| 3072 on both cards | 3391 | 3245 | 24.1 ms |

All nine measured shrink calls, including repeats, took 10.6–24.6 ms. The nine timed
growth calls took 60–745 ms. Full restoration recovered **8263 / 5937 slots** and the
original memory readings. Three additional shrink/grow cycles showed no growing
VRAM footprint.

For coexistence, a 5632 MiB reserve on GPU 1 left 5805 MiB free. The separate
`hold-vram.cpp` process allocated 4096 MiB there, leaving 1576 MiB free including
its runtime overhead. Qwen answered a planning check correctly. Asking to grow
while the holder remained alive stopped at a partial cache, with 1064 MiB free;
another answer was correct. After the holder exited, full restoration succeeded
in 745 ms.

`null` preserves the configured 700 MiB reserve. Here that did not restore every
slot, because the warmed full-cache process had only about 400 MiB free. Explicit
`0` restored the full caches. The first harness stopped on an incorrect assertion
that `null` must restore every slot; it resumed on the same engine using `0`.
The first-attempt log is retained.

## Answers and observed speed

**25/25 complete responses matched their expected JSON**, including six baseline
checks and nineteen elastic-run checks. Reasoning was high, temperature 0, with
no output cap. The longer extraction answers contained 1691–2970 output tokens;
the shorter correctness checks contained 141–514.

| Extraction run, same task | Decode tok/s |
|---|---:|
| Existing released engine 0.1.48 | 159.6 |
| Rebased PR, elastic off | 165.4 |
| Rebased PR, full elastic caches | 176.7 |
| GPU 0 keeps 5120 MiB free | 172.0 |
| GPU 1 keeps 5120 MiB free | 160.7 |
| Final full restoration | 166.2 |

Each row is one naturally completed answer, with different reasoning lengths
and evolving expert caches. These are observations, not a measured speedup or a
precise slowdown estimate. A roughly 32K-input check also passed with both cards
reserving 3072 MiB (131.3 tok/s) and after full restoration (147.4 tok/s).

## Validation and reproduction

- Passed: CUDA compile/link; `python -m unittest serve.test_vram serve.test_server`
  (367 tests); the 25 hardware answers; 19 resize requests; concurrent 4 GiB
  allocation; partial growth while occupied; repeated full restoration.
- No new NVIDIA Xid, PCIe/AER or kernel OOM entries during the test. AER counters
  were unchanged. The normal server and its unchanged config were restored.
- Not tested: HIP/SYCL/Metal builds or execution, concurrent decoding slots,
  all-resident stages, vision requests, actual image generation, or memory-pressure
  fault injection. The holder allocates memory; it does not run a second model.

The JSON files contain all individual results, prompts, answers, reasoning,
timings and memory snapshots. Configs and logs replace private home paths with
`<HOME>`. This is a custom configuration, not setup defaults.

To repeat: build the linked source revision, fill the local asset paths in the
configs, and run the server on `127.0.0.1:8891`. Run
`python probe.py pr-off leads28_en qa_plan qa_logic` with elasticity off, then
start a fresh elastic server and run `python cycles.py`.

Compile `hold-vram.cpp` inside a CUDA development image with
`g++ -std=c++17 hold-vram.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -o hold-vram`.
The harness uses Docker tag `strata-local:cuda13` for that executable, exposes
physical GPU 1 to it, and requires host `nvidia-smi`. Override `STRATA_TEST_URL`
for another server address. Save a first attempt before using `python cycles.py
resume`; that option resumes the recorded run after GPU 0's null-reserve check.
