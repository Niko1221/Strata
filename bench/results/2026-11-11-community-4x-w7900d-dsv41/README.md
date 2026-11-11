# Community benchmark on 4 × W7900D (gfx1100), DeepSeek-V4.1-Flash 2.00

Measured on 2026-11-11 by @Phlimosx. Linux source build of `strata-dsv41` 0.1.59
(`d2f096475`, tag v0.2.13) for gfx1100. This is the first in-tree DeepSeek-V4.1
run on Linux RDNA3 dGPUs. One box; not a comparison with the published R9700
numbers. `strata bench run` was not used (that protocol is the Qwen suite).

## Hardware and software

- GPU and VRAM: 4 × AMD Radeon W7900D, 48 GB each, gfx1100, device id 0x744b, 96 CU, wave32
- CPU: two AMD EPYC 9334 (128 threads, 2 sockets, 2 NUMA nodes)
- RAM: about 1 TiB OS-visible; the engine reported ~206 GB free at start
- Storage: NVMe for the pack (not measured)
- PCIe link: not measured
- OS: Ubuntu 24.04, ROCm 7.2.4
- Strata: tag v0.2.13 / `d2f096475817d4666c364c820b0ca830f751cfc5`; engine 0.1.59; HIP source build `STRATA_HIP_ARCHS=gfx1100`
- Background: shared host; four cards isolated for this run. Power limits not recorded.

## Model and configuration

- `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` revision `cb02422c900df489cea5833e8a92171224059b3c`, IQ2_XXS (2.00 bpw) four-shard layout
- Engram `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (111 GB); Unsloth DSpark Q8_0 at the Unsloth repo root; Unsloth meta shard 1
- Official pack recipe (`dsv41_pack_cmd`): noise 128799, markov 256; 40 × 384 experts, 6 used; 132.1 GB experts; no missing GPU kernels
- Vision off
- Context 8192; `--experts 6` (Full); DSpark default (auto / on); `--prefill-chunk` unset (engine 8192)
- Extra env: `HSA_USERPTR_FOR_PAGED_MEM=0` only
- Sampling: greedy (`--ids`, temperature 0)

```text
strata-dsv41 --pack pack-2.00 --gpus 0,1,2,3 --experts 6 --max-context 8192 --ids @ids.txt --n 32
```

Shard SHA-256 (bytes in parentheses):

| File | SHA-256 |
| --- | --- |
| IQ2_XXS-00001-of-00004.gguf (39075714624) | `d46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050` |
| IQ2_XXS-00002-of-00004.gguf (39388449504) | `cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70` |
| IQ2_XXS-00003-of-00004.gguf (39246891616) | `9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3` |
| IQ2_XXS-00004-of-00004.gguf (20527837312) | `9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059` |
| Engram-IQ4_NL.gguf (110595290816) | `eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a` |
| dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf (7976914240) | `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` |

## Method

Engine `--ids` timings (`prompt:` / `decode:` on stderr). Loading and RAM-store fill (15.0 s, 99.8 GB) are outside those lines. Short prompt: the smoke question, 17 tokens, `--n 32` (model stopped at 8 new tokens / EOS). Three fresh processes. Long prompt: one 4096-token synthetic word list, 32 new tokens, one process. Expert cache was cold at each process start; prompt reuse 0. No HTTP / TTFT.

Default layer split (engine DP): GPU0 layers `[0,42]` + MTP, 10.21 GB experts, 35706 MiB free; GPU1 `[1,12]`; GPU2 `[13,26]`; GPU3 `[27,39]` + head.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s | Decode tok/s | TTFT |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| smoke / greedy, DSpark auto | 17 | 0 | 8 | 3 | 51.39 / 54.19 / 53.00 (median 53.00) | 61.83 / 61.51 / 61.06 (median 61.51) | not measured |
| 4096-token, scalar attention (gfx11 default before this PR) | 4096 | 0 | 32 | 3 | 263.51 / 274.60 / 275.16 (median 274.60) | 90.25 / 90.50 / 90.41 (median 90.41) | not measured |
| 4096-token, `DS4_FAST_ATTN=1` | 4096 | 0 | 32 | 3 | 314.48 / 315.92 / 312.07 (median 314.48) | 91.02 / 91.66 / 91.37 (median 91.37) | not measured |

DSpark on the short runs: 2 rounds, 10 drafts, 6 accepted (4.00 tokens a round). CPU-lane experts during decode: 0. Disk: 0.0 MB/token. Engram: 0.04–0.11 ms/token. 15,234 / 15,360 routed experts in VRAM (126 on CPU, unused in these decodes).

`DS4_FAST_ATTN=1` (chunked-prompt WMMA attention; decode stays scalar): 4K prompt median 274.60 → **314.48 tok/s (+14.5%)**. Short greedy ids identical (`671,6102,294,8760,344,11111,16,1`). Decode unchanged. This PR turns that path on by default for Linux gfx1100. A rebuild of this patch with `DS4_FAST_ATTN` **unset** read a 4K prompt at 312.78 / 312.69 / 313.66 tok/s (median 312.78, same band) and the same greedy ids.

A same-ids `--dspark off` short run (diagnostic, not the product default) was **slower**: decode 36.51 tok/s. `DS4_PROF` on that run, GPU 25.67 ms/token: attention 12.14, routing + shared expert 5.37, experts gate/up 3.55, down + post 3.32, CPU-lane wait 1.06, between layers 0.23.

Forced `DS4_SPLIT=8,11,11,10` (even trunk layers): 4K prompt 274.45 vs 263.51 tok/s; 4K decode 91.50 vs 90.25. Too small for an engine change.

Per-run stderr excerpts: [engine-log-excerpt.txt](engine-log-excerpt.txt), [runs.json](runs.json).

## Correctness and limitations

- `tools/dsv41/smoke.py`: reply contains Paris.
- Greedy ×3 identical ids: `671,6102,294,8760,344,11111,16,1` (decodes to the Paris sentence).
- `--dspark off` short run: the same ids.
- `DS4_FAST_ATTN=1` short run: the same ids.
- `ds4_iq_test`: IQ2_XXS and IQ1_M (the 2.00 expert types) passed; overall **FAILED** on IQ4_XS wmma vs scalar y 6.72e-03 (tol 3e-03). 2.00 does not use IQ4_XS experts. Log: [ds4_iq_test.txt](ds4_iq_test.txt).
- Not run: 4,995-position KL/ppl (`session16.sh` is not in the tree); needle / 32K / 128K (context was 8K); CUDA hardware (this patch compiled as `strata-dsv41` under CUDA 12.9 sm_89 and CUDA 13.0 sm_120 with no GPU); SYCL (no DeepSeek engine in that tree); Windows / Metal; `strata bench run`.

`head wait` in the engine log is `cudaStreamSynchronize` on the last device (GPU3: layers 27–39 + head), not an isolated lm_head GEMM. With DSpark off, attention is the largest DS4_PROF slice.
