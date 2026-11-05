# Community benchmark: DFlash2 vs MTP per-task acceptance and decode on 2x RTX 4060 Ti

Measured on 2026-11-04/05 by yeyeyeason. A six-arm drafter-configuration sweep with a per-task acceptance/decode
replay method; the main limitation is one machine, short temperature-0 prompts, and 1-2 measured rounds per arm
(five rounds on two decisive arms).

## Hardware and software

- 2x NVIDIA GeForce RTX 4060 Ti 16 GB; Intel Core i5-10600KF; 96 GB RAM; NVMe SSD; PCIe: GPU0 x8, GPU1 x4 (probed by the engine):
- Windows 11 Pro; driver 617.42; CUDA 13 (engine's bundled runtime):
- Strata v0.2.10 checkout, commit `ffa80d73`; engine 0.1.56 official release binary
  (`strata-engine-0.1.56-windows-x86_64-cuda-cuda13`, SHA-256 verified):
- Background: Mem Reduct running (unchanged across all arms); no other GPU workload:
- VRAM guards: `--vram-reserve-mib 2048 --vram-reserve-later-mib 1536 --conversation-cache-slots 2`
  (raised after a reproduced #2065 WDDM paging stall during earlier runs; see the linked issue comment for the stall evidence)

## Model and configuration

- SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated, IQ3_S (2-shard GGUF); ISTA-DASLab tokenizer pack `sc117_iq3_s`:
- Vision encoder: CPU (`strata-vision.exe`, mmproj BF16), 6 threads:
- Context 1,048,576 (YaRN x4 over trained 262,144); KV int8, 32,768 resident; conversation cache 16 GiB:
- MTP runtime `MTP_DIR`; DFlash2 pack `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744`
  (Q4_0 + selector-Q4_0 converted pack, SHA-256 verified against upstream pins):
- Frozen expert profile (copy of the machine's learned profile, byte-identical hash across all arms); sampling
  temperature 0, no thinking; per-arm `user_args` pinning:

```text
# MTP arm
--mtp MTP_DIR --spec 4 --spec-min-p 0.5 --pipeline-windows 2
# DFlash2 arms (one of:)
--draft dflash2 --draft-model <pack> --draft-fallback none --mtp-draft-vocab <vocab> --spec {4,6,8} --draft-conf {0.35,0.50,0.70} --pipeline-windows 0
# shared: --pack PACK_DIR --native <shard1> --ple-gguf <shard2> --expert-cache auto
#         --prefill auto:16384 --kv int8 --kv-resident 32768 --max-context 1048576 ... --ple-io direct
```

The sweep's effective engine command lines are in each arm's `selfcheck.txt` (regex-verified against the pinned
values at every start).

## Method

Six arms run serially on the same server code, one engine at a time, each: start -> self-check of the effective
command line -> one warm-up request -> three prompts x two rounds (round 1 fresh, round 2 prefix-reused) ->
unload. Two decisive arms (MTP, DFlash2 conf 0.70 spec 6) were later repeated with five rounds each.
Prompts (in `prompts.json`): a Python merge-function task (44 prompt tokens, 256 cap), a Chinese mystery-fiction
opening (42 tokens, 384 cap), and a FIFO scheduling reasoning task (57 tokens, 256 cap), all temperature 0.

**Replay method** (the part maintainers asked to be published alongside the data):

- **Full-request weighting**: arm/task decode tok/s is the mean weighted by generated tokens, not the unweighted
  mean of run values. Warm rounds (prefix reused) and the cold start are reported separately; quoting only the
  warm round-2 value is what the replay showed to be a selection artifact (see Results).
- **Acceptance** is accepted drafts / offered drafts as the engine reports per request (drafts accepted N of M).
- **Mixed workload score**: harmonic mean of code and fiction decode tok/s (equal token counts), so a policy that
  is fast at one and slow at the other cannot hide behind the arithmetic mean.
- **Oracle check**: the replay computes the score of a hypothetical policy that picks the best measured arm per
  task, and compares it with the best single static arm. If oracle ties static, per-task switching has no headroom
  on this data — this is the check that killed the task-adaptive variant of the proposal in #5271.
- Scripts: `sweep_dflash2.py` (arm runner with self-checks, per-request incremental result files, stall guard),
  `repeat5.py` (five-round repeat), the replay itself in `replay.md` (weights, harmonic mean, oracle).

## Results

Per-task decode tok/s (token-weighted over all measured requests; warm = prefix-reused rounds):

| Arm | Code (warm) | Fiction (warm) | Reasoning (warm) | Code acceptance | Fiction acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP spec 4, spec-min-p 0.5 | 64.1 | 38.9 | 44.6 | 85% | 64% |
| DFlash2 conf 0.35 spec 6 | 73.6 | 33.0 | 46.5 | 75% | 33% |
| DFlash2 conf 0.50 spec 6 | 77.0 | 35.0 | 47.4 | 82% | 43% |
| DFlash2 conf 0.70 spec 6 | **79.9** | 35.1 | 47.1 | **90%** | 52% |
| DFlash2 conf 0.35 spec 4 | 68.7 | 36.1 | 46.1 | 85% | 37% |
| DFlash2 conf 0.35 spec 8 | 78.6 | 32.0 | 49.1 | 74% | 32% |

Five-round repeat (mean +/- stdev, Welch t vs MTP, n=5, all-rounds):
DFlash2 c0.70 code 61.2+/-3.7 vs MTP 55.0+/-1.8 (+11.2%, t=3.3, p<0.05);
fiction 34.7+/-0.9 vs 37.4+/-0.7 (-7.2%, t=-5.3, p<0.05); reasoning 47.6+/-1.9 vs 45.3+/-1.4 (+5.1%, t=2.2, borderline).

Mixed workload (code:fiction 1:1, harmonic mean of warm values): MTP 48.4; DFlash2 static conf 0.70 48.8;
DFlash2 oracle (best arm per task) 48.8 — **oracle ties the best static arm**, so per-task switching has no
measurable headroom on this data. Correctly-weighted, the earlier "conf 0.35 fastest on code" was a warm-round
selection artifact; conf 0.70 is equally fast on code with the highest acceptance.

Per-round drafting cost (engine's `STRATA_DF2_PROFILE`/`STRATA_DF2_TIMING`, same arm): ~5.6 ms per round,
22-27% of a decode round at 48 tok/s; largest stage the draft-head matmul (1.08 ms). CUDA1 free VRAM with
everything loaded: 971 MiB (below the engine's own 1024 MiB #2065 warning line — the raised reserves are why
these runs did not stall).

Per-run data: `per-request.json` (all six arms, every request, from the engine logs), `repeat5/` (five-round
raw values), `quality/` (13-case known-answer spot check on three arms: DFlash2 c0.70 12/13, MTP 10/13,
DFlash2 c0.35 s8 9/13 round 1), engine logs and selfcheck command lines per arm.

## Correctness and limitations

- One machine, one model and quantization, three short prompts, temperature 0, 1-2 rounds per arm for the sweep
  (five rounds only for MTP and DFlash2 conf 0.70); no 1M-context runs, no sampled decoding, no agent/tool
  workloads, no other backends. Quality checking is a 13-case known-answer spot check, not a suite.
- Five sweep requests were lost to the reproduced #2065 WDDM stall before the VRAM guards were raised (their
  engine-side completions were recovered from the engine log; the client-side rows for those five are absent).
- The acceptance/decode inversion at high confidence (acceptance up, decode flat-to-down) matches the
  independently reported #5206 (2x 4060 Ti, conf forced 0.5 -> 0.7): acceptance parity without decode parity.
- Not measured: prompt throughput per arm (short prompts, uninformative), TTFT per arm, memory deltas between arms.
