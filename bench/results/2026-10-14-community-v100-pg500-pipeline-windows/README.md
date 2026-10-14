# Community benchmark on Tesla V100-PCIE-32GB + Tesla PG500-216: `--pipeline-windows 2` on a two-stage layer split

Measured on 2026-10-14 by Warabii1248. Tested: `--pipeline-windows 2` against off, everything else equal, on a two-card sm_70 layer split.
Main limitation: one model, one machine, two engine starts per setting.

## Hardware and software

- GPUs: Tesla V100-PCIE-32GB (CUDA0, layers 0-23, PCIe Gen3 x4) and Tesla PG500-216 (a 32 GB sm_70 compute card, the same
  architecture as the V100; CUDA1, layers 24-47, PCIe Gen3 x16). A GTX 750 Ti driving the display was not used.
  nvidia-smi reports P2P as OK between the two cards; the engine passes the hidden state between the stages through pinned host RAM (no `--peer-device`).
- CPU AMD Ryzen 7 5700X3D (8 cores, 16 threads, no AVX-512); 30 GiB RAM; the GGUF files on a SATA SSD.
- Ubuntu 26.04.1, kernel 7.0.0-38, NVIDIA driver 580.178.04, CUDA 12.9.
- Strata: source build of `main` at `296b59d` (2026-10-14, engine 0.1.44.1) with no changes, `-DSTRATA_EXPERIMENTAL_SM60=ON`,
  `CMAKE_CUDA_ARCHITECTURES=70`.
- Background: the machine's other GPU service was stopped during the runs. Power limit 170 W and 250 W, set on both cards with `nvidia-smi -pl` before each block of runs at that limit; every setting was measured at both.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_XXS (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`, `-00002-of-00002.gguf`),
  revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, installed by `setup.py --model IQ3_XXS` on 2026-10-02 (setup pins the revision and
  checks the files' SHA-256). SHA-256 of the two shards: `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`, `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.
- Pack, MTP draft layer and draft vocabulary (106,299 ids): the ones setup made, unchanged. Expert profile: the repository's `data/expert-profile.bin`.
- Vision: on (`--vision`), the `strata-vision` helper from a Strata 0.1.38 build.
- Context 262,144; `--kv int8`; expert cache auto with every expert resident on both cards (`pre-filled 12288 of 12288`); `--mmap-experts`;
  prefill auto (8,192); no low-RAM mode. `--pipeline-windows 2` keeps 272 MiB of CUDA0 out of the expert cache; residency is unchanged.
- MTP: `--spec 4`, `spec_min_p` 0.5; `--suffix-draft` at its default (on); reasoning off per request; temperature 0; calibration and experimental speed projection not enabled.
  The draft layer stays on the last card with the flag: `STRATA_MTP_STAGE0` (0.1.44's drafter on the first card) is not set, and its default is on only in HIP builds for gfx12.
- Environment: `STRATA_SM70_TABLE=1` (the Volta decode kernels in docs/NVIDIA_V100.md) and `STRATA_PLE_PROBE=0` (skips the start-up PLE read check
  and keeps direct reads; in an earlier run on this machine that check once measured a slow read and moved the run to mmap reads, as in #2040).

```text
strata --serve --pack <pack> --native <shard 1> --ple-gguf <shard 2> --expert-profile <tree>/data/expert-profile.bin --expert-cache auto
  --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp> --max-context 262144 --kv int8 --vision --vram-reserve-mib 700 --mmap-experts
  --layer-split 24 [--pipeline-windows 2]
```

VRAM after load: on 29,104 / 30,750 MiB, off 28,914 / 30,674 MiB (CUDA0 / CUDA1).

## Method

- A fresh engine start per run; on, off, off, on (ABBA), separately for decode and for prompts, at 170 W and at 250 W (16 starts).
- Decode (`decode_probe.py`): three short questions (English, Japanese, code; 37 / 36 / 50 prompt tokens; each reply runs to its cap of
  400 / 600 / 400 tokens). One warm-up pass, then two passes per start (9 requests). Nothing reused between requests.
- Prompt (`prefill_probe.py`): after each start, fresh prompts of 26.7K, 52K and 104K tokens, then another 26.7K and 52K, in that order (a new
  first line each time, nothing reused; no warm-up, so the first 26.7K is the first request after start). The text is C/C++ source with a
  passphrase in the middle; the answer must contain it.
- Timing: the engine's own lines (`prompt N tokens ... read in X ms (Y tok/s)`, `N generated in X ms (Y tok/s)`). The client's streamed
  timings are in the CSV too and agree within 1%.
- Draft layer placement: a second decode ABBA at 170 W and 250 W, `--pipeline-windows 2` in both arms, `STRATA_MTP_STAGE0=1` against unset
  (8 more starts, the same three questions and passes).
- `prefill_probe.py` also sends one short question with a 512-token answer at the end of each start; it is not used here.

## Results

Decode, tok/s from the engine: median (min-max) of the four measured requests per question (two starts x two passes; warm-up left out).
Prompt tokens 37 / 36 / 50, 0 reused, 400 / 600 / 400 generated. TTFT: not measured for these.

| Configuration | English | Japanese | code |
| --- | ---: | ---: | ---: |
| 170 W off | 100.60 (100.6-100.7) | 111.85 (111.8-111.9) | 122.85 (122.8-123.1) |
| 170 W `--pipeline-windows 2` | 106.10 (106.1-106.1), +5.5% | 120.65 (120.6-120.7), +7.9% | 144.80 (144.7-144.9), +17.9% |
| 250 W off | 100.75 (100.7-100.8) | 112.10 (112.0-112.1) | 123.30 (123.1-123.3) |
| 250 W `--pipeline-windows 2` | 106.10 (106.1-106.2), +5.3% | 120.75 (120.7-120.8), +7.7% | 145.25 (145.2-145.3), +17.8% |

Prompt, tok/s from the engine: median of two starts, one request each (the two values are in `prompt.csv`; they differ by 1.2% at most).
26,693-26,694 / 52,020-52,021 / 103,870 prompt tokens, 0 reused. In brackets: TTFT on the client (prompt sent to first content token), median of the two starts.

| | 26.7K (first after start) | 52K | 104K | 26.7K again | 52K again |
|---|---:|---:|---:|---:|---:|
| 170 W off | 1,667.4 (16.1 s) | 2,067.4 (25.2 s) | 2,140.8 (48.6 s) | 1,797.9 (14.9 s) | 2,062.7 (25.3 s) |
| 170 W `--pipeline-windows 2` | 1,674.8 (16.0 s) | 2,075.6 (25.1 s) | 2,145.6 (48.5 s) | 1,809.3 (14.8 s) | 2,070.3 (25.2 s) |
| 250 W off | 1,788.2 (15.0 s) | 2,241.1 (23.3 s) | 2,334.6 (44.6 s) | 1,942.8 (13.8 s) | 2,240.4 (23.3 s) |
| 250 W `--pipeline-windows 2` | 1,788.2 (15.0 s) | 2,248.9 (23.2 s) | 2,342.5 (44.4 s) | 1,952.8 (13.7 s) | 2,246.6 (23.2 s) |

- Prompt differences +0.0% to +0.6%. The first 26.7K after a start is 7-8% slower than the second 26.7K in every arm (medians).
- Drafts accepted / drafted (MTP and suffix drafts together), as the engine prints them (the 9 requests of one start, warm-up included; `decode.csv` adds up to these): with the flag 2,609 / 3,641 (71.7%,
  the same in all four starts), off 2,712 / 3,448-3,450 (78.6-78.7%). Fewer drafts are accepted with the flag, and decode is faster.
- No failures; no skipped cases.

### Draft layer on the first card (`STRATA_MTP_STAGE0=1`)

Decode with `--pipeline-windows 2` in both arms, tok/s from the engine, median (min-max) of the four measured requests per question.
The unset arm repeats the flag-on rows above in separate starts.

| Configuration | English | Japanese | code |
| --- | ---: | ---: | ---: |
| 170 W, unset (last card) | 106.15 (106.1-106.2) | 120.65 (120.6-120.9) | 144.85 (144.7-144.9) |
| 170 W, `STRATA_MTP_STAGE0=1` | 99.00 (98.9-99.0), -6.7% | 114.50 (114.5-114.6), -5.1% | 140.60 (140.5-140.6), -2.9% |
| 250 W, unset (last card) | 106.25 (106.2-106.3) | 120.95 (120.8-121.0) | 145.35 (145.3-145.5) |
| 250 W, `STRATA_MTP_STAGE0=1` | 99.10 (99.0-99.2), -6.7% | 114.65 (114.5-114.7), -5.2% | 140.70 (140.5-140.8), -3.2% |

- The draft layer did move: VRAM after load 30,396 / 29,456 MiB (CUDA0 / CUDA1) against 29,104 / 30,750 unset, and the log reserves
  `+186 MiB for the draft head` on CUDA0. Every expert stayed resident (`pre-filled 12288 of 12288`) in both arms.
- Replies identical to the runs above (same hashes per question); drafts accepted / drafted 2,609 / 3,641 per start in both arms.
- We did not look into why; on this machine the last card is the faster place for the draft layer (the first card is the one on the PCIe x4 link).

### Attached files

- `engine-timing.txt`: the engine's timing line of every request (120: 72 decode, 40 prompt and the probe's one 512-token answer per prompt start),
  under a header line per start in run order. The CSVs are taken from these lines and the probes' output.

- `decode.csv`, one row per request (72 = 8 starts x 9, warm-up rows included): `power_limit_w`; `pipeline_windows` (2 or 0 = off); `start`
  (a / b, the two starts of a setting); `pass` (`warm-up`, 1, 2); `question` (en / ja / code); `prompt_tokens`; `generated_tokens`;
  `decode_tok_s_client` (tok/s from the stream; empty for warm-up, whose speed the probe does not print); `decode_tok_s_engine` (the engine's
  `generated in ... (tok/s)`); `accepted`, `drafted` (drafts of that request, MTP and suffix, as the engine counts them); `reply_sha256_8` (the first 8 hex digits of the SHA-256 of the reply text).
- `decode-mtp-stage0.csv`, one row per request (72 = 8 starts x 9) of the placement runs: the columns of `decode.csv` with `mtp_stage0`
  (1 or 0 = unset) in place of `pipeline_windows`; `--pipeline-windows 2` in every row. Their timing lines are not in `engine-timing.txt`.
- `prompt.csv`, one row per request (40 = 8 starts x 5): `power_limit_w`; `pipeline_windows`; `start`; `order` (1-5 after the start);
  `prompt_tokens`; `prompt_tok_s_client` (prompt tokens / seconds to the first content token, as the probe prints it); `ttft_s_client`
  (`prompt_tokens / prompt_tok_s_client`); `prompt_read_ms_engine` and `prompt_tok_s_engine`
  (the engine's `read in X ms (Y tok/s)`); `recall` (PASS when the answer contains the passphrase).

## Correctness and limitations

- Replies: identical bit for bit per question in all 144 decode requests (72 above: 4 starts with the flag and 4 without, 9 each; and the 72 placement runs; compared in full,
  `decode.csv` and `decode-mtp-stage0.csv` keep the first 8 hex digits of each SHA-256).
- The passphrase check passed in all 40 prompt requests (8 starts x 5).
- One model and one machine; two starts per setting. `STRATA_MTP_STAGE0=1` was measured on decode only.
- Decode was measured on three short questions only; decode after a long context and agent-like conversations were not measured.
- This does not ask for a default change; whether two sm_70 stages should default to on is for the maintainers once other NVIDIA two-stage numbers are in.

Disclosure: prepared with an AI assistant (Claude); the numbers are from runs on the machine above.
