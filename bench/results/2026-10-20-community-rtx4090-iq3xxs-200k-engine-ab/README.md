# Engine A/B: NVIDIA GeForce RTX 4090 - 0.1.45 vs 0.1.45.3

Measured on 2026-10-20 by [Dmitry-B](https://github.com/Dmitry-B). One question: does the engine change
between 0.1.45 and 0.1.45.3 change what this machine feels? The answer is **no, within +-1% at 16K and
32K tokens**, and the short-prompt column on this machine cannot decide anything. The folders contain the
measurements of the repository's own tool, run twice against each binary in alternating order, plus a
longer-context suite (up to the 204,800-token limit of this configuration) kept as a supplement.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090, 23028 MiB reported, 480 W limit, PCIe bus 00000000:01:00.0; link speed
  and width were not read; clocks were not fixed.
- CPU: AMD Ryzen 9 7950X (32 logical CPUs); RAM 46464 MiB; NVMe ADATA LEGEND 960, 1.9 TB.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, NVIDIA driver 615.71.09, CUDA 13.4 (V13.4.92), Python 3.14.4.
- Both engines were **built from source on this machine by `update.sh`** (CUDA arch `[89]`), never
  downloaded: 0.1.45 (md5 `609cc12a4009f303e3f1d83523f30f30…`, kept as `engine/strata.0145.bak`) and
  0.1.45.3 (md5 `46db377627e32c2f71eb439d0f7d43ad`, `engine/BUILD.json` src hash `25c992dfc5c6cb06`).
  Machine details, model files and the exact arguments: [env.json](env.json).

## Configuration (byte-identical between the arms)

The same server config for both arms, only the engine binary differs: [config.json](config.json)
(0.1.45.3) and [config-0.1.45-previous.json](config-0.1.45-previous.json) (0.1.45); the argument lists
are the same. Flash-Next IQ3_XXS, 204,800-token context, INT8 KV, packed experts (`--pack` +
`--mmap-experts` reading `packs/iq3_xxs/experts.bin`), `--expert-cache auto --prefill auto --spec 4
--mtp`, `--pool-workers 10`, GPU vision.

The auto-selected prompt chunk is the same in both engines, so the runs are comparable token by token:

```text
0.1.45    strata serve: prompt chunk auto: 8192 tokens, a 96-slot ring
0.1.45.3  strata serve: prompt chunk auto: 8192 tokens, a 96-slot ring
0.1.45.3  strata: prefill plan: chunk 8192, ring 96 kept: file set 31.7 GiB fits the page cache 40.1 GiB
```

## Method

1. **The repository's own suite, twice per engine, in alternating order.**
   `python tools/strata_bench.py --url http://127.0.0.1:8080 --api-key <key> --runs 3 --long` against the
   production service; the binary was swapped between arms (stop, replace the file, start, wait for
   `/health`), so each arm ran a freshly started engine. Pair 1 order: 0.1.45 then 0.1.45.3; pair 2
   order: 0.1.45.3 then 0.1.45. The tool's own reports for all four runs are in
   [std-suite/](std-suite/) (`*-tool-report.md`, with every request in `*-results.json`).
2. **A longer-context suite of our own** (synthetic code-prose prompts, 4K / 32K / 128K / 160K / 185K /
   200K and a generation-only case, 3 runs each, a random marker in every prompt so the prefix cache is
   not reused, warm-up run before the measured ones). It is kept because the tool above stops at 32K and
   this configuration's whole point is the 204,800-token context. It is **not** comparable with other
   community reports; it is comparable with our own previous reports, and the 0.1.45 baseline of the
   4K/32K/128K cases was re-measured on the old binary in the same session
   ([runs-0.1.45-previous.json](runs-0.1.45-previous.json)).
3. **A control run of the same 0.1.45.3 binary** in the same session
   ([runs-0.1.45.3-control.json](runs-0.1.45.3-control.json)) - that is the noise band of instrument 2.
4. **An interleaved binary A/B** with instrument 2 (2 pairs, both binaries alternating inside one session)
   in [ab-binary/](ab-binary/), to separate the version from page-cache state.

No request of our own (agent traffic) was sent to this server during the measured runs; the GPU is
dedicated to Strata, the OS is not isolated.

## Results: the repository's suite

Pair 1, 0.1.45 first (`--compare-only`, printed by the tool itself):

```text
                                          old          new    change
chat decode tok/s                      103.20       105.90     +2.6%
chat first token (s)                     0.28         0.28     -0.7%
4K prompt tok/s                       1482.40      1294.70    -12.7%
4K first token (s)                       2.73         3.12    +14.4%
16K prompt tok/s                      3245.50      3241.60     -0.1%
16K first token (s)                       4.97         4.97     +0.1%
32K prompt tok/s                      3303.80      3307.10     +0.1%
32K first token (s)                       9.72         9.72     -0.1%
agent cache hit %                       69.43        69.42     -0.0%
agent re-read tokens/session            73.00        74.00     +1.4%
agent turn first token (s)               1.79         1.80     +0.6%
agent decode tok/s                     126.10       121.00     -4.0%
```

Pair 2, 0.1.45.3 first (the table below is printed with 0.1.45 as the base, so the second arm of the
pair is the one labelled `new`):

```text
                                          old          new    change
chat decode tok/s                       96.80       105.60     +9.1%
chat first token (s)                     0.29         0.29     -1.4%
4K prompt tok/s                       1924.60      1523.80    -20.8%
4K first token (s)                       2.10         2.65    +26.2%
16K prompt tok/s                      3258.70      3233.00     -0.8%
16K first token (s)                       4.95         4.99     +0.6%
32K prompt tok/s                      3352.00      3322.60     -0.9%
32K first token (s)                       9.59         9.67     +0.9%
agent cache hit %                       69.43        69.50     +0.1%
agent re-read tokens/session            73.00         4.00     -94.5%
agent turn first token (s)               1.79         1.81     +0.9%
agent decode tok/s                     124.20       135.00     +8.7%
```

What the two pairs say together:

- **16K and 32K prompt reading: unchanged.** Four numbers, -1.3% … +0.1%; the spread of the pairs is
  larger than any difference between the engines.
- **4K prompt: this machine decides, not the engine.** The 4K column moved -12.7% when 0.1.45 ran first
  and -20.8% when it ran second: in one pair the older binary was faster, in the other the newer one was
  26% faster in absolute terms (1294 … 1924 tok/s over four runs of the same setting). The same drift is
  reported by others at 8-28% between engine starts (#1332); on this machine it is visible across sessions
  too, where our own suite measured 1852 tok/s for 0.1.45 in one published session and 2170 tok/s for
  0.1.45.3 today, while its earlier engines gave 2146 and 2157. No short-prompt claim is made in either
  direction.
- **Decode is not established.** `chat decode` is +2.6% and +9.1% in the two pairs, `agent decode` is
  -4.0% and +8.7% - the pairs disagree in sign. Our generation-only case (a 1024-token answer,
  n=3 per arm) gave x0.997 in the full session and x1.028 in the interleaved A/B, while the control run
  of the same binary moved +-1.5% and the per-case band reached +-6.6%.
- **The conversation cache is unchanged**: 69.4% hit in all four runs of the agent case.

## Supplement: the same engines at 128K-200K (our own suite, not comparable with other reports)

Medians of 3 runs; the baseline of 0.1.45 for the short cases was measured on the old binary in the same
session, the long rows come from the 0.1.45 long-context runs we submitted for this machine in PR #3109
(`runs-long-context.json` of that submission), re-measured on the same machine and config. The `control` columns are the same 0.1.45.3 binary run again in the same session -
they are the noise band, and only differences outside them are called differences.

| case | prompt 0.1.45 | prompt 0.1.45.3 | x | control x | decode 0.1.45 | decode 0.1.45.3 | x | control band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 1852.5 | 2170.4 | 1.172 | 1.168 | 137.7 | 135.6 | 0.985 | +-3.2% |
| 32K | 3301.3 | 3356.6 | 1.017 | 1.009 | 125.9 | 135.6 | 1.077 | +-0.6% |
| 128K | 3280.4 | 3305.4 | 1.008 | 1.009 | 124.4 | 123.3 | 0.991 | +-6.6% |
| 160K | 3286.6 | 3310.3 | 1.007 | 1.007 | 125.9 | 134.6 | 1.069 | +-1.6% |
| 194K | 3298.3 | 3314.8 | 1.005 | 1.003 | 115.3 | 133.5 | 1.158 | +-1.3% |
| 200K | 3267.9 | 3292.6 | 1.008 | 1.007 | 130.6 | 134.7 | 1.031 | +-3.1% |
| gen-only | 287.7 | 263.9 | 0.917 | 1.017 | 154.7 | 154.2 | 0.997 | +-3.6% |

The interleaved binary A/B of the same suite (`ab-binary/`, 2 pairs, one fresh engine per arm) gives
32K x1.010 with the newer binary faster in 2 of 2 pairs, 4K x0.902 (0/2) and generation x1.028 (1/2).

Taken together with the tool's own numbers above, the honest reading is: **at 32K and above the change is
at most about +1% and is not separable from the spread of repeated runs on this machine**. Our earlier
reading of the long-context table (+0.5…+1.7%, "confirmed twice") was based on two instruments of the
same kind; the repository's suite is a third, and it lands on zero. We claim nothing here.

## Correctness

- Recall check (needle in a haystack, 32K and 128K, depths 10/50/90): 6 of 6 on both engines -
  [needles.json](needles.json), [needles-0.1.45-previous.json](needles-0.1.45-previous.json).
- Draft acceptance over the whole suite: 4615/6415 = 71.9% (0.1.45) -> 4567/6241 = 73.2% (0.1.45.3);
  the per-request numbers are in every `runs*.json`.
- Test suite: 1881 tests (1012 `tools/`, 869 `serve/`), 57 not passing at this commit, against 1703/16 at
  0.1.44.1. Every delta was traced to this machine or to the tests themselves, none to the engine or the
  server; the summary of the run is [tests-0.2.0-beta-summary.txt](tests-0.2.0-beta-summary.txt)
  and the reasons are filed separately as an issue.
- Answer determinism: on this machine the same binary and the same config do **not** return identical
  answers when the server learns the expert profile between requests (`--expert-profile-save …
  --expert-profile-save-every 10`): in a control run of the same configuration twice, 3 of 7 fixed
  prompts differed. A bit-level comparison of two builds has to freeze that profile; we report it with the
  opt-in measurements.

## Not measured

Concurrency (`--clients`), cold start with an empty page cache, vision requests, the 0.2.0
application-layer features, and anything on a second machine. The opt-in switches of 0.2.0-beta measured
on this card (the sm_86 switch set, `STRATA_GDN_CHUNKED`, `STRATA_IO_PREFETCH`) and the file tier of
#149 are reported separately, not here.

## Files

| file | content |
| --- | --- |
| [std-suite/](std-suite/) | the repository tool's reports for all four arms (2 pairs x 2 engines) |
| [runs.json](runs.json) | our suite on 0.1.45.3 (6 prompt lengths + generation, 3 runs each) |
| [runs-0.1.45.3-control.json](runs-0.1.45.3-control.json) | the same binary again - the noise band |
| [runs-0.1.45-previous.json](runs-0.1.45-previous.json) | the 0.1.45 baseline of the short cases, same session |
| [ab-binary/](ab-binary/) | interleaved binary A/B (which engine answered is recorded per arm) |
| [needles*.json](needles.json) | recall on both engines |
| [config*.json](config.json), [env.json](env.json) | the two server configs (identical arguments) and the machine |
| [tests-0.2.0-beta-summary.txt](tests-0.2.0-beta-summary.txt) | counts and the failing test names of the full unit-test run |
