# Community benchmark on RTX 5070 Ti 16 GB (Linux): five opt-in settings, A/B

Measured on 2026-10-15 (04:38-05:55 CDT) by [qiuzan2001](https://github.com/qiuzan2001). Strata 0.1.44.1 source
build, Swift 1.5 IQ3_S, one GPU. Five A/B comparisons, each of 4 alternating rounds with a fresh server per arm and
round: `--adapt-async 1`, `STRATA_ROUTE_TAIL_SKIP=0` against the default 7, `STRATA_MTP_ADAPTIVE=1` from PR #2529,
`--spec 5` against the default 4, and `--mtp-window 8192` against the default 32768. Main limitation: one machine,
one model size, short generations (256 tokens at most); the first A/B ran at a different context from the other four.
The same machine's speed report is in PR #2580 (folder `2026-10-14-community-rtx-5070-ti-swift-iq3_s`); this report
stands on its own.

## Hardware and software

- **GPU:** NVIDIA GeForce RTX 5070 Ti, 16,303 MiB, power limit 300 W, clocks not locked, PCIe gen 5 card in a
  gen 4 x16 slot (`lspci`: 16 GT/s x16). **CPU:** AMD Ryzen 7 5800X3D, 8 cores / 16 threads, AVX2, no AVX-512
  (7 expert-pool workers plus the host thread). **RAM:** 80 GiB DDR4-3200 (2x 32 GiB + 2x 8 GiB, 76.5 GiB usable).
  **Storage:** SK hynix Gold P31 1 TB NVMe, ext4.
- **OS:** Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; driver 595.99.02; CUDA toolkit 13.1.
- **Strata:** source build at `296b59d2` (0.1.44.1), sm_120, Release, native CPU. For the PR #2529 A/B both arms ran
  that PR's head `931e1dac` (one commit on `296b59d2`), built here the same way.
- **Background:** a GNOME desktop session and several mostly idle terminal programs; no other inference or GPU work.
  In 10-second CPU samples other processes (the desktop's editor, terminal programs, the kernel's memory compaction)
  reached 1 to 2.75 cores at times, in every A/B. The arms alternate, so both saw it.

## Model and configuration

- Swift 1.5 IQ3_S from [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  (the repository's files of 2026-10-14, head `99bb8f7f`): `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
  (SHA-256 `46cb3996...46edd3b`) and `-00002-of-00002.gguf` (`70ce6fb4...b66b608`), both checked locally. Setup's pack,
  the shipped `data/expert-profile.bin` (no profile save), the MTP draft layer `mtp/rt`.
- Base arguments (setup's, kept in each A/B's `cfg_*.json` with paths replaced): `--expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --mtp ... --kv int8 --kv-resident 32768`, greedy, thinking off, no calibration, no
  experimental speed projection, vision off.
- **Context:** the `--adapt-async` A/B ran at **1,048,576 tokens** (`--rope-scaling yarn --rope-scale 4`, 3,124-3,126
  experts in VRAM). The machine's config was then changed to the trained **262,144 tokens** (no YaRN), and the other
  four A/Bs ran there (3,837-3,840 experts in VRAM; 3,855-3,856 in the `--mtp-window 8192` arm).

```text
serve/server.py --engine strata --config cfg_<arm>.json --port 8090      (one server at a time, started by ab_more.py)
engine: --pack <data>/packs/swift-iq3_s --native <data>/models/swift-IQ3_S/...-00001-of-00002.gguf
        --ple-gguf <data>/models/swift-IQ3_S/...-00001-of-00002.gguf --expert-profile <strata>/data/expert-profile.bin
        --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt
        --max-context 262144 --kv int8 --kv-resident 32768
        (the --adapt-async A/B: --max-context 1048576 --rope-scaling yarn --rope-scale 4)
plus the arm's change: --adapt-async 1 | STRATA_ROUTE_TAIL_SKIP=0 | STRATA_MTP_ADAPTIVE=1 (PR #2529's binary in both
arms) | --spec 5 instead of 4 | --mtp-window 8192
```

The environment variables are set by `ab_more.py` in the server's environment; they are not in `cfg_<arm>.json`.

## Method

[ab_more.py](ab_more.py) (standard library; the pattern of `tools/ab_engine.py` with more requests):

- An arm is the base config plus one change. Each round starts a fresh server for each arm, in the order A, B in
  rounds 0 and 2 and B, A in rounds 1 and 3, and discards one short warm-up answer per start. Model loading is not
  timed. The machine's own server was stopped during each A/B and started again after it.
- Each start then sends the same requests in order: eight different short prompts (code, prose, lists; 27-37
  tokens) and one long word-list prompt, 256 new tokens at most, greedy, thinking off. The long prompt is 6,345
  tokens, except in the `--mtp-window` A/B, where it is 63,147 tokens: the draft layer attends to the last
  `--mtp-window` cells, so the two windows differ only once the context is longer than 8,192 cells
  ([STRIX_HALO.md](../../../docs/STRIX_HALO.md) section 5 measured it at 32K and 64K).
- **Speeds are the engine's own timings.** "Pooled short decode" for one arm and round is the sum of `predicted_n` over
  the sum of `predicted_ms` of its eight short answers; the long prompt's decode and prompt tok/s are its own
  `timings`. A change is the B arm over the A arm of the same round.
- Each answer's SHA-256 (first 16 hex digits) is kept, to see whether the arms wrote the same text. The draft,
  generated-token and identical-answer counts below were tallied from `rows.jsonl`; the script prints the decode and
  prompt figures. It writes its engine logs as `engine_<arm>_r<round>.log`; they are stored here as `.txt`.
- **Not measured:** time to first token and total latency (non-streamed requests), RAM and VRAM per request.
  **Failures:** none; every request of every A/B returned an answer.

```text
python ab_more.py CONFIG base: "async:xargs=--adapt-async;1" --rounds 4 --out adapt-async
python ab_more.py CONFIG skip7: "skip0:STRATA_ROUTE_TAIL_SKIP=0" --rounds 4 --out route-tail-skip
python ab_more.py CONFIG "fixed:exe=PR2529/strata" "adaptive:exe=PR2529/strata,STRATA_MTP_ADAPTIVE=1" --rounds 4 --out mtp-adaptive-pr2529
python ab_more.py CONFIG spec4: "spec5:drop=--spec,xargs=--spec;5" --rounds 4 --out spec
python ab_more.py CONFIG win32k: "win8k:xargs=--mtp-window;8192" --rounds 4 --long-words 60000 --out mtp-window
```

## Results

Pooled short decode in tok/s per round (A / B), medians and ranges over the 4 rounds, and the paired change.

| A/B (A against B) | Context | Round 0 | Round 1 | Round 2 | Round 3 | A median (range) | B median (range) | B vs A per round | B faster |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| default against `--adapt-async 1` | 1,048,576 | 72.30 / 73.55 | 73.36 / 72.95 | 73.68 / 72.97 | 73.68 / 75.37 | 73.52 (72.30-73.68) | 73.26 (72.95-75.37) | +1.7, -0.6, -1.0, +2.3% | 2 of 4 |
| tail skip 7 (default) against `STRATA_ROUTE_TAIL_SKIP=0` | 262,144 | 79.05 / 60.34 | 79.87 / 60.15 | 76.52 / 59.89 | 78.41 / 60.08 | 78.73 (76.52-79.87) | 60.11 (59.89-60.34) | -23.7, -24.7, -21.7, -23.4% | 0 of 4 |
| PR #2529 unset against `STRATA_MTP_ADAPTIVE=1` | 262,144 | 80.39 / 77.59 | 78.86 / 78.08 | 79.63 / 78.68 | 80.38 / 78.37 | 80.00 (78.86-80.39) | 78.22 (77.59-78.68) | -3.5, -1.0, -1.2, -2.5% | 0 of 4 |
| `--spec 4` (default) against `--spec 5` | 262,144 | 80.13 / 76.40 | 79.99 / 75.80 | 79.59 / 77.38 | 78.70 / 76.14 | 79.79 (78.70-80.13) | 76.27 (75.80-77.38) | -4.7, -5.2, -2.8, -3.3% | 0 of 4 |
| `--mtp-window 32768` (default) against `8192` | 262,144 | 79.81 / 79.85 | 80.06 / 79.09 | 80.42 / 80.12 | 79.45 / 79.77 | 79.94 (79.45-80.42) | 79.81 (79.09-80.12) | +0.1, -1.2, -0.4, +0.4% | 2 of 4 |

The long prompt in each round (one request per arm and round; decode tok/s, then prompt tok/s, medians of 4):

| A/B | Prompt tokens | A decode | B decode | B decode per round vs A | A prompt | B prompt |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `--adapt-async 1` | 6,345 | 62.0 (60.4-65.2) | 63.1 (58.0-65.8) | -5.2, -4.9, +2.2, +8.9% | 2,584 | 2,599 |
| `STRATA_ROUTE_TAIL_SKIP=0` | 6,345 | 65.95 (62.0-71.2) | 48.30 (44.0-54.5) | -17.0, -24.6, -24.7, -38.2% | 2,709 | 2,683 |
| `STRATA_MTP_ADAPTIVE=1` | 6,345 | 65.0 (60.9-67.1) | 64.3 (63.0-65.2) | -1.5, +0.9, +3.4, -4.3% | 2,702 | 2,716 |
| `--spec 5` | 6,345 | 64.35 (60.7-69.0) | 65.25 (64.4-66.0) | -1.8, +4.8, -4.3, +6.1% | 2,710 | 2,704 |
| `--mtp-window 8192` | 63,147 | 66.45 (61.6-69.1) | 68.20 (65.6-69.1) | +1.9, +4.5, +10.9, -5.1% | 3,071 | 3,075 |

Drafts over the 32 short answers of each arm (accepted of offered) and identical answers:

| A/B | A drafts | B drafts | Generated tokens A / B | Answer pairs byte-identical |
| --- | --- | --- | --- | ---: |
| `--adapt-async 1` | 4,825 of 6,986 | 4,764 of 7,063 | 7,688 / 7,640 | 4 of 36 |
| `STRATA_ROUTE_TAIL_SKIP=0` | 4,760 of 6,870 | 4,873 of 7,118 | 7,551 / 7,775 | 2 of 36 |
| `STRATA_MTP_ADAPTIVE=1` | 4,771 of 6,913 | 4,036 of 5,366 | 7,620 / 7,547 | 5 of 36 |
| `--spec 5` | 4,786 of 6,884 | 5,079 of 7,906 | 7,609 / 7,720 | 8 of 36 |
| `--mtp-window 8192` | 4,775 of 6,901 | 4,767 of 6,913 | 7,596 / 7,580 | 13 of 36 |

What the numbers say, on this machine and model only:

- **The route tail skip (on by default) matters most here:** turning it off cost 21.7-24.7% of short decode in every
  round and 17-38% on the long prompt's decode, with prompt speed within 1% (2,709 against 2,683 tok/s). Put the other
  way, the default is +31% over off (78.73 against 60.11 tok/s), more than the +10-20% the engine's start-up line
  names for 12-16 GB cards.
- **`--spec 5` was slower than the default 4 in every round** (-2.8 to -5.2%), although it landed more drafts per
  answer: the longer verify window costs more than the extra drafts gain here.
- **PR #2529's adaptive MTP window was slower in every round** (-1.0 to -3.5%); it offered fewer drafts (5,366 against
  6,913) and landed a larger share, but fewer in total. Posted on the PR.
- **`--adapt-async 1` and `--mtp-window 8192` changed nothing beyond the noise** of these runs (the asynchronous
  tier did run: the round-0 log reads `asynchronous adaptive tier: 195 rounds, 18624 experts swapped in, 38.2 ms per
  round`; that A/B ran at 1,048,576 tokens, see above) (paired changes -1.2 to
  +2.3% on short decode). The `--mtp-window` long-prompt pairs lean towards 8192 (3 of 4, median +3.2%), but each is
  one answer of 135-171 tokens, so that is not a result.

Per-request data, the arms' configs and every engine log are in the subfolders ([adapt-async](adapt-async/),
[route-tail-skip](route-tail-skip/), [mtp-adaptive-pr2529](mtp-adaptive-pr2529/), [spec](spec/),
[mtp-window](mtp-window/)): `rows.jsonl` (one line per request: arm, round, engine timings, drafts, answer hash and
the answer's first 80 characters), `summary.json`, `cfg_<arm>.json`, `engine_<arm>_r<round>.txt` and a GPU telemetry
summary. GPU power during each A/B: median 125-156 W, at most 251-264 W of the 300 W limit (1-second samples
including loading and idle gaps); temperature at most 57-64 C. The software power-cap flag (`0x4`) appeared in
1-27 of 256-399 busy samples per A/B (27 in the `--mtp-window` one, the A/B with the longest prompts).

## Correctness and limitations

- **Answers differ between arms** in most pairs (the byte-identical counts above), so every decode figure includes
  different text and different draft acceptance; the speeds are paired by round, not by text. No quality check was
  run; the tail skip is documented to change answers slightly.
- **Short generations only:** 256 new tokens at most (the long prompts' answers stopped at 116-186 tokens). Long
  generations, long agent sessions, `--batch` and concurrency were not tested.
- **One model size and one machine.** The first A/B ran at 1,048,576 tokens and the others at 262,144, so its numbers
  are not comparable with theirs. Expert-cache hit rates are left out: with the tail skip on they do not count the
  skipped experts, so they do not compare across settings.
- Not tested: other ranks for `STRATA_ROUTE_TAIL_SKIP` (8, 9), `--spec 3`, other `--spec-min-p` values, `--mtp-window`
  past 63K tokens, PR #2529 with `--spec-follow` pinned text (the author's method), `STRATA_DECODE_TIMING=1`.
- **Units:** VRAM and RAM in MiB and GiB as the engine and `nvidia-smi` print them; installed RAM in GiB.
