# Reproduce — gfx1151 UD-Q4_K_XL vs UD-IQ4_XS, 2026-10-16 / 2026-10-11

Both arms were measured on the machine described in [README.md](README.md). The harnesses are the
ones already in this repository and in the sibling `llama-swap` bench directory; nothing was
custom-written for a submission. Raw outputs were **not** copied into this folder — they live on
the submitting machine and are pinned by SHA-256 below, so a verifier can ask for a file and check
it. Re-running the full chain takes about 2 h per arm plus a model load.

## Harnesses used

| Harness | Where | What it produces |
|---|---|---|
| `speed.py` | `llama-swap` bench dir (`/ai/llama-swap/bench/ds4/speed.py`) | short-prompt code/prose rows, and `pp4k`/`pp16k-1`/`pp41k-1`/`pp64k-1`/`pp128k-1`/`pp245k-1` long rows; reports engine `pp`/`tg`, draft accept, wall time, `MemAvailable` |
| `ctxbench.py` | `bench/ctxbench/ctxbench.py` (this repo) | standard and `hard` correctness batteries, seed-controlled, thinking off |
| `iotrack.py` | `llama-swap` bench dir | disk I/O samples every 5 s during arm B's chain (not used as a result here) |

`speed.py` takes sizes and a floor from the environment; both arms used the same values:
`BENCH_SIZES=41000,64000,128000,245000`, `BENCH_MIN_AVAIL=4`.

## Arm A — UD-IQ4_XS, Strata v0.1.43, 2026-10-11

Served directly, with `llama-swap` stopped (the chain script aborts if `llama-swap` is active or
a `strata` process exists), so nothing else was resident.

```bash
# /ai/llama-swap/bench/ds4/strata-chain.sh, abridged
/ai/strata/start-strata.sh --port 8091 > strata-candidate-server.out 2>&1 &
BENCH_URL=http://127.0.0.1:8091 BENCH_MODEL=x BENCH_SIZES=41000,64000,128000,245000 BENCH_MIN_AVAIL=4 \
  ./speed.py strata-v43-fast          > speed-strata-v43-fast.out
./ctxbench.py --label strata-0.1.43-fast-262k-thinkoff          > results/strata-v43-fast-std.out
./ctxbench.py --battery hard --levels 16000,48000,96000,150000,200000,245000 \
                --label strata-0.1.43-fast-hard                  > results/strata-v43-fast-hard.out
```

Engine arguments (from `/ai/strata/strata-x2.json`):

```
--pack /ai/Strata-data/packs/unsloth-ud-iq4_xs
--native …/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
--expert-profile /ai/strata/data/expert-profile.bin
--expert-cache auto --prefill auto:16384
--spec 4 --spec-min-p 0.5
--mtp /ai/Strata-data/mtp/rt --max-context 262144
--kv int8 --kv-resident 32768 --resident-budget-gib 55
```

Environment: `STRATA_HIPBLASLT_TUNING=/ai/strata/tools/hip/gfx1151-hipblaslt-100401.txt`,
`STRATA_PF_SWITCH_MIN_T=4096`, `STRATA_PREFILL_STREAM_MIN=128`,
`lib_dirs: ["/opt/rocm/core-7.14/lib"]`.

## Arm B — UD-Q4_K_XL, Strata v0.1.44.1, 2026-10-16

Served through `llama-swap` on `:8080` as model `deep`, **with live traffic** (llama-swap could
not be stopped during this window, so Home Assistant requests may interleave — recorded as a
caveat rather than hidden).

```bash
# /ai/llama-swap/bench/ds4/strata-final-chain.sh, abridged
BENCH_URL=http://127.0.0.1:8080 BENCH_MODEL=deep BENCH_SIZES=41000,64000,128000,245000 BENCH_MIN_AVAIL=4 \
  ./speed.py strata-q4kxl-final       > speed-strata-q4kxl-final.out
./ctxbench.py --url http://127.0.0.1:8080 --model deep --label strata-0.1.44.1-q4kxl-final-262k-thinkoff \
                              > results/strata-q4kxl-final-std.out
./ctxbench.py --url http://127.0.0.1:8080 --model deep --battery hard \
  --levels 16000,48000,96000,150000,200000,245000 \
  --label strata-0.1.44.1-q4kxl-final-hard > results/strata-q4kxl-final-hard.out
```

Engine arguments (from `/ai/strata-0.1.44.1/strata-x2-q4kxl.json`): as arm A, except
`--pack …/unsloth-ud-q4_k_xl`, `--native …/UD-Q4_K_XL/…-00001-of-00004.gguf`,
`--spec 6 --spec-min-p 0.6`, `--mtp-window 8192`, `--resident-budget-gib 71`.

## Counting rule for correctness

`README.md`'s accuracy figures count harness rows beginning `ok ` versus rows beginning `BAD` or
`FAIL`, per battery. No task was excluded or reweighted. Standard scores 182 rows in both arms;
hard scores 185 rows in both arms.

## Raw files and SHA-256 (on the submitting machine, not in this folder)

```
58868eefe423b2f48788f491487133051925a4e9bce8288c1ac85f2704efff90  speed-strata-v43-fast.out
2a706daf02ad97504da1f083cd3de831df79ca19a72124f8260082eb3401edee  speed-strata-v43-fast.jsonl
0306f8006b1bc5a5b0fbf803e4fef22e989ea0255c31580a9ed2bf6bd154a6a6  speed-strata-q4kxl-final.out
c5cebdcbaddf7e1758b60378317e279f47fed55e1fc9ecd5584808c88297e036  speed-strata-q4kxl-final.jsonl
4467cf2ea90b2f2a47fbe3348fafd67401a03b9250407b8d1f1653073def6c67  strata-v43-fast-std.out
3049407284f3aaa1515d5788fc8481ecf1f403f6c7ce2ff98f8923000cb33ecd  strata-v43-fast-hard.out
a7b90cc572e6c760d633846b808af918734f7ad418ddc7d4b87b349c4e9c9f69  strata-q4kxl-final-std.out
fc1a198a74af2793773148c9570619f89f576dcafdd4d61c0a71e804c88a465c  strata-q4kxl-final-hard.out
```

Paths: the `speed-*` files under the `llama-swap` bench dir (`bench/ds4/`), the `strata-*-std.out`
and `strata-*-hard.out` under `bench/ctxbench/results/`.

Model files were confirmed by byte count and by the HF revision path
(`models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/…`), the same sizes the two earlier gfx1151
reports record: UD-IQ4_XS 93,682,584,224 B, UD-Q4_K_XL 111,334,654,784 B.

## Not included

Engine binaries, GGUF files, per-token logs, and `iotrack.py` output (about 2 GB across the run).
The `probe-pw-*` and `v-*` sweeps in the same directory are a power-cap and parameter exploration
on arm B's engine; they are **not** part of this comparison and are not cited in `README.md`.
