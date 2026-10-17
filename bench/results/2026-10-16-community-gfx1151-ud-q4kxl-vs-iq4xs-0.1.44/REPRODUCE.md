# Reproduce — gfx1151 UD-Q4_K_XL vs UD-IQ4_XS, 2026-10-16 / 2026-10-11

Both arms were measured on the machine described in [README.md](README.md). The harnesses are the
written for this machine and **not published with this report** (see the table below); nothing
else was custom-written for a submission. Raw outputs were **not** copied into this folder — they live on
the submitting machine and are pinned by SHA-256 below, so a verifier can ask for a file and check
it. Re-running the full chain takes about 2 h per arm plus a model load.

## Harnesses used

| Harness | Where it lives | What it produces |
|---|---|---|
| `speed.py` | **submitter-local script, not in any repository** (submitter's machine, path withheld) | short-prompt code/prose rows, and `pp4k`/`pp16k-1`/`pp41k-1`/`pp64k-1`/`pp128k-1`/`pp245k-1` long rows; reports engine `pp`/`tg`, draft accept, wall time, `MemAvailable` |
| `ctxbench.py` | **submitter-local script, not in any repository** it is **not part of Strata**, nor of any tracked tree on the submitter's machine | standard and `hard` correctness batteries, seed-controlled, thinking off |
| `iotrack.py` | submitter-local script | disk I/O samples every 5 s during arm B's chain (not used as a result here) |

> **Reproducibility limit, stated plainly:** neither harness is published with this report, so the
> exact numbers here cannot be re-derived from the repository alone. The harnesses are small local
> scripts written for this machine; ask the submitter for them and they will be supplied, or
> substitute any OpenAI-compatible load driver that sends the same prompt sizes and reads `pp`/`tg`
> from the usage statistics. The correctness batteries additionally depend on `ctxbench.py`'s task
> set and graders, so that part needs the script to reproduce at all.

`speed.py` takes sizes and a floor from the environment; both arms used the same values:
`BENCH_SIZES=41000,64000,128000,245000`, `BENCH_MIN_AVAIL=4`.

## Arm A — UD-IQ4_XS, Strata v0.1.43, 2026-10-11

Served directly by the engine, with the serving proxy stopped and no other `strata` process
running, so nothing else was resident.

```bash
# abridged
./start-strata.sh --port <local-port> > strata-candidate-server.out 2>&1 &
BENCH_URL=http://127.0.0.1:<local-port> BENCH_MODEL=<any> BENCH_SIZES=41000,64000,128000,245000 BENCH_MIN_AVAIL=4 \
  ./speed.py strata-v43-fast          > speed-strata-v43-fast.out
./ctxbench.py --label strata-0.1.43-fast-262k-thinkoff          > results/strata-v43-fast-std.out
./ctxbench.py --battery hard --levels 16000,48000,96000,150000,200000,245000 \
                --label strata-0.1.43-fast-hard                  > results/strata-v43-fast-hard.out
```

Engine arguments (from the submitter's engine config):

```
--pack unsloth-ud-iq4_xs
--native …/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
--expert-profile <expert-profile.bin>
--expert-cache auto --prefill auto:16384
--spec 4 --spec-min-p 0.5
--mtp <mtp-dir> --max-context 262144
--kv int8 --kv-resident 32768 --resident-budget-gib 55
```

Environment: `STRATA_HIPBLASLT_TUNING=<bundled gfx1151/100401 tuning table>`,
`STRATA_PF_SWITCH_MIN_T=4096`, `STRATA_PREFILL_STREAM_MIN=128`,
`lib_dirs: ["/opt/rocm/core-7.14/lib"]` (the one ROCm install on this host).

## Arm B — UD-Q4_K_XL, Strata v0.1.44.1, 2026-10-16

Served through an OpenAI-compatible proxy, **with live interactive traffic** (the proxy could not
be detached during this window, so unrelated requests may interleave — recorded as a caveat rather
than hidden).

```bash
 # abridged
BENCH_URL=http://127.0.0.1:<proxy-port> BENCH_MODEL=<served-model> BENCH_SIZES=41000,64000,128000,245000 BENCH_MIN_AVAIL=4 \
  ./speed.py strata-q4kxl-final       > speed-strata-q4kxl-final.out
./ctxbench.py --url http://127.0.0.1:<proxy-port> --model <served-model> --label strata-0.1.44.1-q4kxl-final-262k-thinkoff \
                              > results/strata-q4kxl-final-std.out
./ctxbench.py --url http://127.0.0.1:<proxy-port> --model <served-model> --battery hard \
  --levels 16000,48000,96000,150000,200000,245000 \
  --label strata-0.1.44.1-q4kxl-final-hard > results/strata-q4kxl-final-hard.out
```

Engine arguments (submitter's engine config): as arm A, except
`--pack …/unsloth-ud-q4_k_xl`, `--native …/UD-Q4_K_XL/…-00001-of-00004.gguf`,
`--spec 6 --spec-min-p 0.6`, `--mtp-window 8192`, `--resident-budget-gib 71`.

## Counting rule for correctness

`README.md`'s accuracy figures count harness rows beginning `ok ` versus rows beginning `BAD` or
`FAIL`, per battery. No task was excluded or reweighted. Standard scores 182 rows in both arms;
hard scores 185 rows in both arms.

## Raw files and SHA-256 (on the submitting machine, **not** in this PR)

These hashes pin files a reviewer **cannot** download from this pull request — only the README and
this file are in the commit. They are recorded so that a file sent on request can be verified as
the one the tables were built from; treat them as a checksum against a private artefact, not as
published evidence.

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

Paths withheld (submitter's machine). Available on request; verify against the hashes above.

Model files were confirmed by byte count and by the HF revision path
(`models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/…`), the same sizes the two earlier gfx1151
reports record: UD-IQ4_XS 93,682,584,224 B, UD-Q4_K_XL 111,334,654,784 B.

## Not included

Engine binaries, GGUF files, per-token logs, and `iotrack.py` output (about 2 GB across the run).
The `probe-pw-*` and `v-*` sweeps in the same directory are a power-cap and parameter exploration
on arm B's engine; they are **not** part of this comparison and are not cited in `README.md`.
