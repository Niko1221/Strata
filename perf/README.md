# perf/ - the raw runs behind docs/PERF.md

| path | contents |
|---|---|
| `raw/perf-*.jsonl` | prefill + decode curve per arm (`win_perf.sh` → `ctx_curve.py`), one JSON object per rep |
| `raw/tps2-*.txt` | 16-prompt decode harness per arm (`win_tps2.sh` → `win_tps2.py`), last line is the median |
| `raw/needles-*.json` | `tools/needle_bench.py` results (96K / 128K, depth 50) |
| `tps2-medians.tsv` | every `tps2-*` run flattened: label, prefix chars, median tok/s, ms per forward, acceptance, config, env |
| `kl-tables.txt` | the quality tables: `tools/quant_quality.py compare` output, reference fp16 K/V |
| `harness/` | the scripts that produced the above, as they ran on the box |

## Labels and configs

The configs are copies of the production `strata-sc117-iq3_s.json` differing only in the three flags below. They are
not in this repo (they name this box's model paths); they were kept in `/tmp` on the box.

| config | `--kv` | `--kv-resident` | `--prefill` |
|---|---|---|---|
| `cfg-k8v4.json` | k8v4 | 32768 | `auto:8192` |
| `cfg-oct3.json` | oct3 | 32768 | `auto:8192` |
| `cfg-oct3-r64k.json` | oct3 | 65536 | `auto:8192` |
| `cfg-pf32k.json` | oct3 | 65536 | `auto:32768` |
| `cfg-forkA.json` | oct3 | 65536 | `auto:32768` (plus the environment in `docs/FORK-RDNA3-MAPPING.md` §1) |
| `cfg-fork-accel.json` | oct3 | 65536 | `auto:32768`, and `exe` = the fork's `build-hip-oct3/strata` (the `facc-*` arms) |
| production | oct3 | 65536 | `auto:32768` |

Label prefixes in `raw/`: `p`/`z`/`f` = the plain oct3/k8v4 pair, `g64`/`h96`/`p64`/`y24` = context-length arms,
`s24` = window-tail arms, `v` = window V-side arms, `pf32`/`pf8` = prefill-chunk arms, `fork*` = the upstream-switch
arm, `facc-*` = the `STRATA_SPEC_CUM` arms (base / cum25 / cum40 / cum40b / cum60).

## Harness

`harness/launch.sh` starts one engine off `$AB_CFG` with `/tmp/ab_env` as its environment; the three `win_*.sh`
scripts stop the user unit `strata-sc117`, call `launch.sh`, measure, and kill it. They are the box's copies with its
absolute paths (`/home/des/test/strata`, port 8089, the production config for the API key). **None of them restarts
the production unit at the end** - do it by hand and wait for `/health` 200 (`docs/DEPLOY.md` §5).

Prerequisites: a checkout with the Python server (`.venv`), `ab/ctx_curve.py` in it, and `tools/needle_bench.py` for
`win_needle.sh`.

## Not committed

The quality runs themselves are 32 MB probability dumps per arm (`win-6k-*.tsv`, `win-*.tsv`); only the comparison
tables are here. To regenerate them on a box with the model:

```bash
.venv/bin/python tools/quant_quality.py run --kv-format fp16    # the reference
.venv/bin/python tools/quant_quality.py run --kv-format oct3    # and with STRATA_KV_WINDOW=0 / _TAIL=N via --env
.venv/bin/python tools/quant_quality.py compare <ref.tsv> <run.tsv> ...
```
