# Measurements of 2026-10-19

| File | What |
|---|---|
| `vision-20261019-2053.json` | Picture reader on the CPU with engine 0.1.44.1 (`visiontest.py`): answers, times, VRAM/RAM, engine log lines |
| `r3177-20261019-2109.jsonl` | Strata issue #3177 reproduction attempt (`r3177.py`): 55,019-token agent request, 6 runs |
| `folgetests-20261019-2119.jsonl` | `folgetests.py`: test `cpu` (LP-E cores, `--cpuset-cpus 0-13` A/B) and test `frisch` (8 fresh 0.1.44.1 engines) |
| `frisch044-runde2-*.jsonl`, `frisch020-*.jsonl` | fresh engines, 4K prompt: 0.1.44.1 (16 more) and 0.2.0-beta (8 + 16) |
| `strata-0.2.0-beta/` | `t020.py`: prebuilt `hip-rx7900` engine 0.1.45.3: `strata_bench.py --long` (`strata-bench-*`), our `bench.py` (`020-a020.json`, `020-auto.json` = default worker count), needles, engine log excerpts (`log-*.txt`), `doctor.txt`, `engine-version.txt`. `needles-250k.*` is the run cut off by our own model manager (idle TTL while the bench talked to the backend directly); `rerun-*` is the repeat |

Fields: `prompt_tps` / `decode_tps` are the engine's `timings` (tok/s); `draft_n` / `draft_ok` MTP drafts offered/accepted; `sekunden` wall time; `variante` configuration; `lauf` / `paar` / `messung` / `engine` run counters; `neu_token` generated tokens. Model answers of the quality and refusal tasks are left out.
