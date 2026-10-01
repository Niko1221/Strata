# 2026-10-01 — expert-cache tuning pass (`strata-coder-iq1_m.json`)

Config-only pass on the served `strata-coder-iq1_m.json` (engine log `strata-coder-iq1_m.log`).
No `CMakeLists.txt`, no kernel source, no build. No server or benchmark was started here — the host
runs `tools/hip/bench_prefill.py` afterwards.

Baseline from the prompt and the log: ~750–800 t/s fresh prefill, ~36–40 t/s decode, decode
expert-cache hit rate 43–64% (typical 55–63%), KV block hit rate 98–99%, spec acceptance ~75% at
`--spec 4`. Decode is memory-bound on expert-cache misses (`OPTIMISATION.md:433`, `:481`), so this
pass buys cache *capacity* and cache *freshness*, not kernel instructions.

## Keys changed (JSON `args` array)

| Key | Before | After |
| --- | --- | --- |
| `--vram-reserve-mib` | `1024` | `256` |
| `--adapt-every` | `0` | `200` |
| `--pcie-frac` | `0` | **removed** (engine default) |

**Final value (2026-10-01, end of day): `--vram-reserve-mib` = `1024` — back to the original.**
The benchmark below favored 616 (fastest arm), but live use with browsers open (Brave + Zen +
the chat tab) showed both 256 and 616 leave too little for the display: with 616 the loaded engine
had only 532 MiB free, amdgpu then logged `Failed to pin framebuffer` (screen flicker) and
`Not enough memory for command submission`, and prompts crawled at 2–3 tok/s. At 1024 the engine
loads with 918 MiB free, zero kernel errors, and a 16-token probe answers in ~1.5 s. The reserve
protects what the engine leaves free at cache-sizing time; it cannot fence off GPU buffers the
browsers allocate *afterwards*. The table above records the candidate arm's measured state (256).

Plus one new top-level key `"_comment"` recording the `--pcie-frac` removal and the reproducibility
trade. JSON has no comment syntax, and every reader of this file fetches config keys by name
(`serve/server.py:1883` `json.loads`, `:541-565` `engine_args`/`child_env`, `serve/mcp.py:644-662`),
so an unknown key is ignored rather than parsed as an argument. Validated with
`python3 -m json.tool strata-coder-iq1_m.json`.

### `--vram-reserve-mib 1024 → 256`

The reserve is subtracted before `--expert-cache auto` sizes its slots:
`slots = (free_b − reserve) / max_blob`, where `reserve` is `--vram-reserve-mib` plus the draft-head
and (when not borrowing) prefill allowances (`src/program/generate.cpp:2345-2351`), and after the slots
are written the engine shrinks the cache back until that much is free again
(`src/program/generate.cpp:2468-2477`). The served run logs the effect directly —
`expert cache auto: 10.30 GiB free, 1024 MiB reserved -> 3749 slots`
(`strata-coder-iq1_m.log:11`), then `expert cache 4886 slots, 9.30 GiB of VRAM` after per-pair sizing
(`log:12`). Dropping the reserve hands 768 MiB back to the cache: roughly +7% on the uniform-slot
figure (3749 → ~4020), and the profile's per-pair pass turns the same bytes into proportionally more
slots, since a native pack's blobs are smaller than `max_blob`
(`src/program/generate.cpp:2376-2389`). More slots on a memory-bound decode is the whole point:
every extra resident expert is one fewer miss (`docs/DETAILS.md:146-149`).

**Risk — watch the first startup log.** The engine only enforces the reserve immediately after the
cache is zeroed (`src/program/generate.cpp:2466-2477`); the verify-window buffers and the MTP draft
head are allocated later (`log:33-34`), and the served run already ends with **698 MiB free**
(`log:35`) against a configured 1024 — about 326 MiB below the reserve. With a 256 MiB reserve the
same drift lands close to zero. `docs/DETAILS.md:688` documents the failure ("reading the prompt",
GPU pinned at 100%) and the remedy (raise `--vram-reserve-mib`). If the first run prints
`shrinking the expert cache` or a low `... MiB of VRAM free with everything loaded`, put the reserve
back toward 512–1024 and keep the rest of this change set.

### `--adapt-every 0 → 200`

`0` is static residency: the GPU cache never moves (`src/program/generate.cpp:324-326`). `200`
re-enables the adaptive tier — every 200 rounds the most-routed *missing* experts swap into VRAM in
place of the least-routed resident ones (`src/program/generate.cpp:3415`, `:4766`). With
`--resident-cpu-experts` a swap is a RAM↔VRAM copy of the evicted/evicting pair and touches no file
(`docs/DETAILS.md:105-106`, `docs/AMD_HIP_PERFORMANCE.md:39-41`), so the cost is small next to the
misses it prevents: the cache follows the conversation instead of whatever the first request warmed.
200 rounds is deliberately loose — enough to track a long coding session without churning; tighten
only if the hit rate stays low on fresh turns.

### `--pcie-frac 0` removed

`0` forces every expert-cache miss onto the CPU pool. Removing the flag restores the engine default
`-1.0`, i.e. "the model's default": **0.55 for a native pack**, scaled down by a one-off H2D bandwidth
probe and clamped to 0 only below ~4 GB/s (`src/program/generate.cpp:322`, `:1534-1547`). That share
of the misses is fetched over PCIe *while* the CPU computes the rest — the overlap the flag exists for
(`docs/DETAILS.md:107-109`, `:263-264`).

**Trade accepted: exact-token reproducibility is given up.** `docs/DETAILS.md:107-109` states that
`--pcie-frac 0` is what gives "the mapped mode's exact tokens", and `docs/DETAILS.md:463-464` states
that with the default adaptive expert tier a sampled result is not reproducible run to run (use
`--adapt-every 100000` for static residency if seeds must be reproducible). Both halves of that
bargain change in this pass (`--adapt-every 200` *and* the non-zero `--pcie-frac`), so future runs are
for throughput comparison, not token-for-token A/B.

## Not changed (explicitly out of scope this pass)

`--spec 4`, `--spec-min-p 0.5`, `--prefill 8192`, `--pool-workers 7`, and the hipBLASLt tuning table
path (`env.STRATA_HIPBLASLT_TUNING` → `tools/hip/gfx1100-hipblaslt-100200.txt`) are untouched, as is
`CMakeLists.txt` and every kernel source (OPT-1 still needs its reviewed guard fix first —
`OPTIMISATION.md:133-235`).

## Discrepancies and observations (nothing blocked)

All three step-2 premises held on disk: `--vram-reserve-mib 1024`, `--adapt-every 0` and
`--pcie-frac 0` were all present exactly once, so no step had to be stopped. Worth recording anyway:

1. **The config is gitignored** (`.gitignore:37`, `/strata-*.json`): `strata-coder-iq1_m.json` is not
   tracked, so this change has no diff in `git status`. Back it up outside the repo if it matters.
   `strata-coder-iq1_m.json.orig` and `.tuned` (both untracked) differ: `.orig` has `--prefill auto`
   and none of `--adapt-every`/`--pcie-frac`/`--vram-reserve-mib`; `.tuned` has `--pool-workers 15`.
2. **The served config still says `--prefill 8192`,** while `docs/DETAILS.md:118-127` recommends
   `--prefill auto` (and `docs/AMD_HIP_PERFORMANCE.md:33` measured with `8192`). Left alone per the
   instructions — but it is the obvious next lever if prefill is the target rather than decode.
3. **`--pool-workers 7` vs the measured 15.** `docs/AMD_HIP_PERFORMANCE.md:33` and
   `strata-coder-iq1_m.json.tuned` both use 15; this host has no AVX-512 (`log:1`), which may be why 7
   was chosen. Untouched this pass; `--calibrate` (`docs/DETAILS.md:262-271`) is the documented way to
   settle it.
4. **The log is from engine 0.1.27** (`log:29`) while the tree is 0.1.30 (HEAD `30ec18e`), so the
   sizing-line wording differs slightly from `src/program/generate.cpp` today. Slot arithmetic above is
   therefore an estimate; read the new run's own `expert cache auto:` line for the real figure.
5. **The restored `--pcie-frac` default can still be 0** if the H2D probe measures under ~4 GB/s
   (`src/program/generate.cpp:1540-1547`). Check the startup line
   `strata generate: PCIe probe: ... -> pcie_frac ...` on the next run to confirm the arm actually got
   a non-zero share.
6. ~~Nothing here is throughput evidence yet.~~ Superseded: the control/candidate benchmark was run
   the same day — see [Benchmark](#benchmark-2026-10-01-this-host) below. No cancelled-request timing
   line was used (`docs/AMD_HIP_PERFORMANCE.md:65`).

## Benchmark (2026-10-01, this host)

Three arms, one fresh server at a time (port 8080 and `/dev/kfd` verified clean between arms),
`tools/hip/bench_prefill.py`, 9 requests each (1 warmup + 4× fresh 4210/8830 tokens + 4 follow-ups,
128-token cap, `timeout 900`, exit 0 on all arms):

```sh
python3 tools/hip/bench_prefill.py --model qwen3.8-flash-next-coder-iq1_m \
  --url http://127.0.0.1:8080 --engine-log <log> --label <label> --output <json>
```

| Arm | Config | Launcher |
| --- | --- | --- |
| candidate | tuned keys as shipped (`--vram-reserve-mib 256`) | `run-coder-iq1_m.sh` |
| control | three keys reverted to `1024` / `0` / `--pcie-frac 0` | `run-coder-iq1_m-control.sh` |
| reserve616 | tuned keys + `--vram-reserve-mib 616` (engine's own low-VRAM recommendation) | `run-coder-iq1_m-reserve616.sh` |

All numbers below are the **second, healthy run of each arm** (`bench-candidate.json`,
`bench-control.json`, `bench-reserve616.json`). Every fresh request had `reused == 0`; every
non-warmup request returned `finish_reason=length` with exactly 128 generated tokens; per-request
cache-hit rates were paired from the engine logs by prompt-token count.

### Per-request results

**candidate** (engine log `strata-coder-iq1_m-run2.log`, 2680 MiB VRAM free at load)

| kind | trial | prompt tok | reused | prefill t/s | decode t/s | wall s | cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 0 | 15 | 0 | 41.4 | 13.0 | 0.530 | 33.8% |
| fresh | 1 | 4210 | 0 | 770.2 | 32.2 | 9.452 | 48.7% |
| followup | 1 | 4451 | 4203 | 177.9 | 34.0 | 5.171 | 54.8% |
| fresh | 2 | 8830 | 0 | 768.9 | 34.2 | 15.246 | 49.3% |
| followup | 2 | 9071 | 8823 | 177.0 | 32.7 | 5.338 | 57.2% |
| fresh | 3 | 4210 | 0 | 795.8 | 34.4 | 9.017 | 57.3% |
| followup | 3 | 4451 | 4338 | 111.7 | 34.0 | 4.790 | 58.6% |
| fresh | 4 | 8830 | 0 | 774.9 | 38.1 | 14.770 | 58.3% |
| followup | 4 | 9071 | 8823 | 176.3 | 35.3 | 5.046 | 65.2% |

**control** (engine log `strata-coder-iq1_m-control.log`, 918 MiB free at load)

| kind | trial | prompt tok | reused | prefill t/s | decode t/s | wall s | cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 0 | 15 | 0 | 42.2 | 13.3 | 0.520 | 43.0% |
| fresh | 1 | 4210 | 0 | 704.7 | 36.8 | 9.459 | 58.8% |
| followup | 1 | 4451 | 4203 | 153.2 | 38.4 | 4.965 | 63.0% |
| fresh | 2 | 8830 | 0 | 717.9 | 37.3 | 15.747 | 58.4% |
| followup | 2 | 9071 | 8823 | 155.0 | 37.1 | 5.066 | 59.5% |
| fresh | 3 | 4210 | 0 | 731.3 | 37.1 | 9.220 | 57.5% |
| followup | 3 | 4451 | 4337 | 99.5 | 36.9 | 4.631 | 54.3% |
| fresh | 4 | 8830 | 0 | 722.2 | 38.5 | 15.568 | 57.9% |
| followup | 4 | 9071 | 8957 | 101.0 | 36.1 | 4.689 | 53.8% |

**reserve616** (engine log `strata-coder-iq1_m-reserve616-run2.log`, 510 MiB free at load)

| kind | trial | prompt tok | reused | prefill t/s | decode t/s | wall s | cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 0 | 15 | 8 | 64.8 | 16.5 | 0.244 | 44.3% |
| fresh | 1 | 4210 | 0 | 769.1 | 37.0 | 8.944 | 60.4% |
| followup | 1 | 4451 | 4203 | 196.5 | 38.8 | 4.573 | 61.4% |
| fresh | 2 | 8830 | 0 | 775.5 | 37.9 | 14.781 | 59.9% |
| followup | 2 | 9071 | 8958 | 123.1 | 40.6 | 4.085 | 68.1% |
| fresh | 3 | 4210 | 0 | 798.2 | 40.0 | 8.479 | 67.2% |
| followup | 3 | 4451 | 4337 | 123.3 | 39.8 | 4.156 | 63.4% |
| fresh | 4 | 8830 | 0 | 780.5 | 41.7 | 14.402 | 68.5% |
| followup | 4 | 9071 | 8823 | 198.0 | 39.0 | 4.550 | 73.8% |

### Aggregates (run 2 of each arm)

| metric | candidate | control | reserve616 | res616 vs cand | res616 vs ctrl |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh prefill t/s | 777.5 | 719.0 | 780.8 | +0.4% | +8.6% |
| fresh decode t/s | 34.7 | 37.4 | 39.2 | +12.7% | +4.6% |
| fresh wall s (sum) | 48.49 | 49.99 | 46.61 | −3.9% | −6.8% |
| follow-up prefill t/s | 160.7 | 127.2 | 160.2 | −0.3% | +26.0% |
| follow-up decode t/s | 34.0 | 37.1 | 39.6 | +16.3% | +6.5% |
| follow-up wall s (sum) | 20.34 | 19.35 | 17.36 | −14.7% | −10.3% |
| total wall s (8 requests) | 68.83 | 69.35 | 63.97 | −7.1% | −7.8% |
| fresh hit % | 53.4 | 58.1 | 64.0 | +10.6 pp | +5.9 pp |
| follow-up hit % | 59.0 | 57.6 | 66.7 | +7.7 pp | +9.0 pp |
| all-8 hit % | 56.2 | 57.9 | 65.3 | +9.2 pp | +7.4 pp |

### Verdict

- **candidate ≈ control on a healthy machine** — total wall −0.7% (68.83 s vs 69.35 s), fresh
  prefill +8.1%, fresh decode −7.2%, hit rate −1.7 pp. Within single-run noise: the tuned keys
  neither broke nor clearly beat the control here.
- **reserve616 is the best arm** — fastest total wall (−7.1% vs candidate, −7.8% vs control),
  highest decode (+12.7% vs candidate, +4.6% vs control), and the highest cache hit rate on every
  one of the 8 requests (59.9–73.8% vs candidate 48.7–65.2% and control 53.8–63.0%). Fresh prefill
  matches candidate and beats control by 8.6%.
- **Recommendation (as benchmarked): keep `--adapt-every 200` and the `--pcie-frac` removal; of the
  reserves, 616 was fastest** (it eliminates the 256 arm's VRAM-headroom risk while still beating
  both other arms). **Caveat discovered in live use:** with browsers actively holding GPU buffers,
  616 still left only ~530 MiB and the machine fell into the same submission/pin failures — so the
  config ships at `--vram-reserve-mib 1024` (original value) until a way to cap browser VRAM exists;
  re-run the reserve arms (616 vs 1024) on an idle-GPU desktop before revisiting.
- **Caveats:** one run per arm, different ambient browser VRAM load per arm, and hit rate drifts
  between runs — treat ±5–8% total-wall differences as noise. The reserve616 decode/hit ordering was
  nevertheless consistent request-by-request, not just in the mean.

### First attempts quarantined (contamination record)

The first run of each arm (13:55–14:36) is **excluded and superseded** — the machine was not healthy:

- The first reserve616 instance degraded live during its bench: GPU clocks oscillating 34↔2566 MHz
  while the GPU reported only 7–32% busy during active requests, and an 11.2 s prefill for a 15-token
  prompt on an otherwise idle system, worsening run over run. It recovered only on a fresh process.
- `journalctl` shows amdgpu `*ERROR* Not enough memory for command submission!` and
  `Failed to pin framebuffer with error -12` from 14:26:59 to 14:39:37 — inside that contaminated
  window (plus two stray occurrences at 10:06/10:54, before any server). **Zero such errors after
  14:39:37**, i.e. during all three run-2 benches.
- A user abort SIGINT'd a mid-load server (instance A) without detaching; the launch pattern was
  changed to `setsid nohup ... & disown` afterwards, and every arm since verifies port 8080 and
  `/dev/kfd` are clean before starting.
- The candidate's first run also logged the engine warning `152 MiB of VRAM free ... LOW: requests
  may stall; add --vram-reserve-mib 616` — the same config loaded with 2680 MiB free on run 2,
  showing ambient GPU-buffer residency (browsers) swung ~2.5 GB between attempts.
- An earlier claim in this file that the candidate was **+132.4% slower** than control (fresh prefill
  107.8 vs 416.0 t/s, decode 24.4 vs 15.5, total wall 317.1 vs 136.5 s) is **withdrawn** — those two
  runs sat inside the contaminated window.

Quarantined artifacts kept for audit: `bench-candidate-run1-contaminated.json`,
`bench-control-run1-contaminated.json`, `bench-reserve616-run1-contaminated.json`,
`strata-coder-iq1_m-run1.log`, `strata-coder-iq1_m-control-run1.log`,
`strata-coder-iq1_m-reserve616-run1.log`. Clean artifacts: `bench-{candidate,control,reserve616}.json`,
`strata-coder-iq1_m-run2.log`, `strata-coder-iq1_m-control.log`,
`strata-coder-iq1_m-reserve616-run2.log`. The tuned config was verified byte-identical to its
pre-benchmark backup after all arms — the only edits to `strata-coder-iq1_m.json` were the approved
post-benchmark reserve changes `256 → 616 → 1024` (final state differs from the backup on that one
line: `1024` vs the arm's `256`) plus the unrelated addition of `"fit_max_tokens": true` (clamps
oversized `max_tokens` to the remaining context instead of returning 400 — added after a 50k-token
chat request 400'd on `prompt + max_tokens > 65536`).
