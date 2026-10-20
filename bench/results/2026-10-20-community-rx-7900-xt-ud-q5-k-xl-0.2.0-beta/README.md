# Community benchmark: v0.1.44.1 -> v0.2.0-beta on an RX 7900 XT 20 GB (gfx1100), Linux, UD-Q5_K_XL

Measured on 2026-10-20 by loeeeee. The same machine, model and run configuration, measured twice the same day with
`tools/strata_bench.py --runs 3 --long`: once on v0.1.44.1 (the build from
[the 2026-10-15 report](../2026-10-15-community-rx-7900-xt-ud-q5-k-xl/README.md)), then after updating the source
build to the v0.2.0-beta tag (engine 0.1.45.3). The before/after comparison below is speed only. A separate
[hand-run pilot of the draft benchmark protocol](protocol-pilot/README.md) (thinking profile, 12 sanity checks, optional
concurrency, needle recall at 32K and 124K, memory/clock telemetry) was run on v0.2.0-beta only: 12/12 sanity and 6/6 needles
in each of two runs. It is nonconforming and makes no before/after claim.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XT (gfx1100), 20 GiB; CPU: AMD EPYC 7K62, 48 CPUs given to the container; RAM: 128 GiB; storage: not measured
- OS: Linux 7.0.14 (Proxmox kernel) host, NixOS LXC with the GPU passed through (`/dev/kfd`, `/dev/dri`); driver: host amdgpu
- ROCm: nixpkgs system ROCm, **7.2.3 for v0.1.44.1 and 7.2.4 for v0.2.0-beta** (the OS closure was updated between the runs, see Limits)
- Strata: source build via `setup.sh --backend hip` of tag `v0.1.44.1`, then of tag `v0.2.0-beta` (engine 0.1.45.3)
- Background workloads and power limits: not controlled; the server is shared with coding agents. Load average was about 3.5 shortly before the first run.

## Model and configuration

- Unsloth Qwen3.8-Flash-Next UD-Q5_K_XL (6 shards), all routed experts in RAM, 262,144-token context, `--kv int8`, vision off
- Identical run configuration on both builds, setup's output plus `--vram-reserve-mib 1024 --prefill auto:32768`
  (the tuning measured in the 2026-10-15 report); MTP, expert cache and speculation as setup wrote them
- Sampling: temperature 0, thinking off (the suite's default)

## Method

`python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long` against the already running server, one client,
one warm-up per run. The v0.1.44.1 run finished (03:32-03:41 UTC) before the update. v0.2.0-beta was started after the engine
rebuild, left to warm for about two minutes, and measured with the same command. Per-run data: `v0.1.44.1/results.json`,
`v0.2.0-beta/results.json`; the tool's own reports are `v0.1.44.1/README.md` and `v0.2.0-beta/README.md`.

## Results (medians; ranges in parentheses; 3 runs per prompt test, 9 chat requests, 15 agent turns)

| Test | v0.1.44.1 | v0.2.0-beta |
| --- | --- | --- |
| short chat decode, tok/s | 16.2 (8.4-21.5) | 18.6 (14.4-22.3) |
| short chat first token, s | 1.99 (1.69-4.04) | 1.79 (1.51-2.61) |
| 4,019-token prompt, tok/s | 226.2 (172.1-311.1) | 827.7 (798.6-834.0) |
| 4,019-token first token, s | 17.84 (12.97-23.45) | 4.92 (4.89-5.09) |
| 16,058-token prompt, tok/s | 833.3 (676.2-1049.3) | 1152.6 (1094.2-1158.0) |
| 16,058-token first token, s | 19.33 (15.39-23.82) | 13.98 (13.92-14.75) |
| 32,049-token prompt, tok/s | 901.7 (582.1-999.1) | 1050.2 (1045.2-1119.5) |
| 32,049-token first token, s | 35.61 (32.17-55.13) | 30.59 (28.70-30.75) |
| agent session, conversation-cache hit | 68.8 % | 69.3 % |
| agent session, follow-up first token, s | 5.79 (4.46-11.02) | 5.67 (4.35-6.55) |
| agent session, decode tok/s | 22.3 (5.8-30.0) | 21.2 (12.3-27.2) |

No failed requests in either run.

## Correctness and limitations

- **One pair of sessions, not a controlled A/B.** The v0.1.44.1 numbers vary much more (4K prompt 172-311 tok/s; 32K 582-999)
  than the 2026-10-15 tuned run on the same build (4K 577, 16K 888, 32K 1116 tok/s), and its 4K result is far below that
  earlier one, so the 3.7x 4K difference above should not be read as a speedup; the cause of that baseline's low, wide 4K
  numbers is unknown. The decode and agent-turn differences are within the run-to-run ranges. What the pair shows: prompt
  throughput at 16K and 32K is higher (+38 % and +16 % in the medians) and steadier on v0.2.0-beta, and decode and agent
  follow-ups are not measurably different. Against the 2026-10-15 numbers the 32K figure is lower (1050 vs 1116).
- **Two things changed between the runs, not one.** The OS update that moved nixpkgs ROCm 7.2.3 -> 7.2.4 and the Strata
  tag are mixed. No run of v0.2.0-beta on ROCm 7.2.3 or of v0.1.44.1 on ROCm 7.2.4 was made.
- Setup reused a CMake cache from the v0.1.44.1 build; it still pointed at the old ROCm and glibc and the first link of the
  v0.2.0-beta engine failed (`undefined reference: sqrtf@GLIBC_2.43` from librocblas). Moving the `build-hip` directory
  aside and rerunning setup fixed it. Only relevant to a toolchain change between builds, noted for anyone who updates
  ROCm and Strata together.
- Measured only in the protocol pilot, on v0.2.0-beta only: needle recall (6/6 at 32K and 124K), concurrency (capacity 1, queued),
  VRAM/RAM/clock/power telemetry. Not measured on v0.1.44.1, so none of it supports a version comparison. Still not measured:
  262K prompts, storage, and any of these on other backends.
  The report's tool left the GPU name as "AMD Radeon" and replaced the service user name in the Strata version line; both
  were corrected by hand in the copied reports. Nothing else was edited.
