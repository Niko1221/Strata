# RX 6600M (gfx1032), i5-13600: CPU-pool worker count on Windows, one processor group (2026-10-29)

This packet holds the raw engine output behind the RX 6600M section of
[the Windows workers benchmark](../../../docs/benchmarks/windows-zen2-gfx1030-workers.md#one-processor-group-with-streamed-experts-rx-6600m-gfx1032).
The card has 8 GiB of VRAM and holds 1,165 of the model's 12,288 (layer, expert) pairs, so most routed experts
are computed by the CPU pool; the question here was whether that pool's worker count limits decode.

All runs used explicit `--pool-workers` except the two labelled automatic, which used the engine's own choice.
Each setting ran twice, in one session, in the order listed in [timeline.txt](timeline.txt).

## Rig

- AMD Radeon RX 6600M 8 GiB (gfx1032, wave32, 14 multiprocessors), 13th Gen Intel Core i5-13600 (6 performance
  cores, 8 efficiency cores, 14 physical, 20 logical, **one** processor group), 95.7 GB RAM, Windows 11 build
  10.0.26100.3037.
- Engine 0.1.53 built from Strata commit `46c152a1` in a source checkout: `-DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1032 -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON`,
  ROCm 10.2.0a20260930 (TheRock wheels, no admin rights). The hand build for this card is described in
  [docs/RX6600M.md](../../../docs/RX6600M.md).
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ1_M coder pack (native IQ pack, 256 experts, 48 layers), prompt chunk
  automatic, 2,048-cell context, `--spec 4`, `--expert-cache auto` (1,165 slots, 2.25 GiB), greedy sampling,
  the same five-token prompt (ids `760,6511,314,9338,369`) in every run, 64 new tokens except the two length
  runs. CPU pool tasks per phase: 30 (automatic).

Command (the worker count is the only flag that changes between rows; it is omitted for the automatic runs):

```bat
build-hip-gfx1032\strata.exe --pack <pack> ^
  --native <model>-00001-of-00002.gguf ^
  --spec 4 --prefill auto --expert-profile data\expert-profile-coder.bin ^
  --expert-cache auto --max-context 2048 --max-new 64 ^
  --tokens "760,6511,314,9338,369" --pool-workers <N>
```

## Worker sweep, 64 tokens per run

| Explicit workers | Decode tok/s | Decode ms | Pool ms/round | Wait for rings ms/round | Log |
| ---: | --- | ---: | --- | --- | --- |
| automatic | 3.89 | 16,443.6 | 53.6 | 214.8 | [auto-r1.log](auto-r1.log.txt) |
| automatic | 4.08 | 15,672.2 | 52.3 | 207.8 | [auto-r2.log](auto-r2.log.txt) |
| 1 | 3.81 | 16,799.1 | 107.9 | 173.8 | [w01-r1.log](w01-r1.log.txt) |
| 1 | 3.72 | 17,221.3 | 117.8 | 170.9 | [w01-r2.log](w01-r2.log.txt) |
| 3 | 4.16 | 15,398.7 | 63.9 | 192.4 | [w03-r1.log](w03-r1.log.txt) |
| 3 | 4.07 | 15,738.3 | 74.8 | 175.0 | [w03-r2.log](w03-r2.log.txt) |
| 5 | 4.16 | 15,377.7 | 49.8 | 205.3 | [w05-r1.log](w05-r1.log.txt) |
| 5 | 4.06 | 15,768.7 | 54.6 | 207.0 | [w05-r2.log](w05-r2.log.txt) |
| 9 | 4.15 | 15,436.3 | 51.3 | 204.9 | [w09-r1.log](w09-r1.log.txt) |
| 9 | 4.01 | 15,976.2 | 54.7 | 210.5 | [w09-r2.log](w09-r2.log.txt) |
| 13 | 4.17 | 15,355.0 | 46.2 | 208.4 | [w13-r1.log](w13-r1.log.txt) |
| 13 | 4.02 | 15,911.7 | 49.8 | 213.7 | [w13-r2.log](w13-r2.log.txt) |

The automatic choice was 9 workers: five performance cores after the host's, plus four of the eight
efficiency cores. The pool's share of a round falls from 108 ms at 1 worker to 46 ms at 13, and the wait for
the rings rises from 174 to 208 ms, so decode stays inside 3.72-4.17 tok/s and only the 1-worker setting is
outside the run-to-run spread (two runs at the same setting differed by up to 7%). All twelve runs generated
the same 64 token ids, byte for byte: the flag moves threads only, never a result.

## Length check, 9 workers

Same session and settings, explicit `--pool-workers 9`, new-token count varied:

| New tokens | Decode tok/s | Decode ms | Rounds | Decode ms/round | Pool ms/round | Wait for rings ms/round | Log |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 5.71 | 2,800.5 | 15 | 187 | 76.3 | 88.4 | [len16.log](len16.log.txt) |
| 32 | 4.63 | 6,911.4 | 29 | 238 | 62.0 | 155.6 | [len32.log](len32.log.txt) |
| 64 | 4.15 | 15,436.3 | 56 | 276 | 51.3 | 204.9 | [w09-r1.log](w09-r1.log.txt) |
| 64 | 4.01 | 15,976.2 | 56 | 285 | 54.7 | 210.5 | [w09-r2.log](w09-r2.log.txt) |

The 16-token run repeats the 5.70 tok/s this card measured in its first GPU run. Over a longer run the average
round grows from 187 ms to about 280 ms while the pool's share of it falls, and the adaptive tier moved 1,087
to 1,102 experts into the VRAM tier during the 64-token runs (0.05-0.06 ms/round). The cause of the growth was
not separated from the change in the routing mix (13.88 distinct CPU experts per layer at 16 tokens, 9.57 at
64), so these rows are rates over a run of that length, not a different build.

## Limits

- One machine, one card, one model, one prompt. Two runs per setting; the desktop was not idle, ambient
  conditions and VRAM state were not controlled.
- One processor group, so nothing here exercises the group-affinity code path; it is a control for it.
- The 16- and 32-token rows are one run each, not medians.
- `--serve`, contexts above 2,048 cells, MTP drafting and multiple GPUs were not tested, as in
  [docs/RX6600M.md](../../../docs/RX6600M.md).
