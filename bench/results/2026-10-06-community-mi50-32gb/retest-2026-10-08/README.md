# Follow-up (2026-10-08): the 16 GB-limit 252K case on 0.1.40.3 and 0.1.41

Requested in the review of the main run: repeat the case that answered `!!!!...` on 0.1.40.1 (card limited to 16 GB with
`--vram-reserve-mib 16384`, `--max-context 262144`, a ~250K-token prompt with three needles at 10 / 50 / 90 %) and say whether
it shows again. Same MI50 32 GB, same host, same model (IQ2_XS) and same engine arguments as
[../strata-iq2_xs-16gb-ctx262144.json](../strata-iq2_xs-16gb-ctx262144.json) (the configs actually used are the two `*-cfg-*.json` here).

**Result: it did not show again.** On both versions, with two different prompts (seeds 23 and 11), all three needles were
returned (3 of 3) and nothing resembling `!!!!` appeared in the answers or in the engine and server logs.

| Version (commit) | Prompt (seed) | Prefill, 252K cold | Needles | Decode, 300 tokens, prefix reused | Expert cache hit rate in that decode |
|---|---|---:|---|---:|---:|
| 0.1.40.3 (`d5ea713`) | 23 (251,950 tok) | 363 tok/s (693 s) | 3 of 3 | 29.1 tok/s | 90.2% |
| 0.1.40.3 (`d5ea713`) | 11 (252,304 tok) | 364 tok/s (694 s) | 3 of 3 | 33.6 tok/s | 96.0% |
| 0.1.41 (`fb58e0d`) | 23 (251,950 tok) | 362 tok/s (696 s) | 3 of 3 | 29.1 tok/s | 90.2% |
| 0.1.41 (`fb58e0d`) | 11 (252,304 tok) | 361 tok/s (698 s) | 3 of 3 | 31.9 tok/s | 96.2% |

(The prefill figure is client-side for the seed-23 rows and the engine's own `strata serve:` line for the seed-11 rows; both are
in `*-ver-test.out`. The seed-23 prompt is the same text on both versions, and the same answer came back.)

## What was run

- Each version was built separately for gfx906 (`-DSTRATA_HIP_GFX906=ON`, ROCm 7.2.2, same flags as [../build-gfx906.sh](../build-gfx906.sh)),
  each started fresh with [run-retest.sh](run-retest.sh), one version at a time, the card cooled below 50 C before every request.
- **0.1.41 built with no local patch.** The three fixes we carried on 0.1.40.1 are no longer needed there.
- **0.1.40.3 needed one line**: `#define cudaEventBlockingSync hipEventBlockingSync` in
  `include/strata/platform/hip_compat/strata_hip.h` ([local-0.1.40.3.patch](local-0.1.40.3.patch)), which is the `cudaEventBlockingSync` part
  of #1396 that 0.1.41 already has; without it `src/prefill/prefill.cpp` fails to compile for HIP. Nothing else changed.
- One request pair per seed (the answer check, then a 300-token summary). The labels in the output files are in Portuguese
  (`versao ... pronto` = version ... ready, `agulhas` = needles, `tempo` = time, `gerados` = generated).

## Temperature (important for reading this)

Junction temperature (`rocm-smi`, every 10 s, in `*-ver-temps.csv`) peaked at **106 C on both runs** and spent
about 8 minutes at 105 C or above during the 252K prefill (48 and 52 readings at 105 or more, of 152 and 154 per run). The sysfs
critical threshold of this card is 105 C and the emergency one 110 C, so the card was at its own limit and throttled itself.
On 0.1.40.1 (the main run) the same case also sat at 103-104 C. This is the same card, cooling and host, so it is a fair
comparison between the versions, but it means a better-cooled card may behave differently.
Our own safety abort (stop the engine when junction stays at or above a threshold for two reads) was set to 104 C in the main run
and to 107 C here; the first two attempts at this retest were stopped by the 104 C setting before the prompt finished, which is why the setting was raised.

## What this does and does not show

- It shows the `!!!!...` answer did **not** recur in 4 of 4 runs on 0.1.40.3 and 0.1.41 for this case. It does not show what caused it on 0.1.40.1:
  we did not rerun 0.1.40.1 itself, and that failure came from a single run on a card that was at its thermal limit.
- Nothing here tests quality beyond the three-needle check, or any other card, host or model.
