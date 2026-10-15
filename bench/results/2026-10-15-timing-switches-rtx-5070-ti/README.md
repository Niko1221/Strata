# The cost of the engine's timing switches: RTX 5070 Ti 16 GB, Ryzen 7 5800X3D, Swift 1.5 IQ3_S, 2026-10-15

`STRATA_PREFILL_TIMING=1` and `STRATA_DECODE_TIMING=1` (what the Profile page and `tools/strata_profile.py` set) against the engine without them: `tools/ab_engine.py strata-swift-iq3_s.json "off:" "on:STRATA_PREFILL_TIMING=1,STRATA_DECODE_TIMING=1" --rounds 3`, a restart per arm, the arms' order alternating per round, greedy, 256 new tokens, thinking off. Release engine 0.1.44.1 (Strata 0.1.44.1), 262K context, INT8 KV, `--kv-resident 32768`, `--prefill auto`, `--spec 4`, Linux, driver 595.99.02. The engine's own timings (prompt_per_second, predicted_per_second). Raw rows: ab_engine.jsonl.

| Request | Round | off: decode / prompt tok/s | on: decode / prompt tok/s | decode | prompt |
| --- | ---: | --- | --- | ---: | ---: |
| chat0 | 1 | 77.0 / 86 | 78.3 / 84 | +1.7% | -1.8% |
| chat0 | 2 | 78.0 / 82 | 75.9 / 85 | -2.7% | +3.8% |
| chat0 | 3 | 78.6 / 86 | 78.6 / 86 | +0.0% | +0.0% |
| chat1 | 1 | 79.8 / 74 | 81.5 / 82 | +2.1% | +9.7% |
| chat1 | 2 | 78.9 / 83 | 75.0 / 72 | -4.9% | -12.6% |
| chat1 | 3 | 81.4 / 82 | 75.8 / 75 | -6.9% | -8.3% |
| chat2 | 1 | 76.9 / 75 | 77.0 / 73 | +0.1% | -1.7% |
| chat2 | 2 | 79.5 / 75 | 79.5 / 68 | +0.0% | -9.1% |
| chat2 | 3 | 77.1 / 74 | 77.4 / 69 | +0.4% | -6.5% |
| long (8,020-token prompt) | 1 | 70.4 / 2710 | 75.2 / 2582 | +6.8% | -4.7% |
| long (8,020-token prompt) | 2 | 75.4 / 2702 | 74.4 / 2585 | -1.3% | -4.3% |
| long (8,020-token prompt) | 3 | 75.6 / 2699 | 71.0 / 2592 | -6.1% | -3.9% |

The 8K prompt read is 3.9-4.7% slower with the switches on in every pair (the prefill timer records a CUDA event per phase per chunk). The decode moves -6.9% to +6.8% with no consistent sign, so its timestamps per window cost nothing measurable at this noise. The chat prompts are 40-50 tokens: their prompt tok/s is a few tens of milliseconds of work and moves with the noise.
