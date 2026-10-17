# Long-context sweep, 4K to 255K prompt tokens

Companion to the [report above](../README.md). The stock suite tops out at 32K prompt tokens, so this sweep was run
to see how both quantizations behave out to the model's full 262,144-token window.

## Method

`bench3.py` in this folder is a lightly adapted copy of the repository's `bench/bench_vs_llama.py`: the request
body, sampler settings and the server-side `timings` fields it reads are unchanged. It was adapted only to take
cell sizes from the command line, to read one Strata config per model, and to set per-config environment variables.
The VRAM column that script samples with `nvidia-smi` is unavailable on AMD and is ignored.

Prompts are exact lengths built from 8 MB of this repository's own markdown docs and sources with Strata's
tokenizer (`tools/strata_tokenizer.py`). Each request is a fresh prompt with a unique nonce, greedy (temperature 0),
256-token cap, one run per cell, one model loaded at a time. Values are the engine's `prompt_per_second` and
`predicted_per_second`.

```bash
python bench3.py run --machine machine.json --out out
```

Raw records: [`iq2_xs.jsonl`](iq2_xs.jsonl), [`coder_iq1_m.jsonl`](coder_iq1_m.jsonl), [`qwen27b_q4.jsonl`](qwen27b_q4.jsonl).

## Results

### Flash-Next IQ2_XS

| Cell | Prompt tokens | Prefill tok/s | Decode tok/s | First token (s) |
| --- | ---: | ---: | ---: | ---: |
| 4K | 4,052 | 1,068.3 | 79.5 | 3.8 |
| 32K | 32,053 | 1,288.1 | 78.2 | 24.9 |
| 128K | 128,063 | 1,288.6 | 72.1 | 99.4 |
| 255K | 255,127 | 1,157.5 | 68.3 | 220.4 |

### Coder IQ1_M

| Cell | Prompt tokens | Prefill tok/s | Decode tok/s | First token (s) |
| --- | ---: | ---: | ---: | ---: |
| 4K | 4,054 | 1,378.3 | 68.1 | 2.9 |
| 32K | 32,055 | 1,630.6 | 78.0 | 19.7 |
| 128K | 128,065 | 1,526.1 | 66.9 | 83.9 |
| 255K | 255,129 | 1,346.2 | 63.7 | 189.5 |

## Notes

- Prompt throughput is flat from 32K to 128K and ~10 % lower at 255K. The 4K cell is *slower* than 32K because
  fixed per-request overhead (graph capture, cache probing) is a larger share of a short prompt.
- Decode loses little across the range: 14 % (IQ2_XS) and 7 % (Coder) from 4K to 255K. KV streaming
  (`--kv-resident 32768`, 3.09 GiB of pinned RAM) keeps the hot attention window in VRAM and spends the freed VRAM
  on experts.
- The absolute values here are 6-12 % apart from the stock suite's numbers for the same model and a similar prompt
  size. The two runs are different sessions with differently constructed prompts; use each table on its own.

## The third file

`qwen27b_q4.jsonl` is a **dense 27B via llama.cpp**, not a Strata run. It is kept here because it was measured in
the same session on the same card and the comparison is useful: prefill 330 / 279 / 180 tok/s and decode 30.9 /
20.5 / 10.0 tok/s at 4K / 32K / 128K, and the 255K cell never returned (the harness's 3600 s HTTP timeout). See the
appendix of the main report for why.
