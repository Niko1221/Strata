# KV cache precision, measured at the logits (2026-10-12)

Teacher-forced, on the shipping model (Swift Qwen3.8 Flash Next, native IQ3_XXS pack, RTX 5080, CUDA 13.0).
This is the logit-level measurement the KV work has been missing: the parity gate measures attention-output
error on a synthetic cache, and the recall probe measures behaviour end to end. Neither says what a KV
format does to the model's output distribution. This does.

## How

`--dump-prompt-logits PATH` (new) runs the language head over **every prompt position** inside the prefill.
Row *t* is the distribution the model had at position *t* with the KV state as it stood there - the chunk's
residual rows are causal (row *t* read only the cells up to its own position, the GDN recurrence walked to
step *t*, the PLE rows are per position), so this is teacher forcing, not the state after the whole prompt.
The head is the engine's own: `core::lm_head` on a canonical pack, and the native pack's
`lm_head_mix` + `native_quantize_q8_1` + `native_mmvq` - the same calls the verify window makes.

`tools/kv_precision_compare.py` runs the arms and computes, over the prompt positions (row *k* predicts
token *k+1*):

- **Same top-1 as fp16** - argmax agreement.
- **Mean KL(fp16 || arm)** - `sum p_fp16 * (log p_fp16 - log p_arm)`, averaged over positions.
- **Perplexity** - `exp(mean(-log p_arm(token_{k+1})))`, as a ratio to fp16.
- **nats/token vs fp16** - mean of `-log p_arm(t) + log p_fp16(t)`, with SE = sd/sqrt(n); "indistinguishable"
  when |delta| < 2 SE.

Texts: a code-agent style text, a document style text, a chat style text, 512 tokens each (tokenized with
`tools/strata_tokenizer.py`; the engine has no tokenizer). 511 scored positions per arm. Same engine, same
tokens, same expert profile - only `--kv` and the window env differ.

## Results

| Text | KV | Same top-1 as fp16 | Mean KL | Perplexity (x fp16) | nats/token vs fp16 |
|---|---|---:|---:|---:|---:|
| code agent | fp16 | 100.0% | 0.0000 | 1.000 | - |
| | int8 | 93.6% | 0.0227 | 1.022 | -0.022 +- 0.012 (indistinguishable) |
| | q4_0 | 89.5% | 0.0539 | 0.990 | +0.010 +- 0.018 (indistinguishable) |
| | q2 | 73.6% | 0.3241 | 1.021 | -0.021 +- 0.044 (indistinguishable) |
| | q2+window | 75.2% | 0.3193 | 1.045 | -0.044 +- 0.042 (indistinguishable) |
| | q4_0+window | 89.6% | 0.0640 | 1.012 | -0.012 +- 0.020 (indistinguishable) |
| document | fp16 | 100.0% | 0.0000 | 1.000 | - |
| | int8 | 97.1% | 0.0117 | 1.010 | -0.010 +- 0.007 (indistinguishable) |
| | q4_0 | 92.6% | 0.0310 | 1.021 | -0.021 +- 0.012 (indistinguishable) |
| | q2 | 77.5% | 0.2053 | 0.949 | +0.053 +- 0.029 (indistinguishable) |
| | q2+window | 79.1% | 0.2043 | 0.945 | +0.056 +- 0.029 (indistinguishable) |
| | q4_0+window | 92.4% | 0.0307 | 1.030 | -0.030 +- 0.014 |
| chat | fp16 | 100.0% | 0.0000 | 1.000 | - |
| | int8 | 97.1% | 0.0095 | 0.991 | +0.009 +- 0.007 (indistinguishable) |
| | q4_0 | 93.6% | 0.0275 | 0.992 | +0.008 +- 0.011 (indistinguishable) |
| | q2 | 81.1% | 0.1746 | 0.911 | +0.094 +- 0.033 |
| | q2+window | 82.4% | 0.1756 | 0.891 | +0.116 +- 0.034 |
| | q4_0+window | 93.6% | 0.0237 | 0.984 | +0.016 +- 0.013 (indistinguishable) |

First generated token per arm: identical across all six arms on the code-agent and chat texts. On the
document text fp16/int8/q4_0/q4_0+window all pick 18169 while q2 picks 1105 and q2+window picks 3813 -
the 2-bit cache changes the model's next token on that text.

## What this says

- **q4_0 is at the noise floor.** Mean KL 0.028-0.054, top-1 agreement 89.5-93.6%, perplexity within 1-2%
  of fp16, and the nats/token delta is inside 2 SE on 5 of 6 arms. Per-cell cost: 576 B vs 2048 B fp16.
- **q2 (per-channel 2-bit, 80 B/cell/head = 320 B/cell) is measurably lossy at the logits.** Mean KL
  0.175-0.324 - 5-6x q4_0 - and top-1 agreement drops to 73.6-81.1% from 89.5-93.6%. On the chat text the
  perplexity cost is significant (+0.094 +- 0.033 nats/token, and +0.116 +- 0.034 with the window).
  **This is the number that settles "near lossless": q2 is not there.** It agrees with the gate
  (attention output 7.15% vs q4_0's 2.05%) and with the recall probe (q2 loses a short-answer question that
  q4_0 answers).
- **The FP16 sliding window does not recover logit-level precision.** q2+window is within noise of plain q2
  on every metric (top-1 +1.2 to +1.6 points, KL equal, perplexity equal or slightly worse). The window is a
  *no-regression* device - it protects the recent span and the sinks - not a precision device; the loss is
  in the quantized bulk, which the window does not touch.
- **q2's perplexity is *below* fp16 on two texts (0.949, 0.911).** Quantization noise acts as a smoother on
  highly predictable text: the 2-bit cache assigns more probability to the tokens that a sharper model
  spreads over near-ties. Perplexity alone would call q2 an improvement; top-1 agreement and KL are the
  metrics that do not flatter it, which is why the harness prints all four.

## Caveats

- These rows come from the **prefill** path (full chunk attention), not the decode path (sparse QSA). The
  two agree closely at the last prompt position - top-1 and top-5 identical, 93/100 top-100 overlap,
  KL 0.082 nats - but they are not bitwise the same computation, and the absolute perplexity here is the
  prefill path's.
- 512 tokens per text, 511 scored positions. The SEs are within-text; they do not cover text-to-text
  variation, which is why three texts are reported separately rather than pooled.
- `--dump-prompt-logits` is a measurement path: the head runs once per position. It costs ~14 s per 512-token
  arm including engine load, and it changes nothing about a normal run (verified: same first token with and
  without the flag).

## Reproduce

```
cmd.exe /C build-cu130.bat
set KV_ENGINE=C:\Users\blsor\Strata\build-cu130\strata.exe
set KV_LIB_DIRS=C:\Users\blsor\cuda130\nvidia\cu13\bin;C:\Users\blsor\cuda130\nvidia\cu13\bin\x64
.venv\Scripts\python.exe tools\kv_precision_compare.py --max-tokens 512 --keep-logits
```

Raw logits per arm are kept with `--keep-logits` (`tmp\kv-precision\*.logits.bin`); check any of them with
`tools\logits_dump_check.py FILE [EXPECTED_ROWS]`. Numbers above are from
`tmp\kv-precision\kv_precision_compare.json`.
