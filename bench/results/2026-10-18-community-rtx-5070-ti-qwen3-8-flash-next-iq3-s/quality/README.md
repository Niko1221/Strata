# IQ3_S against Q8_0 on this PC: the quant yardstick, and three settings

Measured on 2026-10-18 on the same machine as the speed report in [../README.md](../README.md) (RTX 5070 Ti 16 GB, Core Ultra 7 270K Plus,
63.5 GiB RAM, Windows 11), engine 0.1.45 (`477ac13c`), with `tools/quant_quality.py` of the same commit and its text
set `data/quant_quality/v1` (code, docs, English, German/French/Japanese/Chinese, an agent transcript: 4,995 scored
positions). Q8_0 is Unsloth's file, run on this PC as the reference. Every KL below is the tool's: KL(Q8_0 || run)
over Q8_0's 64 most likely tokens.

- **IQ3_S matches the published yardstick:** 87.51% the same top token as Q8_0, mean KL 0.142, perplexity 4.061
  against Q8_0's 3.997 (docs/QUANTS.md, the R9700 box: 87.69%, 0.144, 4.097 against 4.052).
- **The settings of this PC's server** (GPU expert cache, tail skip off, int8 K/V, the FP8 n-gram table):
  87.11%, KL 0.1435, perplexity 4.036. Both models guess the text's real next token 68.7% of the time.
- **Per text, the gap is the agent transcript:** perplexity 4.72-4.95 in every IQ3_S run with the FP8 table, against
  Q8_0's 4.36 (+8-13%), KL 0.30. Code, docs and the other languages are ties on perplexity; English is below Q8_0's
  in every run (one short story).
- **`STRATA_ROUTE_TAIL_SKIP=7` against `=0`:** +0.0104 KL against Q8_0, at the edge of zero overall (95% range of
  32-position blocks +0.0004 to +0.0196; other block sizes and resamples put its lower end between -0.0003 and
  +0.0005), but positive and separated from zero on four of the five texts (code, docs, English, the other
  languages); the agent text is noisy (-0.0054, -0.055 to +0.037). 1,479 positions worse and 929 better by more than
  0.01. Compared with each other directly: 90.61% the same top token, KL 0.075 (see
  [the observations](#observations-for-the-maintainers)).
- **int8 against FP16 K/V:** no difference against Q8_0 (-0.0001, 95% range -0.0083 to +0.0084) at these lengths.
- **The IQ4_NL n-gram table against FP8:** +0.0033 overall (-0.0028 to +0.0102, not separated), +0.030 on the agent
  text (+0.005 to +0.064) and -0.009 on code (-0.018 to -0.001).

## Runs

| Run (`data/`) | Experts | n-gram table | Tail skip | K/V |
| --- | --- | --- | --- | --- |
| `q8_0` | CPU (1 VRAM slot), 39 GiB in RAM, the rest read from the GGUF | the file's own (Q8_0) | 0 | FP16 |
| `iq3s_cpu` | CPU (1 VRAM slot), every expert in RAM | FP8 (`ple-fp8.gguf`) | 0 | FP16 |
| `iq3s_cpu_iq4nl` | as `iq3s_cpu` | the GGUF's IQ4_NL | 0 | FP16 |
| `skip0` | `--gpu-experts auto`: 4,396 slots, 8.39 GiB | FP8 | 0 | FP16 |
| `skip7` | as `skip0` | FP8 | 7 | FP16 |
| `skip0_int8` | as `skip0` | FP8 | 0 | int8 |

`skip0` run a second time gave byte-identical rows. The Q8_0 run was made once.

## Results

All runs against Q8_0, all five texts (`results.txt` has every table):

| Run | Same top token | Mean KL | KL p99 | Positions with KL > 1 | Perplexity (Q8_0: 3.997) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iq3s_cpu` | 87.51% | 0.1421 | 1.81 | 112 | 4.061 |
| `iq3s_cpu_iq4nl` | 87.43% | 0.1454 | 1.92 | 117 | 4.113 |
| `skip0` | 86.97% | 0.1436 | 1.77 | 122 | 4.068 |
| `skip7` | 86.67% | 0.1540 | 2.13 | 141 | 4.062 |
| `skip0_int8` (this PC's settings) | 87.11% | 0.1435 | 1.81 | 127 | 4.036 |

`skip0_int8` per text:

| Text | Same top token | Mean KL | KL > 1 | Real next token is the top guess: IQ3_S / Q8_0 | Perplexity: IQ3_S / Q8_0 |
| --- | ---: | ---: | ---: | --- | --- |
| code | 93.3% | 0.083 | 20 | 84.2% / 84.4% | 2.035 / 2.040 |
| docs | 77.5% | 0.203 | 27 | 49.7% / 49.7% | 10.010 / 9.974 |
| english | 86.4% | 0.085 | 4 | 60.5% / 59.1% | 4.139 / 4.252 |
| multilingual | 92.5% | 0.047 | 3 | 77.2% / 77.6% | 2.694 / 2.703 |
| agent | 85.9% | 0.299 | 73 | 72.1% / 72.6% | 4.718 / 4.363 |
| all | 87.1% | 0.143 | 127 | 68.7% / 68.7% | 4.036 / 3.997 |

Where `skip0_int8` and Q8_0 pick different top tokens (644 positions), the text's real next token is IQ3_S's pick at
158, Q8_0's at 155 and neither at 331.

Paired differences, per position (run B's KL minus run A's, both against Q8_0). The 95% ranges come from 2,000
bootstrap resamples of blocks of 32 consecutive positions within each text, since neighbouring tokens are not
independent; `results.txt` also has the ranges of single positions, which are a little narrower:

| Change | All texts | code | docs | english | multilingual | agent |
| --- | --- | --- | --- | --- | --- | --- |
| tail skip 7 vs 0 (`skip7` - `skip0`) | **+0.0104** [+0.0004, +0.0196] | +0.0103 [+0.0008, +0.0201] | +0.0108 [+0.0021, +0.0208] | +0.0251 [+0.0172, +0.0343] | +0.0113 [+0.0051, +0.0193] | -0.0054 [-0.0549, +0.0371] |
| IQ4_NL vs FP8 table (`iq3s_cpu_iq4nl` - `iq3s_cpu`) | +0.0033 [-0.0028, +0.0102] | -0.0089 [-0.0179, -0.0014] | -0.0020 [-0.0111, +0.0071] | -0.0014 [-0.0041, +0.0016] | -0.0014 [-0.0053, +0.0018] | +0.0302 [+0.0049, +0.0635] |
| int8 vs FP16 K/V (`skip0_int8` - `skip0`) | -0.0001 [-0.0083, +0.0084] | +0.0038 [-0.0005, +0.0085] | +0.0078 [+0.0002, +0.0163] | -0.0002 [-0.0032, +0.0030] | -0.0018 [-0.0041, +0.0001] | -0.0102 [-0.0440, +0.0287] |
| GPU vs CPU experts (`skip0` - `iq3s_cpu`) | +0.0015 [-0.0034, +0.0067] | -0.0042 [-0.0106, +0.0016] | -0.0020 [-0.0102, +0.0055] | +0.0020 [-0.0006, +0.0049] | -0.0015 [-0.0074, +0.0026] | +0.0131 [-0.0086, +0.0345] |

The table holds 20 per-text ranges, so one or two of them separated from zero by chance alone is expected (int8 on
docs, +0.0078, is one candidate). The last row is the placement alone: 0.54 points of top-token agreement and +0.0015
KL, so top-token agreement and p99 move by that much without a change in the weights or the settings.

## How the runs were made

- **Q8_0:** the six shards of `unsloth/Qwen3.8-Flash-Next-GGUF` at setup's pinned revision (`38bb39ee`), fetched with
  setup.py's own `download()` and checked against its pinned SHA-256 values. The pack is setup's step 6 for the
  unsloth family: `tools/iq_pack.py --gguf ...-Q8_0-00001-of-00006.gguf --out packs/unsloth-q8_0 --compat-bf16` (1.4
  GB, 21 s), so the hyper-connection projections are the BF16 copy (`STRATA_HC_Q8` unset). The config was written by
  hand with setup's arguments for this PC (`configs/q8_0.json`; `--resident-budget-gib 39` is setup's
  `resident_budget_gib("Q8_0", 63.5)`); setup itself was not run. The engine kept 39.00 GiB of experts page-locked
  (8,018 by profile rank) and read the rest from the GGUF: 248 / 168 / 234 / 233 / 235 s per text, 863 GB of file
  reads during the first text. Q8_0 decode speed was not measured.
- **IQ3_S:** this PC's config (`configs/iq3_s.json`) without `--kv-grow` and the conversation-cache flags, with
  `STRATA_IQ3S_MT1=1` in its env; the tool sets the rest (`--expert-cache 1` or `--gpu-experts`, `--kv fp16`, no
  adaptation, no PCIe share, `STRATA_ROUTE_TAIL_SKIP=0` unless `--env` sets it, `STRATA_PREFILL_CPU_SHARE=0`). About
  12 s per text with the experts on the CPU, 8-10 s with the GPU cache.
- **int8 K/V:** `qq_kv.py` runs the tool with its fixed `--kv fp16` replaced by `--kv int8`; nothing else differs.
- The commands: `run.sh`. The engine's start-up lines of every run: `engine-start-excerpts.txt`.
- **On Windows:** `quant_quality.py run` puts the config's `lib_dirs` into `LD_LIBRARY_PATH`, which Windows does not
  read, so the engine cannot find the CUDA DLLs; `run.sh` puts them on `PATH`. Its RAM check (`mem_available_gb`)
  reads `/proc/meminfo` and returns infinity on Windows, so the 16 GB guard does not apply there.

## Observations for the maintainers

- **The tail skip's KL here is larger than DETAILS.md's table.** DETAILS.md ("Answers differ slightly from 0.1.41")
  gives KL(0.1.41 behaviour || tail skip) 0.0058 (RTX 3060, IQ3_XXS) and 0.0032 (A4000, Q2_0), on generated chat text
  through the MTP verify windows. Here, `skip7` against `skip0` directly (`compare skip0.tsv skip7.tsv`): 90.61% the
  same top token, KL 0.0753, p99 1.15; against Q8_0 the skip adds +0.0104, at the edge of zero overall and
  separated on code, docs, English and the other languages, not on the agent text. The method differs (the yardstick's teacher-forced reading of fixed
  text). The decode hit rate the engine logged per text was 19-60% (`skip0`) and 24-69% (`skip7`); this PC's own
  server's request log averages 51-55% (agent use with Claude Code, engines 0.1.38 to 0.1.45). NVFP4.md's yardstick row points the same way
  (0.069 to 0.118 with the skip on). No default change is proposed here; this PC's server runs with
  `STRATA_ROUTE_TAIL_SKIP=0`.
- **At several of the agent text's largest KL positions, Q8_0 is the one far from the text** (positions are the
  `pos` column of the agent segment): at 41, after `<|im_end|>\n<|im_start|>`, the text continues `user`, Q8_0 puts
  90% on `<|im_start|>` and IQ3_S picks `user`; at 145, in pytest's `[100`, the text has `%`, Q8_0 puts 38% on `0`;
  at 793, after `if`, the text has ` m`, Q8_0 puts 28% on ` it`. Observed in this one Q8_0 run; a row-by-row check
  found no position shift (Q8_0's top token equals the previous token at 0.2% of positions). A Q8_0 run on another
  machine would tell whether this is the file or this run.

## Not tested

One text of about 1,000 tokens per subject, so a per-text difference rests on one sample. Short context only (each
text is read alone, `--max-context` 1,512); nothing about 100K+ conversations. Teacher-forced reading only: no
generated answers, no task success. The live server's adaptive expert swaps, `--kv-grow` and conversation cache are
not part of these runs. Q8_0 was run once, on this PC only.

## Files

- `data/q8_0.tsv.gz`: the Q8_0 run in full, the engine's `STRATA_LOGPOS` rows (pos, target id, logp(target), argmax
  id, logp(argmax), argmax == target, logp(`<|im_end|>`), logp(target without `<|im_end|>`), then 64 `id:logp`;
  natural log), gzipped. No Q8_0 reference ships with Strata, so a `quant_quality.py run` of another quant can be
  compared against it (`gunzip`, then `compare q8_0.tsv RUN.tsv`); it is this PC's Q8_0 (CUDA + AVX2 CPU), and the
  same file on another backend differs (docs/QUANTS.md).
- `data/positions.tsv.gz`: one row per position for every IQ3_S run (logp(target), top token, KL against Q8_0), made
  from the full runs by `make_positions.py`; the full IQ3_S runs (about 2.2 MB each, gzipped) are not included.
- `data/<run>.tsv.seg.json`: each run's line range per text, engine arguments and env (paths shortened to
  `<Strata>`, `<Strata-data>`, `<run dir>`).
- `analyze.py`: every table above from `data/positions.tsv.gz` (`python analyze.py`, standard library only);
  `results.txt`: its output.
- `configs/`: the three configs (no API key). `run.sh`, `qq_kv.py`, `engine-start-excerpts.txt`: see above.
