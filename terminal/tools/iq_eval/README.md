# iq_eval: final-intelligence gate for lossy engine or model changes

Decides whether a lossy change (engine, quantization, cache, kernel) keeps the final intelligence of the model. Small output changes are
fine; accuracy on verifiable tasks must not drop. Teacher-forcing KL is only a screen; this is the final gate. It runs the reference
(unchanged) arm and the candidate arm on the same fixed items and tests the paired difference for non-inferiority.

## Files

| file | what it does |
| --- | --- |
| `subsets.py` | Builds `subsets.jsonl` (seed 0) from the pinned datasets: GSM8K 200, MATH-500 150, HumanEval 164 (all), MMLU-Pro 200 (stratified over the 14 categories), JCommonsenseQA 200 = 914 items. Holds the prompt templates (`\boxed{}` for math, "Answer with the letter only" for multiple choice, one code block for HumanEval), identical for every arm. Needs `pyarrow` (the project venv has none; use any Python that has it, for example the one that ships `hf`). |
| `run.py` | Starts `serve/server.py --engine strata --config CFG --port P` like `tools/ab_engine.py` (and its `wait_ready`), then sends each item to `/v1/chat/completions`: temperature 0, fixed `max_tokens` (default 4096), the model's default thinking (no `chat_template_kwargs`). Writes one JSONL row per item: output, reasoning, finish_reason, prompt and completion tokens, seconds. Resumable (ids already done are skipped; a failed request is stored with `error` and retried on the next run). |
| `score.py` | Scores a run: boxed or last number (GSM8K), boxed answer with normalized-string and numeric equivalence (MATH), letter extraction (MMLU-Pro, JCommonsenseQA), unit tests (HumanEval). A reply cut off inside its thinking has no answer and counts wrong. |
| `compare.py` | Compares two scored runs on the shared ids: per-set and pooled accuracy, paired difference with a bootstrap 95% CI (10,000 resamples of item pairs, seed 0), exact McNemar p, mean output tokens and the share cut off at `max_tokens` per arm, and the verdict. |
| `test_iq_eval.py` | pytest: extractors, the HumanEval runner (correct, wrong, timeout), `compare.py`, and `run.py` against a fake HTTP server. No engine, no GPU. |

## Data

The datasets live outside the repo in `D:\exstrata-work\evalsets` (pins, licenses, sizes and sha256 in `pins.json`). They must not be committed.
`subsets.jsonl` is derived from them (including JCommonsenseQA, CC BY-SA 4.0) and stays there too.

## Commands

```
# once: the subsets (deterministic; rerunning gives the same file)
<python with pyarrow> terminal/tools/iq_eval/subsets.py

# each arm: start the engine, run all 914 items, stop the engine
PY=D:\works\projects\experiments\exstrata\.venv\Scripts\python.exe
$PY terminal/tools/iq_eval/run.py CFG_REF.json  --out D:\exstrata-work\iq\ref.jsonl
$PY terminal/tools/iq_eval/run.py CFG_CAND.json --out D:\exstrata-work\iq\cand.jsonl [--exe PATH] [--xargs "a;b"] [--env K=V]
# a server that is already running:  run.py --base-url http://127.0.0.1:8090 --out ...
# smoke test:                        run.py CFG --out x.jsonl --limit 3 --sets gsm8k,jcommonsenseqa

$PY terminal/tools/iq_eval/score.py D:\exstrata-work\iq\ref.jsonl  --out D:\exstrata-work\iq\ref_scored.jsonl
$PY terminal/tools/iq_eval/score.py D:\exstrata-work\iq\cand.jsonl --out D:\exstrata-work\iq\cand_scored.jsonl
$PY terminal/tools/iq_eval/compare.py D:\exstrata-work\iq\ref_scored.jsonl D:\exstrata-work\iq\cand_scored.jsonl [--margin 2]

# tests
$PY -m pytest terminal/tools/iq_eval/test_iq_eval.py
```

`compare.py` exits 0 for NON-INFERIOR, 1 for INCONCLUSIVE, 2 for INFERIOR. Verdict rule on the pooled accuracy (all items, micro): the
bootstrap 95% CI of (candidate - reference) has its lower bound above -margin (default 2 percentage points) = non-inferior; its upper
bound below -margin = inferior; otherwise inconclusive (more items or a repeat is needed, not a pass). Run both arms with the same
`--max-tokens` and the same subsets file.

## Safety: HumanEval runs model-written code

`score.py` executes the code the model wrote, on this machine. Each item gets a fresh Python process (`python -I`, the interpreter that runs
`score.py`; use the project venv), a new temporary directory as cwd and a minimal environment, with a 10 s timeout. There is no sandbox:
no network use is intended, but nothing blocks it, and the code could touch any file the user can. Run it on a machine and account where
that is acceptable.

## How long a full run takes

914 items, decode about 100 tok/s, one request at a time; the time is almost all decoding (prompts are short). Per arm:

| mean output tokens per item | decode time per arm | both arms |
| --- | --- | --- |
| 1,000 | 9,140 s = 2.5 h | 5.1 h |
| 2,000 | 18,280 s = 5.1 h | 10.2 h |
| 3,000 | 27,420 s = 7.6 h | 15.2 h |
| 4,096 (every item hits the cap) | 37,440 s = 10.4 h | 20.8 h |

Add a few minutes per arm for server start-up and for scoring (HumanEval: at most 164 x 10 s, normally about 1 to 3 minutes). With thinking on,
math items tend to use far more tokens than multiple-choice items, so expect the mean between 1,500 and 3,000 until measured; a `--limit 5`
smoke run per set gives the real mean in a few minutes. Use `--sets` to split the work over several sessions; the run is resumable.

## Limits

- A thinking reply that does not finish within `--max-tokens` has no answer and counts wrong in both arms. The `cut` columns of `compare.py`
  show how often this happens; a high rate lowers the sensitivity of the gate. Raise `--max-tokens` for both arms if it is high.
- Greedy decoding on a lossy engine is not bit-reproducible run to run; the paired test treats that as noise, which is the intent. Run the
  reference arm twice if its own noise is in doubt.
- With 914 items the CI half-width is about 1.5 pp when 5% of the items flip between arms; a margin below 2 pp needs more items.
- MATH equivalence is normalized strings plus a numeric check; unusual formats (sets, intervals written differently) can score wrong in both arms alike.
