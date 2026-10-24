# Evaluation framework

Three metrics per test query, measured over the real HTTP API (`/v1/chat/completions`):

- **format** - the reply is a valid OpenAI `chat.completion` (`choices`, `message`, `usage`, a known
  `finish_reason`): the shape every client relies on.
- **relevance** - the reply contains the keywords the query expects (`expect_all`: every one,
  `expect_any`: at least one; case-insensitive).
- **time** - wall seconds per request, and tokens/s: the engine's own `timings` on a real engine,
  `completion_tokens / wall` on the mock.

## Running it

Against the mock engine - no GPU, no model, no downloads (the repo's test convention):

```console
python evals/run_eval.py
```

Against a running server (the real model, started by `START-HERE.bat` / `setup.sh`):

```console
python evals/run_eval.py --base http://127.0.0.1:8095 --api-key KEY --repeats 3
```

`--repeats N` runs every query N times and keeps the median wall time. `--timeout S` is the per-request
limit (also the per-query time gate; a query can lower it with `"max_s"`). `--min-relevance F` is the
keyword coverage a query needs (default 1.0 = all keywords). `--trace` turns on the repo's opt-in tracing
(`serve/tracing.py`) for the run: the mock run exports every query through an in-process OTLP collector and
links each row to its server-side spans (the mock engine says no DONE line, so its traces carry the request
span with token counts and wall time; prefill and decode spans are the real engine's milliseconds); against
a real server each request carries a W3C `traceparent` and the row records the server's answer header, so
rows line up with the server's traces (the server must be started with tracing on). Exit code 0 when every
query passes all three metrics, 1 otherwise - usable as a CI gate. The report lands in
`evals/results/eval-<utc>.json` with one row per query (content preview, problems, timings).

The framework's own test (the harness end to end against the mock, same convention as the serve tests):

```console
python -m unittest evals.test_eval
```

## Queries

`evals/queries.jsonl` - one JSON object per line:

| key | meaning |
| --- | --- |
| `id` | the row's name (unique) |
| `prompt` | the user message |
| `expect_all` | keywords the reply must contain (all) |
| `expect_any` | keywords of which the reply must contain at least one |
| `mock_answer` | what the mock engine replays for this query (so the mock run checks the server's plumbing: request handling, the OpenAI shape, token counts, latency bookkeeping) |
| `max_tokens` | optional output cap (default 256) |
| `max_s` | optional per-query time limit (default: `--timeout`) |

Add a query by adding a line; keep `mock_answer` consistent with the keywords, so the mock run stays
green and a real-model run measures the model, not the plumbing.

## What each run checks

- **Mock run** (`python evals/run_eval.py`): the server itself - request handling, the OpenAI shape,
  usage counts, latency bookkeeping. Runs anywhere, in seconds, with no model.
- **Real run** (`--base` against your server): the model's answers against the same keywords, with the
  engine's own tok/s. A keyword the model misses is a real answer-quality finding; a format problem is
  an API regression.
