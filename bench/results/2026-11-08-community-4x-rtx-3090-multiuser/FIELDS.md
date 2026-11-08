# Fields and units of data/

All times in seconds, rates in tokens per second, timestamps ISO 8601 (local time, CET) unless said otherwise.
Labels: `run-<arm>[-r2]-<kind>`, arm S / C / PT, `-r2` = the second fresh server start of that arm.

## sweep.jsonl (`scripts/bench_concurrency.py`, one line per label and user count)

- `users`: concurrent clients; `rounds`: requests per client, sent one after another; `requests`, `ok`, `errors`, `err_sample`.
- `max_tok`: output cap per request (512, every request ran to the cap: greedy, thinking off); `temperature` 0.
- `prompt_kind`: nominal prompt size (100 = short, 1000 = long); `prompt_tok_avg`: the actual mean prompt tokens (146-148 / 1213).
- `wall_s`: first request sent to last token; `out_tok_total`; `agg_out_tps` = out_tok_total / wall_s; `agg_total_tps`
  includes prompt tokens; `agg_out_tps_steady` / `steady_s`: output rate over the window in which every client was decoding.
- `ts`: start time; `max_tok_min`: 0 (no minimum); `agg_out_tps_steady` is null when no window had every client decoding (S, 3 slots).
- `ttft_p50` / `ttft_p95` / `ttft_max`: client-side time to the first streamed token; `dec_tps_p50` / `_p5` / `_p95`:
  per-request decode rate after the first token; `e2e_p50` / `e2e_p95`: request latency.
- `cached_tok`: prompt tokens the server reported as reused; `replies_sha256`: SHA-256 over all replies of the label.

## sweep.requests.jsonl, ref.requests.jsonl (one line per label and user count)

- `fields` names the columns of each row in `reqs`: `uid` (the prompt), `user`, `max_tok`, `gen` (generated tokens),
  `prompt` (prompt tokens), `t0` / `t_first` / `t_end` (seconds from the label's start), `sha256` (of the reply), `err`.
  `scripts/compare_replies.py` compares the `sha256` per `uid` for the greedy identity check (`<arm>-ident.txt`);
  `ref.requests.jsonl` holds reference short replies from an earlier window on the same machine: `ref-PT-short` (PT, the reference used here) and `ref-I-short` (an earlier integration build, unused).

## multiturn.jsonl (`scripts/bench_multiturn.py --users 16 --turns 3 --doc 8000`)

- `wall_s`; `agg_out_tps`; `turn1` / `turn2` / `turn3`: `prompt` and `cached` tokens (per-request means), `ttft_p50`, `ttft_p95`, `dec_p50`;
  `follow_ttft_p50`: TTFT p50 over turns 2-3; `ts`, `users`, `turns`, `temperature`, `prompt_tok` / `out_tok` (totals), `replies` / `reply_fields` (per-request reply hashes).

## join-*.json (`tools/piped_read_join_bench.py --scenarios cold32k`)

- `cold32k.joiner_ttft_s`, `joiner_prompt_tokens`, `joiner_cached_tokens`; `worst_max_gap_s`: the longest pause between two
  tokens of the 4 running streams while the 32K joiner was read; `streams[]`: per stream tokens, `max_gap_s`, `p99_gap_s`, `stalled_s`; `mean_stalled_s`, `stream_tokens_during_join`, `aggregate_tok_s` (all streams during the join).

## vram-*.csv

One line per 5 s sample: `unix time,` then per card `index,free MiB,GPU utilization %`, cards separated by `;`.

## status-*.json

The server's status JSON at the start of the arm, before its first request (engine version, slots, context, VRAM/RAM).

## *-short.txt, *-long.txt, *-mt.txt, *-ident.txt

The harness's console tables of the same runs.
