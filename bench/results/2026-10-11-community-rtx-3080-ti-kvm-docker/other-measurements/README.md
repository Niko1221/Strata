# Other measurements

Per-run JSON lines from an earlier script, taken the same day on the same machine and engine. Request settings are
the same as the sweep's (thinking off unless the file name says `reason`, greedy, seed 1, streamed, a fresh nonce at
the start of every prompt), but prompt sizes and output caps are its own. File names give the model
(`iq3_s`, `ud-iq4_xs`, `swift-iq3_s`), then the context when it is not 131,072 (`204k`, `262k`, `524k`), then the
test. `p2` means `"parallel": 2`; `bmtp` adds `--batch-mtp`; `pool19` is `--pool-workers 19`.

Fields: `prompt_tokens`, `cache_n` (reused tokens), `prompt_n` (tokens read), `prefill_tps` and `decode_tps`
(tokens/s, the engine's clock), `predicted_n` (generated tokens), `ttft_s` and `wall_s` (seconds, client clock),
`finish`. In `concurrent` files `streams` holds one such record per simultaneous request, `tokens_per_wall_s` is all
generated tokens over the time until both finished, and `embed` is the latency in seconds of an unrelated embedding
model on the machine's other GPU while the requests ran. In `needle` files `needle_found` says whether a six-digit
code planted a tenth of the way into the prompt came back. In `reason` files each line is one question with a known
numeric answer: `correct`, `completion_tokens` (thinking included), `think_chars` and `wall_s`.
