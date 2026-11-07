| case | runs | input tokens | cached | output tokens | prompt tok/s (median, range) | decode tok/s (median, range) | TTFT s | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `short` | 3 | 77 | 0 | 512 | 65.1 ([60.7, 73.4]) | 58.0 ([56.8, 68.0]) | 1.21 | completed,truncated |
| `prompt-4k` | 3 | 4093 | 0 | 512 | 770.0 ([768.6, 771.4]) | 57.9 ([56.3, 66.5]) | 5.35 | truncated |
| `prompt-16k` | 3 | 16387 | 0 | 512 | 1299.5 ([1295.6, 1301.5]) | 58.8 ([52.9, 61.1]) | 12.65 | truncated |

Protocol `strata-qwen38-thinking-v1-draft.1` (corpus sha256 `e75eff8b9feea04f745520b392c4c17d97aefc1ba185e222aa6740efa8f68407`, protocol git commit `9780a8c27d84f408240ed4b07062ab7275f02f09`), profile `qwen3.8-flash-next-thinking`, concurrency 1, 512-token output cap, **streaming**; the suite takes decode from the engine's own `timings.predicted_per_second` and prompt from `usage.prompt_tokens` over the stream.

All three rounds of both long-prompt cases hit the 512-token output cap (`finish_reason: length`) because the thinking model writes a long chain of thought first: that is why 7 of the 9 performance runs report `truncated`, and why the decode column here is a shorter burst than a 512-token answer suggests. The sanity pool ran separately and all 12 completed.

Suite quality gate: **12/12 passed** - instruction 4/4, retrieval 4/4, tool calls 4/4. Model files were re-hashed by the suite itself (shard 1 sha256 `4c1eb2ceb4915e11...`). Concurrency stages were not selected (recorded in the run's `missing` map, together with the draft protocol's unpinned model-card revision).
