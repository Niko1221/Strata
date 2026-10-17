## Summary

**Disk restore made the first token 11.6x faster in the 128K example: 22.611 s for full prefill versus 1.956 s for RESTORE plus generation.** This includes restore time, uses warm OS page cache, and is one measured example. Initial SAVE/prefill is excluded.

![Disk restore latency and speedup](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/refs/heads/bench/cache-latency-1k-8k/docs/measurements/cache-latency-20261008/disk-restore/overview.png)

Compare main (`d5ea713`) with #1489 (`b299af0`) using the public Strata server. Main's session SAVE failed at all 11 sizes in this native non-MTP configuration; #1489 saved/restored successfully. As a secondary comparison, live prefix reuse reached the first token in **0.525 s versus 22.611 s for full prefill: about 43x faster** in the same 128K example. Live reuse avoids the file restore and already works in main; the 43x result is not a disk-restore speedup. Both 128K comparisons are single measured examples. No engine changes are included in this benchmark PR.

## Completed 1K-8K follow-up

![Completed live-prefix TTFT campaign](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/refs/heads/bench/cache-latency-1k-8k/docs/measurements/cache-latency-20261008/full/ttft.png)

![Measured latency percentiles](https://raw.githubusercontent.com/CC-David-CC/Strata-a5500/refs/heads/bench/cache-latency-1k-8k/docs/measurements/cache-latency-20261008/full/percentiles.png)

[Full report, completion chart, percentiles and raw observations](https://github.com/CC-David-CC/Strata-a5500/tree/bench/cache-latency-1k-8k/docs/measurements/cache-latency-20261008/full).

All 3,200 matched cross-build outputs were identical. Cache-disabled versus pinned outputs matched 958/1,600 pairs in each build; this is not a quality-equivalence claim. The 7.2x 8K gain is from existing live reuse, not an additional #1489 speedup.

## Evidence and limits

- 1K-8K: three observations per successful cell. 32K/64K/128K: one each. 135 timed generations succeeded, all with 128 output tokens. Main's 11 SAVE failures are reported as blocked disk cases.
- RTX PRO 6000 Blackwell; unchanged ISTA IQ2_XS; FP16 KV; prefill 8192; temperature 0; reasoning none; MTP off; serial local Responses API. Context capacity: 32K short / 256K long.
- Real filesystem restore with warm OS page cache, not cold SSD reads. No restart, automatic parking, durable-history or quality-equivalence claim. Both CUDA builds and six statistics tests passed.
- The separate 1K-8K live-reuse campaign is now complete: **6,400 measured requests, zero errors, 200 observations per cell**. At 8K, TTFT p50 was **1.574 s uncached versus 0.219 s pinned** in both builds; cached p95 **0.220 s**, p99 **0.223 s**. These percentiles apply only to live reuse, not the smaller disk-restore sample. p99 remains exploratory with 200 observations per cell.

[Full tables, completion charts, raw outputs and reproduction instructions](https://github.com/CC-David-CC/Strata-a5500/tree/bench/cache-latency-1k-8k/docs/measurements/cache-latency-20261008/disk-restore).

## Simple use

Keep the reference document unchanged, mark it with `strata_prefix: {"messages": 1}`, and cap the next response at `max_output_tokens: 128`. Enable `--slot-save-path ./sessions`, then:

```python
import requests
url = "http://127.0.0.1:8080/slots/0"
requests.post(url + "?action=save", json={"filename": "reference.session"}).raise_for_status()
requests.post(url + "?action=restore", json={"filename": "reference.session"}).raise_for_status()
# Send the next /v1/responses request against the restored prefix.
```

Add your authentication headers when configured.
