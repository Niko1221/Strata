# Strata cache latency: 1K–128K prefixes

This benchmark separates the benefit of prefix reuse from differences between
Strata main and the Responses/disk-cache integration in
[#1489](https://github.com/Niko1221/Strata/pull/1489). It uses Strata's public
Python server and native engine directly. There is no downstream profile wrapper.

Results: [completed 1K–8K percentiles, 6,400 measured requests](measurements/cache-latency-20261008/full/),
[disk restore examples](measurements/cache-latency-20261008/disk-restore/),
and [32K–128K single examples](measurements/cache-latency-20261008/long-examples/).

The first pass is deliberately small: three measured requests at each prefix
length, after three recorded warm-ups. Its charts show preliminary medians only.
An additional pass measures one request per cell at 32K, 64K and 128K after one
warm-up, using a 256K context capacity for both builds. These are single examples,
not latency percentiles. It keeps all other engine and sampling settings identical.

The follow-up campaign completed 200 measured requests per cell in ten
blocks of 20: 6,400 requests across two builds, two modes and eight lengths,
with zero request errors. This repeated campaign covers cache-disabled and
pinned live reuse only; disk restore retains the smaller sample counts above.
Warm-ups and requests used to switch conversations are kept in the raw logs
but excluded from the measured population. Build/mode order reverses on alternate
blocks; prefix-length order is deterministically shuffled within each block.

## Fixed comparison

| Item | Setting |
|---|---|
| Main | `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` |
| Integration #1489 | `b299af0e8fc9f7ee1792c63a155707cc166ae7e4` |
| Model | ISTA-DASLab Qwen3.8-Flash-Next-GSQ-RCO IQ2_XS, unchanged public files |
| Hardware | One RTX PRO 6000 Blackwell, 96 GB VRAM; llm-60 |
| Backend | CUDA 13.2, architecture 120; same build options and ggml source |
| Context / KV / prefill chunk | 32,768 for short tests, 262,144 for long examples / FP16 / 8,192 |
| Speculation | Native IQ `--spec 2`; no MTP weights; suffix draft disabled |
| Sampling | Temperature 0, seed 42, reasoning effort none |
| Output | Cap 128 tokens; actual output length recorded for every request |
| Prefix lengths | 1,024 through 8,192 in increments of 1,024, plus single examples at 32,768 / 65,536 / 131,072 |
| API / load | Local HTTP SSE `/v1/responses`, one request at a time, `store: false` |

Actual prompt length is the pinned prefix plus the remaining document fragment,
changing question and template markers. Exact input and reused token counts are
recorded. This is a synthetic Python reference-document workload, not a model
quality evaluation. Both builds receive the same deterministic request corpus.

## Cache modes

- **Cache disabled:** `--prompt-cache 0`; includes full prefill on every request.
- **Pinned prefix:** `--prompt-cache 6`, `strata_prefix: {"tokens": N}`; repeated
  independent questions about one unchanged document, without conversation parking.
- **Disk restore + request:** warm the document, save a session through
  `POST /slots/0?action=save`, replace the live conversation with an unrelated
  request, then `POST /slots/0?action=restore` before asking the next question.
  Charted latency includes the entire restore HTTP call plus the generation
  request. One-time SAVE and initial prefill are reported separately. This uses
  the explicit session API on both builds, not automatic disk-only parking.
- **Switch / RAM:** alternate with an unrelated document, with a 4 GiB RAM
  conversation-cache budget. This is a separate workload from same-prefix requests.
- **Switch / disk:** the integration's disk-only conversation cache, with a 4 GiB
  budget and the same alternating workload. Main lacks this mode; it is marked
  unsupported rather than assigned a synthetic timing.

The initial branch covers the first three modes. Disk restore has three measured
requests per short cell and one per long cell, after one warm-up and one SAVE per
cell. A failed SAVE blocks restore and generation: those are reported as blocked
cases, not successful latency observations. Automatic switching is a separate
follow-up. Cached-token counts determine observed hits; a mode's name is not proof
of reuse. Main's non-MTP snapshot behavior may produce misses, which remain in
the reported latency population.

## What the charts measure

TTFT runs from HTTP request dispatch to the first nonempty **text** delta, not the
early `response.created` event. Completion time ends when the SSE response closes.
The completion chart includes only replies that actually generated 128 tokens;
the summary separately reports early stops and errors. No concurrent queue time
is being tested. Model loading and initial prefix creation are excluded from
steady-state percentiles; warm-up timings remain available separately.

Report p50, p80, p90, p95 and p99 using the linear empirical quantile estimator
(Hyndman–Fan type 7). At 200 observations, p99 depends on roughly two upper-tail
observations; it is exploratory, not an SLA estimate. Pilot charts with fewer than
100 observations per cell deliberately withhold tail curves. Correlated requests
on one machine do not represent a production workload distribution.

The OS file cache is left warm. A disk-cache result therefore measures the normal
filesystem path and can include page-cache hits; it is not a cold-storage or
post-reboot guarantee. Other GPU workloads must remain stopped during the campaign.

## Reproduce

Build the two pinned sources with matching compiler options. Supply a normal
Strata engine config for each, using the same model, tokenizer and engine settings,
but omit all `--prompt-cache` and `--conversation-cache-*` flags; the probe supplies
those. Keep outputs in a disposable benchmark directory.

```bash
python tools/cache_latency.py --source /path/to/source --config base.json \
  --build-label main --source-commit d5ea7133741e67743c0e886bb426c0ce8d69cf6c \
  --mode pinned --samples 20 --output results/main-pinned
python tools/cache_latency_report.py results --output charts
python tools/test_cache_latency_report.py
```

For a matrix, `cache_latency_campaign.py --config campaign.json --output results`
accepts a JSON object with `targets` (comma-separated token counts) and `builds`.
Each build contains `label`, `commit`, `source`, `config`, and a `modes` array.
Run under a detached supervisor so a client-network interruption does not stop it.
The campaign preserves completed-job receipts and reclaims only its generated KV
snapshots. No model weights belong in this branch or its result files.

Example request shape (reference text abbreviated):

```python
request = {
    "model": "cache-benchmark",
    "input": "<unchanged reference document>\nQuestion 000001: Write a Python queue with tests.",
    "strata_prefix": {"tokens": 4096},
    "max_output_tokens": 128,
    "temperature": 0,
    "seed": 42,
    "reasoning": {"effort": "none"},
    "store": False,
    "stream": True,
}
```

Only use `tokens: 4096` when that many initial tokens are unchanged. Applications
can instead mark a document using `strata_prefix: {"messages": 1}` or the
document-character boundary documented in [DETAILS.md](DETAILS.md).
