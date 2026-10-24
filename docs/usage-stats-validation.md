# Lifetime usage statistics: validation

This change extends the counters and range views from [PR #2370](https://github.com/Niko1221/Strata/pull/2370).
The original commits are retained. The additional work covers persistence, writer ownership, visible failures,
shared sequential profiles, undated imports and a permanent all-time output headline. The Usage card is
appended at the bottom of Monitor; the existing speed, hardware and cache layout remains first.

## 0.2.0-beta revalidation

See [the beta report](usage-v020/README.md) for the current revision, results and limits. The older measurements below remain historical evidence.

## Repeat the tests

No model or GPU is required:

```sh
python -m unittest serve.test_server serve.test_usage serve.test_usage_durability serve.test_monitor serve.test_prometheus serve.test_runconfig -q
node --check serve/web/app.js
```

The persistence suite checks abrupt subprocess exit after two quick requests, 64 concurrent writes, a second
process attempting to open the same store, failed atomic replacement and retry, corrupt JSON preservation,
fractional timing reload and undated imports. Serving tests cover tool-call output, partial disconnect,
multiple request threads and the disabled metrics response.

Results on 2026-10-17, after syncing with upstream main `477ac13c`:

| Host | Result | Time |
| --- | --- | --- |
| macOS | 389 tests passed | 116.549 s |
| Windows 11 | 389 tests run: 386 passed, 3 Unix-signal tests skipped | 116.853 s |

`node --check serve/web/app.js` and `git diff --check` also passed. The Windows run used an isolated copy
of this PR, without stopping the production service. Its first package omitted the upstream
`tools/test_strata_tokenizer.py` helper; after supplying that file, the complete suite above passed.


Additional checks on a deployed Windows integration used Chat Completions and Responses with streaming on/off,
plus cancellation and resumption through a separately installed scheduler patch. The saved counters matched
session totals. A native-model smoke request increased the API, disk and visible total by the same four tokens.
These deployed checks used an older server baseline; the tests above exercise the PR source synced to main.

## Cost of persistence

Windows, Ryzen 9 5900X, 128 GB RAM, RTX 3090. With 744 hourly and 31 daily buckets (~150 KB JSON), 100
settled-request writes took **10.78 ms median, 12.30 ms p95, 18.11 ms max**. This measures statistics persistence,
not model throughput. The write happens once at request settlement, not once per token.

Reproduce the same write workload, using a temporary directory on the disk you want to measure:

```python
import statistics
import tempfile
import time
from pathlib import Path
from serve.usage import COUNTERS, Usage

with tempfile.TemporaryDirectory() as folder:
    store = Usage(Path(folder) / "usage.json")
    now = time.time()
    for segment, count, step in (("hour", 744, 3600), ("day", 31, 86400)):
        store.buckets[segment] = [
            dict(t=int(now // step) * step - i * step,
                 **{key: 10000 for key in COUNTERS})
            for i in range(count - 1, -1, -1)
        ]
    elapsed = []
    try:
        for _ in range(100):
            start = time.perf_counter()
            store.record(output_tokens=1000)
            elapsed.append((time.perf_counter() - start) * 1000)
        print({"median_ms": statistics.median(elapsed),
               "p95_ms": sorted(elapsed)[94], "max_ms": max(elapsed)})
    finally:
        store.close()
```

## UI and limits

[Original Monitor sections first](images/usage-stats/desktop.png) · [Usage appended at the bottom](images/usage-stats/usage-bottom.png).
The screenshots use a local preview with synthetic usage counters and a labelled undated import; the hardware
readings come from the preview host and are not an inference benchmark. Both screenshots were taken at 1280×720
after syncing with main and moving the card to the bottom. There was no horizontal overflow.
The unchanged card styling was also checked earlier at 390×844. Disabled metrics kept the usage card hidden.

- Only counters/timestamps are stored; private request text is not persisted by this feature.
- A request killed before settlement is not included. Forced process exit after settlement was tested;
  physical power loss was not.
- Historical imports cannot reconstruct dates, input counts or speeds absent from retained logs.
- Native engine sources are unchanged. No CUDA/HIP/SYCL rebuild or inference speedup is claimed.
