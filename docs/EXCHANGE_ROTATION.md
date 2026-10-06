# Resident expert exchange buffer rotation

When the resident RAM complement exchanges an expert with the GPU cache, the
evicted expert lands in a temporary host buffer. The existing path then copies
those bytes into the promoted expert's old RAM slot. Rotation removes that last
host copy: the temporary buffer becomes resident, and the old RAM slot becomes
the next temporary buffer.

## Enable

Rotation is **off by default**. Set `STRATA_EXCHANGE_ROTATE=1` before starting the
engine with your existing arguments:

```bash
export STRATA_EXCHANGE_ROTATE=1
```

```powershell
$env:STRATA_EXCHANGE_ROTATE = '1'
```

Only the exact value `1` opts in. Set it to `0` or remove it to restore the copy
path. This setting does not select resident RAM mode or change any preset.
It applies only when all layers have equal-size expert blocks, the resident
complement is fully pinned and GPU-mapped, and the exchange buffers can also be
pinned and mapped. Other layouts keep the copy path and log why rotation could
not be enabled. Configurations without a resident complement do not use it.

The startup diagnostic confirms activation:

```text
FileExpertSource: exchange buffer rotation enabled: ...; no host commit memcpy
```

Serving diagnostics report the cumulative rotated block count and avoided
`memcpy` payload bytes. This counts the copied payload once, not read plus write
traffic, and does not include the GPU transfers, which still occur.

## Ownership and synchronization

The original allocations own their memory until `FileExpertSource::close()`.
Rotation updates slot ownership and keeps each host pointer paired with its GPU
alias. Atomic slot IDs let background residency queries observe those immutable
pairs. Compute readers and GPU transfers must still finish before the existing
serialized commit; rotation adds no overlapping compute or new synchronization
policy. Reserve the maximum exchange capacity before rotation starts: growing a
live exchange arena is rejected because it can now hold resident experts.

## Tests

The ownership test can run without CUDA:

```bash
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread -Iinclude tests/core/exchange_storage_test.cpp -o /tmp/exchange_storage_test
/tmp/exchange_storage_test
```

With an existing CUDA build configured:

```bash
cmake --build build --target strata exchange_storage_test file_expert_source_test
ctest --test-dir build -R '^(exchange_storage_test|file_expert_source_test)$' --output-on-failure
./build/file_expert_source_test --rotation-gpu
compute-sanitizer --tool memcheck --error-exitcode 71 ./build/file_expert_source_test --rotation-gpu
```

The CPU test checks byte preservation, pointer/alias ownership, guards, invalid
operations and concurrent residency queries across 12,304 exchanges. The CUDA
fixture performs 64 exchanges in each of three modes: copy with pinned RAM,
rotation with pinned RAM, and requested rotation with pageable RAM (fallback).
It checks exact bytes, GPU alias reads, file fallback, capacity, and close/reopen.
The GPU fixture is explicit; the ordinary CTest invocation does not run it.

On 2026-10-04, the clean patch on upstream
`6f32ec070f23ced9f50e704d854d775da52591ab` built on Linux with GCC 15.2 and CUDA
13.3 for an RTX 4090. Both CTest tests, the ASan/UBSan ownership test, and the
CUDA fixture passed. Compute Sanitizer reported **0 errors**. AMD and Windows
GPU execution were not tested for this patch.

## One native token-identical A/B

Measured on 2026-10-04 using the clean rotation build at
`18a30ad775bce86e99a54e6dc552ccb72c66d16e`, on upstream
`6f32ec070f23ced9f50e704d854d775da52591ab`. The model is the standard
GSQ-RCO **Q2_0**, with its matching IQ4_NL PLE shard. Both shards were verified
against the publisher's SHA-256 hashes at the pinned revision in the receipt.
The native pack uses the existing `tools/iq_pack.py`; no compatibility conversion
or Q8 PLE reader extension is needed.

Both arms used the same Release binary (GCC 13.3, CUDA 13.2), RTX PRO 6000
Blackwell 96GB, Ryzen 9 7950X and 128GB installed RAM. A fresh engine per arm
generated exactly 1,024 tokens after the same 1,024-token counting prompt.
The context allocation was 40,960, with 12,000 GPU expert slots, a 16.19 GiB
pinned/mapped resident complement, FP16 KV, greedy target-only decoding and
zero offered drafts, reused prompt tokens or file-tier blob reads during
generation. Only `STRATA_EXCHANGE_ROTATE` changed. Transfer completion is
awaited before admission.

The GPU cache was deliberately capped at 12,000 slots (15.45 GiB) to exercise
resident RAM exchanges. Q2 can fit entirely on this 96GB GPU; this is a storage
path comparison, not the fastest full-GPU configuration. It does not change
the project's presets or enable resident mode for existing users.

| Rotation | Decode tokens/s | Resident exchanges | Avoided host-copy payload | Output IDs |
|---|---:|---:|---:|---|
| `0` | 104.90 | 3,802 | 0 bytes | 1,024, identical |
| `1` | 113.44 | 3,802 | 5,255,884,800 bytes | 1,024, identical |

Decode rates exclude startup and prefill. The observed throughput change was
+8.1% in this one pair; repeated timing or a general speedup is not established.
The out-of-vocabulary EOS sentinel `2147483647` forces the exact output length,
so this is not a model-quality evaluation. Speculative-path token parity,
AMD and Windows GPU execution remain untested.

The same Blackwell binary passed both CTest checks and the three-mode CUDA
fixture; Compute Sanitizer reported 0 errors. The
[receipt](../bench/results/exchange-rotation-ab.json) contains source, binary,
model, pack and profile hashes, arguments, timings and diagnostics. The
[raw IDs](../bench/results/exchange-rotation-token-ids.json) retain the input
and both outputs. To check the token comparison without a model:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('bench/results')
r = json.loads((p / 'exchange-rotation-ab.json').read_text())
raw = (p / r['token_ids_file']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == r['token_ids_file_sha256']
d = json.loads(raw)
assert len(d['input']) == len(d['off']) == len(d['on']) == 1024
assert d['off'] == d['on']
for name, arm in zip(('off', 'on'), r['arms']):
    sha = hashlib.sha256(json.dumps(d[name], separators=(',', ':')).encode()).hexdigest()
    assert sha == arm['output_token_ids_sha256']
print('1,024 identical output token IDs')
PY
```
