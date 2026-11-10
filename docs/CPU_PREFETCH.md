# Idle CPU expert prefetch (experimental)

This opt-in path warms CPU caches with predicted next-layer experts while the GPU
runs. It uses the existing expert-pool workers, without an extra thread pool. It
reads stable RAM arena bytes only. It does not change routing, copy experts into
VRAM, or replace the expert arithmetic. It is a decode experiment; it does not
accelerate the GPU prompt-read path.

Build with `-DSTRATA_CPU_PREFETCH=ON` and native experts enabled. Both the build
switch and the runtime setting are off by default. With the build switch off,
the original pool and generate code is compiled without the prefetch hooks.
The switch is wired into the root CPU test and CUDA/HIP engine builds. HIP is
unvalidated; the standalone SYCL and Metal builds do not implement this switch.
For an eligible engine, set `STRATA_CPU_PREFETCH=4` to warm up to four experts;
`1`, `2` and `3` warm fewer, and `0` disables it. Values are clamped to 0..4.
Four is the retained local setting, not a recommendation for every machine.

The path requires x86 AVX2, at least two pool workers, native experts in a private
RAM arena, and BF16 router weights. File mappings and mutable complement buffers
are excluded. Serialized windows and batch groups are required: pipelined windows
and multiple batch groups are rejected when the runtime setting is enabled.
The routers are copied from their owning GPU once at startup. A Q4 prediction
uses an owned snapshot of the previous layer's activations; predictions can miss.
GPU-, peer- and remote-resident experts are excluded from the warm-up candidates.

On the existing two-node NUMA split, workers claim tickets only from their own
node's queue. Each node reads its gate, up and down half, matching the native row
split. This does not allocate or migrate arena pages: it retains the upstream
arena-placement limitation. Both nodes need an actual pool worker. Single-node
runs divide each matrix's byte range among the workers.

Foreground dispatch cancels warming and joins every claimed job before using the
pool again. Warm-up never crosses a layer-split stage's last layer. Request-end
reporting and object destruction also stop the work. The
`strata cpu-prefetch` line reports prediction selections, routed hits and cache
lines read. It is not a measured cache-hit rate or a throughput guarantee.

CPU regression tests (no model or GPU required):

```sh
cmake -S . -B build-prefetch -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_CPU_PREFETCH=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-prefetch --target pool_idle_test idle_prefetch_test numa_idle_test
ctest --test-dir build-prefetch --output-on-failure -R '^(pool_idle_test|idle_prefetch_test|numa_idle_test|numa_idle_prediction)$'
```

The synthetic NUMA tests run on Linux; they exercise queue and range behavior,
not physical page placement or remote-memory bandwidth. HIP execution and
end-to-end output parity require separate validation.

The CPU warming idea is adapted from Basalt commit
`ca7d07418cdf6ee08462c560102a7798c94886cc`; see [its MIT license](../BASALT-LICENSE.txt).
This change adds multiple predictions, mixed expert layouts, owned activation
snapshots, cancellation and NUMA-local work partitioning. DAG scheduling and
cross-window asynchronous expert refills are separate experiments.
