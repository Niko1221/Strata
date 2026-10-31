# Reproduction notes

The commands below describe how to repeat the saved workload. They are not a
claim of a second benchmark run. Use the hashes and exact inputs in provenance.json
to check assets; model weights and built engines are not included in this report.

## Sources and builds

Clone `https://github.com/dhsgud/Strata.git` and make separate checkouts of the
two evidence commits linked in README.md. Build the engine from each checkout.
For the integrated configuration, run the Python server from the 0.1.54 checkout
and point its config at the 0.1.52 executable.

The measured build options were:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=OFF \
  -DSTRATA_BUILD_DSV41=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp
cmake --build build --target strata file_expert_source_test -j2
```

The supplied ggml directory must be llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. The historical build used an extracted
pinned tree, so CMake's printed `ggml commit` inherited the enclosing Strata git
revision; that diagnostic is not the llama.cpp source pin. Binary hashes identify
the measured artifacts; this report does not promise reproducible binary hashes
from a new build directory.

## Model assets

Download the two GGUF shards at the revision in provenance.json and verify their
hashes. With the 0.1.52 evidence checkout and its Python dependencies installed:

```sh
python tools/iq_pack.py --gguf /path/to/model/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --out /path/to/model/pack-contiguous --compat-bf16 --experts-bin
python tools/mtp_fetch.py fetch --out /path/to/model/mtp
python tools/mtp_pack.py --src /path/to/model/mtp --experts q2_0 \
  --out /path/to/model/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py --gguf /path/to/model/mtp/mtp-q2_0.gguf \
  --out /path/to/model/mtp/rt
```

Check that MTP fetch resolves the revision recorded in provenance.json. The
original tokenizer was under `pack/tokenizer`; use the matching tokenizer
generated from the same GGUF and compare its recorded file hashes. Point the
config's `tokenizer` field at that directory. The exact rendered template is
also included as chat_template.jinja. Put the included numeric expert profile
at the config's `--expert-profile` path. Do not substitute a different profile
when comparing these configurations.

## Server and requests

Copy the selected sanitized config to a private file. Replace all `<...>` paths
and `REPLACE_WITH_YOUR_API_KEY`, including `exe`, `cwd`, `lib_dirs` and `log`.
For baseline and baseline-fresh, use the baseline config and old server.
For candidate use the new server and new engine. For integrated use the new
server and old engine. In the appropriate server checkout:

```sh
python serve/server.py --engine strata --config /path/to/private-config.json --port 8080 --gpu 0
```

From another terminal, run the report's runner. `--source` is the 0.2.8 checkout
for benchmark helpers, even when the server under test is 0.2.7. Use a new output
directory for every arm:

```sh
python /path/to/report/run_checks.py \
  --source /path/to/source-0154 --config /path/to/private-config.json \
  --url http://127.0.0.1:8080 --out /path/to/new-results
```

The original order was baseline, candidate, baseline-fresh, integrated. The first
baseline process had previous use, so its exact prior state cannot be recreated.
The other arms began after a server/engine restart, without dropping the OS page
cache. Each runner invocation waits for readiness, then performs 12 sanity cases
before the three coding requests. Preserve both long requests and the sanity
outputs. Model loading is outside request timings.

For a new causal comparison, additionally isolate one change, repeat and swap
run order, control background activity, record memory/clock/I/O behavior, and
retain every run. Those controls were not part of this historical dataset.

## Saved validation

On the 0.1.54 experimental source, the saved logs record:

- `python -m unittest discover -s serve -p 'test_*.py'`: 439 tests passed.
- `ctest --test-dir build --output-on-failure -R '^(auto_cache_test|ple_residency_test|nv_defaults_test|arch_defaults_test|window_limits_test|conversation_cache_test)$'`:
  six tests passed (build these targets first).
- `build/file_expert_source_test`, and the same executable with
  `STRATA_DIRECT_COMPLETION=1 STRATA_COLD_PERSISTENT=1 STRATA_COLD_CPU=1 STRATA_DIRECT_POLL_US=2000`
  and `--rotation-gpu`: passed on Linux/CUDA.
- `c++ -std=c++17 -O2 -pthread -I include tools/test_persistent_job.cpp -o build/test_persistent_job`,
  then `build/test_persistent_job`: passed.
- The equivalent command for `tools/test_fixed_io_ring.cpp`, followed by that
  executable: passed. This tests an experimental path disabled in the timed runs.

These logs belong to the tested source snapshots, not to fresh builds made while
preparing the contribution. Other backends and operating systems remain untested.
