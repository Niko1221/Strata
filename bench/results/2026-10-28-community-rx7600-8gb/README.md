# Community benchmark: RX 7600 8 GB, Core i9-10900

Measured on 2026-10-28 by [ominiverdi](https://github.com/ominiverdi). This
tests Strata 0.1.53 on a headless Radeon RX 7600 8 GB (`gfx1102`) with the
original Qwen3.8-Flash-Next Q2_0 and the expert-pruned Coder IQ1_M, one GPU,
and a 32,768-token context limit.

In the repository's standard three-run suite, Q2_0 processed fresh 4K and 16K
prompts at median **215.9 and 237.9 tok/s**, and Coder IQ1_M at **176.9 and
174.4 tok/s**. Median short-chat decode was **27.2 tok/s for Q2_0** and
**19.8 tok/s for Coder**. Q2_0 found all six recall needles at actual prompt
lengths up to 22,646 tokens. These results establish that both models run on
this headless 8 GB card; they do not establish performance at larger contexts,
with vision, or under concurrent load.

## Hardware and software

- AMD Radeon RX 7600 8 GB, Navi 33, `gfx1102`, PCI ID `1002:7480` (ASUS
  subsystem `1043:05fb`). PCIe 4.0 x8 capability and negotiated link, 16.0 GT/s
  x8. The engine measured 6.8 GB/s host-to-device and selected `pcie_frac 0.19`.
- Intel Core i9-10900, 10 cores / 20 threads, AVX2 and no AVX-512. The engine
  selected nine pool workers plus its host thread.
- 64 GB installed RAM, 62.7 GiB usable; 48 GiB swap. Model files and packs were
  on a Lexar NM610 PRO 1 TB NVMe SSD.
- Debian GNU/Linux 13 (trixie), kernel `6.12.113+deb13-amd64`, in-tree
  `amdgpu` driver. The server was headless and no other GPU workload ran.
- Strata project and engine 0.1.53, commit `46c152a1`; llama.cpp `3cf0325`.
  Local HIP release build for `gfx1102`; see [BUILD.json](data/BUILD.json).
- User-local TheRock ROCm wheels `7.10.0a20251120`; the device probe reported
  runtime/driver `70253210`. CMake 4.4.3, Ninja 1.13.2, HIP Clang 22.0.0 and
  Python 3.13.5.
- No `gfx1102` tuning table exists for this build's hipBLASLt 100200, so dense
  prompt GEMMs used plain hipBLAS.

`strata-device --list-devices` reported `AMD Radeon RX 7600`, `gfx1102`,
wave32 and 7.984 GiB. Its arena allocation/over-allocation self-test passed.
The full output is in [strata-device.txt](data/strata-device.txt); additional
host facts are in [system-facts.txt](data/system-facts.txt).

The initial model build used:

```text
-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF
-DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON
-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON
-DCMAKE_HIP_ARCHITECTURES=gfx1102
```

Setup printed its expected warning that `gfx1102` builds but has not been
maintainer-validated on a real card.

## Models and configuration

The model files came from `main` in these repositories; setup did not record a
fixed Hub revision:

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
  - `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`
  - `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`
- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`
  - `IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`
  - `IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf`

The four local files, engine, profiles and prepared packs were SHA-256 hashed
after testing; byte sizes and hashes are in
[artifact-hashes.json](data/artifact-hashes.json). Hashes for configurations and
`BUILD.json` refer to the original runtime files; the attached copies have paths
sanitized. The Coder and original second shards are byte-identical, as expected
for their shared n-gram table. Vision was disabled, so no image encoder was
downloaded or used.

Both models used one request slot, a 32,768-token context, int8 KV, automatic
expert-cache sizing and prefill, and the default MTP vocabulary:

```text
--expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5 --mtp <draft-layer>
--max-context 32768 --kv int8
```

The complete sanitized configurations are
[strata-q2_0.json](data/strata-q2_0.json) and
[strata-coder-iq1_m.json](data/strata-coder-iq1_m.json).

| Model | Expert arena | GPU expert cache | Prefill cap | Engine-reported free VRAM after load |
| --- | ---: | ---: | ---: | ---: |
| Q2_0 | 32,400 MiB | 1,730 slots / 2,280 MiB | 3,584 tokens | 634 MiB |
| Coder IQ1_M | 23,981 MiB | 750 slots / 1,480 MiB | 1,280 tokens | 616 MiB |

Manual post-request snapshots, not sampled peaks, showed 7.41 GiB VRAM and
34 GiB used host RAM for Q2_0, and 7.48 GiB VRAM and 25 GiB used host RAM for
Coder. Swap allocation was 22 MiB and 220 MiB respectively, but swap I/O and
process attribution were not measured. No out-of-memory error occurred.

## Method and reproduction

Setup was run sequentially; the models and engines were never loaded at the
same time. The recorded choices can be reproduced with:

```bash
./setup.sh --backend hip --model Q2_0 --context 32768 --vision no \
  --parallel 1 --no-start --yes
./setup.sh --backend hip --family coder --model IQ1_M --context 32768 \
  --vision no --parallel 1 --no-start --yes
```

The repository's `tools/strata_bench.py` suite was then run with three
repetitions and `--long`:

```bash
python tools/strata_bench.py --config strata-q2_0.json --runs 3 --long
python tools/strata_bench.py --url http://127.0.0.1:8080 --runs 3 --long
```

The second command used the already-running Coder configuration attached above.
For Q2_0 the tool started a fresh engine. For Coder, two coding smoke requests
had already run in that engine session before the suite. Model loading is not
included in either measurement.

The suite disabled thinking, set temperature to zero, discarded one short chat
and one 1K warm-up, and then ran:

- three questions three times, with a 100-token cap, for short-chat decode;
- three fresh 4K and three fresh 16K word-list prompts, each with a 24-token cap,
  for prompt throughput;
- three five-turn agent sessions for conversation-cache reuse.

Every fresh prompt carried a unique tag and reported zero reused tokens. The
engine stayed loaded between a model's runs, so its expert cache and OS file
cache remained warm. Engine prompt/decode timings supply tok/s; the client
measured TTFT and total latency over loopback. The 24-token prompt-test outputs
are deliberately too short for a stable decode summary, so decode is reported
from the nine 100-token chat requests instead.

The requested 32K case was skipped by the tool because a 32,768-token context
did not leave room for prompt overhead and output. There were no failed or
cancelled benchmark requests.

## Results

Each value is the median **[minimum–maximum]**. Short chat has nine requests;
each fresh-prompt row has three.

### Q2_0

| Workload | Prompt tokens | Reused | Generated | Prompt tok/s | Decode tok/s | TTFT s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short chat | 46 [43–51] | 0 | 100 | not summarized | 27.2 [23.3–38.1] | 1.167 [1.032–1.286] | 4.725 [3.785–5.296] |
| Fresh 4K | 4,019 | 0 | 24 | 215.9 [215.9–225.5] | not summarized | 18.659 [17.865–18.659] | 19.418 [18.724–19.568] |
| Fresh 16K | 16,058 | 0 | 24 | 237.9 [235.3–238.3] | not summarized | 67.542 [67.437–68.283] | 68.254 [68.170–69.021] |

The agent sessions had a median 69.45% cache hit rate [69.25–69.47%], three
earlier-turn tokens re-read per session [1–74], median turn decode 29.6 tok/s
[13.5–34.4], and median turn TTFT 12.653 s [8.159–15.429].

Raw records: [q2_0-results.json](data/q2_0-results.json). Engine startup and
request lines: [q2_0-engine.txt](data/q2_0-engine.txt).

### Coder IQ1_M

| Workload | Prompt tokens | Reused | Generated | Prompt tok/s | Decode tok/s | TTFT s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short chat | 46 [43–51] | 0 | 100 | not summarized | 19.8 [13.2–29.8] | 1.486 [1.316–3.760] | 7.689 [4.796–8.978] |
| Fresh 4K | 4,019 | 0 | 24 | 176.9 [167.9–177.8] | not summarized | 22.775 [22.661–24.006] | 23.803 [23.620–25.155] |
| Fresh 16K | 16,058 | 0 | 24 | 174.4 [172.2–174.6] | not summarized | 92.145 [92.024–93.304] | 94.233 [93.241–94.361] |

The agent sessions had a median 69.34% cache hit rate [69.31–69.45%], three
earlier-turn tokens re-read per session [2–5], median turn decode 21.4 tok/s
[10.2–27.2], and median turn TTFT 19.217 s [11.795–21.630].

Raw records: [coder-iq1_m-results.json](data/coder-iq1_m-results.json). Engine
startup and request lines: [coder-engine.txt](data/coder-engine.txt).

## Correctness and limitations

### Long-context recall

The repository's unchanged `tools/needle_bench.py` was run against Q2_0:

```bash
python tools/needle_bench.py --url http://127.0.0.1:8080 \
  --lengths 16k,24k --depths 10,50,90 --out q2_0-needles.json
```

All six answers exactly matched their expected code words:

| Requested length | Actual prompt tokens | Depths | Result |
| --- | ---: | --- | ---: |
| 16K | 14,998–14,999 | 10%, 50%, 90% | 3/3 |
| 24K | 22,644–22,646 | 10%, 50%, 90% | 3/3 |

Raw results: [q2_0-needles.json](data/q2_0-needles.json). This checks only six
synthetic retrieval cases, not general long-context quality.

Coder IQ1_M also generated a Python interval-merging function that parsed and
passed five local behavioral cases, including empty, overlapping, touching and
nested intervals and an input-nonmutation check. That was a smoke check, not a
standardized coding benchmark; its result is not included in the speed table.

This is one card, host and software stack. Vision, concurrent clients,
`--kv-grow`, contexts above 32K, sustained thermal behavior, sampled decoding,
hipBLASLt calibration and a same-workload comparison against another engine
were not tested. Memory figures are snapshots rather than telemetry peaks.

## Test-build appendix

After the model benchmarks, the same source tree was reconfigured with
`STRATA_BUILD_TESTS=ON`. CTest registered 176 tests: 161 passed, nine skipped
and six failed. The full output is retained in [ctest.txt](data/ctest.txt), but
these are not counted as model-request failures:

- `ple_parity`: external binary fixtures were absent.
- `platform_memory_test`: `mlock` failed under the host's memlock limit.
- `hip_prefill_mmq_parity`: every printed Q2_0 numerical screen passed before
  the optional host quantizer was reported unavailable.
- `isa_guard_check`: two IQ AVX-512 public functions were absent from its
  allow-list; the same functions are tracked in issue #4213.
- `vmm_test`: a multi-chunk HIP memset failed after remapping. These model runs
  did not enable `--kv-grow`.
- `conversation_copy_streams_test`: first HIP copy-stream initialization used
  151,552 KiB, above that test's 16 MiB bound.

The final two observations may merit focused issues with minimal reproductions;
ordinary 32K Q2_0 and Coder serving remained stable throughout this report.
