# Community benchmark on AMD Instinct MI50 32 GB (single card, gfx906)

Measured on 2026-10-10 by phoenixclyde. One MI50 32 GB runs a Qwen3.8-Flash-Next build at its
native 262,144-token context **on a single card**, with 4-bit KV and the whole expert cache in VRAM.
The upstream AMD notes list "a single-card run" as not done for gfx906, and the earlier gfx906 report
on this repository is two 16 GB cards with a layer split. Main limitation: one card, one user, no
concurrency measurement, and the weights are a community abliterated variant rather than an official
quantization.

## Hardware and software

- **GPU:** 1x AMD Instinct MI50 32 GB (Vega 20, gfx906, wave64; PCI id `1002:66a1`; VBIOS
  `113-D1631711-100`). VRAM **34,342,961,152 B** as read from the card. One card, no peer, no NVLink.
- **CPU / RAM / storage:** Intel Xeon E5-2696 v4 (22C/44T, AVX2, **no AVX-512**); **128.7 GiB** DDR4;
  SATA SSD.
- **PCIe:** the root port caps the link at 8 GT/s x16, so the whole path negotiates **Gen 3 x16**
  (the card's own hop is 16 GT/s). Read with `lspci -vv` hop by hop. **No power limit** was set.
- **OS / runtime:** Ubuntu, kernel **7.0.0-34-generic** with the in-kernel `amdgpu` driver.
  ROCm **7.14**, from the community image `mixa3607/rocm-gfx906:7.14-complete`, because current AMD
  ROCm releases no longer ship gfx906 libraries.
- **Engine:** Strata **0.1.41**, release image `strata-gfx906:0.1.41` (source build with
  `-DSTRATA_HIP_GFX906=ON`; `STRATA_USE_HIP` not defined). Engine binary md5
  **`160611838796d4b172ec3a642ccd988d`**, 24,982,264 B.
- **Background workloads:** the machine's other GPUs (an RTX 2070, the ASPEED BMC) do no inference
  here; the container is given only this MI50.

## Model and configuration

- **Model:** `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`, quant **`unsloth-UD-IQ4_XS`**
  (Qwen3.8-Flash-Next, 262,144-token native window). A community abliterated variant, not an official
  quantization.
- **Custom pack:** `/data/packs/huihui-ud-iq4_xs`, prepared with the repository's `tools/iq_pack.py`;
  `index.txt` lists **1079 tensors**, `dense.bin` is **1,384 MiB**. Expert types: 44 of the 48 layers
  use `IQ3_S` gate/up with `IQ4_NL` down, 4 use `IQ4_XS` with `Q8_0` (`native_experts.txt`).
- **Vision encoder:** served, `--vision` on. No image was used in the measurements below.
- **Context / KV / cache:** `--max-context 262144`; `--kv k8v4` (4-bit KV); `--expert-cache auto`,
  **9761 resident expert slots / 21.99 GiB of VRAM**; `--prefill auto`; `--prompt-cache-every 8192`;
  `--conversation-cache-mib 4096`. No low-RAM mode, no calibration, no speed projection.
- **MTP / reasoning / sampling:** adaptive MTP draft head over a 106,299-token draft vocabulary,
  `--mtp-window 8192`, `--spec-min-p 0.70`; thinking off per request
  (`chat_template_kwargs: {"enable_thinking": false}`); greedy (`temperature 0`).
- **Serve-side patches:** two local overrides in the serve layer (AMD PCIe readings in
  `telemetry.py`; a sparkline-axis fix in `web/app.js`). They do not touch the engine or its output;
  the engine's own diff is empty.

```text
strata --serve --pack /data/packs/huihui-ud-iq4_xs --native /models/<...>/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --resident-budget-gib 76 --expert-profile /src/data/expert-profile.bin --expert-cache auto \
  --pcie-frac 0.20 --spec 4 --spec-min-p 0.70 --mtp /data/mtp/rt --max-context 262144 --kv k8v4 \
  --prefill auto --prompt-cache-every 8192 --expert-profile-save /data/profiles/huihui-ud-iq4_xs-learned.bin \
  --conversation-cache-mib 4096 --vision --mtp-window 8192
```

## Method

- **Speed:** the main arm (`--kv k8v4`) uses three prompt lengths (**4,096 / 32,768 / 128,000**
  tokens); the KV reference arm (`--kv fp16`) uses **two** of them (**32,768 / 128,000**) -- the
  4,096 tier adds nothing to a KV comparison and one engine run per arm is the expensive part.
  **Every tier that was run has three measured runs** (9 for `k8v4`, 6 for `fp16`), output
  cap **256**, greedy, one request at a time. Every run puts a different nonce at the head of the
  prompt, so the first token differs and **nothing can be reused**; the engine reports `reused 0` for
  all of them. `speed.py` in this folder drives it and reads the engine's own
  `strata serve: prompt ...` line (prompt ms, decode ms, reused/read counts, draft acceptance) plus the
  `decode expert cache hit rate` line. Decode throughput is never derived from generated / total time.
- **Recall:** the repository's `tools/needle_bench.py`, run inside the container (its haystack is
  built from the repository, so it needs the source root at `/src`), lengths **32k** and **128k** at
  depths **10/50/90**.
- **Units:** GiB for memory, tokens for lengths. `TTFT` was not measured -- requests were
  non-streaming, so total latency is given instead.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `--kv k8v4`, 4,096-token target | 3,141 | 0 | 186 / 256 / 234 | 3 | 393.80 [361.70..394.30] | 44.70 [38.30..45.40] | not measured |
| `--kv k8v4`, 32,768-token target | 26,038-26,040 | 0 | 41 / 256 / 256 | 3 | 460.70 [457.30..461.50] | 40.70 [38.40..47.50] | not measured |
| `--kv k8v4`, 128,000-token target | 93,337-93,339 | 0 | 256 / 256 / 256 | 3 | 458.80 [458.80..460.70] | 40.70 [35.40..46.80] | not measured |
| `--kv fp16`, 32,768-token target | 26,039-26,041 | 0 | 173 / 256 / 256 | 3 | 456.30 [456.00..457.30] | 34.90 [34.80..38.40] | not measured |
| `--kv fp16`, 128,000-token target | 93,340-93,341 | 0 | 256 / 256 / 256 | 3 | 458.00 [457.90..459.50] | 37.80 [35.20..38.70] | not measured |

Every run is published above and in `speed.json` (one record per run, with the engine's own timing line).
All 15 runs report `reused 0`: the nonces make each prompt new work.

Prompt throughput is flat from 32,768 tokens up (about 458-461 tok/s), so prefill is not
over-subscribed at 93k prompt tokens. Decode sits at 35-47 tok/s.

The two `--kv` arms differ in one setting only, and the difference is visible in the decode column and
in the cache hit rate, not in prefill:

| `--kv` | Prompt tok/s @32768 | Decode tok/s @32768 | Decode tok/s @93339 | Cache hit @32768 | Resident expert slots | KV VRAM |
| --- | --- | --- | --- | --- | ---: | --- |
| `k8v4` | 460.70 | 40.70 | 40.70 | 90.8% | 9,761 | 4-bit |
| `fp16` | 456.30 | 34.90 | 37.80 | 83.1% | 8,047 | 16-bit |

4-bit KV frees about 3.8 GiB of VRAM, which the engine spends on 1,714 more resident experts; the
decode cache hit rate follows (+3.4 to +7.7 points) and decode runs 7.7%-16.6% faster at equal
prompt throughput. Prefill is unchanged (within 1%).

Per-run `drafts accepted` is in the per-run table and in `speed.json`; it moves run to run even at a
fixed prompt, as the specification warns.

## Correctness and limitations

### Recall (the repository's `tools/needle_bench.py`, run inside the container)

`--kv k8v4` (the production setting):

| Length | Depth | Result | Actual prompt tokens | Seconds |
| --- | ---: | --- | ---: | ---: |
| 128k | 10% | **FOUND** | 126,640 | 264 |
| 128k | 50% | **FOUND** | 126,638 | 231 |
| 128k | 90% | **FOUND** | 126,638 | 264 |
| 32k | 10% | **FOUND** | 33,286 | 74 |
| 32k | 50% | **FOUND** | 33,286 | 74 |
| 32k | 90% | **FOUND** | 33,287 | 41 |

`--kv fp16`, same prompts:

| Length | Depth | Result | Actual prompt tokens | Seconds |
| --- | ---: | --- | ---: | ---: |
| 128k | 10% | **FOUND** | 126,640 | 265 |
| 128k | 50% | **FOUND** | 126,638 | 231 |
| 128k | 90% | **FOUND** | 126,638 | 247 |
| 32k | 10% | **FOUND** | 33,286 | 76 |
| 32k | 50% | **FOUND** | 33,286 | 73 |
| 32k | 90% | **FOUND** | 33,287 | 41 |

**Paired by (length, depth): 6 pairs, 0 disagreements** -- both settings found all six.

`262k` was attempted on the k8v4 arm and the script skipped it itself
(`262k: skipped (the server's context is 262144)`): it targets 98% of the length and adds 200 tokens
for the answer, so a 262,144-token context cannot hold its own name. Recorded as a skipped case, not a
failure. In `needles.json` this is the object `"k8v4_262k": {"skipped": true, "reason": ...}`
(an earlier revision wrote a bare `null`, which read as "missing" rather than "skipped by design"). No `128k` run was near that limit (126,638-126,640 actual prompt tokens).

### Limits

- **Single card, single user, no concurrency measurement.** One request is in flight at a time; the
  engine's batch/throughput behaviour on this card is untested.
- **TTFT was not measured** -- the requests were non-streaming, so the table carries total latency
  instead: 13.2-13.6 s at 3,141 prompt tokens, 57.9-63.2 s at 26,038-26,040, 209.0-210.0 s at
  93,337-93,339 (client-side, includes prefill, decode and transport).
- **Prompt lengths are actual, not targets.** The script estimates 2.4 characters per token; this
  repository text actually runs at 3.0-3.3, so the 4,096/32,768/128,000 targets landed at
  3,141 / 26,038-26,040 / 93,337-93,341. The table reports what was measured; the targets are labels.
- **The weights are a community abliterated variant** (`huihui-ai/...-abliterated-GGUF`,
  `unsloth-UD-IQ4_XS`), not an official quantization, so this is not directly comparable to reports
  that used the official files. The pack is built with the repository's own `tools/iq_pack.py`.
- **Not paired with the earlier gfx906 report** (2x MI50 16 GB, Coder IQ1_M, layer split, 131k
  context): different machine, model, quantization and card count.
- **A distribution-level quality gate was tried and abandoned on this card.** Teacher-forced
  top-256 log-probability comparisons (the engine's own `STRATA_LOGPOS` probe) show that on gfx906
  the measured KLD between two `--kv` settings is dominated by which experts happened to be resident,
  not by the KV dtype: two runs of the *same* dtype and context but 96 different resident slots
  already differ by KLD 0.284 (argmax 91.5%), while 1224 slots give 0.366. Since changing `--kv`
  necessarily changes how much VRAM the KV occupies, and therefore the cache, the two effects cannot
  be separated on one card. That is why this report answers the quality question with the recall
  check above instead.
- **Vision is enabled but untested** here; no image was sent.
- Calibration and experimental speed projection were both off.


### Per-run data

| Arm | Target | Run | Prompt tokens | Reused | Prompt ms | Prompt tok/s | Generated | Decode ms | Decode tok/s | Drafts accepted | Decode cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `--kv k8v4` | 4,096 | 0 | 3,141 | 0 | 8,684 | 361.70 | 186 | 4,856 | 38.30 | 106/131 | 89.9% |
| `--kv k8v4` | 4,096 | 1 | 3,141 | 0 | 7,965 | 394.30 | 256 | 5,638 | 45.40 | 153/188 | 93.1% |
| `--kv k8v4` | 4,096 | 2 | 3,141 | 0 | 7,976 | 393.80 | 234 | 5,238 | 44.70 | 140/163 | 89.9% |
| `--kv k8v4` | 32,768 | 0 | 26,038 | 0 | 56,943 | 457.30 | 41 | 863 | 47.50 | 19/23 | 97.4% |
| `--kv k8v4` | 32,768 | 1 | 26,039 | 0 | 56,421 | 461.50 | 256 | 6,296 | 40.70 | 145/189 | 90.8% |
| `--kv k8v4` | 32,768 | 2 | 26,040 | 0 | 56,527 | 460.70 | 256 | 6,660 | 38.40 | 146/192 | 89.1% |
| `--kv k8v4` | 128,000 | 0 | 93,339 | 0 | 202,601 | 460.70 | 256 | 7,226 | 35.40 | 149/186 | 88.4% |
| `--kv k8v4` | 128,000 | 1 | 93,337 | 0 | 203,444 | 458.80 | 256 | 6,288 | 40.70 | 152/191 | 92.6% |
| `--kv k8v4` | 128,000 | 2 | 93,339 | 0 | 203,439 | 458.80 | 256 | 5,472 | 46.80 | 152/183 | 94.9% |
| `--kv fp16` | 32,768 | 0 | 26,041 | 0 | 56,946 | 457.30 | 173 | 4,973 | 34.80 | 100/115 | 81.1% |
| `--kv fp16` | 32,768 | 1 | 26,040 | 0 | 57,110 | 456.00 | 256 | 7,331 | 34.90 | 150/194 | 83.1% |
| `--kv fp16` | 32,768 | 2 | 26,039 | 0 | 57,067 | 456.30 | 256 | 6,673 | 38.40 | 152/193 | 88.6% |
| `--kv fp16` | 128,000 | 0 | 93,340 | 0 | 203,140 | 459.50 | 256 | 6,767 | 37.80 | 159/199 | 89.2% |
| `--kv fp16` | 128,000 | 1 | 93,341 | 0 | 203,819 | 458.00 | 256 | 7,268 | 35.20 | 152/192 | 86.4% |
| `--kv fp16` | 128,000 | 2 | 93,341 | 0 | 203,847 | 457.90 | 256 | 6,613 | 38.70 | 161/199 | 89.5% |

The engine's own timing lines for every request are in `engine-timing-lines.txt`.
