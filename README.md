# Community bench: RTX 5070 Ti **Laptop** 12 GB, Flash-Next IQ3_XXS, 262K context

Results-only submission, no engine changes. Engine 0.1.41 (release), Windows 11.

The other two 5070 Ti rows in the index are desktop cards. This one is the 12 GB laptop part, which is
the smallest VRAM anything in this table has run a 262K context on. The interesting result is not that
it runs — it is that **decode speed is flat from a 4K prompt to a 128K prompt.**

## Hardware

- RTX 5070 Ti Laptop GPU, 12,227 MiB VRAM, driver 616.64, 140 W power limit, AC power
- AMD Ryzen 9 9955HX. Windows reports 8C/16T for this part (the SKU name says 16-Core); 16 logical
  processors, engine picked 8 participating threads.
- 61.7 GB RAM
- YMTC PC411 1 TB NVMe SSD
- Windows 11 Home China, build 26100

## Software and model

- Strata engine **0.1.41**, `source: release`, CUDA 13.0, `archs [75, 86, 89, 120]` (from `engine/BUILD.json`)
- Qwen3.8-Flash-Next, **IQ3_XXS**
- Server started through `START-HERE.bat`; the running config was:

```
--max-context 262144 --kv int8 --kv-resident 32768 --vram-reserve-mib 700
--expert-cache auto --prefill auto --spec 4 --vision
--pcie-frac 0.20 --spec-min-p 0.70 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97
```

The last five are this machine's own `--calibrate` output, not defaults — do not copy them.

- At startup the expert cache filled **1585 of 1585 slots** (2.61 GiB of VRAM):

```
strata generate: expert cache 1585 slots, 2.61 GiB of VRAM
strata generate: pre-filled 1585 of 1585 slots from the profile in 0.3 s (10687 MB/s)
```

## Method

`measure.py` (included) drives the OpenAI-compatible endpoint at `http://127.0.0.1:8080` and reads both
figures back out of the engine's own log rather than timing them itself:

```
strata serve: prompt N tokens = R reused + P read in T ms (X tok/s), G generated in U ms (Y tok/s)
strata serve: decode expert cache hit rate: H% (a hits / b lookups)
```

`X` is prefill, `Y` is decode, `H` is the decode-time expert-cache hit rate. All three are the engine's
numbers.

- 3 runs per tier, **256-token cap**, non-streaming.
- Each run puts a unique nonce at the very start of the prompt, so the server reports `0 reused` and no
  run rides on the previous run's prefix cache. Every row below is a genuinely fresh prompt.
- Prompt body is a repeated English sentence; the task tail asks for a 1..200 list so the generation
  reaches the cap instead of stopping early.
- Medians over the 3 runs. Per-run JSON in this folder.

## Results

| Prompt tokens (actual) | Prefill tok/s | Decode tok/s | Decode range | Expert cache hit | Wall |
|---|---|---|---|---|---|
| 4,184 | 788.5 | 63.8 | 60.7 – 63.9 | 57.3 % | 9.3 s |
| 32,855 | 901.4 | 64.1 | 63.4 – 64.1 | 57.4 % | 40.5 s |
| **131,166** | **886.8** | **64.4** | 59.7 – 65.3 | 58.0 % | 152 s |
| 196,696 | 869.4 | 61.2 | 59.7 – 62.9 | 56.4 % | 231 s |

192K is beyond the 4K / 32K / 128K the index uses; it is here because 262K is the claim this row is
making. Same method. Note the wall time: a 197K prompt costs 231 s of prefill — the context is
*usable*, not *cheap to fill*.

## What stands out

**Decode is flat across a 47x prompt-length increase: 63.8 → 64.1 → 64.4 → 61.2 tok/s.** Prefill is
flat too (788 → 901 → 887 → 869 tok/s). Context length costs this machine almost nothing to *decode*;
what it costs is *prefill time* — a 197K prompt takes 231 s to read at ~870 tok/s.

The expert cache is the reason, and it is also why `nvidia-smi` is the wrong instrument here. VRAM holds
a cache of hot experts that shares space with the KV cache. Without KV streaming, growing the context
evicts experts instead of adding VRAM — same 12 GB card, same model, `204800` with `--kv-resident`
absent measured **207 expert slots and 2.3 tok/s** (2 replies, indicative). With `--kv-resident 32768`
the KV cache lives in RAM, the 128K run cached 58.0 % of its expert lookups, and the slot count holds
at 1585–1676.

So the honest summary of this row is not "12 GB runs 262K". It is: **on 12 GB, `--kv-resident` is what
decides whether a long context is usable at all**, and once it is on, the speed is roughly constant.

## Limitations

- One laptop, one model, one engine version. Nothing here is closed for the bench; other desktop
  applications were running as usual, so the absolute numbers are pessimistic by an unknown amount.
- The prompt body is repeated English prose, not code or prose in another language. Prefill on this
  model is content-sensitive — the RTX 4090 rows in the index show a code/prose spread — so treat the
  prefill column as valid for this prompt shape only.
- No clock, temperature or power draw was captured. The 140 W limit is the card's, not a measured draw.
- Non-streaming requests, so no separate TTFT; the engine's prompt time is the closest available proxy.
- The 204800-without-streaming figure quoted above is 2 replies from an earlier session, not part of
  this sweep. It is a direction, not a measurement.
- CUDA 13.0 release build only; no source build was compared.
