# Supplementary notes: RX 6900 XT (gfx1030), measurements outside the standard protocol

These numbers come from the same machine and build as the standard run in this folder. They are **not** part of
`strata-qwen38-thinking-v1-draft.1` and are not comparable with standard results: different prompts, one request
per cell, no cap, and no repetition except where stated. They are here because the `rx6000` package has never been
reported on a real RX 6000 card, and these are the questions a reader of that section will have.

## Machine

- AMD Radeon RX 6900 XT 16 GB, gfx1030 (Navi 21), stock. `LnkSta: Speed 16GT/s, Width x16` (PCIe Gen4 x16),
  Resizable BAR 16 GB active.
- Intel Core i3-13100T, 4 cores / 8 threads, `acpi-cpufreq` with the `performance` governor (≈3.9 GHz all-core
  under load). 96 GB DDR4-3200 installed over 2 channels (32+16 / 32+16), 78.5 GiB visible to the guest.
- **QEMU/KVM guest on Proxmox VE 9.2 with the GPU passed through** (`vfio-pci`, `x-vga=1`, q35 + OVMF,
  `cpu: host`). I/O goes through virtio to the host's NVMe (WD Blue SN570 1 TB). Absolute numbers are therefore
  not bare-metal numbers.
- Ubuntu 24.04.4 LTS, kernel 6.8.0-124-generic. **System ROCm is 6.3.0 and is not used by the package**: the
  `rx6000` engine package carries its own HIP runtime. The RX 6000 docs point at a system ROCm 7; this run shows
  the released package does not need one.
- Strata Terminal 0.2.8, engine **0.1.54 `linux-x86_64-hip-rx6000`**, selected automatically. No source build.
- Model: ISTA-DASLab Qwen3.8-Flash-Next **IQ3_S**, 2 shards, 83.6 GB, sha256
  `4c1eb2ce…` and `316b46f3…` (the same files and hashes on ModelScope and Hugging Face).
- Engine options in every run below: `--max-context 262144 --kv k8v4 --kv-resident 32768 --vision
  --pool-cores 1,3,5 --spec 4 --spec-min-p 0.5` plus MTP, and `STRATA_HIP_PROMPT_F16=1` unless the row says
  otherwise.

## `STRATA_HIP_PROMPT_F16=1` is worth 2.1x on the prompt path

The engine prints a tip about this switch on gfx103x. Measured here on an 8,192-token prompt (the same 5,065-token
request body used for every row), one request per row, decode cap 120 tokens:

| `STRATA_HIP_PROMPT_F16` | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: |
| off | **488.2** | 42.0 |
| on | 1,023.7 – 1,035.3 over ten runs (median ≈1,029) | 42.3 – 45.1 |

The off arm was run once; the effect (2.1x) is far larger than the spread of the ten on-arms. Decode is
unchanged, so this is a prompt-path-only win, as `docs/AMD_HIP.md` describes. It is the only lever on this card
that moved a number by more than noise.

## Negative results, so nobody repeats them

| Setting | Decode tok/s (400-token answer) | Note |
| --- | ---: | --- |
| `--pool-workers` 2 | 42.6 | `--pool-cores 1,5` |
| `--pool-workers` 3 | 42.9 | default for this machine, `--pool-cores 1,3,5` |
| `--pool-workers` 4 | 42.8 | `--pool-cores 1,2,3,4` |
| `--ple-io direct` (default) | 42.9 | |
| `--ple-io mmap` | 42.4 | the guest cannot cache a 28.8 GB table in the RAM that is left |

Decode on this machine is not bound by CPU expert compute: three worker counts spanning 2 to 4 threads gave the
same number, which also means the CPU-side optimisations (huge pages, THP, worker placement) have nothing to win
here. That is the reason none of them were pursued further.

## Context size is nearly free; depth is not

Only `--context` changed between rows. Expert slots and KV RAM are read from the startup log; the prefill column is
the same 5,065-token request as above.

| `--context` | rope scaling | expert cache slots | KV in pinned RAM | 8K prefill tok/s |
| --- | --- | ---: | ---: | ---: |
| 65536 | none | 4,830 | 0.60 GiB | 1,029.9 |
| 131072 | none | 4,768 | 1.20 GiB | 1,028.7 |
| 262144 | none (native window) | 4,647 | 2.39 GiB | 1,030.4 |
| 524288 | yarn, factor 2 (derived automatically) | 4,412 | 4.78 GiB | 1,023.7 |
| 1048576 | yarn, factor 4 (derived automatically) | 3,944 | 9.56 GiB | not measured |

Prefill is flat from 64K to 512K. On a 16 GB card this is the KV streaming path doing the work: 32,768 of the
context's cells stay in VRAM (98.8–99.1 % of block reads hit VRAM in the standard run's requests, 5–61 MiB read
from RAM per request), and the rest of the KV lives in pinned RAM.

Long prompts over loopback, one request each, from `--context 131072` / `524288`:

| Prompt tokens | Prefill tok/s | Decode tok/s at that depth | Context setting |
| ---: | ---: | ---: | --- |
| 30,061 | 1,105.4 – 1,112.5 | 39.3 – 42.8 | 128K |
| 46,177 | 1,071.3 | 45.9 | 128K |
| 75,909 | 1,001.5 | 37.9 | 128K |
| 212,561 | 988.9 | 35.7 | 256K |
| 300,061 | 927.1 | 34.0 | 512K (yarn 2) |

Both of the last two are past the model's 262,144 trained window for the second one only; the 212K row is inside
it. `--context 1048576` also loads and serves (the startup log shows `yarn, factor 4`), though no long prompt was
run at that size.

## Pictures on gfx1030 work, with the CPU reader

`strata-vision` logged `on the CPU, 3 threads, no warm-up` (expected: this card has 16 GB, the GPU reader starts
from 24 GB). A synthetic image — the text "STRATA 7391", a red circle and a blue triangle on white — was read
correctly (text, colours, shapes, and the white background). 370-token image prompt, 200-token answer, 13.4 s end
to end. The reader used no VRAM. Checked by hand, not through the protocol.

## Not measured

- The Windows `rdna2` package, and any Windows machine.
- Any quantization other than IQ3_S.
- Per-request clocks, power limits and temperatures. The `rocm-smi` readings taken during a request were 45 °C and
  183 W of a 255 W limit at 15.3/16.0 GiB VRAM.
- Concurrency, long soaks, `--layer-split` (single card), and the serving suites.
- Anything on bare metal: see the virtualization note above.
