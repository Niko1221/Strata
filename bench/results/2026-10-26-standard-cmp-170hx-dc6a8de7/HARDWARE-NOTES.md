# Hardware notes: 4x CMP 170HX (40 GB, PCIe Gen2 x16)

These cards are **not stock**. Every number in this report was measured on modified
hardware, so treat it as "CMP 170HX after the community unlock", not as a stock card.

## 1. HBM: 10 GB -> 40 GB per card

A stock CMP 170HX exposes roughly 10 GB of usable HBM2e. These four cards are unlocked
to **40 GB each** — `nvidia-smi` reports 40,960 MiB — using the published research on the
card's secure coprocessor (Pry, *A Canary in the Crypto Mine: Defeating Stack Protection
in a GPU Secure Coprocessor*, Zenodo DOI 10.5281/zenodo.20916112) and the community tooling
that followed it.

We did **not** get 80 GB: that needs a GSP-aperture change we have not made, so usable VRAM
stays at 40 GB per card and the whole model has to fit inside that envelope.

## 2. PCIe: x4 -> x16, and Gen1 -> Gen2

Stock, this card is wired **x4** and links at Gen1. Ours carry two changes:

- the community **capacitor mod** that restores the full **x16** electrical lanes;
- the **Gen2** rate (5 GT/s) from the register override, made persistent with a dracut
  hook so it survives reboots.

Verified on the machine this report came from:

```
$ nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv,noheader
0, NVIDIA CMP 170HX, 2, 16, 40960 MiB
1, NVIDIA CMP 170HX, 2, 16, 40960 MiB
2, NVIDIA CMP 170HX, 2, 16, 40960 MiB
3, NVIDIA CMP 170HX, 2, 16, 40960 MiB
```

The engine's own probe at start-up measures **6.2 GB/s host->device** (Gen2 x16 tops out at
7.877 GB/s).

**Why it matters for reading this report:** in a multi-card layer split the engine crosses
PCIe at every layer boundary. On a stock x4 Gen1 link (about 0.99 GB/s) the same
configuration would produce a completely different result. Any comparison against a stock
170HX has to account for the width and the generation, not just for the card model.

## 3. The rest of the machine

| Item | Value |
|---|---|
| Mainboard | Supermicro X10DRG-Q (Powerleader PR4964GW OEM) |
| CPUs | 2x Xeon E5-2690 v4, 2 NUMA nodes, **no AVX-512** |
| RAM | 247.5 GiB |
| Cards | 4x CMP 170HX (GA100, sm_80): PCI 02:00 + 03:00 on CPU0, 82:00 + 83:00 on CPU1 |
| Model storage | NVMe ZHITAI TiPlus7100s 2 TB |
| OS | Ubuntu 26.04 LTS, 7.0.0-29-generic |

Two cards per CPU socket is what makes a two-engine arrangement possible on this box: two
engines, two cards each, one engine per NUMA node.

## 4. What configuration this report measured

The measured server is **one engine on two cards** (group A: PCI 02:00 + 82:00, 4 slots),
started from the config in [`engine-config.json`](engine-config.json). The twin engine
(group B: PCI 03:00 + 83:00, port 8081) was running but had **no client** during the run.
The engine options in effect:

```text
--pack /storage/Strata-data/packs/iq3_s
--native Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
--ple-gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
--expert-profile expert-profile.bin  --expert-cache auto
--prefill auto  --spec 4  --max-context 1048576
--kv q4_0  --kv-resident 20480  --ple-io ram
--mtp /storage/Strata-data/mtp/rt  --pipeline-windows 2
--prompt-cache 24 --prompt-cache-every 4096 --prompt-cache-spread
--vision  --vram-reserve-mib 700  --host-core first
STRATA_SIDE_MIXER=2 STRATA_ATTN_DAILY=1 STRATA_GR_UP75=1 STRATA_GU_SWIGLU=1
STRATA_QSA_ROWS=1 STRATA_B6_MMVQ_ROWS=1 STRATA_CPU_IQ_ALL_EXACT1=1 STRATA_DEQUANT_WIDE=1
```

The eight `STRATA_*` switches are the sm_86 bit-exact set from `docs/DETAILS.md`, applied
here to sm_80 by hand; on this machine they changed generated tokens by zero bytes in our
identity checks.

## 5. Caveats specific to this hardware

- The unlock is unsupported by NVIDIA, voids the warranty, and is per-card irreversible in
  parts (the fuse override). Nothing here is a recommendation to do it.
- 40 GB per card is what we have. **Do not read this report as an 80 GB result** — we could
  not make 80 GB usable.
- The cards sit behind a modded x16 link; a stock 170HX on x4 Gen1 will be much slower at
  anything that crosses PCIe, which for this engine includes multi-card layer splits.
