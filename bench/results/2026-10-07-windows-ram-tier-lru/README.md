# An LRU part for the RAM budget, and an elastic one (Windows)

Measured October 6-7, 2026, on top of upstream `82f46a8` with #833's batched stager reads ported (the first commit
of this branch). Everything here is opt-in: without `--resident-lru-gib` and `STRATA_LRU_KEEP_FREE_GIB` the engine
behaves as before.

## What changes

`--resident-budget-gib N` fills N GiB of RAM once, at start, with the experts the expert profile ranks hottest
after the GPU caches. The profile is global: a conversation about something else keeps reading the drive for
experts that are cold in the profile but hot for it.

- `--resident-lru-gib L` keeps L GiB of the budget for the experts the decode reads from the files, least recently
  used out first. The pool is the unbuffered reads' stage pool, so a hit costs nothing extra. The prompt path copies
  from it too, but does not add to it or refresh it: a prompt chunk sweeps every cold expert once, and letting it
  in would push out what the decode reuses.
- `STRATA_LRU_KEEP_FREE_GIB=G` makes that part elastic. A watcher checks the available RAM twice a second. Below G
  it frees the coldest LRU buffers (back to the OS at once, never paged) and lowers the pool's cap; with RAM free
  again it raises the cap a GiB at a time up to L. The pool never shrinks below 1 GiB: a decode window claims a few
  dozen buffers at once, and a smaller pool allocated and freed them every window (decode 43 tok/s instead of 50
  under pressure). On Windows it watches the available commit as well as the available RAM: every LRU buffer is
  committed memory, and with the commit charge at its limit another program's allocation fails even with RAM free
  (measured on this PC: 112-120 of 121 GiB committed while serving, with 8-9 GB of RAM free).
- `STRATA_TIER_TRACE=<file>` writes one line per expert blob served outside the GPU caches (time, decode or prompt
  path or fill, RAM copy or files, layer, expert, bytes). It is how the LRU was chosen: replaying a traced session
  of six answers on different topics against policies of the same 14 GiB, an LRU fed by the decode's reads cut the
  decode's drive reads from 182 to ~100 GiB; LFU-decay and an LRU fed by the prompt path did worse.

The serve log reports the LRU's hits (`RAM tier LRU ...`) and, when elastic, what it holds and what it gave back.

## Configuration

| Component | Measured configuration |
| --- | --- |
| System | Windows 11 Pro, Core i7-12700K (8P + 4E, AVX2, AVX-VNNI), 64 GB DDR5-6000 |
| GPUs | 2x RTX 5060 Ti 16 GB (CUDA0 on CPU lanes PCIe 5.0 x8, probed 25 GB/s; CUDA1 on the chipset PCIe 4.0 x4, 7.2 GB/s); the RTX 3060 Ti runs the image encoder |
| Storage | `experts.bin` on a Samsung 990 PRO 2 TB (CPU lanes) |
| Build | MSVC 14.43, CUDA 13.0.1, Release, `STRATA_PORTABLE=ON`, sm_120 |
| Model | Qwen3.8-Flash-Next GSQ-RCO IQ3_S (experts.bin 50.3 GB), MTP draft layer |
| Runtime | `--max-context 262144 --kv int8 --kv-resident 32768 --vision`, layer split 30, `--pipeline-windows 2`, `--prefill auto:16384`, `--spec 4 --spec-min-p 0.85`, `STRATA_UNBUFFERED_LOAD=1` |

Each run: one 8K-token and one 32K-token prompt, then six 600-token answers on six different topics (fixed prompts,
greedy, thinking off), through the server. Decode tok/s is the mean of the six answers. The first run after the
server's start measured low in every series, so a warm-up run was discarded; repeated runs of one configuration
differ by about 1-2 tok/s.

## Results

With `--spec-min-p 0.5` (setup's value) and `--prefill auto`, no pipeline:

| RAM budget | Decode tok/s | Prompt tok/s 8K / 32K | Lowest available RAM |
| --- | ---: | ---: | ---: |
| 14 GiB by profile (before) | 41.9 | 662 / 1,340 | 22.6 GB |
| 10 GiB by profile + 4 GiB LRU | 47.0, 45.1 | 629 / 1,282 | 23.2 GB |

With the runtime of the table above (`--spec-min-p 0.85`, which alone moved decode from ~56 to ~62 here):

| RAM budget | Decode tok/s | Prompt tok/s 8K / 32K | Lowest available RAM |
| --- | ---: | ---: | ---: |
| 12 GiB by profile + 6 GiB LRU | 61.4, 59.3 | 743 / 1,329, 768 / 1,412 | 13-14 GB |
| 12 + 18 GiB LRU, elastic, floor 6 GiB | 63.4, 65.1, 65.4, 64.2 | ~950 / ~1,540 | 6.5-9 GB |
| 12 + 24 GiB LRU, elastic | 62.4, 63.2 | ~970 / ~1,610 | ~6 GB |
| 6 + 24 GiB LRU, elastic | 62.3, 62.3 | ~1,030 / ~1,630 | 11 GB |

Another program taking 20 GiB while the server decodes (a process that allocates and touches the memory, holds it
60 s, then frees it), elastic 12 + 18 GiB, floor 6 GiB:

| Phase | LRU held | Decode tok/s |
| --- | ---: | ---: |
| Before | 18 GiB | 63.5 |
| While the other program holds 20 GiB (it got them within 6-7 s) | ~2 GiB | 48-55 |
| After it frees them | back to 18 GiB within ~20 s | 60 -> 62 -> 70-73 |

With a plain (not elastic) 18 GiB LRU under the same load, decode stayed at ~60 tok/s but the available RAM fell to
0 and Windows paged the other program out instead.

## Not kept

- Offering idle LRU buffers to Windows (`OfferVirtualMemory` / `ReclaimVirtualMemory`): it works - under pressure
  the OS discarded them, never paged - but every offer and reclaim moves hundreds of pages and decode fell from 59
  to 27-40 tok/s depending on how long a buffer had to be idle before it was offered.
- A 1 GiB LRU on top of the budget: too small to see reuse (the cold experts' reuse distance is long).
- The routing lookahead reading the predicted blobs when reads are unbuffered: it tripled the bytes read and was
  slower.
