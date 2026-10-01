# RX 9070 XT / gfx1201 / Windows — measured on this PC

Engine 0.1.30, commit `amdwin-integration` (upstream 30ec18e + #247 + #302 + the HIP-ordinal fix),
ROCm 10.2.0a20260930 from AMD's `nightly.repo.amd.com/rocm/whl-next/` into `.venv`.

## Host

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, gfx1201, wave32, 32 CUs, 15.922 GiB (15.774 free at load) |
| iGPU | AMD Radeon(TM) Graphics, gfx1036 — **takes HIP ordinal 0**, see the ordinal fix in `git log` |
| CPU | Ryzen 7 7800X3D, 8c/16t, AVX-512 |
| RAM | 63.1 GB |
| OS | Windows 11, driver 70260201, HIP runtime 70260201 |
| Port | 8090 (8080 was taken by SABnzbd) |

## Model load: Coder IQ1_M, 32K context

```
23.42 GiB loaded into RAM at 3.33 GiB/s
expert cache   5047 slots, 9.59 GiB VRAM (PROFILE-ranked, no eviction)
draft layer    839 MiB VRAM
503 MiB VRAM free with everything loaded
```

Load takes ~3 min cold, ~20 s when the expert arena is already resident in the page cache.
`expert arena: cudaHostRegister PORTABLE ok; large pages refused for 25149046724 B
(GetLargePageMinimum=2097152, VirtualAlloc error 1314); using 4 KB pages` — expected on Windows,
which does not grant large pages to a user process; the engine falls back and says so.

## Decode

128-token generation: **29.2 tok/s** (and 28.9 tok/s on a repeat run — ~29 tok/s stable).

## Correctness

| prompt | result |
|---|---|
| `What is 17*23? Reply with just the number.` | `391` PASS |
| `What is the capital of France? One word.` | `Paris` PASS |
| `Write a Python function that reverses a string. Code only.` | valid `def reverse_string(s: str) -> str: return s[::...]` PASS |

The model **thinks before answering**: with the default reasoning effort a 40-token budget is
entirely consumed by `reasoning_content` and `content` comes back `null` (finish_reason `length`).
That is the model's normal behaviour, not a backend fault — pass `"reasoning_effort": "none"`
(or a larger `max_tokens`) to get the answer.

## Prefill: PR #302 measured, not assumed

Cold prefill, ~1,875-token prompts with a per-trial random salt so the prefix cache cannot serve it.
Four trials per arm, median reported; `STRATA_PREFILL_TIMING=1` on both.

| arm | cold prefill | slot_waits | copied_events | GPU timeline |
|---|---:|---:|---:|---:|
| control (defaults 1/1/1) | **173 tok/s** | 8,270 | 8,270 | 10,338 ms |
| #302 batching (8/16/16) | **181 tok/s** | 1,034 | 1,034 | 9,876 ms |

**+4.6% median (a 4-8% spread across trials), not the +62% #302 reports.**

The engine's own counters show #302 doing exactly what it was written to do — dependency
granularity really did collapse from 8,270 individual waits to 1,034 (8x fewer, matching
`completion_batch=8`) — but **"wait copy" was only 2.2-2.3% of GPU time in both arms**:

```
control:  ... wait copy 237 (2.3%) ...
batched:  ... wait copy 214 (2.2%) ...
```

So there was almost nothing there to win. #302's +62% was measured with Q2_0 (8270 copies, and per
the PR's own numbers a copy-wait category of 2,746 ms out of a 6,144 ms `Prefill::run`) — a
bandwidth-bound regime where the copy engine really is the constraint. On this card, with 16 GB of
VRAM holding 5,047 of the experts, the run is bound elsewhere entirely:

| phase | share of GPU time |
|---|---:|
| gdn (recurrence + out proj) | **42%** |
| qsa proj | 13% |
| dequant | 11-12% |
| gemm (gate/up + down) | 11-12% |
| **wait copy** | **2.2%** |

#302 is a correct, worthwhile change — it is just not the bottleneck for *this* configuration.
A user with less VRAM (more expert traffic, more copies to schedule) would see more of its benefit
than this 16 GB card does. The three env vars are opt-in and default to 1/1/1, which is the right
call: nothing is lost by leaving them off.

## Where the speed actually is

29 tok/s decode is close to what the repo's own AMD numbers predict (30.8 tok/s for a 4K decode on
a 9070 XT, measured by a contributor on a Ryzen 9 3900X). The 7800X3D is not the constraint: the
expert cache holds 5,047 of 12,288 experts, so ~60% of routed experts still go to the CPU, and the
gdn + gdn-out-proj pair alone is 42% of GPU time.

The obvious lever is VRAM, not this code: the repo notes a card with more VRAM is faster *because
more of the model fits on the GPU*, and a big GPU does not reduce the RAM needed.
