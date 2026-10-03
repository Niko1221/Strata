# V100 L3 conversation-cache benchmark — 2026-10-03

## Purpose

This benchmark measures the disk-backed conversation cache when two agents alternate. It compares a cold prompt read with a disk restore of the same conversation state.

## System and configuration

- GPUs: two Tesla V100-PCIE-16GB cards
- Layer split: layer 20
- KV format: INT8
- Maximum context: 524,288 tokens
- Model storage and cache storage: Kingston NVMe SSD, ext4
- Disk-cache budget: 25 GiB
- Host-RAM conversation cache: disabled (`--conversation-cache-mib 0`)
- Prompt checkpoints: enabled (`--prompt-cache 6`)
- Sampling: greedy
- Speculative decode: fixed one-token parity mode

The test used the sequence A, B, A+. Conversation A contained 33,725 prompt tokens. B was an unrelated 61-token conversation. A+ added 22 prompt tokens after A. The baseline and disk-cache runs used the same engine settings. `STRATA_STATE_HASH=1` compared the complete main-model state after each request.

## Results

| Measurement | Result |
|---|---:|
| Cold read of conversation A | 20,332.7 ms |
| L3 record size for A | 989,486,234 bytes (943.6 MiB) |
| Capture plus asynchronous file-write work | 3,536.1 ms |
| File read for A | 1,614.6 ms |
| Two-GPU restore for A | 315.8 ms |
| Read plus restore | 1,930.4 ms |
| Complete A+ prompt phase, including 22 new tokens | 3,015.1 ms |
| Cold-read time divided by complete resumed prompt time | 6.74x |
| Complete resumed prompt latency reduction | 85.2% |
| Reused prompt tokens | 33,725 |

The restored output tokens and all reported main-model state hashes matched the cold baseline. The test verified two stage images in the disk record and used no complete-conversation RAM cache.

The capture must complete before the active GPU session can be overwritten. The file write starts asynchronously after capture and can overlap the next request. The reported 3,536.1 ms is the sum of capture and file-write work, not a request stall measurement.

## Additional validation

A shorter 1,957-token A/B/A run also passed output and byte-exact state parity. Its 266,814,154-byte record loaded in 409.7 ms and restored across two GPUs in 89.5 ms.

Separate real-model runs verified both behaviors below with the same 25 GiB store:

- A parked conversation survived an engine restart and resumed with byte-exact state parity.
- A record with a modified payload byte was rejected, removed, and replaced by a cold prompt read with matching output.

## Evidence

The raw test artifacts were produced outside the repository under the model-storage benchmark directory. The permanent harness is `tools/conversation_cache_disk.py`; its verifier tests are in `tools/test_conversation_cache_disk.py`.
