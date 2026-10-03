# Coder IQ1_M on RX 6900 XT and Threadripper 3990X

Measured on 2026-10-03 (Asia/Tokyo). This folder records installation and real hardware measurements on Windows. Results are added after inference completes; installation and the device self-test alone are not throughput measurements.

## Hardware and software

- AMD Radeon RX 6900 XT, 16 GiB VRAM, gfx1030.
- AMD Ryzen Threadripper 3990X, 64 physical cores / 128 logical processors.
- 128 GiB installed RAM (127.9 GiB visible), eight 16 GiB DDR4 modules configured at 3200 MT/s.
- Windows 11 Pro, build 10.0.26300; AMD display driver 32.0.21045.5002.
- Model storage: C: NVMe SSD, CSSD-M2B2TPG3VNF.
- Strata v0.1.38, upstream commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
- Official Windows HIP release binary; bundled ROCm `10.2.0a20260930`, hipBLASLt `100500`.
- Normal interactive desktop. Unrelated applications were not stopped. GPU power limits and driver settings were not changed.

Hardware snapshots, binary hashes, and the passing device self-test are stored alongside this report. GPU runtime detection/self-test passed before downloading the model.

## Installation

Source: <https://github.com/Niko1221/Strata/releases/tag/v0.1.38>

```powershell
.\.venv\Scripts\python.exe setup.py --yes --family coder --model IQ1_M --backend hip --context 65536 --vision no --host 127.0.0.1 --no-start
```

Application: `C:\Dev\Strata`; model data: `C:\Dev\Strata-data`.

Coder source: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, pinned revision `5348543e0147355ac9cbcb031184a3546350988e`. Coder retains 256 experts per layer; its IQ1_M label does not mean every tensor is uniformly one bit. Setup also obtains and prepares the original checkpoint's MTP draft tensors.

## Measurement method

`benchmark.py` sends streaming OpenAI-compatible requests to the real local engine. Each configuration uses one separate warm-up and three measured requests per selected workload. Requests use temperature 0, reasoning effort `none`, images off, and a 256-token output cap. Actual generated tokens and early stops are retained.

The short workload requests a Python task queue implementation. The longer workloads use synthetic Python modules, sized close to 4,096 and 32,768 tokens through the server's token-count endpoint. Full request bodies, output text, usage, engine timings, and stream events are retained in each run's JSONL.

Each trial changes an early nonce to avoid reusing a long prompt. The engine's actual fresh/reused token counts are authoritative. The expert cache remains adaptive and warm within a server session; this is different from prompt-prefix reuse. Model loading is excluded from request timings.

- Prompt throughput: fresh prompt tokens divided by the engine's prompt time.
- Decode throughput: engine-generated tokens divided by the engine's decode time.
- TTFT: client request start to the first nonempty generated text or reasoning delta; empty stream messages are ignored.
- Total latency: client request start to stream completion.

Example, with the local server already ready:

```powershell
.\.venv\Scripts\python.exe bench/results/2026-10-03-rx6900xt-coder/benchmark.py --cases short,4k,32k --runs 3 --out bench/results/2026-10-03-rx6900xt-coder/default --log strata-coder-iq1_m.log
```

## Results

Pending real inference measurements.

## Scope

These measurements concern this Windows installation and these synthetic coding workloads. They do not establish overall coding quality, full 64K-context correctness, or performance on another operating system. The generated 256-token excerpts can be incomplete by design.
