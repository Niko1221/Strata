# Method

How the two arms in `README.md` were produced, so the run can be repeated.

## The two arms

One RX 6700 XT (gfx1031) run as gfx1030, one engine session per arm, the model reloaded from cold between arms:

| arm | `env` in `strata-swift-iq3_s.json` | prompt GEMM route |
|---|---|---|
| default | `HSA_OVERRIDE_GFX_VERSION=10.3.0` | rocBLAS f16 in, f32 out |
| f16 | `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `STRATA_HIP_PROMPT_F16=1` | rocBLAS f16 in and out |

The engine says which route it took on its own startup line, so each arm is identified in
`engine-log-default.txt` / `engine-log-f16.txt` (`STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and
out` against `tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here`).

Nothing else changed between arms: same pack, same GGUF, same flags, same server, same session order. The card was
idempotent between runs (no resets, `rocBLAS error` lines: zero in both arms).

## The suite

`python tools/strata_bench.py --url http://127.0.0.1:8080 --api-key <key> --long --out <dir>` - the built-in
benchmark against the server that is already running, three runs per test, medians with ranges, thinking off, a
unique tag per prompt so the fresh-prompt tests do not get help from the cache. The suite is `chat` (three short
questions), about 4K prompt tokens, about 16K, an agent session (five turns that each add about 1,500 words of a
project file and continue from the model's own earlier answers), and with `--long` about 32K prompt tokens. Timings
are the engine's own `prompt_per_second` / `predicted_per_second`; first-token time is measured by the client.

`results-default.json` and `results-f16.json` are the tool's `results.json` as written, field for field: every
request with its token counts, the engine's timings, the client's first-token times, any error, and the machine
facts block.

The suite ran in this order: default arm, then the server was restarted and the f16 arm ran, with
`--compare <the default arm's folder>` so the tool prints its own side-by-side table; that table is quoted in the
README. A first attempt at the default arm produced only `HTTP 401` responses because the API key did not reach the
tool; that attempt is not in the data (its 401 noise stays in the local run log) and `--out` was a scratch folder in
every run, so nothing was written into `bench/results/` until the two good arms were assembled by hand.

## Extra scripts

- `scripts/prompt_test.py` - one fixed code-like word-list prompt of about 11-12K tokens through
  `/v1/chat/completions` with `max_tokens`, printing the engine's own prompt and decode rates. Used for the matched
  11,103-token reads quoted in the README (three reads on the default route and one fresh read on the FP16 route, all
  in one server session; a re-read of the same text hits the prefix cache - 11,098 of 11,103 tokens reused, 5 read, in
  102 ms - so the FP16 arm contributes one fresh read and the matched pair is one fresh read per arm, not two
  independent samples). `STRATA_URL` and `STRATA_KEY` come from the environment.
- `scripts/rocblas_gemm_probe.cpp` - a `rocblas_gemm_ex` probe at `N = 10240 x T = 1024 x K = 2560` (a shape the
  prompt path uses) in four types, each checked against a CPU reference. Build and run:

  ```text
  hipcc -x hip rocblas_gemm_probe.cpp -o probe -lrocblas -I/opt/rocm/include -L/opt/rocm/lib
  HSA_OVERRIDE_GFX_VERSION=10.3.0 ./probe
  ```

  `rocblas_gemm_ex` takes alpha and beta as the compute type (f32), not the input type: a `__half` alpha reads as a
  denormal, the output comes out about zero, and the kernel looks broken when it is not.

## Sanitizing

The home-directory paths of this machine are written as `/home/<user>` and `/media/<user>`, the machine's LAN address
as `<lan-ip>`, and the API key is not in any file here (the run config holds it; the scripts read it from the
environment via `STRATA_KEY`). The engine log excerpts are the startup lines only.

## Not measured on this machine

PCIe link width and speed (`lspci -vv` needs a password here), RAM speed and timings, GPU temperatures, power
limits and clock behaviour under load, the CPU governor, the model files' read throughput in isolation, and
`vm.swappiness` / GTT behaviour under memory pressure. The model's GGUF sits on a `fuseblk` (FUSE) mount on a SATA
SSD, which is not the usual setup for the expert streaming path and was not compared against a native mount.
