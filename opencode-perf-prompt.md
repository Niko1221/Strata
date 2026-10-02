# Task: Improve Strata decode/prefill performance on the RX 7900 GRE (gfx1100)

You are working in the Strata repository (`/home/jason/Projects/Strata`). The current
served configuration is `strata-coder-iq1_m.json` (engine log: `strata-coder-iq1_m.log`).
Observed throughput: ~750–800 t/s fresh prefill, ~36–40 t/s decode, decode expert-cache
hit rate 58–63%, KV block hit rate 98–99%, spec acceptance ~75% at `--spec 4`.

`OPTIMISATION.md` and `docs/AMD_HIP_PERFORMANCE.md` are verified research documents —
trust their measurements over any assumption you make yourself. Decode is **memory-bound**
(expert-cache misses), so cache-hit improvements dominate kernel-instruction wins.

## Steps (do them in order, stay bounded)

1. Read `OPTIMISATION.md`, `docs/AMD_HIP_PERFORMANCE.md`, `docs/DETAILS.md` (sections on
   `--adapt-every`, `--pcie-frac`, `--resident-cpu-experts`, `--prefill auto`), and
   `strata-coder-iq1_m.json`.

2. Apply this config change set to `strata-coder-iq1_m.json` (JSON `args` array):
   - `--vram-reserve-mib 1024` → `256` (more VRAM into `--expert-cache auto` slots)
   - `--adapt-every 0` → `200` (cache follows the conversation; swaps are RAM↔VRAM
     copies with `--resident-cpu-experts`, so they are cheap)
   - Remove `--pcie-frac 0` (restore the default non-zero share so the GPU fetches
     part of the misses over PCIe in parallel; note in a comment that reproducibility
     of exact tokens is given up)
   Do NOT change `--spec`, `--spec-min-p`, `--prefill`, `--pool-workers`, or the
   hipBLASLt tuning table path in this pass.

3. Do NOT touch `CMakeLists.txt` or any kernel source (OPT-1 needs a reviewed guard
   fix first — out of scope here).

4. Verify the JSON is still valid (`python3 -m json.tool strata-coder-iq1_m.json`).

5. Write a short summary of exactly which keys changed and why, referencing the doc
   lines that justify each change, into `docs/perf-notes/2026-10-01-cache-tuning.md`
   (create the directory).

## Hard rules

- Never use cancelled-request timing lines as throughput evidence.
- Do not run the GPU server or benchmarks; the host (not opencode) will run
  `tools/hip/bench_prefill.py` afterwards.
- Keep all edits surgical (`str_replace`-style), never rewrite whole files.
- If any step's premise is false on disk (e.g. a flag is already set), stop that step,
  record the discrepancy in the notes file, and continue with the rest.
