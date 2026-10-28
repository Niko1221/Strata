# Doubling decode on the M1 Max, quant frozen — chunked plan

Engineering plan, kept beside the code it changes. Written 2026-10-28 after a full
forensics session (evidence paths below; everything here was measured on this machine).

## The deal

Decode of the huihui UD-Q4_K_XL pack serves ~21 tok/s (essays; 29–31 on good
draft streaks) on the 64 GB M1 Max. The pack stays exactly as it is — no requant, no
format change — so every win must come from *time*, never from *bytes*.
Goal: ~2× (38–45 tok/s). Realistic floor of the plan as written: 1.5–1.7×.

## The budget (measured, T1 window, 33.7 ms/token wall)

| phase | ms/token | rate now | ceiling evidence |
|---|---|---|---|
| expert gate/up (moe_gu 8,1,2 Q4_K, 47×) | 5.0 | ~110 GB/s live / 181 bench | pure read 310–358 GB/s |
| expert down (moe_down 4,1,4 Q5_1, 43×) | 4.3 | ~112 live / 233 bench | **live-vs-bench 2× anomaly, unexplained** |
| dense gemv (qkv 180, hc ~200, output 243 GB/s) | ~19.5 | mixed | sweeps hit 233 for these shapes |
| ssm_out / attn_gate / router gemv | 2.0 / 1.9 / 1.9 | 65 / 69 / 37 GB/s | shapes sweep at 150–230 |
| norms, steps, route, misc | ~3 | fine | — |
| drafts + sampling (outside the window) | ~5 | — | overlap target ~2 |

Fixed facts: DRAM read ceiling 358 GB/s (rt_bench, 2026-10-28); scattered 3 MB arena
slots read at 309–317 GB/s (rt_bench scatter); no packed format decodes faster than
~250 Gval/s on Apple GPUs (IQ4_XS 259, Q4_K 259, IQ2/IQ3 table formats stall at
150–260 Gval/s but never beat it; expert_parity benches, `/tmp/bw` and session notes);
T2 verify window costs 1.6× T1 (weights re-decode per token); spec depth 3/4 measured
no gain at today's kernel speeds (draft accept ~65–75 %); MOE_DN2 no gain; HC_Q8
defaults ON; tiles gu 8,1,2 / dn 4,1,4 confirmed optimal on Q4_K/Q5_1 (23-config sweep).

## Chunk C-EDGE — riding to upstream's bleeding edge (one session; priority C0)

**DONE 2026-10-28** (`fc5f7097`, tag `pre-edge-20261028` = fallback). All 51 commits kept;
engine now 0.1.52-metal. 20-file merge resolved: code = ours for the Mac engine +
theirs for upstream's work, combined by hand where both edited one function
(`park_source`'s `ParkSource` re-took our `max_cells > 0` drafter gates at its four
construction sites; `verify.cpp` kept upstream's tap + our `out` arg; `mtl_rt.mm`
keepalive/zero-copy came in under `STRATA_DS4_RUNTIME`). Docs stayed ours;
`serve/server.py` kept our answer-room rule (upstream's is DeepSeek-only).
Verification (single-engine windows): ctest 4/4; `gemv_parity`/`expert_parity` ALL
PASS; greedy answers IDENTICAL (content + reasoning, 2 prompts) against the pre-merge
server; needle 20k @ 50 % FOUND (18 942 tokens, 81 s); parking A/B follow-up tokens
IDENTICAL, restore 7.6 ms (the script's `reused > 0` and `SNAPSHOT_VERIFY` criteria
are CUDA-only, unmet on Metal by design); decode measured ~10 % FASTER than the
same-evening control (18.78 vs 21.09 s, 36.83 vs 39.96 s). Pitfalls found: the
repo's `.gitattributes` uses the mergiraf driver with no driver configured in
worktrees (silent ours-only "conflicts" — redo with `git merge-file`); parking and
the serve window MUST not overlap (two engines = swap storm = token-0 stall abort).

Facts measured 2026-10-28 (all in this repo's git):
- `upstream/main` (Niko1221/Strata) is 1071 commits newer than our merge-base
  (`011c9e60`, 2026-10-21); our branch `metal-live` is 51 commits ahead — those 51
  ARE the Mac engine (parking, cut-flag handshake, sampling, batch, M1 tiles,
  session port, host sampler) plus PRs #3333/#3615/#3301/#3527 merged locally.
- Upstream did NOT touch `window.cpp`; upstream grew a parallel DS4 Mac line
  (`metal/src/engine/ds4_*`). `git merge-tree` previews **20 conflicting files**:
  `generate.cpp`, `conversation_snapshot.cpp`, `conversation_state.cpp`,
  `conversation_cache.hpp/_test.cpp`, `conversation_snapshot.hpp`, `verify.hpp/cpp`,
  `prefill.cpp`, `mtl_rt.h/.mm`, `runtime.hpp/cpp`, `stubs.cpp`, `stubs_0145.cpp`,
  `metal/CMakeLists.txt`, `serve/server.py`, `.gitignore`, docs (DETAILS/MAC).
- The "needle lost at 17K+" fix (`845941f5`) is in `src/dsv41/ds4_kernels.cu`
  (DeepSeek CUDA) — it does NOT apply to the Mac engine. Our QSA path is clean:
  20 000-token needle at 50 % depth answered correctly on the live huihui server
  (2026-10-28, `tools/needle_bench.py`).
- Upstream Metal fixes worth reading into during the merge: `2a220c1d` (sparse_attn
  binding read stale rows — short prompts and follow-up turns answered differently
  every run), `8810e00c` (prompt hc_mix reads each row once: bit-identical, +4.5 %
  at 3K prompt), `7c77f388` (commit queued GPU work before the CPU lane), and
  `42890ca7` (stubs for the layer-major prefill kernels — REQUIRED, the Mac build
  shares `prefill.cpp`). Upstream `7a34c60d` (KVA keeps the draft layer off
  approximated cells) must be reconciled with our `c84cb517` max_cells > 0 gates.
- Engine version crosses 0.1.47-metal → 0.1.48 (upstream 0.2.3).

Steps (one session):
1. `git worktree add /tmp/edge metal-edge` (branch off `metal-live`); merge
   `upstream/main` there. Live tree and serving box untouched.
2. Resolve the 20 conflicts with intent: ours = the Mac engine (window.cpp line);
   theirs = DS4/KVA/prefill rework. Re-run the stub regeneration recipe
   (gen_stubs.py, union of link logs, last `-I*/include` wins — the
   stubs-regen skill) after the first failed link.
3. Full `ninja` (not just strata — new sources arrive), `ctest`, parity tools compile.
4. Serve-verification window (harness; end STARTED + health-checked):
   `tools/needle_bench.py` battery incl. >17k, `parking_test.py` + disk slot
   save/restore, byte-identity vs the saved w3/BASE answers, tok/s regression
   gate 20.7/23.0/22.2 ±2 %, one kprof run for class-comparability with C0.
5. Only then fast-forward `metal-live` to `metal-edge`; keep the pre-merge tag
   (`git tag pre-edge-20261028 metal-live`) as the fallback; settings live in
   the untracked config json + scripts and survive untouched.

## Chunks (each fits one regular session)

### C0 — Wildcard forensics (half session; env/knob only) — DO FIRST
The down kernel is the one class at 2× its bench time live (99 µs/layer vs 52 µs in
`expert_parity --gu Q4_K --dn Q5_1 --gucfg 8,1,2 --dncfg 4,1,4`), same tile, same
bytes. Suspects: kprof serialized-replay inflation; 33 GB arena residency/page
locality (bench arena is 12.6 GB); clock state under server load.
- Re-run live KPROF (`STRATA_METAL_KPROF=1`) with a warmed conversation; override dn
  tile via env; re-run `expert_parity --bench --layers 48` to grow the bench arena.
- Exit: per-class attribution of the live−bench delta; if a locality/tile fix exists,
  commit it (worth ~1.5–2 ms/token for free and it re-sizes C2/C4's payoff).

### C1 — Dense loser tiles (one session; small code)
ssm_out (65), attn_gate (69), router ffn_gate_inp (37 GB/s) run below what their
shapes sweep (gemv_parity results in session notes: Q4_K 2560×10240 → 233 @8,2,4;
Q8_0 512×2560 → 167 @8,1,4). Map each live shape to its swept tile in the gemv_tile
table (`metal/src/engine/window.cpp:127–152`), A/B with KPROF classes + tok/s.
- Exit: the three classes ≥130 GB/s live; expected −1.5…−2.5 ms/token; committed.

### C2 — fp16/bf16 hot-expert tier, design + bench spike (one session; bench code only)
The escape from the decode tax at frozen bytes: dequant the HOT experts once at
stream-in (idle CPU/NEON during the streaming copy) into a 16-bit shadow copy; decode
reads unpacked data (dot = 1 fma, no unpack). Not a requant — the pack is untouched.
- Decide fp16 vs bf16 (bf16 `Dec` is instantiated today; fp16 needs new ones; parity
  must hold with fp32 accumulators — 4-bit values fit both formats' mantissas).
- Sizing: fp16 expert = 7.9 MB vs 3.07 MB packed; top-N from the expert profile's
  frequency ranking, e.g. 1 500 experts ≈ 12 GB tier inside the 33 GB arena knob.
- Bench it in `expert_parity` first (bf16/fp16 gu+dn launch target ≤40/≤30 µs vs
  101/52 today). Exit: measured bench numbers + design note (tier layout, replacement,
  kernel variant selection) + parity numbers; committed bench/test only.

### C3 — Hot-tier engine integration (one to two sessions; the big one)
Shadow buffer in the arena, dequant-on-stage in the streaming path, per-launch kernel
variant by slot flag, arena accounting (env `STRATA_METAL_HOT16=<GB>`, opt-in;
default path byte-identical per repo rules). A/B with KPROF + tok/s + mactop + the
needle battery.
- Exit: live decode ≥ +20 % opt-in; committed.

### C4 — half2 unpack for the dense gemvs (one to two sessions; MSL surgery)
Apple fp16 vector rate is 2×: unpack to packed half and dot in half vectors with fp32
accumulators, across `Dec<>`/`unit_combine` (`metal/kernels/units.metal`) or a
gemv-specific path if that proves cleaner. All 48+ instantiations must stay parity
(mathMode Safe, explicit fma). C0's attribution decides how much is real.
- Exit: dense classes +25 % live, parity green; committed.

### C5 — Draft overlap + window depth re-tune (one session; engine)
Second command queue for MTP draft rounds (GPU sits at ~37 % utilisation); then
re-test `--spec 3` — deeper verify windows were losers at 33.7 ms/token but become
cheap once per-token window cost flattens (T2 slope drops after C3).
- Exit: draft overhead ≤3 ms; best spec depth recorded and set in the serve config.

### C6 — Combine, document, offer upstream (half session)
Full battery on the final engine; measured-numbers table in `docs/DETAILS.md`
(repo convention: numbers with what they were measured on); optional draft PR
(pattern: #4384, branch off upstream main).

## Order and stacking

C-EDGE (upstream rebase, section above) first: it re-numbers the ground under
every class timing, and it carries real Mac fixes. Then C0 → C1 (parallel-safe)
→ C2 → C3 → C4 → C5 → C6. C1 and C4 both touch dense: C1 is tiles (cheap,
first), C4 is unpack code (deep). C3 + C4 are independent and each is worth
~+20–30 %; C0 and C5 are the multipliers around them.

## Machine and harness notes for the next session

- Harness: `metal/tools/serve_huihui.sh {status|stop|start|run CMD}` (self-hosted
  box: this assistant runs on the huihui server, port 8180). ALWAYS leave the server
  started and health-checked (`/v1/models` → loaded, one live chat probe) — an
  experiment window ending with the server down is a bug in the script, not the box.
- Warm boot is ~4 s to loaded, ~11 s to first answer (page cache hot). Do NOT run
  other GPU work during timed A/B arms.
- Engine forensics: `STRATA_METAL_KPROF=1` (window.cpp, committed b06a9afd) prints
  per-class serialized GPU time once per token-class at the 20th occurrence.
- DRAM: `mactop --headless --format json --count 20 -i 1000` →
  `soc_metrics.dram_bw_combined_gbs` (powermetrics lost its DRAM samplers on macOS 27).
- No GNU `timeout` on this Mac. Scripts run `set -u` — define every loop variable.
- Benches: `build-metal/expert_parity --gu Q4_K --dn Q5_1 --bench [--sweep]
  [--gucfg L,R,S] [--dncfg L,R,S] [--layers N]`; `build-metal/gemv_parity --one ...`;
  `build-metal/rt_bench {bw|scatter|dispatch}`. Rebuild ~11 s: `cd build-metal &&
  ninja strata` (MSL files compile at engine start — kernel edits need no rebuild,
  only a server restart).
- Drive for live A/B: `/tmp/bw/drive.py "prompt" max_tokens` (three-request shape used
  by the control: hello / twelve-paragraph essay / three-paragraph convolutional
  codes; control at spec 2 = 20.7 / 23.0 / 22.2 tok/s, 2026-10-28).
- Evidence dir `/tmp/bw/` (w1–w4 scripts + logs, sweeps; ephemeral — move anything
  still needed into the repo or this doc).
- Fork rules: keep new paths opt-in, defaults byte-identical, titles prefixed
  (`[Metal] …`), never push the fork branch without permission.
