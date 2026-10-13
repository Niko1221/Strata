# Suffix-Reuse: KV Cache Splicing for Edited Contexts

**Design and benchmark document** for the `--cache-suffix-reuse` feature and its adaptive-cadence extension (`--prompt-cache-dense-tail`). Covers branch `sfx-043-fixed` (all-quant splice, base v0.1.43) plus `sfx-future` @ `fd64fdf6` (dense-tail cadence, partial-prefix engagement, per-request checkpoint gate — 11 commits, review-clean).
All numbers measured on: RTX 3090 (24 GB), Qwen3.8-Flash-Next abliterated GSQ-RCO IQ3_XXS (2-file pack + PLE), 131k context, MTP spec-4, unless noted. Probes deterministic (temperature 0).

---

## 1. Summary

Strata already reuses the KV cache when a new prompt *extends* the previous one — the append case. It reuses nothing when the prompt is *edited in the middle*: one changed token invalidates the entire cache, and the whole prompt is re-read from scratch. This PR closes that gap. When a prompt repeats an earlier one with a change in the middle, the engine finds how much of the old context still matches, keeps that KV, and rotates the cached K rows to their new positions (V rows are position-free and pass through untouched). On a 5.7k-token ledger prompt with a one-row mid-file edit, reuse goes from **0% to 89%** and prefill from **10.4 s to 3.4 s**, with answers provably correct. The splice works with **every KV format the engine supports** — fp16, int8, int8+STRATA_KV_ROT, k8v4, q4_0 — because the rotation is applied in dequantized space and re-encoded with each format's own writer math. The feature is opt-in (`--cache-suffix-reuse N`), fails closed to today's exact re-read behavior, and the default path is unchanged.

## 2. Background: what Strata caches today, and the gap

Upstream v0.1.43 ships two reuse mechanisms:

- **Conversation checkpoints** (`--prompt-cache N`, `--prompt-cache-every N`): the engine snapshots the KV at turn boundaries.
- **Conversation parking** (`--conversation-cache-mib N`): finished conversations' KV is kept in RAM so a later message in the same chat can mount it.

Both are **prefix-mounted**: the cached KV is valid only while the new prompt is the old prompt plus new text at the end. Measured on the probe workload:

- Append turn (flag off): `prompt 3294 tokens = 3240 reused + 54 read` — **98.3% reuse**, 2.1 s.
- Mid-edit turn (flag off): `prompt 3294 tokens = 0 reused + 3294 read` — **0% reuse**, 10.4 s. The engine re-reads text it read seconds ago, because after the edit the cached sequence is no longer a prefix of the new one.

## 3. Why this matters: agents that edit their context

The gap is not an edge case — it is the *dominant* pattern for agentic and context-management workloads:

- **Agent loops with external memory.** An agent that keeps verbatim working notes (files re-sent each turn — the "context-level management" / CLM pattern) rewrites those notes constantly. Every rewrite is a mid-prompt edit.
- **Tool harnesses.** A file listing, a code window, or a scratchpad embedded in the prompt changes between calls while the surrounding system prompt and history stay identical.
- **Observed, not hypothesized.** In a 68-turn benchmark session against an engine with only prefix reuse, **41 of 66 requests (62%) got 0% reuse** — each note rewrite triggered a full re-prefill. The same session's pure appends got 97–98%. (kv-store trace, seed 42; 45.5% aggregate reuse despite the appends.)

Recent serving literature identifies the same pattern. CacheSlide (FAST '26) names it *Relative-Position-Dependent Caching*: agent prompts contain segments that keep their relative order while their absolute positions shift, and reports 3.1–4.3× latency reductions when serving reuse for them. vLLM's reuse RFC explicitly names absolute-position mismatch as the reason mid-prompt reuse is skipped. llama.cpp ships `--cache-reuse` (KV shifting) for exactly this shape, off by default. Strata's answer is below.

## 4. How it works

Given the old prompt O and the new prompt N with a common prefix of length P (the edit at P):

1. **Partial-prefix mount.** The engine matches N against parked conversations and mounted checkpoints, finds the deepest usable cached point ≤ P, and mounts it (`narrowed_resume_target`).
2. **The splice.** The unchanged tail [P, n) is cached at old positions; the K rows are rotated in place by the RoPE delta (R(p) → R(p+δ)) so they describe the *new* sequence. V rows are never touched — V is added post-attention and is position-free. Recurrence states (GDN/PLE) are *not* rotated forward from a parked state (that produces stale, wrong answers — we measured it); they are replayed from the narrowed checkpoint.
3. **Quantized formats: the dequant → rotate → requant sandwich.** int8, k8v4 and q4_0 K rows are quantized codes, not floats. The splice dequantizes with each format's own dequant math, rotates in fp32, and re-encodes with the *writer's exact* quantizer — one requant, the same noise structure as a fresh write.
   - int8: codes + scales, per-64 groups.
   - int8 + STRATA_KV_ROT=1: K is stored Hadamard-rotated (H·R(p)·k). Because H·H = I, the splice runs dequant → H → R(δ) → H → requant, which equals the exact stored form at the new position: H·R(δ)·H·(H·R(p)·k) = H·R(p+δ)·k.
   - k8v4: the hybrid's K half is byte-identical int8 storage (same codes/scales at the same offsets); V (rotated Q4_0) is untouched.
   - q4_0: whole K rows are Q4_0 blocks (18 B per 32 values); dequant/requant per block with the parity-reference quantizer.
4. **Fail-closed.** Unsupported memory layouts (`--kv-resident > 0`, `--layer-split`) refuse with a logged reason and fall through to the re-read path. Off (the default), behavior is byte-identical to today (pinned by `suffix_reuse_off_parity_test`).
5. **Adaptive cadence (`--prompt-cache-dense-tail N`, `sfx-future`).** Checkpoints are taken sparsely through the head of the prompt and every `--prompt-cache-every` tokens within the last N tokens — cheap cold read, dense anchors exactly where mid-text edits land. The periodic gate is per-request (it reads the request's own `pp_next_check`, not a value persisting across requests, so a warm read banks anchors at its piece boundaries). The engagement rule prefers a partial mount only when its LCP is at least the strict match's depth AND its anchor is deeper (`p >= s` *and* `L > s`) — enforced across conversations (a different conversation's deeper-but-shallower-LCP partial never mounts; pinned by T16.7/T16.7b). When an edit sits deeper than every checkpoint, the engine fail-closes: it restores the deepest checkpoint at or before the edit and reads forward (§5.8), rather than splicing recurrence state across the edit.

## 5. Benchmarks

### 5.1 Before / after: the mid-edit turn (int8, the daily format)

Three-turn ledger probe: a 160-row ledger (≈5.7k tokens), one row inserted at row 99 (edit), a second insert (edit), temperature 0, exact-count scoring.

| Turn | Before (flag off) | After (`--cache-suffix-reuse 64`) |
|---|---|---|
| A: original prompt | 0/5670 reused, 10.4 s | 0/5670 reused (cold), 10.4 s |
| B: insert one row → answer 161 | **0/5702 reused, 10.4 s full re-read** | **3072/5702 reused (54%), 20.2 s** |
| C: insert second row → answer 162 | **0/5736 reused, 10.4 s full re-read** | **5120/5736 reused (89.3%), 3.4 s** |

Answers: 160 / 161 / 162 all correct, including double-edit and revert discrimination (revert correctly answers 160 again). Engine log line, turn C: `prompt 5736 tokens = 5120 reused + 616 read in 3439 ms`.

### 5.2 All quant formats: steady-state and splice-path proof

| Format | Splice fired (trace) | Edit answer | Steady-state chain (3 turns) |
|---|---|---|---|
| fp16 | ✓ | correct | (Phase-2 measurement: 24/24 correctness) |
| int8 | ✓ | 161 ✓ | GREEN — 89% reuse on turn C |
| int8 + STRATA_KV_ROT=1 | ✓ (`fired=1`, `status=1`) | 161 ✓ | — |
| k8v4 | ✓ (`fired=1`, `status=1`) | 161 ✓ | — |
| q4_0 | ✓ (`fired=1`, `status=1`) | 161 ✓ | **GREEN — A 45.2 s cold; B 54%; C 5120/5736 (89.3%), 3.4 s, 162 ✓** |

q4_0's full chain matches int8's GREEN shape exactly (validator: `sfx_steady_state_q4.py`).

### 5.3 Numerical accuracy of the splice (unit tests, CPU)

The double-quant noise (first write + splice requant) measured against a fresh write, in code steps:

- int8 unrotated: **1.17**
- int8 + KV_ROT (H-sandwich): **1.27** — within 10% of the accepted unrotated path; bitwise-identical to an independent sandwich-of-stored reference.
- q4_0: **1.53 q4-scale units** — coarser codec (16 levels vs int8's 254), same ballpark.

Cells outside the edit range are byte-identical after the splice (all formats).

### 5.4 Long-context generation parity (old vs new engine)

Same seed (42), temperature 0, ~8k-token prompt with 12 seeded facts at spaced depths: exact recall + arithmetic + a 250-word generation.

| | Strata .40.3 (base) | .43 + suffix-reuse |
|---|---|---|
| Recall + arithmetic | 10/10 (2 runs) | 10/10 (2 runs) |
| Long generation | 1973 chars, natural stop | 1972 chars, natural stop |
| Turn-2 append generation | correct | correct (served 98% from cache) |

### 5.5 CLM on ContextBench: retention and reuse (assembled from the benchmark campaigns)

**Retention: CLM vs traditional summarization** (P3 final, matched instances, seeds 42+43, prefix-only engine — the splice does not affect this comparison):

| Arm (context mgmt) | Total (4 families) | Exact-value queries | kv-store | needle (multi-seed 42/43/44) |
|---|---|---|---|---|
| **CLM** (verbatim offload) | **292/429 = 0.681** | **full preservation** | **0.875 mean** (24/24, 18/24) | 1.000 / 0.750 / 0.833 |
| Floor (raw context) | 253/429 = 0.590 | partial | 0.312 | 0.675 mean (P2) |
| **Summarization** | 122/429 = 0.284 | **0/96 — absolute zero** | **0.000** | 0.295 mean (P2) |

Summarization destroys exact values (0/96 recoveries) — every ID, count and token the compressor touches is gone. CLM preserves them by never compressing: the agent re-injects its notes verbatim. That verbatim re-injection is precisely the mid-prompt-edit workload the splice serves.

**Reuse: before the splice (prefix-only)** — two complementary measurements:
- *Usage trace* (kv-store seed 42, 68 turns): 45.5% aggregate, but split 25 pure-append requests at 97–98% vs **41 edit requests at 0%** — every CLM note-rewrite paid a full re-prefill.
- *Campaign profile* (P0b, 4 families, 248 requests, 14,336 budget): needle 1.000 (96/96) at **16.1% reuse**, sudoku 1.000 (15/15) at 57.6%, kv 1.000 (24/24) at 43.1%, log-triage 0.583 (14/24) at 21.5%; total 28.9% (491,299/1,700,229 tokens), 2,378 s prefill saved. Reuse tracks workload shape: append-heavy families reuse; insert/edit families (needle's mid-file inserts, log's rolling edits) are the starved ones.

**After the splice (fixed engine, .43 + suffix-reuse):** the engine-level edit-turn measurement is §5.1/§5.2 — a mid-prompt edit now reuses 89% (5120/5736) instead of 0%, at 3.4 s instead of 10.4 s, answers provably correct on every KV format. The ContextBench families whose reuse was starved (needle 16.1%, log 21.5%) are exactly the edit-shaped workloads this converts. A full 4-family re-run on the fixed engine with the dense-tail configuration exists and is §5.8: reuse rose exactly where predicted (log 21.5% → 77.2%, kv 43.1% → 72.3%), with the remaining failure mode (needle's fail-close gap-read) measured and decomposed there rather than hidden.

### 5.6 The honest cost: checkpoint granularity

Reuse requires a checkpoint at or before the edit position. With `--prefill auto` (one big chunk) there are no mid-prompt checkpoints and mid-edit reuse does not fire (appends still reuse). The proven configuration is `--prefill 1024 --prompt-cache-every 1024`, which costs cold-read throughput: 46.6 s vs 10.9 s for a cold 5.7k prompt (4.3×) on this GPU. `--prefill 2048` halves the checkpoint density and recovers most of the cold-read cost. This trade-off is why the feature is opt-in and the knob is exposed.

### 5.7 Dense-tail cadence: the trade-off dissolved (`sfx-future`)

`--prompt-cache-dense-tail 4096` with `--prefill auto`: sparse checkpoints through the head, dense through the last 4096 tokens. Same three-turn ledger probe, same seed, dense-tail stack (T21 per-request gate):

| Turn | Edit-path config (§5.1: prefill 1024 + every 1024) | Dense-tail config (prefill auto + dense-tail 4096) |
|---|---|---|
| A: cold read | 46.6 s | **42.4 s** |
| B: append → 161 ✓ | 20.2 s (54%) | 27.9 s |
| C: edit → 162 ✓ | 3.4 s (89.3%) | **10.1 s (81.0% — 4646/5736)** |

All answers correct (160/161/162). The C-turn anchor lands on a dense-tail checkpoint at 4646 (just below the second edit at 5230) — strict-mount, no splice needed. Checkpoint cost at existing boundaries is ~100–150 ms per save (measured T19: pipeline sync 0.0 ms, state save 100–150 ms); the per-boundary ~8 s cost of the §5.6 config is the *pipeline restart between chunks*, which dense-tail's single-chunk head avoids.

### 5.8 The remaining gap — and the honest failure mode

When an edit lands deeper than every checkpoint, the engine fail-closes: restore the deepest checkpoint at or before the edit position, read the rest fresh (826 tokens in the §5.7 probe). Correctness is never traded. The cost is bounded but real, and it is the dominant residual inefficiency. A 4-family ContextBench campaign on the dense-tail config (seed 42, budget 14,336, CLM-skill agent) measured where this bites:

| Family | base .40.3 (reward / reuse) | dense-tail stack (reward / reuse) |
|---|---|---|
| needle | 1.000 / 16.1% | 0.500 / 17.0% |
| sudoku | 1.000 / 57.6% | 0.867 / 73.5% |
| kv-store | 1.000 / 43.1% | **1.000 / 72.3%** |
| log-triage | 0.583 / 21.5% | **0.625 / 77.2%** |

Two findings. First, the reuse gains are real and large (kv +29.2 pp, log +55.7 pp) and scores on those families rose, not fell — mounted cache answers exactly as fresh-prefilled text does. Second, needle fell to 0.500 — and the decomposition isolates why it is not the code: the same binary with the §5.1 config scores 0.750 on needle, identical to the flags-off arm's 0.750 (both within the seed band of the multi-seed needle mean 0.847 ± 0.171); only the dense-tail *configuration* drops it to 0.500, and needle's reuse was unchanged (16.1% → 17.0%) — the failure is not cache corruption but the fail-close gap-read colliding with that seed's unbounded-growth agent trajectory (23 messages, no compaction).

### 5.9 FW-2b — state-cadence checkpoints close the gap (2026-10-13)

`--prompt-cache-state-every N` (FW-2b) adds a side pool of *state-only* checkpoints (the recurrence state `checkpoint_save` already copies for periodic anchors — gdn/ple/tails/dead/block_pos host vectors, no K/V) at a fine cadence, and the splice-narrowed restore picks the deepest anchor ≤ P across **both** pools. Fail-close semantics are unchanged — the gap [anchor, P) is still read through the normal chunk loop; the anchor just sits closer to the edit. 0 = off = byte-identical.

Same binary, same seed 42, same dt configuration, only the flag differs:

| arm | reward | reuse | wall | prefill ms/token |
|---|---|---|---|---|
| dt + state-every 256 | **0.833** (80/96) | **25.2%** | 1252 s | 3.66 |
| dt (control) | 0.750 (72/96) | 0.0% | 992 s | 4.02 |

Needle recovers into the seed band (0.847 ± 0.171) and per-token prefill *drops* 9% despite a longer trajectory. The control's per-request table shows the mechanism: all 17 of its mid-file rewrites found no anchor (0% reuse, ~44 s full prefill each); with the state ladder, the compaction-phase tail-preserving rewrites mount at 38–66% (9 of 26 requests; the growing phase and head-rewrites legitimately have nothing to mount).

Extended to seeds 43/44 (same protocol), the reward margin does **not** reproduce — what holds is the reuse and stability effect:

| seeds 42/43/44 | reward (mean ± sd) | reuse (mean ± sd) |
|---|---|---|
| dt + state-every 256 | 0.792 ± 0.042 | **15.5% ± 10.6** |
| dt (control) | 0.792 ± 0.072 | 7.2% ± 7.6 |

Every run of both arms sits inside the multi-seed needle band; the feature is reward-neutral across seeds while roughly **doubling mean reuse** (2.1×) and halving reward variance.

Re-certifying the other three families on the full FW-2b stack (dt + state-every 256, seed 42) gives the campaign's strongest table:

| family | base .40.3 | dt (P0e) | **dt + state-256 (FW-2b)** |
|---|---|---|---|
| needle | 1.000 / 16.1% | 0.500 / 17.0% | 0.792 ± 0.042 / 15.5% (3 seeds) |
| sudoku | 1.000 / 57.6% | 0.867 / 73.5% | **1.000 / 68.9%** |
| kv-store | 1.000 / 43.1% | 1.000 / 72.3% | **1.000 / 71.8%** |
| log-triage | 0.583 / 21.5% | 0.625 / 77.2% | **0.625 / 69.9%** |

Every family at or above its base score, reuse 1.2–4.3× base, and sudoku's single-seed dip is gone. Read FW-2b as a reuse/latency optimization with a safety profile: the 0.500 failure mode of §5.8's single-seed dt run never appears (arm minimum 0.750), and the seed-42 draw where reuse visibly rescued reward (0.750 → 0.833) illustrates the upside when a trajectory's rewrites land where only the state ladder has anchors. Steady-state chain (dt-steady10/10b): answers 160/161/162, C-turn reuse 81.0% @ 10.8 s, cold and append turns identical to pre-FW-2b — the side pool taxes nothing (captures ~105 ms each, host-RAM pool capped at 2 GiB ≈ 10 anchors at needle scale).

## 6. Purpose in context management (CLM) terms

Verbatim-offload context management — the agent keeps its working state in files and re-injects them every turn — is the strongest known defense against context decay (our measurements: verbatim offload preserves exact values that summarization destroys; 0/96 exact-value recoveries after summarization vs full preservation offloaded). But it converts every working-state update into a mid-prompt edit, which until now meant a full re-prefill per update. The splice makes the CLM pattern affordable at the engine level: the agent rewrites its notes, and the engine re-uses everything up to the rewrite. The 41-of-66 zero-reuse trace (§3) is a CLM session; the same shape with the splice is §5.1.

**Prior-session usage evidence.** Three independent benchmark campaigns (P2 multi-seed, P3 matched 4-family, P0b replication, seeds 42–44) against prefix-only reuse consistently show the same usage shape: a CLM agent's live context grows 17k → 41k chars over 68 turns as notes accumulate, 25 of 66 requests (the pure appends) ride 97–98% reuse while 41 rewrites each pay a full re-prefill — the engine spends most of its busy time re-reading text the agent already had. Task scores themselves were unaffected by cache state (kv-store 24/24 → 18/24 across P3/P0b runs with identical configs; needle 1.000/0.750/0.833 across seeds vs prefix-only mean 0.847 ± 0.171), which is expected — cache changes compute cost, not token visibility — and is the evidence that the splice's win is purely time-to-answer, with no accuracy tax to hide. The exact-value stakes are documented across sessions too: verbatim offload preserved every exact value where summarization recovered 0/96 — the pattern the splice makes affordable at engine level.

## 7. Relationship to prior work

- **llama.cpp `--cache-reuse` / KV cache shifting** — production precedent for position-shifted cache reuse on quantized caches (issue #18497's logs show chunk reuse with KV shifting on q8_0 caches; off by default; #3825 documents the shift machinery's history). Same edit shape, same off-by-default posture. Strata's splice differs in the quantized path: dequant → rotate → requant with each format's writer math (incl. the Hadamard-domain sandwich), and cross-request conversation checkpoints rather than per-slot reuse.
- **CacheSlide (FAST '26)** — identifies Relative-Position-Dependent Caching in agent workflows and solves it with position-insensitive encodings and weighted correction attention (3.1–4.3× latency reduction). Strata's splice is the exact-arithmetic version for RoPE models: rotate the kept K by the position delta rather than making positions insensitive.
- **Adaptive KV Cache Reuse (arXiv 2605.24022)** — survey framing reuse across request shapes; positions CacheSlide and prefix/PIC approaches.
- **PBKV (arXiv 2605.06472)** — prediction-driven cache management for dynamic agent workflows.
- **vLLM reuse RFC (#2614)** — names absolute-position mismatch as the blocker for mid-prompt reuse and scopes reuse to prefixes; the problem statement this PR answers on different machinery.

## 8. Configuration

Two new flags; the rest is existing upstream machinery:

```
--cache-suffix-reuse 64          # the feature (opt-in; 0 = off, default)
--prompt-cache-state-every 256   # FW-2b: fine-cadence state-only checkpoints (0 = off = prior behavior)
--prompt-cache-dense-tail 4096   # adaptive cadence (opt-in; 0 = off = uniform cadence)
--conversation-cache-mib 4096    # upstream flag; default 0 = parking off (needed: nothing to splice from without it)
--prompt-cache 6                 # upstream default (>= 2 required by the validator)
--prefill 1024 --prompt-cache-every 1024   # uniform checkpoint density (the §5.1 edit-path config; not needed with dense-tail)
```

Refused (logged, falls back to full re-read, never crashes): `--kv-resident > 0`, `--layer-split`, rotated formats with head_dim ≠ 256 (fwht256 is order-256).

## 9. Verification summary

- CPU ctest **53/53** (39 upstream + 14 suffix-reuse), re-run after every commit.
- Suffix-reuse suites: partial-prefix **74/0** (incl. T16.7 cross-conversation pin), dense-tail **23/23**.
- CUDA (sm_86, toolkit 13.0) builds clean; `-Wall -Wextra` single-header syntax check clean.
- Live GPU probes: splice fired and answers correct on every format (§5.2); dense-tail chain GREEN with all answers correct (§5.7).
- Independent fail-closed code reviews: all-quant splice **APPROVE** (3 MINOR doc findings, closed); `sfx-future` **r1 REJECT → fix → r2 APPROVE** (1 MAJOR cross-conversation dominance bug found and fixed TDD-style; zero new findings at r2).
- Not tested: HIP compile (no AMD hardware available to the author — the new code is host-side C++ behind the existing transport layer; an AMD build pass is requested from the community/maintainers).

## 10. Limitations and future work

- The fail-close gap-read (§5.8) leaves bounded re-read when an edit lands deeper than every checkpoint. Next milestone: state-cadence checkpoints plus a replay path (restore the deepest state checkpoint at or before the edit, prefill forward — attention over the gap reads valid arena K/V). The env gate `STRATA_SFX_REPLAY` and the `partial_gap_read_from` seam exist and are tested; the state-cadence banking is not built yet. Projected anchors: 59.7% (parked) / 91.2% (session-local) on the §5.7 probe shape.
- Streaming-ring (`--kv-resident`) and layer-split layouts remain refused — splicing a ring requires cell re-homing, not just rotation.
- Recurrence state (GDN/PLE) is replayed from the narrowed checkpoint rather than transformed; a forward-state transform would cut the remaining gap-read further.

## References

1. CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving — Liu et al., USENIX FAST '26. https://www.usenix.org/system/files/fast26-liu-yang.pdf
2. llama.cpp issue #18497 — cache-reuse effectiveness on quantized caches (KV-shift logs). https://github.com/ggml-org/llama.cpp/issues/18497
3. llama.cpp issue #3825 — `llama_kv_cache_seq_shift` position-delta semantics. https://github.com/ggml-org/llama.cpp/issues/3825
4. llama.cpp discussion #8860 — prefix cache reuse semantics (maintainer confirmation). https://github.com/ggml-org/llama.cpp/discussions/8860
5. Adaptive KV Cache Reuse for Fast Long-Context LLM Serving — arXiv:2605.24022. https://arxiv.org/abs/2605.24022
6. Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management (PBKV) — arXiv:2605.06472. https://arxiv.org/abs/2605.06472
7. vLLM prompt-cache reuse RFC — absolute-position mismatch discussion. https://github.com/vllm-project/vllm/issues/2614
