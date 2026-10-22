# Community KLD report: route tail-skip on the NVFP4-QAD pack

Measured on 2026-10-22 by jerry78424. What was tested: the per-position cost of `STRATA_ROUTE_TAIL_SKIP`
(7/8/9 vs 0) on the **NVFP4-QAD** pack (`nvidia/Qwen3.8-Flash-Next-FP4-Local`, one of the FP4 cards in
`setup.py`'s `MODELS`) with the committed teacher-forced
yardstick (`tools/quant_quality.py`). The engine's NVFP4 guard (`src/program/generate.cpp`, "off by
default with NVFP4 experts") rests on a teacher-forced KL of 0.035 measured on the GPTQ fork pack;
the repo carries no measurement of the skip on the QAD pack. **The guard holds**: rank 7 moves the
answer distribution by mean KL 0.054 with top-1 agreement down 8.3 pt against an exactly-zero A/A
floor; even rank 9 (the mildest) loses 5.9 pt.

## Hardware and software

- GPU: RTX 5090 32 GB (sm_120); CPU: Ryzen 7 9800X3D; RAM: 187 GiB; Windows 11 10.0.26200; driver 617.42; CUDA 13.4
- Storage and PCIe link width: not recorded (the arms miss experts over PCIe per `engine-log-excerpt.txt`, but the link was not measured)
- Engine: `strata 0.1.46.1` (based on `9ae93f7b`), a source build, stamped per arm into every `out-*.json` identity
- No background workloads (the arms ran on a quiet GPU inside a maintenance window; GPU verified free between arms)

## Model and configuration

- Model: `nvidia/Qwen3.8-Flash-Next-FP4-Local` (NVFP4-QAD), prepared pack + `mtp/rt` + `ple-nvfp4.gguf`
  (Hub revision not recorded); MTP draft layer bound (`--spec 5`) and computed in every window
  (`mtp.prefill`) — the teacher-forced pass commits the given tokens, so the drafter proposes nothing,
  but its rows are part of the windows the guard scans, identical across arms
- Expert cache: `--expert-cache auto` — the engine's own start-up sizing decided **8930 slots** on this card
  (see `engine-log-excerpt.txt`), pre-filled fully from the committed profile
  (`--expert-profile data/expert-profile.bin`)
- Run window: `--max-context 2048` (seg+512), fixed by the yardstick
- Yardstick-fixed (quant_quality `FIXED`/`FIXED_ENV`): `--adapt-every 100000` (tier frozen), `--prompt-cache 0`,
  `--suffix-draft 0`, `--kv fp16`, `--prefill 512`, `--short-read` seg+64 (every position scores through the
  decode windows), `STRATA_PREFILL_CPU_SHARE=0`, `STRATA_ROUTE_TAIL_SKIP=0` for all arms except the arm's own
   `--env`; the driver passes the config's `--pcie-frac 0.35` through (the yardstick's 0 would not
   disable the skip — its scan runs before the PCIe/CPU placement decision in `src/core/expert_source.cpp`
   — but would compute every remaining miss on the CPU, outside the streaming regime the guard exists
   for). That the skip fired: `skips-A*.txt` (1.7 M experts at rank 7); that streaming was live: the
   `read by the GPU over PCIe` lines in `engine-log-excerpt.txt`)
- The engine command line as actually received (paths as placeholders): `reproduce/config.json`

## Method

Teacher-forced, like the quant-quality table: the fixed id set is read through the decode windows with
`STRATA_LOGPOS`/`STRATA_LOGPOS_TOPK=64`, one engine child per arm, KL/top-1/ppl from
`quant_quality.read_run/compare` over the reference's renormalized top-64 sets. Driver and commands:
`reproduce/README.md` (`ab_kld.py` reuses the yardstick's `engine_args`/`FIXED`/`read_run`/`compare`
plus a `--test-flag` for the PCIe share and the arm identity; the per-arm RTS rides the yardstick's
own `--env`).

- Corpus: 3 x 1536 evenly spaced slices per file over ten corpus files — nine public NVIDIA documents
  (Ada and Blackwell architecture whitepapers, consumer and PRO; the DLSS 4 announcement article, the
  DLSS 5 report, the DLSS SR/FG/RR guides) plus `dlss-prompt3.txt`, a page-marked concatenation of six
  of them, so the DLSS guides and Blackwell whitepapers carry double weight (the robustness check under
  Results removes exactly this duplication) — = 30 segments,
  **46,050 positions** (each segment's first token is not scoreable), tokenized with the pack tokenizer.
  `reproduce/dlss.ids.json` is committed (corpus hash `f553e2c3`) and is the authority for re-running;
  `reproduce/corpus.json` carries the per-file sha256 prefixes (`dlss-prompt3.txt` is locally derived —
  its sha256 pins the exact page-marker formatting, and a full corpus rebuild is best-effort because
   PDF-to-text conversion varies). A memorization scan preceded the matrix (`reproduce/corpus.json` `entropy_dead99_pct`: per-file dead-99 %
  8.3-24.4 %, no recitation), so the positions are live enough to judge; the dead positions stay in and dilute
  every arm equally.
- Arms: A0 (reference), A0b (A/A rerun — the floor is measured, not assumed), A7, A8, A9.
- Smoke first: one arm, two segments, positions and skip lines checked before the full matrix.

## Results

| arm | RTS | mean KL | p99 KL | top-1 agree | ppl | missed experts skipped |
|---|---|---|---|---|---|---|
| A0b | 0 | 0.000000 | 0.000 | 100.00% | 6.570 | 0 |
| A7 | 7 | 0.054316 | 0.789 | 91.71% | 6.412 | 1,682,818 |
| A8 | 8 | 0.039486 | 0.536 | 92.80% | 6.525 | 1,141,890 |
| A9 | 9 | 0.027325 | 0.366 | 94.07% | 6.479 | 578,302 |

All arms exit 0, all scored all 46,050 positions (`out-*.json`); `kld-matrix.json` is the `compare`
output (every arm carries the same stamped identity - build/corpus/topk/context; the identity's
`context` shows `262144` - the driver as it ran that day stamped it from the arm's input config,
which this report does not ship; the committed driver stamps the `--max-context` the engine actually
received, 2048). The A/A floor is
exactly zero — the teacher-forced pass reproduced byte-identically on this machine, so every arm
number sits above a floor of 0.

Reading: the guard's own bar (KL 0.035, and that figure's source is the GPTQ pack) is exceeded at
ranks 7 and 8, and the profile is spiky (p99 up to 0.79, an order of magnitude above the mean - not
uniform drift). Top-1 agreement drops 5.9-8.3 pt at every rank. ppl does **not**
degrade (6.41-6.53 vs 6.57): on this corpus the skip reads as mild denoising of the tail experts —
the damage signal is the argmax divergence and the p99 spikes. Conclusion from this machine's side:
the NVFP4 default-off guard is correct for the QAD pack too; `STRATA_ROUTE_TAIL_SKIP=7` stays an
opt-in.

Robustness (recomputed from the same per-position tsvs, no re-run): dropping the three
`dlss-prompt3` segments - the six-document concatenation - leaves 27 segments / 41,445 positions and
moves nothing that matters: rank 7 KL 0.055112 (p99 0.825, top-1 91.78 %), rank 8 KL 0.040059 (0.560,
92.81 %), rank 9 KL 0.027191 (0.370, 94.12 %), A/A floor still exactly 0; the subset's base ppl is
6.450 and the arms stay below it as above. The double weight of the six duplicated documents is not
what the conclusion rests on.

## Correctness and limitations

- The top-1 number is teacher-forced (the arm's greedy pick differs from the reference's on the same
  true prefix at that % of positions) — a per-step divergence proxy, not a free-run acceptance rate.
  Free-run acceptance, decode tok/s and the speed half were **not measured** (speed was to be measured
  only for a rank that survived the quality gate; none did).
- One machine, one corpus (NVIDIA technical English), one engine build; cold prompt cache per arm by
  the yardstick's design. Single run per arm — legitimate here because the A/A rerun reproduced
  byte-identically (floor exactly 0); run-to-run determinism is empirical on this machine, not a code
  guarantee (the engine documents near-tie flips from float ordering).
- The 8.3-24.4 % dead-99 % positions are included; a per-segment cut (drop positions where the
  reference scores p>0.99) is the follow-up if a rank ever lands near the bar.
- The ~46 MB per-position tsvs are not committed (regenerable; `reproduce/README.md`).
