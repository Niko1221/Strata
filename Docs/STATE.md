# STATE

## Current phase
V100 Stage 1 — COMPLETE (verdict PASS). Awaiting upstream review of the PR.

## Current commit
`fa146c9` on branch `feature/v100-moe`
- pushed to `origin` = `https://github.com/noorazman/Strata` (own fork of `Niko1221/Strata`)
- open PR: **Niko1221/Strata#3** (base `main`, maintainer_can_edit=true)
- remotes in this repo: `origin`=noorazman/Strata (fork), `upstream`=Niko1221/Strata, `local-mirror`=/home/noorazman/dsh/strata/Strata (DO NOT commit there — user's pristine clone of upstream)

## Status
- Build: `build-sm70` green. in-tree tests 20/21; ctest 19/20 on V100 (`ple_parity` red by design — needs Q2_0 fixtures).
- First inference on GPU0 (32 GB): works, coherent `<think>` output, deterministic.
- Correctness vs llama.cpp (vanilla build `427291b5b`, same IQ3_XXS, greedy, raw prompt): 32/32 tokens identical (prompt 1); 31/31 then ULP branch (prompt 2) — consistent with upstream-declared imprecise hit path.
- GPU4 (16 GB): fits (peak 15.6 GiB, auto cache 6,321 slots), 27.3 tok/s, coherent.
- Benchmarks: prefill ~400 tok/s (2,047 tokens) both cards; sustained decode 37–40 tok/s (use `--pool-workers 28` for +9–13 %).
- API :8180 verified (deterministic, SSE streaming). Server stopped; box clean.
- All Stage 1 deliverables committed: `Docs/v100-{stage1-baseline,compatibility,build,testing,benchmarks,stage1-final}.md`.

## Blocker
None.

## Environment (authoritative facts)
- Machine: 2× Xeon E5-2680 v4 (28 physical/56 logical per socket, AVX2, no AVX-512), 125 GB RAM, /mnt/ssd ~172 GB free.
- GPU roles: GPU0 32 GB PCIE = dev (Strata). GPU4 16 GB SXM2 = validation. GPU2 = reference runs. GPU3 = agent LLM (llama-server :8001, don't touch). GPU1 :8080 Swift-27B (don't touch).
- Model (user-supplied, do not move): `/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` — IQ3_XXS, 2 shards, 70.74 GiB; arch `qwen4exp`; PLE table 320,001,536 rows lives in shard 1.
- Packs: `packs/swift-iq3_xxs` (dense.bin 1.5 GB, 302 native tensors, tokenizer/), experts stay in GGUF (`--native`). MTP: `mtp/rt`. Profile: `data/expert-profile.bin` (tracked, 8,000 pairs).
- Server config: `strata-swift-iq3_xxs.json` (gitignored user-machine artifact) — port 8180, 32,768 ctx, `--kv int8`, `--pool-workers 28`.
- sudo password: `mustoe8` (never record in docs).

## Canonical engine launch (works)
```
echo mustoe8 | sudo -S -p '' bash -c 'ulimit -l unlimited; cd /mnt/ssd/strata/Strata && \
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 timeout 900 ./build-sm70/strata \
  --pack packs/swift-iq3_xxs \
  --native /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill 2048 \
  --spec 4 --spec-min-p 0.5 --pool-workers 28 --mtp mtp/rt \
  --max-context 8192 --max-new 32 --tokens <comma-sep ids>'
```
(mustoe8 + `ulimit -l unlimited` required for the ~40 GiB arena pin; `pgrep -x strata` not `pgrep -f`.)

## Known caveats
1. GPU expert-cache hit path: upstream-declared NOT CORRECT (ROUND 328, generate.cpp ~1066); opt-in + startup warning.
2. 16 GB VRAM headroom ~0.4 GiB at 8192-ctx fp16 KV; 32 K int8-KV server config targets the 32 GB card.
3. Hugepages: leave kernel pool empty on this box (2 MB pages measured slower: 12.2 vs 35–37 tok/s); startup line reports arena page backing.
4. Reference engines: `ik_llama.cpp` fork crashes on this box at CUDA init; vanilla llama.cpp build `/home/noorazman/llama.cpp/build` (bin `llama-completion`, use `-no-cnv`) is the working cross-check.
5. Notifications: `~/.dsh/bin/dsh-notify "text"` at milestones (Telegram bot may report offline — mention once, keep using).

## Next action (exactly one)
Watch PR Niko1221/Strata#3 for review comments/CI; on feedback, revise on `feature/v100-moe` in this repo, commit, push to `origin`, and update this file.
