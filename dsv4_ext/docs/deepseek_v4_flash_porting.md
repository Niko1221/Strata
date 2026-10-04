# DeepSeek-V4-Flash: porting notes (step 1)

Rule of this tree: **additive**. Nothing under the Strata checkout is modified, so the fork stays aligned with
the maintainer. Strata is consumed read-only (headers/objects) in later steps.

## Facts read from the real GGUF header (shard 1 of UD-IQ1_M, `general.architecture = deepseek4`)

| Item | Value |
| --- | --- |
| layers | 43 (+1 extra entry in `compress_ratios`: len 44, inferred MTP layer) |
| n_embd / heads / kv heads / head_dim | 4096 / 64 / 1 / 512 (rope dims 64) |
| LoRA ranks | q 1024, output 1024 in 8 groups |
| MoE | 256 experts, top-6, 1 shared, ff 2048, gating_func id 4, scale 1.5, top-k renorm |
| hash routing | `hash_layer_count = 3` |
| swiglu clamp | per-layer arrays (`swiglu_clamp_exp`, `swiglu_clamp_shexp`), 43 entries each |
| indexer | 64 heads, key_length 128, top_k 512 |
| compress ratios | layers 0,1 = 0, then alternating 4 / 128, last (extra) = 0 |
| hyper-connections | 4 streams, 20 Sinkhorn iterations, eps 1e-6 |
| rope | YaRN factor 16, orig ctx 65536, base 10000, compress base 160000 |
| tokenizer | gpt2 BPE (`joyai-llm` pre), vocab 129280, BOS 0, EOS 1, pad 2, add_bos false |

## NOT known yet (do not assume)

- Tensor names, shapes and ggml types: shard 1 has **0 tensors**; `shapes.txt` from all 3 shards is needed.
- Whether `expert_gating_func = 4` is sqrtsoftplus (the HF `scoring_func`): assumed, to be checked against llama.cpp.
- The exact math of hyper-connections, compressor, indexer and hash routing (to be taken from the reference
  `inference/model.py` / transformers `modular_deepseek_v4.py`, then compared against a small PyTorch model).
- Whether Strata's native CPU expert kernel handles these experts (IQ1_M with 4096x2048). Needs
  `cpu/native_expert.hpp`, `cpu/expert.hpp`, `cpu/expert_layout.hpp`.

## Design (Strata's philosophy, as an extension)

router (always authoritative) -> residency table (layer, expert) -> HIT: GPU grouped kernel / MISS: CPU pool,
both fed the SAME quantized activation and fp32 scales, each writing its own routed row, then combine.
The profile only decides which experts are resident; it never touches routing.

## Roadmap (each step is tested before the next)

1. [done, CPU-tested] GGUF header reader, config, expert inventory from real shapes, VRAM plan (`--expert-vram-pct`).
2. Tensor-name map + loader plan from `shapes.txt`.
3. Components with parity tests vs a small PyTorch reference: router (sqrtsoftplus + bias), hash routing,
   hyper-connection, compressor/indexer attention, shared expert.
4. All-resident baseline, then HIT/MISS tier on top of Strata's ExpertCache/ExpertPool (read-only use).
5. Profile preload/save, static then adaptive, benchmarks P=0/25/50/75/100.

## Findings from the official `inference/model.py` (verified by reading, implemented in `ops.cpp`)

- Layers with `compress_ratio == 0` (pure sliding window) use `rope_theta` WITHOUT YaRN; compressed layers use
  `compress_rope_theta` WITH YaRN. Two different rope tables.
- Rotary is applied only to the last `rope_head_dim` (64) elements of q, k and compressed kv, and is **inverted** on the
  attention output before `wo_a`.
- Router: `scores = sqrt(softplus(logits))`; bias shifts only the top-k selection; weights come from the unbiased scores,
  are normalised to sum 1, then times `route_scale`. Hash layers take experts from `tid2eid[token_id]`.
- Expert: `silu(min(gate, limit)) * clamp(up, -limit, limit)` computed in float32, routing weight applied BEFORE `w2`.
- Shared expert is added after the routed sum; the routed sum accumulates in float32.
- Ratio-4 layers use an overlapping compressor (2x width) plus an indexer (top-512); ratio-128 layers have no indexer.
- `hc_head_*` (final) and `hc_*_fn/base/scale` per layer match the GGUF tensor shapes ((2+4)*4 = 24 mixes).

Still missing from the reference: `kernel.py` (`hc_split_sinkhorn`, `sparse_attn`, `act_quant` semantics).
