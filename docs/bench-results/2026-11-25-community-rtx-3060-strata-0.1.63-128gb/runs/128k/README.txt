Strata non-standard synthetic long-context experiment
UTC: 2026-11-25T18:37:17+00:00
GPU: NVIDIA RTX 3060 12 GiB (Minisforum DEG1 OCuLink), GPU Unlock ON
Model: Qwen3.8-Flash-Next Q2_0 (ISTA-DASLab)
Context configured: 131072
Nominal input target: 128000 tokens
Speed runs: 3 repetitions, output cap 512, needle depth 50 percent
Recall runs: needle depth 10 and 90 percent, output cap 2048
Effective recall input can be shorter than nominal target to reserve generation tokens.
Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
  presence_penalty 0.0, repetition_penalty 1.0,
  reasoning_effort xhigh, reasoning_budget_tokens 0,
  chat_template_kwargs.enable_thinking true,
  chat_template_kwargs.preserve_thinking true.
Drafting OFF; GPU Unlock ON. One request at a time.
IMPORTANT: Same sampling as strata-qwen38-thinking-v1-draft.1,
  but the synthetic long-context workloads are NOT the standard protocol.
Run after standard benchmark on same Strata instance; caches may have warmed.
Public release requires inspection for hostname/path/IP and secrets.
