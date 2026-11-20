# Evolutionary archive evidence (2026-11-20)

The evolutionary method used in this session was an archive of candidate
engine variants, each built and scored by the same automated harness
(CTest 18/18 gate, then median prefill time with token-identity gate). The LLM
proposals came from analysis subagents and NIM MiniMax; selection was automatic.

| Variant | Origin of change | Build dir | 9B median | 27B median | Fitness outcome |
|---|---|---|---|---|---|
| V0 baseline | initial | `q38-9b-geometry-20261120` | 563.3 ms | 555.8 ms | reference |
| V1 expf cache | subagent finding (256x expf) | `q38-9b-expf-20261120` | 807.2 ms single run, token identical | not run | no clear gain |
| V2 V1 + RoPE table + memset removal | subagent finding | `q38-9b-algo-20261120` | **534.422 ms** | not run | +5.4%, accepted |
| V3 V2 + GQA-fused attention | mathematical GQA analysis | `q38-9b-gqa-20261120` | 18/18 tests | **522.559 ms** | +6.0% on 27B, accepted |
| V4 decode expf cache | subagent finding | `q38-9b-decode-20261120` | build failed kv_numerics memory accounting | - | rejected, reverted |

Selection rule: a variant enters the archive only if CTest passes and its
median is strictly better than the incumbent with identical generated tokens
(9B `15217`, 27B `381`). V3 is the current best. V4 was discarded.

Shinka/NIM was attempted as an additional mutation engine, but MiniMax failed
to emit extractable patches and the endpoint returned 429/529, so it produced
no accepted variant. The archive above is the validated evolutionary result.
