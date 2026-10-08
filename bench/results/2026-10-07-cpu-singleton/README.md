# Bit-exact AVX2 singleton CPU experts

On an RTX 3060 12 GB / i7-12700KF / 62 GB PC, the IQ3_XXS pack's CPU expert pool spends most of
its time on gate/up rows. In an exploratory engine 0.1.40 run of 256 chat tokens, gate/up took
14.81 ms per verify window and down took 7.41 ms. Of 52,554 distinct CPU expert groups, 43,839 served
one token. The one-token gate/up path normally uses ggml's dot product. This experiment keeps its
floating-point accumulation and horizontal-sum order while using Strata's AVX2 weight decode.
Only IQ2_XXS, IQ2_XS, IQ3_XXS and IQ3_S, on CPUs with AVX2 but without AVX-512, are eligible.
`STRATA_CPU_IQ_ALL_EXACT1=1` opts in;
the default is unchanged.

## Correctness

With this branch's pinned ggml build, `iq_avx2_parity` compared 5,120 gate/up output rows per format
against ggml: **0 bit differences** on P-core 2, E-core 16 and the non-VNNI AVX2 path. On real
IQ3_XXS-pack expert rows, `native_expert_parity` also found 0 differences for the four tested IQ3_S
and two tested IQ3_XXS layers (1,920 outputs per layer). An A/B with the same 52-token chat prompt
and 512 generated tokens matched both emitted tokens and meaningful state hashes in every pair.
The same CPU parity check passed on both core types with `STRATA_PORTABLE=ON` and the pinned ggml
AVX2 build. The opt-in path is not taken on AVX-512 CPUs.

## Speed

Single-thread microbenchmark: 64 MB of expert blobs, one token per expert, 3 repeats; median gate/up
milliseconds per expert on this i7-12700KF. The same down kernel runs in both arms.

| Gate/up format | P-core 2 ggml / opt-in | E-core 16 ggml / opt-in |
| --- | ---: | ---: |
| IQ2_XXS | 0.181 / 0.166 | 0.313 / 0.295 |
| IQ2_XS | 0.209 / 0.192 | 0.444 / 0.358 |
| IQ3_S | 0.343 / 0.179 | 0.718 / 0.406 |
| IQ3_XXS | 0.220 / 0.189 | 0.402 / 0.386 |

End-to-end A/B with engine 0.1.40.2 on the same RTX 3060 / i7-12700KF, IQ3_XXS pack, 160K configured context,
52-token chat prompt, 512 generated tokens, 4 alternating A/B pairs, GPU stage profiling off:
median A **45.28 tok/s**, median B **46.42 tok/s**, median *paired* change **+2.45%** (pair range
+1.51% to +3.66%). Tokens and state matched in every pair. This is one workload on one PC; it
does not establish a gain on other models, CPUs or contexts.

An attempted 512-token refactoring-prompt A/B changed draft acceptance in one of four pairs even
though its emitted text matched; no speed claim is made from that request. Draft-policy decisions
use observed round times, and one-token arithmetic parity cannot make that policy deterministic.

Reproduce the isolated kernel check with `cmake --build build --target iq_avx2_parity strata`, then
`build/iq_avx2_parity --cpu 2` and `build/iq_avx2_parity --cpu 16`. For the non-VNNI path use
`STRATA_NO_AVXVNNI=1 build/iq_avx2_parity --cpu 2`. For microbenchmarks use
`build/iq_avx2_parity --bench --dispatch --cpu 2 --nt 1 --mb 64 --reps 3 --pairs
iq2_xxs/iq4_nl,iq2_xs/q2_0,iq3_s/iq4_nl,iq3_xxs/iq4_nl` (repeat with `--cpu 16`).
For real expert rows, run `build/native_expert_parity <IQ3_XXS-shard1.gguf> 20 28 34 43 35 44`.
