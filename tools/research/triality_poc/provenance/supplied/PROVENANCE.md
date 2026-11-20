# Provenance / scope

Date: 2026-10-23 JST. Draft and PoC produced in response to the user's request.

## Source of the proposed design

The user's explicit Strata/Bonsai/TurboQuant specification and the supplied
`strata_triality_attention_plan_v3.md` define the same-lift Spin(8) views,
input-conditioned Q adapter, immutable original weights, separate codec layer,
and local-only operation. The user's design and prior numerical claims are not
presented as new measurements in this package.

Cayley–Dickson/Clifford conventions agree with the user's earlier CPU reference
archive. The current implementation uses PyTorch tensors and was tested afresh.
No pretrained model, external corpus, private credential or original large model
file is bundled. The included teacher/adapter checkpoints are tiny synthetic
models trained during this response.

## Pinned future integration targets (not modified or qualified by this PoC)

- Strata: 011c9e600a93bbc9fd451c563501b5831c32bbca
- zapabob/llama.cpp: 2c8e9655cdc6ea31051d4437a71492e717d046a4
- PrismML-Eng/llama.cpp: 73bfe6cc485554822f80c473a3c0c4f25e80f41b
- zapabob/Turboquant-CUDA: d85c787733f6a973d7cfe953d11276701a3c09e0
- Its llama.cpp gitlink: 81b08be15c554918a8cfe20d5f0846038082430f
- TheTom/llama-cpp-turboquant: 643956891dfdf51aecdbd05ca508dd21e21e8195
- TheTom/turboquant_plus: ba52ad107d1fdd02bc9be8fd85308226b75c905b

User-provided large-model identity:
`zapabobouj/Hikari-Bonsai-27B-Nimble-LoRA-GGUF`.
Expected SHA256 (not independently rehashed in this environment):
`0ab2a1f237b73e5002ecd8c9b764249b90ee59eca749a6226c44bb43b1036a13`.

## Fresh evidence

- evidence/red_algebra.txt and red_split.txt: failed contract fixtures before the fix.
- evidence/validation_dtype_failure.txt: intermediate dtype failure, retained.
- evidence/final_pytest.txt: final full suite, including explicit CUDA skip.
- evidence/validation.json: fresh CPU FP64 / statistical validation.
- evidence/tiny_lm/results.json: complete 3-seed CPU experiment.
- evidence/mutations.json: selected source mutants detected on temporary copies.
- evidence/cpu_operator_benchmark.json: CPU operator timings, NOT GPU/LLM throughput.
- evidence/summary.json: transparent aggregation, including negative comparisons.

Whole-project mutation coverage, C++/CUDA product builds, long-context inference,
Windows process recovery, real-model PPL and speed are NOT established here.

## Literature

`paper/references.json` and `paper/references.bib` contain 26 open-access scholarly
references. The review is selective, not an exhaustive priority search. Some
sources are preprints; this package does not claim they are all peer-reviewed.
Official PyTorch installation/documentation is listed separately in the paper.

## Authorship and licence

The design origin is the user's specification. ChatGPT assisted with literature
synthesis, deductions, code and experiment/report generation. No journal
submission or human affiliation is invented. Research claims and authorship
should be reviewed by the responsible human researcher before publication.
This package does not relicense Strata, Prism, TurboQuant, TheTom, PyTorch, model
weights, cited papers, or any other external project.
