# RTX PRO adaptive configuration experiments

Private research on llm-60: RTX PRO 6000 Blackwell Workstation Edition 96 GB,
Ryzen 9 7950X, 128 GB RAM. Base: upstream 0.1.38, `99f3dbd`.

This branch retains upstream's adaptive-cache completion wait (`322ea76`,
`1e5f6c1`) and PCIe probe/group sizing improvements. `STRATA_ADAPT_NOWAIT`
stays off. Local compatibility ports add non-MTP serving, full Q8_0 PLE
rows/direct reads, and the tested output/EOS commit limit. The PCIe launch
size from current upstream and its Q5_0 direct-reader support are preserved.

The earlier fixed 16,000-slot, PCIe-fraction-zero runs remain historical evidence.
The active performance configurations use automatic expert-cache capacity,
adaptive swaps, and automatic PCIe fraction. Measurements must record the actual
slots, RAM complement, CPU/PCIe split, swaps, output hash and first divergence.
More output speed is not evidence of matching semantics when placement changes.

Configuration paths share this engine and vary independently:

| Path | Weights | KV | Scope |
| --- | --- | --- | --- |
| q4-int8-kv | Full UD-Q4_K_XL | int8 | All experts on GPU when they fit |
| q4-fp16-kv | Full UD-Q4_K_XL | fp16 | Separate precision/performance path |
| q8-int8-kv | Full Q8_0 | int8 | Adaptive VRAM/RAM placement |
| q8-fp16-kv | Full Q8_0 | fp16 | Re-evaluate cache capacity after KV growth |
| esp | Q4 initially | int8 initially | Model-changing control-vector experiment |

Each path checks serial, MTP, n-gram and combined drafting. Start with 65,536
actual input tokens / 73,728 allocated context, then validate 131,072 / 139,264
for configurations that fit. Use identical prompt tokens, record actual output
length and natural stops, and report output and effective throughput separately.
Select useful candidates for reverse-order repeats rather than treating single
observations as small performance gains.

ESP is off in the engine optimization comparisons. Its optional projection
changes model activations and answers; measure generated-code behavior as well
as throughput and draft acceptance. An ESP-on answer is not an exactness control
for ESP-off. Loaded-but-disabled projection should match the no-vector arm.

This branch is a test base. No performance claim is made until its results are
recorded. No system service or public endpoint is installed by these experiments.
