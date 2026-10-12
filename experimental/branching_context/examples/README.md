# Runnable example: model-free only

`python3 examples/branching_demo.py`

Demonstrates an immutable prefix, three branches with distinct known input suffixes, revisioned rewinding, preserved old futures, a deliberately simple read-only evaluation function, CAS adoption and process-level store reopen. The "model" is an exact-rational mean-of-tokens toy. It **does not** execute any neural network, produce a KV cache, demonstrate physical prefix sharing, save GPU model state, or certify G1–G5.

The reference store restricts model token IDs to nonnegative integers. The toy itself uses a negative example suffix, so the script uses a bijective zigzag encoding at the fixture boundary. This is not an inference-backend encoding or workaround for invalid token IDs. Real fixtures must use actual tokenizer token IDs.
