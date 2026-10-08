"""Original synthetic lookup fixture; no model, allocation or file side effects.

Call build_prompt_ids with encode(text), using the matching model tokenizer
and parse_special=True. The caller owns tokenizer loading and engine execution.
This public extraction has not been run as a portable end-to-end campaign.
"""
from hashlib import sha256

FACTS = {
    "FIRST": "maple-copper-731",
    "SECOND": "violet-anchor-284",
    "THIRD": "silver-orbit-596",
    "LAST": "cedar-lantern-842",
}
EXPECTED_TOKENS = 60270
EXPECTED_SHA256 = "5ec64b0c7e58d274b0b652888cea66fcff15608b1d4f30a611ad4f1ff35a8399"
QUESTION = "Return all four IMPORTANT KEY values FIRST, SECOND, THIRD and LAST, exactly as written. Do not calculate or guess."


def wrap(reference):
    return (
        "<|im_start|>system\nYou are a careful assistant. Reason carefully, then answer accurately.\n"
        + reference + "<|im_end|>\n<|im_start|>user\n" + QUESTION
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


def build_prompt_ids(encode):
    lines = [
        f"Record {i:05d}: storage shard {i % 83}; status archived; checksum {(i * 15485863) % 999999937:09d}; no action required.\n"
        for i in range(3000)
    ]
    while len(encode(wrap("".join(lines)))) > 61000:
        lines = lines[:-20]
        if not lines:
            raise ValueError("unexpected tokenizer or prompt overhead")
    for position, (key, value) in zip(
        (4, len(lines) // 3, 2 * len(lines) // 3, len(lines) - 4), FACTS.items()
    ):
        lines[position] = f"IMPORTANT KEY {key} = {value}.\n"
    ids = encode(wrap("".join(lines)))
    digest = sha256(str(ids).encode("ascii")).hexdigest()
    if len(ids) != EXPECTED_TOKENS or digest != EXPECTED_SHA256:
        raise ValueError(f"fixture/tokenizer mismatch: {len(ids)} tokens, SHA256 {digest}")
    return ids


def passes_lookup(text):
    return "</think>" in text and all(value in text.split("</think>")[-1] for value in FACTS.values())
