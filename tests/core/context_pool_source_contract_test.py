"""Host-only wiring check; NOT a substitute for a native server journey.

The CPU capture fixture tests real allocation capacities and the production pool.
This companion pins the disabled server task to the verified pre-patch source
(54828c7efbcdf9474697e70922d851f6707f4b32), without requiring git at test time.
Hashes omit comments/whitespace only; they are not hashes of a patched build.
"""
import hashlib
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
source = (root / "src/program/generate.cpp").read_text()


def normalized(text):
    return "".join(re.sub(r"//[^\n]*", "", text).split())


def digest(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def block_after(text, marker):
    start = text.index(marker) + len(marker)
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    raise AssertionError("unterminated block")


pool = source[source.index("class CheckpointPool {"):source.index("/// STRATA_CKPT_THREADS:")]
assert digest(pool) == "b93b131fba5e8a8a474bdcd99d39811a8acfac94c47dd1656e3dd9c0a5958b81"
topup = block_after(source, "auto pool_topup = [&] {")
disabled = block_after(topup, "if (!strata::core::ContextLockPolicy::enabled()) {")
start = disabled.index("pool_refill = std::async")
end = disabled.index("\n                });", start) + len("\n                });")
assert digest(disabled[start:end]) == "ba12700ee32d2ea3c6eeb9924002f2f271318ade5dde3a77491d8ff5ed8191ca"
assert disabled[end:].strip() == "return;"  # cannot fall through to the gated task

park = normalized(block_after(source, "auto park_source = [&](const ParkSource& src, size_t held) -> bool {"))
assert "constboolplanned_context=strata::core::ContextLockPolicy::enabled();" in park
assert "if(planned_context&&pool_refill.valid())pool_refill.get();" in park
assert "if(planned_context&&ckpt_pool.on()&&n_st==0)" in park
late = park.index("if(!planned_context){if(ckpt_pool.on()&&n_st==0)")
assert late > park.index("while(!admit()")  # legacy take/move only AFTER admission
assert "if(planned_context)storage.clear();elsefor(ConvCheckpoint&c:storage)ckpt_pool.give(std::move(c));" in park
assert "if(planned_context)image.stage_images.reserve(n_st);elsepool_topup();" in park
assert park.index("elsepool_topup();") < park.index("image.stage_images.push_back")
assert "if(stored||!planned_context){return_captured_moves.on=false;restore_moves.on=false;}" in park
assert park.index("restore_moves.on=false;") > park.index("conversations.put(std::move(image),held)")
state = (root / "src/core/conversation_state.cpp").read_text()
exact = block_after(state[state.index("bool checkpoint_save("):], "if (ContextLockPolicy::enabled()) {")
for payload in ("gdn", "ple", "tails", "dead", "block_pos"):
    assert f"c.{payload}.reserve(" in exact
copy = normalized((root / "src/core/conversation_copy.hpp").read_text())
assert "if(ContextLockPolicy::enabled())o.stage_parts.reserve(c.stage_parts.size());" in copy
assert "out.reserve(ContextLockPolicy::enabled()?memory_bytes_sum({cs.size(),moved_count}):cs.size());" in copy
snapshot = normalized((root / "src/core/conversation_snapshot.cpp").read_text())
assert "if(ContextLockPolicy::enabled())a.items.reserve(items);" in snapshot
print("baseline pool/task source hashes and context-only server wiring passed (static, not GPU runtime)")
