"""Apply reviewed, hunk-specific native diff3 resolutions; fail on changed conflicts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("H:/Strata/evidence/upstream-sync-20261116/native-resolution.json")
CONFLICT = re.compile(
    r"^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n",
    re.MULTILINE | re.DOTALL,
)


def resolve(path: str, rules: list) -> dict:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    matches = list(CONFLICT.finditer(text))
    if len(matches) != len(rules):
        raise ValueError(f"Unexpected conflicts: {path}: {len(matches)}")
    index = 0

    def replace(match: re.Match) -> str:
        nonlocal index
        rule = rules[index]
        index += 1
        ours, theirs = match.groups()
        return rule(ours, theirs)

    merged = CONFLICT.sub(replace, text)
    if re.search(r"^(<<<<<<<|=======|>>>>>>>)", merged, re.MULTILINE):
        raise ValueError("Unresolved conflict")
    target.write_text(merged, encoding="utf-8", newline="\n")
    return {
        "path": path, "resolved_hunks": len(rules),
        "before_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(merged.encode()).hexdigest(),
    }


def q36(ours: str, theirs: str) -> str:
    if "hs_stream_mmq_ms" not in ours or "FQ::experts_native_ext" not in theirs:
        raise ValueError("Q36 conflict changed")
    # Keep the new fused/int8 paths. The existing diagnostic timer still
    # measures only MMQ, so it must live inside that branch, not around all paths.
    text = theirs.replace(
        "            } else {\n                MMQ::quantize",
        "            } else {\n"
        "                const auto hs_mmq = hotspot_on ? Clock::now() : Clock::time_point{};\n"
        "                MMQ::quantize",
        1,
    )
    return text.replace(
        "                prow = rinv;\n",
        "                prow = rinv;\n"
        "                if (hotspot_on) {\n"
        "                    CK(cudaStreamSynchronize(st));\n"
        "                    hs_stream_mmq_ms += ms_since(hs_mmq);\n"
        "                }\n",
        1,
    )


def shrink(ours: str, theirs: str) -> str:
    if "cleanup_pending_" not in ours or "refused(\"unmap\"" not in theirs:
        raise ValueError("Expert cache conflict changed")
    # Preserve fork fail-closed ownership after an API failure and upstream
    # accurate mapped-prefix accounting plus deterministic failure injection.
    text = ours.replace(
        "const CUresult unmapped = v.unmap(p, (size_t) seg_size_[(size_t) i]);",
        "const CUresult unmapped = refused(\"unmap\", i) ? (CUresult) -1\n"
        "            : v.unmap(p, (size_t) seg_size_[(size_t) i]);",
    ).replace(
        "if (unmapped == CUDA_SUCCESS) seg_mapped_[(size_t) i] = 0;",
        "if (unmapped == CUDA_SUCCESS) {\n"
        "            seg_mapped_[(size_t) i] = 0;\n"
        "            mapped_segs_ = i;\n"
        "        }",
    )
    return text.replace(
        "? v.release((CUmemGenericAllocationHandle) segs_[(size_t) i]) : (CUresult) -1;",
        "? (refused(\"release\", i) ? (CUresult) -1\n"
        "               : v.release((CUmemGenericAllocationHandle) segs_[(size_t) i]))\n"
        "            : (CUresult) -1;",
    )


def main() -> None:
    if OUT.exists():
        raise ValueError("Retain prior resolution receipt")
    both = lambda ours, theirs: ours + theirs
    ours = lambda ours, theirs: ours
    rows = [
        resolve("CMakeLists.txt", [
            lambda a, b: a.replace("VERSION 0.5.9", "VERSION 0.6.0"), both,
        ]),
        resolve("metal/CMakeLists.txt", [
            lambda a, b: a.replace('STRATA_VERSION="0.5.9"', 'STRATA_VERSION="0.6.0"'),
        ]),
        resolve("src/program/generate.cpp", [
            lambda a, b: a.removeprefix("#endif\n") + b, both, both,
        ]),
        resolve("src/qwen35/q36_engine.cu", [q36]),
        resolve("src/core/expert_cache.cpp", [ours, shrink, both]),
    ]
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump({"source": "reviewed_three_way_hunks", "files": rows}, stream, indent=2)


if __name__ == "__main__":
    main()
