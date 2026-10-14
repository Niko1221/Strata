"""serve/jev_golden.py - the jev golden battery for the M3.0 systemone endpoint.

Format: one JSON object per line (jsonl), the donor's battery shape:
  {"state": {...}, "questions": {qid: {type, instructions, criteria?}}, "expect": {qid: {...}}}

`expect` pins only INVARIANTS, never model-specific answers (the port's target model is
Qwen3.8-Flash-Next, not the donor's 27B - exact probabilities are a re-run artifact):
  - noul: 0.0 <= noul <= 1.0
  - choice: choice is a valid index; probabilities sum to 1 (+/- 1e-6); confidence in [0, 1]
  - score: score in [0, len(levels)-1]; probabilities sum to 1
  - shared: deterministic across repeats (same request twice -> identical answers, greedy path)

The golden battery is run by tests/systemone_golden_test.py against a LIVE engine
(--engine strata, --jev-brow on the engine args); it is skipped when no engine is configured.
"""
from __future__ import annotations

import json
from pathlib import Path

# The battery: 12 items covering every qtype, criteria shape, and validation arm.
BATTERY = [
    # 1. noul, minimal
    {"state": {"hp": 7, "mp": 3},
     "questions": {"q1": {"type": "noul", "instructions": "Is the player alive (hp > 0)?"}},
     "expect": {"q1": {"type": "noul"}}},
    # 2. choice, object criteria
    {"state": {"enemy_dist": 4.2},
     "questions": {"q1": {"type": "choice", "instructions": "Pick the next action.",
                          "criteria": {"attack": 0, "defend": 1, "flee": 2}}},
     "expect": {"q1": {"type": "choice", "options": 3}}},
    # 3. choice, array criteria
    {"state": {"tile": "water"},
     "questions": {"q1": {"type": "choice", "instructions": "Which tool?",
                          "criteria": ["boat", "bridge", "swim"]}},
     "expect": {"q1": {"type": "choice", "options": 3}}},
    # 4. score
    {"state": {"threat": "high"},
     "questions": {"q1": {"type": "score", "instructions": "Rate the threat.",
                          "criteria": ["low", "medium", "high", "critical"]}},
     "expect": {"q1": {"type": "score", "levels": 4}}},
    # 5. multi-question same state
    {"state": {"pos": [3, 4], "fuel": 12},
     "questions": {"q1": {"type": "noul", "instructions": "Fuel below 20?"},
                   "q2": {"type": "choice", "instructions": "Move where?",
                          "criteria": {"north": 0, "south": 1}}},
     "expect": {"q1": {"type": "noul"}, "q2": {"type": "choice", "options": 2}}},
    # 6. rich nested state
    {"state": {"party": [{"name": "A", "hp": 30}, {"name": "B", "hp": 0}],
                "weather": {"kind": "rain", "wind": 7.5}},
     "questions": {"q1": {"type": "noul", "instructions": "Is any party member down?"}},
     "expect": {"q1": {"type": "noul"}}},
    # 7. choice with 2 options (boundary)
    {"state": {"door": "locked"},
     "questions": {"q1": {"type": "choice", "instructions": "Open how?",
                          "criteria": {"key": 0, "force": 1}}},
     "expect": {"q1": {"type": "choice", "options": 2}}},
    # 8. score with 2 levels (boundary)
    {"state": {"light": "dim"},
     "questions": {"q1": {"type": "score", "instructions": "Visibility level.",
                          "criteria": ["poor", "good"]}},
     "expect": {"q1": {"type": "score", "levels": 2}}},
    # 9. unicode state values
    {"state": {"npc_says": "你好，旅人。", "gold": 50},
     "questions": {"q1": {"type": "noul", "instructions": "Does the NPC speak Chinese?"}},
     "expect": {"q1": {"type": "noul"}}},
    # 10. large-ish state (two questions share the rendered prefix)
    {"state": {"log": ["event-%d" % i for i in range(40)]},
     "questions": {"q1": {"type": "noul", "instructions": "Was event-7 logged?"},
                   "q2": {"type": "noul", "instructions": "Was event-31 logged?"}},
     "expect": {"q1": {"type": "noul"}, "q2": {"type": "noul"}}},
    # 11. instructions with punctuation/quotes
    {"state": {"quote": "he said \"run\""},
     "questions": {"q1": {"type": "noul", "instructions": "Did he say 'run'? (yes/no)"}},
     "expect": {"q1": {"type": "noul"}}},
    # 12. all three types in one request
    {"state": {"mode": "combat", "hp": 80},
     "questions": {"q1": {"type": "noul", "instructions": "In combat?"},
                   "q2": {"type": "choice", "instructions": "Stance?",
                          "criteria": {"aggressive": 0, "cautious": 1}},
                   "q3": {"type": "score", "instructions": "Urgency 0-2.",
                          "criteria": ["routine", "elevated", "urgent"]}},
     "expect": {"q1": {"type": "noul"}, "q2": {"type": "choice", "options": 2},
                "q3": {"type": "score", "levels": 3}}},
]


def load_battery() -> list[dict]:
    return json.loads(json.dumps(BATTERY))       # deep copy, dict order preserved


def check_invariants(item: dict, answers: dict) -> list[str]:
    """-> [] when every invariant holds; else a list of human-readable violations."""
    bad: list[str] = []
    for qid, exp in item["expect"].items():
        a = answers.get(qid)
        if a is None:
            bad.append(f"{qid}: no answer")
            continue
        if a.get("type") != exp["type"]:
            bad.append(f"{qid}: type {a.get('type')!r} != {exp['type']!r}")
            continue
        if exp["type"] == "noul":
            v = a.get("noul")
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                bad.append(f"{qid}: noul={v!r} out of [0,1]")
        elif exp["type"] == "choice":
            n = exp["options"]
            c = a.get("choice")
            if not isinstance(c, int) or not 0 <= c < n:
                bad.append(f"{qid}: choice={c!r} not a valid index < {n}")
            ps = a.get("probabilities")
            if not isinstance(ps, list) or len(ps) != n:
                bad.append(f"{qid}: probabilities has {len(ps) if isinstance(ps, list) else '?'} entries, want {n}")
            else:
                s = sum(ps)
                if abs(s - 1.0) > 1e-6:
                    bad.append(f"{qid}: probabilities sum {s!r} != 1")
            cf = a.get("confidence")
            if not isinstance(cf, (int, float)) or not 0.0 <= cf <= 1.0:
                bad.append(f"{qid}: confidence={cf!r} out of [0,1]")
        elif exp["type"] == "score":
            n = exp["levels"]
            sc = a.get("score")
            if not isinstance(sc, (int, float)) or not 0.0 <= sc <= n - 1:
                bad.append(f"{qid}: score={sc!r} out of [0,{n - 1}]")
            ps = a.get("probabilities")
            if not isinstance(ps, list) or len(ps) != n:
                bad.append(f"{qid}: probabilities has {len(ps) if isinstance(ps, list) else '?'} entries, want {n}")
            else:
                s = sum(ps)
                if abs(s - 1.0) > 1e-6:
                    bad.append(f"{qid}: probabilities sum {s!r} != 1")
            cf = a.get("confidence")
            if not isinstance(cf, (int, float)) or not 0.0 <= cf <= 1.0:
                bad.append(f"{qid}: confidence={cf!r} out of [0,1]")
    return bad


def battery_path() -> Path:
    return Path(__file__).resolve().parent / "jev_golden.jsonl"


def write_battery() -> Path:
    p = battery_path()
    with open(p, "w", encoding="utf-8") as f:
        for item in BATTERY:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return p
