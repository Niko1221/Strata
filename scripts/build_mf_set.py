#!/usr/bin/env python3
"""build_mf_set.py - build a LABELLED two-field decision set from banking77.

Why this dataset: the #3743 thread's A/B used banking77 with two fields (`is_card`, `group`), so
numbers measured here are directly comparable with leonardomurillo's.  Each item carries the gold
answers for BOTH fields, taken from the intent label - no model in the loop when labelling.

  state.utterance         the customer's message (banking77 text)
  questions.is_card       noul: "does this concern a card?"
  questions.group         choice over 5 intent groups
  gold.is_card            1 when the label contains "card", else 0
  gold.group              card | top_up | transfer | cash | other   (deterministic from the label)

Balanced on is_card (half yes, half no) so the metric the thread flagged (is_card, p 0.002) has
power at small n.

usage: build_mf_set.py --parquet test.parquet --labels label_names.json --out mf-set.jsonl
                       [--per-class 100] [--seed 7]
"""
import argparse
import json
import random

GROUPS = [
    ("card", lambda s: "card" in s),
    ("top_up", lambda s: "top_up" in s),
    ("transfer", lambda s: "transfer" in s),
    ("cash", lambda s: "cash" in s or "withdrawal" in s),
    ("other", lambda s: True),          # the fallback: must be last
]


def group_of(label: str) -> str:
    for name, test in GROUPS:
        if test(label):
            return name
    return "other"


def is_card(label: str) -> int:
    return 1 if "card" in label else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=100, help="items per is_card class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", action="append", default=[],
                    help="jsonl whose utterances must NOT reappear (train/held-out split)")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    names = json.load(open(args.labels))
    table = pq.read_table(args.parquet).to_pydict()
    rows = list(zip(table["text"], table["label"]))

    yes = [(t, n) for t, i in rows for n in [names[i]] if is_card(n)]
    no = [(t, n) for t, i in rows for n in [names[i]] if not is_card(n)]
    if args.exclude:
        skip = set()
        for path in args.exclude:
            for line in open(path):
                line = line.strip()
                if line:
                    skip.add(json.loads(line)["state"]["utterance"])
        before = len(yes) + len(no)
        yes = [(t, n) for t, n in yes if t not in skip]
        no = [(t, n) for t, n in no if t not in skip]
        print(f"excluded {len(skip)} utterances: {before} -> {len(yes) + len(no)} candidate rows "
              f"(yes {len(yes)}, no {len(no)})")
    rnd = random.Random(args.seed)
    rnd.shuffle(yes)
    rnd.shuffle(no)
    picked = [(t, n, 1) for t, n in yes[: args.per_class]] + [(t, n, 0) for t, n in no[: args.per_class]]
    rnd.shuffle(picked)

    with open(args.out, "w") as f:
        for i, (text, label, card) in enumerate(picked):
            f.write(json.dumps({
                "id": f"b77-{i:04d}",
                "state": {"utterance": text},
                "questions": {
                    "is_card": {"type": "noul",
                                "instructions": "Does this customer request concern a card?"},
                    "group": {"type": "choice",
                              "instructions": "Which group does this customer request belong to?",
                              "criteria": {name: None for name, _ in GROUPS}},
                },
                "gold": {"is_card": card, "group": group_of(label)},
                "intent": label,
            }) + "\n")
    print(f"wrote {len(picked)} items -> {args.out} "
          f"(is_card yes {sum(1 for *_, c in picked if c == 1)}, no {sum(1 for *_, c in picked if c == 0)})")


if __name__ == "__main__":
    main()
