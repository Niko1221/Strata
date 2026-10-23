#!/usr/bin/env python3
"""build_gqa_set.py - build a labelled GQA binary set for /v1/systemone (noul), with images.

GQA's labels live in `testdev_balanced_instructions` (12,578 rows: id, imageId, question, answer,
isBalanced, ...) and its images live in a SEPARATE config `testdev_balanced_images` (398 rows:
id + image.src).  Answers are free text, so only the yes/no questions are usable as `noul`; those
are what this keeps, balanced 50/50.

Rows are sampled at a stride across the split rather than taken from the head, so the sample is not
biased to whatever order the viewer returns them in.

  build_gqa_set.py --out /opt/data/eval/gqa/gqa-600.jsonl [--per-class 300] [--seed 11]

Writes the jsonl plus gqa-600.manifest.json (counts, repo revision, sha256 of each image as used).
"""
import argparse
import base64
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DS = "lmms-lab-encoder/GQA"
ROWS = "https://datasets-server.huggingface.co/rows"


def token():
    for line in open("/opt/data/llm-swap/.hf-token"):
        return line.strip()
    return ""


def get_json(url, tok, tries=4):
    for i in range(tries):
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok,
                                                   "User-Agent": "suzie-gqa"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i + 1 < tries:
                time.sleep(2 * (i + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def rows_page(config, split, offset, length, tok):
    q = urllib.parse.urlencode({"dataset": DS, "config": config, "split": split,
                                "offset": offset, "length": length})
    return get_json(f"{ROWS}?{q}", tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stride-pages", type=int, default=25, help="how many 100-row pages to sample")
    args = ap.parse_args()

    tok = token()
    first = rows_page("testdev_balanced_instructions", "testdev", 0, 1, tok)
    total = first["num_rows_total"]
    print(f"testdev_balanced_instructions: {total} rows")

    step = max(1, total // args.stride_pages)
    cand = []
    for off in range(0, total, step):
        page = rows_page("testdev_balanced_instructions", "testdev", off, 100, tok)
        for r in page["rows"]:
            row = r["row"]
            ans = str(row.get("answer", "")).strip().lower()
            if ans in ("yes", "no"):
                cand.append({"uid": f"gqa-{row['id']}", "image_id": row["imageId"],
                             "question": row["question"], "gold": 1 if ans == "yes" else 0,
                             "balanced": bool(row.get("isBalanced"))})
        time.sleep(0.15)
    print(f"binary (yes/no) candidates from {len(cand)} sampled rows: "
          f"yes {sum(1 for c in cand if c['gold'] == 1)} / no {sum(1 for c in cand if c['gold'] == 0)}")

    # dedupe identical (image, question) pairs
    seen, uniq = set(), []
    for c in cand:
        k = (c["image_id"], c["question"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    rnd = random.Random(args.seed)
    yes = [c for c in uniq if c["gold"] == 1]
    no = [c for c in uniq if c["gold"] == 0]
    rnd.shuffle(yes)
    rnd.shuffle(no)
    picked = yes[: args.per_class] + no[: args.per_class]
    rnd.shuffle(picked)
    if len(picked) < 2 * args.per_class:
        print(f"WARNING: only {len(picked)} items available "
              f"(yes {len(yes)}, no {len(no)}) - reduce --per-class")
    print(f"picked {len(picked)} items over {len({c['image_id'] for c in picked})} images")

    print("fetching image rows...")
    img_map = {}
    off, total_img = 0, None
    while True:
        page = rows_page("testdev_balanced_images", "testdev", off, 100, tok)
        total_img = page["num_rows_total"]
        for r in page["rows"]:
            row = r["row"]
            img_map[row["id"]] = row["image"]["src"]
        off += 100
        if off >= total_img:
            break
        time.sleep(0.15)
    print(f"image rows: {len(img_map)} (total {total_img})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    used, missing, written = {}, [], 0
    with open(out, "w") as f:
        for c in picked:
            src = img_map.get(c["image_id"])
            if not src:
                missing.append(c["image_id"])
                continue
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        src, headers={"Authorization": "Bearer " + tok, "User-Agent": "suzie-gqa"}),
                        timeout=60) as r:
                    raw = r.read()
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                missing.append(f"{c['image_id']} ({e})")
                continue
            used[c["image_id"]] = hashlib.sha256(raw).hexdigest()[:16]
            f.write(json.dumps({
                "id": c["uid"],
                "qid": c["uid"],
                "image_b64": base64.b64encode(raw).decode(),
                "image_sha": used[c["image_id"]],
                "q": {"type": "noul", "instructions": c["question"]},
                "gold": c["gold"],
                "image_id": c["image_id"],
            }) + "\n")
            written += 1
    manifest = {"dataset": DS, "config": "testdev_balanced", "split": "testdev",
                "rows_total": total, "binary_candidates": len(uniq),
                "written": written, "images_used": len(used),
                "missing": missing[:10], "image_sha": used,
                "seed": args.seed, "per_class": args.per_class}
    Path(str(out).replace(".jsonl", ".manifest.json")).write_text(json.dumps(manifest, indent=1))
    print(f"wrote {written} items -> {out} ({len(used)} images, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
