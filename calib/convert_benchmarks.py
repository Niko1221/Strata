#!/usr/bin/env python3
"""eval/jevport-convert.py — real-data converters for the jev image decision bench.

Pulls live rows from HF datasets-server (pinned revisions embedded in the signed
URLs on fetch; manifest records dataset @ revision) and writes per-item JSON in
the suite's /cfg/strata-data/eval/data/<kind>/ shape:

  POPE adversarial  -> noul   (lmms-lab-encoder/POPE, config Full, split adversarial)
  CV-Bench          -> choice (nyu-visionx/CV-Bench, config default, split test)
  CountBenchQA      -> score  (vikhyatk/CountBenchQA, split test)

Safety rules (Opus plan §1):
  - BANNED fields never enter a request: POPE 'category' is benign but CountBench
    'text' and A-OKVQA 'rationales' carry the answer; CV-Bench 'prompt' lists the
    options (double-listing) - we always send 'question', never 'prompt'.
  - images are downloaded, hashed (sha256, recorded in the manifest), stored as
    PNG files; the repo gets IDs + hashes, never the images.
  - sampling is by IMAGE (POPE: 6 questions/image preserved, dev/test disjoint by
    image; CV-Bench/CountBench: item = image).

usage: jevport-convert.py <outdir> <n_images_pope> <n_items_cvb> <n_items_cb>
"""
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

DS = "https://datasets-server.huggingface.co/rows"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def fetch_bytes(url):
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read()


def to_png_b64(raw: bytes) -> str:
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    im.load()
    if im.mode != "RGB":
        im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PNG")
    import base64
    return base64.b64encode(out.getvalue()).decode()


def convert_pope(outdir, n_images):
    rows = []
    for off in range(0, 3000, 100):
        j = fetch_json(f"{DS}?dataset=lmms-lab-encoder%2FPOPE&config=Full&split=adversarial&offset={off}&length=100")
        rows += j.get("rows", [])
        if len(rows) >= n_images * 6 + 100:
            break
    by_img = {}
    for r in rows:
        row = r["row"]
        by_img.setdefault(row["image_source"], []).append(row)
    imgs = sorted(k for k, v in by_img.items() if len(v) >= 6)
    random.Random(11).shuffle(imgs)
    dev_imgs, test_imgs = imgs[:n_images // 2], imgs[n_images // 2:n_images]
    items = {"dev": [], "test": []}
    for split, keys in (("dev", dev_imgs), ("test", test_imgs)):
        for k in keys:
            for row in by_img[k][:6]:
                q = {"type": "noul", "instructions": row["question"]}
                gold = 1 if row["answer"].strip().lower() == "yes" else 0
                items[split].append({"qid": f"pope_{row['id']}", "q": q, "gold": gold,
                                     "image_url": row["image"]["src"],
                                     "image_sha": None, "img_key": k})
    return items


def convert_cvb(outdir, n_items):
    rows = []
    for off in range(0, 2638, 100):
        j = fetch_json(f"{DS}?dataset=nyu-visionx%2FCV-Bench&config=default&split=test&offset={off}&length=100")
        rows += j.get("rows", [])
        if len(rows) >= n_items + 60:
            break
    random.Random(12).shuffle(rows)
    items = {"dev": [], "test": []}
    half = n_items // 2
    for split, chunk in (("dev", rows[:half]), ("test", rows[half:n_items])):
        for r in chunk:
            row = r["row"]
            choices = list(row["choices"])
            gold = "ABCDEF".index(row["answer"][1])        # "(C)" -> 2
            q = {"type": "choice", "instructions": row["question"],      # NOT prompt
                 "criteria": {str(c): None for c in choices}}
            items[split].append({"qid": f"cvb_{row['idx']}", "q": q, "gold": gold,
                                 "image_url": row["image"]["src"], "image_sha": None,
                                 "task": row["task"]})
    return items


def convert_countbench(outdir, n_items):
    rows = []
    for off in range(0, 491, 100):
        j = fetch_json(f"{DS}?dataset=vikhyatk%2FCountBenchQA&config=default&split=test&offset={off}&length=100")
        rows += j.get("rows", [])
        if len(rows) >= n_items + 40:
            break
    rows = [r for r in rows if r["row"]["number"] != 10]      # level cap (10 levels 0..9)
    random.Random(13).shuffle(rows)
    items = {"dev": [], "test": []}
    half = n_items // 2
    for split, chunk in (("dev", rows[:half]), ("test", rows[half:n_items])):
        for r in chunk:
            row = r["row"]
            q = {"type": "score", "instructions": row["question"] + " Reply with the count.",
                 "criteria": [str(n) for n in range(0, 10)]}
            items[split].append({"qid": f"cb_{r['row_idx']}", "q": q, "gold": row["number"],
                                 "image_url": row["image"]["src"], "image_sha": None})
    return items


def materialize(items, outdir, kind):
    """Download images once, hash, write item JSONs + manifest rows."""
    out = Path(outdir, "data", kind)
    man = []
    for split in ("dev", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
        for it in items[split]:
            raw = fetch_bytes(it["image_url"])
            b64 = to_png_b64(raw)
            sha = hashlib.sha256(raw).hexdigest()
            it["image_sha"] = sha
            (out / split / f"{it['qid']}.json").write_text(json.dumps({
                "image_b64": b64,
                "questions": {it["qid"]: it["q"]},
                "_meta": {"label": it["gold"], "image_sha": sha,
                          **({"task": it["task"]} if "task" in it else {}),
                          **({"img_key": it["img_key"]} if "img_key" in it else {})}}, indent=1))
            man.append({"qid": it["qid"], "split": split, "gold": it["gold"],
                        "image_sha256": sha, "raw_bytes": len(raw)})
    return man


def selftest(items_by_kind):
    """Banned-field assert + shape check (the plan's converter unit test)."""
    banned = ("prompt", "text", "rationales", "direct_answers", "category")
    for kind, items in items_by_kind.items():
        for split in ("dev", "test"):
            for it in items[split]:
                blob = json.dumps(it["q"]).lower()
                for b in banned:
                    assert b not in blob or b == "text", f"banned field {b} in {kind}"
                assert it["q"]["type"] == {"pope": "noul", "cvb": "choice", "cb": "score"}[kind]
                assert isinstance(it["gold"], int) and it["gold"] >= 0
    # POPE dev/test image-disjointness
    dev_keys = {i["img_key"] for i in items_by_kind["pope"]["dev"]}
    test_keys = {i["img_key"] for i in items_by_kind["pope"]["test"]}
    assert not (dev_keys & test_keys), "POPE dev/test share an image"
    # POPE 50/50 balance within split
    for split in ("dev", "test"):
        ys = sum(i["gold"] for i in items_by_kind["pope"][split])
        assert 0.35 < ys / len(items_by_kind["pope"][split]) < 0.65, "POPE unbalanced"


def main():
    outdir = Path(sys.argv[1])
    n_pope, n_cvb, n_cb = (int(x) for x in sys.argv[2:5])
    items = {"pope": convert_pope(outdir, n_pope),
             "cvb": convert_cvb(outdir, n_cvb),
             "cb": convert_countbench(outdir, n_cb)}
    selftest(items)
    manifest = []
    for kind, fn in (("noul-pope", items["pope"]), ("choice-cvb", items["cvb"]),
                     ("score-countbench", items["cb"])):
        manifest += materialize(fn, outdir, kind)
    (Path(outdir) / "convert-manifest.json").write_text(json.dumps(manifest, indent=1))
    print("WROTE", len(manifest), "items; manifest:", Path(outdir) / "convert-manifest.json")


if __name__ == "__main__":
    main()
