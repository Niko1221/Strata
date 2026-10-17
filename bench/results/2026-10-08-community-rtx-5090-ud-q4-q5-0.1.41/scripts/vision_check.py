#!/usr/bin/env python3
"""vision_check.py - does the running Strata server read images correctly? A generated picture with known answers.

    python3 vision_check.py --port 8080 --out ~/vision-check/q5      # needs Pillow (the Strata .venv has it)

The picture (saved as image.png in --out): a red circle, a blue square, and the number 4271 in large black digits,
on white. Three greedy questions with one expected word each; the answers, timings and the server's /v1/status go to
--out/result.json. Exit code 0 when all three answers contain the expected word.
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

QUESTIONS = [("What number is written in this image? Reply with the digits only.", "4271"),
             ("What color is the circle in this image? Reply with one word.", "red"),
             ("What shape is the blue object in this image? Reply with one word.", "square")]


def picture():
    im = Image.new("RGB", (768, 512), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 260, 260), fill=(220, 20, 20))
    d.rectangle((508, 60, 708, 260), fill=(20, 60, 220))
    try:
        font = ImageFont.load_default(size=140)
    except TypeError:          # Pillow < 10.1
        font = ImageFont.load_default()
    d.text((384, 390), "4271", fill="black", font=font, anchor="mm")
    return im


def ask(port, png_b64, question):
    body = {"model": "strata", "temperature": 0, "reasoning_effort": "none", "max_tokens": 24,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64}},
                {"type": "text", "text": question}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return (msg.get("content") or "").strip(), round(time.time() - t0, 2), out.get("usage")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    im = picture()
    im.save(a.out / "image.png")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/status", timeout=10) as r:
        status = json.load(r)
    rows, ok = [], 0
    for q, want in QUESTIONS:
        try:
            ans, s, usage = ask(a.port, b64, q)
        except Exception as e:  # noqa: BLE001 - a refused image is a result
            ans, s, usage = f"ERROR: {e}", None, None
        good = want.lower() in ans.lower()
        ok += good
        rows.append({"question": q, "expected": want, "answer": ans, "ok": good, "seconds": s, "usage": usage})
        print(f"{'ok  ' if good else 'FAIL'} {want!r:10} <- {ans!r} ({s} s)")
    res = {"model": status.get("model"), "engine": status.get("engine"), "passed": ok, "of": len(QUESTIONS),
           "rows": rows}
    (a.out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{status.get('model')} engine {status.get('engine')}: {ok}/{len(QUESTIONS)}")
    sys.exit(0 if ok == len(QUESTIONS) else 1)


if __name__ == "__main__":
    main()
