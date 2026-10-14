#!/usr/bin/env python3
"""Hand the GPU between Strata (this box) and ComfyUI (another box) over HTTP.

ComfyUI's own controls (ComfyUI/server.py, ComfyUI/main.py):
  POST /free    {"unload_models": true, "free_memory": true}  unload every model + empty the cache
  POST /prompt  {"prompt": {...}, "front": true}              queue a workflow at the head of the queue
  GET  /history/{prompt_id}   GET /view?filename=..   GET /system_stats   GET /queue
  POST /interrupt {"prompt_id": ...}   POST /queue {"clear": true} / {"delete": [id]}
Strata's controls (serve/server.py, docs/DETAILS.md "Sharing the GPU"):
  POST /unload   POST /load   POST /v1/vram {"reserve_mib": N}   (the last needs "vram_elastic": true)
  POST /v1/vram {"reserve_mib": N, "gpu": 1}  - on a layer split, one card's cache only (0 = the main card)

Strata's restore call is `{"reserve_mib": null}` (the reserve it started with); ComfyUI's is POST /free,
which drops the weights and lets the next prompt load them again.  So one image run is: give the GPU to
Comfy, queue, wait, hand the GPU back.

Examples:
  python tools/comfy_vram_handoff.py status
  python tools/comfy_vram_handoff.py run wf.json --prompt "a poster that says HELLO" --out out/
  python tools/comfy_vram_handoff.py run wf.json --reserve 6000        # soft handoff, no full unload
  python tools/comfy_vram_handoff.py run wf.json --reserve 16000 --gpu 1   # one card of a 2-card split
  python tools/comfy_vram_handoff.py free                              # Comfy gives its VRAM back
  python tools/comfy_vram_handoff.py unload --lazy                    # Strata gives its VRAM back

Env: COMFY_URL (default http://127.0.0.1:8188), STRATA_URL (default http://127.0.0.1:8080),
     STRATA_API_KEY.  ComfyUI has no authentication: keep its port on the local network only.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# the nodes that take the prompt in ComfyUI's Qwen-Image-2.1 templates
PROMPT_NODES = {"TextEncodeQwenImage21", "TextEncodeQwenImageEdit", "TextEncodeQwenImageEditPlus"}


def http(method, url, payload=None, key="", timeout=60.0):
    """One request; returns (status, parsed body or text)."""
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")   # Strata's /unload, /load, /v1/vram need JSON
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    try:
        return 200, json.loads(raw)
    except json.JSONDecodeError:
        return 200, raw


class Comfy:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def vram(self):
        code, s = http("GET", f"{self.url}/system_stats")
        if code != 200:
            return None
        return [(g["name"], g["vram_free"], g["vram_total"]) for g in s.get("gpu", [])]

    def busy(self):
        _, q = http("GET", f"{self.url}/queue")
        return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))

    def free(self, unload=True, memory=True):
        """POST /free: the flags are consumed by the worker thread, so they land when the running
        prompt finishes, and within ~1 s when the queue is idle (main.py: flags = q.get_flags())."""
        return http("POST", f"{self.url}/free", {"unload_models": unload, "free_memory": memory})

    def queue(self, workflow, front=True):
        code, r = http("POST", f"{self.url}/prompt", {"prompt": workflow, "front": front})
        if code != 200:
            raise SystemExit(f"ComfyUI rejected the workflow ({code}): {json.dumps(r)[:1500]}")
        return r["prompt_id"]

    def wait(self, prompt_id, timeout):
        """Poll /history until the job lands there; returns its outputs."""
        end, shown = time.time() + timeout, False
        while time.time() < end:
            _, h = http("GET", f"{self.url}/history/{prompt_id}")
            if prompt_id in h:
                st = h[prompt_id].get("status") or {}
                if st.get("status_str") == "error":
                    raise SystemExit(f"ComfyUI failed: {json.dumps(st)[:1500]}")
                return h[prompt_id].get("outputs", {})
            if not shown:
                print(f"  queued {prompt_id}; running ...", flush=True)
                shown = True
            time.sleep(0.5)
        http("POST", f"{self.url}/interrupt", {"prompt_id": prompt_id})
        raise SystemExit(f"timed out after {timeout} s waiting for {prompt_id} (interrupted)")

    def download(self, outputs, outdir):
        files = []
        Path(outdir).mkdir(parents=True, exist_ok=True)
        for node in outputs.values():
            for img in node.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                with urllib.request.urlopen(f"{self.url}/view?{q}", timeout=120) as r:
                    dest = Path(outdir) / img["filename"]
                    dest.write_bytes(r.read())
                files.append(str(dest))
        return files


class Strata:
    def __init__(self, url, key=""):
        self.url = url.rstrip("/")
        self.key = key

    def post(self, path, payload=None):
        code, r = http("POST", f"{self.url}{path}", payload if payload is not None else {}, key=self.key)
        return code, r

    def unload(self, timeout):
        """POST /unload: 409 while a request runs; 503 while the engine is still releasing memory."""
        end = time.time() + timeout
        while True:
            code, r = self.post("/unload")
            body = r if isinstance(r, str) else json.dumps(r)
            if code == 200 and (isinstance(r, dict) and r.get("status") in ("unloaded", "not loaded")):
                return r
            if code in (409, 503) and time.time() < end:
                time.sleep(2.0)
                continue
            raise SystemExit(f"Strata /unload failed ({code}): {body[:400]}")

    def load(self):
        return self.post("/load")

    def reserve(self, mib, gpu=None):
        """POST /v1/vram: shrink the expert cache so `mib` MiB stays free for Comfy (needs
        "vram_elastic": true); None grows it back.  `gpu` is Strata's GPU number of a layer split
        (0 = the main card); None: every GPU of the run.  A running request finishes first."""
        payload = {"reserve_mib": mib}
        if gpu is not None:
            payload["gpu"] = gpu
        code, r = self.post("/v1/vram", payload)
        if code != 200:
            raise SystemExit(f"Strata /v1/vram failed ({code}): {json.dumps(r)[:400]}")
        return r


def apply_edits(workflow, args):
    """--prompt into the Qwen text-encode nodes, --seed into KSamplers, --set NODE.KEY=VALUE."""
    touched = 0
    for node in workflow.values():
        cls, inputs = node.get("class_type", ""), node.get("inputs", {})
        if args.prompt and cls in PROMPT_NODES and "text" in inputs:
            inputs["text"] = args.prompt
            touched += 1
        if args.seed is not None and cls == "KSampler" and "seed" in inputs:
            inputs["seed"] = args.seed
    for spec in args.set:
        nid, rest = spec.split(".", 1)
        key, raw = rest.split("=", 1)
        if nid not in workflow:
            raise SystemExit(f"--set: node {nid} is not in the workflow")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        workflow[nid]["inputs"][key] = value
    if args.prompt and not touched:
        print(f"  note: no prompt node ({', '.join(sorted(PROMPT_NODES))}) found; the workflow's own text is used",
              file=sys.stderr)
    return workflow


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    ap.add_argument("--strata", default=os.environ.get("STRATA_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--strata-key", default=os.environ.get("STRATA_API_KEY", ""))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="VRAM free on both boxes")
    p = sub.add_parser("run", help="unload the LLM -> generate -> free Comfy -> reload the LLM")
    p.add_argument("workflow", help="workflow JSON in ComfyUI's API format (Save (API Format))")
    p.add_argument("--prompt")
    p.add_argument("--seed", type=int)
    p.add_argument("--set", action="append", default=[], metavar="NODE.KEY=VALUE")
    p.add_argument("--out", default="comfy-out")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--reserve", type=int, metavar="MIB", help="shrink Strata's VRAM use by reserving MIB for "
                                                              "Comfy instead of unloading it (needs vram_elastic)")
    p.add_argument("--gpu", type=int, metavar="N", help="with --reserve: take it from that GPU of a layer split "
                                                        "only (0 = the main card); default: every GPU")
    p.add_argument("--no-llm", action="store_true", help="no Strata on this GPU: only talk to ComfyUI")
    p.add_argument("--keep-loaded", action="store_true", help="leave ComfyUI's models in VRAM after the run")
    p.add_argument("--no-reload", action="store_true", help="do not pre-load the LLM; its next request loads it")
    sub.add_parser("free", help="ComfyUI: unload its models and empty the cache")
    sub.add_parser("unload", help="Strata: unload the model now")
    sub.add_parser("load", help="Strata: load the model now")
    p = sub.add_parser("vram", help="Strata: POST /v1/vram")
    p.add_argument("reserve_mib", type=int, help="MiB to keep free for other programs (0: take it all back)")
    p.add_argument("--gpu", type=int, metavar="N", help="only that GPU of a layer split (0 = the main card); "
                                                        "default: every GPU of the run")

    args = ap.parse_args()
    comfy, strata = Comfy(args.comfy), Strata(args.strata, args.strata_key)

    def show_vram():
        for name, free, total in (comfy.vram() or []):
            print(f"  comfy {name}: {free / 1024**2:.1f} GiB free of {total / 1024**2:.1f}")

    if args.cmd == "status":
        show_vram()
        code, r = http("GET", f"{strata.url}/v1/status")
        print(f"  strata: {json.dumps(r.get('vram') if code == 200 and isinstance(r, dict) else r)[:300]}")
        return

    if args.cmd == "free":
        print(comfy.free(), "-> ComfyUI unloads at the end of the running prompt (or within ~1 s when idle)")
        time.sleep(1.5)
        show_vram()
        return

    if args.cmd == "unload":
        print(" ->", strata.unload(args.timeout if hasattr(args, "timeout") else 60))
        return

    if args.cmd == "load":
        print(" ->", strata.load())
        return

    if args.cmd == "vram":
        print(" ->", strata.reserve(args.reserve_mib, args.gpu))
        return

    # run
    workflow = apply_edits(json.loads(Path(args.workflow).read_text(encoding="utf-8-sig")), args)
    if not args.no_llm:
        if args.reserve:
            print(f"giving Comfy {args.reserve} MiB of VRAM (Strata keeps serving) ...", flush=True)
            strata.reserve(args.reserve, args.gpu)
        else:
            print("unloading the Strata model ...", flush=True)
            strata.unload(60)
    print("queueing the workflow ...", flush=True)
    prompt_id = comfy.queue(workflow)
    outputs = comfy.wait(prompt_id, args.timeout)
    files = comfy.download(outputs, args.out)
    print(f"{len(files)} file(s): {', '.join(files) or 'none'}")
    if not args.keep_loaded:
        comfy.free()
        time.sleep(1.5)
        show_vram()
    if args.reserve and not args.no_llm:
        strata.reserve(None, args.gpu)
    elif not args.no_llm and not args.no_reload:
        strata.load()


if __name__ == "__main__":
    main()
