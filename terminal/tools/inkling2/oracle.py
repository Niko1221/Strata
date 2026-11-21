"""Inkling-2-Mini oracle: run the transformers PR #51050 reference on CPU and dump per-layer-application tensors.

usage: oracle.py --dtype float32|bfloat16 --out DIR [--prompts prompts.json] [--gen N]

Dumps (safetensors, one file per prompt):
  ids, logits (last position + all positions), and for each decoder-layer call k (0..47, in call order):
  call{k}.layer, call{k}.in_h, call{k}.in_B, call{k}.msgs (loop passes 1-2 only), call{k}.out
Also the greedy continuation (--gen tokens) as text + ids in summary.json.
"""
import argparse, json, os, time
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

MODEL = "/var/fastasfuck/inkling2/Inkling-2-Mini"

DEFAULT_PROMPTS = [
    {"name": "raw_en", "text": "The capital of France is"},
    {"name": "raw_fr", "text": "Le petit chat est monté sur le toit parce que"},
    {"name": "raw_code", "text": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"},
    {"name": "chat_sky", "chat": [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    dtype = getattr(torch, a.dtype)
    prompts = json.load(open(a.prompts)) if a.prompts else DEFAULT_PROMPTS

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(MODEL, dtype=dtype, device_map="cpu")
    model.eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layers = list(text_model.layers)
    calls = []

    def hook(mod, args, kwargs, out):
        if not calls_enabled[0]:
            return
        h = args[0] if args else kwargs["hidden_states"]
        B = args[1] if len(args) > 1 else kwargs["block_states"]
        msgs = args[2] if len(args) > 2 else kwargs.get("messages")
        calls.append({"layer": layers.index(mod), "in_h": h, "in_B": B, "msgs": msgs, "out": out})

    calls_enabled = [True]
    for L in layers:
        L.register_forward_hook(hook, with_kwargs=True)

    summary = {"dtype": a.dtype, "prompts": []}
    for p in prompts:
        if "chat" in p:
            ids = tok.apply_chat_template(p["chat"], tokenize=True, add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        else:
            ids = tok(p["text"], return_tensors="pt").input_ids
        calls.clear()
        calls_enabled[0] = True
        t = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        fwd_s = time.time() - t
        assert len(calls) == 48, len(calls)
        tensors = {"ids": ids[0].to(torch.int64).contiguous(), "logits": out.logits[0].float().contiguous().clone()}
        for k, c in enumerate(calls):
            tensors[f"call{k}.layer"] = torch.tensor([c["layer"]], dtype=torch.int64)
            tensors[f"call{k}.in_h"] = c["in_h"][0].float().contiguous().clone()
            tensors[f"call{k}.in_B"] = c["in_B"][0].float().contiguous().clone()
            tensors[f"call{k}.out"] = c["out"][0].float().contiguous().clone()
            if c["msgs"] is not None:
                tensors[f"call{k}.msgs"] = c["msgs"][0].float().contiguous().clone()
        save_file(tensors, os.path.join(a.out, f"{p['name']}.safetensors"))

        calls_enabled[0] = False
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=a.gen, do_sample=False)
        new = g[0, ids.shape[1]:].tolist()
        top5 = torch.topk(out.logits[0, -1].float(), 5)
        rec = {"name": p["name"], "n_tokens": ids.shape[1], "fwd_s": round(fwd_s, 1),
               "top5_ids": top5.indices.tolist(), "top5_logits": [round(x, 4) for x in top5.values.tolist()],
               "gen_ids": new, "gen_text": tok.decode(new)}
        summary["prompts"].append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print("ORACLE-DONE", flush=True)


if __name__ == "__main__":
    main()
