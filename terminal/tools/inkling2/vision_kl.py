"""Inkling-2-Mini with a picture, end to end (transformers PR #51050 on the CPU): all-position logits of a chat prompt
with one picture, and its greedy continuation, for strata-ink2 --image --dump-logits (compare with kl_cmp-style KL).

usage: vision_kl.py --model DIR --out PREFIX --image FILE [--image FILE ...] [--dtype float32] [--gen N]

Per picture k (PREFIX.k.*): .pt {"ids": the processor-expanded prompt, "logits": [T][V] fp32, "gen": greedy ids},
.ids1 (the prompt with one <|unused_200054|> per picture: what strata-ink2 --tokens-file takes with --image).
One model load for all pictures.
"""
import argparse, json, time
import torch
from transformers import AutoProcessor, Inkling2LoopForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--prompt", default="Describe this picture in one sentence.")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    proc = AutoProcessor.from_pretrained(a.model)
    t = time.time()
    model = Inkling2LoopForConditionalGeneration.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cpu").eval()
    print(f"loaded in {time.time() - t:.0f}s", flush=True)
    for k, img in enumerate(a.image):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": a.prompt}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", reasoning_effort=a.effort)
        ids = inp["input_ids"][0].tolist()
        one, prev = [], None
        for x in ids:
            if not (x == proc.image_token_id and prev == proc.image_token_id):
                one.append(x)
            prev = x
        open(f"{a.out}.{k}.ids1", "w").write(",".join(map(str, one)))
        t = time.time()
        with torch.no_grad():
            logits = model(**inp, use_cache=False).logits[0].float()
            fwd = time.time() - t
            g = model.generate(**inp, max_new_tokens=a.gen, do_sample=False)
        gen = g[0, len(ids):].tolist()
        torch.save({"ids": inp["input_ids"][0], "logits": logits, "gen": gen}, f"{a.out}.{k}.pt")
        print(json.dumps({"image": img, "grid": inp["image_grid_thw"][0].tolist(), "tokens": len(ids), "fwd_s": round(fwd, 1),
                          "gen_ids": gen, "gen_text": proc.tokenizer.decode(gen)}, ensure_ascii=False), flush=True)
    print("VISION-KL-DONE", flush=True)


if __name__ == "__main__":
    main()
