"""Inkling-2-Mini image path reference (transformers PR #51050, fp32 on the CPU), for strata-ink2's ViT + projector.

usage: vision_oracle.py --model DIR --image FILE --out PREFIX [--prompt TEXT] [--layers]

Writes, for one picture through the model's own processor:
  PREFIX.pix    the processor's pixel tensor: int32 rows, int32 cols, then float32 [rows*cols][40][40][3]
                (strata-ink2 --vis-pixels reads it: the ViT on exactly these numbers)
  PREFIX.emb    the vision tower's output (one row per image token): float32 [rows*cols][2304]
  PREFIX.ids    the chat prompt with the picture (processor-expanded placeholders), comma-separated token ids
  PREFIX.ids1   the same prompt with one placeholder per picture (what Strata's server sends the engine)
  PREFIX.layers (--layers) the ViT hidden states after embeddings+ln_pre and after each layer: float32 [24][1+4n][1024]
Only the vision tower is loaded (its 422 tensors), so this runs in seconds; kl_run.py-style full-model logits of the
multimodal prompt are vision_kl.py's job.
"""
import argparse, json, re
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

RENAMES = [(r"^backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
           (r"^backbone\.embeddings\.", "embeddings."), (r"^backbone\.ln_pre\.", "pre_layernorm."),
           (r"^backbone\.layers\.", "layers."), (r"^proj_in\.", "merger.linear_1."), (r"^proj_out\.", "merger.linear_2."),
           (r"\.ln_1\.", ".layernorm_before."), (r"\.ln_2\.", ".layernorm_after."), (r"\.mlp\.c_fc\.", ".mlp.fc1."),
           (r"\.mlp\.c_proj\.", ".mlp.fc2."), (r"\.ls_1\.gamma$", ".lambda_1"), (r"\.ls_2\.gamma$", ".lambda_2")]


def load_vision(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    vm = Inkling2VisionModel(cfg.vision_config).float().eval()
    sd = {}
    with safe_open(f"{model_dir}/model-00001-of-00001.safetensors", "pt") as f:
        for k in f.keys():
            if not k.startswith("model.visual."):
                continue
            n = k[len("model.visual."):]
            for a, b in RENAMES:
                n = re.sub(a, b, n)
            sd[n] = f.get_tensor(k).float()
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Describe this picture in one sentence.")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--layers", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(16)
    proc = AutoProcessor.from_pretrained(a.model)
    msgs = [{"role": "user", "content": [{"type": "image", "image": a.image}, {"type": "text", "text": a.prompt}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                   reasoning_effort=a.effort)
    pix, grid = inp["pixel_values"].float(), inp["image_grid_thw"]
    t, rows, cols = grid[0].tolist()
    ids = inp["input_ids"][0].tolist()
    img = proc.image_token_id
    one, prev = [], None
    for x in ids:                      # one placeholder per picture, as the template renders it before expansion
        if not (x == img and prev == img):
            one.append(x)
        prev = x
    with open(a.out + ".pix", "wb") as f:
        np.array([rows, cols], dtype=np.int32).tofile(f)
        pix.numpy().astype(np.float32).tofile(f)
    open(a.out + ".ids", "w").write(",".join(map(str, ids)))
    open(a.out + ".ids1", "w").write(",".join(map(str, one)))
    vm = load_vision(a.model)
    with torch.no_grad():
        out = vm(pixel_values=pix, image_grid_thw=grid, output_hidden_states=a.layers)
    out.pooler_output.float().numpy().tofile(a.out + ".emb")
    if a.layers:
        torch.stack([h.float() for h in out.hidden_states]).numpy().tofile(a.out + ".layers")
    print(json.dumps({"image": a.image, "grid": [t, rows, cols], "pix_shape": list(pix.shape), "tokens": len(ids),
                      "image_tokens": ids.count(img), "pix_min": pix.min().item(), "pix_max": pix.max().item(),
                      "hidden_states": len(out.hidden_states) if a.layers else 0}))


if __name__ == "__main__":
    main()
