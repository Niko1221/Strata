# Inkling-2-Mini validation tools

How `strata-ink2ref` and `strata-ink2` were checked against the model's reference implementation, the transformers
pull request huggingface/transformers#51050, run in fp32 on the CPU.

1. `setup-venv.sh DIR`: a venv with that pull request and CPU torch.
2. `oracle.py --model CKPT --out DIR`: per-layer-application outputs and logits of a few prompts, plus their greedy
   continuations; `ref_run.sh PROMPT` compares `strata-ink2ref --dump-layers --dump-logits` with them layer by layer.
3. `kl_run.py --model CKPT --text FILE --out X.pt`: all-position logits of a long text (`--mode w4a4` reproduces vLLM's
   quantized expert activations); `cmp_logits.py` and `kl_cmp.py` give KL, max logit difference and top-1 agreement
   against an engine's `--dump-logits`; `long_check.sh` runs the long-context and position-offset cases.
4. `tok_check.py CKPT REPO [TEXT...]`: Strata's tokenizer against the Hugging Face one on a corpus and chat prompts;
   `pack_check.py` loads the extracted tokenizer pack the way the server does.
5. Pictures: `vision_oracle.py --model CKPT --image FILE --out P` writes the processor's pixel tensor (`P.pix`) and the
   vision tower's rows (`P.emb`); `strata-ink2 --vis-pixels P.pix --dump-image-emb X` runs the engine's ViT on the same
   pixels and `emb_cmp.py P.emb X` compares them (max |d|, cosine per token).  `vision_kl.py --model CKPT --image FILE
   --out P` gives the full multimodal prompt's logits and greedy answer (`P.0.pt`, the prompt with one placeholder per
   picture in `P.0.ids1`); `strata-ink2 --tokens-file P.0.ids1 --image FILE --dump-logits L` then `vis_cmp.py P.0.pt L`
   give KL and top-1 before, inside and after the picture.  Use PNG (or BMP) for strict comparisons: the reference
   decodes JPEG with libjpeg-turbo, the engine with stb_image (the server hands strata-ink2 Pillow-decoded JPEGs).
