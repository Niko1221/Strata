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
