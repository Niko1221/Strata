import subprocess, sys
sys.path.insert(0, "tools")
import strata_tokenizer as T

gguf, exe = sys.argv[1], "build/src/qwen36/strata-qwen36"
tok = T.Tokenizer.from_gguf(gguf)
prompt = "The capital of France is"
ids = tok.encode(prompt, parse_special=False)
eos = tok.encode("<|im_end|>", parse_special=True)

p = subprocess.Popen([exe, "--serve", "--native", gguf, "--max-context", "4096", "--eos-ids", ",".join(map(str, eos))],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
while not p.stdout.readline().startswith("READY"):
    pass
p.stdin.write(f"GEN 40 temperature=0 {','.join(map(str, ids))}\n"); p.stdin.flush()
out = []
for line in p.stdout:
    if line.startswith("T "): out.append(int(line.split()[1]))
    if line.startswith(("DONE", "ERR")): print(line.strip()); break
print(prompt + tok.decode(out))
p.stdin.write("QUIT\n"); p.stdin.flush()
