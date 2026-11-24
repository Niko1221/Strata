"""Check a candidate Inkling engine against a captured Responses request, without installing it.

Run with Strata stopped so the candidate has the GPU to itself. This checks greedy
output against cold prefill and verifies the engine's reported shared-prefix reuse.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TERMINAL), str(TERMINAL / "tools")]
from serve import frontend, inkling_dialect, responses
from strata_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--draft", choices=("auto", "off"), default="auto")
    args = parser.parse_args()
    vocab = json.loads((args.tokenizer / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    cfg = json.loads((args.tokenizer / "tokenizer.json").read_text())
    tok = Tokenizer(tokens, (args.tokenizer / "merges.txt").read_text().split("\n"),
                    json.loads((args.tokenizer / "token_type.json").read_text()),
                    cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)
    template = frontend.ChatTemplate(args.tokenizer / "chat_template.jinja")
    req = json.loads(args.request.read_text())
    tools = responses.prompt_tools(req, responses.request_tools(req)[0])
    messages = inkling_dialect.prepare_messages(responses.input_messages(req))
    count = 0
    for message in messages:
        if message.get("role") not in ("system", "developer"):
            break
        count += 1
    if count == 0:
        raise SystemExit("The capture must contain leading system instructions.")
    kw = inkling_dialect.template_kwargs({})
    prefix = tok.encode(template.render(messages[:count], tools=tools, add_generation_prompt=False, **kw))

    def prompt(digit):
        head = copy.deepcopy(messages[:count])
        text = template.render(head + [{"role": "user", "content": f"Reply with exactly the single digit {digit}."}],
                               tools=tools, **kw)
        return tok.encode(inkling_dialect.finish_prompt(text, has_tools=bool(tools)))

    a, b = prompt(3), prompt(5)
    if a[:len(prefix)] != prefix or b[:len(prefix)] != prefix:
        raise SystemExit("The rendered system/tool boundary is not a token prefix.")
    reset_tok = 2 if (a[0] == 1 or b[0] == 1) else 1
    # K8V8 and two slots match this installation; --draft off isolates speculative decoding.
    command = [str(args.engine.resolve()), "--native", str(args.model.resolve()), "--serve",
               "--max-context", str(max(16384, len(a) + 1024)), "--parallel", "2", "--kv", "k8v8", "--draft", args.draft]
    engine = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def line():
        value = engine.stdout.readline()
        if not value:
            raise RuntimeError(f"engine exited ({engine.poll()}); see its stderr")
        if value.startswith("ERR "):
            raise RuntimeError(value.strip())
        return value

    def generate(ids, pin=None, max_new=8):
        option = f" pin={pin}" if pin else ""
        engine.stdin.write(f"GEN {max_new} temperature=0{option} " + ",".join(map(str, ids)) + "\n")
        engine.stdin.flush()
        emitted = []
        while True:
            value = line()
            if value.startswith("T "):
                emitted.append(int(value.split()[1]))
            if value.startswith("DONE "):
                return emitted, int(value.split()[-1])

    def record(label, ids, pin=None, cold=False):
        if cold:
            # A one-token unrelated request forces the next request to reset this slot.
            generate([reset_tok], max_new=0)
        output, reused = generate(ids, pin)
        print(json.dumps({"case": label, "draft": args.draft, "reused": reused,
                          "tokens": output, "text": tok.decode(output)}, ensure_ascii=True), flush=True)
        return output, reused

    try:
        chunk = 0
        while True:
            value = line()
            if value.startswith("INFO "):
                fields = dict(part.split("=", 1) for part in value.split()[1:] if "=" in part)
                chunk = int(fields.get("chunk", 0))
            if value.startswith("READY "):
                break
        if chunk <= 0:
            raise RuntimeError("engine did not report its prefill chunk size")
        boundary = (len(prefix) // chunk) * chunk
        if boundary <= 0:
            raise RuntimeError("shared prefix is shorter than one prefill chunk")
        print(f"Shared boundary: {len(prefix)} tokens; aligned pin={boundary}; chunk={chunk}; draft={args.draft}", flush=True)
        # Compare like-for-like cold runs before judging checkpoint restoration.
        unpinned_a1, r0 = record("cold_A_unpinned_1", a, cold=True)
        unpinned_a2, r1 = record("cold_A_unpinned_2", a, cold=True)
        pinned_a1, r2 = record("cold_A_pinned_1", a, boundary, cold=True)
        pinned_a2, r3 = record("cold_A_pinned_2", a, boundary, cold=True)
        warm_b, rb = record("warm_B_from_A", b, boundary)
        warm_a, ra = record("warm_A_from_B", a, boundary)
        cold_b_pinned, r4 = record("cold_B_pinned", b, boundary, cold=True)
        cold_b_unpinned, r5 = record("cold_B_unpinned", b, cold=True)
        checks = {
            "all_cold_runs_reused_zero": all(r == 0 for r in (r0, r1, r2, r3, r4, r5)),
            "unpinned_cold_A_repeat": unpinned_a1 == unpinned_a2,
            "pinned_cold_A_repeat": pinned_a1 == pinned_a2,
            "prefill_split_A_parity": pinned_a1 == unpinned_a1,
            "shared_B_reuse": rb == boundary,
            "shared_A_reuse": ra == boundary,
            "restore_B_matches_same_split_cold": warm_b == cold_b_pinned,
            "restore_A_matches_same_split_cold": warm_a == pinned_a2,
            "prefill_split_B_parity": cold_b_pinned == cold_b_unpinned,
        }
        print(json.dumps({"checks": checks}, indent=2), flush=True)
        if not all(checks.values()):
            raise RuntimeError("GPU checks failed: " + ", ".join(k for k, ok in checks.items() if not ok))
        print(f"PASS: {boundary} shared tokens reused; cold repeats, split prefill and restoration agree.")
    finally:
        # Always reap the CUDA process, including startup errors and broken stdin pipes.
        try:
            if engine.poll() is None:
                engine.stdin.write("QUIT\n")
                engine.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            engine.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            engine.wait(timeout=20)
        except subprocess.TimeoutExpired:
            engine.kill()
            engine.wait()
        engine.stdout.close()


if __name__ == "__main__":
    main()
