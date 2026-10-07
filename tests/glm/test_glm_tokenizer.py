"""tests/glm/test_glm_tokenizer.py - Strata's tokenizer with the `glm` pre-tokenizer against Hugging Face `tokenizers`.

GLM-5.3 ships a tokenizer.json (no GGUF), and the server tokenizes with tools/strata_tokenizer.py, so the two must give
the same ids.  `tokenizers` is the reference implementation of that file; it is not one of Strata's dependencies, so
without it this test says SKIPPED and exits 2 (a test that did not find its reference verified nothing).

    python tests/glm/test_glm_tokenizer.py --model D:/models/GLM-5.3-colibri-int4-g64
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

CORPUS = [
    "The capital of France is",
    "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "\r\n\r\n", "tabs\tand\tspaces    end",
    "def f(x):\n\treturn x  # comment\n", "for (int i = 0; i < 10; ++i) { sum += a[i]; }",
    "1234567890", "3.14159 and 2,718,281 and 10^100", "007 James Bond", "Ⅻ ½ ²³ ١٢٣",
    "你好，世界！今天天气很好。", "日本語のテキストとカタカナ", "한국어 문장입니다", "مرحبا بالعالم", "Привет, мир",
    "😀🚀🇺🇸 emoji and 👨‍👩‍👧 family", "e\u0301\u0301 combining marks", "\x00\x01\x7f control bytes",
    "\u00a0non-breaking\u00a0space", "MixedCASE_and-dashes", "I'm you're they'll we'd IT'S", "\u2028\u2029",
    "<think>reasoning</think>answer", "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "[gMASK]<sop><|system|>You are helpful.<|user|>Hi<|assistant|><think></think>Hello",
    "<|observation|><tool_response>{\"temp\": 21}</tool_response>",
    "x" * 3000, "ab " * 500,
    "https://example.com/path?q=1&r=2#frag", "SELECT * FROM t WHERE a='b';",
    "    indented\n        more\n",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the GLM-5.3 model folder (its tokenizer.json)")
    a = ap.parse_args()
    try:
        from tokenizers import Tokenizer as HF
    except ImportError:
        print("SKIPPED: the reference (pip package `tokenizers`) is not installed")
        return 2
    import strata_tokenizer as ST
    path = pathlib.Path(a.model) / "tokenizer.json"
    ref = HF.from_file(str(path))
    mine = ST.Tokenizer.from_hf_json(path, "glm")
    bad = 0
    for s in CORPUS:
        want = ref.encode(s, add_special_tokens=False).ids
        got = mine.encode(s, parse_special=True)
        back = mine.decode(got)
        ok = got == want and back == s
        if not ok:
            bad += 1
            print(f"FAIL {s[:40]!r}: ours {got[:12]}... ref {want[:12]}... round trip {back == s}")
    print(f"{len(CORPUS) - bad} of {len(CORPUS)} strings: the same ids as tokenizers, and decode(encode(s)) == s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
