"""Teacher-forced precision of the KV cache formats, measured against --kv fp16 with one engine
binary, one token list and one expert profile: only --kv and the window environment differ.

This rebuilds tools/kv_precision_compare.py, the script the earlier KV results READMEs cite
(bench/results/2026-09-27-kv-q4/README.md, the section "Quality (teacher-forced,
tools/kv_precision_compare.py)"); the file itself was no longer in the tree.

WHERE THE ROWS COME FROM
  --dump-prompt-logits PATH writes one logits row per PROMPT position, from the PREFILL, where the chunk
  residual stream already exists: row k is the distribution the model had at position k with the KV state
  as it stood there, so row k predicts token k+1. That is the teacher-forcing a precision table needs.
  --dump-logits cannot serve this on a native (IQ) pack: it yields one row per GENERATED token and no
  prompt rows, which is why the 2026-10-11 write-up says that table is the one metric it is missing.

  The last row (position n_prompt-1) is a decode path row, not a prefill row, and its target token is the
  first GENERATED token, which is not in the prompt. So:
    * top-1 agreement and mean KL use EVERY row - they need no target;
    * perplexity and nats/token use rows 0..n_prompt-2, the rows whose target token is known.
  The row counts are printed, so this is visible rather than assumed.

METRICS (per arm, against the --kv fp16 reference on the same text)
  same top-1 as fp16 (%)   argmax agreement
  mean KL(fp16 || arm)     sum over v of p_fp16(v) * (log p_fp16(v) - log p_arm(v)), averaged over rows
  perplexity              exp(mean over positions of -log p_arm(token k+1))
  nats/token vs fp16      mean of (-log p_arm(t) + log p_fp16(t)) with SE = sd / sqrt(n), labelled
                          "indistinguishable" when the absolute delta is under 2 SE

THE ARMS
  fp16 (the reference), int8, q4_0, q2, q2k (K 2-bit per-channel + V 4-bit, 448 B/cell for 2 KV heads), q2 and
  q2k with STRATA_KV_WINDOW=128 STRATA_KV_SINK=8, and q4_0 with the same window environment.

COST, STATED PLAINLY
  strata generate (non-serve) takes ONE prompt per process and the engine takes about 40 s to load, so the
  harness pays the load once per (arm, text): 6 arms x 3 texts = 18 loads, roughly 12 to 20 minutes on an
  RTX 5080. Several prompts cannot be fed through one non-serve process, and the serve path has no
  prompt-logits sink, so the load cost is accepted rather than hidden.

  Run it with no arguments for every arm and every text; --arms and --texts take comma separated names to
  restrict the run. KV_ENGINE overrides the exe path, the KV_DEV_ENGINE-style override the other KV tools
  use. The engine arguments come from the args list of the serve configuration, with only --kv replaced;
  --no-capture is NOT set, because a native pack captures no graphs and the prompt path is what runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata_tokenizer as ST  # noqa: E402

ROOT = HERE.parents[0]
CONFIG = ROOT / "strata-swift-iq3_xxs.json"

ARMS = [
    dict(name="fp16", kv="fp16", env=dict()),
    dict(name="int8", kv="int8", env=dict()),
    dict(name="q4_0", kv="q4_0", env=dict()),
    dict(name="q2", kv="q2", env=dict()),
    # q2k: K 2-bit per-channel + V 4-bit (448 B/cell for 2 KV heads, against q2's 320 and q4_0's 576). It sits
    # between the two by construction, and the point of the arm is to check that on logits, not on the parity
    # probe's softmax: expect it under q2's top-1 loss and over q4_0's.
    dict(name="q2k", kv="q2k", env=dict()),
    dict(name="q2+window", kv="q2", env=dict(STRATA_KV_WINDOW="128", STRATA_KV_SINK="8")),
    dict(name="q2k+window", kv="q2k", env=dict(STRATA_KV_WINDOW="128", STRATA_KV_SINK="8")),
    dict(name="q4_0+window", kv="q4_0", env=dict(STRATA_KV_WINDOW="128", STRATA_KV_SINK="8")),
    # Window-size sweep. A window at least as long as the prompt covers every scored position, so the sweep
    # only means anything with texts LONGER than the largest window - pass --max-tokens above that.
    dict(name="q2+win512", kv="q2", env=dict(STRATA_KV_WINDOW="512", STRATA_KV_SINK="8")),
    dict(name="q2+win1024", kv="q2", env=dict(STRATA_KV_WINDOW="1024", STRATA_KV_SINK="8")),
    dict(name="q4_0+win512", kv="q4_0", env=dict(STRATA_KV_WINDOW="512", STRATA_KV_SINK="8")),
]
REFERENCE = "fp16"
CODE_AGENT = """Task: add a KV cache parity check to the engine and make the suite run in CI.

Read the existing suites first, in this order: the 4-bit parity suite, the int8 parity suite, and the
streamed-versus-resident parity suite with ring restore. They live under the kernels directory of the
source tree, one file per suite, and each one is registered with the test runner.

The new file must follow the same shape: a reference built in host memory, the kernel path writing into a
device buffer, and a comparison that reports the FIRST mismatching byte rather than a count. A count tells
you the size of a bug; an offset tells you where it is, and only one of the two can be acted on.

def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else len(b)

Notes from the last review, in the order they were raised:
1. The tie-break in the group maximum must be deterministic across lanes. Equal absolute value with
   opposite signs gave one block two scales, and the parity suite passed anyway because it compared the
   dequantized values, which agree for one of the two scales.
2. The host copy for the new format has to exist before the residency map is built, or the prompt path
   writes through the map entries marked -1. That is an out-of-bounds write with no symptom until the next
   request reads a K row that was never written.
3. Do not compare the selection ids before the selection blocks are made resident. The decode path used to
   run attention twice, and the first run read the blocks of the previous window.

Plan, in order: write the reference dequantization in the test, not in the library; write K, V and the
rotated Q through the kernel and read them back; compare bitwise and print the first mismatch together with
the bucket the cell falls in, which is one of sink, window and bulk; add the target to the build file and to
the test runner so the KV parity selection picks it up; run the six KV suites on the same toolchain and
paste the counts into the pull request.

def add_suite(cmake, name):
    lines = cmake.splitlines()
    if any(name in line for line in lines):
        return cmake
    block = declaration_line(name) + link_line(name) + registration_line(name)
    return cmake.rstrip() + "\\n" + block

The registration helper appends three lines to the build file when the suite name is not already there: one
that declares the executable and its source, one that links it against the kernel library and the GPU
runtime, and one that registers the test with the runner. The helper is idempotent, because a suite added
twice is a build failure with a message that points at the build file and not at the author.

The CI job runs the KV parity selection with output on failure and fails on any suite that does not report
a check count. A suite that prints nothing and exits 0 is the failure mode to design for: it is
indistinguishable from a suite that passed, and it is exactly what a stubbed test body produces. The suite
must print the number of checks it ran, and that number must be greater than zero.
"""

DOCUMENT = """Archive index 4417. The archive holds routine correspondence from the regional office, catalogued by
year and by storage vault, transcribed by hand and verified twice before it is filed. The following pages
summarise what the cataloguers found in the accessions of 1974 through 1979, and what has been done with
them since.

The accession of 1974 arrived as forty-one boxes with no index. The correspondence is largely between the
regional office and the district surveyors, and it concerns the standard of the boundary markers rather
than the boundary itself. Reading it in order, the same three questions return every few weeks: whether a
marker set in stone counts as a marker set in iron, whether a surveyor may act for both parties, and what
a district office is to do when the notes of a surveyor contradict the plan he filed. The answers are
consistent, and they are boring: the stone counts as the iron when the depth is recorded, a surveyor may
not act for both parties, and the filed plan governs.

The accession of 1975 is the one the cataloguers value most. It contains the minute books of the district
office, twelve volumes, which record not only the decisions but the cost of reaching them. A hearing in
that decade averaged nine hours of clerk time and two hours of magistrate time. The minute books let that
be measured per case, and the measurement is what made the reform of 1979 possible: the reform reduced the
average to six hours of clerk time, and it did so by standardising the forms rather than by changing the
law. The law was, on the evidence of these volumes, not the bottleneck.

The accession of 1976 is agricultural: drainage, commons, and the inclosure awards for eleven parishes. The
awards are drafted in a hand that is difficult to read and easy to misdate, because the clerk of that
period wrote the year as a two-digit number and dropped the apostrophe. Cataloguers should date an award by
the witness list, not by the numeral. In four cases the numeral and the witnesses disagree, and in all four
the witnesses are the later date.
"""

DOCUMENT = DOCUMENT + """
The accession of 1977 is the smallest and the most awkward. It is the correspondence of a single clerk,
kept at home, and it is personal. It contains no official material. It was donated by a grandson with a
letter saying the family had kept it in a cupboard under the stairs since 1961. The condition is poor: the
lower inches of most pages are water-damaged, and the paper is brittle where it was folded. The
conservators recommend interleaving with tissue and a cold room, and they recommend against flattening the
folds, because the folds are the evidence of how the clerk carried the letters.

The accession of 1978 is the engineering drawings for the river works, one hundred and nine sheets, rolled.
The rolls are numbered, and the numbers are complete, which is unusual. The drawings are on linen and
several are traced copies of drawings held elsewhere, which matters for dating: a traced copy is dated by
the tracing, not by the original, and the tracing office did not always copy the date.

The accession of 1979 is the personnel records of the office, and it is closed for seventy-five years
under the rules in force at the time of transfer. The cataloguers recorded the closure in the index and in
the box, and the box is sealed. Nothing in this summary describes what is inside it.

Three points recur across the whole set and are worth stating once. First, the index is a finding aid and
not a history: it records what is in a box, and it is silent about what was thrown away. Second, the
transcriptions are verified twice, and the second verification catches about one character in nine
hundred, most of them in dates. Third, the accessions are not comparable in size, and a rate computed
across them, damage per box or cost per case, is a rate over a mixture, not a property of any one of them.
"""

CHAT = """user: I keep hearing that a KV cache is just a memory of the past tokens. Is that the whole story?

assistant: It is the right first approximation and it hides the interesting part. The cache stores, for
every past position, the key and value vectors of the attention layers, so a new token can attend to the
past without recomputing it. What it hides is that the cache is also the thing that gets quantized,
evicted and paged out, and every one of those operations changes what the model can see. A model with a
2-bit cache and a model with an fp16 cache are the same weights reading two different memories.

user: So quantizing the cache changes the answers?

assistant: Sometimes measurably, sometimes not at all, and the honest answer is that you have to measure
it. The way to measure it is teacher-forced: feed the same text to both configurations, take the
distribution at every position, and compare. Top-1 agreement tells you how often the argmax is the same.
Mean KL tells you how far the whole distribution moved. Perplexity tells you what the model thinks of the
text it was given. Nats per token tells you the average cost of the missing bits, with a standard error so
you know when a difference is a difference.

user: And what does that show for 4-bit?

assistant: On the measurements recorded in this repository, int8 is indistinguishable from fp16, within
about two standard errors everywhere. Four-bit costs measurable precision, and it costs more as the
context grows: perplexity up about eight percent at a thousand tokens and about twelve percent at eight
thousand on a document, roughly neutral on code and on chat. Retrieval holds: needles at 1K, 32K, 128K
and 262K come back. So the format is usable, and it is a user option rather than a default.
"""

CHAT = CHAT + """

user: What about keeping the recent tokens in full precision?

assistant: That is the sliding window with attention sinks. The last hundred and twenty-eight cells and
the first eight are kept in fp16, and the bulk stays quantized. It removes about eight percent of the key
error in two of the three position buckets and none in the bulk. Measured end to end at four thousand
through a hundred and thirty-one thousand tokens, on probes that need both the bulk and the window, it
changed nothing: not quality, not speed. Its purpose is to make two and three bit caches usable, not to
make four bit better.

user: Why is the bulk the part that matters for a long document?

assistant: Because the questions of a long document point into the bulk. Put six facts in the body of a
document and six in the last hundred tokens before the question, and the first six are decided by the
precision of the bulk. The window is exact by construction, so it cannot be the reason a fact is lost. The
bulk is where the error is, and the loss comes from where the error is.

user: If I change only one thing, what should I change?

assistant: Measure before you choose. Run the same text through fp16 and through the format you are
considering, and read the nats per token together with its standard error. If the interval covers zero,
the format is free and you should take the memory. If it does not, you have a number for what you are
paying, and you can decide with that number in hand instead of a rumour about bit depth.
"""

TEXTS = [("code agent", CODE_AGENT), ("document", DOCUMENT), ("chat", CHAT)]


# ------------------------------------------------------------------ the tokenizer
def load_tokenizer(cfg: dict) -> ST.Tokenizer:
    """The pack's exported tokenizer (the engine has none): vocab.json, merges.txt, token_type.json."""
    t = Path(cfg["tokenizer"])
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types)


# ------------------------------------------------------------------ logits files
def read_header(path: Path) -> tuple[int, int]:
    """int32 n_vocab, int32 n_rows, then n_rows rows of n_vocab float32. A header that does not describe the
    file is refused here rather than turned into a metric that reads as authoritative."""
    with open(path, "rb") as f:
        hdr = f.read(8)
    if len(hdr) < 8:
        raise ValueError("%s: no header, the file is %d bytes" % (str(path), len(hdr)))
    vals = [int(v) for v in np.frombuffer(hdr, dtype="<i4")]
    n_vocab, n_rows = vals[0], vals[1]
    size = path.stat().st_size
    want = 8 + n_vocab * 4 * n_rows
    if size != want:
        raise ValueError("%s: the header says %d x %d = %d bytes, the file is %d"
                         % (str(path), n_vocab, n_rows, want, size))
    return n_vocab, n_rows


def block(f, n_vocab: int, r0: int, nr: int) -> np.ndarray:
    """One block of rows. Read in blocks: 1024 rows x 248,320 vocab is 1.0 GB, and two arms at once is
    two of those. A block of 16 rows is 15 MB, which is what the KL term needs live."""
    f.seek(8 + r0 * n_vocab * 4)
    buf = f.read(nr * n_vocab * 4)
    if len(buf) != nr * n_vocab * 4:
        raise ValueError("short read at row %d" % r0)
    return np.frombuffer(buf, dtype="<f4").reshape(nr, n_vocab)


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    z = np.exp(x - m)
    return x - m - np.log(z.sum(axis=1, keepdims=True))


def compare(ref_path: Path, arm_path: Path, tokens: list[int]) -> dict:
    """The per-arm numbers against the reference, on the same tokens.

    Row k predicts token k+1. The last row predicts the first GENERATED token, which is not in the prompt,
    so the target-dependent metrics (perplexity, nats/token) stop at row n_rows-2; top-1 agreement and the
    KL need no target and run over every row. n_rows and n_scored are returned so the table says which is
    which.
    """
    n_vocab, n_rows = read_header(ref_path)
    a_vocab, a_rows = read_header(arm_path)
    if (n_vocab, n_rows) != (a_vocab, a_rows):
        raise ValueError("%s: %d x %d does not match the reference %d x %d"
                         % (str(arm_path), a_vocab, a_rows, n_vocab, n_rows))
    scored = min(n_rows, len(tokens) - 1)
    top1_same, top1_n = 0, 0
    kl_sum, kl_n = 0.0, 0
    d_nats, n_nats = 0.0, 0
    nats_i: list[float] = []
    d_i: list[float] = []
    nonfinite = 0
    B = 16
    with open(ref_path, "rb") as fr, open(arm_path, "rb") as fa:
        for r0 in range(0, n_rows, B):
            nr = min(B, n_rows - r0)
            xr = block(fr, n_vocab, r0, nr).astype(np.float64)
            xa = block(fa, n_vocab, r0, nr).astype(np.float64)
            nonfinite += int(np.count_nonzero(~np.isfinite(xr)))
            nonfinite += int(np.count_nonzero(~np.isfinite(xa)))
            lr, la = log_softmax(xr), log_softmax(xa)
            pr = np.exp(lr)
            top1_same += int(np.count_nonzero(np.argmax(xr, axis=1) == np.argmax(xa, axis=1)))
            top1_n += nr
            kl_sum += float(np.sum(np.sum(pr * (lr - la), axis=1)))
            kl_n += nr
            take = min(nr, max(0, scored - r0))
            if take > 0:
                rows = np.arange(r0, r0 + take)
                idx = np.arange(take)
                tgt = np.array([tokens[int(r) + 1] for r in rows], dtype=np.int64)
                nats_i += [-float(v) for v in la[idx, tgt]]
                d_vals = [float(-a + b) for a, b in zip(la[idx, tgt], lr[idx, tgt])]
                d_i += d_vals
                d_nats += float(sum(d_vals))
                n_nats += take
    if nonfinite:
        raise ValueError("%s and %s: %d non-finite logits" % (str(arm_path), str(ref_path), nonfinite))
    mean_d = d_nats / n_nats if n_nats else float("nan")
    se = float("nan")
    if n_nats > 1:
        se = float(np.asarray(d_i, dtype=np.float64).std(ddof=1)) / math.sqrt(n_nats)
    return dict(
        n_rows=n_rows,
        n_scored=n_nats,
        top1_same_pct=100.0 * top1_same / top1_n if top1_n else float("nan"),
        mean_kl=kl_sum / kl_n if kl_n else float("nan"),
        perplexity=math.exp(-d_nats / n_nats) if n_nats else float("nan"),
        nats_per_token=float(np.mean(nats_i)) if nats_i else float("nan"),
        nats_delta_vs_fp16=mean_d,
        nats_delta_se=se,
        indistinguishable=bool(n_nats > 1 and abs(mean_d) < 2.0 * se),
    )


# ------------------------------------------------------------------ the engine
def engine_args(cfg: dict, kv: str) -> list[str]:
    """The serve configuration's args, with only --kv replaced. --no-capture is NOT set: a native pack
    captures no graphs, and the prompt path is the one that runs."""
    out, i, src = [], 0, list(cfg["args"])
    while i < len(src):
        if src[i] == "--kv":
            out += ["--kv", kv]
            i += 2
            continue
        out.append(src[i])
        i += 1
    return out


# The CUDA 13.0 runtime the CUDA 13.0 build needs. CUDA 13.2 miscompiles IQ3_XXS on this box (see
# bench/results/2026-10-11-kv-window/README.md), and the serve configuration's lib_dirs point at 13.2
# because the installed release is built against it. A precision table run on the wrong runtime is a
# measurement of a different engine, so the pair is chosen from the exe path and can be overridden.
# KV_CUDA130 names the root of a CUDA 13.0 toolkit (e.g. pip's cuda-toolkit components, .../nvidia/cu13).
_CU130 = os.environ.get("KV_CUDA130")
CUDA130_LIBS = [os.path.join(_CU130, "bin"), os.path.join(_CU130, "bin", "x64")] if _CU130 else []


def lib_dirs(cfg: dict, exe: str) -> list[str]:
    override = os.environ.get("KV_LIB_DIRS")
    if override:
        return [p for p in override.split(os.pathsep) if p]
    if "build-cu130" in str(exe).replace("\\", "/") and CUDA130_LIBS:
        return CUDA130_LIBS
    return list(cfg.get("lib_dirs") or [])


def run_engine(cfg: dict, exe: str, arm: dict, tokens: list[int], out: Path, log: Path,
               max_new: int = 1, mode: str = "prompt") -> int:
    tok_file = out.with_suffix(".tokens.txt")
    tok_file.write_text(",".join(str(t) for t in tokens), encoding="utf-8")
    # mode="prompt": teacher-forced rows, one per prompt position (the prefill path).
    # mode="decode": one row per GENERATED token, from the speculative acceptance loop - the decode path,
    # which is where the FP16 window overlay and the quantized bulk actually act.
    dump_flag = "--dump-prompt-logits" if mode == "prompt" else "--dump-logits"
    cmd = [exe] + engine_args(cfg, arm["kv"]) + [
        "--tokens-file", str(tok_file),
        "--max-new", str(max_new),
        "--greedy",
        dump_flag, str(out),
    ]
    env = dict(os.environ)
    env.update(arm["env"])
    libs = lib_dirs(cfg, exe)
    if libs:
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(libs + [env.get("PATH", "")])
        else:
            env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    t0 = time.time()
    with open(log, "wb") as lf:
        p = subprocess.run(cmd, cwd=cfg.get("cwd"), stdout=lf, stderr=subprocess.STDOUT, env=env)
    print("    %s: exit=%d in %.1fs" % (out.name, p.returncode, time.time() - t0), flush=True)
    if p.returncode != 0:
        print("      last log lines:", flush=True)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
            print("      " + line, flush=True)
    return p.returncode


def first_generated(log: Path) -> int | None:
    """The run's own `output :` line. With --max-new 1 this is the first generated token, which is what
    proves the measurement flag did not alter the run."""
    m = re.search(r"^output\s*:\s*(.*)$", log.read_text(encoding="utf-8", errors="replace"), re.M)
    if not m:
        return None
    ids = m.group(1).split()
    return int(ids[0]) if ids else None


def generated_tokens(log: Path) -> list[int]:
    """All of the run's `output :` line: the tokens the engine emitted, in order."""
    m = re.search(r"^output\s*:\s*(.*)$", log.read_text(encoding="utf-8", errors="replace"), re.M)
    if not m:
        return []
    return [int(t) for t in m.group(1).split() if t.lstrip("-").isdigit()]


def compare_decode(ref_path: Path, ref_tokens: list[int], arm_path: Path, arm_tokens: list[int]) -> dict:
    """Decode-path comparison. Row i of a --dump-logits file is the distribution that produced generated
    token i, so the two files are compared step by step - the path the FP16 window overlay and the
    quantized bulk actually act on (prefill writes the overlay; decode attention reads it).

    Greedy arms diverge: once an arm emits a different token, its later rows are conditioned on a
    different prefix, and comparing them compares different contexts. So the KL runs over the AGREEMENT
    PREFIX only, and the prefix length is itself the headline number - how far an arm generates
    identically to the reference before it departs.
    """
    nv_r, nr_r = read_header(ref_path)
    nv_a, nr_a = read_header(arm_path)
    n = min(len(ref_tokens), len(arm_tokens), nr_r, nr_a)
    agree = 0
    while agree < n and ref_tokens[agree] == arm_tokens[agree]:
        agree += 1
    kl: list[float] = []
    kl0 = None
    with open(ref_path, "rb") as fr, open(arm_path, "rb") as fa:
        for r0 in range(0, max(agree, 1), 16):
            nr = min(16, max(agree, 1) - r0)
            xr = block(fr, nv_r, r0, nr).astype(np.float64)
            xa = block(fa, nv_a, r0, nr).astype(np.float64)
            lr, la = log_softmax(xr), log_softmax(xa)
            step = np.sum(np.exp(lr) * (lr - la), axis=1).tolist()
            kl.extend(step[:nr] if r0 < agree else [])
            if r0 == 0:
                kl0 = step[0]
    return dict(agree=agree, requested=n, generated=len(arm_tokens), rows_ref=nr_r, rows_arm=nr_a,
                mean_kl=float(np.mean(kl)) if kl else 0.0,
                max_kl=float(np.max(kl)) if kl else 0.0,
                kl_first_step=kl0,
                diverged_at=agree if agree < n else -1,
                ref_token=ref_tokens[agree] if agree < len(ref_tokens) else None,
                arm_token=arm_tokens[agree] if agree < len(arm_tokens) else None)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--arms", default=",".join(d["name"] for d in ARMS))
    ap.add_argument("--texts", default="all", help="all, or a substring of a text name")
    ap.add_argument("--tmp", default=str(ROOT / "tmp" / "kv-precision"))
    ap.add_argument("--json", default=str(ROOT / "tmp" / "kv-precision" / "kv_precision_compare.json"))
    ap.add_argument("--keep-logits", action="store_true", help="leave the per-arm logits files in place")
    ap.add_argument("--max-tokens", type=int, default=1024, help="truncate each text to this many tokens")
    ap.add_argument("--write-tokens", action="store_true",
                    help="tokenize, write tmp/<text>.tokens.txt for each text, and exit without running the "
                         "engine (for hand-run engine checks: the 64-row check, the two-path argmax check, "
                         "the first-token check)")
    ap.add_argument("--token-limit", type=int, default=0,
                    help="with --write-tokens, truncate to this many tokens instead of --max-tokens")
    ap.add_argument("--decode", type=int, default=0,
                    help="decode-path mode: greedy-generate this many tokens per arm with --dump-logits and "
                         "compare the per-step distributions over the agreement prefix. This is the path the "
                         "FP16 window overlay acts on (prefill writes it, decode attention reads it), so it "
                         "is the only logit-level measurement a window change can move. Skips the "
                         "teacher-forced pass.")
    ap.add_argument("--repeat", type=int, default=3,
                    help="with --decode: runs per arm. Greedy decode is not reproducible run to run on this "
                         "engine, so decode-path numbers are means over repeats with their spread.")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    exe = os.environ.get("KV_ENGINE") or cfg["exe"]
    print("engine: %s" % exe, flush=True)
    print("runtime lib dirs: %s" % ", ".join(lib_dirs(cfg, exe)), flush=True)
    if not Path(exe).exists():
        print("DEFECT: the engine path does not exist: %s" % exe)
        return 2

    arm_of = dict((d["name"], d) for d in ARMS)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for n in arms:
        if n not in arm_of:
            print("unknown arm %s; known: %s" % (n, ", ".join(arm_of)))
            return 2
    if REFERENCE not in arms:
        arms.insert(0, REFERENCE)

    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(cfg)
    prompts: dict[str, list[int]] = dict()
    for name, body in TEXTS:
        if a.texts != "all" and a.texts.lower() not in name.lower():
            continue
        limit = a.token_limit if a.token_limit > 0 else a.max_tokens
        ids = tok.encode(body, parse_special=True)[: limit]
        prompts[name] = ids
        print("text %s: %d tokens" % (name, len(ids)), flush=True)
    if a.write_tokens:
        # The token files the hand-run engine checks need. The engine has no tokenizer, so these are the
        # input; the harness's own runs write the same files.
        for name, ids in prompts.items():
            tag = re.sub("[^a-z0-9]+", "-", name.lower())
            p = tmp / (tag + ".tokens.txt")
            p.write_text(",".join(str(t) for t in ids), encoding="utf-8")
            print("wrote %s (%d ids)" % (p, len(ids)))
        return 0
    if not prompts:
        print("no texts selected")
        return 2

    if a.decode > 0:
        # Greedy decode on this engine is NOT reproducible run to run: the same command gave q4_0 6/64
        # identical tokens in one run and 1/64 in the next (QSA top-k tie-breaks and the atomic partial-
        # group KV writes are order-sensitive). So each arm is run --repeat times and the numbers are
        # means with their spread; a single run per arm is a sample, not a measurement.
        reps = max(1, a.repeat)
        dec: dict[str, dict[str, list]] = dict((name, dict((an, []) for an in arms)) for name in prompts)
        for name, ids in prompts.items():
            tag = re.sub("[^a-z0-9]+", "-", name.lower())
            for rep in range(reps):
                dfiles: dict[str, Path] = dict()
                dtoks: dict[str, list[int]] = dict()
                for an in arms:
                    out = tmp / ("%s-%s.r%d.decode.bin" % (tag, an, rep))
                    log = tmp / ("%s-%s.r%d.decode.log" % (tag, an, rep))
                    if run_engine(cfg, exe, arm_of[an], ids, out, log, max_new=a.decode, mode="decode") != 0:
                        return 1
                    dfiles[an] = out
                    dtoks[an] = generated_tokens(log)
                for an in arms:
                    dec[name][an].append(
                        compare_decode(dfiles[REFERENCE], dtoks[REFERENCE], dfiles[an], dtoks[an]))
                if not a.keep_logits:
                    for p in dfiles.values():
                        p.unlink(missing_ok=True)
        print()
        print("Decode-path precision from --dump-logits: row i is the distribution that produced generated")
        print("token %d, %d runs per arm (greedy decode is not reproducible run to run on this engine, so" % (a.decode, reps))
        print("these are means with min-max spread). KL runs over the agreement prefix - the steps where")
        print("both arms emitted the same token - and KL @ 0 is always comparable (same context, the last")
        print("prompt position), which is the number to read when arms diverge immediately.")
        print("This is the path the FP16 window overlay acts on: prefill writes it, decode attention reads it.")
        print()
        print("| %-14s | %-12s | %15s | %15s | %15s | %15s |"
              % ("Text", "KV", "Identical prefix", "Mean KL", "Max KL", "KL @ 0"))
        print("|" + "---|" * 5)
        for name in prompts:
            for an in arms:
                rs = dec[name][an]
                def stat(key):
                    vals = [r[key] for r in rs if r[key] is not None]
                    return (sum(vals) / len(vals), min(vals), max(vals)) if vals else (0.0, 0.0, 0.0)
                pa, pmin, pmax = stat("agree")
                ma, mmin, mmax = stat("mean_kl")
                xa, xmin, xmax = stat("max_kl")
                k0, k0min, k0max = stat("kl_first_step")
                print("| %-14s | %-12s | %7.1f %5.0f-%-4.0f | %7.4f %4.3f-%-4.3f | %6.4f %4.3f-%-4.3f | %7.4f %4.3f-%-4.3f |"
                      % (name if an == arms[0] else "", an, pa, pmin, pmax, ma, mmin, mmax,
                         xa, xmin, xmax, k0, k0min, k0max))
        payload = dict(engine=exe, config=str(a.config), reference=REFERENCE, mode="decode",
                       max_new=a.decode, repeats=reps,
                       arms=[dict(name=an, kv=arm_of[an]["kv"], env=arm_of[an]["env"]) for an in arms],
                       texts=dict((n, dict(n_tokens=len(ids))) for n, ids in prompts.items()),
                       metrics=dict((an, dict((n, dec[n][an]) for n in prompts)) for an in arms))
        Path(a.json).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print()
        print("wrote %s" % a.json)
        return 0

    results: dict[str, dict[str, dict]] = dict((an, dict()) for an in arms)
    firsts: dict[str, dict[str, int]] = dict()
    for name, ids in prompts.items():
        tag = re.sub("[^a-z0-9]+", "-", name.lower())
        files: dict[str, Path] = dict()
        firsts[name] = dict()
        for an in arms:
            out = tmp / (tag + "-" + an + ".logits.bin")
            log = tmp / (tag + "-" + an + ".log")
            if run_engine(cfg, exe, arm_of[an], ids, out, log) != 0:
                return 1
            files[an] = out
            firsts[name][an] = first_generated(log)
        ref = files[REFERENCE]
        for an in arms:
            results[an][name] = compare(ref, files[an], ids)
        if not a.keep_logits:
            for p in files.values():
                p.unlink(missing_ok=True)

    print()
    print("Teacher-forced precision from --dump-prompt-logits, reference --kv %s. One engine binary, one"
          % REFERENCE)
    print("token list, one expert profile per run; only --kv and the window environment differ.")
    print("top-1 and KL run over every prompt row; perplexity and nats/token over the rows whose target")
    print("token is in the prompt (row k predicts token k+1, so the last row is excluded).")
    print()
    print("| %-14s | %-12s | %18s | %9s | %10s | %34s |"
          % ("Text", "KV", "Same top-1 as fp16", "Mean KL", "Perplexity", "vs fp16 (nats/token)"))
    print("|" + "---|" * 6)
    for name in prompts:
        for an in arms:
            r = results[an][name]
            delta = "%+.3f +- %.3f" % (r["nats_delta_vs_fp16"], r["nats_delta_se"])
            if an != REFERENCE and r["indistinguishable"]:
                delta += " (indistinguishable)"
            print("| %-14s | %-12s | %17.1f%% | %9.4f | %10.3f | %34s |"
                  % (name if an == arms[0] else "", an, r["top1_same_pct"], r["mean_kl"],
                     r["perplexity"], delta))
    print()
    print("rows per arm (n_rows; n_scored = rows with a target token in the prompt):")
    for name in prompts:
        for an in arms:
            r = results[an][name]
            print("  %-14s %-12s n_rows=%d n_scored=%d" % (name, an, r["n_rows"], r["n_scored"]))
    print()
    print("first generated token per arm (identical across arms is the flag-did-not-alter-the-run check):")
    for name in prompts:
        print("  " + name + ": " + ", ".join("%s=%s" % (an, firsts[name][an]) for an in arms))

    payload = dict(
        engine=exe,
        config=str(a.config),
        reference=REFERENCE,
        arms=[dict(name=an, kv=arm_of[an]["kv"], env=arm_of[an]["env"]) for an in arms],
        texts=dict((n, dict(n_tokens=len(ids), first_tokens=firsts[n])) for n, ids in prompts.items()),
        metrics=results,
    )
    Path(a.json).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print()
    print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
