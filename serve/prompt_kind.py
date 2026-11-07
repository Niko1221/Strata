"""Label a prompt by the kind of answer it is asking for: the key a per-request retune uses.

The same engine does not have to be tuned the same way for every request.  On this box the code lane decodes
about 114 tok/s and the prose lane 65-69 tok/s on the same config, because MTP draft acceptance is much higher
on code; the settings that gate that (``--spec-min-p``) are per-request keys, so the floor can follow the
prompt instead of being one value for the whole session.

This is a heuristic on purpose: it reads the prompt's own text, costs microseconds, has no model, and is
deterministic - so a measurement can be repeated and a wrong label can be explained from the reasons it prints.
It is deliberately BIASED TO PROSE: a label only changes a setting when the code evidence is clear, because an
unnecessary change costs more than a missed one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CODE = "code"
PROSE = "prose"

# a fenced block is the strongest single signal (```lang ... ```)
_FENCE = re.compile(r"```")
# a line that IS code rather than prose about code
_CODE_LINE = re.compile(
    r"^\s{3,}\S"                                             # any indented line (code, not a wrapped paragraph)
    r"|^\s*(?:def|class|import|from|return|yield|async|await|lambda|pass|raise|try|except|finally|with)\b"
    r"|^\s*(?:if|elif|else|for|while|switch|case|do|break|continue)\b[^\n]*[:(]"
    r"|^\s*(?:function|const|let|var|export|module\.|public|private|protected|static|final|void|int|float|"
    r"double|char|bool|struct|enum|typedef|template|namespace|using|package|func|impl|fn|use|mod)\b"
    r"|^\s*(?:#include|#define|#!|//|/\*|\*|-->|<!--)\s"
    r"|^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\b"
    r"|\}\s*$|\{\s*$|\)\s*;?\s*$|;\s*$|=>|\|>|::|->"
)
# instruction verbs, paired with the noun they must appear near to count
_CODE_VERB = re.compile(
    r"\b(?:write|implement|refactor|debug|fix|add|generate|create|port|optimi[sz]e|compile|build|patch|"
    r"convert|migrate|profile|benchmark|test)\b", re.I)
_CODE_NOUN = re.compile(
    r"\b(?:function|class|method|module|script|query|endpoint|regex|regexp|test|tests|patch|diff|api|sql|"
    r"hook|handler|parser|crate|package|library|service|struct|macro|kernel|unit test|typescript|javascript|"
    r"python|rust|golang|c\+\+|bash|yaml|json|dockerfile|makefile|cmakelists|middleware|cli|flag|command|"
    r"tool|worker|daemon|pipeline|migration|schema|suite|app|server|client|plugin|bot)\b", re.I)
_PROSE_VERB = re.compile(
    r"\b(?:explain|describe|summari[sz]e|compare|contrast|outline|brainstorm|translate|rephrase|critique|"
    r"review|plan|draft (?:an? )?(?:email|letter|story|poem|post|blog|message)|tell me|walk me through)\b", re.I)
_ERROR_MARK = re.compile(
    r"(?:Traceback \(most recent call last\)|^\w*Error:|^\w*Exception|panicked at|segmentation fault|"
    r"core dumped|stderr|exit code|at line \d+|File \"[^\"]+\", line \d+)", re.M)
_DIFF_MARK = re.compile(r"^(?:diff --git |--- a/|\+\+\+ b/|@@ )", re.M)
_PATH = re.compile(r"\b[\w./-]+\.(?:py|c|cc|cpp|h|hpp|js|jsx|ts|tsx|go|rs|java|kt|rb|sh|sql|json|ya?ml|toml|"
                   r"ini|cfg|md|html|css)\b", re.I)
_CODE_PUNCT = re.compile(r"[(){}\[\];=<>]")
# `code` in backticks: the error text, a flag or a symbol a person pasted - code evidence without a block
_INLINE_CODE = re.compile(r"`[^`\n]{2,80}`")

THRESHOLD = 0.35          # code_score - prose_score at which a prompt is called code


@dataclass
class Label:
    kind: str
    score: float
    reasons: dict = field(default_factory=dict)

    def __str__(self) -> str:
        top = ", ".join(f"{k}={v:+.2f}" for k, v in sorted(self.reasons.items(), key=lambda kv: -abs(kv[1]))[:4])
        return f"{self.kind} ({self.score:+.2f}: {top})"


def _lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def classify(text: str) -> Label:
    """Label one turn's text.  Never raises; an empty or unreadable prompt is prose (no change)."""
    text = text if isinstance(text, str) else ""
    if not text.strip():
        return Label(PROSE, 0.0, {"empty": 0.0})
    reasons: dict = {}
    code = 0.0
    prose = 0.0

    fences = min(len(_FENCE.findall(text)) // 2, 2)
    if fences:
        code += 0.45 * fences
        reasons["fenced blocks"] = 0.45 * fences

    lines = _lines(text)
    is_code_line = [_CODE_LINE.search(ln) is not None for ln in lines]
    if lines:
        hit = sum(is_code_line)
        ratio = hit / len(lines)
        if ratio:
            c = min(ratio * 1.2, 0.6)
            code += c
            reasons["code lines"] = c

    # the instruction part: verbs only count near a code noun, so "fix the wording" stays prose
    verb = _CODE_VERB.search(text)
    instruction_code = bool(verb and _CODE_NOUN.search(text))
    if instruction_code:
        code += 0.55
        reasons["code instruction"] = 0.55
    if _ERROR_MARK.search(text):
        code += 0.30
        reasons["error/stack"] = 0.30
    if _DIFF_MARK.search(text):
        code += 0.30
        reasons["diff"] = 0.30
    paths = len(_PATH.findall(text))
    if paths:
        c = min(0.35 * paths, 0.50)
        code += c
        reasons["file names"] = c
    spans = len(_INLINE_CODE.findall(text))
    if spans:
        c = min(0.20 * spans, 0.30)
        code += c
        reasons["inline code"] = c

    letters = sum(len(ln) for ln in lines) or 1
    density = len(_CODE_PUNCT.findall(text)) / letters
    if density > 0.01:
        c = min(density * 8.0, 0.40)
        code += c
        reasons["code punctuation"] = c

    if _PROSE_VERB.search(text):
        prose += 0.30
        reasons["prose instruction"] = -0.30
    if not fences and not instruction_code and not paths and not any(is_code_line):
        prose += 0.20
        reasons["no code shape"] = -0.20
    words = max(len(text.split()), 1)
    if len(lines) <= 4 and words > 40:                     # one long paragraph: an explanation request
        prose += 0.15
        reasons["one long paragraph"] = -0.15

    score = round(code - prose, 3)
    return Label(CODE if score >= THRESHOLD else PROSE, score, reasons)


def _last_user_text(messages, context_chars: int = 1200) -> str | None:
    """The LAST user turn, with the tail of the turn before it as context; None when there is no user turn.

    "Explain what this does" is code when the thing before it was code, which a single-turn reading misses.
    """
    turns = [m for m in (messages or []) if isinstance(m, dict)]
    users = [m for m in turns if (m.get("role") or "").lower() == "user"]
    if not users:
        return None
    text = _content(users[-1])
    if len(users) > 1:
        text = _content(users[-2])[-context_chars:] + "\n" + text
    return text


def messages_text(messages, context_chars: int = 1200) -> str:
    """The text a chat request is judged on (see _last_user_text), or "" when there is no user turn."""
    return _last_user_text(messages, context_chars) or ""


def classify_messages(messages, context_chars: int = 1200) -> Label:
    """Label a chat request by the text messages_text() returns."""
    text = _last_user_text(messages, context_chars)
    if text is None:
        return Label(PROSE, 0.0, {"no user turn": 0.0})
    return classify(text)


def _content(message: dict) -> str:
    """A message's text.  OpenAI content is a string or a list of parts; images and tool blocks carry no text."""
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return ""


# -----------------------------------------------------------------------------------------------------------
# A second, narrower label: does this prompt want a SHORT DIRECT answer (so thinking is wasted)?
#
# Measured reason to have it: on this box a needle question spent 313-405 reasoning characters before answering
# a single line, i.e. 25-50 % of the generated tokens on that request are thinking nobody reads.  The rule is
# deliberately narrow and biased to "let it think": a request that asks for explanation, design or code keeps
# its thinking, and anything ambiguous does too.
# -----------------------------------------------------------------------------------------------------------

_DIRECT_STRONG = re.compile(
    r"(?:^|[^a-z])(?:reply|answer|respond).{0,40}?(?:only|just|one word|single word|word|number|numbers|value|name|"
    r"id|json|yaml|csv|xml|yes or no|true or false)(?:[^a-z]|$)"
    r"|(?:^|[^a-z])(?:extract|translate|convert|reformat|format|parse|transform).{0,60}?(?:json|yaml|csv|xml|"
    r"markdown|number|value|name|id|email|date|address|french|german|spanish)(?:[^a-z]|$)"
    r"|(?:^|[^a-z])(?:output|print|return|give me|give).{0,20}?(?:only|just)(?:[^a-z]|$)"
    r"|(?:one[- ]word|single word|just the (?:value|name|number|answer|word|id)|only the (?:value|name|number|"
    r"answer|word|id))"
    r"|(?:^|[^a-z])what is the (?:value|name|number|id|price|date|version) of",
    re.I)
# a transformation of the text that was given (the payload may be code; the task is still mechanical)
_TRANSFORM = re.compile(r"(?:^|[^a-z])(?:extract|translate|convert|reformat|format|parse|transform)(?:[^a-z]|$)", re.I)
# MEASURED COST of getting this wrong (accuracy campaign 2026-11-04, 26 prompts x 3 rounds x 2 arms): with the
# scratchpad suppressed, "Is 91 divisible by 7? Reply yes or no." answered "No" (it is: 13 x 7), "Convert 2048 MiB
# to GiB" answered 1.9999999999999998, and a CSV row came out in the wrong field order. Those three prompts took
# the lever from 88.5 % to 76.9 % correct. So anything that has to COMPUTE or ORDER keeps its thinking, however
# directly its wording asks for the answer. Extraction still gets suppressed; arithmetic does not.
_COMPUTE = re.compile(
    r"(?:^|[^a-z])(?:divisible|modulo|remainder|percent(?:age)?|average|mean|sum|product|difference|largest|"
    r"smallest|larger|smaller|greater|less than|how many|how much|how long|how far|how often|count|counts|"
    r"per (?:second|minute|hour|day|token)|in order|ordered|fields in|columns in|sorted|convert)(?:[^a-z]|$)"
    r"|\b(?:mib|gib|kib|mb|gb|kb|bytes?|seconds?|minutes?|hours?|days?)\b"   # a UNIT means a conversion
    r"|\d\s*[+*/x\u00d7\u00f7]\s*\d",                     # 17 + 25, 45 x 12 - not '-' (dates look like that)
    re.I)
_REASONING = re.compile(
    r"(?:^|[^a-z])(?:explain|why|how does|how do|prove|derive|design|compare|contrast|analy[sz]e|critique|evaluate|"
    r"plan|step[- ]by[- ]step|walk me through|reason about|trade-?offs?|architecture|refactor|debug|implement|"
    r"write (?:a|an|the|me)? ?(?:function|class|script|test|program|module))(?:[^a-z]|$)",
    re.I)

def wants_direct_answer(text: str) -> Label:
    """A SHORT DIRECT answer request: extraction, formatting, translation, a one-word reply.

    Returns a Label whose `kind` is "direct" or "reason" - the two names are what a caller logs, and the score
    is positive only when the direct-reading evidence clearly outweighs the reasoning-reading evidence.  A code
    prompt is never direct: a bug or a design question needs its thinking.
    """
    text = text if isinstance(text, str) else ""
    if not text.strip():
        return Label(PROSE, 0.0, {"empty": 0.0})
    reasons: dict = {}
    strong = len(_DIRECT_STRONG.findall(text))
    reasoning = len(_REASONING.findall(text))
    compute = len(_COMPUTE.findall(text))
    score = min(0.9, 0.55 * strong) - min(0.9, 0.45 * reasoning) - min(0.9, 0.6 * compute)
    if strong:
        reasons["asks for a value/format"] = 0.55 * min(strong, 2)
    if reasoning:
        reasons["asks for reasoning"] = -0.45 * min(reasoning, 2)
    if compute:
        reasons["needs arithmetic/ordering"] = -0.6 * min(compute, 2)
    kind = classify(text)
    if kind.kind == CODE and score > 0 and not _TRANSFORM.search(text):
        score -= 0.5                      # code: keep the thinking even when the wording sounds direct
        reasons["code prompt (keeps thinking)"] = -0.5
    return Label("direct" if score >= 0.4 else "reason", round(score, 3), reasons)
