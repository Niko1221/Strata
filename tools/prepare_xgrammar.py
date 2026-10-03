"""Prepare the one pinned native grammar dependency at build time, never server startup.

No Git operations, Python bindings, TVM or CUDA extension.
Preserves upstream licenses. Applies the listed cooperative resource guards and
the dynamic-object schema correction; original/patched hashes are in PIN.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

XGRAMMAR = ("mlc-ai/xgrammar", "97787376faee5ed8466cfad57c99855e4ce2f6aa",
            "87bc16759aadc535e3585551cfc086d0db704a1c12d0a97a3261650a62731719")
DLPACK = ("dmlc/dlpack", "bbd2f4d32427e548797929af08cfe2a9cbb3cf12",
          "f5dcb30f8a3d1a41d48b5d5b3fe1631dfce11de76f9e7e5bd22c6dc75a758885")
BACKEND = "xgrammar-0.2.8-strata-budget1-json1"


def extract(pin, dest):
    repo, commit, digest = pin
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"dependency archive size/hash mismatch: {repo}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        total = 0
        for item in archive:
            parts = PurePosixPath(item.name).parts[1:]
            if not parts:
                continue
            if any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
                raise RuntimeError("unsafe dependency archive path")
            target = dest.joinpath(*parts)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.isfile():
                total += item.size
                if total > 128 * 1024 * 1024:
                    raise RuntimeError("expanded dependency exceeds 128 MiB")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as source, target.open("xb") as output:
                    output.write(source.read())
            else:
                raise RuntimeError("dependency archive contains a link or special file")
    return {"repository": repo, "commit": commit, "url": url, "sha256": digest}


def patch(dest):
    # Each exact anchor is checked against the pinned source. These hooks bound
    # grammar expansion, FSM growth and ambiguous Earley state work. They do not
    # alter a transition, token mask, acceptance rule or sampling distribution.
    edits = {
        "cpp/grammar_builder.cc": [
            ("int32_t GrammarBuilder::AddGrammarExpr(const GrammarExpr& grammar_expr) {", 1,
             "\n  strata::grammar::detail::work(2 + grammar_expr.data_len);"
             "\n  strata::grammar::detail::bound(grammar_->grammar_expr_data_.size() + 2 + grammar_expr.data_len, 65536);"
             "\n  strata::grammar::detail::bound(grammar_->grammar_expr_indptr_.size() + 1, 8192);")],
        "cpp/fsm.cc": [
            ("  int AddState() {", 1,
             "\n    strata::grammar::detail::work();"
             "\n    strata::grammar::detail::bound(edges_.size() + 1, 8192);"),
            ("  void AddEdge(int from, int to, int32_t min, int32_t max) {", 1,
             "\n    strata::grammar::detail::work();"
             "\n    strata::grammar::detail::bound(edges_[from].size() + 1, 8192);")],
        "cpp/earley_parser.h": [
            ("  void Enqueue(const ParserState& state) {", 1,
             "\n    strata::grammar::detail::work();"
             "\n    strata::grammar::detail::bound(tmp_process_state_queue_.size() + tmp_states_to_be_added_.size() + 1, 8192);"),
            ("  void EnqueueWithoutProcessing(const ParserState& state) {", 1,
             "\n    strata::grammar::detail::work();"
             "\n    strata::grammar::detail::bound(tmp_process_state_queue_.size() + tmp_states_to_be_added_.size() + 1, 8192);")],
        "cpp/earley_parser.cc": [
            ("  for (const auto& state : latest_states) {", 2, "\n    strata::grammar::detail::work();"),
            ("  while (!tmp_process_state_queue_.empty()) {", 3, "\n    strata::grammar::detail::work();")],
        "cpp/grammar_compiler.cc": [
            ("    for (int i = interval.first; i < interval.second; ++i) {", 1,
             "\n      strata::grammar::detail::work();"),
            ("  std::optional<ThreadPool> thread_pool;", 1, "\n  size_t strata_mask_bytes = 0;"),
            ("    auto cur_adaptive_token_mask_cache = grammar_matcher.GetAdaptiveTokenMask(is_root_rule);", 1,
             "\n    strata_mask_bytes += MemorySize(cur_adaptive_token_mask_cache);"
             "\n    strata::grammar::detail::bound(strata_mask_bytes, 16 * 1024 * 1024);"
             "\n    strata::grammar::detail::work();")],
    }
    changed = {}
    for name, replacements in edits.items():
        path = dest / name
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        for anchor, count, addition in replacements:
            if text.count(anchor) != count:
                raise RuntimeError(f"resource-guard source drift: {name}: {anchor}")
            text = text.replace(anchor, anchor + addition)
        text = '#include "strata/core/grammar_budget.hpp"\n' + text
        result = text.encode("utf-8")
        path.write_bytes(result)
        changed[name] = {"original_sha256": hashlib.sha256(raw).hexdigest(),
                         "patched_sha256": hashlib.sha256(result).hexdigest()}
    # The pinned converter omits additionalProperties when patternProperties is
    # present without named properties. Keep the extra-key branch, as its named
    # properties path already does. Full final validation still checks overlap
    # between patterns and additional values, and property-name restrictions.
    path = dest / "cpp/json_schema_converter.cc"
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    anchor = '''        }
      } else {
        int32_t key_rule_id = CreateRule(spec.property_names, rule_name + "_name");'''
    replacement = '''        }
        if (additional_property) {
          int32_t key_expression = spec.property_names
              ? RuleRef(CreateRule(spec.property_names, rule_name + "_extra_name"))
              : KeyPatternExpression();
          int32_t value_rule_id =
              CreateRule(additional_property, rule_name + "_" + additional_suffix);
          property_choices.push_back(Sequence(
              {beginning_separator,
               FormatOtherProperty(key_expression, value_rule_id, rule_name,
                                   additional_suffix, additional_property)}
          ));
        }
      } else {
        int32_t key_rule_id = CreateRule(spec.property_names, rule_name + "_name");'''
    if text.count(anchor) != 1:
        raise RuntimeError("JSON converter source drift: dynamic object additional properties")
    result = text.replace(anchor, replacement).encode("utf-8")
    path.write_bytes(result)
    changed["cpp/json_schema_converter.cc"] = {
        "original_sha256": hashlib.sha256(raw).hexdigest(),
        "patched_sha256": hashlib.sha256(result).hexdigest()}
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="new dependency directory (for example build/xgrammar)")
    args = ap.parse_args()
    dest = args.out.resolve()
    marker = dest / "PIN.json"
    if marker.exists():
        pin = json.loads(marker.read_text(encoding="utf-8"))
        if pin.get("backend") != BACKEND:
            raise RuntimeError("existing dependency has a different pin; choose a new directory")
        for name, hashes in pin["patched_files"].items():
            if hashlib.sha256((dest / name).read_bytes()).hexdigest() != hashes["patched_sha256"]:
                raise RuntimeError(f"existing dependency was edited: {name}; choose a new directory")
        print(dest)
        return
    if dest.exists():
        raise RuntimeError("destination already exists without a complete pin; choose a new directory")
    dest.mkdir(parents=True)
    dependencies = [extract(XGRAMMAR, dest), extract(DLPACK, dest / "3rdparty/dlpack")]
    changed = patch(dest)
    marker.write_text(json.dumps({"backend": BACKEND, "dependencies": dependencies,
                                 "patched_files": changed}, indent=2) + "\n", encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
