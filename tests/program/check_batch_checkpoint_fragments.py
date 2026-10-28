#!/usr/bin/env python3
"""Compile real checkpoint-admission/return fragments with synthetic CPU state.

This is not a backend build or an inference test. The checkpoint join is a stub;
no device copy, model, user input, network, or third-party Python package is used.
Run from any directory; --output preserves generated C++, commands and logs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BASE = "5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d"
PATHS = ("src/program/generate.cpp", "sycl/src/program/generate.cpp")


def between(text, start, end, *, last=False):
    at = text.rfind(start) if last else text.find(start)
    if at < 0:
        raise RuntimeError(f"missing source anchor: {start}")
    stop = text.find(end, at)
    if stop < 0:
        raise RuntimeError(f"missing end anchor: {end}")
    return text[at:stop]


def fragments(text, native):
    admission = between(
        text, "                    const ConvCheckpoint* best = nullptr;",
        "                    const auto tc4 = Clock::now();" if native else
        '                    std::fprintf(stderr, "strata batch: slot %d takes', last=True)
    handback = between(
        text, "                    checks.clear();   // the main session's checkpoints were of the conversation it held before",
        '                    std::fprintf(stderr, "strata batch: slot %d gave back')
    retention = between(
        text, "            if (as_tail) {   // only one tail checkpoint stays alive:",
        "            if (ckpt_trace && checks.size() < before_evict) {" if native else
        "            return true;\n        };")
    return admission, handback, retention


RESET = """                    // An ordinary whole-slot return replaces the old chain; its tail identity cannot follow it.
                    if (slot_ck == nullptr && !bs[(size_t) slot_source].img &&
                        !(bs[(size_t) slot_source].partial && admit_slot == slot_source)) tail_ckpt_len = -1;
"""


PRELUDE = r'''
#include "strata/program/batch_checkpoints.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <numeric>
#include <string>

using ConvCheckpoint = strata::core::ConversationCheckpoint;
using Clock = std::chrono::steady_clock;
static long fail_at = -1;
static bool count_allocations = false, failed_allocation = false, move_enabled = false;
static size_t allocations = 0;
static int joins = 0;
static std::vector<ConvCheckpoint>* pending = nullptr;
void* operator new(std::size_t n) {
    if (count_allocations) ++allocations;
    if (fail_at == 0) { fail_at = -1; failed_allocation = true; throw std::bad_alloc(); }
    if (fail_at > 0) --fail_at;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
// libstdc++ stable_sort uses nothrow temporary storage. Route that through the
// same injected allocator so sanitizer allocation/deallocation families match.
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (const std::bad_alloc&) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n); } catch (const std::bad_alloc&) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
bool slot_ckpt_move_env() { return move_enabled; }
namespace strata::core {
// Synchronous synthetic completion, not evidence about actual async GPU copies.
void conversation_checkpoint_join(const void* address) {
    if (address != nullptr) return;
    ++joins;
    if (pending != nullptr) {
        for (auto& c : *pending) {
            c.gdn[0] = 222;
            for (auto& part : c.stage_parts) part.gdn[0] = 223;
        }
        pending = nullptr;
    }
}
}
'''


CONTEXT = r'''
struct BSlot {
    bool img = false, partial = false;
    int64_t partial_from0 = 0;
    ConvCheckpoint::PromptPlan partial_plan;
    std::vector<int32_t> ids;
    std::vector<ConvCheckpoint> checks;
};
struct Context {
    struct Options { int prompt_cache = 6; bool prompt_cache_spread = false; } o;
    std::vector<ConvCheckpoint> checks;
    std::vector<int32_t> live;
    std::vector<strata::core::ConversationImageKey> live_imgs;
    std::vector<BSlot> bs = std::vector<BSlot>(2);
    std::vector<int> stages;
    CheckpointPool ckpt_pool;
    int admit_slot = 0, lent_slot = -1, slot_source = 1;
    int64_t resume = 0, turn_at = -1, moved_in = -1, tail_ckpt_len = -1;
    int64_t slot_tokens = 0, resumed_from0 = 0, resume_plan_at = 0, resume_plan_end = 0;
    uint64_t check_clock = 1000;
    bool from_live = false;
    ConvCheckpoint::PromptPlan resume_plan;
    const ConvCheckpoint* slot_ck = nullptr;
    explicit Context(bool pool = false) : ckpt_pool(pool, 8) {}
    void native_admit(); void sycl_admit();
    void native_return(); void sycl_return();
    void native_retain(ConvCheckpoint c, int64_t L, bool as_tail);
    void sycl_retain(ConvCheckpoint c, int64_t L, bool as_tail);
};
'''


TESTS = r'''
static int checks_run = 0, failures = 0;
#define CHECK(expr) do { ++checks_run; if (!(expr)) { ++failures; \
    std::printf("FAIL line %d: %s\n", __LINE__, #expr); } } while (false)
bool equal(const ConvCheckpoint& a, const ConvCheckpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.gdn != b.gdn || a.ple != b.ple ||
        a.tails != b.tails || a.dead != b.dead || a.block_pos != b.block_pos ||
        a.used != b.used || a.dev_tag != b.dev_tag || a.pinned != b.pinned ||
        a.stage_parts.size() != b.stage_parts.size() || a.plan.chunk != b.plan.chunk ||
        a.plan.from != b.plan.from || a.plan.to != b.plan.to || a.plan.lend != b.plan.lend ||
        a.plan.pipe != b.plan.pipe || a.plan.single != b.plan.single || a.plan.chunked != b.plan.chunked) return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!equal(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}
bool equal(const std::vector<ConvCheckpoint>& a, const std::vector<ConvCheckpoint>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!equal(a[i], b[i])) return false;
    return true;
}
ConvCheckpoint point(size_t length, size_t stages) {
    ConvCheckpoint c;
    c.ids.resize(length); std::iota(c.ids.begin(), c.ids.end(), 10);
    c.gdn = {11, 12, static_cast<uint8_t>(length)}; c.ple = {21, 22};
    c.tails = {31, 32, 33}; c.dead = {41}; c.block_pos = {51, 52};
    c.used = length; c.dev_tag = length + 100;
    c.plan = {4, 0, 80, 2, 0.5, true, true};
    for (size_t k = 0; k < stages; ++k) {
        c.stage_parts.push_back(point(length, 0));
        c.stage_parts.back().gdn[1] = static_cast<uint8_t>(70 + k);
        c.stage_parts.back().dev_tag += k + 1;
    }
    return c;
}
void setup(Context& s, size_t stages = 2) {
    s.stages.resize(stages); s.live = point(67, 0).ids;
    // Deliberately unsorted: erasing a moved source entry must not invalidate later copies.
    for (size_t n : {21, 61, 5, 37, 13}) s.checks.push_back(point(n, stages));
}
std::vector<size_t> lengths(const std::vector<ConvCheckpoint>& v) {
    std::vector<size_t> out; for (const auto& c : v) out.push_back(c.ids.size());
    std::sort(out.begin(), out.end()); return out;
}
const ConvCheckpoint* find(const std::vector<ConvCheckpoint>& v, size_t n) {
    for (const auto& c : v) if (c.ids.size() == n) return &c;
    return nullptr;
}
using Method = void (Context::*)();
void assert_complete_copies(const std::vector<ConvCheckpoint>& copies,
                            const std::vector<ConvCheckpoint>& original) {
    for (const auto& c : copies) {
        const auto* donor = find(original, c.ids.size());
        CHECK(donor != nullptr);
        if (donor) CHECK(equal(c, *donor));
    }
}
void admissions(Method admit, bool native) {
    for (size_t stage_count = 0; stage_count <= 3; ++stage_count) {
        for (int budget : {0, 1, 2, 3, 5, 9}) {
            Context s; setup(s, stage_count); s.o.prompt_cache = budget;
            const auto source = s.checks; joins = 0; (s.*admit)();
            std::vector<size_t> want;
            if (budget == 1) want = {61};
            else if (budget == 2) want = {5, 61};
            else if (budget == 3) want = {5, 37, 61};
            else if (budget > 3) want = {5, 13, 21, 37, 61};
            CHECK(lengths(s.bs[0].checks) == want);
            CHECK(equal(s.checks, source));
            CHECK(joins == (native && budget > 0 ? 1 : 0));
            assert_complete_copies(s.bs[0].checks, source);
            for (auto& c : s.bs[0].checks) {
                c.gdn[0] = 99;
                for (auto& part : c.stage_parts) part.ple[0] = 98;
            }
            CHECK(equal(s.checks, source));
        }
        Context s; setup(s, stage_count);
        auto bad = point(64, stage_count); bad.ids[0] = -1; s.checks.push_back(bad);
        bad = point(65, stage_count); bad.stage_parts.push_back(point(65, 0)); s.checks.push_back(bad);
        bad = point(63, stage_count); bad.imgs.push_back({0, 1}); s.checks.push_back(bad);
        s.checks.push_back(point(67, stage_count)); // The current live extent is not a prior checkpoint.
        s.checks.push_back(point(70, stage_count)); // Outside the copied range.
        s.checks.push_back(point(0, stage_count));
        if (stage_count != 0) {
            bad = point(62, stage_count); bad.stage_parts.back().ids[0] = -2; s.checks.push_back(bad);
            bad = point(66, stage_count); bad.stage_parts.back().imgs.push_back({0, 2}); s.checks.push_back(bad);
        }
        (s.*admit)(); CHECK(lengths(s.bs[0].checks) == (std::vector<size_t>{5,13,21,37,61}));
        Context image; setup(image, stage_count); image.bs[0].img = true;
        (image.*admit)(); CHECK(image.bs[0].checks.empty());
    }
    if (!native) return;
    for (bool pool : {false, true}) {
        for (bool pinned : {false, true}) {
            Context s(pool); setup(s); s.turn_at = 61; s.checks[1].pinned = pinned;
            if (pool) s.ckpt_pool.give(point(67, 2));
            const auto source = s.checks; const auto* storage = s.checks[1].gdn.data();
            move_enabled = true; (s.*admit)(); move_enabled = false;
            CHECK(s.bs[0].checks.size() == 5);
            assert_complete_copies(s.bs[0].checks, source);
            CHECK(s.checks.size() == (pinned ? 5U : 4U));
            if (pinned) CHECK(equal(s.checks, source));
            else CHECK(find(s.bs[0].checks, 61)->gdn.data() == storage);
        }
    }
    Context borrowed; setup(borrowed); borrowed.checks.erase(borrowed.checks.begin() + 1);
    borrowed.bs[1].checks.push_back(point(61, 2)); borrowed.lent_slot = 1; borrowed.resume = 61;
    borrowed.turn_at = 61;
    const auto donor = borrowed.bs[1].checks, own = borrowed.checks;
    move_enabled = true; borrowed.native_admit(); move_enabled = false;
    CHECK(lengths(borrowed.bs[0].checks) == (std::vector<size_t>{5,13,21,37,61}));
    CHECK(equal(borrowed.bs[1].checks, donor)); CHECK(equal(borrowed.checks, own));
    Context evicted; setup(evicted); evicted.o.prompt_cache = 2; evicted.turn_at = 61;
    evicted.checks[1].used = 1; evicted.checks[3].pinned = true;
    const auto untouched = evicted.checks; move_enabled = true; evicted.native_admit(); move_enabled = false;
    CHECK(lengths(evicted.bs[0].checks) == (std::vector<size_t>{5,37})); CHECK(equal(evicted.checks, untouched));
    Context moved; setup(moved); moved.moved_in = 61;
    move_enabled = true; moved.native_admit(); move_enabled = false;
    CHECK(moved.checks.size() == 4); CHECK(find(moved.checks, 61) == nullptr);
    Context ordinary; setup(ordinary); const auto ordinary_source = ordinary.checks;
    move_enabled = true; ordinary.native_admit(); move_enabled = false;
    CHECK(equal(ordinary.checks, ordinary_source));
    for (bool spread : {false, true}) {
        Context policy; setup(policy); policy.o.prompt_cache = 3; policy.o.prompt_cache_spread = spread;
        policy.checks[0].used = 90; policy.checks[1].used = 1;
        policy.checks[3].used = 2; policy.checks[4].used = 80;
        policy.native_admit();
        CHECK(lengths(policy.bs[0].checks) == (spread ? std::vector<size_t>{5,37,61}
                                                     : std::vector<size_t>{5,13,21}));
    }
    Context async; setup(async); pending = &async.checks; async.native_admit();
    CHECK(pending == nullptr);
    for (const auto& c : async.bs[0].checks) {
        CHECK(c.gdn[0] == 222);
        for (const auto& part : c.stage_parts) CHECK(part.gdn[0] == 223);
    }
}
void allocation_sweep(Method admit, bool native) {
    for (size_t stages = 0; stages <= 3; ++stages) {
        for (bool pool : {false, true}) {
            if (pool && !native) continue;
            Context measure(pool); setup(measure, stages); measure.turn_at = 61;
            if (pool) measure.ckpt_pool.give(point(67, stages));
            move_enabled = native; allocations = 0; count_allocations = true;
            (measure.*admit)(); count_allocations = false;
            const size_t total = allocations;
            CHECK(total > 0);
            for (size_t fault = 0; fault < total; ++fault) {
                Context s(pool); setup(s, stages); s.turn_at = 61;
                if (pool) s.ckpt_pool.give(point(67, stages));
                const auto source = s.checks;
                failed_allocation = false; fail_at = static_cast<long>(fault);
                bool escaped = false;
                try { (s.*admit)(); } catch (const std::bad_alloc&) { escaped = true; }
                fail_at = -1;
                CHECK(!escaped); CHECK(failed_allocation); CHECK(s.bs[0].checks.size() <= 5);
                assert_complete_copies(s.bs[0].checks, source);
                // A failed metadata/copy allocation must not consume the move donor.
                // stable_sort may recover from its own failed temporary allocation.
                if (s.bs[0].checks.size() < 5 || !native) CHECK(equal(s.checks, source));
                else {
                    CHECK(s.checks.size() == 4);
                    CHECK(find(s.checks, 61) == nullptr);
                    assert_complete_copies(s.checks, source);
                }
            }
            move_enabled = false;
            std::printf("%s allocation sweep: stages=%zu pool=%d allocation sites=%zu\n",
                        native ? "native" : "sycl", stages, pool, total);
        }
    }
}
void returns(Method give_back, bool native) {
    for (bool pinned : {false, true}) {
        Context s; setup(s); s.bs[1].checks = s.checks; s.bs[1].checks[0].pinned = pinned;
        s.bs[1].ids = s.live; s.slot_tokens = 67; s.tail_ckpt_len = 21;
        const auto donor = s.bs[1].checks;
        (s.*give_back)();
        CHECK(s.tail_ckpt_len == -1); CHECK(equal(s.bs[1].checks, donor));
        CHECK(lengths(s.checks) == (std::vector<size_t>{5,13,21,37,61}));
        auto c = point(65, 2);
        if (native) s.native_retain(std::move(c), 65, true); else s.sycl_retain(std::move(c), 65, true);
        CHECK(find(s.checks, 21) != nullptr); CHECK(s.tail_ckpt_len == 65);
    }
    // BYIELD and selected-checkpoint bookkeeping deliberately do not take the reset.
    for (int special : {0, 1, 2}) {
        Context s; setup(s); s.bs[1].checks = s.checks; s.bs[1].ids = s.live;
        s.slot_tokens = 67; s.tail_ckpt_len = 21;
        if (special == 0) { s.bs[1].partial = true; s.admit_slot = 1; s.bs[1].partial_from0 = 9; }
        if (special == 1) { s.slot_ck = &s.bs[1].checks[1]; s.slot_tokens = 61; }
        if (special == 2) s.bs[1].img = true;
        (s.*give_back)(); CHECK(s.tail_ckpt_len == 21);
        if (special == 0) { CHECK(!s.bs[1].partial); CHECK(s.resumed_from0 == 9); }
        if (special == 1) CHECK(s.live.size() == 61);
    }
    Context filter; setup(filter); filter.bs[1].checks = filter.checks; filter.bs[1].ids = filter.live;
    auto bad = point(24, 2); bad.ids[0] = -1; filter.bs[1].checks.push_back(bad);
    bad = point(30, 2); bad.imgs.push_back({0, 3}); filter.bs[1].checks.push_back(bad);
    filter.slot_tokens = 37; (filter.*give_back)();
    CHECK(lengths(filter.checks) == (std::vector<size_t>{5,13,21,37}));
    if (native) {
        Context s; setup(s); s.bs[1].checks = s.checks; s.bs[1].ids = s.live;
        s.slot_tokens = 67; s.admit_slot = 1; move_enabled = true; s.native_return(); move_enabled = false;
        CHECK(s.bs[1].checks.empty()); CHECK(s.checks.size() == 5);
    }
}
int baseline_probe() {
    for (bool native : {true, false}) {
        Context s; setup(s); if (native) s.native_admit(); else s.sycl_admit();
        if (s.bs[0].checks.size() != 5) {
            ++failures; std::printf("REGRESSION %s admission_chain\n", native ? "native" : "sycl");
        }
        Context r; setup(r); r.bs[1].checks = r.checks; r.bs[1].ids = r.live;
        r.slot_tokens = 67; r.tail_ckpt_len = 21;
        if (native) r.native_return(); else r.sycl_return();
        const bool stale_tail = r.tail_ckpt_len != -1;
        if (native) r.native_retain(point(65, 2), 65, true); else r.sycl_retain(point(65, 2), 65, true);
        if (stale_tail && find(r.checks, 21) == nullptr) {
            ++failures; std::printf("REGRESSION %s whole_slot_tail\n", native ? "native" : "sycl");
        }
    }
    std::printf("baseline: %d expected regression failures\n", failures);
    return failures ? 1 : 0;
}
int main(int argc, char**) {
    if (argc > 1) return baseline_probe();
    admissions(&Context::native_admit, true); admissions(&Context::sycl_admit, false);
    allocation_sweep(&Context::native_admit, true); allocation_sweep(&Context::sycl_admit, false);
    returns(&Context::native_return, true); returns(&Context::sycl_return, false);
    std::printf("patched: %d checks, %d failures\n", checks_run, failures);
    return failures ? 1 : 0;
}
'''


def generate(sources):
    pool = between(sources[0], "class CheckpointPool {", "\n};") + "\n};\n"
    result = PRELUDE + pool + CONTEXT
    for index, source in enumerate(sources):
        native = index == 0
        name = "native" if native else "sycl"
        admission, handback, retention = fragments(source, native)
        result += f"\nvoid Context::{name}_admit() {{\n    BSlot& sl = bs[0];\n" + admission
        if native:
            result += "    (void)tc3; (void)move_best;\n"
        result += "}\n"
        result += f"\nvoid Context::{name}_return() {{\n" + handback + "}\n"
        result += f"\nvoid Context::{name}_retain(ConvCheckpoint c, int64_t L, bool as_tail) {{\n" + retention
        if native:
            result += "    (void)te0; (void)before_evict;\n"
        result += "}\n"
    return result + TESTS


def run(command, log, *, expected=0):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("$ " + shlex.join(map(str, command)) + "\n" + result.stdout)
    if result.returncode != expected:
        raise RuntimeError(f"exit {result.returncode}, expected {expected}; see {log}")
    if "AddressSanitizer" in result.stdout or "runtime error:" in result.stdout:
        raise RuntimeError(f"sanitizer diagnostic; see {log}")
    return result.stdout


def validate(args, output):
    patched = [(ROOT / path).read_text() for path in PATHS]
    base = [subprocess.check_output(["git", "show", f"{args.base}:{path}"], cwd=ROOT, text=True) for path in PATHS]
    for i in range(2):
        _, returned, retained = fragments(patched[i], i == 0)
        _, old_returned, old_retained = fragments(base[i], i == 0)
        if returned.count(RESET) != 1 or returned.replace(RESET, "") != old_returned:
            raise RuntimeError(f"return block changed beyond tail reset: {PATHS[i]}")
        if retained != old_retained:
            raise RuntimeError(f"retention block unexpectedly changed: {PATHS[i]}")
    modes = [("optimized", ["-O2"])]
    if not args.no_sanitize:
        modes.append(("sanitized", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]))
    summaries = []
    for version, sources in (("base", base), ("patched", patched)):
        source = output / f"{version}.cpp"
        source.write_text(generate(sources))
        for mode, flags in modes:
            executable = output / f"{version}-{mode}"
            command = shlex.split(args.cxx) + ["-std=c++20", "-Wall", "-Wextra", "-Wno-mismatched-new-delete"]
            command += flags + ["-I", str(ROOT / "include"), str(source), "-o", str(executable)]
            run(command, output / f"{version}-{mode}-compile.log")
            result = run([str(executable)] + (["baseline"] if version == "base" else []),
                         output / f"{version}-{mode}-run.log", expected=1 if version == "base" else 0)
            if version == "base":
                for backend in ("native", "sycl"):
                    for problem in ("admission_chain", "whole_slot_tail"):
                        if f"REGRESSION {backend} {problem}" not in result:
                            raise RuntimeError(f"baseline did not reproduce {backend} {problem}")
            summary = f"{version}/{mode}: {result.splitlines()[-1]}"
            print(summary, flush=True)
            summaries.append(summary)
    manifest = {"base": args.base, "results": summaries,
                "production_source_sha256": {path: hashlib.sha256(source.encode()).hexdigest()
                                             for path, source in zip(PATHS, patched)},
                "limitations": "CPU fragments only; join is a stub; no full backend build or GPU execution"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Return-block boundary and unchanged retention checks: passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE, help="git revision for the original regression (default: patch base)")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"), help="C++ compiler command")
    parser.add_argument("--output", type=Path, help="directory to retain generated sources, logs and binaries")
    parser.add_argument("--no-sanitize", action="store_true", help="run only optimized builds")
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        validate(args, output)
    else:
        with tempfile.TemporaryDirectory(prefix="strata-checkpoint-fragments-") as directory:
            validate(args, Path(directory))


if __name__ == "__main__":
    main()
