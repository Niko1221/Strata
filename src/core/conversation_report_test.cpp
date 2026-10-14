// src/core/conversation_report_test.cpp - reporting a failed running-state restore must never throw.
//
// The serve path catches the bad_alloc a decode raises when there is no RAM left. The first version of that handler
// then built its error string ("checkpoint restore failed: " + e.what()), which wants a heap block on the one path
// that has none, so a second bad_alloc escaped the catch and ended the process for a reason that had nothing to do
// with the state. report_restore() (strata/core/conversation_report.hpp) reports from a literal or from what() and
// swallows a throw in the copy into the caller's string. These tests arm the allocator to fail and check both halves
// of that claim: the pattern the handler used really does throw, and the one it uses now really does not.
//
// Host-only: no CUDA, no device, no model. It runs on any box that compiles the conversation tests.

#include "strata/core/conversation_report.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

using strata::core::conversation_detail::report_restore;

// The allocator, armable. While `failing` is set, a request of `fail_from_bytes` or more throws bad_alloc and is
// counted; smaller requests still succeed, so the runtime can do whatever small bookkeeping a throw needs and the
// test says something about the code under test rather than about the C++ library. Aligned new is left alone.
// Global, because replacing the allocation functions is only allowed at file scope.
static bool failing = false;
static std::size_t fail_from_bytes = 0;
static std::size_t refused = 0;

static void* allocate(std::size_t n) {
    if (failing && n >= fail_from_bytes) { ++refused; throw std::bad_alloc(); }
    void* p = std::malloc(n == 0 ? 1 : n);
    if (p == nullptr) { if (failing) { ++refused; throw std::bad_alloc(); } std::abort(); }
    return p;
}
static void disarm() { failing = false; fail_from_bytes = 0; }

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    if (failing && n >= fail_from_bytes) { ++refused; return nullptr; }
    return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    if (failing && n >= fail_from_bytes) { ++refused; return nullptr; }
    return std::malloc(n == 0 ? 1 : n);   // every replaceable form allocates here, so every delete form frees here
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// A sink whose assignment always fails: the deterministic form of "the message cannot be built".
struct throwing_sink {
    int assignments = 0;
    void operator=(const char*) { ++assignments; throw std::bad_alloc(); }
};

// A sink that fails only on a message too long for its inline buffer, the way a std::string does under pressure.
struct long_only_sink {
    std::string text;
    int refused = 0;
    void operator=(const char* s) {
        if (std::strlen(s) > 15) { ++refused; throw std::bad_alloc(); }
        text = s;
    }
};

const char* const long_message = "not enough RAM to widen the checkpoint's running state";
const char* const fallback = "checkpoint restore failed";

// The shape the handler had: a message assembled on the failure path.
bool assembled_message_still_throws() {
    std::string error;
    bool threw = false;
    failing = true; fail_from_bytes = 16; refused = 0;
    try {
        error = std::string("checkpoint restore failed: ") + "out of memory";
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    disarm();
    check(threw, "assembling the error string on the failure path does throw bad_alloc (the bug being guarded)");
    check(refused > 0, "and it is the assembly that asked for the block");
    return threw;
}

// The same failure, reported the way the handler does now.
void guard_holds_under_a_failing_allocator() {
    std::string why;
    const char* logged = nullptr;
    bool threw = false;
    failing = true; fail_from_bytes = 16; refused = 0;
    try {
        logged = report_restore(why, long_message, fallback);
    } catch (...) {
        threw = true;
    }
    disarm();
    check(!threw, "reporting a restore failure does not throw when the message cannot be copied");
    check(logged == long_message, "and the pointer it returns is the literal, so the log line still prints");
}

// A message that fits the string's inline buffer needs no heap at all: nothing is even asked for.
void short_message_asks_for_no_heap() {
    const char* short_message = "no room";   // seven characters: the inline buffer of every implementation here
    std::string why;
    const char* logged = nullptr;
    bool threw = false;
    failing = true; fail_from_bytes = 1; refused = 0;   // refuse every request, however small
    try {
        logged = report_restore(why, short_message, fallback);
    } catch (...) {
        threw = true;
    }
    disarm();
    check(!threw, "a short message is reported without throwing");
    check(refused == 0, "and nothing was asked of the allocator on the way");
    check(logged == short_message && why == short_message, "with the text in both the returned pointer and the sink");
}

void throwing_sink_is_swallowed() {
    throwing_sink sink;
    const char* logged = nullptr;
    bool threw = false;
    try {
        logged = report_restore(sink, long_message, fallback);
    } catch (...) {
        threw = true;
    }
    check(!threw, "a sink that throws on every assignment does not carry the throw out of the report");
    check(sink.assignments == 1, "it was offered the message once");
    check(logged == long_message, "and the caller still gets the message to log");
}

void sink_that_fails_only_on_length() {
    long_only_sink sink;
    const char* logged = nullptr;
    bool threw = false;
    try {
        logged = report_restore(sink, long_message, fallback);
    } catch (...) {
        threw = true;
    }
    check(!threw && sink.refused == 1, "a sink that fails only on a long message does not throw out either");
    check(logged == long_message, "the long message is still what gets logged");

    const char* short_message = "no room";
    const char* logged2 = report_restore(sink, short_message, fallback);
    check(sink.text == short_message && logged2 == short_message, "and a short one still reaches the sink");
}

// Which message is chosen: what was given, else the literal.
void message_selection() {
    std::string why;
    check(report_restore(why, nullptr, fallback) == fallback, "no message at all falls back to the literal");
    check(report_restore(why, "", fallback) == fallback, "an empty message falls back to the literal too");
    check(report_restore(why, long_message, fallback) == long_message, "a message is used when there is one");
    check(why == long_message, "and it is the text the caller's string received");
}

// what() is read while the exception is alive, which is the rule the callers follow: the pointer does not outlive
// the handler, so the report is made inside it.
void what_is_reported_inside_the_handler() {
    std::string why;
    std::string captured;
    bool threw = false;
    try {
        try {
            throw std::runtime_error("the stored checkpoint holds a codec this build does not know");
        } catch (const std::exception& e) {
            const char* logged = report_restore(why, e.what(), fallback);
            captured = logged;   // copied here: after the handler, what() is gone
        }
    } catch (...) {
        threw = true;
    }
    check(!threw, "a std::exception during a restore is reported, not rethrown");
    check(captured == "the stored checkpoint holds a codec this build does not know", "with the exception's own words");
    check(why == captured, "and the same text in the caller's string");
}

// The same, with the allocator failing while what() is being reported. Two things make this the sharpest of these
// checks. The message is longer than a std::string's small-string buffer - 15 bytes on libstdc++, 22 on libc++ - so
// constructing the exception took a heap block and what() hands out storage the exception owns, not a literal the
// test could read whenever it likes. And the allocator is armed only after that construction: what is being tested
// is that reporting the failure does not need the block that is being withheld.
//
// what() does not outlive the handler, so the text the report returned is compared HERE, while `e` is alive. Nothing
// is copied out to keep it alive: a std::string built for the purpose would itself want the heap this test refuses,
// and would then be measuring its own allocation rather than the report's. Only booleans leave the handler.
void what_survives_a_failing_allocator() {
    const char* const words = "the pooled indexer state could not be widened: there is no RAM left for the state";
    check(std::strlen(words) > 22, "the message is past every implementation's small-string buffer");
    std::string why;
    bool threw = false, refused_one = false, own_pointer = false, text_ok = false;
    try {
        throw std::runtime_error(words);   // its text is heap storage, taken before the allocator was armed
    } catch (const std::exception& e) {
        // 16: the first request a std::string makes once the text no longer fits its 15-byte inline buffer
        failing = true; fail_from_bytes = 16; refused = 0;
        try {
            const char* logged = report_restore(why, e.what(), fallback);
            own_pointer = (logged == e.what());          // the exception's own text, not a copy of it
            text_ok = (std::strcmp(logged, words) == 0);  // read while the exception that owns it is still alive
        } catch (...) {
            threw = true;
        }
        refused_one = (refused > 0);   // read here, with everything else that belongs to the handler
        disarm();                      // before any assertion: check() prints, and printing may allocate
    }
    check(!threw, "a restore failure is reported without throwing even when the copy cannot allocate");
    check(refused_one, "and the copy really was refused, so the pressure was actually applied");
    check(own_pointer, "the pointer returned is the exception's own text");
    check(text_ok, "and that text is the exception's words, compared while the exception was still alive");
}
}  // namespace

int main() {
    assembled_message_still_throws();
    guard_holds_under_a_failing_allocator();
    short_message_asks_for_no_heap();
    throwing_sink_is_swallowed();
    sink_that_fails_only_on_length();
    message_selection();
    what_is_reported_inside_the_handler();
    what_survives_a_failing_allocator();

    std::printf("conversation_report_test: %d checks passed\n", checks);
}