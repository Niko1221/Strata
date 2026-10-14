#pragma once

namespace strata::core::conversation_detail {

/// Copies `message` into the caller's sink and returns the pointer to log, allocating nothing of its own except
/// that copy, which is attempted inside a try. Host-only, no CUDA, header-only.
///
/// This exists for the paths that report an allocation failure. There the error string is the thing that fails: a
/// handler that builds `"restore failed: " + e.what()` throws on its way out of the catch, and the escaping
/// `bad_alloc` ends the engine for a reason that has nothing to do with the state. So the message is always a
/// string that already exists - a literal, or `what()` - never one assembled here, and a throw in the copy is
/// swallowed because the caller gets its false either way and the log line still prints from the literal.
///
/// `sink` is a template so a test can hand it an assignment that throws on purpose; a `std::string&` is the real
/// caller. Returns `message` when it holds text, otherwise `fallback`, which must outlive the call.
template<class Sink>
const char* report_restore(Sink&& sink, const char* message, const char* fallback) {
    const char* what = (message != nullptr && *message != '\0') ? message : fallback;
    try {
        sink = what;
    } catch (...) {
        // Losing the text in the caller's string beats throwing past the serve loop, which is the whole point.
    }
    return what;
}

}  // namespace strata::core::conversation_detail