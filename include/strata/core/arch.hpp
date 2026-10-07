// include/strata/core/arch.hpp - which model family a file is, and the metadata prefix that names its keys.
//
// The engine began as a single-architecture engine: `check_architecture` refused anything whose
// `general.architecture` was not `qwen4exp`, and every GGUF key it read was spelled with that prefix at the
// call site.  GLM-5.3-Flash (`glm5-next`) is the second family, so the prefix becomes a variable and lives
// here rather than in a dozen string literals.
//
// Deliberately a LEAF header - <string> and nothing else.  Both the artifact layer (which reads the file)
// and the core layer (which owns the geometry) need to name an arch, and neither should have to include the
// other to do it.
#pragma once

#include <string>

namespace strata::core {

enum class Arch {
    Unknown,    ///< sentinel: "read it from the file's general.architecture"
    Qwen4Exp,   ///< Qwen3.8-Flash-Next and its Coder / Swift / Unsloth variants
    Glm5Next,   ///< GLM-5.3-Flash (312B total, 17B active)
};

/// The arch's GGUF metadata prefix: every key this engine reads is `<prefix>.<name>`.
inline const char* arch_meta_prefix(Arch a) {
    switch (a) {
    case Arch::Qwen4Exp: return "qwen4exp";
    case Arch::Glm5Next: return "glm5-next";
    default: return "";
    }
}

/// A name for messages and logs.
inline const char* arch_name(Arch a) {
    switch (a) {
    case Arch::Qwen4Exp: return "Qwen3.8-Flash-Next (qwen4exp)";
    case Arch::Glm5Next: return "GLM-5.3-Flash (glm5-next)";
    default: return "unknown";
    }
}

/// `general.architecture` -> Arch.  Both spellings the reference accepts are accepted here: the artifact on
/// disk says `glm5-next`, and `glm5next` is its alias in llama.cpp's own arch table.
inline bool arch_from_string(const std::string& s, Arch& out) {
    if (s == "qwen4exp") { out = Arch::Qwen4Exp; return true; }
    if (s == "glm5-next" || s == "glm5next") { out = Arch::Glm5Next; return true; }
    return false;
}

/// What to tell a user whose file is neither.
inline const char* arch_list() { return "'qwen4exp' or 'glm5-next'"; }

/// Does this family carry a PLE (per-layer embedding) table?  It is a `qwen4exp` feature - GLM-5.3-Flash has
/// no such tensor, so a loader that REQUIRES one refuses a GLM file before it can ever name the real gap.
inline bool arch_has_ple(Arch a) { return a == Arch::Qwen4Exp; }

}  // namespace strata::core
