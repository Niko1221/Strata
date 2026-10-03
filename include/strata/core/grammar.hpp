// Native raw-GBNF constraints. Immutable compilation is shared; progress is private.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::grammar {

inline constexpr const char* kBackend = "xgrammar-0.2.8-strata-budget1-json1";
inline constexpr size_t kMaxSourceBytes = 8192;
inline constexpr size_t kMaxHistoryTokens = 8192;
inline constexpr size_t kMaxOutputBytes = 65536;

struct ProtocolTokens {
    int32_t think_end = -1, call_start = -1, call_end = -1;
};
struct Scope {
    bool thinking = false, tools = false;
    uint32_t reasoning_tokens = 0; // zero: no separate thinking limit
    bool enabled() const { return thinking || tools; }
};

struct Vocabulary {
    // Actual emitted bytes. Special/control/unused IDs have empty strings and
    // cannot be selected as text. Stop IDs are separate control transitions.
    std::vector<std::string> bytes;
    std::vector<int32_t> stop_ids;
    ProtocolTokens protocol;
    std::vector<int32_t> text_mask, newline_ids;
    std::string identity; // stable diagnostic fingerprint, not an authentication token
    static std::shared_ptr<const Vocabulary> from_pack(const std::filesystem::path& path,
                                                      std::vector<int32_t> stop_ids);
    static std::shared_ptr<const Vocabulary> from_bytes(std::vector<std::string> bytes,
                                                       std::vector<int32_t> stop_ids, ProtocolTokens protocol = {});
};

struct Compiled;
struct Checkpoint {
    std::shared_ptr<const Compiled> compiled;
    std::vector<int32_t> tokens;
    Scope scope;
};

// Explicit local debugging only; no HTTP/native-pipe inspector endpoint. The
// mask is the union of viable continuations, not an enumerated parse tree or a
// probability distribution. Caller-owned occurrence IDs are separate from the
// exact-source/backend/vocabulary definition fingerprint.
struct TokenPreview {
    int32_t id;
    std::string bytes; // at most 32 raw bytes; render escaped/hex, not as UTF-8
    bool stop, bytes_truncated;
};
struct Inspection {
    std::string definition, mask_fingerprint;
    size_t committed_tokens, committed_bytes, legal_tokens;
    bool accepting, terminated, preview_complete;
    uint64_t work_used;
    std::vector<TokenPreview> preview;
};

struct PrefixMasks {
    std::vector<int32_t> bits; // row-major, ceil(vocabulary/32) words per row
    int rows = 0;
    int blocked_draft = -1; // zero-based proposal that has no continuation
    bool end_draft = false; // the last proposal examined is a legal end control
};

class Matcher {
public:
    // JSON bounds/alternatives and discarded speculative masks share this
    // sequence budget. Raw GBNF retains its smaller existing default.
    static constexpr uint64_t kJsonWorkLimit = 64000000;
    explicit Matcher(std::shared_ptr<const Compiled> compiled, uint64_t work_limit = 2000000, Scope scope = {});
    ~Matcher();
    Matcher(Matcher&&) noexcept;
    Matcher& operator=(Matcher&&) noexcept;
    Matcher(const Matcher&) = delete;
    Matcher& operator=(const Matcher&) = delete;
    const std::vector<int32_t>& mask();
    // Masks before each reachable proposal, then its replacement/bonus row.
    // No row follows an illegal/end proposal. x (the pending feedback token)
    // is already committed and must NOT be passed again. At most seven drafts.
    // Tentative work is charged here, but tentative progress is discarded.
    void prefix_masks(const int32_t* drafts, int count, PrefixMasks& out);
    bool allows(int32_t token);
    bool accept(int32_t token); // illegal token: false with no state change
    bool complete() const;     // accepting prefix, possibly with legal continuations
    bool terminated() const;   // an allowed end control was committed
    bool failed() const;
    Matcher fork() const;
    Checkpoint checkpoint() const;
    void restore(const Checkpoint& checkpoint);
    const std::vector<int32_t>& tokens() const;
    const std::string& identity() const;
    uint64_t work_used() const;
    // Classification of the last committed token, never a speculative draft.
    // Only "answer" advances the user's grammar. Protocol separators are control.
    const char* channel() const;
    const char* phase() const;
    Inspection inspect(size_t preview_limit = 16) const; // bounded 0..64; no parent progress/budget change
private:
    struct Impl;
    explicit Matcher(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Compiler {
public:
    // Bounded string lengths expand the schema automaton across the vocabulary.
    // Codex's 36-character title uses >100M work units with Qwen's 248K tokens.
    // The independent 2.5s deadline and 16 MiB compiled-size limit still apply.
    static constexpr uint64_t kJsonWorkLimit = 150000000;
    explicit Compiler(std::shared_ptr<const Vocabulary> vocabulary, size_t entries = 8,
                      size_t cache_bytes = 64 * 1024 * 1024);
    ~Compiler();
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;
    std::shared_ptr<const Compiled> compile(const std::string& source,
                                          uint64_t work_limit = 5000000, bool json_schema = false);
    size_t cache_entries() const;
    size_t cache_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void validate_source(const std::string& source);
bool valid_utf8(const std::string& text);

} // namespace strata::grammar
