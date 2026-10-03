#include "strata/core/grammar.hpp"
#include "strata/core/grammar_budget.hpp"

#include <xgrammar/config.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <list>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace strata::grammar {
using namespace std::chrono_literals;

static std::string source_identity(const std::string& source) {
    uint64_t h = 14695981039346656037ull;
    for (unsigned char byte : source) h = (h ^ byte) * 1099511628211ull;
    std::ostringstream out;
    out << "root-gbnf-v1-fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << h;
    return out.str(); // diagnostic only; cache and checkpoints also compare actual ownership/source
}

bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        uint32_t c = (uint8_t) s[i++];
        if (c < 128) continue;
        int n;
        uint32_t minimum;
        if (c >= 0xc2 && c <= 0xdf) { n = 1; c &= 0x1f; minimum = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { n = 2; c &= 0x0f; minimum = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { n = 3; c &= 7; minimum = 0x10000; }
        else return false;
        if (i + n > s.size()) return false;
        while (n--) {
            const uint8_t next = (uint8_t) s[i++];
            if ((next & 0xc0) != 0x80) return false;
            c = (c << 6) | (next & 0x3f);
        }
        if (c < minimum || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff)) return false;
    }
    return true;
}

void validate_source(const std::string& source) {
    if (source.empty() || source.size() > kMaxSourceBytes || source.find('\0') != std::string::npos ||
        !valid_utf8(source))
        throw std::runtime_error("grammar must be 1..8192 UTF-8 bytes without a raw NUL");
    // An admission scan only; XGrammar remains the sole parser and matcher.
    // Initially exclude attributes, lookahead, macros, regex and range expansion.
    static const std::regex attributes(R"((^|\n)[ \t]*[A-Za-z_][A-Za-z0-9_.-]*[ \t]*\[[^\n]*\][ \t]*::=)");
    if (std::regex_search(source, attributes)) throw std::runtime_error("grammar rule attributes are unsupported");
    bool quote = false, cls = false, comment = false, escape = false;
    int depth = 0, alternatives = 0, rules = 0;
    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        if (comment) { if (c == '\n') comment = false; continue; }
        if (escape) { escape = false; continue; }
        if (quote || cls) {
            if (c == '\\') escape = true;
            else if (quote && c == '"') quote = false;
            else if (cls && c == ']') cls = false;
            continue;
        }
        if (c == '#') comment = true;
        else if (c == '"') quote = true;
        else if (c == '[') cls = true;
        else if (c == '(' && ++depth > 32) throw std::runtime_error("grammar nesting exceeds 32 groups");
        else if (c == ')' && --depth < 0) throw std::runtime_error("unmatched grammar closing group");
        else if (c == '|' && ++alternatives > 256) throw std::runtime_error("grammar exceeds 256 alternatives");
        else if (c == ':' && source.compare(i, 3, "::=") == 0 && ++rules > 128)
            throw std::runtime_error("grammar exceeds 128 rules");
        else if (c == '{' || c == '}' || c == '/' || c == '@' || c == '<' || c == '>' || c == '!' || c == '$')
            throw std::runtime_error("unsupported grammar extension (ranges, regex, macros, lookahead or token literals)");
    }
    if (depth != 0 || quote || cls || escape) throw std::runtime_error("unclosed grammar group, literal or character class");
}

struct Compiled {
    std::shared_ptr<const Vocabulary> vocabulary;
    std::string source;
    std::string identity;
    xgrammar::CompiledGrammar grammar;
    size_t bytes;
    bool json_schema;
    Compiled(std::shared_ptr<const Vocabulary> v, std::string s, xgrammar::CompiledGrammar g, bool json)
        : vocabulary(std::move(v)), source(std::move(s)),
          identity(std::string(kBackend) + ":" + vocabulary->identity + (json ? ":json-schema:" : ":") + source_identity(source)),
          grammar(std::move(g)), bytes(grammar.MemorySizeBytes() + source.size()), json_schema(json) {}
};

struct Matcher::Impl {
    enum class Phase { reasoning, choice, tool, after_tool, answer };
    std::shared_ptr<const Compiled> compiled;
    xgrammar::GrammarMatcher matcher;
    detail::WorkBudget budget;
    std::vector<int32_t> history, bitmask;
    size_t output_bytes = 0;
    bool dirty = true, failed = false;
    Scope scope;
    Phase phase;
    const char* channel = "answer";
    bool stopped = false;
    size_t separator_bytes = 0;
    uint32_t reasoning_tokens = 0;
    std::vector<int32_t> answer_mask;
    explicit Impl(std::shared_ptr<const Compiled> c, uint64_t limit, Scope s)
        : compiled(std::move(c)), matcher(compiled->grammar, std::nullopt, false), budget{limit},
          bitmask((compiled->vocabulary->bytes.size() + 31) / 32), scope(s),
          phase(s.thinking ? Phase::reasoning : s.tools ? Phase::choice : Phase::answer) {
        const auto& t = compiled->vocabulary->protocol;
        if (s.reasoning_tokens > kMaxHistoryTokens || (!s.thinking && s.reasoning_tokens))
            throw std::runtime_error("invalid native thinking budget");
        if ((s.thinking && t.think_end < 0) || (s.tools && (t.call_start < 0 || t.call_end < 0)))
            throw std::runtime_error("grammar scope requires the Qwen reasoning/tool special tokens");
    }
    Impl(const Impl& from)
        : compiled(from.compiled), matcher(from.matcher.Fork()), budget(from.budget), history(from.history),
          bitmask(from.bitmask), output_bytes(from.output_bytes), dirty(from.dirty), failed(from.failed),
          scope(from.scope), phase(from.phase), channel(from.channel), stopped(from.stopped),
          separator_bytes(from.separator_bytes), reasoning_tokens(from.reasoning_tokens), answer_mask(from.answer_mask) {}
    void usable() const {
        if (failed) throw std::runtime_error("grammar matcher failed; discard this generation");
    }
};

Matcher::Matcher(std::shared_ptr<const Compiled> c, uint64_t limit, Scope s) {
    if (!c) throw std::runtime_error("missing compiled grammar");
    detail::WorkBudget budget{limit};
    detail::WorkScope scope(budget, 1000ms);
    impl_ = std::make_unique<Impl>(std::move(c), limit, s);
    impl_->budget = budget;
    scope.finish();
}
Matcher::Matcher(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Matcher::~Matcher() = default;
Matcher::Matcher(Matcher&&) noexcept = default;
Matcher& Matcher::operator=(Matcher&&) noexcept = default;

const std::vector<int32_t>& Matcher::mask() {
    auto& p = *impl_;
    p.usable();
    if (terminated()) throw std::runtime_error("grammar matcher is terminal");
    if (!p.dirty) return p.bitmask;
    try {
        detail::WorkScope scope(p.budget, 1000ms);
        const auto& v = *p.compiled->vocabulary;
        auto allow = [&](int32_t id) { p.bitmask[id / 32] |= (int32_t) (1u << (id % 32)); };
        if (p.phase == Impl::Phase::reasoning || p.phase == Impl::Phase::tool) {
            p.bitmask = v.text_mask;
            if (p.phase == Impl::Phase::reasoning && p.scope.reasoning_tokens &&
                p.reasoning_tokens >= p.scope.reasoning_tokens)
                std::fill(p.bitmask.begin(), p.bitmask.end(), 0);
            allow(p.phase == Impl::Phase::reasoning ? v.protocol.think_end : v.protocol.call_end);
        } else {
            int64_t shape[2] = {1, (int64_t) p.bitmask.size()};
            DLTensor tensor{};
            tensor.data = p.bitmask.data(); tensor.device = {kDLCPU, 0};
            tensor.ndim = 2; tensor.dtype = {kDLInt, 32, 1}; tensor.shape = shape;
            p.matcher.FillNextTokenBitmask(&tensor);
            if (p.phase != Impl::Phase::answer) {
                p.answer_mask = p.bitmask;
                // Qwen's envelope separator is at most two LF bytes. An
                // unbounded formatting alternative can starve the answer when
                // its first legal token has a lower score than a newline.
                for (auto id : v.newline_ids)
                    if (v.bytes[id].size() <= 2 - p.separator_bytes) allow(id);
                if (p.scope.tools) allow(v.protocol.call_start);
                if (p.phase == Impl::Phase::after_tool) for (auto id : v.stop_ids) allow(id);
            }
        }
        // Always exclude padding beyond the vocabulary, including from an all-true mask.
        const size_t n = p.compiled->vocabulary->bytes.size();
        if (n % 32) p.bitmask.back() = (int32_t) ((uint32_t) p.bitmask.back() & ((1u << (n % 32)) - 1));
        if (std::all_of(p.bitmask.begin(), p.bitmask.end(), [](int32_t word) { return word == 0; }))
            throw std::runtime_error("grammar has no tokenizer-realizable continuation");
        scope.finish();
        p.dirty = false;
        return p.bitmask;
    } catch (...) { p.failed = true; throw; }
}

bool Matcher::allows(int32_t token) {
    if (token < 0 || (size_t) token >= impl_->compiled->vocabulary->bytes.size()) return false;
    const auto& m = mask();
    return ((uint32_t) m[(size_t) token / 32] & (1u << (token % 32))) != 0;
}

void Matcher::prefix_masks(const int32_t* drafts, int count, PrefixMasks& out) {
    impl_->usable();
    if (count < 0 || count > 7 || (count && !drafts))
        throw std::runtime_error("grammar speculative prefix requires 0..7 drafts");
    out.rows = 1; out.blocked_draft = -1; out.end_draft = false;
    out.bits = mask();
    if (!count) return;
    try {
        auto tentative = fork();
        try {
            detail::WorkScope scope(tentative.impl_->budget, 1000ms);
            for (int i = 0; i < count; ++i) {
                detail::work(); // include cheap/cached proposal steps in the request budget
                if (!tentative.accept(drafts[i])) { out.blocked_draft = i; break; }
                if (tentative.terminated()) { out.end_draft = true; break; }
                const auto& next = tentative.mask();
                out.bits.insert(out.bits.end(), next.begin(), next.end());
                ++out.rows;
            }
            scope.finish();
        } catch (...) {
            impl_->budget = tentative.impl_->budget;
            throw;
        }
        impl_->budget = tentative.impl_->budget;
    } catch (...) { impl_->failed = true; throw; }
}

bool Matcher::accept(int32_t token) {
    auto& p = *impl_;
    p.usable();
    if (!allows(token)) return false;
    try {
        const size_t bytes = p.compiled->vocabulary->bytes[(size_t) token].size();
        if (p.history.size() >= kMaxHistoryTokens || p.output_bytes + bytes > kMaxOutputBytes)
            throw std::runtime_error("grammar resource limit: generation history exceeds its bound");
        detail::WorkScope scope(p.budget, 1000ms);
        const auto before = p.phase;
        const auto& v = *p.compiled->vocabulary;
        const bool stop = std::find(v.stop_ids.begin(), v.stop_ids.end(), token) != v.stop_ids.end();
        if (p.phase == Impl::Phase::reasoning) {
            if (token != v.protocol.think_end) ++p.reasoning_tokens;
            p.channel = token == v.protocol.think_end ? "control" : "reasoning";
            if (token == v.protocol.think_end) p.phase = Impl::Phase::choice;
        } else if (p.phase == Impl::Phase::tool) {
            p.channel = "tool";
            if (token == v.protocol.call_end) p.phase = Impl::Phase::after_tool;
        } else if (p.phase != Impl::Phase::answer && p.scope.tools && token == v.protocol.call_start) {
            p.channel = "tool"; p.phase = Impl::Phase::tool;
        } else if (p.phase == Impl::Phase::after_tool && stop) {
            p.channel = "control"; p.stopped = true;
        } else if (p.phase != Impl::Phase::answer &&
                   !((uint32_t) p.answer_mask[token / 32] & (1u << (token % 32)))) {
            // A newline that the grammar permits belongs to the answer. Only
            // other newline-only tokens are protocol framing, never output.
            p.channel = "control";
            p.separator_bytes += bytes;
        } else {
            if (!p.matcher.AcceptToken(token)) throw std::runtime_error("native grammar rejected an allowed token");
            p.channel = stop ? "control" : "answer"; p.phase = Impl::Phase::answer;
        }
        detail::work();
        scope.finish();
        p.history.push_back(token); p.output_bytes += bytes;
        if (before != p.phase) p.separator_bytes = 0;
        p.dirty = before != p.phase || p.phase == Impl::Phase::answer ||
                  p.phase == Impl::Phase::choice || p.phase == Impl::Phase::after_tool ||
                  (p.phase == Impl::Phase::reasoning && p.scope.reasoning_tokens &&
                   p.reasoning_tokens >= p.scope.reasoning_tokens);
        return true;
    } catch (...) { p.failed = true; throw; }
}

bool Matcher::complete() const {
    impl_->usable();
    return impl_->phase == Impl::Phase::after_tool ||
           (impl_->phase != Impl::Phase::reasoning && impl_->phase != Impl::Phase::tool && impl_->matcher.IsCompleted());
}
bool Matcher::terminated() const { impl_->usable(); return impl_->stopped || impl_->matcher.IsTerminated(); }
const char* Matcher::channel() const { impl_->usable(); return impl_->channel; }
const char* Matcher::phase() const {
    impl_->usable();
    switch (impl_->phase) {
        case Impl::Phase::reasoning: return "reasoning";
        case Impl::Phase::choice: return "choice";
        case Impl::Phase::tool: return "tool";
        case Impl::Phase::after_tool: return "after_tool";
        default: return "answer";
    }
}
bool Matcher::failed() const { return impl_->failed; }
const std::vector<int32_t>& Matcher::tokens() const { return impl_->history; }
const std::string& Matcher::identity() const { return impl_->compiled->identity; }
uint64_t Matcher::work_used() const { return impl_->budget.used; }

Inspection Matcher::inspect(size_t preview_limit) const {
    impl_->usable();
    if (preview_limit > 64) throw std::runtime_error("grammar inspection preview is limited to 64 tokens");
    const auto& p = *impl_;
    Inspection out{identity(), "none:terminal", p.history.size(), p.output_bytes, 0,
                   complete(), terminated(), true, p.budget.used, {}};
    if (out.terminated) return out;
    // Read-only observation: work to derive an uncached mask is charged to a
    // private copy. Even an inspection failure cannot poison/advance the parent.
    auto budget = p.budget;
    detail::WorkScope scope(budget, 1000ms);
    Matcher view(std::make_unique<Impl>(p));
    const auto& bits = view.mask();
    uint64_t hash = 14695981039346656037ull;
    for (int32_t word : bits)
        for (int byte = 0; byte < 4; ++byte)
            hash = (hash ^ (uint8_t) ((uint32_t) word >> (8 * byte))) * 1099511628211ull;
    std::ostringstream fingerprint;
    fingerprint << "mask-v1-fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    out.mask_fingerprint = fingerprint.str();
    const auto& vocab = *p.compiled->vocabulary;
    for (size_t id = 0; id < vocab.bytes.size(); ++id) {
        if (!((uint32_t) bits[id / 32] & (1u << (id % 32)))) continue;
        ++out.legal_tokens;
        if (out.preview.size() < preview_limit) {
            const auto& bytes = vocab.bytes[id];
            out.preview.push_back({(int32_t) id, bytes.substr(0, 32),
                std::find(vocab.stop_ids.begin(), vocab.stop_ids.end(), (int32_t) id) != vocab.stop_ids.end(),
                bytes.size() > 32});
        }
    }
    out.preview_complete = out.preview.size() == out.legal_tokens;
    scope.finish();
    return out;
}

Matcher Matcher::fork() const {
    impl_->usable();
    detail::WorkScope scope(impl_->budget, 1000ms);
    auto copy = std::make_unique<Impl>(*impl_);
    scope.finish();
    return Matcher(std::move(copy));
}

Checkpoint Matcher::checkpoint() const {
    impl_->usable();
    return {impl_->compiled, impl_->history, impl_->scope};
}

void Matcher::restore(const Checkpoint& checkpoint) {
    impl_->usable();
    if (checkpoint.compiled != impl_->compiled || checkpoint.scope.thinking != impl_->scope.thinking ||
        checkpoint.scope.tools != impl_->scope.tools ||
        checkpoint.scope.reasoning_tokens != impl_->scope.reasoning_tokens)
        throw std::runtime_error("grammar/tokenizer/backend identity mismatch in checkpoint");
    if (checkpoint.tokens.size() > kMaxHistoryTokens) throw std::runtime_error("grammar checkpoint is too large");
    Matcher fresh(impl_->compiled, impl_->budget.remaining, impl_->scope);
    for (int32_t t : checkpoint.tokens)
        if (!fresh.accept(t)) throw std::runtime_error("invalid grammar checkpoint token");
    fresh.impl_->budget.used += impl_->budget.used;
    *this = std::move(fresh);
}

struct Compiler::Impl {
    std::shared_ptr<const Vocabulary> vocabulary;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
    std::list<std::shared_ptr<const Compiled>> cache;
    size_t entries_limit, bytes_limit, bytes = 0;
    Impl(std::shared_ptr<const Vocabulary> v, size_t entries, size_t maximum)
        : vocabulary(std::move(v)), tokenizer(vocabulary->bytes, xgrammar::VocabType::RAW,
                                              (int) vocabulary->bytes.size(), vocabulary->stop_ids, false),
          compiler(tokenizer, 1, false), entries_limit(entries), bytes_limit(maximum) {}
};

Compiler::Compiler(std::shared_ptr<const Vocabulary> v, size_t entries, size_t bytes) {
    if (!v || entries > 64 || bytes > 256 * 1024 * 1024)
        throw std::runtime_error("invalid grammar compiler configuration");
    xgrammar::SetMaxRecursionDepth(128);
    impl_ = std::make_unique<Impl>(std::move(v), entries, bytes);
}
Compiler::~Compiler() = default;
size_t Compiler::cache_entries() const { return impl_->cache.size(); }
size_t Compiler::cache_bytes() const { return impl_->bytes; }

std::shared_ptr<const Compiled> Compiler::compile(const std::string& source, uint64_t limit, bool json_schema) {
    if (json_schema) {
        if (source.empty() || source.size() > kMaxSourceBytes || source.find('\0') != std::string::npos || !valid_utf8(source))
            throw std::runtime_error("JSON schema must be 1..8192 UTF-8 bytes without a raw NUL");
    } else validate_source(source);
    auto& p = *impl_;
    for (auto i = p.cache.begin(); i != p.cache.end(); ++i)
        if ((*i)->source == source && (*i)->json_schema == json_schema) {
            const auto compiled = *i;
            p.cache.splice(p.cache.begin(), p.cache, i);
            return compiled;
        }
    detail::WorkBudget budget{limit};
    detail::WorkScope scope(budget, 2500ms);
    // Schema conversion feeds the same compiler/matcher and native token masks.
    // strict_mode=false preserves the schema's own additionalProperties/items
    // semantics; it is unrelated to the HTTP strict flag. Keep property order.
    auto grammar = json_schema ? p.compiler.CompileJSONSchema(source, true, std::nullopt, std::nullopt, false, 2, false)
                               : p.compiler.CompileGrammar(source, "root");
    auto result = std::make_shared<Compiled>(p.vocabulary, source, std::move(grammar), json_schema);
    if (result->bytes > 16 * 1024 * 1024)
        throw std::runtime_error("grammar resource limit: compiled grammar exceeds 16 MiB");
    scope.finish();
    Matcher initial(result);
    initial.mask(); // No unproductive/unrealizable grammar enters the cache or falls back to plain decoding.
    if (p.entries_limit && result->bytes <= p.bytes_limit) {
        while (p.cache.size() >= p.entries_limit || p.bytes + result->bytes > p.bytes_limit) {
            p.bytes -= p.cache.back()->bytes; p.cache.pop_back();
        }
        p.cache.push_front(result); p.bytes += result->bytes;
    }
    return result;
}

} // namespace strata::grammar
