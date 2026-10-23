// GrammarRegistry unit tests (no GPU): registration, cache hits, parse errors, and the LRU cap
// with slot-pinned entries surviving eviction.
#include "strata/grammar/grammar_registry.hpp"

#include <cstdio>
#include <memory>
#include <string>

using strata::grammar::GrammarRegistry;
using strata::grammar::GrammarVocab;

static std::shared_ptr<const GrammarVocab> make_vocab() {
    auto v = std::make_shared<GrammarVocab>();
    v->pieces = {"a", "b", "<|eog|>"};
    v->control = {0, 0, 1};
    v->eog = {0, 0, 1};
    return v;
}

static int failures = 0;
#define CHECK(cond)                                                                                                  \
    do {                                                                                                             \
        if (!(cond)) {                                                                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                              \
            ++failures;                                                                                              \
        }                                                                                                            \
    } while (0)

int main(void) {
    { // a valid grammar registers, a find hits, and the entry carries the vocab and mask width
        GrammarRegistry reg(make_vocab());
        std::string err;
        CHECK(reg.register_grammar("0000000000000001", "root ::= \"ab\"\n", err));
        CHECK(err.empty());
        auto e = reg.find("0000000000000001");
        CHECK(e != nullptr);
        CHECK(e && e->proto != nullptr);
        CHECK(e && e->vocab && e->vocab->pieces.size() == 3);
        CHECK(e && e->mask_words == 1);
        CHECK(reg.size() == 1);
    }
    { // re-registering the same key returns the same entry, with no reparse
        GrammarRegistry reg(make_vocab());
        std::string err;
        CHECK(reg.register_grammar("0000000000000002", "root ::= \"ab\"\n", err));
        auto first = reg.find("0000000000000002");
        CHECK(reg.register_grammar("0000000000000002", "root ::= \"ab\"\n", err));
        auto again = reg.find("0000000000000002");
        CHECK(first.get() == again.get());
        CHECK(reg.size() == 1);
    }
    { // a broken grammar fails with a message, and leaves the registry untouched
        GrammarRegistry reg(make_vocab());
        std::string err;
        CHECK(!reg.register_grammar("0000000000000003", "root ::= notdefined\n", err));
        CHECK(!err.empty());
        CHECK(reg.find("0000000000000003") == nullptr);
        CHECK(reg.size() == 0);
    }
    { // 65 registrations evict the oldest: the registry stays at 64 and the first key is gone
        GrammarRegistry reg(make_vocab(), 64);
        std::string err;
        char sha[32];
        for (int i = 0; i < 65; ++i) {
            std::snprintf(sha, sizeof(sha), "00000000000000%02x", i);
            CHECK(reg.register_grammar(sha, "root ::= \"ab\"\n", err));
        }
        CHECK(reg.size() == 64);
        std::snprintf(sha, sizeof(sha), "00000000000000%02x", 0);
        CHECK(reg.find(sha) == nullptr);
        std::snprintf(sha, sizeof(sha), "00000000000000%02x", 64);
        CHECK(reg.find(sha) != nullptr);
    }
    { // a slot holding a shared_ptr pins its entry: overflow registers past the cap, the pin lives
        GrammarRegistry reg(make_vocab(), 64);
        std::string err;
        CHECK(reg.register_grammar("00000000000000aa", "root ::= \"ab\"\n", err));
        auto pinned = reg.find("00000000000000aa");
        CHECK(pinned != nullptr);
        char sha[32];
        for (int i = 0; i < 70; ++i) {
            std::snprintf(sha, sizeof(sha), "0000000000000b%02x", i);
            CHECK(reg.register_grammar(sha, "root ::= \"ab\"\n", err));
        }
        CHECK(reg.find("00000000000000aa") != nullptr);
    }
    if (failures) {
        std::printf("%d registry test(s) FAILED\n", failures);
        return 1;
    }
    std::printf("All grammar registry tests passed.\n");
    return 0;
}
