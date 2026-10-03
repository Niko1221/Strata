// Synthetic proposals through the production native matcher/commit algorithms.
#include "strata/core/grammar.hpp"
#include "strata/program/speculative_window.hpp"
#include <iostream>
#include <stdexcept>

using namespace strata::grammar;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

int main() {
    try {
        std::vector<std::string> bytes(260);
        for (int i = 0; i < 256; ++i) bytes[i] = std::string(1, (char) i);
        bytes[257] = "ab"; bytes[258] = "abc";
        auto vocab = Vocabulary::from_bytes(bytes, {256});
        Compiler compiler(vocab);
        auto compiled = compiler.compile("root ::= \"abcdefg\"");
        int checks = 0;
        for (int bad = 0; bad < 7; ++bad) {
            Matcher matcher(compiled);
            int32_t drafts[] = {'a','b','c','d','e','f','g'};
            drafts[bad] = 'z';
            PrefixMasks rows;
            const auto before = matcher.work_used();
            matcher.prefix_masks(drafts, 7, rows);
            REQUIRE(rows.rows == bad + 1 && rows.blocked_draft == bad && !rows.end_draft);
            REQUIRE(matcher.tokens().empty() && matcher.work_used() > before);
            Matcher reference(compiled);
            for (int row = 0; row < rows.rows; ++row) {
                const auto& mask = reference.mask();
                REQUIRE(std::equal(mask.begin(), mask.end(), rows.bits.begin() + row * mask.size()));
                if (row < bad) REQUIRE(reference.accept(drafts[row]));
            }
            ++checks;
        }
        // The end proposal is checked by the preceding row, never fed back as
        // a prefix requiring another sample. A legal multi-byte token can end it.
        Matcher end(compiler.compile("root ::= \"abc\""));
        const int32_t ending[] = {258, 256, 'z'};
        PrefixMasks masks;
        end.prefix_masks(ending, 3, masks);
        REQUIRE(masks.rows == 2 && masks.end_draft && masks.blocked_draft == -1);
        REQUIRE(end.tokens().empty());
        REQUIRE(end.accept(258) && end.complete() && !end.terminated());
        end.prefix_masks(ending + 1, 2, masks);
        REQUIRE(masks.rows == 1 && masks.end_draft && end.tokens().size() == 1);
        ++checks;
        // Split UTF-8: the illegal continuation's descendants are not sampled.
        Matcher unicode(compiler.compile("root ::= \"\xc3\xa9!\""));
        const int32_t utf8[] = {0xc3, 'x', '!'};
        unicode.prefix_masks(utf8, 3, masks);
        REQUIRE(masks.rows == 2 && masks.blocked_draft == 1);
        REQUIRE(unicode.tokens().empty() && unicode.accept(0xc3) && unicode.accept(0xa9));
        ++checks;
        // Equality, each rejection position, budget and EOS share one boundary.
        const std::vector<int64_t> stops{256};
        for (int rows = 1; rows <= 8; ++rows) for (int mismatch = 0; mismatch < rows; ++mismatch)
            for (int budget = 1; budget <= 9; ++budget) for (int eos = -1; eos < rows; ++eos) {
                std::vector<int32_t> in(rows, 'a'), selected(rows, 'a');
                if (mismatch + 1 < rows) in[mismatch + 1] = 'z';
                if (eos >= 0) selected[eos] = 256;
                const int want = std::min({rows, mismatch + 1, budget, eos >= 0 ? eos + 1 : rows});
                const auto kept = strata::program::retained_window(in.data(), selected.data(), rows, budget, stops);
                REQUIRE(kept.count == want && kept.eos == (eos >= 0 && eos + 1 == want));
                for (int i = 1; i < kept.count; ++i) REQUIRE(in[i] == selected[i - 1]);
                for (int i = 0; i + 1 < kept.count; ++i) REQUIRE(selected[i] != 256);
                ++checks;
            }
        // Ordinary CLI decoding preserves its explicit stop-EOS setting.
        const int32_t control_inputs[]{'x',256,'a'}, control_outputs[]{256,'a','b'};
        REQUIRE(strata::program::retained_window(control_inputs, control_outputs, 3, 9, stops).count == 1);
        REQUIRE(strata::program::retained_window(control_inputs, control_outputs, 3, 9, stops, false).count == 3);
        // Discarded proposal work cannot reset the request's cooperative budget.
        Matcher bounded(compiled, 1000);
        bool exhausted = false;
        const int32_t legal[] = {'a','b','c','d','e','f','g'};
        for (int i = 0; i < 10000 && !exhausted; ++i) {
            try { bounded.prefix_masks(legal, 7, masks); }
            catch (const std::exception& e) {
                REQUIRE(std::string(e.what()).find("resource limit") != std::string::npos);
                exhausted = true;
            }
        }
        REQUIRE(exhausted && bounded.failed() && bounded.tokens().empty());
        std::cout << "PASS: " << checks << " native prefix/commit cases; synthetic proposals, no model execution\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
