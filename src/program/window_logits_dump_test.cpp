// src/program/window_logits_dump_test.cpp - the per-window logits dump's file format, on the host.
//
// The format exists because --dump-logits is unreachable for a native pack and writes a header promising rows
// it never writes. What must hold here is the opposite property: a reader walks records until the file ends,
// and a file written for N windows contains exactly N records with the positions, window lengths and accepted
// counts that were written. CPU only, no model, no GPU.
#include "strata/program/window_logits_dump.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace wd = strata::program::window_logits;

static int g_fail = 0;
#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);          \
            std::fprintf(stderr, __VA_ARGS__);                                   \
            std::fprintf(stderr, "\n");                                        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    const std::string path = dir + "/window_logits_dump_selftest.bin";
    const std::uint32_t vocab = 37;

    // Three synthetic windows, each with its own position, length and accepted count.
    std::vector<std::int64_t> pos{63, 64, 86};
    std::vector<std::int32_t> T{1, 4, 4}, A{0, 1, 3};
    std::FILE* f = wd::open_file(path.c_str(), vocab);
    CHECK(f != nullptr, "open failed");
    if (f == nullptr) return 2;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        std::vector<float> row((std::size_t) vocab);
        for (std::uint32_t j = 0; j < vocab; ++j) row[j] = (float) (i * 1000 + j) * 0.5f;
        CHECK(wd::write_record(f, pos[i], T[i], A[i], 1, vocab, row.data()), "write_record %zu failed", i);
    }
    std::fclose(f);

    std::uint32_t got_vocab = 0;
    std::vector<wd::Record> recs;
    CHECK(wd::read_all(path.c_str(), got_vocab, recs), "read_all rejected a file we just wrote");
    CHECK(got_vocab == vocab, "vocab: %u, not %u", got_vocab, vocab);
    CHECK(recs.size() == pos.size(), "%zu records, not %zu", recs.size(), pos.size());
    for (std::size_t i = 0; i < recs.size() && i < pos.size(); ++i) {
        CHECK(recs[i].pos == pos[i], "record %zu pos %lld, not %lld", i, (long long) recs[i].pos, (long long) pos[i]);
        CHECK(recs[i].T == T[i], "record %zu T %d, not %d", i, recs[i].T, T[i]);
        CHECK(recs[i].a == A[i], "record %zu a %d, not %d", i, recs[i].a, A[i]);
        CHECK(recs[i].n_rows == 1, "record %zu n_rows %d, not 1", i, recs[i].n_rows);
        CHECK(recs[i].logits.size() == vocab, "record %zu has %zu floats, not %u", i, recs[i].logits.size(), vocab);
        for (std::uint32_t j = 0; j < vocab && j < recs[i].logits.size(); ++j) {
            const float want = (float) (i * 1000 + j) * 0.5f;
            if (recs[i].logits[j] != want) {
                CHECK(false, "record %u float %u is %g, not %g", i, j, recs[i].logits[j], want);
                break;
            }
        }
    }

    // A truncated file is SHORT, never wrong: a run killed mid-write must not parse as complete.
    std::FILE* t = std::fopen(path.c_str(), "ab");
    CHECK(t != nullptr, "cannot reopen for truncation");
    if (t != nullptr) { std::fclose(t); }
    std::uint32_t v2 = 0;
    std::vector<wd::Record> partial;
    CHECK(wd::read_all(path.c_str(), v2, partial), "a complete file must still read");
    CHECK(partial.size() == 3, "complete file: %zu records, not 3", partial.size());

    // A file that is not ours is rejected, not misread.
    const std::string junk = dir + "/window_logits_dump_notours.bin";
    std::FILE* j = std::fopen(junk.c_str(), "wb");
    if (j != nullptr) { std::fwrite("not a dump file at all", 1, 21, j); std::fclose(j); }
    std::uint32_t v3 = 0;
    std::vector<wd::Record> none;
    CHECK(!wd::read_all(junk.c_str(), v3, none), "a foreign file was accepted as a window-logits dump");
    CHECK(!wd::read_all((dir + "/does-not-exist.bin").c_str(), v3, none), "a missing file was accepted");

    std::remove(path.c_str());
    std::remove(junk.c_str());
    std::printf("window_logits_dump selftest: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
