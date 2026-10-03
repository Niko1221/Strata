// include/strata/program/window_logits_dump.hpp - the per-verify-window logits probe.
//
// WHY THIS EXISTS, and why it is not `--dump-logits`: a native pack never runs the per-token loop that
// `--dump-logits` writes from (`if (native_pack) { spec_pos = pos; break; }` leaves it before the dump site),
// so that flag writes a header promising one row per position and then writes NOTHING. A header that
// describes a different file from the one written is worse than no diagnostic: a reader that trusts it
// reports "0 differing positions" from a file with no positions in it.
//
// So this format streams RECORDS and never states a count it has not written. A reader walks records until
// the file ends; a truncated run is short, not wrong. Each record is one verify window:
//
//     magic  "STRWLLOG"   8 bytes, once, at the head of the file
//     u32    version      1
//     u32    n_vocab      floats per row
//     then, per window, repeated:
//       i64  pos          the position the window starts at
//       i32  T            the window length (drafts verified with the real token)
//       i32  a            how many drafts were accepted
//       i32  n_rows       rows carried by this record (1: the row that chose the next token)
//       f32  logits[n_rows * n_vocab]
//
// All little-endian, matching the engine's other dumps. Default off: the caller reads the environment once
// and opens nothing until it is set.
#ifndef STRATA_PROGRAM_WINDOW_LOGITS_DUMP_HPP
#define STRATA_PROGRAM_WINDOW_LOGITS_DUMP_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace strata {
namespace program {
namespace window_logits {

inline constexpr char kMagic[8] = {'S', 'T', 'R', 'W', 'L', 'L', 'O', 'G'};
inline constexpr std::uint32_t kVersion = 1;

/// Open the file and write the header. Returns null on any failure, and says why on stderr.
inline std::FILE* open_file(const char* path, std::uint32_t n_vocab) {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "strata: window-logits dump: cannot write %s\n", path);
        return nullptr;
    }
    const bool ok = std::fwrite(kMagic, 1, sizeof kMagic, f) == sizeof kMagic &&
                    std::fwrite(&kVersion, sizeof kVersion, 1, f) == 1 &&
                    std::fwrite(&n_vocab, sizeof n_vocab, 1, f) == 1;
    if (!ok) {
        std::fprintf(stderr, "strata: window-logits dump: cannot write the header of %s\n", path);
        std::fclose(f);
        return nullptr;
    }
    return f;
}

/// One window. `logits` holds `n_rows * n_vocab` floats. False means the record did not land, and the file
/// is left as it was: a short file is honest, a padded one would not be.
inline bool write_record(std::FILE* f, std::int64_t pos, std::int32_t T, std::int32_t a, std::int32_t n_rows,
                         std::uint32_t n_vocab, const float* logits) {
    if (f == nullptr || logits == nullptr || n_rows <= 0) return false;
    const bool head = std::fwrite(&pos, sizeof pos, 1, f) == 1 && std::fwrite(&T, sizeof T, 1, f) == 1 &&
                      std::fwrite(&a, sizeof a, 1, f) == 1 && std::fwrite(&n_rows, sizeof n_rows, 1, f) == 1;
    if (!head) return false;
    const std::size_t want = (std::size_t) n_rows * n_vocab;
    return std::fwrite(logits, sizeof(float), want, f) == want;
}

/// What a reader gets back for one record.
struct Record {
    std::int64_t pos = 0;
    std::int32_t T = 0;
    std::int32_t a = 0;
    std::int32_t n_rows = 0;
    std::vector<float> logits;
};

/// Walk a dump: the header, then every record until the file ends. `vocab` receives n_vocab.
/// False means the header is not one of ours - the file is then not a window-logits dump at all.
inline bool read_all(const char* path, std::uint32_t& vocab, std::vector<Record>& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    char magic[8];
    std::uint32_t version = 0;
    const bool head = std::fread(magic, 1, sizeof magic, f) == sizeof magic &&
                      std::fread(&version, sizeof version, 1, f) == 1 &&
                      std::fread(&vocab, sizeof vocab, 1, f) == 1;
    if (!head || std::memcmp(magic, kMagic, sizeof magic) != 0 || version != kVersion || vocab == 0) {
        std::fclose(f);
        return false;
    }
    for (;;) {
        Record r;
        if (std::fread(&r.pos, sizeof r.pos, 1, f) != 1) break;
        if (std::fread(&r.T, sizeof r.T, 1, f) != 1) break;
        if (std::fread(&r.a, sizeof r.a, 1, f) != 1) break;
        if (std::fread(&r.n_rows, sizeof r.n_rows, 1, f) != 1) break;
        if (r.n_rows <= 0) break;
        r.logits.resize((std::size_t) r.n_rows * vocab);
        const std::size_t want = r.logits.size();
        if (std::fread(r.logits.data(), sizeof(float), want, f) != want) break;   // a truncated tail
        out.push_back(std::move(r));
    }
    std::fclose(f);
    return true;
}

}  // namespace window_logits
}  // namespace program
}  // namespace strata

#endif  // STRATA_PROGRAM_WINDOW_LOGITS_DUMP_HPP
