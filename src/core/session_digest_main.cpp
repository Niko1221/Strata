// strata-session-digest: one base session file's format-independent digest, for host-side orchestration.  No model,
// no GPU.  Prints one stable key=value per line:
//   version=<1|2|3>  tokens=<hex>  kv=<hex>  state=<hex>  state_exact=<0|1>  state_ref=<0|1>  state_rht=<0|1>
// Exit 0: the digest was printed.  1: the file was refused (the reason on stderr).  2: bad usage.
#include "strata/core/conversation_file.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
void usage() {
    std::fprintf(stderr,
                 "usage: strata-session-digest --file <path> --model <fingerprint> --config <fingerprint>\n"
                 "  --model, --config: the model and config fingerprints the file must carry (decimal or 0x hex)\n"
                 "prints: version= tokens= kv= state= state_exact= state_ref= state_rht=\n");
}

bool parse_fingerprint(const char* text, uint64_t& out) {
    if (!text || !*text) return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 0);
    if (errno != 0 || !end || *end) return false;
    out = (uint64_t) value;
    return true;
}
} // namespace

int main(int argc, char** argv) {
    std::string file;
    uint64_t model = 0, config = 0;
    bool have_model = false, have_config = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-session-digest: %s needs a value\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        if (arg == "--file") { file = value(); continue; }
        if (arg == "--model") { have_model = parse_fingerprint(value(), model); continue; }
        if (arg == "--config") { have_config = parse_fingerprint(value(), config); continue; }
        std::fprintf(stderr, "strata-session-digest: unknown argument %s\n", arg.c_str());
        usage();
        return 2;
    }
    if (file.empty() || !have_model || !have_config) {
        std::fprintf(stderr, "strata-session-digest: --file, --model and --config are all required\n");
        usage();
        return 2;
    }
    strata::core::SessionFileDigest digest;
    std::string error;
    strata::core::SessionStatus status;
    if (!strata::core::session_file_digest(file, {model, config}, digest, error, &status)) {
        std::fprintf(stderr, "strata-session-digest: %s\n", error.c_str());
        return 1;
    }
    std::printf("version=%u\ntokens=%016llx\nkv=%016llx\nstate=%016llx\nstate_exact=%d\nstate_ref=%d\n"
                "state_rht=%d\n",
                digest.file_version, (unsigned long long) digest.tokens, (unsigned long long) digest.kv,
                (unsigned long long) digest.state, digest.state_exact ? 1 : 0, digest.state_ref ? 1 : 0,
                digest.state_rht ? 1 : 0);
    return 0;
}
