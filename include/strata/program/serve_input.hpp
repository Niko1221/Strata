// Existing stdin command stream plus one bounded, inert grammar payload frame.
#pragma once
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace strata::program {
struct ServeRequest {
    std::string line, grammar, error;
    bool fatal = false;
    bool thinking = false, tools = false;
    bool json_schema = false;
    uint32_t reasoning_tokens = 0;
};

// GENG1 <UTF-8 byte count>\n<exact grammar bytes>\nGEN <arguments>\n
// GENG2 <byte count> <thinking:0|1> <tools:0|1> adds final-answer scope.
// GENG3 <byte count> <json_schema:0|1> <thinking:0|1> <tools:0|1> <reasoning tokens>
// Both versions permit CHECKG as the final command (compile only).
// A malformed/truncated frame terminates this pipe; its body is never reparsed
// as commands. The caller owns the existing queue and STOP/cancellation flag.
class ServeInput {
public:
    explicit ServeInput(std::function<std::ptrdiff_t(char*, size_t)> read) : read_(std::move(read)) {}
    std::optional<ServeRequest> next() {
        if (finished_) return std::nullopt;
        ServeRequest request;
        try {
            bool terminated = false;
            if (!line(request.line, terminated)) { finished_ = true; return std::nullopt; }
            if (request.line.rfind("GENG", 0) != 0) return request;
            const bool scoped = request.line.rfind("GENG2 ", 0) == 0;
            const bool json_frame = request.line.rfind("GENG3 ", 0) == 0;
            if (!terminated || (!scoped && !json_frame && request.line.rfind("GENG1 ", 0) != 0))
                throw std::runtime_error("unsupported or truncated grammar frame header");
            size_t count = 0;
            const char* start = request.line.data() + 6;
            const char* end = request.line.data() + request.line.size();
            if (json_frame) {
                std::istringstream header(request.line.substr(6));
                int json = -1, thinking = -1, tools = -1;
                int64_t budget = -1;
                std::string extra;
                if (!(header >> count >> json >> thinking >> tools >> budget) || header >> extra ||
                    json < 0 || json > 1 || thinking < 0 || thinking > 1 || tools < 0 || tools > 1 ||
                    budget < 0 || budget > 8192 || (!thinking && budget))
                    throw std::runtime_error("invalid GENG3 flags or reasoning budget");
                request.json_schema = json; request.thinking = thinking; request.tools = tools;
                request.reasoning_tokens = (uint32_t) budget;
            } else if (scoped) {
                if (end - start < 5 || end[-4] != ' ' || end[-2] != ' ' ||
                    (end[-3] != '0' && end[-3] != '1') || (end[-1] != '0' && end[-1] != '1'))
                    throw std::runtime_error("grammar scope requires thinking/tools boolean flags");
                request.thinking = end[-3] == '1'; request.tools = end[-1] == '1';
                if (!request.thinking && !request.tools) throw std::runtime_error("empty grammar scope");
                end -= 4;
            }
            const auto number = json_frame ? std::from_chars_result{end, std::errc{}} : std::from_chars(start, end, count);
            if (number.ec != std::errc{} || number.ptr != end || count < 1 || count > 8192)
                throw std::runtime_error("grammar frame length must be 1..8192 bytes");
            request.grammar = bytes(count);
            if (bytes(1) != "\n") throw std::runtime_error("grammar frame delimiter is missing");
            if (!line(request.line, terminated) || !terminated ||
                (request.line != "CHECKG" && request.line.rfind("GEN ", 0) != 0) ||
                request.line.find('\0') != std::string::npos)
                throw std::runtime_error("grammar frame needs one complete GEN or CHECKG command");
            return request;
        } catch (const std::exception& error) {
            finished_ = request.fatal = true;
            request.error = error.what();
            return request;
        }
    }
private:
    std::function<std::ptrdiff_t(char*, size_t)> read_;
    std::string buffer_;
    bool finished_ = false;
    bool more() {
        char chunk[4096];
        const auto n = read_(chunk, sizeof chunk);
        if (n < 0) throw std::runtime_error("native input read failed");
        if (n == 0) return false;
        buffer_.append(chunk, (size_t) n);
        return true;
    }
    bool line(std::string& out, bool& terminated) {
        for (;;) {
            const size_t nl = buffer_.find('\n');
            if (nl != std::string::npos) {
                if (nl > 4 * 1024 * 1024) throw std::runtime_error("native input line exceeds 4 MiB");
                out.assign(buffer_, 0, nl); buffer_.erase(0, nl + 1);
                if (!out.empty() && out.back() == '\r') out.pop_back();
                terminated = true;
                return true;
            }
            if (buffer_.size() > 4 * 1024 * 1024) throw std::runtime_error("native input line exceeds 4 MiB");
            if (!more()) { out.swap(buffer_); terminated = false; return !out.empty(); }
        }
    }
    std::string bytes(size_t count) {
        while (buffer_.size() < count)
            if (!more()) throw std::runtime_error("truncated grammar frame payload");
        std::string out = buffer_.substr(0, count);
        buffer_.erase(0, count);
        return out;
    }
};

inline std::string protocol_error(const std::string& message) {
    std::string out = message.substr(0, 512);
    // Source-bearing diagnostics are UTF-8 too. A byte limit must not leave a
    // partial character that kills the Python stdout decoder instead of ERR.
    if (message.size() > out.size() && !out.empty()) {
        size_t start = out.size() - 1;
        while (start && ((unsigned char) out[start] & 0xc0) == 0x80) --start;
        const auto lead = (unsigned char) out[start];
        const size_t size = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
        if (start + size > out.size()) out.resize(start);
    }
    for (char& ch : out) if ((unsigned char) ch < 32 || ch == 127) ch = ' ';
    return out;
}
} // namespace strata::program
