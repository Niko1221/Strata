// include/strata/artifact/safetensors_reader.hpp - Safetensors shard header parser and exact pread helper.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace strata::artifact {

struct TensorSlice {
    int shard_idx = -1;
    uint64_t file_offset = 0;
    uint64_t byte_len = 0;
};

inline void pread_exact(int fd, void* buf, size_t count, uint64_t offset) {
    uint8_t* dst = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t n = ::pread(fd, dst + done, count - done, static_cast<off_t>(offset + done));
        if (n <= 0) {
            std::fprintf(stderr, "pread failed at offset %llu (wanted %zu, got %zd)\n",
                         (unsigned long long) (offset + done), count - done, n);
            std::exit(1);
        }
        done += static_cast<size_t>(n);
    }
}

inline void parse_safetensors_header(int fd, int shard_idx,
                                     std::unordered_map<std::string, TensorSlice>& index_out) {
    uint64_t hlen = 0;
    pread_exact(fd, &hlen, 8, 0);
    std::string hdr(hlen, '\0');
    pread_exact(fd, hdr.data(), hlen, 8);
    const uint64_t data_base = 8 + hlen;

    size_t pos = 0;
    while (pos < hdr.size()) {
        size_t k1 = hdr.find('"', pos);
        if (k1 == std::string::npos) break;
        size_t k2 = hdr.find('"', k1 + 1);
        if (k2 == std::string::npos) break;
        std::string key = hdr.substr(k1 + 1, k2 - k1 - 1);
        size_t colon = hdr.find(':', k2 + 1);
        if (colon == std::string::npos) break;
        size_t brace_open = hdr.find('{', colon);
        if (brace_open == std::string::npos) break;

        int depth = 1;
        size_t cur = brace_open + 1;
        while (cur < hdr.size() && depth > 0) {
            if (hdr[cur] == '{') depth++;
            else if (hdr[cur] == '}') depth--;
            cur++;
        }
        if (key != "__metadata__") {
            std::string body = hdr.substr(brace_open, cur - brace_open);
            size_t doff = body.find("\"data_offsets\"");
            if (doff != std::string::npos) {
                size_t b1 = body.find('[', doff);
                size_t b2 = body.find(']', b1);
                unsigned long long start = 0, end = 0;
                if (std::sscanf(body.substr(b1 + 1, b2 - b1 - 1).c_str(), "%llu,%llu", &start, &end) == 2) {
                    index_out[key] = TensorSlice{shard_idx, data_base + start, end - start};
                }
            }
        }
        pos = cur;
    }
}

}  // namespace strata::artifact
