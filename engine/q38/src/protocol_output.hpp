#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>

// Wire format consumed by StrataEngine._parse_done. The seven integer zeros
// are drafts accepted/offered, reuse, hits/lookups, RAM/file blobs; file MB is
// the following float. Keep prompt_read and offloaded in positions 14 and 15.
inline std::string q38_format_done(int produced, size_t prompt_tokens,
                                   double prompt_ms, double decode_ms,
                                   const char *finish, size_t prompt_read) {
    char line[256];
    int n=std::snprintf(line,sizeof(line),
        "DONE %d %zu %.1f %.1f %s 0 0 0 0 0 0 0 0.0 %zu 0\n",
        produced,prompt_tokens,prompt_ms,decode_ms,finish,prompt_read);
    if(n<0 || size_t(n)>=sizeof(line)) throw std::runtime_error("DONE format overflow");
    return std::string(line,size_t(n));
}
