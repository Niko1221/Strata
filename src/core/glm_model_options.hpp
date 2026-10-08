#pragma once
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core::glm_model_options {
inline bool integer_list(const std::string& text, std::vector<int>& out) {
    out.clear();
    const char* p = text.c_str();
    for (;;) {
        while (*p == ' ') ++p;
        if (*p < '0' || *p > '9') return false;
        errno = 0;
        char* end = nullptr;
        const long n = std::strtol(p, &end, 10);
        if (errno == ERANGE || n > INT_MAX) return false;
        p = end;
        while (*p == ' ') ++p;
        out.push_back((int) n);
        if (*p == '\0') return true;
        if (*p++ != ',') return false;
    }
}
inline bool devices_ok(const std::vector<int>& devs, int visible, int layers, int max_parts) {
    if (devs.size() > (size_t) max_parts || devs.size() > (size_t) layers) return false;
    for (size_t i = 0; i < devs.size(); ++i) {
        if (devs[i] < 0 || devs[i] >= visible) return false;
        for (size_t j = 0; j < i; ++j) if (devs[i] == devs[j]) return false;
    }
    return true;
}
}
