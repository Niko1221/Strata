// engine/metal/src/hy4flash_gguf/dump.cpp - see dump.h.
#include "dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace strata::hy4_flash {

std::string Dump::open(const std::string& dir, const std::vector<int32_t>& tokens) {
    if (::mkdir(dir.c_str(), 0755) != 0) {
        struct stat st {};
        if (::stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return "cannot create the dump directory " + dir;
    }
    dir_ = dir;
    tokens_ = tokens;
    // An earlier run's rows are kept, as long as it was the same sequence - a manifest that mixes two token
    // lists would have a comparison read nodes from two different inputs.
    FILE* f = std::fopen((dir + "/manifest.txt").c_str(), "r");
    if (f == nullptr) return {};
    std::string want = "# tokens";
    for (int32_t t : tokens) want += " " + std::to_string(t);
    char* line = nullptr;
    size_t cap = 0;
    bool same = false;
    while (::getline(&line, &cap, f) > 0) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.rfind("# tokens", 0) == 0) {
            same = s == want;
            continue;
        }
        if (s.empty() || s[0] == '#') continue;
        const size_t tab = s.rfind('\t');
        if (tab == std::string::npos) continue;
        kept_.push_back({s, s.substr(tab + 1)});
    }
    std::free(line);
    std::fclose(f);
    if (!same) kept_.clear();   // another sequence: start over rather than mix two inputs
    return {};
}

void Dump::add(const std::string& name, const char* type, std::vector<int64_t> ne_one, int pos_axis,
               const void* data, size_t bytes_per_pos, int n_pos) {
    if (dir_.empty()) return;
    auto it = nodes_.find(name);
    if (it == nodes_.end()) {
        Node n;
        n.type = type;
        n.ne_one = std::move(ne_one);
        n.pos_axis = pos_axis;
        it = nodes_.emplace(name, std::move(n)).first;
        order_.push_back(name);
    }
    Node& n = it->second;
    const size_t at = n.data.size();
    n.data.resize(at + bytes_per_pos * (size_t) n_pos);
    std::memcpy(n.data.data() + at, data, bytes_per_pos * (size_t) n_pos);
    n.n_pos += n_pos;
}

void Dump::add_whole(const std::string& name, const char* type, std::vector<int64_t> ne, const void* data,
                     size_t bytes) {
    if (dir_.empty()) return;
    Node n;
    n.type = type;
    n.ne_one = std::move(ne);
    n.whole = true;
    n.data.assign((const uint8_t*) data, (const uint8_t*) data + bytes);
    if (!nodes_.count(name)) order_.push_back(name);
    nodes_[name] = std::move(n);
}

std::string Dump::close() {
    if (dir_.empty()) return {};
    const std::string mpath = dir_ + "/manifest.txt";
    FILE* mf = std::fopen(mpath.c_str(), "w");
    if (mf == nullptr) return "cannot write " + mpath;
    std::fprintf(mf, "# dump_reference v3 (TAB-separated): file\ttype\tne0\tne1\tne2\tne3\tbytes\tname\n");
    std::fprintf(mf, "# engine strata_hy4_flash_metal (plan P1.4): the same node names and shapes as the reference's dump\n");
    std::fprintf(mf, "# tokens");
    for (int32_t t : tokens_) std::fprintf(mf, " %d", t);
    std::fprintf(mf, "\n");
    std::string err;
    // the rows of earlier runs this one does not replace
    for (const Kept& k : kept_)
        if (!nodes_.count(k.name)) std::fprintf(mf, "%s\n", k.line.c_str());
    for (const std::string& name : order_) {
        Node& n = nodes_[name];
        // the full shape: the per-position dims with the position axis inserted where the reference has it
        int64_t ne[4] = {1, 1, 1, 1};
        if (n.whole) {
            for (size_t i = 0; i < n.ne_one.size() && i < 4; ++i) ne[i] = n.ne_one[i];
        } else {
            // ggml order; numpy sees (ne3, ne2, ne1, ne0), and check_block.py's POS_AXIS is a numpy axis, so
            // numpy axis `pos_axis` is ggml index 3 - pos_axis.
            const int gpos = 3 - n.pos_axis;
            int g = 0;
            for (int i = 0; i < 4; ++i) {
                if (i == gpos) {
                    ne[i] = n.n_pos;
                    continue;
                }
                ne[i] = g < (int) n.ne_one.size() ? n.ne_one[(size_t) g] : 1;
                ++g;
            }
        }
        char file[320];
        std::snprintf(file, sizeof file, "%s.bin", name.c_str());
        const std::string path = dir_ + "/" + file;
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            err = "cannot write " + path;
            break;
        }
        const bool ok = n.data.empty() || std::fwrite(n.data.data(), 1, n.data.size(), f) == n.data.size();
        std::fclose(f);
        if (!ok) {
            err = "short write to " + path;
            break;
        }
        std::fprintf(mf, "%s\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%s\n", file, n.type.c_str(), (long long) ne[0],
                     (long long) ne[1], (long long) ne[2], (long long) ne[3], n.data.size(), name.c_str());
        n.data.clear();
        n.data.shrink_to_fit();
    }
    std::fclose(mf);
    return err;
}

}  // namespace strata::hy4_flash
