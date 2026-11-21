// hy4_convcache.h - a conversation cache for `strata-hy4 --serve`: parks the resident sequence on the SSD when another prompt takes the card, brings it back
// when its conversation returns, and rewinds the resident sequence to a shorter common prefix.  The idea of Strata's conversation cache disk tier (PR #5835),
// written for the Hy4-Flash runner's state.
//
// What a sequence's state is (hy4_graph.hip.h Runner): per sparse layer (12) the MLA rows kcache [pos][576 halves], the IndexPool rows ivg [pos][256 halves] and the
// pooled keys [pos/4][128 halves]; per sliding-window layer (30) a ring of `ring` rows (slot = pos % ring).  Rows of a position depend only on the tokens up to it,
// so the sparse layers can be cut at any length.  A ring taken at position P holds the positions [P - ring, P); a query at s needs (s - window, s], so a ring taken
// at P serves every start s with P - s <= ring - window (minus a margin for the verify rows DFly2 writes past the end).  Two ring sets are kept per sequence: the
// live one (at the end of the sequence) and a copy taken when the last prompt was read (the prompt end: the point where the next turn of a chat diverges, since a
// chat template usually renders the previous answer differently from how it was generated).
//
// File (one per parked sequence, <dir>/<16 hex>.h4cc, written to .tmp then renamed): a 512-byte header, the token ids, the ring sets, then per sparse layer its
// rows, IndexPool rows and pooled groups.  Pictures never enter the cache (their rows depend on the picture, not only on the ids).
#pragma once
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hy4_graph.hip.h"

namespace hy4 {

struct CcHeader {
    char magic[8];          // "H4CC0001"
    uint32_t hdr_bytes;     // 512
    int32_t n_layers, ring, window, ntok, nring;
    int32_t ring_pos[2];    // the position each ring set was taken at (set 0 = ntok)
    uint64_t swa_mask;      // bit l = layer l is a sliding-window layer
    uint64_t model_bytes;   // the size of the model file (with model[]: the identity of the model)
    char model[256];
    uint8_t pad[512 - 8 - 4 - 4 * 5 - 4 * 2 - 8 - 8 - 256];
};
static_assert(sizeof(CcHeader) == 512, "CcHeader is 512 bytes");

struct CcEntry {
    std::string path;
    CcHeader h{};
    std::vector<int32_t> toks;
    uint64_t bytes = 0;
    double used = 0;
};

class ConvCache {
public:
    bool on = false;
    std::string dir;
    uint64_t budget = 0;
    int min_tokens = 1024;
    // the prompt-end ring copy of the resident sequence
    int ck_pos = -1;

    static constexpr size_t ROW_B = (size_t)576 * 2, IVG_B = (size_t)256 * 2, POOL_B = (size_t)128 * 2;

    bool init(Runner& r, const std::string& d, double gib, int min_tok, const std::string& model_path) {
        dir = d;
        budget = (uint64_t)(gib * 1073741824.0);
        min_tokens = std::max(64, min_tok);
        if (dir.empty() || budget == 0) return false;
        ::mkdir(dir.c_str(), 0755);
        struct stat sb{};
        if (::stat(model_path.c_str(), &sb) != 0) return false;
        model_bytes_ = (uint64_t)sb.st_size;
        std::string base = model_path;
        size_t sl = base.find_last_of("/\\");
        if (sl != std::string::npos) base = base.substr(sl + 1);
        model_ = base.substr(0, 255);
        nl_ = r.c.n_layers;
        if (nl_ > 64) return false;
        ring_ = r.ring;
        window_ = r.c.window;
        swa_mask_ = 0;
        nswa_ = 0;
        for (int l = 0; l < nl_; ++l)
            if (r.c.is_swa[l]) {
                swa_mask_ |= 1ull << l;
                ++nswa_;
            }
        HCK(hipHostMalloc((void**)&stg_, STG, hipHostMallocDefault));
        HCK(hipHostMalloc((void**)&ckbuf_, ring_set_bytes(), hipHostMallocDefault));
        scan();
        on = true;
        fprintf(stderr, "[convcache] %s: %zu parked conversation(s), %.2f of %.1f GiB, ring %d, rewind up to %d tokens, parks sequences of %d+ tokens\n", dir.c_str(), ents_.size(),
                total() / 1073741824.0, budget / 1073741824.0, ring_, rewind_limit(), min_tokens);
        return true;
    }

    // how far a ring set taken at P can be rewound
    int rewind_limit() const { return ring_ - window_ - 32; }

    // the best start a ring set at P gives for a common prefix of n tokens (prompt of T0 tokens: at least one token must be read): -1 when it gives none
    int start_for(int n, int P, int T0) const {
        if (P <= 0) return -1;
        int s = std::min(std::min(n, P), T0 - 1);
        if (s <= 0 || P - s > rewind_limit()) return -1;
        return s;
    }

    // after the prompt is read: keep a host copy of the rings at the prompt end
    void checkpoint(Runner& r, int pos) {
        if (!on) return;
        HCK(hipStreamSynchronize(r.st));
        size_t off = 0;
        for (int l = 0; l < nl_; ++l)
            if (swa(l)) {
                HCK(hipMemcpy(ckbuf_ + off, r.kcache[l], ring_b(), hipMemcpyDefault));
                off += ring_b();
            }
        ck_pos = pos;
    }
    // the resident sequence is rewound to the prompt-end copy: put its rings back
    void restore_checkpoint(Runner& r) {
        HCK(hipStreamSynchronize(r.st));
        size_t off = 0;
        for (int l = 0; l < nl_; ++l)
            if (swa(l)) {
                HCK(hipMemcpy(r.kcache[l], ckbuf_ + off, ring_b(), hipMemcpyDefault));
                off += ring_b();
            }
    }

    struct Pick {
        int ent = -1, start = 0, set = -1;
    };
    // the parked sequence that gives the longest start for ids
    Pick best(const std::vector<int32_t>& ids) const {
        Pick p;
        const int T0 = (int)ids.size();
        for (size_t i = 0; i < ents_.size(); ++i) {
            const CcEntry& e = ents_[i];
            int n = common(e.toks.data(), e.h.ntok, ids);
            for (int k = 0; k < e.h.nring; ++k) {
                int s = start_for(n, e.h.ring_pos[k], T0);
                if (s > p.start) {
                    p.ent = (int)i;
                    p.start = s;
                    p.set = k;
                }
            }
        }
        return p;
    }

    static int common(const int32_t* a, int na, const std::vector<int32_t>& b) {
        int n = std::min(na, (int)b.size()), i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return i;
    }

    // write the resident sequence tok[0, fwd) to the SSD (with the prompt-end ring copy when it belongs to it); drops parked sequences it supersedes
    bool park(Runner& r, const std::vector<int32_t>& tok, int fwd) {
        if (!on || fwd < min_tokens || (int)tok.size() < fwd) return false;
        const double t0 = now_s();
        HCK(hipDeviceSynchronize());
        CcEntry e;
        CcHeader& h = e.h;
        memset(&h, 0, sizeof h);
        memcpy(h.magic, "H4CC0001", 8);
        h.hdr_bytes = sizeof(CcHeader);
        h.n_layers = nl_;
        h.ring = ring_;
        h.window = window_;
        h.ntok = fwd;
        h.swa_mask = swa_mask_;
        h.model_bytes = model_bytes_;
        snprintf(h.model, sizeof h.model, "%s", model_.c_str());
        h.nring = 1;
        h.ring_pos[0] = fwd;
        h.ring_pos[1] = -1;
        if (ck_pos > 0 && ck_pos < fwd && fwd - ck_pos > 0) {
            h.nring = 2;
            h.ring_pos[1] = ck_pos;
        }
        e.toks.assign(tok.begin(), tok.begin() + fwd);
        uint64_t key = 1469598103934665603ull;
        for (int32_t t : e.toks) key = (key ^ (uint32_t)t) * 1099511628211ull;
        char name[64];
        snprintf(name, sizeof name, "%016llx.h4cc", (unsigned long long)key);
        e.path = dir + "/" + name;
        const std::string tmp = e.path + ".tmp";
        int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            fprintf(stderr, "[convcache] cannot write %s: %s\n", tmp.c_str(), strerror(errno));
            return false;
        }
        bool ok = wr(fd, &h, sizeof h) && wr(fd, e.toks.data(), (size_t)fwd * 4);
        // ring set 0: the live rings; ring set 1: the prompt-end copy
        for (int l = 0; ok && l < nl_; ++l)
            if (swa(l)) ok = d2f(fd, r.kcache[l], ring_b());
        if (ok && h.nring == 2) ok = wr(fd, ckbuf_, ring_set_bytes());
        for (int l = 0; ok && l < nl_; ++l) {
            if (swa(l)) continue;
            ok = d2f(fd, r.kcache[l], (size_t)fwd * ROW_B) && d2f(fd, r.ivg[l], (size_t)fwd * IVG_B) && d2f(fd, r.pooled[l], (size_t)(fwd / 4 + 1) * POOL_B);
        }
        ok = ::close(fd) == 0 && ok;
        if (!ok || ::rename(tmp.c_str(), e.path.c_str()) != 0) {
            ::unlink(tmp.c_str());
            fprintf(stderr, "[convcache] writing %s failed\n", e.path.c_str());
            return false;
        }
        e.bytes = file_bytes(h);
        e.used = now_wall();
        // a parked sequence whose ids are a prefix of this one (an earlier turn of the same conversation) is superseded
        for (size_t i = 0; i < ents_.size();) {
            const CcEntry& o = ents_[i];
            if (o.path == e.path || (o.h.ntok <= fwd && common(o.toks.data(), o.h.ntok, e.toks) == o.h.ntok)) {
                if (o.path != e.path) ::unlink(o.path.c_str());
                ents_.erase(ents_.begin() + (long)i);
            } else {
                ++i;
            }
        }
        ents_.push_back(std::move(e));
        evict();
        fprintf(stderr, "[convcache] parked %d tokens (%s, %.2f GB, ring sets at %d%s) in %.0f ms\n", fwd, name, ents_.back().bytes / 1e9, h.ring_pos[0],
                h.nring == 2 ? (" and " + std::to_string(h.ring_pos[1])).c_str() : "", (now_s() - t0) * 1000.0);
        return true;
    }

    // bring a parked sequence back up to p.start tokens
    bool restore(Runner& r, const Pick& p, std::string& err) {
        const CcEntry& e = ents_[(size_t)p.ent];
        const int s = p.start;
        const double t0 = now_s();
        if (!r.kv_ensure(s + 16)) {
            err = "context too small for the parked sequence";
            return false;
        }
        HCK(hipDeviceSynchronize());
        int fd = ::open(e.path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = std::string("cannot open ") + e.path;
            return false;
        }
        const CcHeader& h = e.h;
        uint64_t off = sizeof(CcHeader) + (uint64_t)h.ntok * 4;
        bool ok = true;
        uint64_t rs = off + (uint64_t)p.set * ring_set_bytes();
        for (int l = 0; ok && l < nl_; ++l)
            if (swa(l)) {
                ok = f2d(fd, rs, r.kcache[l], ring_b());
                rs += ring_b();
            }
        off += (uint64_t)h.nring * ring_set_bytes();
        const size_t kvb = (size_t)h.ntok * ROW_B, ivb = (size_t)h.ntok * IVG_B, plb = (size_t)(h.ntok / 4 + 1) * POOL_B;
        for (int l = 0; ok && l < nl_; ++l) {
            if (swa(l)) continue;
            ok = f2d(fd, off, r.kcache[l], (size_t)s * ROW_B) && f2d(fd, off + kvb, r.ivg[l], (size_t)s * IVG_B) &&
                 f2d(fd, off + kvb + ivb, r.pooled[l], (size_t)(s / 4 + 1) * POOL_B);
            off += kvb + ivb + plb;
        }
        ::close(fd);
        if (!ok) {
            err = std::string("reading ") + e.path + " failed";
            return false;
        }
        touch((size_t)p.ent);
        ck_pos = -1;
        fprintf(stderr, "[convcache] restored %d of %d parked tokens (ring set at %d) in %.0f ms\n", s, h.ntok, h.ring_pos[p.set], (now_s() - t0) * 1000.0);
        return true;
    }

    int parked() const { return (int)ents_.size(); }

private:
    static constexpr size_t STG = (size_t)64 << 20;
    uint8_t* stg_ = nullptr;
    uint8_t* ckbuf_ = nullptr;
    std::vector<CcEntry> ents_;
    std::string model_;
    uint64_t model_bytes_ = 0, swa_mask_ = 0;
    int nl_ = 0, ring_ = 0, window_ = 0, nswa_ = 0;

    bool swa(int l) const { return (swa_mask_ >> l) & 1; }
    size_t ring_b() const { return (size_t)ring_ * ROW_B; }
    size_t ring_set_bytes() const { return (size_t)nswa_ * ring_b(); }
    uint64_t file_bytes(const CcHeader& h) const {
        int nsp = h.n_layers - __builtin_popcountll(h.swa_mask);
        return sizeof(CcHeader) + (uint64_t)h.ntok * 4 + (uint64_t)h.nring * ring_set_bytes() +
               (uint64_t)nsp * ((uint64_t)h.ntok * (ROW_B + IVG_B) + (uint64_t)(h.ntok / 4 + 1) * POOL_B);
    }
    uint64_t total() const {
        uint64_t t = 0;
        for (const CcEntry& e : ents_) t += e.bytes;
        return t;
    }
    static double now_wall() {
        struct timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
    }
    void touch(size_t i) {
        ents_[i].used = now_wall();
        struct timespec t[2];
        clock_gettime(CLOCK_REALTIME, &t[0]);
        t[1] = t[0];
        ::utimensat(AT_FDCWD, ents_[i].path.c_str(), t, 0);
    }
    void evict() {
        while (total() > budget && !ents_.empty()) {
            size_t lo = 0;
            for (size_t i = 1; i < ents_.size(); ++i)
                if (ents_[i].used < ents_[lo].used) lo = i;
            fprintf(stderr, "[convcache] dropped %s (%d tokens): over %.1f GiB\n", ents_[lo].path.c_str(), ents_[lo].h.ntok, budget / 1073741824.0);
            ::unlink(ents_[lo].path.c_str());
            ents_.erase(ents_.begin() + (long)lo);
        }
    }
    static bool wr(int fd, const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        while (n > 0) {
            ssize_t w = ::write(fd, b, n);
            if (w <= 0) {
                if (w < 0 && errno == EINTR) continue;
                return false;
            }
            b += w;
            n -= (size_t)w;
        }
        return true;
    }
    static bool rd(int fd, uint64_t off, void* p, size_t n) {
        uint8_t* b = (uint8_t*)p;
        while (n > 0) {
            ssize_t g = ::pread(fd, b, n, (off_t)off);
            if (g <= 0) {
                if (g < 0 && errno == EINTR) continue;
                return false;
            }
            b += g;
            off += (uint64_t)g;
            n -= (size_t)g;
        }
        return true;
    }
    // device (or mapped host) memory -> file, through the pinned staging buffer
    bool d2f(int fd, const void* src, size_t n) {
        const uint8_t* s = (const uint8_t*)src;
        while (n > 0) {
            size_t k = std::min(n, STG);
            HCK(hipMemcpy(stg_, s, k, hipMemcpyDefault));
            if (!wr(fd, stg_, k)) return false;
            s += k;
            n -= k;
        }
        return true;
    }
    bool f2d(int fd, uint64_t off, void* dst, size_t n) {
        uint8_t* d = (uint8_t*)dst;
        while (n > 0) {
            size_t k = std::min(n, STG);
            if (!rd(fd, off, stg_, k)) return false;
            HCK(hipMemcpy(d, stg_, k, hipMemcpyDefault));
            d += k;
            off += k;
            n -= k;
        }
        return true;
    }
    void scan() {
        DIR* dp = ::opendir(dir.c_str());
        if (!dp) return;
        while (struct dirent* de = ::readdir(dp)) {
            std::string n = de->d_name;
            if (n.size() < 6) continue;
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) {
                ::unlink((dir + "/" + n).c_str());
                continue;
            }
            if (n.compare(n.size() - 5, 5, ".h4cc") != 0) continue;
            CcEntry e;
            e.path = dir + "/" + n;
            int fd = ::open(e.path.c_str(), O_RDONLY);
            if (fd < 0) continue;
            bool ok = rd(fd, 0, &e.h, sizeof e.h);
            const CcHeader& h = e.h;
            ok = ok && memcmp(h.magic, "H4CC0001", 8) == 0 && h.hdr_bytes == sizeof(CcHeader) && h.n_layers == nl_ && h.ring == ring_ && h.window == window_ &&
                 h.swa_mask == swa_mask_ && h.model_bytes == model_bytes_ && strncmp(h.model, model_.c_str(), sizeof h.model) == 0 && h.ntok > 0 && h.nring >= 1 && h.nring <= 2;
            struct stat sb{};
            ok = ok && ::fstat(fd, &sb) == 0 && (uint64_t)sb.st_size == file_bytes(h);
            if (ok) {
                e.toks.resize((size_t)h.ntok);
                ok = rd(fd, sizeof(CcHeader), e.toks.data(), (size_t)h.ntok * 4);
                e.bytes = (uint64_t)sb.st_size;
                e.used = sb.st_mtim.tv_sec + sb.st_mtim.tv_nsec * 1e-9;
            }
            ::close(fd);
            if (ok) {
                ents_.push_back(std::move(e));
            } else {
                fprintf(stderr, "[convcache] ignoring %s (another model, ring or a damaged file)\n", e.path.c_str());
            }
        }
        ::closedir(dp);
        evict();
    }
};

}  // namespace hy4
