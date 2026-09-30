// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

namespace {

/// LOCAL PORT (Z620): raw CPUID leaf 1, shared by the AVX2 and AVX1 probes below.
/// Kept as one helper because both probes need the same OSXSAVE/XCR0 dance, and duplicating it is how
/// the two end up disagreeing about what "this CPU has AVX" means.
struct Leaf1 {
    bool osxsave = false;
    unsigned ecx = 0;
    unsigned long long xcr0 = 0;
};

Leaf1 read_leaf1() {
    Leaf1 s;
    unsigned r[4] = {0, 0, 0, 0};
    auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, (int) leaf, (int) sub);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
    };
    unsigned maxleaf[4] = {0, 0, 0, 0};
    cpuid(0, 0);
    maxleaf[0] = r[0];
    if (maxleaf[0] < 1) return s;
    cpuid(1, 0);
    s.ecx = r[2];
    s.osxsave = ((r[2] >> 27) & 1u) != 0;
    if (!s.osxsave) return s;                             // no OSXSAVE: the OS is not saving YMM
#if defined(_MSC_VER)
    s.xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    s.xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    return s;
}

}  // namespace

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}
    }();
    return ok;
}

std::string cpu_name() {
    unsigned r[12] = {};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000u);
    if ((unsigned) x[0] < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuidex(x, (int) (0x80000002u + i));
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0x80000000u, a, b, c, d);
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) __cpuid(0x80000002u + i, r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
#endif
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}

/// The AVX1 rung: the CPU has AVX (256-bit float), SSSE3 and SSE4.1, but no AVX2.
///
/// Deliberately does NOT require FMA3 or F16C.  Those arrived a generation after AVX (Ivy Bridge,
/// 2011), while this probe exists for the generation before that (Sandy Bridge / Westmere, 2008-2010).
/// q2_avx1.cpp does a software fp16 decode and a mul+add rather than using them, so demanding them
/// here would reject exactly the CPUs the kernel is written for.  Verified by EXECUTION on a Xeon
/// E5-2680 (family 6 model 45 stepping 7): FMA (leaf 1 ECX 12) = 0, F16C (ECX 29) = 0, and
/// `vfmadd*` / `vcvtph2ps` each raise #UD there, while `pmaddubsw` works.
///
/// STRATA_FORCE_AVX2 is upstream's "skip the widest path" switch and only has a VALUE meaning at '1' -
/// cpu_avx512_ok() above tests `f[0] == '1'`.  An earlier revision of this probe tested mere PRESENCE
/// of the variable, which combined with the same test in cpu_avx512_ok() to make all three rungs return
/// false at once on an AVX-512 machine, and the startup gate then exited.  Hence `f[0] == '1'` here too.
bool cpu_avx1_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        if (const char* f = std::getenv("STRATA_FORCE_AVX1"); f != nullptr && f[0] == '1') return true;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        if (!((ecx1 >> 27) & 1u)) return false;            // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the YMM state
        const bool avx   = ((ecx1 >> 28) & 1u) != 0;      // leaf 1 ECX 28
        const bool ssse3 = ((ecx1 >> 9) & 1u) != 0;       // leaf 1 ECX 9
        const bool sse41 = ((ecx1 >> 19) & 1u) != 0;      // leaf 1 ECX 19
        return avx && ssse3 && sse41;
    }();
    return ok;
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) {
        q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    } else if (cpu_avx2_ok()) {
        q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
    } else if (cpu_avx1_ok()) {
        q2_0_gguf_rows_multi_avx1(w, row_bytes, nblocks, a, nt, out, r0, r1);
    } else {
        // Nothing vectorised is available; s2_expert_scalar is the only correct answer left, and it is
        // slow enough that it is better to say so than to decode at a crawl.
        std::fprintf(stderr,
                     "strata: no usable CPU kernel: this CPU has neither AVX-512 (F/BW/VL/VNNI/VBMI), AVX2, "
                     "nor AVX+FMA+F16C. The scalar fallback exists for tests only.\n");
        std::exit(1);
    }
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else if (cpu_avx2_ok()) act_quant_q8_1_avx2(x, n, a);
    else if (cpu_avx1_ok()) act_quant_q8_1_avx1(x, n, a);
    else std::exit(1);
}

/// LOCAL PORT (Z620): the startup gate, in the same file as the dispatch and for the same reason.
///
/// `cpu_require_expert_support()` lives in `expert.cpp`, which is compiled with `/arch:AVX512`, and the
/// CMakeLists comment above says why that is a hazard: a TU built with the flag may use those instructions
/// ANYWHERE in its code, so calling into it from a CPU without AVX-512 can trap inside what is supposed to
/// be the error message. This TU carries no per-file ISA flag, so asking here is safe everywhere.
///
/// The check is the full ladder, not just the AVX-512 rung, so a Sandy Bridge is told what it actually gets
/// (the AVX1 kernel) instead of being refused for a feature it was never going to use.
void cpu_require_expert_support_any() {
    if (cpu_avx512_ok()) return;
    if (cpu_avx2_ok()) return;
    if (cpu_avx1_ok()) return;
    std::fprintf(stderr,
                 "strata: this CPU cannot run the expert kernel: no AVX-512 (F/BW/VL/VNNI/VBMI), no AVX2, "
                 "and no AVX+SSSE3+SSE4.1.\n"
                 "        The fastest path Strata has on this CPU would be its scalar fallback, which exists "
                 "for tests only and is far too slow to decode with.\n");
    std::exit(1);
}

// ================================ LOCAL PORT: the canonical expert path, dispatched ================
//
// `pool.cpp` is the CPU expert worker loop and it calls `s2_expert_vnni_q` / `s2_expert_gu_rows` /
// `s2_expert_down_rows` and their `_multi` forms directly.  All of those are DEFINED in `expert.cpp`, a
// translation unit compiled with `/arch:AVX512`, so on a CPU without AVX-512 calling them is not merely
// slow - it is a trap.  Upstream's answer is to refuse the CPU at startup (`cpu_require_expert_support`).
//
// This adds the missing middle: each entry point picks the AVX-512 original or the AVX1 port.  Two
// properties matter and both come from where the dispatch LIVES:
//
//   * it is in this TU, which carries no per-file ISA flag, so evaluating `cpu_avx512_ok()` is safe
//     anywhere.  Putting the same check in `expert.cpp` would risk the compiler emitting AVX-512 into
//     the check itself.
//   * the AVX-512 originals are only CALLED when `cpu_avx512_ok()` has already returned true, which is
//     exactly the condition under which calling them is defined.  On a Sandy Bridge the branch is not
//     taken and the AVX1 port runs instead.
//
// Each wrapper is a straight forward-through: the AVX1 and AVX-512 versions have the same contract, and
// the parity test (s2_avx1_parity.cpp) checks that claim against an independent reference rather than
// taking it on trust.

// The AVX-512 originals, renamed from s2_expert_* in expert.cpp so these can be told apart.  The rename
// is the only change to that file and it is mechanical.

void s2_expert_vnni_q_any(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    if (cpu_avx512_ok()) s2_expert_vnni_q(blob, a1, out, ws);
    else s2_expert_vnni_q_avx1(blob, a1, out, ws);
}

void s2_expert_gu_rows_any(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    if (cpu_avx512_ok()) s2_expert_gu_rows(blob, a1, ff, r0, r1);
    else s2_expert_gu_rows_avx1(blob, a1, ff, r0, r1);
}

void s2_expert_down_rows_any(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    if (cpu_avx512_ok()) s2_expert_down_rows(blob, a2, out, r0, r1);
    else s2_expert_down_rows_avx1(blob, a2, out, r0, r1);
}

void s2_expert_gu_rows_multi_any(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                                 int r1) {
    if (cpu_avx512_ok()) s2_expert_gu_rows_multi(blob, a1, n_tokens, ff, r0, r1);
    else s2_expert_gu_rows_multi_avx1(blob, a1, n_tokens, ff, r0, r1);
}

void s2_expert_down_rows_multi_any(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out,
                                   int r0, int r1) {
    if (cpu_avx512_ok()) s2_expert_down_rows_multi(blob, a2, n_tokens, out, r0, r1);
    else s2_expert_down_rows_multi_avx1(blob, a2, n_tokens, out, r0, r1);
}

void s2_expert_vnni_multi_any(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                              ExpertScratchMulti& ws) {
    // The experimental oracle contract (`expert_set_oracle_q8_0`) is implemented only in the AVX-512 TU,
    // and it is off by default, so on this CPU fall back to one full expert per token rather than
    // pretending the two agree.  Upstream's own multi path does the same thing for this flag.
    if (expert_oracle_q8_0_enabled()) {
        for (int t = 0; t < n_tokens; ++t) s2_expert_vnni_q_any(blob, *a1[t], out[t], ws.single);
        return;
    }
    if (cpu_avx512_ok()) s2_expert_vnni_multi(blob, a1, n_tokens, out, ws);
    else s2_expert_vnni_multi_avx1(blob, a1, n_tokens, out, ws);
}

#if !defined(STRATA_NATIVE_EXPERTS)
// Without ggml-cpu no native pack loads (expert_layout_load refuses), so these are never reached.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
#endif

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (!in) {
        L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
        g_layout = L;
        return true;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    err = "this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS";
    return false;
#else
    L.native = true;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            if (!line.empty() && line[0] == '#') {
                // "# strata native experts vN: ...".  A version this engine does not know may mean columns it
                // would misread, so it is refused rather than parsed as far as the known columns go.
                static const char tag[] = "# strata native experts v";
                if (line.compare(0, sizeof tag - 1, tag) == 0) {
                    L.version = std::atoi(line.c_str() + sizeof tag - 1);
                    if (L.version > kExpertLayoutVersion) {
                        err = "native_experts.txt is v" + std::to_string(L.version) + "; this engine reads up to v" +
                              std::to_string(kExpertLayoutVersion) + " (the pack was written by a newer tools/"
                              "iq_pack.py: update the engine, or repack with this one)";
                        return false;
                    }
                }
                // v3 packs record their expert count in the header; a pruned model (GSQ-RCO Coder) ships
                // fewer experts than the canonical geometry the caller passes, which is a compile-time
                // default, so the header wins.
                const size_t at = line.find("(n_expert ");
                if (at != std::string::npos) L.n_expert = std::atoll(line.c_str() + at + 10);
            }
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, gt = -1, dt = -1;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dox = 0;
        if (!(ss >> l >> gt >> dt >> off >> blob) || l < 0 || l >= n_layers) {
            err = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        NativeFmt f;
        if (!native_fmt((int) gt, (int) dt, H, FF, f, err)) return false;
        if (f.bytes != blob) {
            err = "native_experts.txt: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        if (ss >> go >> uo >> dox) {   // v2 lines: the GGUF offsets
            if (L.gguf_off.empty()) L.gguf_off.assign((size_t) (3 * n_layers), 0);
            L.gguf_off[(size_t) (3 * l)] = go;
            L.gguf_off[(size_t) (3 * l + 1)] = uo;
            L.gguf_off[(size_t) (3 * l + 2)] = dox;
            std::string file;             // v3: the shard that holds this layer (a file name beside --native)
            if (ss >> file) {
                if (L.gguf_file.empty()) L.gguf_file.assign((size_t) (3 * n_layers), std::string());
                // One name covers all three roles.  When a shard boundary falls inside a layer the packer writes
                // "gate,up,down" instead (v4), an empty field meaning the --native shard.  A GGUF file name has no
                // comma, so the split is unambiguous; an engine older than v4 fails to open such a "file" loudly.
                // The comma form is from #255 (gopinath87607).
                std::vector<std::string> parts;
                for (size_t from = 0;;) {
                    const size_t comma = file.find(',', from);
                    parts.push_back(file.substr(from, comma == std::string::npos ? comma : comma - from));
                    if (comma == std::string::npos) break;
                    from = comma + 1;
                }
                if (parts.size() == 1) {
                    const std::string one = parts[0];   // not parts.assign(3, parts[0]): that aliases the element
                    parts.assign(3, one);                //   the assignment is about to overwrite
                }
                std::string extra;
                if (parts.size() != 3 || (ss >> extra)) {
                    err = "native_experts.txt: layer " + std::to_string(l) + ": the shard column is one name or "
                          "gate,up,down, not: " + line;
                    return false;
                }
                for (size_t r = 0; r < 3; ++r) L.gguf_file[(size_t) (3 * l) + r] = parts[r];
            }
        }
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) L.n_expert;
    }
    L.total = at;
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::kernels::cpu
