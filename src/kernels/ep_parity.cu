// src/kernels/ep_parity.cpp - expert parallelism (--expert-parallel-device) without a model:
//
//   A. the two plans of a pair (resident_plan_ep, ep_kernels.hpp) against a host replay of their contract.
//      Each expert is resident on card 0, on card 1, on both, or on neither.  Owner: card 0 when it holds the
//      expert, else card 1.  For each role the plan must be resident_plan's layout over exactly the entries that
//      card owns: distinct owned experts in routing order, each with its entries ascending, ptr[g] = the slot on
//      that card, counts[1] = its entries.  An expert on neither card leaves an empty plan and sets *plan_err.
//      With every expert on card 0, role 1 is resident_plan's all-resident plan word for word.
//
//   B. the mapped row/flag exchange round trip (ep_push, ep_wait, ep_copy_uncached, ep_send_rows,
//      ep_copy_peer_rows), first as plain kernels and then captured into the two per-window graphs the engine
//      records.  Five epochs: card 0 pushes window-dependent ids and q8_1 bytes and card 1's checksum of what it
//      copied must match the host's, or the push/wait protocol lost the window; card 1 computes window-dependent
//      rows for its half of the entries and card 0's parts must hold them bit-exact while its own rows are
//      untouched.
//
// GPU (two cards), synthetic, no model.  Needs two devices; on a one-GPU box it prints SKIP and returns 0.
#include "strata/kernels/ep_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

constexpr int NE = 512;          // expert ids in the pool
constexpr int K = 10;            // routing width
constexpr int NT = 2;            // window tokens
constexpr int NENT = NT * K;     // entries this window
constexpr int NW = 64;           // row width (floats)
constexpr long long CAPX = 128;  // the plan's capacity in entries
constexpr int WINDOWS = 5;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void fill_ids_kernel(int32_t* ids, int n, const uint32_t* ctr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = (int32_t) ((*ctr - 1) * 100u + (uint32_t) (i % 50));
}

__global__ void fill_xq_kernel(uint32_t* xq, int words, const uint32_t* ctr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) xq[i] = (*ctr - 1) * 1000u + (uint32_t) i;
}

__global__ void fill_parts_kernel(float* parts, int n, int row) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * row; i += gridDim.x * blockDim.x) parts[i] = -1.0f;
}

__global__ void fill_rows_kernel(float* rows, int n, int row, const uint32_t* ctr) {
    const int r = blockIdx.x;
    const float base = (float) (*ctr - 1) * 10000.0f + (float) r * 100.0f;
    for (int i = threadIdx.x; i < row; i += blockDim.x) rows[(size_t) r * row + i] = base + (float) i;
}

/// card 1: a checksum of everything it copied out of the inbox, to mapped memory
__global__ void checksum_kernel(const uint32_t* in, int words, uint32_t* out) {
    __shared__ uint32_t s[256];
    uint32_t acc = 0;
    for (int i = threadIdx.x; i < words; i += blockDim.x) acc += in[i] * 2654435761u + (uint32_t) i;
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off /= 2) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s[0];
}

struct HostPlan {
    std::vector<int> ge;                    // the owned experts, routing order
    std::vector<std::vector<int>> gi;       // their entry indices
};

HostPlan host_plan(const std::vector<int32_t>& ids, const std::vector<int32_t>& r0, const std::vector<int32_t>& r1,
                   int role) {
    HostPlan out;
    for (int i = 0; i < (int) ids.size(); ++i) {
        const int e = ids[(size_t) i];
        const bool in_range = e >= 0 && e < NE;
        const bool own = role == 1 ? (in_range && r0[(size_t) e] >= 0)
                                   : (in_range && r0[(size_t) e] < 0 && r1[(size_t) e] >= 0);
        if (!own) continue;
        int g = 0;
        while (g < (int) out.ge.size() && out.ge[(size_t) g] != e) ++g;
        if (g == (int) out.ge.size()) { out.ge.push_back(e); out.gi.emplace_back(); }
        out.gi[(size_t) g].push_back(i);
    }
    return out;
}

long long plan_words() {
    const long long ptr_off = ((4 + (CAPX + 1) + 2 * CAPX) + 1) & ~1ll;
    return ptr_off + 4 * CAPX + (CAPX + 1) + 1;
}

/// One card's plan over `ids`: compare word for word with the host replay; returns failures added.
int check_plan(int role, const std::vector<int32_t>& ids, std::vector<int32_t> r0, std::vector<int32_t> r1,
               int32_t* d_r0, int32_t* d_r1, int32_t* d_ids, int32_t* d_plan, uint32_t* d_err, int dev,
               bool missing_case, int case_id) {
    const size_t words = (size_t) plan_words();
    int fails = 0;
    if (missing_case) { r0[(size_t) ids[0]] = -1; r1[(size_t) ids[0]] = -1; }
    cudaSetDevice(dev);
    cudaMemcpy(d_r0, r0.data(), NE * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r1, r1.data(), NE * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    std::vector<int32_t> plan(words, 0x5a5a5a5a);
    cudaMemcpy(d_plan, plan.data(), words * 4, cudaMemcpyHostToDevice);
    uint32_t zero = 0;
    cudaMemcpy(d_err, &zero, 4, cudaMemcpyHostToDevice);
    k::resident_plan_ep(d_ids, (int) ids.size(), K, d_r0, d_r1, role, NE, (const uint8_t*) (uintptr_t) 0x40000000,
                        nullptr, 4096, d_plan, CAPX, nullptr, d_err);
    check("resident_plan_ep");
    cudaMemcpy(plan.data(), d_plan, words * 4, cudaMemcpyDeviceToHost);
    uint32_t err = 0;
    cudaMemcpy(&err, d_err, 4, cudaMemcpyDeviceToHost);
    if (missing_case) {
        if (err != 1 || plan[0] != 0 || plan[1] != 0) {
            std::printf("FAIL case %d role %d: a missing expert gave err %u groups %d\n", case_id, role, err, plan[0]);
            ++fails;
        }
        return fails;
    }
    const HostPlan hp = host_plan(ids, r0, r1, role);
    const int32_t* start = plan.data() + 4;
    const int32_t* dst = start + CAPX + 1;
    const int32_t* tok = dst + CAPX;
    const long long ptr_off = ((4 + (CAPX + 1) + 2 * CAPX) + 1) & ~1ll;
    const auto* ptr = (const unsigned long long*) (plan.data() + ptr_off);
    const int32_t* start2 = plan.data() + ptr_off + 4 * CAPX;
    int ents = 0;
    for (const auto& v : hp.gi) ents += (int) v.size();
    bool ok = err == 0 && plan[0] == (int) hp.ge.size() && plan[1] == ents && plan[2] == 0 &&
              start[hp.ge.size()] == ents && start2[0] == ents;
    if (!ok)
        std::printf("FAIL case %d role %d: counts groups %d (want %d) entries %d (want %d) tail %d/%d err %u\n",
                    case_id, role, plan[0], (int) hp.ge.size(), plan[1], ents, start[hp.ge.size()], start2[0], err);
    const auto& own = role == 1 ? r0 : r1;
    int at = 0;
    for (size_t g = 0; ok && g < hp.ge.size(); ++g) {
        if (ptr[g] != (unsigned long long) (0x40000000ull + (size_t) own[(size_t) hp.ge[g]] * 4096ull)) {
            std::printf("FAIL case %d role %d: group %zu ptr %llx want %llx (expert %d slot %d)\n", case_id, role, g,
                        (unsigned long long) ptr[g],
                        (unsigned long long) (0x40000000ull + (size_t) own[(size_t) hp.ge[g]] * 4096ull),
                        hp.ge[g], own[(size_t) hp.ge[g]]);
            ok = false;
            break;
        }
        if (start[g] != at) {
            std::printf("FAIL case %d role %d: group %zu start %d want %d\n", case_id, role, g, start[g], at);
            ok = false;
            break;
        }
        for (int i : hp.gi[g]) {
            if (dst[at] != i || tok[at] != i / K) {
                std::printf("FAIL case %d role %d: entry %d dst %d (want %d) tok %d (want %d)\n", case_id, role, at,
                            dst[at], i, tok[at], i / K);
                std::printf("  ids:");
                for (int x : ids) std::printf(" %d", x);
                std::printf("\n  r0:");
                for (int x : ids) std::printf(" %d", r0[(size_t) x]);
                std::printf("\n  r1:");
                for (int x : ids) std::printf(" %d", r1[(size_t) x]);
                std::printf("\n  host groups %zu:", hp.ge.size());
                for (size_t gg = 0; gg < hp.ge.size(); ++gg) {
                    std::printf(" e%d=[", hp.ge[gg]);
                    for (int i2 : hp.gi[gg]) std::printf("%d,", i2);
                    std::printf("]");
                }
                std::printf("\n  kern groups %d start:", plan[0]);
                for (int gg = 0; gg <= plan[0] && gg < 12; ++gg) std::printf(" %d", start[gg]);
                std::printf(" dst:");
                for (int gg = 0; gg < std::min(plan[1], 24); ++gg) std::printf(" %d", dst[gg]);
                std::printf("\n");
                ok = false;
                break;
            }
            ++at;
        }
    }
    if (!ok) ++fails;
    return fails;
}

uint32_t inbox_checksum(const std::vector<int32_t>& ids, int window, int xq_words) {
    // the inbox slot's words: NENT ids, zero padding to 64 words, then the xq words
    std::vector<uint32_t> w(64 + (size_t) xq_words, 0);
    std::memcpy(w.data(), ids.data(), ids.size() * 4);
    for (int i = 0; i < xq_words; ++i) w[64 + (size_t) i] = (uint32_t) ((window - 1) * 1000 + i);
    uint32_t acc = 0;
    for (size_t i = 0; i < w.size(); ++i) acc += w[i] * 2654435761u + (uint32_t) i;
    return acc;
}

}  // namespace

int main() {
    setbuf(stdout, nullptr);
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev < 2) {
        std::printf("ep_parity: SKIP: two devices are needed (found %d)\n", ndev);
        return 0;
    }
    int fails = 0, cases = 0;

    // ---------------- A. the two plans of a pair ----------------
    {
        std::mt19937 rng(1402);
        const size_t words = (size_t) plan_words();
        int32_t* d_ids0 = nullptr; int32_t* d_r00 = nullptr; int32_t* d_r10 = nullptr; int32_t* d_plan0 = nullptr;
        int32_t* d_ids1 = nullptr; int32_t* d_r01 = nullptr; int32_t* d_r11 = nullptr; int32_t* d_plan1 = nullptr;
        int32_t* d_plan_ref = nullptr;
        uint32_t* d_err0 = nullptr; uint32_t* d_err1 = nullptr;
        cudaSetDevice(0);
        cudaMalloc(&d_ids0, NENT * 4); cudaMalloc(&d_r00, NE * 4); cudaMalloc(&d_r10, NE * 4);
        cudaMalloc(&d_plan0, words * 4); cudaMalloc(&d_plan_ref, words * 4); cudaMalloc(&d_err0, 4);
        cudaSetDevice(1);
        cudaMalloc(&d_ids1, NENT * 4); cudaMalloc(&d_r01, NE * 4); cudaMalloc(&d_r11, NE * 4);
        cudaMalloc(&d_plan1, words * 4); cudaMalloc(&d_err1, 4);
        for (int pool : {3, 10, 40, 160, 512}) {
            for (int n : {1, 2, 5, 9, 17}) {
                for (int rep = 0; rep < 6; ++rep) {
                    ++cases;
                    std::vector<int32_t> ids((size_t) n), r0((size_t) NE, -1), r1((size_t) NE, -1);
                    std::vector<int> pick((size_t) pool);
                    for (int& p : pick) p = (int) (rng() % NE);
                    for (auto& v : ids) v = pick[rng() % (size_t) pool];
                    for (int e = 0; e < NE; ++e) {
                        const unsigned r = rng() % 100;
                        if (r < 45) r0[(size_t) e] = (int) (rng() % 9000);
                        else if (r < 90) r1[(size_t) e] = (int) (rng() % 9000);
                        else { r0[(size_t) e] = (int) (rng() % 9000); r1[(size_t) e] = (int) (rng() % 9000); }
                    }
                    const bool make_missing = rep == 5;
                    fails += check_plan(1, ids, r0, r1, d_r00, d_r10, d_ids0, d_plan0, d_err0, 0, make_missing, cases);
                    fails += check_plan(2, ids, r0, r1, d_r01, d_r11, d_ids1, d_plan1, d_err1, 1, make_missing, cases);
                }
            }
        }
        // role 1 with every expert on card 0 is resident_plan's all-resident plan word for word
        ++cases;
        std::vector<int32_t> ids((size_t) NENT), r0((size_t) NE), r1((size_t) NE, -1);
        std::mt19937 rng2(77);
        for (int i = 0; i < NENT; ++i) ids[(size_t) i] = (int) (rng2() % NE);
        for (int e = 0; e < NE; ++e) r0[(size_t) e] = (int) (rng2() % 9000);
        cudaSetDevice(0);
        cudaMemcpy(d_ids0, ids.data(), NENT * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r00, r0.data(), NE * 4, cudaMemcpyHostToDevice);
        std::vector<int32_t> a(words, 0x11111111), b(words, 0x11111111);
        cudaMemcpy(d_plan0, a.data(), words * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_plan_ref, b.data(), words * 4, cudaMemcpyHostToDevice);
        uint32_t zero = 0;
        cudaMemcpy(d_err0, &zero, 4, cudaMemcpyHostToDevice);
        k::resident_plan_ep(d_ids0, NENT, K, d_r00, d_r10, 1, NE, (const uint8_t*) (uintptr_t) 0x40000000, nullptr, 4096,
                            d_plan0, CAPX, nullptr, d_err0);
        k::resident_plan(d_ids0, NENT, K, d_r00, NE, (const uint8_t*) (uintptr_t) 0x40000000, nullptr, 4096,
                         d_plan_ref, CAPX, nullptr, 0, nullptr, d_err0);
        check("plan compare");
        cudaMemcpy(a.data(), d_plan0, words * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_plan_ref, words * 4, cudaMemcpyDeviceToHost);
        if (a != b) { std::printf("FAIL: role 1 with every expert on card 0 differs from resident_plan\n"); ++fails; }
        cudaFree(d_ids0); cudaFree(d_r00); cudaFree(d_r10); cudaFree(d_plan0); cudaFree(d_plan_ref); cudaFree(d_err0);
        cudaSetDevice(1);
        cudaFree(d_ids1); cudaFree(d_r01); cudaFree(d_r11); cudaFree(d_plan1); cudaFree(d_err1);
    }
    std::printf("ep_parity plans: %d cases, %d failures\n", cases, fails);

    // ---------------- B. the mapped exchange round trip (EP_PARITY_NO_B=1: the plans only) ----------------
    if (std::getenv("EP_PARITY_NO_B") == nullptr) {
        const int xq_words = 128;                        // 512 bytes of "q8_1 rows"
        const size_t slot_bytes = 256 + (size_t) xq_words * 4;
        const size_t flags = (size_t) (2 * WINDOWS + 2) * 64;
        const size_t rows_bytes = (size_t) NENT * NW * 4;
        void* h_inbox = nullptr; void* h_inflag = nullptr; void* h_ret = nullptr; void* h_rows = nullptr;
        void* h_err = nullptr;
        cudaSetDevice(0);
        cudaHostAlloc(&h_inbox, 2 * slot_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostAlloc(&h_inflag, flags, cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostAlloc(&h_ret, flags, cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostAlloc(&h_rows, rows_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostAlloc(&h_err, 256, cudaHostAllocMapped | cudaHostAllocPortable);
        std::memset(h_inbox, 0, 2 * slot_bytes);
        std::memset(h_inflag, 0, flags);
        std::memset(h_ret, 0, flags);
        std::memset(h_rows, 0, rows_bytes);
        std::memset(h_err, 0, 256);
        void *inbox0 = nullptr, *inflag0 = nullptr, *ret0 = nullptr, *rows0 = nullptr, *err0 = nullptr;
        void *inbox1 = nullptr, *inflag1 = nullptr, *ret1 = nullptr, *rows1m = nullptr, *err1 = nullptr;
        cudaHostGetDevicePointer(&inbox0, h_inbox, 0);
        cudaHostGetDevicePointer(&inflag0, h_inflag, 0);
        cudaHostGetDevicePointer(&ret0, h_ret, 0);
        cudaHostGetDevicePointer(&rows0, h_rows, 0);
        cudaHostGetDevicePointer(&err0, h_err, 0);
        cudaSetDevice(1);
        cudaHostGetDevicePointer(&inbox1, h_inbox, 0);
        cudaHostGetDevicePointer(&inflag1, h_inflag, 0);
        cudaHostGetDevicePointer(&ret1, h_ret, 0);
        cudaHostGetDevicePointer(&rows1m, h_rows, 0);
        cudaHostGetDevicePointer(&err1, h_err, 0);
        if (inbox0 != inbox1)
            std::printf("ep_parity: the two cards' aliases differ (%p / %p); both are used\n", inbox0, inbox1);
        check("host mappings");

        int32_t ids_h[NENT], res0_h[NE];
        for (int i = 0; i < NENT; ++i) ids_h[i] = i * 7 % 50;   // every id below 50; ownership by parity
        for (int e = 0; e < NE; ++e) res0_h[e] = (e % 2 == 0) ? e % 100 : -1;
        int32_t *d_ids = nullptr, *d_res0_1 = nullptr, *d_in1 = nullptr, *d_res0 = nullptr;
        float* d_rows1 = nullptr; float* d_parts = nullptr;
        uint32_t *d_ctr0 = nullptr, *d_ctr1 = nullptr, *d_done1 = nullptr, *d_ck = nullptr, *d_xq = nullptr;
        cudaSetDevice(0);
        cudaMalloc(&d_ids, NENT * 4);
        cudaMalloc(&d_xq, (size_t) xq_words * 4);
        cudaMalloc(&d_res0, NE * 4);
        cudaMalloc(&d_parts, rows_bytes);
        cudaMalloc(&d_ctr0, 64);
        cudaMemset(d_ctr0, 0, 64);
        cudaSetDevice(1);
        cudaMalloc(&d_res0_1, NE * 4);
        cudaMalloc(&d_in1, 2 * slot_bytes);
        cudaMalloc(&d_rows1, rows_bytes);
        cudaMalloc(&d_ctr1, 64);
        cudaMemset(d_ctr1, 0, 64);
        cudaMalloc(&d_done1, 64);
        cudaMemset(d_done1, 0, 64);
        cudaMalloc(&d_ck, 64);
        cudaMemset(d_ck, 0, 64);
        cudaMemcpy(d_res0_1, res0_h, NE * 4, cudaMemcpyHostToDevice);

        hipStream_t s0 = nullptr, s1 = nullptr;
        cudaSetDevice(0); cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        cudaSetDevice(1); cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

        for (int phase = 1; phase <= 2; ++phase) {   // 1: plain kernels; 2: captured graphs
            uint32_t zero4[16] = {};
            cudaMemcpy(d_ctr0, zero4, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_ctr1, zero4, 4, cudaMemcpyHostToDevice);
            std::memset(h_inflag, 0, flags);
            std::memset(h_ret, 0, flags);
            std::memset(h_err, 0, 256);
            for (int e = 0; e < NE; ++e) res0_h[e] = (e % 2 == 0) ? e % 100 : -1;
            cudaSetDevice(0);
            cudaMemcpy(d_res0, res0_h, NE * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(d_ids, ids_h, NENT * 4, cudaMemcpyHostToDevice);

            auto body1 = [&]() {   // card 1's half of one window (device 1 current)
                cudaSetDevice(1);
                k::ep_bump(d_ctr1, s1);
                k::ep_wait((const uint32_t*) inflag1, d_ctr1, 200000000u, (uint32_t*) ((uint8_t*) err1 + 8), 1, s1);
                k::ep_copy_uncached(d_in1, inbox1, slot_bytes, s1);
                checksum_kernel<<<1, 256, 0, s1>>>((const uint32_t*) d_in1, (int) (slot_bytes / 4), d_ck);
                check("checksum");
                fill_rows_kernel<<<NENT, 128, 0, s1>>>(d_rows1, NENT, NW, d_ctr1);
                check("fill_rows");
                k::ep_send_rows((float*) rows1m, d_rows1, (const int32_t*) d_in1, d_res0_1, NENT, NW,
                                d_done1, (uint32_t*) ret1, d_ctr1, s1);
            };
            auto body0 = [&]() {   // card 0's half of one window (device 0 current)
                cudaSetDevice(0);
                k::ep_bump(d_ctr0, s0);
                fill_ids_kernel<<<(NENT + 127) / 128, 128, 0, s0>>>(d_ids, NENT, d_ctr0);
                fill_xq_kernel<<<(xq_words + 127) / 128, 128, 0, s0>>>(d_xq, xq_words, d_ctr0);
                fill_parts_kernel<<<8, 256, 0, s0>>>(d_parts, NENT, NW);
                check("fill bodies");
                k::ep_push(d_ids, NENT, (const uint8_t*) d_xq, (size_t) xq_words * 4, (int32_t*) inbox0,
                           (uint8_t*) inbox0 + 256, (uint32_t*) inflag0, d_ctr0, s0);
                k::ep_wait((const uint32_t*) ret0, d_ctr0, 200000000u, (uint32_t*) err0, 2, s0);
                k::ep_copy_peer_rows(d_parts, (const float*) rows0, d_ids, d_res0, NENT, NW, s0);
            };

            cudaGraphExec_t gA = nullptr, gB = nullptr;
            if (phase == 2) {
                cudaSetDevice(1); cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal);
                cudaSetDevice(0); cudaStreamBeginCapture(s0, cudaStreamCaptureModeThreadLocal);
                body1();
                body0();
                cudaGraph_t ga = nullptr, gb = nullptr;
                cudaSetDevice(0); cudaStreamEndCapture(s0, &ga);
                cudaSetDevice(1); cudaStreamEndCapture(s1, &gb);
                check("end capture");
                cudaSetDevice(0); cudaGraphInstantiate(&gA, ga, nullptr, nullptr, 0);
                cudaSetDevice(1); cudaGraphInstantiate(&gB, gb, nullptr, nullptr, 0);
                check("instantiate");
                cudaGraphDestroy(ga); cudaGraphDestroy(gb);
            }
            for (int w = 1; w <= WINDOWS; ++w) {
                if (phase == 1) {
                    cudaSetDevice(1); body1();
                    cudaSetDevice(0); body0();
                } else {
                    cudaSetDevice(1); cudaGraphLaunch(gB, s1);
                    cudaSetDevice(0); cudaGraphLaunch(gA, s0);
                }
                cudaSetDevice(0); cudaStreamSynchronize(s0);
                cudaSetDevice(1); cudaStreamSynchronize(s1);
                uint32_t ck_host = 0, e0 = 0, e1 = 0;
                cudaMemcpy(&ck_host, d_ck, 4, cudaMemcpyDeviceToHost);
                std::memcpy(&e0, (uint8_t*) h_err + 0, 4);
                std::memcpy(&e1, (uint8_t*) h_err + 8, 4);
                std::vector<int32_t> want_ids((size_t) NENT);
                for (int i = 0; i < NENT; ++i) want_ids[(size_t) i] = (w - 1) * 100 + (i % 50);
                const uint32_t want_ck = inbox_checksum(want_ids, w, xq_words);
                std::vector<float> parts_h((size_t) NENT * NW);
                cudaMemcpy(parts_h.data(), d_parts, rows_bytes, cudaMemcpyDeviceToHost);
                int bad_rows = 0, stale_own = 0;
                for (int r = 0; r < NENT; ++r) {
                    const bool peer = (ids_h[r] % 2 == 1);
                    const float base = (float) (w - 1) * 10000.0f + (float) r * 100.0f;
                    if (peer) {
                        for (int i = 0; i < NW; ++i)
                            if (parts_h[(size_t) r * NW + i] != base + (float) i) { ++bad_rows; break; }
                    } else {
                        for (int i = 0; i < NW; ++i)
                            if (parts_h[(size_t) r * NW + i] != -1.0f) { ++stale_own; break; }
                    }
                }
                if (ck_host != want_ck || bad_rows || stale_own || e0 || e1) {
                    std::printf("FAIL %s window %d: checksum %llx (want %llx) peer rows wrong %d, own rows touched %d, "
                                "errs %u/%u\n", phase == 1 ? "plain" : "graph", w, (unsigned long long) ck_host,
                                (unsigned long long) want_ck, bad_rows, stale_own, e0, e1);
                    ++fails;
                }
            }
            if (phase == 2) { cudaGraphExecDestroy(gA); cudaGraphExecDestroy(gB); }
        }
        cudaSetDevice(0); cudaStreamDestroy(s0);
        cudaSetDevice(1); cudaStreamDestroy(s1);
        cudaSetDevice(0);
        cudaFree(d_ids); cudaFree(d_xq); cudaFree(d_res0); cudaFree(d_parts); cudaFree(d_ctr0);
        cudaSetDevice(1);
        cudaFree(d_res0_1); cudaFree(d_in1); cudaFree(d_rows1); cudaFree(d_ctr1); cudaFree(d_done1); cudaFree(d_ck);
        cudaFreeHost(h_inbox); cudaFreeHost(h_inflag); cudaFreeHost(h_ret); cudaFreeHost(h_rows); cudaFreeHost(h_err);
    }
    std::printf("ep_parity: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
