#include "kernels.hpp"
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int C = 3, H = fnx::NQH, D = fnx::HD, K = fnx::NKVH;
constexpr int NP = 4, CAP = 384;

struct Fixture {
    sycl::queue q;
    std::vector<void*> allocations;
    explicit Fixture(int card)
        : q((dpct::select_device(dpct::device_count() == 1 ? 0 : card), dpct::get_current_device()),
            sycl::property_list{sycl::property::queue::in_order{},
                                sycl::property::queue::enable_profiling{}}) {}
    ~Fixture() {
        for (void* p : allocations) sycl::free(p, q);
    }
    template <typename T> T* upload(const std::vector<T>& data) {
        auto* p = sycl::malloc_device<T>(data.size(), q);
        auto* pinned = sycl::malloc_host<T>(data.size(), q);
        if (!p || !pinned) throw std::bad_alloc();
        allocations.push_back(p);
        std::copy(data.begin(), data.end(), pinned);
        q.memcpy(p, pinned, data.size() * sizeof(T)).wait_and_throw();
        sycl::free(pinned, q);
        return p;
    }
    template <typename T> std::vector<T> read(T* p, size_t n) {
        auto* pinned = sycl::malloc_host<T>(n, q);
        if (!pinned) throw std::bad_alloc();
        q.memcpy(pinned, p, n * sizeof(T)).wait_and_throw();
        std::vector<T> result(pinned, pinned + n);
        sycl::free(pinned, q);
        return result;
    }
};

void check(int device, int count, bool dense, int position, int selection, bool paired,
           int page_rows = 96, int cards = 2, bool zero_query = false, bool zero_values = false,
           float query_scale = 0.015f, bool full_range = false) {
    const int R = page_rows, OWN = H / cards;
    const size_t BLOCK = (size_t) C * OWN * (D + 2);
    std::cout << "SETUP device=" << device << " count=" << count
              << " dense=" << dense << std::endl;
    Fixture f(device);
    const size_t page = fnx::kvp_page(device, true);
    const std::vector<int> table{3, 1, 0};
    auto* pt = f.upload(table);
    auto* pos = f.upload(std::vector<int>{position});
    auto* counts = f.upload(std::vector<int>(C, dense ? -1 : count));
    std::vector<int> ids(C * CAP);
    for (int r = 0; r < C; ++r)
        for (int i = 0; i < CAP; ++i)
            ids[r * CAP + i] = selection == 1 ? 2 * i :
                                selection == 2 ? (i / 4) * 8 + i % 4 : i;
    auto* cells = f.upload(ids);
    std::vector<float> qg(C * H * 2 * D);
    for (size_t i = 0; i < qg.size(); ++i)
        qg[i] = zero_query ? 0.0f : float(int(i * 19 % 97) - 48) * query_scale;
    auto* dq = f.upload(qg);
    auto* out = f.upload(std::vector<float>(C * OWN * D));
    std::vector<sycl::half> keys[2], values[2];
    float* partial[2];
    std::vector<float> got[2], normalized[2];
    auto offset = [&](int cell, int h, int d) {
        return (size_t) table[cell / R] * page * 2 +
               ((size_t) (cell % R) * K + h) * D + d;
    };
    auto scale = [&](int cell, int h, int d) {
        return (size_t) table[cell / R] * page +
               (size_t) R * K * D / 2 + ((size_t) (cell % R) * K + h) * 8 + d / 32;
    };
    auto ncell = [&](int card, int r) {
        if (!dense) return count;
        const int p = position + r, b = p / 4;
        return ((b + cards - 1 - card) / cards) * 4 + (b % cards == card ? p % 4 + 1 : 0);
    };
    auto at = [=](int r, int h) {
        return (((size_t) (h / OWN) * C + r) * OWN + h % OWN) * (D + 2);
    };
    const int code_max = full_range ? 127 : 63;
    for (int card = 0; card < 2; ++card) {
        keys[card].resize(NP * page);
        values[card].resize(NP * page);
        auto* kb = reinterpret_cast<int8_t*>(keys[card].data());
        auto* vb = reinterpret_cast<int8_t*>(values[card].data());
        for (int cell = 0; cell < 3 * R; ++cell)
            for (int h = 0; h < K; ++h) {
                for (int d = 0; d < D; ++d) {
                    kb[offset(cell, h, d)] = (cell * 7 + h * 11 + d * 3 + card * 17) % (2 * code_max + 1) - code_max;
                    vb[offset(cell, h, d)] = (cell * 13 + h * 5 + d * 7 + card * 19) % (2 * code_max + 1) - code_max;
                }
                for (int g = 0; g < 8; ++g) {
                    const int sg = paired ? g / 2 : g;
                    keys[card][scale(cell, h, 32 * g)] = sycl::half(0.002f * (1 + (cell + h + sg) % 7));
                    values[card][scale(cell, h, 32 * g)] = zero_values ? sycl::half(0.0f) :
                        sycl::half(0.003f * (1 + (2 * cell + h + sg) % 11));
                }
            }
        auto* dk = f.upload(keys[card]);
        auto* dv = f.upload(values[card]);
        partial[card] = f.upload(std::vector<float>(cards * BLOCK, 12345.f));
        strata::kernels::QsaSplitAttnLayout layout{
            pt, counts, pos, page, R, K, card, cards, C, 0, 1, CAP};
        for (int first : {1, 0, 2}) {
            layout.first_query = first;
            std::cout << "RUN partial card=" << card << " first=" << first << std::endl;
            if (!strata::kernels::qsa_prompt_attn_esimd_split(
                    dq, reinterpret_cast<uint16_t*>(dk), reinterpret_cast<uint16_t*>(dv),
                    cells, layout, partial[card], &f.q))
                throw std::runtime_error("Split ESIMD layout rejected");
        }
        got[card] = f.read(partial[card], cards * BLOCK);
        normalized[card].resize(C * H * D);
        for (int r = 0; r < C; ++r)
            for (int h = 0; h < H; ++h) {
                const int n = ncell(card, r), kh = h / (H / K);
                const size_t p = at(r, h);
                if (n == 0) {
                    if (got[card][p + D] != -INFINITY || got[card][p + D + 1] != 0)
                        throw std::runtime_error("Empty partial has wrong max/sum");
                    for (int d = 0; d < D; ++d)
                        if (got[card][p + d] != 0) throw std::runtime_error("Empty partial is not zero");
                    continue;
                }
                std::vector<double> scores(n);
                double maximum = -INFINITY;
                for (int i = 0; i < n; ++i) {
                    const int cell = dense ? i : ids[r * CAP + i];
                    double dot = 0;
                    for (int d = 0; d < D; ++d)
                        dot += qg[((size_t) r * H + h) * 2 * D + d] *
                               double(kb[offset(cell, kh, d)]) * float(keys[card][scale(cell, kh, d)]);
                    scores[i] = dot / std::sqrt(double(D));
                    maximum = std::max(maximum, scores[i]);
                }
                double sum = 0;
                for (double& score : scores) { score = std::exp(score - maximum); sum += score; }
                if (std::fabs(got[card][p + D] - maximum) > 0.001 ||
                    std::fabs(got[card][p + D + 1] - sum) > 0.001 * sum)
                    throw std::runtime_error("Partial max/sum disagrees with FP64");
                for (int d = 0; d < D; ++d) {
                    double value = 0;
                    for (int i = 0; i < n; ++i) {
                        const int cell = dense ? i : ids[r * CAP + i];
                        value += scores[i] * double(vb[offset(cell, kh, d)]) *
                                 float(values[card][scale(cell, kh, d)]);
                    }
                    const float actual = got[card][p + d] / got[card][p + D + 1];
                    normalized[card][((size_t) r * H + h) * D + d] = actual;
                    if (!std::isfinite(actual) || std::fabs(actual - value / sum) > 0.001)
                        throw std::runtime_error("KV32 attention disagrees with FP64");
                }
            }
        if (paired) {
            constexpr int PS = 128;
            std::vector<int8_t> pk(3 * PS * K * D), pv(pk.size());
            std::vector<uint16_t> sk(3 * PS * K * 4), sv(sk.size());
            std::vector<float> queries(C * H * D);
            std::vector<int> steps(C * strata::kernels::kStepCount), ppids(C * CAP);
            for (int cell = 0; cell < 3 * R; ++cell)
                for (int h = 0; h < K; ++h) {
                    const size_t row = ((size_t) (cell / PS) * K + h) * PS + cell % PS;
                    for (int d = 0; d < D; ++d) {
                        pk[row * D + d] = kb[offset(cell, h, d)];
                        pv[row * D + d] = vb[offset(cell, h, d)];
                    }
                    for (int g = 0; g < 4; ++g) {
                        sk[row * 4 + g] = reinterpret_cast<uint16_t*>(keys[card].data())[scale(cell, h, 64 * g)];
                        sv[row * 4 + g] = reinterpret_cast<uint16_t*>(values[card].data())[scale(cell, h, 64 * g)];
                    }
                }
            for (int r = 0; r < C; ++r) {
                steps[r * strata::kernels::kStepCount + strata::kernels::kStepWidth] = ncell(card, r);
                for (int i = 0; i < CAP; ++i) ppids[r * CAP + i] = dense ? i : ids[r * CAP + i];
                for (int h = 0; h < H; ++h)
                    std::copy_n(qg.data() + ((size_t) r * H + h) * 2 * D, D,
                                queries.data() + ((size_t) r * H + h) * D);
            }
            strata::kernels::QsaAttnPools pools;
            pools.k_q = f.upload(pk); pools.v_q = f.upload(pv);
            pools.k_scale = f.upload(sk); pools.v_scale = f.upload(sv);
            pools.page_table = f.upload(std::vector<int>{0, 1, 2});
            strata::kernels::QsaShapes shape;
            shape.n_head = H; shape.n_head_kv = K; shape.head_dim = D; shape.page_size = PS;
            auto* ppout = f.upload(std::vector<float>(C * H * D));
            if (!strata::kernels::qsa_prompt_attn_esimd(f.upload(queries), pools,
                    f.upload(ppids), f.upload(steps), CAP, shape, ppout, C, &f.q))
                throw std::runtime_error("Direct PP ESIMD path rejected");
            const auto pp = f.read(ppout, C * H * D);
            for (size_t i = 0; i < pp.size(); ++i)
                if (std::fabs(pp[i] - normalized[card][i]) > 0.00002f)
                    throw std::runtime_error("KV64-equivalent fixture differs from direct PP kernel");
        }
    }
    if (cards == 2 && (dense || count > 0)) {
        for (int owner = 0; owner < 2; ++owner) {
            fnx::attn_prompt_merge(partial[owner] + owner * BLOCK,
                partial[1 - owner] + owner * BLOCK, dq, out, C, owner, &f.q);
            const auto merged = f.read(out, C * OWN * D);
            for (int r = 0; r < C; ++r)
                for (int lh = 0; lh < OWN; ++lh) {
                    const int h = owner * OWN + lh;
                    const size_t p = at(r, h);
                    const double m = std::max(got[0][p + D], got[1][p + D]);
                    const double w0 = std::isfinite(got[0][p + D]) ? std::exp(got[0][p + D] - m) : 0;
                    const double w1 = std::isfinite(got[1][p + D]) ? std::exp(got[1][p + D] - m) : 0;
                    const double z = w0 * got[0][p + D + 1] + w1 * got[1][p + D + 1];
                    for (int d = 0; d < D; ++d) {
                        const double value = (w0 * got[0][p + d] + w1 * got[1][p + d]) / z;
                        const double gate = 1.0 / (1.0 + std::exp(-qg[((size_t) r * H + h) * 2 * D + D + d]));
                        const float actual = merged[((size_t) r * OWN + lh) * D + d];
                        if (!std::isfinite(actual) || std::fabs(actual - value * gate) > 0.00002)
                            throw std::runtime_error("All-cards partial merge/gate ABI mismatch");
                    }
                }
        }
    }
    std::cout << "PASS device=" << device << " count=" << count << " dense=" << dense
              << " position=" << position << " selection=" << selection
              << " paired_scales=" << paired << " page_rows=" << R << " cards=" << cards
              << " zero_query=" << zero_query << " zero_values=" << zero_values
              << " query_scale=" << query_scale << " full_range=" << full_range << std::endl;
}
}

int main(int argc, char** argv) {
    try {
        int first = 0, last = 2;
        if (argc != 1) {
            if (argc != 3 || std::string(argv[1]) != "--card")
                throw std::runtime_error("Usage: sycl_native_attention_test [--card 0|1]");
            first = std::stoi(argv[2]);
            if (first < 0 || first > 1) throw std::runtime_error("Select GPU0 or GPU1");
            last = first + 1;
        }
        for (int card = first; card < last; ++card) {
            check(card, 0, false, 0, 0, false);
            check(card, 0, true, 0, 0, false);
            check(card, 0, true, 383, 0, false);
            check(card, 33, false, 0, 1, false);
            check(card, 129, false, 0, 2, false);
            check(card, 193, false, 0, 0, true);
            check(card, 193, false, 0, 0, false, 128);
            check(card, 193, false, 0, 0, false, 96, 4);
            check(card, 129, false, 0, 2, false, 96, 2, true);
            check(card, 33, false, 0, 1, false, 96, 2, false, true);
            check(card, 257, false, 0, 0, false, 96, 2, false, false, 0.06f);
            check(card, 288, false, 0, 0, false, 96, 2, false, false, 0.06f, true);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
