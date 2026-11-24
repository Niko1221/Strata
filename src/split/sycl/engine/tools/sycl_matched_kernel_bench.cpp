#include "sycl_kernel_bench.hpp"
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"

namespace {
void attention(sycl::queue& q, int card, int rows, int selected, bool sparse,
               int warmup, int repeats) {
    using namespace kernel_bench;
    constexpr int H = fnx::NQH, K = fnx::NKVH, D = fnx::HD;
    constexpr int extent = 16384, PS = 4;
    const int R = fnx::kvp_rows(card), pages = (extent + R - 1) / R;
    const size_t page = fnx::kvp_page(card, true);
    const std::string name = "attn_q" + std::to_string(rows) + "_kv" +
        std::to_string(selected) + (sparse ? "_sparse" : "_contiguous");
    std::cout << "SETUP card=" << card << " case=" << name
              << " heads=" << H << " kv_heads=" << K << " dim=" << D
              << " pp_page=" << PS << " split_page=" << R
              << " split_page_halves=" << page << std::endl;
    Fixture f(q);
    std::vector<int> table(pages), pptable(extent / PS);
    for (int i = 0; i < pages; ++i) table[i] = pages - 1 - i;
    for (int i = 0; i < extent / PS; ++i) pptable[i] = extent / PS - 1 - i;
    std::vector<sycl::half> kb((size_t)pages * page), vb(kb.size());
    auto* kbytes = reinterpret_cast<int8_t*>(kb.data());
    auto* vbytes = reinterpret_cast<int8_t*>(vb.data());
    std::vector<int8_t> pk((size_t)extent * K * D), pv(pk.size());
    std::vector<uint16_t> sk((size_t)extent * K * 4), sv(sk.size());
    for (int c = 0; c < extent; ++c)
        for (int h = 0; h < K; ++h) {
            const size_t split_code = (size_t)table[c / R] * page * 2 +
                ((size_t)(c % R) * K + h) * D;
            const size_t split_scale = (size_t)table[c / R] * page +
                (size_t)R * K * D / 2 + ((size_t)(c % R) * K + h) * 8;
            const size_t pp_row = ((size_t)pptable[c / PS] * K + h) * PS + c % PS;
            for (int d = 0; d < D; ++d) {
                const int8_t k = (c * 7 + h * 11 + d * 3) % 127 - 63;
                const int8_t v = (c * 13 + h * 5 + d * 7) % 127 - 63;
                kbytes[split_code + d] = pk[pp_row * D + d] = k;
                vbytes[split_code + d] = pv[pp_row * D + d] = v;
            }
            for (int g = 0; g < 8; ++g) {
                // Paired KV32 scales represent exactly the same values as PP KV64.
                sycl::half ks((1 + (c + h + g / 2) % 7) / 256.f);
                sycl::half vs((1 + (2 * c + h + g / 2) % 11) / 256.f);
                kb[split_scale + g] = ks;
                vb[split_scale + g] = vs;
                sk[pp_row * 4 + g / 2] = sycl::bit_cast<uint16_t>(ks);
                sv[pp_row * 4 + g / 2] = sycl::bit_cast<uint16_t>(vs);
            }
        }
    std::vector<float> queries((size_t)rows * H * D), qg(queries.size() * 2);
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                const float v = (int(((size_t)r * H * D + h * D + d) * 19 % 97) - 48) * 0.015f;
                queries[((size_t)r * H + h) * D + d] = v;
                qg[((size_t)r * H + h) * 2 * D + d] = v;
            }
    std::vector<int> ids((size_t)rows * selected);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < selected; ++i)
            ids[(size_t)r * selected + i] = sparse
                ? (i / 4) * (extent / selected) * 4 + i % 4 : i;
    std::vector<int> steps((size_t)rows * strata::kernels::kStepCount);
    for (int r = 0; r < rows; ++r)
        steps[(size_t)r * strata::kernels::kStepCount + strata::kernels::kStepWidth] = selected;
    auto* dq = f.upload(queries);
    auto* dqg = f.upload(qg);
    auto* di = f.upload(ids);
    auto* ds = f.upload(steps);
    auto* ppout = f.alloc<float>(queries.size());
    auto* partial = f.alloc<float>((size_t)rows * H * (D + 2));
    strata::kernels::QsaAttnPools pools;
    pools.k_q = f.upload(pk); pools.v_q = f.upload(pv);
    pools.k_scale = f.upload(sk); pools.v_scale = f.upload(sv);
    pools.page_table = f.upload(pptable);
    auto shape = strata::kernels::qsa_real_shapes();
    shape.n_head = H; shape.n_head_kv = K; shape.head_dim = D; shape.page_size = PS;
    strata::kernels::QsaSplitAttnLayout layout{
        f.upload(table), f.upload(std::vector<int>(rows, selected)),
        f.upload(std::vector<int>{32767}), page, R, K, card, 2,
        rows, 0, rows, selected};
    auto* dk = f.upload(kb);
    auto* dv = f.upload(vb);
    auto pp = [&] {
        if (!strata::kernels::qsa_prompt_attn_esimd(
                dq, pools, di, ds, selected, shape, ppout, rows, &q))
            throw std::runtime_error("PP attention launch rejected");
    };
    auto split = [&] {
        if (!strata::kernels::qsa_prompt_attn_esimd_split(
                dqg, reinterpret_cast<uint16_t*>(dk), reinterpret_cast<uint16_t*>(dv),
                di, layout, partial, &q))
            throw std::runtime_error("All-cards attention launch rejected");
    };
    pp(); split(); q.wait_and_throw();
    auto expected = f.read(ppout, queries.size());
    auto raw = f.read(partial, (size_t)rows * H * (D + 2));
    std::vector<float> normalized(expected.size());
    constexpr int own = H / 2;
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < H; ++h) {
            const size_t off = (((size_t)(h / own) * rows + r) * own + h % own) * (D + 2);
            if (!(raw[off + D + 1] > 0)) throw std::runtime_error("Invalid attention partial sum");
            for (int d = 0; d < D; ++d)
                normalized[((size_t)r * H + h) * D + d] = raw[off + d] / raw[off + D + 1];
        }
    compare(expected, normalized, 0.00002f, 0.0001f, name);
    paired(q, card, name, pp, split, warmup, repeats);
}
}

int main(int argc, char** argv) {
    try {
        int card = 0, warmup = 5, repeats = 20, query_rows = 0, selected_cells = 0;
        std::string suite = "all";
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            std::string option = argv[i];
            if (option == "--card") card = std::stoi(argv[i + 1]);
            else if (option == "--warmup") warmup = std::stoi(argv[i + 1]);
            else if (option == "--repeats") repeats = std::stoi(argv[i + 1]);
            else if (option == "--suite") suite = argv[i + 1];
            else if (option == "--query-rows") query_rows = std::stoi(argv[i + 1]);
            else if (option == "--selected") selected_cells = std::stoi(argv[i + 1]);
            else throw std::runtime_error("Unknown benchmark option");
        }
        if (card < 0 || card > 1 || warmup < 1 || repeats < 3 ||
            (query_rows != 0 && query_rows != 256 && query_rows != 1024 && query_rows != 4096) ||
            (selected_cells != 0 && selected_cells != 128 && selected_cells != 1024 && selected_cells != 2048) ||
            ((query_rows || selected_cells) && suite != "attention") ||
            (suite != "all" && suite != "attention" && suite != "hc" && suite != "hc-update" &&
             suite != "hc-swap-native" && suite != "hc-swap-flags" && suite != "hc-swap-events" &&
             suite != "hc-swap-own" && suite != "hc-swap-own-serial" &&
             suite != "hc-swap-own-stages" && suite != "attention-copy" && suite != "attention-reuse" &&
             suite != "hc-swap-own-events"))
            throw std::runtime_error("Invalid benchmark options");
        if (suite == "hc-swap-own-events") {
            setenv("STRATA_KPROF", "0", 1);
            if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two GPUs");
            for (int rows : {1, 511, 512, 513, 4095})
                fnx::check_hc_swap("own-events", repeats, rows, fnx::N, 4);
            return 0;
        }
        if (suite == "attention-reuse") {
            setenv("STRATA_KPROF", "0", 1);
            if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two GPUs");
            for (int sender : {0, 1})
                for (int rows : {1, 511, 4095})
                    for (bool ack : {false, true})
                        fnx::check_attention_landing_reuse(ack, sender, repeats, rows);
            return 0;
        }
        if (suite == "attention-copy") {
            setenv("STRATA_KPROF", "0", 1);
            if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two GPUs");
            for (bool direct : {false, true})
                for (int rows : {1, 511, 4095})
                    fnx::check_attention_copy(direct, repeats, rows);
            return 0;
        }
        if (suite == "hc-swap-native" || suite == "hc-swap-flags" || suite == "hc-swap-events" ||
            suite == "hc-swap-own" || suite == "hc-swap-own-serial" ||
            suite == "hc-swap-own-stages") {
            setenv("STRATA_KPROF", "0", 1);
            if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two GPUs");
            if (suite == "hc-swap-own-stages") {
                for (const auto* mode : {"own-stages", "own-stages-direct"})
                    for (int rows : {511, 512, 513, 4095})
                        fnx::check_hc_swap(mode, repeats, rows, fnx::N, 4);
                return 0;
            }
            if (suite == "hc-swap-own-serial") {
                for (int rows : {511, 512, 513, 4095})
                    fnx::check_hc_swap("own-serial", repeats, rows, fnx::N, 4);
                return 0;
            }
            if (suite == "hc-swap-own") {
                for (const auto* mode : {"own-ring", "own-direct"})
                    for (int rows : {511, 512, 513, 4095})
                        fnx::check_hc_swap(mode, repeats, rows, fnx::N, 4);
                return 0;
            }
            const std::string mode = suite.substr(8);
            fnx::check_hc_swap(mode, repeats, 129, 17, 4);
            if (mode == "events") {
                fnx::check_hc_swap(mode, repeats, 3, 17, 8);
                fnx::check_hc_swap(mode, repeats, 4096, fnx::N, 4);
                fnx::check_hc_swap(mode, repeats, 513, fnx::N / 2, 8);
                fnx::check_hc_swap(mode, repeats, 129, fnx::D, 4);
            }
            return 0;
        }
        setenv("STRATA_KPROF", "1", 1);
        int devices = dpct::device_count();
        if (devices < 1 || devices > 2) throw std::runtime_error("Select only GPU0/1");
        dpct::select_device(devices == 1 ? 0 : card);
        sycl::device device = dpct::get_current_device();
        sycl::queue q(sycl::context(device), device,
            sycl::property_list{sycl::property::queue::in_order{},
                                sycl::property::queue::enable_profiling{}});
        std::cout << "DEVICE card=" << card << " name="
                  << q.get_device().get_info<sycl::info::device::name>()
                  << " warmup=" << warmup << " repeats=" << repeats
                  << " split_fixed=" << (std::getenv("STRATA_SPLIT_ATTN_FIXED") ?
                      std::getenv("STRATA_SPLIT_ATTN_FIXED") : "default")
                  << " ABBA no-E2E" << std::endl;
        if (suite == "all" || suite == "attention") {
            for (int rows : {256, 1024, 4096}) {
                if (query_rows && rows != query_rows) continue;
                for (int selected : {128, 1024, 2048}) {
                    if (selected_cells && selected != selected_cells) continue;
                    for (bool sparse : {false, true})
                        attention(q, card, rows, selected, sparse, warmup, repeats);
                }
            }
        }
        if (suite != "attention")
            for (int rows : {256, 1024, 4095, 4096})
                fnx::benchmark_hc(q, card, rows, warmup, repeats, suite == "hc-update");
        std::cout << "PASS kernel-only benchmark card=" << card << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
