// Compares dsv4/ops.cpp against tests/golden.txt (numpy port of the official model.py).  Usage: test_ops tests/golden.txt
#include "dsv4/ops.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static std::map<std::string, std::vector<double>> G;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static bool near(const std::string& name, const std::vector<float>& got, double tol) {
    const auto& ref = G.at(name);
    if (ref.size() != got.size()) { std::fprintf(stderr, "FAIL %s: size %zu vs %zu\n", name.c_str(), got.size(), ref.size()); ++g_fail; return false; }
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - got[i]));
    if (worst > tol) { std::fprintf(stderr, "FAIL %s: max abs diff %.3g > %.3g\n", name.c_str(), worst, tol); ++g_fail; return false; }
    std::printf("ok  %-28s max|diff| = %.3g\n", name.c_str(), worst);
    return true;
}
static void same_ints(const std::string& name, const std::vector<int>& got) {
    const auto& ref = G.at(name);
    bool ok = ref.size() == got.size();
    for (size_t i = 0; ok && i < ref.size(); ++i) ok = (int) ref[i] == got[i];
    if (!ok) { std::fprintf(stderr, "FAIL %s (indices differ)\n", name.c_str()); ++g_fail; } else std::printf("ok  %-28s exact\n", name.c_str());
}
static std::vector<float> F(const std::string& n) { std::vector<float> v; for (double d : G.at(n)) v.push_back((float) d); return v; }

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "tests/golden.txt");
    if (!f) { std::fprintf(stderr, "cannot open golden file\n"); return 2; }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line); std::string name; size_t n; is >> name >> n;
        std::vector<double> v(n); for (auto& x : v) is >> x;
        G[name] = v;
    }
    using namespace dsv4;
    struct GC { const char* n; Score fn; bool bias, hash; } cases[] = {
        {"sqrtsoftplus", Score::SqrtSoftplus, true, false}, {"hash", Score::SqrtSoftplus, false, true},
        {"softmax", Score::Softmax, false, false}, {"sigmoid", Score::Sigmoid, true, false}};
    for (auto& c : cases) {
        const std::string p = std::string("gate_") + c.n;
        auto lg = F(p + "_logits"); std::vector<float> bias; if (c.bias) bias = F(p + "_bias");
        const int32_t h[6] = {17, 3, 200, 41, 99, 250};
        int32_t idx[6]; float w[6];
        gate_route(lg.data(), 256, 6, c.fn, c.bias ? bias.data() : nullptr, c.hash ? h : nullptr, 1.5f, idx, w);
        same_ints(p + "_idx", std::vector<int>(idx, idx + 6));
        near(p + "_w", std::vector<float>(w, w + 6), 1e-6);
    }
    for (const char* tag : {"yarn", "plain"}) {
        std::vector<float> cs, sn;
        const bool y = std::string(tag) == "yarn";
        yarn_table(64, 64, y ? 65536 : 0, y ? 160000.0 : 10000.0, 16.0, 32, 1, cs, sn);
        near(std::string("rope_") + tag + "_cos", cs, 1e-4);
        near(std::string("rope_") + tag + "_sin", sn, 1e-4);
        for (bool inv : {false, true}) {
            auto x = F(std::string("rope_") + tag + "_x");
            rotary(x.data(), 64, 64, &cs[37 * 32], &sn[37 * 32], inv);
            near(std::string("rope_") + tag + (inv ? "_inv" : "_fwd"), x, 1e-4);
        }
    }
    {   // rotary only touches the LAST rope_dim elements of a longer head vector
        std::vector<float> x(128, 1.f), cs(32, 0.f), sn(32, 1.f);  // rotate by 90 degrees
        rotary(x.data(), 128, 64, cs.data(), sn.data(), false);
        CHECK(x[0] == 1.f && x[63] == 1.f && x[64] == -1.f && x[65] == 1.f);
    }
    struct W { const char* n; int w, s, p; } wins[] = {{"win_prefill", 8, 11, 0}, {"win_short", 8, 1, 3}, {"win_wrap", 8, 1, 13}, {"win_exact", 8, 1, 7}};
    for (auto& t : wins) {
        std::vector<int> o; int r, c; window_topk(t.w, t.s, t.p, o, r, c);
        CHECK(G.at(std::string(t.n) + "_shape")[0] == r && G.at(std::string(t.n) + "_shape")[1] == c);
        same_ints(t.n, o);
    }
    struct C { const char* n; int r, s, p, o; } cmps[] = {{"cmp_prefill", 4, 11, 0, 11}, {"cmp_decode", 4, 1, 11, 8}, {"cmp_decode2", 128, 1, 300, 128}};
    for (auto& t : cmps) {
        std::vector<int> o; int r, c; compress_topk(t.r, t.s, t.p, t.o, o, r, c);
        CHECK(G.at(std::string(t.n) + "_shape")[0] == r && G.at(std::string(t.n) + "_shape")[1] == c);
        same_ints(t.n, o);
    }
    {
        auto g = F("swiglu_gate"), u = F("swiglu_up"); std::vector<float> o(64);
        swiglu_clamped(g.data(), u.data(), 64, 10.f, o.data());
        near("swiglu_out", o, 1e-5);
    }
    {   // rmsnorm sanity: unit weights, [3,4] -> rms = sqrt(12.5)
        float x[2] = {3, 4}, y[2];
        rmsnorm(x, nullptr, 0.f, 2, y);
        CHECK(std::fabs(y[0] - 3.f / std::sqrt(12.5f)) < 1e-6f);
    }
    {   // hyper-connections: pre / post / comb / hc_post / hc_head
        const int HC = 4, D = 16;
        auto x = F("hc_x"), fn = F("hc_fn"), sc = F("hc_scale"), bs = F("hc_base"), sub = F("hc_sub");
        std::vector<float> y(D), post(HC), comb(HC * HC), out(HC * D);
        hc_pre(x.data(), HC, D, fn.data(), sc.data(), bs.data(), 1e-6f, 20, 1e-6f, y.data(), post.data(), comb.data());
        near("hc_pre_out", y, 1e-5); near("hc_post_w", post, 1e-5); near("hc_comb", comb, 1e-5);
        hc_post(sub.data(), x.data(), post.data(), comb.data(), HC, D, out.data());
        near("hc_post_out", out, 1e-5);
        // Sinkhorn property: comb is (almost) doubly stochastic
        for (int j = 0; j < HC; ++j) { double r = 0; for (int k = 0; k < HC; ++k) r += comb[j * HC + k]; CHECK(std::fabs(r - 1.0) < 1e-3); }
        auto hf = F("hch_fn"), hs = F("hch_scale"), hb = F("hch_base");
        std::vector<float> hy(D);
        hc_head(x.data(), HC, D, hf.data(), hs.data(), hb.data(), 1e-6f, 1e-6f, hy.data());
        near("hch_out", hy, 1e-5);
    }
    {   // sparse attention with sink and masked (-1) slots
        auto q = F("sa_q"), kv = F("sa_kv"), sk = F("sa_sink"); std::vector<int> ix;
        for (double v : G.at("sa_idx")) ix.push_back((int) v);
        std::vector<float> o(4 * 16);
        sparse_attn_token(q.data(), 4, 16, kv.data(), ix.data(), (int) ix.size(), sk.data(), 0.25f, o.data());
        near("sa_out", o, 1e-5);
        std::vector<int> none(3, -1); std::vector<float> z(4 * 16, 9.f);
        sparse_attn_token(q.data(), 4, 16, kv.data(), none.data(), 3, sk.data(), 0.25f, z.data());
        CHECK(z[0] == 0.f && z[63] == 0.f);
    }
    std::printf("%s\n", g_fail ? "OPS TESTS FAILED" : "OPS TESTS PASSED");
    return g_fail ? 1 : 0;
}
