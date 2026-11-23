// The sampler of origin/main f65f2d614 (before ink_top_k_ids), verbatim, kept only so ink_sample_topk_test can compare token streams.
// sample_logits and verify_draft_logits, namespace ink2old.
namespace ink2old {
using ink2::SampleOpts;
inline int32_t sample_logits(float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        if (hist.size() > start) {
            size_t from = (hist.size() - start > static_cast<size_t>(s.last_n)) ? (hist.size() - static_cast<size_t>(s.last_n)) : start;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    if (s.greedy()) {
        int32_t best = 0;
        float mx = lg[0];
        for (int i = 1; i < V; ++i) {
            if (lg[i] > mx) { mx = lg[i]; best = i; }
        }
        return best;
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    std::vector<double> p(K);
    double mx = lg[idx[0]], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((lg[idx[i]] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    for (int i = 0; i < keep; ++i) z += p[i];
    std::uniform_real_distribution<double> u(0.0, z);
    const double r = u(rng);
    double c = 0.0;
    for (int i = 0; i < keep; ++i) {
        c += p[i];
        if (r <= c) return idx[i];
    }
    return idx[keep - 1];
}
inline int32_t verify_draft_logits(
    float* lg, int V, const SampleOpts& s, const std::vector<int32_t>& hist,
    std::mt19937_64& rng, int32_t draft_tok, bool& accepted) {
    accepted = false;
    if (s.greedy()) {
        const int32_t tok = ink2old::sample_logits(lg, V, s, hist, rng);   // (qualified: ADL would also find ink2::sample_logits)
        accepted = (tok == draft_tok);
        return tok;
    }
    if (s.penalties() && s.last_n > 0) {
        size_t start = static_cast<size_t>(std::max(0, s.pen_from));
        if (hist.size() > start) {
            size_t from = (hist.size() - start > static_cast<size_t>(s.last_n)) ? (hist.size() - static_cast<size_t>(s.last_n)) : start;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
            for (const auto& kv : cnt) {
                if (kv.first < 0 || kv.first >= V) continue;
                float& x = lg[kv.first];
                if (s.rep != 1.0f) x = (x > 0.0f) ? (x / s.rep) : (x * s.rep);
                x -= static_cast<float>(kv.second) * s.freq + (s.pres != 0.0f ? s.pres : 0.0f);
            }
        }
    }
    const int K = std::min(std::max(1, s.top_k), std::min(V, 128));
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    std::vector<double> p(K);
    double mx = lg[idx[0]], sum = 0.0;
    for (int i = 0; i < K; ++i) {
        p[i] = std::exp((lg[idx[i]] - mx) / static_cast<double>(s.temperature));
        sum += p[i];
    }
    for (double& x : p) x /= sum;
    int keep = K;
    if (s.min_p > 0.0f) {
        const double thr = static_cast<double>(s.min_p) * p[0];
        keep = 1;
        while (keep < K && p[keep] >= thr) ++keep;
    }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        int k2 = 0;
        while (k2 < keep) {
            c += p[k2++];
            if (c >= static_cast<double>(s.top_p)) break;
        }
        keep = std::max(1, k2);
    }
    double z = 0.0;
    int draft_pos = -1;
    for (int i = 0; i < keep; ++i) {
        z += p[i];
        if (idx[i] == draft_tok) draft_pos = i;
    }
    if (z <= 0.0) return idx[0];
    const double p_draft = (draft_pos >= 0) ? (p[draft_pos] / z) : 0.0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (draft_pos >= 0 && (keep == 1 || u01(rng) < p_draft)) {
        accepted = true;
        return draft_tok;
    }
    // Rejected: sample from residual p(x) over i != draft_pos
    const double z_rem = z - (draft_pos >= 0 ? p[draft_pos] : 0.0);
    if (z_rem <= 0.0) {
        accepted = (draft_pos >= 0);
        return (draft_pos >= 0) ? draft_tok : idx[0];
    }
    const double r = u01(rng) * z_rem;
    double c = 0.0;
    int last_valid = idx[0];
    for (int i = 0; i < keep; ++i) {
        if (i == draft_pos) continue;
        last_valid = idx[i];
        c += p[i];
        if (r <= c) return idx[i];
    }
    return last_valid;
}
}  // namespace ink2old
