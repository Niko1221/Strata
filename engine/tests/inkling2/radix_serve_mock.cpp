#define INKLING_DEFINE_STBI
#include "ink_serve.h"

#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <limits>

namespace {

void check(bool condition) {
    if (!condition) {
        std::fprintf(stderr, "mock backend state mismatch\n");
        std::abort();
    }
}

struct Checkpoint : ink2::PrefixCheckpoint {
    explicit Checkpoint(std::vector<int32_t> tokens)
        : PrefixCheckpoint(static_cast<int>(tokens.size())), tokens(std::move(tokens)) {}
    size_t bytes() const override {
        return 400000 + (length > 1024 ? static_cast<size_t>(length) * 500 : 0);
    }
    std::vector<int32_t> tokens;
};

class Backend : public ink2::InkBackend {
public:
    const char* backend_name() const noexcept override { return "mock"; }
    std::string device_info() const override { return {}; }
    ink2::BackendStats stats() const override { return {}; }
    bool init(const ink2::InkModelFile&, const ink2::BackendOpts&, std::string&) override { return true; }
    bool alloc_slot(int, int, ink2::KvDtype, std::string&) override { return true; }
    void free_slot(int) override {}
    void reset_slot(int slot) override { live_[slot].clear(); }
    void snapshot_slot(int) override {}
    void rollback_slot(int, int) override {}
    bool selfcheck(std::string&) override { return true; }

    bool supports_radix_cache() const override { return true; }

    std::unique_ptr<ink2::PrefixCheckpoint> save_prefix(int slot, int length, const ink2::ReservePrefix& reserve) override {
        check(static_cast<size_t>(length) == live_[slot].size());
        auto checkpoint = std::make_unique<Checkpoint>(live_[slot]);
        if (!reserve(checkpoint->bytes(), 0)) return {};
        return checkpoint;
    }

    bool restore_prefix(int slot, const ink2::PrefixCheckpoint& source) override {
        live_[slot] = static_cast<const Checkpoint&>(source).tokens;
        return true;
    }

    bool forward_batch(const ink2::StepBatch& batch, float* logits, ink2::TapCapture*, std::string&) override {
        for (int i = 0; i < batch.n_tokens; ++i) {
            auto& slot = live_[batch.slot_ids[i]];
            check(static_cast<size_t>(batch.pos[i]) == slot.size());
            slot.push_back(batch.token_ids[i]);
            if (batch.want_logits[i]) {
                const auto sum = std::accumulate(slot.begin(), slot.end(), 0u,
                    [](unsigned hash, int32_t token) { return hash * 33 + token; });
                std::fill_n(logits, vocab, -5.0f);
                logits[1 + sum % (vocab - 1)] = 5.0f;
                if (slot.back() == 31) logits[0] = std::numeric_limits<float>::quiet_NaN();
                logits += vocab;
            }
        }
        return true;
    }

    static constexpr int vocab = 32;

private:
    std::vector<int32_t> live_[2];
};

} // namespace

int main(int argc, char** argv) {
    check(argc == 2);
    ink2::InkModelFile model;
    model.m.c.vocab_unpadded = Backend::vocab;
    ink2::ServeConfig cfg;
    cfg.bopts.radix_cache_bytes = std::strtoul(argv[1], nullptr, 10) << 20;
    cfg.bopts.parallel = 2;
    cfg.anti_loop = false;
    Backend backend;
    return ink2::run_serve_loop(cfg, model, backend);
}
