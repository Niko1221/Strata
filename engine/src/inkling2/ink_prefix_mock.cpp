// Exercise the real serve scheduler with recurrent state and append-only global KV, without a GPU.
#include "ink_serve.h"
#include <cstdlib>

namespace {
void require(bool ok) { if (!ok) std::abort(); }
class MockBackend final : public ink2::InkBackend {
    struct Checkpoint { std::vector<int32_t> prefix; uint64_t recurrent = 0; };
    std::vector<int32_t> global[2];
    uint64_t recurrent[2] = {};
    int position[2] = {};
    Checkpoint prompt[2], shared[2];
    // INK_TEST_CKPT=1: the backend also has the checkpoints of the chunked prefix cache (ckpt_save: any number per store, by handle)
    const bool marks_on = std::getenv("INK_TEST_CKPT") != nullptr;
    struct Mark { bool used = false; std::vector<int32_t> prefix; uint64_t recurrent = 0; };
    std::vector<Mark> marks;
    bool save(int s, int n, Checkpoint& ck) {
        require(position[s] == n);
        if (std::getenv("INK_TEST_NO_CHECKPOINT")) return false;
        ck.prefix.assign(global[s].begin(), global[s].begin() + n);
        ck.recurrent = recurrent[s];
        return true;
    }
    bool restore(int s, int n, const Checkpoint& ck) {
        if (ck.prefix.empty() || static_cast<int>(ck.prefix.size()) != n) return false;
        // A snapshot of recurrent state alone cannot repair overwritten global KV.
        require(global[s].size() >= ck.prefix.size());
        require(std::equal(ck.prefix.begin(), ck.prefix.end(), global[s].begin()));
        position[s] = n;
        recurrent[s] = ck.recurrent;
        return true;
    }
public:
    const char* backend_name() const noexcept override { return "prefix-test"; }
    std::string device_info() const override { return "mock"; }
    ink2::BackendStats stats() const override { return {}; }
    bool init(const ink2::InkModelFile&, const ink2::BackendOpts&, std::string&) override { return true; }
    bool alloc_slot(int s, int, ink2::KvDtype, std::string&) override { reset_slot(s); return true; }
    void free_slot(int s) override { reset_slot(s); }
    void reset_slot(int s) override { global[s].clear(); recurrent[s] = 0; position[s] = 0; prompt[s] = {}; shared[s] = {}; }
    void snapshot_slot(int) override {}
    void rollback_slot(int, int) override { std::abort(); }
    bool selfcheck(std::string&) override { return true; }
    bool save_prompt_checkpoint(int s, int n) override { return save(s, n, prompt[s]); }
    bool restore_prompt_checkpoint(int s, int n) override { return restore(s, n, prompt[s]); }
    bool supports_shared_checkpoint() const override { return !std::getenv("INK_TEST_NO_CHECKPOINT"); }
    bool save_shared_checkpoint(int s, int n) override { return save(s, n, shared[s]); }
    bool restore_shared_checkpoint(int s, int n) override { return restore(s, n, shared[s]); }
    size_t ckpt_bytes() const override { return marks_on ? size_t(1) << 20 : 0; }
    int ckpt_save(int s) override {
        if (!marks_on) return -1;
        size_t h = 0;
        while (h < marks.size() && marks[h].used) ++h;
        if (h == marks.size()) marks.emplace_back();
        marks[h] = {true, std::vector<int32_t>(global[s].begin(), global[s].begin() + position[s]), recurrent[s]};
        return static_cast<int>(h);
    }
    bool ckpt_restore(int s, int h) override {
        require(h >= 0 && h < static_cast<int>(marks.size()) && marks[h].used);
        const Mark& m = marks[h];
        require(global[s].size() >= m.prefix.size());      // (as restore above: the KV up to the mark must still be the mark's)
        require(std::equal(m.prefix.begin(), m.prefix.end(), global[s].begin()));
        position[s] = static_cast<int>(m.prefix.size());
        recurrent[s] = m.recurrent;
        return true;
    }
    void ckpt_free(int h) override {
        require(h >= 0 && h < static_cast<int>(marks.size()) && marks[h].used);
        marks[h] = {};
    }
    bool kv_copy_prefix(int d, int s, int n) override {
        require(static_cast<int>(global[s].size()) >= n);
        if (static_cast<int>(global[d].size()) < n) global[d].resize(n);
        std::copy(global[s].begin(), global[s].begin() + n, global[d].begin());
        return true;
    }
    bool forward_batch(const ink2::StepBatch& b, float* logits, ink2::TapCapture*, std::string&) override {
        int row = 0;
        for (int i = 0; i < b.n_tokens; ++i) {
            int s = b.slot_ids[i], p = b.pos[i];
            require(position[s] == p);
            global[s].resize(p + 1);
            global[s][p] = b.token_ids[i];
            recurrent[s] = recurrent[s] * 131 + b.token_ids[i];
            ++position[s];
            if (b.want_logits[i]) {
                std::fill(logits + row * 64, logits + (row + 1) * 64, -10.0f);
                logits[row * 64 + 1 + recurrent[s] % 62] = 10.0f;
                ++row;
            }
        }
        return true;
    }
};
}
int main() {
    ink2::InkModelFile model;
    model.m.c.vocab_unpadded = 64;
    model.m.c.eos_token_id = 63;
    model.m.c.eot_token_id = 63;
    ink2::ServeConfig cfg;
    cfg.bopts.parallel = 2;
    cfg.bopts.chunk = 64;
    cfg.bopts.max_ctx = 32768;
    cfg.draft_mode = "off";
    MockBackend backend;
    return ink2::run_serve_loop(cfg, model, backend);
}
