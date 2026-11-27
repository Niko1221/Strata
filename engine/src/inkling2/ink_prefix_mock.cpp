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
    // what a real backend's teardown does to the stack below main (CUDA / driver calls, free lists): it reuses the frames the serve loop had
#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    static void scribble_stack(int depth) {
        volatile unsigned char buf[16384];
        for (size_t i = 0; i < sizeof buf; ++i) buf[i] = 0xA5;
        if (depth > 0) scribble_stack(depth - 1);
        buf[0] = buf[sizeof buf - 1];
    }
public:
    // INK_TEST_SLOW_TEARDOWN_MS: the real backends take seconds to free a model after the serve loop returns (QUIT); the shutdown test uses this
    // to hold that window open while the client has already closed stdin, and the stack below main is overwritten as a real teardown does.
    ~MockBackend() override {
        if (const char* ms = std::getenv("INK_TEST_SLOW_TEARDOWN_MS")) {
            scribble_stack(16);
            std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(ms)));
        }
    }
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
    if (const char* t = std::getenv("INK_TEST_TAIL_ROLE")) cfg.tail_role_token = std::atoll(t);   // --tail-role-token
    int rc;
    {
        MockBackend backend;
        rc = ink2::run_serve_loop(cfg, model, backend);
    }                                   // the backend is torn down here, as in ink_main.cpp
    ink2::leave_after_quit(rc);
}
