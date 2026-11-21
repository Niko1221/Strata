#pragma once

#include "collectives.hpp"
#if defined(FNX_DISTRIBUTED)
#include "engine.hpp"
#include "strata/tp/packet.hpp"

namespace fnx::distributed {
enum class Op : uint8_t {
    window, verify, commit, logits, top_tokens, prefill, pass_rows, images,
    add_lane, warm_graphs, use_lane, step, verify_rows, commit_rows, reset,
    save, load, drop, memory, reclaim, churn, lent, draft, draft_lanes,
    callback_end, shutdown
};

inline void write(strata::tp::Packet& p, const Engine::Penalties::Row& row) {
    strata::tp::write(p, row.repeat); strata::tp::write(p, row.freq); strata::tp::write(p, row.present);
    strata::tp::write(p, row.counts);
}
inline void write(strata::tp::Packet& p, const Engine::Penalties& pen) {
    strata::tp::write(p, uint32_t(pen.rows.size()));
    for (const auto& row : pen.rows) write(p, row);
}
inline void write(strata::tp::Packet& p, const Engine::Penalties::Row* row) {
    strata::tp::write(p, bool(row));
    if (row) write(p, *row);
}
inline void write(strata::tp::Packet& p, const Engine::DraftReq& r) {
    strata::tp::write(p, r.lane); strata::tp::write(p, r.T); strata::tp::write(p, r.src_row); strata::tp::write(p, r.pos0);
    strata::tp::write(p, r.from_main); strata::tp::write(p, r.tokens); strata::tp::write(p, bool(r.pen));
    if (r.pen) write(p, *r.pen);
    strata::tp::write(p, r.cands);
}
inline void write(strata::tp::Packet& p, const std::vector<Engine::DraftReq>& reqs) {
    strata::tp::write(p, uint32_t(reqs.size()));
    for (const auto& r : reqs) write(p, r);
}
bool needs_dispatch();
void begin_call(Op op, const std::string& packet);
void end_call();
void callback(const std::function<void()>& body);
void shutdown();
uint64_t save_id();
void release_state(uint64_t id);
void keep_state(const std::shared_ptr<Engine::SeqState>& state, uint64_t id);

class Call {
    bool entered_ = false;
public:
    template<class... Args> Call(Op op, const Args&... args) {
        strata::tp::Packet p;
        if (needs_dispatch()) {
            using strata::tp::write;
            using fnx::distributed::write;
            write(p, op); (write(p, args), ...);
        }
        begin_call(op, p.bytes()); entered_ = true;
    }
    Call(Op op, const std::vector<Engine::Image>& images);
    ~Call() { if (entered_) end_call(); }
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;
};
}
#define FNX_CALL(...) fnx::distributed::Call distributed_call(fnx::distributed::Op::__VA_ARGS__)
#else
#define FNX_CALL(...)
#endif
