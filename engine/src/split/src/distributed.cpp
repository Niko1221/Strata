#include "distributed.hpp"
#include "control.hpp"
#include "strata/tp/mpi_runtime.hpp"
#include "strata/tp/nccl_session.hpp"
#include <exception>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <array>

extern char** environ;

namespace fnx::distributed {
namespace {
strata::tp::MpiRuntime* mpi = nullptr;
strata::tp::NcclSession* nccl = nullptr;
Engine* engine = nullptr;
bool running = false, worker = false;
int depth = 0;
uint64_t next_state = 0, pending_state = 0;
std::map<uint64_t, std::shared_ptr<Engine::SeqState>> states;
std::map<int, std::vector<std::vector<float>>> image_rows;
void execute(const std::string& frame);
void bulk(void* data, size_t bytes) {
    auto* at = static_cast<std::byte*>(data);
    while (bytes) {
        const size_t count = std::min(bytes, strata::tp::max_prompt_bytes);
        mpi->broadcast({at, count}); at += count; bytes -= count;
    }
}
strata::tp::NcclSession& gpu() {
    if (!nccl) throw std::runtime_error("distributed GPU transport is not initialized");
    return *nccl;
}
}
struct Runtime::Impl {
    strata::tp::MpiRuntime control;
    std::optional<strata::tp::NcclSession> tensors;
    bool ready = false;
    Impl(int& argc, char**& argv) : control(argc, argv, std::chrono::seconds(300)) {}
};
Runtime::Runtime(int& argc, char**& argv) {
    if (mpi) throw std::runtime_error("only one distributed runtime can own a process");
    impl_ = std::make_unique<Impl>(argc, argv);
    mpi = &impl_->control;
}
Runtime::~Runtime() {
    states.clear(); image_rows.clear(); engine = nullptr;
    nccl = nullptr;
    mpi = nullptr;
    impl_->tensors.reset();
    if (impl_->ready && !std::uncaught_exceptions()) impl_->control.finish();
}
void Runtime::run(Engine& local) {
    if (running) throw std::runtime_error("distributed engine is already serving");
    engine = &local; running = true;
    if (coordinator()) return;
    worker = true;
    while (running) {
        try { execute(command("")); }
        catch (const Refused&) { /* The coordinator reports the same refused request. */ }
    }
    states.clear(); image_rows.clear();
}
void Runtime::initialize() {
    if (impl_->control.size() != FNX_NCARD) throw std::runtime_error("MPI world size differs from the compiled card count");
    impl_->control.agree("strata-split-distributed-v1;cards=" + std::to_string(FNX_NCARD));
    strata::tp::Packet config;
    std::vector<std::string> keys;
    std::vector<std::pair<std::string, std::string>> settings;
    for (char** item = environ; *item; ++item) {
        const std::string entry(*item);
        if (entry.rfind("FNX_", 0) != 0) continue;
        const auto equals = entry.find('=');
        keys.push_back(entry.substr(0, equals));
        settings.emplace_back(entry.substr(0, equals), entry.substr(equals + 1));
    }
    if (coordinator()) strata::tp::write(config, settings);
    const auto bytes = command(config.bytes());
    if (!coordinator()) {
        for (const auto& key : keys) unsetenv(key.c_str());
        strata::tp::PacketReader r(bytes);
        const auto count = r.count();
        for (uint32_t i = 0; i < count; ++i) {
            const auto key = r.string(); const auto value = r.string();
            if (setenv(key.c_str(), value.c_str(), 1)) throw std::runtime_error("cannot apply distributed engine setting");
        }
        r.finish();
    }
    auto require = [](const char* key, const char* value) {
        const char* current = std::getenv(key);
        if (current && std::string(current) != value)
            throw std::runtime_error(std::string(key) + " must be " + value + " for distributed resident execution");
        if (setenv(key, value, 1)) throw std::runtime_error("cannot configure distributed engine");
    };
    require("FNX_ARENA", "0");
    require("FNX_RING_F32", "1");
    require("FNX_EXPERT_RAM_GIB", "off");
    impl_->tensors.emplace(impl_->control, std::chrono::seconds(300));
    nccl = &*impl_->tensors;
    impl_->ready = true;
}
bool enabled() { return mpi != nullptr; }
int first_card() { return mpi ? mpi->rank() : 0; }
int end_card() { return mpi ? mpi->rank() + 1 : FNX_NCARD; }
bool coordinator() { return !mpi || mpi->rank() == 0; }
void sum(float* values, size_t count, void* stream) { gpu().sum(values, count, static_cast<cudaStream_t>(stream)); }
void gather(const float* input, float* output, size_t count, void* stream) {
    gpu().gather(input, output, count, static_cast<cudaStream_t>(stream));
}
void exchange(const float* input, float* output, size_t stride, size_t offset, size_t count, void* stream) {
    gpu().exchange(input, output, stride, offset, count, static_cast<cudaStream_t>(stream));
}
uint64_t minimum(uint64_t local) { return mpi ? mpi->minimum(local) : local; }
void gather_host(const void* local, size_t bytes, void* result, const std::vector<int>& sizes) {
    if (!mpi) throw std::runtime_error("distributed host gather has no runtime");
    size_t total = 0;
    for (int size : sizes) {
        if (size < 0 || size_t(size) > 128 * 1024 * 1024 - total) throw std::runtime_error("host gather exceeds 128 MiB");
        total += size;
    }
    mpi->gather({static_cast<const std::byte*>(local), bytes}, {static_cast<std::byte*>(result), total}, sizes);
}
void gather_rows(void* values, size_t stride, size_t bytes) {
    if (!mpi || mpi->size() == 1) return;
    if (!values || bytes > stride || bytes > 128 * 1024 * 1024 / size_t(mpi->size()))
        throw std::runtime_error("invalid distributed host row layout");
    auto* base = static_cast<std::byte*>(values);
    std::vector<int> sizes(mpi->size(), int(bytes));
    if (bytes == stride) { gather_host(base + size_t(mpi->rank()) * stride, bytes, base, sizes); return; }
    std::vector<std::byte> packed(bytes * mpi->size());
    gather_host(base + size_t(mpi->rank()) * stride, bytes, packed.data(), sizes);
    for (int rank = 0; rank < mpi->size(); ++rank)
        std::memcpy(base + size_t(rank) * stride, packed.data() + size_t(rank) * bytes, bytes);
}
std::string command(const std::string& root) {
    if (!mpi) throw std::runtime_error("distributed control has no runtime");
    return mpi->command(root);
}

unsigned results(float* values, int* ids, unsigned* errors, int stride, int rows, float& gpu_ms, double* card_ms) {
    struct Result {
        std::array<float, MAXT> values{};
        std::array<int, MAXT> ids{};
        unsigned error = 0;
        float ms = 0;
    };
    std::vector<Result> all(FNX_NCARD);
    const int rank = first_card();
    auto& local = all[rank];
    std::copy_n(values + rank * stride, rows, local.values.begin());
    std::copy_n(ids + rank * stride, rows, local.ids.begin());
    local.error = errors[rank]; local.ms = gpu_ms;
    gather_rows(all.data(), sizeof(Result), sizeof(Result));
    unsigned error = 0; gpu_ms = 0;
    for (int c = 0; c < FNX_NCARD; ++c) {
        std::copy_n(all[c].values.begin(), rows, values + c * stride);
        std::copy_n(all[c].ids.begin(), rows, ids + c * stride);
        errors[c] = all[c].error; error |= errors[c];
        gpu_ms = std::max(gpu_ms, all[c].ms);
        if (card_ms) card_ms[c] = all[c].ms;
    }
    return error;
}

bool needs_dispatch() { return running && !worker && depth == 0; }
void begin_call(Op, const std::string& packet) {
    if (needs_dispatch()) command(packet);
    ++depth;
}
void end_call() { --depth; }
void pass_rows(int rows) { Call call(Op::pass_rows, rows); }
void shutdown() {
    if (!running || worker) return;
    strata::tp::Packet p; strata::tp::write(p, Op::shutdown); command(p.bytes()); running = false;
}
uint64_t save_id() { return needs_dispatch() ? ++next_state : worker && depth == 0 ? pending_state : 0; }
void keep_state(const std::shared_ptr<Engine::SeqState>& state, uint64_t id) { state->distributed_id = id; }
void release_state(uint64_t id) {
    if (id && needs_dispatch()) { Call call(Op::drop, id); }
}

Call::Call(Op op, const std::vector<Engine::Image>& images) {
    const bool dispatch = needs_dispatch();
    strata::tp::Packet p;
    if (dispatch) {
        strata::tp::write(p, op); strata::tp::write(p, uint32_t(images.size()));
        for (const auto& im : images) {
            if (im.n < 0 || im.n > engine->max_ctx()) throw Refused("image cells exceed context");
            strata::tp::write(p, im.cell); strata::tp::write(p, im.n);
            strata::tp::write(p, im.nx); strata::tp::write(p, im.ny); strata::tp::write(p, bool(im.rows));
        }
    }
    begin_call(op, p.bytes()); entered_ = true;
    if (dispatch) for (const auto& im : images)
        if (im.rows) bulk(const_cast<float*>(im.rows), size_t(im.n) * N * sizeof(float));
}

void callback(const std::function<void()>& body) {
    if (!running) { body(); return; }
    const int previous = depth; depth = 0;
    try {
        if (coordinator()) {
            bool failed = true;
            auto finish = [&] {
                strata::tp::Packet p; strata::tp::write(p, Op::callback_end); strata::tp::write(p, failed); command(p.bytes());
            };
            try { body(); failed = false; }
            catch (...) { finish(); throw; }
            finish();
        } else {
            for (;;) {
                const auto bytes = command("");
                strata::tp::PacketReader r(bytes);
                if (r.scalar<Op>() == Op::callback_end) {
                    const bool failed = r.boolean(); r.finish();
                    if (failed) throw Refused("coordinator callback ended the request");
                    break;
                }
                execute(bytes);
            }
        }
    } catch (...) { depth = previous; throw; }
    depth = previous;
}

namespace {
Engine::Penalties::Row read_row(strata::tp::PacketReader& r) {
    Engine::Penalties::Row row;
    row.repeat = r.scalar<float>(); row.freq = r.scalar<float>(); row.present = r.scalar<float>();
    const auto n = r.count(); row.counts.reserve(n);
    for (uint32_t i = 0; i < n; ++i) { const int id = r.scalar<int>(), count = r.scalar<int>(); row.counts.emplace_back(id, count); }
    return row;
}
Engine::Penalties read_penalties(strata::tp::PacketReader& r) {
    Engine::Penalties p; const auto n = r.count(); p.rows.reserve(n);
    for (uint32_t i = 0; i < n; ++i) p.rows.push_back(read_row(r));
    return p;
}
void execute(const std::string& frame) {
    strata::tp::PacketReader r(frame);
    const Op op = r.scalar<Op>();
    switch (op) {
        case Op::window: { auto v = r.vector<int>(); r.finish(); engine->window(v); break; }
        case Op::verify: { auto v = r.vector<int>(); r.finish(); engine->verify(v); break; }
        case Op::commit: { int n = r.scalar<int>(); r.finish(); engine->commit(n); break; }
        case Op::logits: { int n = r.scalar<int>(); r.finish(); engine->last_logits(n); break; }
        case Op::top_tokens: { int n = r.scalar<int>(); auto p = read_penalties(r); r.finish(); engine->top_tokens(n, p); break; }
        case Op::prefill: { auto v = r.vector<int>(); int after = r.scalar<int>(); r.finish(); engine->prefill(v, after); break; }
        case Op::pass_rows: { int n = r.scalar<int>(); r.finish(); engine->set_pass_rows(n); break; }
        case Op::images: {
            const auto n = r.count();
            std::vector<Engine::Image> images(n); std::vector<bool> has_rows(n);
            std::vector<std::vector<float>> rows(n);
            for (uint32_t i = 0; i < n; ++i) {
                auto& im = images[i]; im.cell = r.scalar<int>(); im.n = r.scalar<int>();
                im.nx = r.scalar<int>(); im.ny = r.scalar<int>(); has_rows[i] = r.boolean();
                if (im.n < 0 || im.n > engine->max_ctx()) throw std::runtime_error("image payload exceeds context");
            }
            r.finish();
            for (uint32_t i = 0; i < n; ++i) if (has_rows[i]) {
                rows[i].resize(size_t(images[i].n) * N);
                bulk(rows[i].data(), rows[i].size() * sizeof(float)); images[i].rows = rows[i].data();
            }
            image_rows[engine->lane()] = std::move(rows); engine->set_images(images); break;
        }
        case Op::add_lane: r.finish(); engine->add_lane(); break;
        case Op::warm_graphs: r.finish(); engine->warm_graphs(); break;
        case Op::use_lane: { int n = r.scalar<int>(); r.finish(); engine->use_lane(n); break; }
        case Op::step: {
            auto lanes = r.vector<int>(); auto tokens = r.vector<int>(); r.finish(); engine->step(lanes, tokens); break;
        }
        case Op::verify_rows: {
            auto lanes = r.vector<int>(); auto counts = r.vector<int>(); auto tokens = r.vector<int>();
            r.finish(); engine->verify_rows(lanes, counts, tokens); break;
        }
        case Op::commit_rows: { auto v = r.vector<int>(); r.finish(); engine->commit_rows(v); break; }
        case Op::reset: r.finish(); engine->reset(); image_rows.erase(engine->lane()); break;
        case Op::save: {
            const bool kv = r.boolean(); pending_state = r.scalar<uint64_t>(); r.finish();
            states[pending_state] = engine->save_state(kv); pending_state = 0; break;
        }
        case Op::load: {
            const auto id = r.scalar<uint64_t>(); r.finish();
            engine->load_state(*states.at(id)); image_rows.erase(engine->lane()); break;
        }
        case Op::drop: { const auto id = r.scalar<uint64_t>(); r.finish(); states.erase(id); break; }
        case Op::memory: r.finish(); engine->free_vram_mib(); break;
        case Op::reclaim: { const double hold = r.scalar<double>(); r.finish(); engine->arena_reclaim(hold); break; }
        case Op::churn: r.finish(); engine->arena_churn(); break;
        case Op::lent: r.finish(); engine->arena_lent(); break;
        case Op::draft: {
            const int T = r.scalar<int>(); const bool main = r.boolean(); const int row = r.scalar<int>();
            auto tokens = r.vector<int>(); const int pos = r.scalar<int>();
            const bool has_pen = r.boolean(); Engine::Penalties::Row pen;
            if (has_pen) pen = read_row(r);
            const bool cands = r.boolean(); r.finish();
            engine->draft(T, main, row, tokens, pos, has_pen ? &pen : nullptr, cands); break;
        }
        case Op::draft_lanes: {
            const auto n = r.count(); std::vector<Engine::DraftReq> reqs(n); std::vector<Engine::Penalties::Row> penalties(n);
            for (uint32_t i = 0; i < n; ++i) {
                auto& q = reqs[i]; q.lane = r.scalar<int>(); q.T = r.scalar<int>(); q.src_row = r.scalar<int>(); q.pos0 = r.scalar<int>();
                q.from_main = r.boolean(); q.tokens = r.vector<int>();
                if (r.boolean()) { penalties[i] = read_row(r); q.pen = &penalties[i]; }
                q.cands = r.boolean();
            }
            r.finish(); engine->draft_lanes(reqs); break;
        }
        case Op::shutdown: r.finish(); running = false; break;
        default: throw std::runtime_error("unknown distributed engine operation");
    }
}
}
}
