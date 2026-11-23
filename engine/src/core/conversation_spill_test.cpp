// The conversation cache's SSD tier: its folder per engine process and its budget (conversation_spill.hpp).  No GPU, no model.
#include "strata/core/conversation_spill.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <iterator>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
constexpr uint64_t GiB = uint64_t(1) << 30;
void touch(const fs::path& p) { std::ofstream(p) << "x"; }
}  // namespace

int main() {
    // the budget: min(32 GiB, 25% of the free space), never into the floor
    check(spill_default_budget(400 * GiB, 4 * GiB) == 32 * GiB, "400 GiB free: 32 GiB");
    check(spill_default_budget(40 * GiB, 4 * GiB) == 10 * GiB, "40 GiB free: a quarter, 10 GiB");
    check(spill_default_budget(5 * GiB, 4 * GiB) == 1 * GiB, "5 GiB free, floor 4: the 1 GiB above the floor");
    check(spill_default_budget(4 * GiB, 4 * GiB) == 0 && spill_default_budget(3 * GiB, 4 * GiB) == 0, "at or under the floor: none");
    SpillBudget b{10 * GiB, 0, 0};
    check(b.reserve(6 * GiB) && !b.reserve(5 * GiB) && b.reserve(4 * GiB), "reservations count until the writes end");
    b.finish(6 * GiB, 5 * GiB);
    check(b.used == 5 * GiB && b.reserved == 4 * GiB && !b.over(), "a write ends: its reservation goes, its bytes stay");
    b.finish(4 * GiB, 0);
    check(b.used == 5 * GiB && b.reserved == 0, "a failed write leaves nothing");
    check(!b.reserve(6 * GiB) && b.reserve(5 * GiB), "the cap includes what is written");
    b.finish(5 * GiB, 6 * GiB);
    check(b.over(), "a file larger than its reservation shows as over the cap");
    b.drop(6 * GiB);
    check(!b.over() && b.used == 5 * GiB, "a dropped entry gives its bytes back");
    check(!SpillBudget{0, 0, 0}.reserve(1), "no budget: nothing");

    // the names a clean may touch
    check(spill_detail::is_namespace_name("123-0123456789abcdef"), "pid-16 hex");
    for (const char* n : {"123-0123456789ABCDEF", "123-0123456789abcde", "-0123456789abcdef", "12a-0123456789abcdef",
                          "notes", "123-0123456789abcdef0", "123_0123456789abcdef"})
        check(!spill_detail::is_namespace_name(n), n);

    // the folders: two live engines on one root keep each other's folder; a dead one's goes; nothing else is touched
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(std::random_device{}()));
    {   // --conversation-spill-persist: a stable folder per model fingerprint; entries survive a restart, LRU at the cap, a grown conversation replaces its prefix
        auto img = [](std::vector<int32_t> ids, uint8_t fill) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 20, fill);
            s.cvec = true;
            return s;
        };
        const fs::path proot = fs::temp_directory_path() / ("strata-spill-persist-" + std::to_string(std::random_device{}()));
        fs::remove_all(proot, ec);
        const SessionFileIdentity A{0xa1, 5}, B{0xb2, 5}, A2{0xa1, 6};
        const uint64_t cap = uint64_t(5) << 20;   // three entries (about 1 MiB each, 2 MiB reserved per write)
        std::string e;
        size_t cleaned = 0, adopted = 99;
        {   // run 0: another model leaves an entry
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &B, &adopted) && s.persistent() && adopted == 0, "persist: a first run on a new root");
            check(s.put(img({90, 91}, 1), B, e) && s.size() == 1, "persist: model B's entry");
        }
        check(fs::exists(proot / spill_detail::persist_name(B.model)), "persist: the stable folder outlives the run");
        {   // run 1: model A
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &A, &adopted) && s.persistent() && adopted == 0 && cleaned == 1, "persist: model A's start removes B's folder");
            check(!fs::exists(proot / spill_detail::persist_name(B.model)) && s.dir().filename() == spill_detail::persist_name(A.model), "persist: B's folder gone, A's is fp-<hex>");
            check(s.put(img({1, 2, 3, 4}, 7), A, e) && s.put(img({10, 11}, 8), A, e) && s.put(img({20, 21}, 9), A, e) && s.size() == 3, "persist: three entries written");
            ConversationSpill other;   // a second live engine of the same model: falls back to its own per-process folder
            check(other.open(proot, cap, e, nullptr, &A, &adopted) && !other.persistent() && adopted == 0 && other.dir() != s.dir(), "persist: a second engine of the model gets its own folder");
            other.close();
        }
        {   // run 2: a restart, same fingerprint
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &A, &adopted) && adopted == 3 && s.size() == 3 && s.bytes() > 0, "restart: the three entries are adopted");
            auto m = s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
            check(m.tokens == 4 && m.live, "restart: found by token prefix");
            SavedConversation back;
            check(s.take_serial(m.serial, A, {}, back, e) && back.live.ids == std::vector<int32_t>({1, 2, 3, 4}) &&
                  back.live.gdn == std::vector<uint8_t>(1 << 20, 7), "restart: read back byte for byte");
            check(s.size() == 3 && s.find(m.serial), "a read leaves the entry in place");
            // least recently USED goes at the cap: {10,11} was written after {1,2,3,4} but the read made {1,2,3,4} the freshest
            check(s.put(img({30, 31}, 4), A, e) && s.size() == 3, "LRU: a fourth at the cap");
            check(s.best(std::vector<int32_t>{10, 11, 0}, {}, true).tokens == 0, "LRU: the least recently used ({10,11}) went");
            check(s.best(std::vector<int32_t>{1, 2, 3, 4, 0}, {}, true).tokens == 4 && s.best(std::vector<int32_t>{20, 21, 0}, {}, true).tokens == 2 &&
                  s.best(std::vector<int32_t>{30, 31, 0}, {}, true).tokens == 2, "LRU: the read one and the others stay");
            // supersede: the same conversation grown replaces its prefix; an entry that is no prefix is never removed for it
            const auto old = s.best(std::vector<int32_t>{1, 2, 3, 4, 0}, {}, true);
            const fs::path oldf = s.find(old.serial)->file;
            check(s.put(img({1, 2, 3, 4, 5, 6}, 6), A, e) && s.size() == 3 && !fs::exists(oldf) && !s.find(old.serial), "supersede: P+Q replaces P");
            check(s.best(std::vector<int32_t>{20, 21, 0}, {}, true).tokens == 2 && s.best(std::vector<int32_t>{30, 31, 0}, {}, true).tokens == 2,
                  "supersede: entries that are no prefix stay");
            check(s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 0}, {}, true).tokens == 6, "supersede: the grown one serves the resume");
        }
        {   // the same model, other settings: the files are not this run's
            ConversationSpill s;
            check(s.open(proot, cap, e, nullptr, &A2, &adopted) && adopted == 0 && s.size() == 0, "restart: another config fingerprint adopts nothing");
        }
        {   // restart: another model fingerprint
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &B, &adopted) && adopted == 0 && s.size() == 0 && cleaned == 1, "restart: another model fingerprint deletes the entries");
            check(!fs::exists(proot / spill_detail::persist_name(A.model)), "restart: the old folder is gone");
        }
        {   // a damaged index file: that entry is dropped, the others are adopted
            ConversationSpill s;
            check(s.open(proot, cap, e, nullptr, &A, &adopted) && s.put(img({1, 2}, 1), A, e) && s.put(img({3, 4}, 1), A, e), "damage: two entries");
            const fs::path f0 = s.entry(0).file;
            s.close();
            fs::resize_file(fs::path(f0).replace_extension(".meta"), 9, ec);
            ConversationSpill t;
            check(t.open(proot, cap, e, nullptr, &A, &adopted) && adopted == 1 && !fs::exists(f0), "damage: a cut index file drops its entry");
        }
        fs::remove_all(proot, ec);
    }
    fs::remove_all(root, ec);
    {
        SpillNamespace a, b2;
        std::string err;
        check(a.open(root, err) && a.is_open() && fs::is_directory(a.dir()), "the first engine's folder");
        touch(a.file_for(7));
        const fs::path dead = root / "4242-0123456789abcdef";
        fs::create_directories(dead);
        touch(dead / "conv-1.session");                               // a crashed engine's file, no lock file left
        const fs::path dead2 = root / "4243-fedcba9876543210";
        fs::create_directories(dead2);
        touch(dead2 / ".lock");                                       // a crashed engine's lock file, nobody holds it
        fs::create_directories(root / "keepme");
        touch(root / "keepme" / "conv-1.session");
        touch(root / "notes.txt");
        size_t cleaned = 99;
        check(b2.open(root, err, &cleaned), "a second engine on the same root");
        check(cleaned == 2 && !fs::exists(dead) && !fs::exists(dead2), "the two dead folders went");
        check(fs::exists(a.file_for(7)), "the first engine's live folder and file stay");
        check(fs::exists(root / "keepme" / "conv-1.session") && fs::exists(root / "notes.txt"), "nothing else under the root is touched");
        check(a.dir() != b2.dir() && a.file_for(7) != b2.file_for(7), "each engine writes its own files");
#if !defined(_WIN32)
        const fs::path outside = fs::temp_directory_path() / ("strata-spill-outside-" + std::to_string(std::random_device{}()));
        fs::create_directories(outside);
        touch(outside / "keep.txt");
        fs::create_directory_symlink(outside, root / "4244-0123456789abcdef", ec);
        check(!ec && SpillNamespace::clean_dead(root) == 0 && fs::exists(outside / "keep.txt"), "a link named like a folder is never followed");
        fs::remove(root / "4244-0123456789abcdef", ec);
        fs::remove_all(outside, ec);
#endif
        const fs::path adir = a.dir();
        a.close();
        check(!fs::exists(adir) && fs::exists(b2.dir()), "an orderly end removes its own folder only");
    }
    check(fs::exists(root) && !fs::exists(root / "4242-0123456789abcdef"), "both engines ended");
    std::string err;
    check(!SpillNamespace().open(root / "notes.txt" / "sub", err) && !err.empty(), "a root that cannot be a folder is refused with a reason");

    // the entries: written as session files, the oldest go first, matched like the RAM cache, read back exactly
    {
        auto img = [](std::vector<int32_t> ids, uint8_t fill, std::vector<int32_t> ckpt = {}) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 20, fill);                  // 1 MiB of running state
            s.cvec = true;
            if (!ckpt.empty()) { ConversationCheckpoint c; c.ids = ckpt; c.gdn.assign(4096, (uint8_t) (fill + 1)); s.checkpoints.push_back(c); }
            return s;
        };
        const SessionFileIdentity id{11, 22};
        ConversationSpill sp;
        // room for two entries of ~2 MiB each (1 MiB payload + the 1 MiB margin per write)
        check(sp.open(root, (uint64_t(7) << 19), err) && sp.enabled(), "the tier opens");
        check(sp.put(img({1, 2, 3, 4}, 7, {1, 2}), id, err) && sp.put(img({5, 6, 7}, 9), id, err) && sp.size() == 2, "two entries written");
        const fs::path first = sp.entry(0).file;
        check(fs::exists(first) && fs::exists(sp.entry(1).file), "their files exist");
        check(sp.bytes() == sp.entry(0).bytes + sp.entry(1).bytes && sp.budget().reserved == 0, "the budget counts what is written");
        check(sp.put(img({8, 9}, 3), id, err) && sp.size() == 2 && !fs::exists(first), "a third pushes out the oldest, its file deleted");
        auto m = sp.best(std::vector<int32_t>{5, 6, 7, 1}, {}, true);
        check(m.tokens == 3 && m.live && sp.entry(m.index).live.ids == std::vector<int32_t>({5, 6, 7}), "a prompt finds the entry it continues");
        check(sp.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a dropped entry is not found");
        check(sp.best(std::vector<int32_t>{5, 6, 7, 1}, {}, false).tokens == 0, "other steering: not found");
        SavedConversation back;
        const fs::path taken = sp.entry(m.index).file;
        check(sp.take(m.index, id, {}, back, err) && back.live.ids == std::vector<int32_t>({5, 6, 7}) &&
              back.live.gdn == std::vector<uint8_t>(1 << 20, 9), "taken back byte for byte");
        check(sp.size() == 1 && !fs::exists(taken) && sp.bytes() == sp.entry(0).bytes, "a taken entry leaves the tier");
        // a checkpoint match
        check(sp.put(img({20, 21, 22, 23}, 5, {20, 21}), id, err), "an entry with a checkpoint");
        auto mc = sp.best(std::vector<int32_t>{20, 21, 99}, {}, true);
        check(mc.tokens == 2 && !mc.live, "a prompt that shares only the checkpoint finds it");
        // another model's identity: refused, the entry goes, out unchanged
        SavedConversation keep = img({1}, 1);
        check(!sp.take(mc.index, SessionFileIdentity{11, 23}, {}, keep, err) && keep.live.ids == std::vector<int32_t>({1}), "another identity: refused");
        check(sp.size() == 1, "the refused entry is gone");
        // a cut file: refused by the bounded read
        {
            const fs::path f = sp.entry(0).file;
            fs::resize_file(f, fs::file_size(f) / 2, ec);
            SavedConversation x;
            check(!ec && !sp.take(0, id, {}, x, err) && sp.size() == 0 && !fs::exists(f), "a cut file is refused and removed");
        }
        // larger than the whole tier: refused, nothing written
        ConversationSpill small;
        std::string e2;
        check(small.open(root, 1 << 20, e2) && !small.put(img({1, 2}, 4), id, e2) && !e2.empty(), "an image larger than the tier is refused");
        size_t files = 0;
        for (auto& x : fs::directory_iterator(small.dir())) if (x.path().filename() != ".lock") ++files;
        check(files == 0, "nothing of a refused put is left on the disk");
        small.close();
        // #879: clear() drops every entry and its file, the tier stays open
        check(sp.put(img({30, 31}, 2), id, err) && sp.put(img({40, 41}, 2), id, err) && sp.size() == 2, "two more entries");
        const fs::path c0 = sp.entry(0).file, c1 = sp.entry(1).file;
        check(sp.clear() == 2 && sp.size() == 0 && sp.bytes() == 0 && !fs::exists(c0) && !fs::exists(c1), "clear() drops them and their files");
        check(sp.enabled() && fs::is_directory(sp.dir()) && sp.put(img({50}, 1), id, err) && sp.size() == 1, "the tier stays open after clear()");
        sp.close();
    }
    {   // the cost rule, with l620's measured speeds (a 236 MB image written in 120 ms, read back in 220 ms; 5 prompt tokens per ms)
        SpillCost c;
        check(c.reread_tokens_per_ms == 2.0 && c.round_trip_ms(2000000) == 4.0, "the defaults: 2 tokens per ms, 1 GB/s each way");
        for (int i = 0; i < 40; ++i) {
            c.saw_prompt(8000, 1600.0);
            c.saw_write(236000000, 120.0);
            c.saw_read(236000000, 220.0);
        }
        check(c.reread_tokens_per_ms > 4.99 && c.reread_tokens_per_ms < 5.01, "the prompt speed converges to the measured one");
        check(c.round_trip_ms(236000000) > 339.0 && c.round_trip_ms(236000000) < 341.0, "the round trip converges to 120 + 220 ms");
        check(!c.worth(51, 236000000), "a 51-token conversation is not worth the disk (re-read about 10 ms of tokens)");
        check(!c.worth(2000, 264000000), "a 2K one is not either here (400 ms against about 1.25 x 380 ms)");
        check(c.worth(8000, 350000000), "an 8K one is (1600 ms against about 1.25 x 504 ms)");
        const SpillCost before = c;
        c.saw_prompt(100, 1.0);       // fewer than kMinTokens: a request's fixed costs, not the prompt speed
        c.saw_prompt(4000, 0.0);      // no time: ignored
        c.saw_write(0, 5.0);          // nothing written: ignored
        c.saw_read(1000, -1.0);       // a clock that went back: ignored
        check(c.reread_tokens_per_ms == before.reread_tokens_per_ms && c.write_bytes_per_ms == before.write_bytes_per_ms &&
              c.read_bytes_per_ms == before.read_bytes_per_ms, "samples that say nothing about the speeds are ignored");
        SpillCost slow;   // a card that reads prompts at 1 token per ms: the same 2K conversation is worth spilling
        for (int i = 0; i < 40; ++i) { slow.saw_prompt(2000, 2000.0); slow.saw_write(264000000, 130.0); slow.saw_read(264000000, 250.0); }
        check(slow.worth(2000, 264000000), "on a slower card a 2K conversation is worth the disk");
        check(!c.worth(0, 1) && !slow.worth(0, 1), "an empty conversation never is");
    }
    {   // 2c: the background writer
        using namespace std::chrono_literals;
        struct Gate {   // holds the writer inside a write until opened
            std::mutex mu; std::condition_variable cv; bool open = false; int entered = 0;
            void enter() { std::unique_lock<std::mutex> lk(mu); ++entered; cv.notify_all(); cv.wait(lk, [&] { return open; }); }
            void release() { { std::lock_guard<std::mutex> lk(mu); open = true; } cv.notify_all(); }
            void wait_entered(int n) { std::unique_lock<std::mutex> lk(mu); cv.wait_for(lk, 5s, [&] { return entered >= n; }); }
        };
        Gate gate;
        std::vector<int> order;
        std::mutex order_mu;
        {
            BackgroundWriter<int> w([&](const int& v) -> uint64_t {
                gate.enter();
                std::lock_guard<std::mutex> lk(order_mu);
                order.push_back(v);
                return v == 3 ? 0 : (uint64_t) v * 10;   // item 3: a failed write
            });
            w.push(1);
            gate.wait_entered(1);
            w.push(2);   // one waits behind the one in flight
            std::atomic<bool> third_in{false};
            std::thread t([&] { w.push(3); third_in = true; });
            std::this_thread::sleep_for(150ms);
            check(!third_in.load(), "a third push waits while one is in flight and one is queued");
            int held = 0;
            w.for_each_held([&](const int&) { ++held; });
            check(held == 2 && !w.idle(), "the queued one and the one in flight are held");
            gate.release();
            t.join();
            w.wait_idle();
            check(third_in.load() && w.idle(), "the queue drains");
            auto done = w.take_done();
            check(done.size() == 3 && done[0].item == 1 && done[0].serial == 10 && done[1].item == 2 && done[1].serial == 20 &&
                  done[2].item == 3 && done[2].serial == 0, "written in order, each handed back with its serial (0: failed)");
            check(order == std::vector<int>({1, 2, 3}) && w.take_done().empty(), "handed back once");
        }
        Gate g2;
        std::vector<int> written;
        {
            BackgroundWriter<int> w([&](const int& v) -> uint64_t { g2.enter(); written.push_back(v); return 1; });
            w.push(7);
            g2.wait_entered(1);
            w.push(8);
            auto queued = w.take_queued();
            check(queued.size() == 1 && queued[0] == 8, "take_queued gives the queued one back unwritten");
            w.push(9);   // and the writer takes more after it
            g2.release();
        }   // the destructor: 7 finishes, 9 is not written (queued at the end)
        check(written == std::vector<int>({7}) || written == std::vector<int>({7, 9}),
              "the destructor finishes the write in flight and does not hang");
        // a SavedConversation through the real SSD tier, while the serve thread matches and takes by serial
        auto img = [](std::vector<int32_t> ids, uint8_t fill) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 18, fill);
            s.cvec = true;
            return s;
        };
        const SessionFileIdentity id{31, 41};
        ConversationSpill sp;
        std::string err;
        check(sp.open(root, uint64_t(64) << 20, err), "a tier for the writer");
        {
            BackgroundWriter<SavedConversation> w([&](const SavedConversation& s) -> uint64_t {
                uint64_t serial = 0;
                std::string e;
                return sp.put(s, id, e, &serial) ? serial : 0;
            });
            for (int k = 0; k < 6; ++k) w.push(img({k, k + 1, k + 2}, (uint8_t) k));
            for (int k = 0; k < 200; ++k) {   // the serve thread matching while the writer puts
                const auto m = sp.best(std::vector<int32_t>{2, 3, 4, 9}, {}, true);
                if (m.tokens > 0) (void) sp.find(m.serial);   // (may be gone or not yet listed: it must only not crash or race)
            }
            w.wait_idle();
            auto done = w.take_done();
            check(done.size() == 6 && sp.size() == 6, "six written, six listed");
            const auto m = sp.best(std::vector<int32_t>{2, 3, 4, 9}, {}, true);
            check(m.tokens == 3, "the entry a prompt continues is found");
            const auto e = sp.find(m.serial);
            check(e && e->live.ids == std::vector<int32_t>({2, 3, 4}), "found by its serial");
            SavedConversation back;
            check(sp.take_serial(m.serial, id, {}, back, err) && back.live.gdn == std::vector<uint8_t>(1 << 18, 2) &&
                  !sp.find(m.serial) && sp.size() == 5, "taken by its serial, byte for byte; it leaves the list");
            check(!sp.take_serial(m.serial, id, {}, back, err) && !sp.drop_serial(m.serial), "a gone serial: nothing");
            check(sp.drop_serial(done[0].serial) && sp.size() == 4, "dropped by its serial");
            // a written image still in RAM: the prefix it gives, by the RAM cache's rule
            const auto p = conversation_image_prefix(done[5].item, std::vector<int32_t>{5, 6, 7, 1}, {}, true);
            check(p.first == 3 && p.second, "a held image's live prefix");
            check(conversation_image_prefix(done[5].item, std::vector<int32_t>{5, 6, 7}, {}, false).first == 0, "other steering: none");
        }
        sp.close();
    }
#if !defined(_WIN32)
    // v3 parts entries (the durable-commit spec, A-D): the index is the commit point, an append reuses the blocks already
    // on the disk instead of rewriting them, and an entry whose bytes or ids cannot be verified is never handed out.
    {
        const SessionFileIdentity id{11, 22};
        auto fill = [](ConversationBuffer& b, uint8_t seed) {
            b.visit(0, b.size(), [&](uint8_t* p, size_t n, size_t off) {
                for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) ((off + i) * 31u + seed);   // equal prefixes stay equal
                return true;
            });
        };
        // `cells` INT8 K/V cells a layer of a 2-layer image: every part's size depends only on `cells`, so a grown image
        // appends to each part (the old bytes are a prefix of the new ones).
        auto kv_img = [&](std::vector<int32_t> ids, int64_t cells, uint8_t seed) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(4096, seed);
            s.cvec = true;
            s.kv.resize(2);
            for (auto& k : s.kv) {
                k.format = 0;
                k.cells = cells;
                k.heads = 2;
                k.head_dim = 256;
                k.page_size = 4;
                k.pooled_rows = cells / 4 + 1;
                k.idx_dim = 128;
                k.k.resize((size_t) cells * 512);
                k.v.resize((size_t) cells * 512);
                k.k_scale.resize((size_t) cells * 16);
                k.v_scale.resize((size_t) cells * 16);
                k.pooled.resize((size_t) (cells / 4 + 1) * 128 * 4);
                for (size_t p = 0; p < 5; ++p) fill(spill_parts::part_of(k, p), (uint8_t) (seed + p));
            }
            return s;
        };
        auto same_kv = [](const SavedConversation& a, const SavedConversation& b) {
            if (a.kv.size() != b.kv.size() || a.live.ids != b.live.ids) return false;
            for (size_t i = 0; i < a.kv.size(); ++i) {
                const ConversationKv& x = a.kv[i];
                const ConversationKv& y = b.kv[i];
                if (x.cells != y.cells || x.format != y.format || x.pooled_rows != y.pooled_rows || !(x.k == y.k) ||
                    !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) || !(x.pooled == y.pooled))
                    return false;
            }
            return true;
        };
        const uint64_t tier = uint64_t(256) << 20;

        // (i) append reuse: one conversation grows, its entry keeps folder and serial, and it reads back as the new image
        {
            const fs::path proot = root / "parts-append";
            ConversationSpill sp;
            check(sp.open(proot, tier, err) && sp.enabled(), "a parts tier opens");
            sp.set_parts_mode(true);
            const SavedConversation a8 = kv_img({1, 2, 3, 4}, 512, 3);
            uint64_t first = 0;
            check(sp.put(a8, id, err, &first) && sp.size() == 1 && sp.entry(0).serial == first, "a first parts entry");
            const fs::path dir = sp.entry(0).file;
            check(fs::exists(dir / "parts.idx") && fs::exists(dir / "ids.bin"), "the entry's index and ids");
            const uint64_t bytes8 = sp.bytes();
            const SavedConversation a16 = kv_img({1, 2, 3, 4, 5, 6}, 1024, 3);
            // the write's own account (stderr, STRATA_SPILL_DEBUG) is what says whether the old blocks were kept
            const fs::path slog = root / "append.log";
            std::fflush(stderr);
            const int saved = ::dup(STDERR_FILENO);
            std::FILE* lf = std::fopen(slog.c_str(), "w");
            check(saved >= 0 && lf != nullptr, "the write log is redirected");
            if (lf != nullptr) ::dup2(::fileno(lf), STDERR_FILENO);
            ::setenv("STRATA_SPILL_DEBUG", "1", 1);
            uint64_t grown = 0;
            const bool put_ok = sp.put(a16, id, err, &grown);
            std::fflush(stderr);
            ::unsetenv("STRATA_SPILL_DEBUG");
            if (lf != nullptr) {
                ::dup2(saved, STDERR_FILENO);
                std::fclose(lf);
            }
            if (saved >= 0) ::close(saved);
            std::ifstream lr(slog);
            const std::string log_text((std::istreambuf_iterator<char>(lr)), std::istreambuf_iterator<char>());
            check(put_ok && grown == first && sp.size() == 1 && sp.entry(0).file == dir,
                  "a grown image appends in place: same folder, same serial");
            check(sp.bytes() > bytes8, "the entry accounts the bytes it now holds");
            check(log_text.find("appended in place") != std::string::npos, "the write appended to the entry it already had");
            check(log_text.find("MiB kept") != std::string::npos, "and it kept the old blocks (they were not rewritten)");
            SavedConversation back;
            check(sp.take_serial(first, id, {}, back, err) && same_kv(back, a16), "the appended entry reads back byte for byte");
            check(sp.find(first).has_value(), "a read keeps the folder listed: the next park extends that generation");
            const SavedConversation other = kv_img({9, 9, 9}, 512, 7);
            uint64_t second = 0;
            check(sp.put(other, id, err, &second) && second != first && sp.size() == 2 && sp.entry(1).file != dir,
                  "another conversation's image gets a folder of its own");
            sp.close();
        }

        // (ii) a part that does not match its index is dropped at adopt, never handed out
        {
            const fs::path proot = root / "parts-corrupt";
            uint64_t serial = 0;
            fs::path dir;
            {
                ConversationSpill sp;
                std::string e;
                check(sp.open(proot, tier, e), "the corrupting tier opens");
                sp.set_parts_mode(true);
                check(sp.put(kv_img({2, 4, 6}, 512, 5), id, e, &serial), "one entry to corrupt");
                dir = sp.entry(0).file;
                sp.close();
            }
            {
                std::fstream f(dir / "kv-0-0.bin", std::ios::in | std::ios::out | std::ios::binary);
                f.seekp(16);
                f.put('\x7f');   // one byte inside the committed K
            }
            ConversationSpill sp;
            check(sp.open(proot, tier, err) && sp.size() == 0 && !fs::exists(dir),
                  "a part whose bytes do not match the index is dropped, its folder with it");
            sp.close();
        }

        // (iii) an unreferenced tail is harmless (the old generation still reads); a truncated part drops the entry
        {
            const fs::path proot = root / "parts-tail";
            uint64_t serial = 0;
            fs::path dir;
            const SavedConversation img = kv_img({3, 1, 4}, 512, 8);
            {
                ConversationSpill sp;
                std::string e;
                check(sp.open(proot, tier, e), "the tail tier opens");
                sp.set_parts_mode(true);
                check(sp.put(img, id, e, &serial), "one entry to cut");
                dir = sp.entry(0).file;
                sp.close();
            }
            {   // bytes past the index's extents (an append that died before its commit): the old generation still reads
                const fs::path part = dir / "kv-0-0.bin";
                fs::resize_file(part, fs::file_size(part) + 4096, ec);
            }
            {
                ConversationSpill sp;
                check(sp.open(proot, tier, err) && sp.size() == 1, "a part with garbage past its committed size is kept");
                SavedConversation back;
                check(sp.take_serial(serial, id, {}, back, err) && same_kv(back, img), "and it still reads back the old generation");
                const fs::path part = sp.find(serial)->file / "kv-0-0.bin";
                sp.close();
                fs::resize_file(part, fs::file_size(part) / 3, ec);   // now below the committed size
            }
            ConversationSpill sp;
            check(sp.open(proot, tier, err) && sp.size() == 0, "a part that lost committed bytes drops the entry");
            sp.close();
        }

        // (iv) ids that do not match the index: the entry donates its bytes, it is never matched to a prompt
        {
            const fs::path proot = root / "parts-ids";
            uint64_t serial = 0;
            fs::path dir;
            // longer than chain_of's 4096-id window, so the chain survives the turn even though the ids are stale
            const std::vector<int32_t> ids_a(4096, 5);
            const SavedConversation img = kv_img(ids_a, 512, 2);
            {
                ConversationSpill sp;
                std::string e;
                check(sp.open(proot, tier, e), "the ids tier opens");
                sp.set_parts_mode(true);
                check(sp.put(img, id, e, &serial), "one entry with ids");
                dir = sp.entry(0).file;
                sp.close();
            }
            {
                uint64_t len = 0, hash = 0;
                std::string e;
                const SavedConversation stale = kv_img({7, 7, 7}, 512, 2);   // another conversation's ids
                check(spill_parts::write_ids(dir, stale, len, hash, e), "a stale ids sidecar is written");
            }
            ConversationSpill sp;
            check(sp.open(proot, tier, err) && sp.size() == 1, "an entry whose ids do not match the index is kept for its bytes");
            sp.set_parts_mode(true);
            check(sp.entry(0).live.ids.empty(), "it carries no ids");
            check(sp.best(std::vector<int32_t>{5, 5, 5, 5}, {}, true).tokens == 0 &&
                  sp.best(std::vector<int32_t>{7, 7, 7}, {}, true).tokens == 0,
                  "no prompt matches it: bytes only");
            std::vector<int32_t> ids_b = ids_a;
            ids_b.push_back(5);
            ids_b.push_back(5);
            const SavedConversation grown = kv_img(ids_b, 1024, 2);
            uint64_t again = 0;
            check(sp.put(grown, id, err, &again) && again == serial && sp.size() == 1,
                  "the next park still extends its bytes");
            sp.close();
        }
    }
#endif

    fs::remove_all(root, ec);
    std::printf("conversation_spill_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
