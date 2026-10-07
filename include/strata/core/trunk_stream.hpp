// include/strata/core/trunk_stream.hpp
#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <set>
#include <map>

namespace strata::core {
class WeightTable;

struct TrunkLayer {
    uint64_t file_off = 0;
    uint64_t nbytes = 0;
    std::vector<std::string> tensors;
};

class TrunkStreamer {
public:
    TrunkStreamer() = default;
    ~TrunkStreamer();

    TrunkStreamer(const TrunkStreamer&) = delete;
    TrunkStreamer& operator=(const TrunkStreamer&) = delete;

    // Initialize the streaming trunk reader.
    // Given the parsed WeightTable index rows, identify the layer tensors (`blk.<layer>.*`).
    // It allocates pinned or device memory for `npin` pinned layers and `nslot` ring buffer slots.
    // If budget_bytes is large enough, all layers might be pinned.
    bool open(const std::string& pack_dir, WeightTable& wt, uint64_t budget_bytes, std::string& err);

    // Make the layer L resident, wait for it if it's being prefetch, and update the WeightTable's data pointers
    // to point to the correct slot in the arena.
    bool bind_layer(int L, WeightTable& wt, std::string& err);

    // Asynchronously prefetch layer L into the next available slot.
    void prefetch_layer(int L);

    // Report stats
    void report() const;

private:
    void io_loop();
    bool read_layer_sync(int L, void* dst, std::string& err);

    std::string pack_dir_;
    int fd_ = -1;
    bool direct_ = false;

    int n_layers_ = 0;
    std::vector<TrunkLayer> layers_;

    int npin_ = 0;
    int nslot_ = 0;
    uint64_t slot_bytes_ = 0;

    std::vector<void*> pin_slots_;
    std::vector<void*> ring_slots_;

    std::vector<int> layer_of_slot_; // for ring slots
    std::vector<int> slot_of_layer_; // -1 if not resident, 0..npin-1 for pinned, npin..npin+nslot-1 for ring

    int ring_next_ = 0;

    // IO thread
    std::thread io_thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    
    // Request to IO thread
    int req_layer_ = -1;
    int req_slot_ = -1;
    bool req_busy_ = false;
    bool req_done_ = false;
    std::string io_err_;

    // Stats
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t bytes_read_ = 0;
    double load_seconds_ = 0.0;
};

} // namespace strata::core
