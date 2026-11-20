#include "strata/program/joined_prefetch.hpp"
#include <atomic>
#include <cassert>
#include <stdexcept>

std::atomic<int> live{0}, completed{0};
struct Scratch { Scratch() { ++live; } ~Scratch() { --live; } };
void read_batch() { thread_local Scratch scratch; ++completed; }

int main() {
    using strata::program::JoinedPrefetch;
    JoinedPrefetch ahead;
    for (int i=0; i<128; ++i) {
        ahead.start(read_batch);
        assert(ahead.valid());
        ahead.get();
        assert(!ahead.valid() && live==0 && completed==i+1);
    }
    { JoinedPrefetch early_exit; early_exit.start(read_batch); }
    assert(live==0 && completed==129);
    ahead.start([] { read_batch(); throw std::runtime_error("read failed"); });
    bool caught=false;
    try { ahead.get(); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && live==0 && !ahead.valid());
    ahead.get(); // error is consumed once
    int sync=0;
    for(int failure=0; failure<2; ++failure) {
        ahead.start([&] { ++sync; }, [failure](auto) -> std::thread {
            if(failure==0) throw std::bad_alloc();
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        });
        assert(ahead.valid() && sync==failure+1);
        ahead.get();
    }
    ahead.start([] { throw std::runtime_error("sync failure"); }, [](auto)->std::thread {throw std::bad_alloc();});
    caught=false;
    try { ahead.get(); } catch(const std::runtime_error&) { caught=true; }
    assert(caught);
}
