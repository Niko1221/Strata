#include "strata/core/idle_cache.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
using strata::core::IdleDeviceArena;
using strata::core::IdleSnapshot;
static void check(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
int main() {
    if (!strata::core::vmm_available()) return 77;
    try {
        std::string err;
        IdleDeviceArena a, b;
        check(a.allocate((9u << 20) + 31, true, err), err.c_str());
        check(b.allocate(65536, true, err), err.c_str());
        std::vector<uint8_t> expected(a.size()), second(b.size(), 43), host(1234567, 71), empty;
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<uint8_t>(i * 13 + 7);
        check(cudaMemcpy(a.data(), expected.data(), a.size(), cudaMemcpyHostToDevice) == cudaSuccess, "upload a");
        check(cudaMemcpy(b.data(), second.data(), b.size(), cudaMemcpyHostToDevice) == cudaSuccess, "upload b");
        check(a.remember_constant(a.data(), 64, err), err.c_str());
        check(!a.remember_constant(static_cast<uint8_t*>(a.data()) + a.size(), 1, err), "bounds"); err.clear();
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "stream");
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess, "capture");
        const auto grerr = cudaMemsetAsync(b.data(), 89, b.size(), stream);
        check(grerr == cudaSuccess, cudaGetErrorString(grerr));
        check(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "capture end");
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess, "graph instantiate");
        IdleSnapshot snap;
        check(snap.init("idle-cache-test-files", 3600, err), err.c_str());
        const std::vector<IdleDeviceArena*> arenas{&a, &b};
        const std::vector<std::vector<uint8_t>*> hosts{&host, &empty};
        const auto address = a.data();
        for (int cycle = 0; cycle < 3; ++cycle) {
            check(snap.save(arenas, hosts, err), err.c_str());
            check(snap.host_bytes() == host.size(), "host byte count");
            check(std::filesystem::file_size(snap.path()) == snap.disk_bytes(), "disk byte count");
            check(a.unmap(err) && b.unmap(err), err.c_str());
            check(!a.resident() && !b.resident(), "physically released");
            std::vector<uint8_t>().swap(host);
            check(a.map(err) && b.map(err), err.c_str());
            check(a.data() == address, "address stable");
            check(snap.restore(arenas, hosts, err), err.c_str());
            std::vector<uint8_t> out(a.size());
            check(cudaMemcpy(out.data(), a.data(), out.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "read restored");
            check(out == expected && host == std::vector<uint8_t>(1234567, 71), "snapshot exact");
            check(cudaGraphLaunch(exec, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "graph after remap");
            out.resize(b.size());
            check(cudaMemcpy(out.data(), b.data(), out.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "graph output");
            check(out == std::vector<uint8_t>(out.size(), 89), "graph still uses correct address");
            check(snap.erase(err), err.c_str());
        }
        check(snap.save(arenas, hosts, err), err.c_str());
        { std::fstream f(snap.path(), std::ios::in | std::ios::out | std::ios::binary); f.seekp(256); f.put('!'); }
        check(!snap.restore(arenas, hosts, err), "corruption rejected"); err.clear();
        check(snap.erase(err), err.c_str());
        check(snap.save(arenas, hosts, err), err.c_str());
        std::filesystem::resize_file(snap.path(), 19);
        check(!snap.restore(arenas, hosts, err), "truncation rejected"); err.clear();
        check(snap.erase(err), err.c_str());
        check(!snap.restore(arenas, hosts, err), "missing file rejected"); err.clear();
        check(snap.save(arenas, hosts, err, 0), err.c_str());
        check(!snap.restore(arenas, hosts, err), "expired snapshot rejected"); err.clear();
        const std::filesystem::path abandoned = "idle-cache-test-files/strata-kv-idle-abandoned.tmp";
        { std::ofstream f(abandoned); f << 'x'; }
        std::filesystem::last_write_time(abandoned, std::filesystem::file_time_type::clock::now() - std::chrono::hours(2));
        IdleSnapshot cleanup;
        check(cleanup.init("idle-cache-test-files", 3600, err), err.c_str());
        check(!std::filesystem::exists(snap.path()), "expired abandoned snapshot cleaned on startup");
        check(!std::filesystem::exists(abandoned), "incomplete abandoned snapshot cleaned on startup");
        check(snap.erase(err), err.c_str());
        check(a.unmap(err) && a.fresh(err), err.c_str());
        std::vector<uint8_t> fresh(a.size());
        check(cudaMemcpy(fresh.data(), a.data(), fresh.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "fresh copy");
        for (size_t i = 0; i < fresh.size(); ++i) check(fresh[i] == (i < 64 ? expected[i] : 0), "fresh keeps only constants");
        check(!a.allocate(8, true, err), "double allocation rejected"); err.clear();
        IdleDeviceArena normal;
        check(normal.allocate(4096, false, err), err.c_str());
        check(!normal.unmap(err), "normal allocation protected"); err.clear();
        IdleSnapshot invalid;
        { std::ofstream f("idle-cache-test-files/not-a-directory"); f << 'x'; }
        check(!invalid.init("idle-cache-test-files/not-a-directory/sub", 3600, err), "invalid directory rejected");
        std::filesystem::remove("idle-cache-test-files/not-a-directory");
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
        std::puts("PASS: VMM graph replay, 3 exact multi-region restores, constants, corruption, truncation, missing file, expired-file cleanup, allocation and path failures");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
