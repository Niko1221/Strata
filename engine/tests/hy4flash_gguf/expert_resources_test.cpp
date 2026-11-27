#include "strata/hy4flash_gguf/host_memory.hpp"
#include "strata/hy4flash_gguf/cpu/expert_layout.hpp"
#include <chrono>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-hy4-expert-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
void test_host_memory() {
    using namespace strata::hy4_flash;
    constexpr uint64_t GiB = 1ull << 30;
    HostMemory m;
#if defined(__linux__)
    TempDirectory t;
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((t.path / rel).parent_path());
        std::ofstream(t.path / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (t.path / "meminfo").string(), cg = (t.path / "cgroup").string(),
                      root = (t.path / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        m.commit = 0;
        const bool ok = host_available_memory(m, mi, cg, root);
        require(m.commit == ~0ull, "Linux memory probe retained a commit limit");
        return ok;
    };
    // no cgroup line at all: MemAvailable alone (before #633: "cannot determine")
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 36 * GiB - 48 * GiB / 64 && m.cgroup_limit == 48 * GiB, "v2 limit");
    // 149: with "file" (active + inactive cache) the active part counts too: 12 GiB of cache, 4 GiB anonymous
    put("fs/box/memory.stat", "file " + std::to_string(12 * GiB) + "\ninactive_file " + std::to_string(4 * GiB) +
                              "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 44 * GiB - 48 * GiB / 64 && m.cgroup_limit == 48 * GiB,
            "v2 limit, the active page cache credited");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n"), "v2 malformed limit accepted");
    fs::remove(t.path / "fs/box/memory.max");
    require(!probe("0::/box\n"), "v2 group without memory.max accepted");
    // v1: the memory controller's group (a hybrid line list), a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    put("mountinfo", "41 1 0:9 /containers/team " + root + "/memory rw - cgroup memory rw,memory\n");
    put("fs/memory/docker/abc/memory.stat", "total_inactive_file " + std::to_string(4 * GiB) +
                                            "\ntotal_dirty 0\ntotal_writeback 0\n");
    require(probe("4:memory:/containers/team/docker/abc\n") && m.cgroup_limit == 32 * GiB &&
            m.available == 26 * GiB - 32 * GiB / 64, "v1 subtree mount with reclaimable file cache");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v1 unlimited");
    // a v1 group not visible here (another namespace): MemAvailable alone
    require(probe("4:memory:/elsewhere\n") && m.available == 100 * GiB, "v1 group not mounted");
#else
    require(host_available_memory(m) && m.available > 0 && m.cgroup_limit == ~0ull, "this PC's RAM");
#if defined(_WIN32)
    require(m.commit != ~0ull, "Windows memory probe did not report commit capacity");
#else
    require(m.commit == ~0ull, "a platform without commit accounting imposed a limit");
#endif
    (void) GiB;
#endif
}

}

void dense_prefix_layout() {
    namespace cpu = strata::hy4_flash::cpu;
    constexpr int width = 1536, layers = 42, experts = 512;
    constexpr uint64_t gate = width * (width / 256) * 144;
    constexpr uint64_t down_q6 = width * (width / 256) * 210;
    TempDirectory dir;
    std::ostringstream rows;
    uint64_t total = 0;
    for (int i = 0; i < layers; ++i) {
        const uint64_t blob = 2 * gate + (i % 2 ? down_q6 : gate);
        rows << i << " 12 " << (i % 2 ? 14 : 12) << ' ' << total << ' ' << blob << ' '
             << total << ' ' << total + experts * gate << ' ' << total + 2 * experts * gate << '\n';
        total += experts * blob;
    }
    std::ofstream(dir.path / "native_experts.txt") << "# strata native experts v4 (n_expert " +
               std::to_string(experts) + ", total " + std::to_string(total) + ")\n" + rows.str();

    std::string err;
    const bool ok = cpu::expert_layout_load(dir.path.string(), layers, experts, err, 1, width, width, 10.f);
    require(ok, "load dense-prefix expert layout: " + err);
    const auto& layout = cpu::expert_layout();
    require(layout.native && layout.block_base == 1 && layout.n_layers == layers && layout.n_expert == experts &&
          layout.n_embd == width && layout.n_ff == width, "dense-prefix pack geometry");
    uint64_t offset = 0;
    for (int i = 0; i < layers; ++i) {
        const auto& fmt = layout.fmt.at(i);
        const uint64_t blob = 2 * gate + (i % 2 ? down_q6 : gate);
        require(layout.block_of(i) == i + 1 && fmt.n_embd == width && fmt.n_ff == width &&
              fmt.d_type == (i % 2 ? 14 : 12) && fmt.swiglu_clamp == 10.f &&
              fmt.h_bytes > cpu::kNativeHBytes && fmt.h_bytes <= cpu::kNativeHBytesMax &&
              fmt.act_bytes <= cpu::kNativeActBytes && layout.bytes.at(i) == blob &&
              layout.offset.at(i) == offset, "dense-prefix expert ordinal " + std::to_string(i));
        offset += experts * blob;
    }
    require(!strata::kernels::cpu::expert_layout_load(dir.path.string(), layers, experts, err), "default geometry rejects the wide pack");
}

int main() { test_host_memory(); dense_prefix_layout(); }
