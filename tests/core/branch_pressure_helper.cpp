// Separate real-CUDA process holding physical memory until explicitly released.
// The registered API test records this binary's hash and its measured receipts.
#include "branch_pressure.hpp"
#include <iostream>
#include <string>

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !strata::core::vmm_available()) {
        std::cerr << "BLOCKED: actual CUDA VMM device required\n";
        return 2;
    }
    BranchPressure pressure;
    std::cout << "{\"status\":\"ready\",\"granularity\":" << strata::core::vmm_granularity() << "}" << std::endl;
    std::string command;
    bool filled = false;
    while (std::getline(std::cin, command)) {
        if (command == "fill" && !filled) {
            const bool good = pressure.fill();
            filled = true;
            const auto errors = strata::core::vmm_failure_counters();
            std::cout << "{\"status\":\"" << (good ? "full" : "error")
                      << "\",\"held_bytes\":" << pressure.allocated_bytes
                      << ",\"physical_allocations\":" << pressure.handles.size()
                      << ",\"free_before\":" << pressure.free_before
                      << ",\"free_after\":" << pressure.free_after_fill
                      << ",\"total_bytes\":" << pressure.total
                      << ",\"driver_failures\":" << errors.driver_failures
                      << ",\"out_of_memory_failures\":" << errors.out_of_memory_failures
                      << ",\"last_error\":" << errors.last_error
                      << ",\"last_operation\":\"" << errors.last_operation << "\"}" << std::endl;
            if (!good) return 1;
        } else if (command == "release" && filled) {
            const bool good = pressure.release();
            std::cout << "{\"status\":\"" << (good ? "released" : "error")
                      << "\",\"released_bytes\":" << pressure.released_bytes
                      << ",\"free_after\":" << pressure.free_after_release << "}" << std::endl;
            return good ? 0 : 1;
        } else {
            std::cerr << "Invalid pressure-helper transition\n";
            return 1;
        }
    }
    return 1; // EOF still releases all owned allocations through RAII.
}
