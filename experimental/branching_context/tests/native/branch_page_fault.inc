        // Test executable only: alter an actual driver allocation. The normal
        // pipe has no fault command. The caller supplies an audited page handle.
        if (command == "test-corrupt-page") {
            uint64_t handle = 0, offset = 0;
            std::string extra;
            if (!(in >> handle >> offset) || (in >> extra) || !handle || offset >= vmm_granularity()) {
                error_reply("invalid test page and byte offset"); continue;
            }
            VmmMapping alias;
            uint8_t before = 0, after = 0;
            if (!alias.reserve(vmm_granularity()) || !alias.map(0, handle, true) ||
                cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(&before, alias.base() + offset, 1, cudaMemcpyDeviceToHost) != cudaSuccess) {
                error_reply("cannot inspect actual test allocation"); continue;
            }
            const uint8_t changed = before ^ 1;
            if (cudaMemcpy(alias.base() + offset, &changed, 1, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(&after, alias.base() + offset, 1, cudaMemcpyDeviceToHost) != cudaSuccess ||
                !alias.release()) {
                error_reply("cannot alter actual test allocation"); continue;
            }
            std::cout << "{\"status\":\"ok\",\"driver_handle\":" << handle
                      << ",\"offset\":" << offset << ",\"bytes_written\":1,\"before\":" << unsigned(before)
                      << ",\"after\":" << unsigned(after) << "}" << std::endl;
            continue;
        }
