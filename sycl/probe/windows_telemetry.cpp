// Read-only IGCL sampler. Driver calls live outside the API server process.
// Build with Intel's igcl_api.h; ControlLib.dll comes from the graphics driver.
#include <windows.h>
#define CTL_APIEXPORT
#include "igcl_api.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string quote(const char* text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
        if (*p == '"' || *p == '\\') out << '\\' << *p;
        else if (*p < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(*p);
        else out << *p;
    }
    return out.str() + '"';
}

static void sensor(std::ostream& out, const char* key, const ctl_oc_telemetry_item_t& item, ctl_units_t units) {
    if (!item.bSupported || item.units != units) return;
    double v;
    switch (item.type) {
    case CTL_DATA_TYPE_DOUBLE: v = item.value.datadouble; break;
    case CTL_DATA_TYPE_FLOAT: v = item.value.datafloat; break;
    case CTL_DATA_TYPE_UINT64: v = static_cast<double>(item.value.datau64); break;
    case CTL_DATA_TYPE_INT64: v = static_cast<double>(item.value.data64); break;
    case CTL_DATA_TYPE_UINT32: v = item.value.datau32; break;
    case CTL_DATA_TYPE_INT32: v = item.value.data32; break;
    default: return;
    }
    if (std::isfinite(v) && v >= 0) out << ',' << quote(key) << ':' << std::setprecision(16) << v;
}

int main(int argc, char** argv) {
    unsigned index = 0, samples = 0;
    DWORD parent_pid = 0;
    std::string bdf;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        std::string arg = argv[i++];
        if (arg == "--index") index = std::stoul(argv[i]);
        else if (arg == "--parent") parent_pid = std::stoul(argv[i]);
        else if (arg == "--samples") samples = std::stoul(argv[i]);
        else if (arg == "--pci-bdf") bdf = argv[i];
        else return 2;
    }
    HANDLE parent = parent_pid ? OpenProcess(SYNCHRONIZE, FALSE, parent_pid) : nullptr;
    if (parent_pid && !parent) return 3;
    auto lib = LoadLibraryExW(L"ControlLib.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!lib) { std::cerr << "Intel ControlLib.dll unavailable\n"; return 4; }
#define FN(name) auto name = reinterpret_cast<decltype(&::name)>(GetProcAddress(lib, #name))
    FN(ctlInit); FN(ctlEnumerateDevices); FN(ctlGetDeviceProperties); FN(ctlClose);
    FN(ctlPowerTelemetryGet); FN(ctlEnumMemoryModules); FN(ctlMemoryGetProperties); FN(ctlMemoryGetState);
    FN(ctlPciGetProperties); FN(ctlPciGetState);
#undef FN
    if (!ctlInit || !ctlEnumerateDevices || !ctlGetDeviceProperties) return 5;
    ctl_init_args_t args{};
    args.Size = sizeof(args); args.AppVersion = CTL_IMPL_VERSION; args.flags = CTL_INIT_FLAG_USE_LEVEL_ZERO;
    ctl_api_handle_t api{};
    if (ctlInit(&args, &api) != CTL_RESULT_SUCCESS) return 6;
    uint32_t count = 0;
    if (ctlEnumerateDevices(api, &count, nullptr) || !count) return 7;
    std::vector<ctl_device_adapter_handle_t> devices(count);
    if (ctlEnumerateDevices(api, &count, devices.data())) return 7;
    ctl_device_adapter_handle_t device = nullptr;
    std::string name, address;
    unsigned matched = 0;
    for (auto candidate : devices) {
        ctl_device_adapter_properties_t p{};
        p.Size = sizeof(p); p.Version = 2;
        LUID luid{}; p.pDeviceID = &luid; p.device_id_size = sizeof(luid);
        if (ctlGetDeviceProperties(candidate, &p) || p.pci_vendor_id != 0x8086 || p.device_type != CTL_DEVICE_TYPE_GRAPHICS) continue;
        std::string candidate_bdf = std::to_string(p.adapter_bdf.bus) + ":" + std::to_string(p.adapter_bdf.device) + "." + std::to_string(p.adapter_bdf.function);
        if ((!bdf.empty() && bdf == candidate_bdf) || (bdf.empty() && matched == index)) {
            device = candidate; p.name[sizeof(p.name) - 1] = '\0'; name = p.name; address = candidate_bdf; break;
        }
        ++matched;
    }
    if (!device) { std::cerr << "Requested Intel adapter unavailable\n"; return 8; }
    std::vector<ctl_mem_handle_t> memories;
    if (ctlEnumMemoryModules && ctlMemoryGetProperties && ctlMemoryGetState) {
        uint32_t n = 0;
        if (!ctlEnumMemoryModules(device, &n, nullptr) && n) {
            std::vector<ctl_mem_handle_t> all(n);
            if (!ctlEnumMemoryModules(device, &n, all.data())) for (auto mem : all) {
                ctl_mem_properties_t p{}; p.Size = sizeof(p);
                if (!ctlMemoryGetProperties(mem, &p) && p.location == CTL_MEM_LOC_DEVICE) memories.push_back(mem);
            }
        }
    }
    unsigned emitted = 0;
    do {
        if (parent && WaitForSingleObject(parent, 0) != WAIT_TIMEOUT) break;
        std::ostringstream out;
        out << "{\"name\":" << quote(name.c_str()) << ",\"pci_bdf\":" << quote(address.c_str());
        ctl_power_telemetry_t t{}; t.Size = sizeof(t);
        if (ctlPowerTelemetryGet && !ctlPowerTelemetryGet(device, &t)) {
            sensor(out, "timestamp", t.timeStamp, CTL_UNITS_TIME_SECONDS);
            sensor(out, "energy", t.gpuEnergyCounter, CTL_UNITS_ENERGY_JOULES);
            sensor(out, "activity", t.globalActivityCounter, CTL_UNITS_TIME_SECONDS);
            sensor(out, "temp", t.gpuCurrentTemperature, CTL_UNITS_TEMPERATURE_CELSIUS);
            sensor(out, "vram_temp", t.vramCurrentTemperature, CTL_UNITS_TEMPERATURE_CELSIUS);
        }
        uint64_t total = 0, free = 0;
        bool valid = !memories.empty();
        for (auto mem : memories) {
            ctl_mem_state_t s{}; s.Size = sizeof(s);
            if (ctlMemoryGetState(mem, &s) || !s.size || s.free > s.size) { valid = false; break; }
            total += s.size; free += s.free;
        }
        if (valid) out << ",\"mem_total\":" << total << ",\"mem_used\":" << total - free;
        ctl_pci_properties_t p{}; p.Size = sizeof(p);
        p.address.Size = sizeof(p.address); p.maxSpeed.Size = sizeof(p.maxSpeed);
        if (ctlPciGetProperties && !ctlPciGetProperties(device, &p) && p.maxSpeed.gen > 0)
            out << ",\"pcie_gen_max\":" << p.maxSpeed.gen;
        ctl_pci_state_t state{}; state.Size = sizeof(state); state.speed.Size = sizeof(state.speed);
        if (ctlPciGetState && !ctlPciGetState(device, &state)) {
            if (state.speed.gen > 0) out << ",\"pcie_gen\":" << state.speed.gen;
            if (state.speed.width > 0) out << ",\"pcie_width\":" << state.speed.width;
        }
        std::cout << out.str() << "}\n" << std::flush;
        if (!std::cout || (samples && ++emitted >= samples)) break;
        if (parent) { if (WaitForSingleObject(parent, 1000) != WAIT_TIMEOUT) break; }
        else Sleep(1000);
    } while (true);
    if (ctlClose) ctlClose(api);
    if (parent) CloseHandle(parent);
    FreeLibrary(lib);
    return 0;
}
