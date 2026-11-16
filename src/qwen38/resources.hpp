#pragma once
#include <windows.h>
#include <psapi.h>
#include <filesystem>
#include <stdexcept>
#include <cstdint>
namespace strata::qwen38 {
struct ResourceError : std::runtime_error {using std::runtime_error::runtime_error;};
constexpr uint64_t GiB=1024ull*1024*1024, GPU_RESERVE=768ull*1024*1024;
inline void host_resources(){
    MEMORYSTATUSEX m{};m.dwLength=sizeof(m);PERFORMANCE_INFORMATION p{};p.cb=sizeof(p);
    if(!GlobalMemoryStatusEx(&m)||!GetPerformanceInfo(&p,sizeof(p))||m.ullAvailPhys<3*GiB||p.CommitTotal>p.CommitLimit||uint64_t(p.CommitLimit-p.CommitTotal)*p.PageSize<4*GiB)
        throw ResourceError("requires available RAM >= 3 GiB and commit >= 4 GiB");
    for(const auto& disk: {std::pair<const char*,uint64_t>{"H:/",10*GiB},{"C:/",5*GiB}}){
        std::error_code ec;auto sp=std::filesystem::space(disk.first,ec);
        if(ec||sp.available<disk.second)throw ResourceError(std::string("disk floor: ")+disk.first);
    }
}
}
