#pragma once
#include <windows.h>
#include <psapi.h>
#include <stdexcept>
#include <string>
#include <cstdint>
namespace strata::qwen38 {
struct ResourceError : std::runtime_error {using std::runtime_error::runtime_error;};
constexpr uint64_t GiB=1024ull*1024*1024, GPU_RESERVE=768ull*1024*1024;
inline void host_resources(){
    MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
    if(!GlobalMemoryStatusEx(&m)||m.ullAvailPhys<3*GiB)
        throw ResourceError("requires available RAM >= 3 GiB and commit >= 4 GiB");
    if(m.ullAvailPageFile<4*GiB){
        PERFORMANCE_INFORMATION p{};p.cb=sizeof(p);
        if(!GetPerformanceInfo(&p,sizeof(p))||p.CommitTotal>p.CommitLimit||uint64_t(p.CommitLimit-p.CommitTotal)*p.PageSize<4*GiB)
            throw ResourceError("requires available RAM >= 3 GiB and commit >= 4 GiB");
    }
    ULARGE_INTEGER avail{},tot{},free_b{};
    if(!GetDiskFreeSpaceExW(L"H:\\",&avail,&tot,&free_b)||avail.QuadPart<10*GiB)
        throw ResourceError("disk floor: H:/");
    if(!GetDiskFreeSpaceExW(L"C:\\",&avail,&tot,&free_b)||avail.QuadPart<5*GiB)
        throw ResourceError("disk floor: C:/");
}
}
