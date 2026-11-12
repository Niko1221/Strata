// Bandwidth-only 2x2 test requested by user; no byte validation or inference.
#include "strata/models/deepseek41/model.hpp"
#include "expert_host_layout.hpp"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace d = strata::models::deepseek41;
using Clock = std::chrono::steady_clock;
constexpr uint64_t MiB=1ull<<20, GiB=1ull<<30, page=2*MiB, slot=16*MiB;
constexpr uint64_t expected_bytes=156662231040ull, expected_arena=156711780352ull;
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
uint64_t epoch_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
struct Slice{int fd;uint64_t offset,bytes;};
struct Blob{int layer,id;uint64_t offset,bytes;std::array<Slice,3> slices;};
void read_full(const Slice& s,uint8_t* dst){uint64_t n=0;while(n<s.bytes){const auto got=pread(s.fd,dst+n,s.bytes-n,s.offset+n);if(got<0&&errno==EINTR)continue;need(got>0,"GGUF pread failed");n+=got;}}
void read_blob(const Blob& b,uint8_t* dst){uint64_t n=0;for(const auto& s:b.slices){read_full(s,dst+n);n+=s.bytes;}need(n==b.bytes,"projection sizes");}
uint64_t mem_available(){std::ifstream f("/proc/meminfo");std::string line;while(std::getline(f,line))if(line.rfind("MemAvailable:",0)==0){uint64_t kib=0;std::istringstream(line.substr(13))>>kib;return kib*1024;}return 0;}
uint64_t huge_bytes(uint8_t* ptr,uint64_t bytes){
    std::ifstream f("/proc/self/smaps");std::string line;bool own=false;uintptr_t map_end=0;uint64_t total=0;
    const uintptr_t begin=reinterpret_cast<uintptr_t>(ptr),end=begin+bytes;
    while(std::getline(f,line)){uintptr_t lo=0,hi=0;char dash=0;std::istringstream h(line);
        if(h>>std::hex>>lo>>dash>>hi && dash=='-'){own=lo>=begin&&hi<=end;map_end=hi;}
        if(own&&line.rfind("AnonHugePages:",0)==0){uint64_t kib=0;std::istringstream(line.substr(14))>>kib;total+=kib*1024;if(map_end==end)return total;}
    }return total;
}
void coverage(uint8_t* ptr,uint64_t bytes,bool huge,const char* phase){const auto n=huge_bytes(ptr,bytes);
    std::cout<<"{\"kind\":\"coverage\",\"phase\":"<<std::quoted(phase)<<",\"bytes\":"<<bytes<<",\"anon_huge_bytes\":"<<n<<"}\n";
    need(n==(huge?bytes:0),"actual page coverage does not match arm");
}
int main(int argc,char** argv){std::cout<<std::unitbuf<<std::setprecision(12);try{
    need(argc==4,"usage: transfer MODEL HOT_IDS.tsv plan|A|B|C|D");
    const std::string arm=argv[3];const bool plan=arm=="plan",huge=arm=="C"||arm=="D",split=arm=="B"||arm=="D";
    need(plan||arm=="A"||arm=="B"||arm=="C"||arm=="D","invalid arm");
    auto* model=new d::Model(argv[1]);
    std::vector<uint64_t> offsets(40*384,0);std::ifstream hot(argv[2]);int l=0,e=0,hot_count=0;
    need(bool(hot),"hot IDs file");while(hot>>l>>e){need(l>=0&&l<40&&e>=0&&e<384,"hot ID range");need(offsets[l*384+e]!=UINT64_MAX,"duplicate hot ID");offsets[l*384+e]=UINT64_MAX;++hot_count;}
    need(hot.eof()&&hot_count==4645,"frozen hot count");
    const auto sizes=model->expert_bytes();const auto boundaries=d::make_expert_host_segments({sizes.begin(),sizes.end()},384,offsets,40);
    need(boundaries.size()==41&&boundaries.back()==expected_arena,"current arena/segment mismatch");
    std::map<std::string,int> fds;std::vector<Blob> blobs;uint64_t total=0;const char* suffixes[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
    for(l=0;l<40;++l){
        const auto x=model->expert(l,0);
        std::cout<<"{\"kind\":\"format\",\"layer\":"<<l<<",\"gu_type\":"<<x.type<<",\"down_type\":"<<x.down_type<<",\"bytes\":"<<sizes[l]<<"}\n";
        for(e=0;e<384;++e)if(offsets[l*384+e]!=UINT64_MAX){
            Blob b{l,e,offsets[l*384+e],sizes[l],{}};uint64_t sum=0;
            for(int p=0;p<3;++p){const auto& view=model->tensor("blk."+std::to_string(l)+"."+suffixes[p]);const auto& file=*view.file;
                if(!fds.count(file.path())){const int fd=open(file.path().c_str(),O_RDONLY);need(fd>=0,"GGUF open");fds[file.path()]=fd;}
                need(view.bytes%384==0,"expert tensor divisibility");const uint64_t n=view.bytes/384;
                b.slices[p]={fds.at(file.path()),file.data_start()+view.info->offset+n*e,n};sum+=n;
            }
            need(sum==b.bytes&&b.bytes<slot,"expert blob size");
            const auto next=std::upper_bound(boundaries.begin(),boundaries.end(),b.offset);
            need(next!=boundaries.end()&&b.offset+b.bytes<=*next,"expert crosses segment");
            blobs.push_back(b);total+=b.bytes;
        }
    }
    need(blobs.size()==10715&&total==expected_bytes,"full cold complement mismatch");
    std::cout<<"{\"kind\":\"layout\",\"cold_count\":"<<blobs.size()<<",\"payload_bytes\":"<<total<<",\"arena_bytes\":"<<boundaries.back()<<",\"boundaries\":[";
    for(size_t i=0;i<boundaries.size();++i)std::cout<<(i?",":"")<<boundaries[i];
    std::cout<<"]}\n";
    if(plan){std::cout<<"{\"kind\":\"plan_complete\",\"cuda_called\":false}\n";return 0;}
    need(mem_available()>=expected_arena+16*GiB,"insufficient MemAvailable for full arena and 16GiB headroom");
    need(sysconf(_SC_PAGESIZE)==4096,"base page size");uint64_t pmd=0;std::ifstream("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size")>>pmd;need(pmd==page,"PMD page size");
    const auto alloc_start=Clock::now();void* reservation=mmap(nullptr,expected_arena+3*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);need(reservation!=MAP_FAILED,"mmap");
    auto* source=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(reservation)+2*page-1)/page*page);
    need(!mprotect(source,expected_arena,PROT_READ|PROT_WRITE),"mprotect");need(!madvise(source,expected_arena,huge?MADV_HUGEPAGE:MADV_NOHUGEPAGE),"page advice");
    std::memset(source,0,expected_arena);
    for(int attempt=0;huge&&huge_bytes(source,expected_arena)!=expected_arena&&attempt<4;++attempt){
        const int status=madvise(source,expected_arena,25),error=status?errno:0;
        std::cout<<"{\"kind\":\"collapse\",\"attempt\":"<<attempt+1<<",\"status\":"<<status<<",\"errno\":"<<error<<"}\n";
    }
    coverage(source,expected_arena,huge,"after_allocation");
    std::cout<<"{\"kind\":\"allocation\",\"base_address\":"<<reinterpret_cast<uintptr_t>(source)<<",\"ms\":"<<ms(alloc_start)<<"}\n";
    const auto load_start=Clock::now();
    for(auto& b:blobs){read_blob(b,source+b.offset);
        std::cout<<"{\"kind\":\"source\",\"layer\":"<<b.layer<<",\"expert\":"<<b.id<<",\"offset\":"<<b.offset<<",\"bytes\":"<<b.bytes<<"}\n";
    }
    std::cout<<"{\"kind\":\"load\",\"bytes\":"<<total<<",\"ms\":"<<ms(load_start)<<"}\n";
    coverage(source,expected_arena,huge,"before_register");
    cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));size_t free=0,gpu_total=0;ck(cudaMemGetInfo(&free,&gpu_total));need(free>=GiB,"insufficient GPU free memory");
    const auto reg_start=Clock::now();
    if(split){for(size_t i=1;i<boundaries.size();++i)ck(cudaHostRegister(source+boundaries[i-1],boundaries[i]-boundaries[i-1],cudaHostRegisterPortable));}
    else ck(cudaHostRegister(source,expected_arena,cudaHostRegisterPortable));
    std::cout<<"{\"kind\":\"registered\",\"arm\":"<<std::quoted(arm)<<",\"count\":"<<(split?40:1)<<",\"bytes\":"<<expected_arena<<",\"ms\":"<<ms(reg_start)<<",\"gpu\":"<<std::quoted(prop.name)<<"}\n";
    coverage(source,expected_arena,huge,"after_register");
    uint8_t* gpu=nullptr;ck(cudaMalloc(&gpu,8*slot));
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    sleep(2); // Identical quiet interval after loading and registration in every arm.
    for(int pass=0;pass<=3;++pass){
        struct Layer{uint64_t bytes=0,count=0;float event_ms=0;double wall_ms=0;};std::array<Layer,40> results{};size_t cursor=0;double event_ms=0;
        ck(cudaDeviceSynchronize());const auto wall_start=Clock::now();const auto start_ns=epoch_ns();
        for(l=0;l<40;++l){auto& row=results[l];const auto t=Clock::now();ck(cudaEventRecord(start,stream));
            while(cursor<blobs.size()&&blobs[cursor].layer==l){const auto& b=blobs[cursor];ck(cudaMemcpyAsync(gpu+(cursor%8)*slot,source+b.offset,b.bytes,cudaMemcpyHostToDevice,stream));row.bytes+=b.bytes;++row.count;++cursor;}
            ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));row.wall_ms=ms(t);ck(cudaEventElapsedTime(&row.event_ms,start,end));event_ms+=row.event_ms;
        }
        const auto end_ns=epoch_ns();const double wall_ms=ms(wall_start);need(cursor==blobs.size(),"measurement coverage");
        std::cout<<"{\"kind\":\"pass\",\"pass\":"<<pass<<",\"warmup\":"<<(pass==0?"true":"false")<<",\"bytes\":"<<total<<",\"count\":"<<blobs.size()<<",\"start_ns\":"<<start_ns<<",\"end_ns\":"<<end_ns<<",\"event_ms\":"<<event_ms<<",\"wall_ms\":"<<wall_ms<<",\"event_GBps\":"<<total/event_ms/1e6<<",\"wall_GBps\":"<<total/wall_ms/1e6<<"}\n";
        for(l=0;l<40;++l){const auto& r=results[l];std::cout<<"{\"kind\":\"layer\",\"pass\":"<<pass<<",\"layer\":"<<l<<",\"bytes\":"<<r.bytes<<",\"count\":"<<r.count<<",\"event_ms\":"<<r.event_ms<<",\"wall_ms\":"<<r.wall_ms<<",\"event_GBps\":"<<r.bytes/r.event_ms/1e6<<"}\n";}
    }
    coverage(source,expected_arena,huge,"after_measurement");
    std::cout<<"{\"kind\":\"complete\",\"bandwidth_completed\":true,\"byte_validation_performed\":false,\"bytes\":"<<total<<",\"count\":"<<blobs.size()<<",\"explicit_unregister\":false}\n";
    // Keep the established process-exit cleanup path; do not exercise a new
    // explicit unregister lifecycle in a bandwidth attribution experiment.
    std::_Exit(0);
}catch(const std::exception& e){std::cout<<"{\"kind\":\"error\",\"message\":"<<std::quoted(e.what())<<"}\n";std::_Exit(1);}}
