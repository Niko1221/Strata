#include "kv.hpp"
#include "upload_reader.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
namespace Q=strata::qwen38;
static void demand(bool b,const char* m){if(!b)throw std::runtime_error(m);}
template<class F>static void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}demand(rejected,"invalid input accepted");}
int main(int argc,char** argv){try{
    demand(argc==2,"reference profile path required");
    {
        const auto request=Q::KvConfig::admit("auto");Q::KvBudget budget{123456789,4567890,345678,23456,16777216};const uint64_t reserve=768ull*1024*1024,ctx=4096;
        uint64_t fixed=budget.weights+budget.recurrent+budget.scratch+budget.upload_scratch+budget.allocator_allowance+reserve;
        uint64_t thresholds[3];const char* modes[]={"f16","f16_turbo4","spin8_k8v4"};
        for(int i=0;i<3;++i){auto c=Q::KvConfig::admit(modes[i]);thresholds[i]=fixed+Q::kv_storage_bytes(c,ctx,16)+c.scratch_bytes()+c.coefficient_bytes()+Q::kv_allocator_slack(c,ctx,16);}
        demand(thresholds[0]>thresholds[1]&&thresholds[1]>thresholds[2],"policy order payload");
        for(int i=0;i<3;++i){auto exact=Q::select_kv(request,ctx,16,budget,thresholds[i],reserve);demand(std::string(exact.config.name())==modes[i]&&exact.required==thresholds[i],"auto exact-fit threshold");
            demand(exact.payload==Q::kv_storage_bytes(exact.config,ctx,16),"auto changed requested context");
            if(i<2){auto below=Q::select_kv(request,ctx,16,budget,thresholds[i]-1,reserve);demand(std::string(below.config.name())==modes[i+1],"one-byte-below transition");}
            else rejects([&]{Q::select_kv(request,ctx,16,budget,thresholds[i]-1,reserve);});
        }
        for(const char* mode:{"f16","k8v4","spin8_k8v4","f16_turbo2","f16_turbo3","f16_turbo4"}){auto explicit_request=Q::KvConfig::admit(mode);auto selected=Q::select_kv(explicit_request,ctx,16,budget,UINT64_MAX,reserve);demand(selected.requested==mode&&std::string(selected.config.name())==mode,"explicit changed");rejects([&]{Q::select_kv(explicit_request,ctx,16,budget,selected.required-1,reserve);});}
        rejects([&]{Q::select_kv(request,UINT64_MAX,16,budget,UINT64_MAX,reserve);});
        rejects([&]{Q::select_kv(request,ctx,16,{UINT64_MAX,1,0,0,0},UINT64_MAX,reserve);});
        rejects([&]{Q::select_kv(request,0,16,budget,UINT64_MAX,reserve);});
        rejects([]{Q::KvConfig::admit("auto","plus");});
        std::puts("PASS production auto selector exact/below/none/overflow/explicit/context boundaries");
    }
    for(const char* mode:{"f16","k8v4","spin8_k8v4","f16_turbo2","f16_turbo3","f16_turbo4"}){auto c=Q::KvConfig::admit(mode);demand(std::string(c.name())==mode,"mode mismatch");}
    for(const char* view:{"vector","plus","minus"}){auto compiled=Q::KvConfig::admit("spin8_k8v4",view),file=Q::KvConfig::admit("spin8_k8v4",view,argv[1]);demand(compiled.profile_sha==file.profile_sha&&compiled.matrix_sha==file.matrix_sha,"compiled reference byte mismatch");std::printf("PROFILE view=%s profile=%s matrix=%s\n",view,compiled.profile_sha.c_str(),compiled.matrix_sha.c_str());}
    rejects([]{Q::KvConfig::admit("turbo3");});rejects([]{Q::KvConfig::admit("f16","plus");});rejects([]{Q::KvConfig::admit("k8v4","","absent");});rejects([]{Q::KvConfig::admit("spin8_k8v4","all");});rejects([]{Q::KvConfig::admit("spin8_k8v4","","absent");});
    const auto profile=std::filesystem::current_path()/"q38-kv-invalid.profile";demand(!std::filesystem::exists(profile),"fixture already exists");
    for(const char* invalid:{"", "spin8-profile-v2\n0 1 .2\n", "spin8-profile-v1\n", "spin8-profile-v1\n0 0 .2\n", "spin8-profile-v1\n0 8 .2\n", "spin8-profile-v1\n0 1 nan\n", "spin8-profile-v1\n0 1 .2 extra\n"}){
        {std::ofstream f(profile,std::ios::binary);f<<invalid;}rejects([&]{Q::KvConfig::admit("spin8_k8v4","vector",profile.string());});
    }
    {std::ofstream f(profile,std::ios::binary);f<<std::string(65537,' ');}rejects([&]{Q::KvConfig::admit("spin8_k8v4","vector",profile.string());});std::filesystem::remove(profile);
    const auto path=std::filesystem::current_path()/"q38-upload-fixture.bin";demand(!std::filesystem::exists(path),"upload fixture already exists");
    size_t bytes=Q::UploadReader::chunk_bytes+257;std::vector<uint8_t> input(bytes);for(size_t i=0;i<bytes;++i)input[i]=uint8_t((i*71+i/511)&255);
    {std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(input.data()),input.size());demand(bool(f),"fixture write failed");}
    {
        Q::UploadReader reader(path.string(),bytes);demand(reader.staging_bytes()==16*1024*1024,"unbounded staging");
        size_t consumed=0,calls=0,checks=0;
        reader.read(17,bytes-17,[&]{++checks;},[&](uint64_t offset,const uint8_t* p,size_t n){demand(n<=Q::UploadReader::chunk_bytes,"oversized chunk");demand(std::equal(p,p+n,input.data()+17+offset),"upload byte mismatch");consumed+=n;++calls;});
        demand(consumed==bytes-17&&calls==2&&checks==calls,"chunk/resource checks");
        auto no=[]{};size_t invalid_deliveries=0;auto count_delivery=[&](uint64_t,const uint8_t*,size_t){++invalid_deliveries;};
        rejects([&]{reader.read(bytes,1,no,count_delivery);});demand(invalid_deliveries==0,"out-of-range read delivered data");
        rejects([&]{reader.read(UINT64_MAX,2,no,count_delivery);});demand(invalid_deliveries==0,"overflowing offset delivered data");
        rejects([&]{reader.read(1,UINT64_MAX,no,count_delivery);});demand(invalid_deliveries==0,"overflowing length delivered data");
        // File shrinks after admission: no partial first chunk may reach consume.
        std::filesystem::resize_file(path,1024);bool delivered=false;
        rejects([&]{reader.read(0,bytes,no,[&](uint64_t,const uint8_t*,size_t){delivered=true;});});demand(!delivered,"short read delivered partial data");
    }
    rejects([&]{Q::UploadReader reader(path.string(),bytes);});std::filesystem::remove(path);
    rejects([]{Q::UploadReader reader("missing-q38-upload",1);});
    std::puts("PASS KV CPU modes/profile/audit and bounded upload byte/bounds/short-read checks");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
