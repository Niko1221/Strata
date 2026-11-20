#include "kv.hpp"
#include "spin8/spin8.h"
#include "reference_profile.hpp"
extern "C" {
#include "sha256.h"
}
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstring>

namespace strata::qwen38 {
namespace {
void demand(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::string hash(const void* bytes,size_t length){
    unsigned char digest[32];spin8_sidecar_sha256_hash(digest,static_cast<const unsigned char*>(bytes),length);
    const char* hex="0123456789abcdef";std::string out;for(auto b:digest){out+=hex[b>>4];out+=hex[b&15];}return out;
}
}
KvConfig KvConfig::admit(const std::string& mode,const std::string& view,const std::string& path){
    KvConfig c;
    if(mode=="auto"){c.mode=KvMode::f16;c.automatic=true;}else if(mode=="f16")c.mode=KvMode::f16;else if(mode=="k8v4")c.mode=KvMode::k8v4;
    else if(mode=="spin8_k8v4")c.mode=KvMode::spin8_k8v4;
    else if(mode=="f16_turbo2")c.mode=KvMode::f16_turbo2;
    else if(mode=="f16_turbo3")c.mode=KvMode::f16_turbo3;
    else if(mode=="f16_turbo4")c.mode=KvMode::f16_turbo4;
    else throw std::runtime_error("unsupported KV mode (Turbo K is not qualified)");
    if(c.mode!=KvMode::spin8_k8v4){demand(view.empty()&&path.empty(),"Spin8 profile/view requires spin8_k8v4");return c;}
    if(view.empty()||view=="vector")c.view=0;else if(view=="plus")c.view=1;else if(view=="minus")c.view=2;
    else throw std::runtime_error("invalid Spin8 view");
    std::string profile(reinterpret_cast<const char*>(q38_reference_profile),sizeof(q38_reference_profile));
    if(!path.empty()){
        std::ifstream f(path,std::ios::binary);demand(bool(f),"cannot read Spin8 profile");profile.clear();char ch;
        while(f.get(ch)){demand(profile.size()<65536,"Spin8 profile exceeds 64 KiB");profile+=ch;}
        demand(!f.bad(),"Spin8 profile read failed");
    }
    std::istringstream lines(profile);std::string line;std::getline(lines,line);
    if(!line.empty()&&line.back()=='\r')line.pop_back();demand(line=="spin8-profile-v1","invalid Spin8 profile version");
    std::vector<spin8_plane> planes;
    while(std::getline(lines,line)){
        auto comment=line.find('#');if(comment!=std::string::npos)line.resize(comment);
        std::istringstream row(line);row>>std::ws;if(row.eof())continue;
        int a=-1,b=-1;double angle=0;std::string extra;
        demand(bool(row>>a>>b>>angle)&&!(row>>extra)&&a>=0&&a<8&&b>=0&&b<8&&a!=b&&std::isfinite(angle),"invalid Spin8 plane");
        demand(planes.size()<4096,"too many Spin8 planes");planes.push_back({uint32_t(a),uint32_t(b),angle});
    }
    demand(!planes.empty(),"empty Spin8 profile");spin8_plan* raw=nullptr;
    demand(spin8_plan_create(planes.data(),planes.size(),&raw)==SPIN8_OK,"Spin8 plan rejected");
    std::unique_ptr<spin8_plan,decltype(&spin8_plan_free)> plan(raw,spin8_plan_free);
    spin8_audit_report report{};demand(spin8_plan_audit(plan.get(),&report)==SPIN8_OK,"Spin8 plan audit failed");
    double matrix[64];demand(spin8_plan_matrix(plan.get(),spin8_view(c.view),matrix)==SPIN8_OK,"Spin8 matrix unavailable");
    for(int i=0;i<64;++i){c.matrix[i]=float(matrix[i]);matrix[i]=c.matrix[i];demand(std::isfinite(matrix[i]),"nonfinite Spin8 coefficient");}
    spin8_matrix_audit rounded{};demand(spin8_audit_matrix(matrix,&rounded)==SPIN8_OK,"Spin8 F32 audit failed");
    demand(rounded.det_sign==1&&rounded.max_error<=2e-6&&std::abs(rounded.log_abs_det)<=2e-6,"Spin8 F32 coefficients violate SO(8)");
    c.profile_sha=hash(profile.data(),profile.size());
    // Canonical little-endian IEEE754 coefficient fingerprint, independent of padding.
    std::array<unsigned char,256> encoded{};for(int i=0;i<64;++i){uint32_t bits;std::memcpy(&bits,&c.matrix[i],4);for(int j=0;j<4;++j)encoded[i*4+j]=uint8_t(bits>>(8*j));}
    c.matrix_sha=hash(encoded.data(),encoded.size());return c;
}
const char* KvConfig::name()const{switch(mode){case KvMode::f16:return "f16";case KvMode::k8v4:return "k8v4";case KvMode::spin8_k8v4:return "spin8_k8v4";case KvMode::f16_turbo2:return "f16_turbo2";case KvMode::f16_turbo3:return "f16_turbo3";default:return "f16_turbo4";}}
int KvConfig::turbo()const{return mode>=KvMode::f16_turbo2?int(mode)-int(KvMode::f16_turbo2)+1:0;}
const char* KvConfig::key_format()const{return mode==KvMode::k8v4||mode==KvMode::spin8_k8v4?"q8_0":"f16";}
const char* KvConfig::value_format()const{return mode==KvMode::f16?"f16":turbo()==1?"turbo2":turbo()==2?"turbo3":turbo()==3?"turbo4":"q4_0";}
const char* KvConfig::view_name()const{return mode!=KvMode::spin8_k8v4?"none":view==0?"vector":view==1?"plus":"minus";}
size_t KvConfig::key_row_bytes()const{return mode==KvMode::k8v4||mode==KvMode::spin8_k8v4?8*34:512;}
size_t KvConfig::value_row_bytes()const{return mode==KvMode::f16?512:turbo()?2*(18+16*turbo()):8*18;}
size_t KvConfig::scratch_bytes(const KvShape& shape,uint64_t capacity)const{
    const int q=shape.q_elements(),kv=shape.kv_elements();
    size_t bytes=4+(mode==KvMode::f16?0:size_t(q)*4)+(mode==KvMode::spin8_k8v4?(size_t(q)+size_t(kv)+size_t(kv))*4:0);
    if(mode==KvMode::f16&&capacity)bytes+=size_t(kv_batched_score_bytes(shape,capacity));
    return bytes;
}
size_t KvConfig::coefficient_bytes()const{return mode==KvMode::spin8_k8v4?256:0;}
namespace {
uint64_t add_checked(uint64_t a,uint64_t b){demand(b<=UINT64_MAX-a,"KV budget overflow");return a+b;}
uint64_t mul_checked(uint64_t a,uint64_t b){demand(!a||b<=UINT64_MAX/a,"KV budget overflow");return a*b;}
}
uint64_t kv_storage_bytes(const KvConfig& c,uint64_t capacity,uint64_t layers,const KvShape& shape){
    return mul_checked(mul_checked(mul_checked(layers,uint64_t(shape.kv_heads)),capacity),c.key_row_bytes()+c.value_row_bytes());
}
uint64_t kv_batched_score_bytes(const KvShape& shape,uint64_t capacity){
    // Fixed cap keeps KV scratch independent of capacity; the number of tokens
    // that fit per sub-batch is derived from it in KvCache. The opt-in override
    // lets the tuning harness explore sub-batch granularity; the default path
    // stays byte-identical to the validated 32 MiB configuration.
    uint64_t cap=32ull*1024*1024;
    if(const char* env=std::getenv("STRATA_Q38_SCORE_MB")){
        int mb=std::atoi(env);
        if(mb>=8&&mb<=128)cap=uint64_t(mb)*1024*1024;
    }
    const uint64_t per_token=mul_checked(mul_checked(uint64_t(shape.q_heads),capacity),4);
    return std::max(cap,per_token);
}
uint64_t kv_allocator_slack(const KvConfig& c,uint64_t capacity,uint64_t layers,const KvShape& shape){
    // CUDA13/SM120 measurements show 2 MiB backing granularity, including a
    // small-allocation pool. Budget each owned allocation independently; this
    // is a conservative allowance, not claimed exact physical residency.
    constexpr uint64_t granule=2*1024*1024;uint64_t slack=0;
    auto allocation=[&](uint64_t bytes){if(bytes)slack=add_checked(slack,(granule-bytes%granule)%granule);};
    uint64_t rows=mul_checked(mul_checked(layers,uint64_t(shape.kv_heads)),capacity);
    allocation(mul_checked(rows,c.key_row_bytes()));allocation(mul_checked(rows,c.value_row_bytes()));
    allocation(4);if(c.mode!=KvMode::f16)allocation(uint64_t(shape.q_elements())*4);
    else allocation(kv_batched_score_bytes(shape,capacity));
    if(c.mode==KvMode::spin8_k8v4){allocation(256);allocation(uint64_t(shape.q_elements())*4);allocation(uint64_t(shape.kv_elements())*4);allocation(uint64_t(shape.kv_elements())*4);}
    return slack;
}
KvSelection select_kv(const KvConfig& request,uint64_t capacity,uint64_t layers,const KvBudget& budget,uint64_t available,uint64_t reserve,const KvShape& shape){
    demand(capacity&&layers,"empty KV allocation budget");
    uint64_t fixed=reserve;for(auto n:{budget.weights,budget.recurrent,budget.scratch,budget.upload_scratch,budget.allocator_allowance})fixed=add_checked(fixed,n);
    const char* candidates[]={"f16","f16_turbo4","spin8_k8v4"};uint64_t smallest=UINT64_MAX;
    for(int i=0;i<(request.automatic?3:1);++i){
        auto c=request.automatic?KvConfig::admit(candidates[i]):request;
        uint64_t payload=kv_storage_bytes(c,capacity,layers,shape),slack=kv_allocator_slack(c,capacity,layers,shape);
        uint64_t total=add_checked(add_checked(add_checked(add_checked(fixed,payload),c.scratch_bytes(shape,capacity)),c.coefficient_bytes()),slack);smallest=std::min(smallest,total);
        if(total<=available)return {c,request.automatic?"auto":request.name(),request.automatic?(i?"vram_budget_compression":"f16_fits"):"explicit",available,reserve,total,payload,slack};
    }
    throw std::runtime_error("CUDA KV budget cannot fit requested context: required="+std::to_string(smallest)+" available="+std::to_string(available)+" reserve="+std::to_string(reserve));
}
}
