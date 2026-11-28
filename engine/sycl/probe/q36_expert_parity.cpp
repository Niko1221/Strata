// Qwen3.6 real-expert gate for n_embd=2048,n_ff=512. Independent GGML CPU
// quant/vec_dot and dequantized float references, then SYCL native grouped experts.
// Usage: q36_expert_parity MODEL.gguf [LAYER ...] (defaults: 0,34,39,40).
// Define STRATA_Q36_EXPERT_CPU_ONLY to run the two CPU references without a GPU.
// GPU build links strata_kernels, ggml-cpu, ggml-base; CPU-only needs ggml-cpu/base.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#ifndef STRATA_Q36_EXPERT_CPU_ONLY
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace K = strata::kernels;
namespace {
constexpr int H=2048, FF=512, NT=8, NG=2;
void require(bool yes,const std::string& why) { if(!yes) throw std::runtime_error(why); }
double rel(const std::vector<float>& a,const std::vector<float>& b) {
    require(a.size()==b.size(),"reference size mismatch");double num=0,den=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite expert output");
        num+=std::abs(double(a[i])-b[i]);den+=std::abs(double(b[i]));
    }
    return num/(den+1e-30);
}
struct Expert {
    K::NativeExpertLayout L;
    std::vector<uint8_t> blob;
    std::vector<float> gate,up,down,reference,cpu;
    int id;
};
std::vector<float> inputs(int layer) {
    std::mt19937 rng(3600+layer);std::normal_distribution<float> normal(0,1);
    std::vector<float> x(NT*H);
    for(int t=0;t<NT;++t)for(int j=0;j<H;++j) {
        float v=normal(rng);
        if(t==0)v=0;
        if(t==2)v*=0.1f;
        if(t==3)v=(j%257==0)?1.5f:0.0f;
        if(t==4)v=0.5f;
        if(t==5)v=(j&1)?-0.5f:0.5f;
        if(t==6)v*=2.0f;
        if(t==7)v*=0.3f;
        x[t*H+j]=v;
    }
    return x;
}
Expert load(strata::GgufFile& gguf,int layer,int id,const std::vector<float>& x) {
    const std::string prefix="blk."+std::to_string(layer)+".ffn_";
    auto* g=gguf.find(prefix+"gate_exps.weight");auto* u=gguf.find(prefix+"up_exps.weight");auto* d=gguf.find(prefix+"down_exps.weight");
    require(g&&u&&d,"missing expert tensors at layer "+std::to_string(layer));
    require(g->shape==std::vector<uint64_t>({H,FF,256})&&u->shape==g->shape&&d->shape==std::vector<uint64_t>({FF,H,256}),"unexpected Qwen3.6 geometry");
    require(g->type==u->type,"mixed gate/up types are not supported by this gate");
    Expert e;e.id=id;auto& L=e.L;L.gu_type=g->type;L.d_type=d->type;L.n_embd=H;L.n_ff=FF;
    L.gu_row=ggml_row_size((ggml_type)L.gu_type,H);L.d_row=ggml_row_size((ggml_type)L.d_type,FF);
    L.up_off=FF*L.gu_row;L.down_off=2*L.up_off;L.bytes=L.down_off+H*L.d_row;
    e.blob.resize(L.bytes);
    std::memcpy(e.blob.data(),gguf.tensor_data(*g)+size_t(id)*L.up_off,L.up_off);
    std::memcpy(e.blob.data()+L.up_off,gguf.tensor_data(*u)+size_t(id)*L.up_off,L.up_off);
    std::memcpy(e.blob.data()+L.down_off,gguf.tensor_data(*d)+size_t(id)*H*L.d_row,H*L.d_row);
    const auto* gt=ggml_get_type_traits((ggml_type)L.gu_type);const auto* dt=ggml_get_type_traits((ggml_type)L.d_type);
    const auto* gc=ggml_get_type_traits_cpu((ggml_type)L.gu_type);const auto* dc=ggml_get_type_traits_cpu((ggml_type)L.d_type);
    require(gt&&gt->to_float&&dt&&dt->to_float&&gc&&gc->vec_dot&&dc&&dc->vec_dot,"ggml reference missing a type");
    const auto* ga=ggml_get_type_traits_cpu(gc->vec_dot_type);const auto* da=ggml_get_type_traits_cpu(dc->vec_dot_type);
    require(ga&&ga->from_float&&da&&da->from_float,"ggml activation quantizer missing");
    e.gate.resize(FF*H);e.up.resize(FF*H);e.down.resize(H*FF);
    for(int r=0;r<FF;++r) {
        gt->to_float(e.blob.data()+r*L.gu_row,e.gate.data()+r*H,H);
        gt->to_float(e.blob.data()+L.up_off+r*L.gu_row,e.up.data()+r*H,H);
    }
    for(int r=0;r<H;++r)dt->to_float(e.blob.data()+L.down_off+r*L.d_row,e.down.data()+r*FF,FF);
    e.reference.resize(NT*H);e.cpu.resize(NT*H);
    std::vector<uint8_t> act(ggml_row_size(gc->vec_dot_type,H)),hq(ggml_row_size(dc->vec_dot_type,FF));
    std::vector<float> h(FF),hc(FF);
    for(int t=0;t<NT;++t) {
        ga->from_float(x.data()+t*H,act.data(),H);
        for(int r=0;r<FF;++r) {
            double a=0,b=0;
            for(int j=0;j<H;++j) {a+=double(e.gate[r*H+j])*x[t*H+j];b+=double(e.up[r*H+j])*x[t*H+j];}
            h[r]=float((a/(1+std::exp(-a)))*b);
            float ac=0,bc=0;
            gc->vec_dot(H,&ac,0,e.blob.data()+r*L.gu_row,0,act.data(),0,1);
            gc->vec_dot(H,&bc,0,e.blob.data()+L.up_off+r*L.gu_row,0,act.data(),0,1);
            hc[r]=(ac/(1+std::exp(-ac)))*bc;
        }
        da->from_float(hc.data(),hq.data(),FF);
        for(int r=0;r<H;++r) {
            double out=0;for(int j=0;j<FF;++j)out+=double(e.down[r*FF+j])*h[j];e.reference[t*H+r]=float(out);
            dc->vec_dot(FF,e.cpu.data()+t*H+r,0,e.blob.data()+L.down_off+r*L.d_row,0,hq.data(),0,1);
        }
    }
    const double ec=rel(e.cpu,e.reference);
    require(ec<3e-2,"GGML CPU expert exceeds existing 3% float-reference gate");
    for(int j=0;j<H;++j)require(e.reference[j]==0&&e.cpu[j]==0,"zero-input CPU expert is not zero");
    std::printf("layer=%d expert=%d gu=%s down=%s shape=%dx%d blob=%zu ggml_cpu_vs_float=%.6g\n",layer,id,ggml_type_name((ggml_type)L.gu_type),ggml_type_name((ggml_type)L.d_type),H,FF,L.bytes,ec);
    return e;
}

// Keep the zero/canary fixture separate: the original nt=1 window only read x[0]
// (all zero), so it could pass without exercising nonzero decode arithmetic.
enum class Destination { Identity, Affine, Reverse };
struct GroupCase {
    const char* name;
    std::vector<int32_t> token_order;
    bool reverse_experts;
    int second_group_rotation;
    Destination destinations;
};
std::vector<GroupCase> group_cases() {
    return {
        {"zero_t1", {0}, false, 0, Destination::Affine},
        {"nonzero_t1_dense", {1}, false, 0, Destination::Identity},
        {"nonzero_t1_large_swapped", {6}, true, 0, Destination::Reverse},
        {"nonzero_t1_sparse_swapped", {3}, true, 0, Destination::Affine},
        {"nonzero_t1_alternating", {5}, false, 0, Destination::Affine},
        {"grouped_t3", {0,1,2}, false, 0, Destination::Affine},
        {"grouped_t3_permuted", {3,1,0}, true, 1, Destination::Reverse},
        {"grouped_t8", {0,1,2,3,4,5,6,7}, false, 0, Destination::Affine},
        {"grouped_t8_permuted", {7,6,5,4,3,2,1,0}, true, 3, Destination::Identity},
    };
}
struct CaseData {
    std::array<int,NG> expert_index;
    std::array<int32_t,NG+1> starts;
    std::vector<int32_t> dst,tok;
    std::vector<float> expected,cpu_expected;
    int zero_rows=0,nonzero_rows=0;
};
CaseData case_data(const std::vector<Expert>& es,const GroupCase& c) {
    require(es.size()==NG,"unexpected expert group count");
    const int nt=int(c.token_order.size()),count=NG*nt;
    require(nt>0&&nt<=NT,"invalid expert fixture width");
    CaseData v;v.dst.resize(count);v.tok.resize(count);
    v.expected.resize(count*H);v.cpu_expected.resize(count*H);
    std::vector<bool> seen(count,false);
    for(int g=0;g<NG;++g) {
        const int ei=c.reverse_experts?NG-1-g:g;
        v.expert_index[g]=ei;v.starts[g]=g*nt;
        for(int t=0;t<nt;++t) {
            const int e=g*nt+t,token=c.token_order[(t+g*c.second_group_rotation)%nt];
            require(token>=0&&token<NT,"fixture token out of range");
            const int target=c.destinations==Destination::Identity?e:
                c.destinations==Destination::Reverse?count-1-e:(5*e+3)%count;
            require(!seen[target],"fixture output destinations are not a permutation");seen[target]=true;
            v.dst[e]=target;v.tok[e]=token;
            std::copy_n(es[ei].reference.data()+token*H,H,v.expected.data()+target*H);
            std::copy_n(es[ei].cpu.data()+token*H,H,v.cpu_expected.data()+target*H);
            double signal=0;for(int j=0;j<H;++j)signal+=std::abs(double(v.expected[target*H+j]));
            if(token==0) {require(signal==0,"zero fixture reference is not zero");++v.zero_rows;}
            else {require(signal>1e-12,"nonzero fixture reference has no signal");++v.nonzero_rows;}
        }
    }
    v.starts[NG]=count;
    require(c.token_order!=std::vector<int32_t>{0}||v.nonzero_rows==0,"zero fixture contains nonzero rows");
    require(c.token_order==std::vector<int32_t>{0}||v.nonzero_rows>0,"nonzero fixture is vacuous");
    return v;
}
void check_rows(const std::vector<float>& got,const CaseData& v,const char* source) {
    for(size_t e=0;e<v.tok.size();++e) {
        double signal=0;
        for(int j=0;j<H;++j) {
            const float value=got[size_t(v.dst[e])*H+j];
            require(std::isfinite(value),std::string(source)+" produced a nonfinite row");
            if(v.tok[e]==0)require(value==0,std::string(source)+" zero-input expert is not zero");
            signal+=std::abs(double(value));
        }
        if(v.tok[e]!=0)require(signal>1e-12,std::string(source)+" nonzero expert row has no signal");
    }
}
void cpu_case_gate(const std::vector<Expert>& es) {
    for(const auto& c:group_cases()) {
        const auto v=case_data(es,c);check_rows(v.cpu_expected,v,"CPU");
        const double ec=rel(v.cpu_expected,v.expected);
        require(ec<3e-2,std::string("GGML CPU case exceeds existing 3% gate: ")+c.name);
        std::printf(" CPU case=%s entries_per_expert=%zu zero_rows=%d nonzero_rows=%d cpu_vs_float=%.6g mapping=ok\n",
                    c.name,c.token_order.size(),v.zero_rows,v.nonzero_rows,ec);
    }
}

#ifndef STRATA_Q36_EXPERT_CPU_ONLY
template<class T> struct Buffer {
    sycl::queue& q;T* p;
    Buffer(sycl::queue& q,size_t n):q(q),p(sycl::malloc_device<T>(n,q)){require(p!=nullptr,"GPU allocation failed");}
    ~Buffer(){sycl::free(p,q);}Buffer(const Buffer&)=delete;
};
void dequant_gate(sycl::queue& q,const Expert& e,Buffer<uint8_t>& blob) {
    const auto& L=e.L;
    Buffer<float> fp(q,size_t(H)*FF);Buffer<uint16_t> half(q,size_t(H)*FF);
    for(int role=0;role<3;++role) {
        const int type=role==2?L.d_type:L.gu_type;const size_t off=role==0?0:role==1?L.up_off:L.down_off;
        const auto& expected=role==0?e.gate:role==1?e.up:e.down;
        K::iq_dequant_f32(type,blob.p+off,H*FF,fp.p,&q);K::iq_dequant_f16(type,blob.p+off,H*FF,half.p,&q);q.wait_and_throw();
        std::vector<float> got(H*FF);std::vector<uint16_t> got_half(H*FF);
        q.memcpy(got.data(),fp.p,got.size()*sizeof(float)).wait();q.memcpy(got_half.data(),half.p,got_half.size()*sizeof(uint16_t)).wait();
        double max_error=0,max_half_error=0,scale=0;size_t half_differences=0;
        for(size_t j=0;j<got.size();++j) {
            require(std::isfinite(got[j]),"nonfinite GPU dequantization");
            max_error=std::max(max_error,std::abs(double(got[j])-expected[j]));scale=std::max(scale,std::abs(double(expected[j])));
            const auto h=ggml_fp32_to_fp16(expected[j]);half_differences+=h!=got_half[j];
            const auto actual=ggml_fp16_to_fp32(got_half[j]);require(std::isfinite(actual),"nonfinite GPU half dequantization");
            max_half_error=std::max(max_half_error,std::abs(double(actual)-ggml_fp16_to_fp32(h)));
        }
        require(max_error<=1e-7+scale*2e-6,"GPU F32 dequantization differs from GGML");
        require(max_half_error<=1e-7+scale*1e-3,"GPU F16 dequantization differs materially from GGML");
        std::printf(" dequant role=%d type=%s max_f32=%.6g max_f16=%.6g fp16_bit_differences=%zu\n",role,ggml_type_name((ggml_type)type),max_error,max_half_error,half_differences);
    }
}
void gpu_gate(sycl::queue& q,const std::vector<Expert>& es,const std::vector<float>& x) {
    const auto& L=es[0].L;
    require(K::native_expert_supported(L.gu_type,L.d_type,H,FF),"native_expert_supported rejected real Qwen3.6 types; enable Qwen35/Q6K feature gates");
    auto actual=K::native_expert_layout(L.gu_type,L.d_type,H,FF);
    require(actual.up_off==L.up_off&&actual.down_off==L.down_off&&actual.bytes==L.bytes,"GPU/ggml blob layouts differ");
    Buffer<uint8_t> blob0(q,L.bytes),blob1(q,L.bytes);q.memcpy(blob0.p,es[0].blob.data(),L.bytes).wait();q.memcpy(blob1.p,es[1].blob.data(),L.bytes).wait();
    dequant_gate(q,es[0],blob0);
    Buffer<float> dx(q,x.size()),out(q,NG*NT*H+32);Buffer<uint8_t> xq(q,NT*H/32*36),scratch(q,K::native_expert_scratch_bytes(NG*NT,FF));
    Buffer<unsigned long long> ptr(q,NG);Buffer<int32_t> start(q,NG+1),ng(q,1),dst(q,NG*NT),tok(q,NG*NT);
    q.memcpy(dx.p,x.data(),x.size()*sizeof(float)).wait();K::quantize_q8_1_rows(dx.p,NT,H,xq.p,&q);
    const unsigned long long blobs[]={reinterpret_cast<unsigned long long>(blob0.p),reinterpret_cast<unsigned long long>(blob1.p)};
    const int32_t n=NG;q.memcpy(ng.p,&n,sizeof(n)).wait();
    for(const auto& c:group_cases()) {
        const auto v=case_data(es,c);const int count=int(v.tok.size());
        const unsigned long long ptrs[]={blobs[v.expert_index[0]],blobs[v.expert_index[1]]};
        std::vector<float> guarded(count*H+32,-123456.0f);
        q.memcpy(ptr.p,ptrs,sizeof(ptrs)).wait();q.memcpy(start.p,v.starts.data(),sizeof(v.starts)).wait();
        q.memcpy(dst.p,v.dst.data(),count*sizeof(int32_t)).wait();q.memcpy(tok.p,v.tok.data(),count*sizeof(int32_t)).wait();
        q.memcpy(out.p,guarded.data(),guarded.size()*sizeof(float)).wait();
        K::native_expert_grouped(actual,ptr.p,start.p,ng.p,dst.p,tok.p,NG,count,xq.p,scratch.p,out.p+16,&q);q.wait_and_throw();
        q.memcpy(guarded.data(),out.p,guarded.size()*sizeof(float)).wait();
        for(int j=0;j<16;++j)require(guarded[j]==-123456.0f&&guarded[16+count*H+j]==-123456.0f,"grouped expert wrote beyond output");
        std::vector<float> got(guarded.begin()+16,guarded.begin()+16+count*H);check_rows(got,v,"GPU");
        const double ef=rel(got,v.expected),ec=rel(got,v.cpu_expected);
        require(ef<3e-2&&ec<5e-2,std::string("native expert CPU/GPU parity failed: ")+c.name+"; GPU q8_1 and CPU q8_K are different activation formats");
        std::printf(" grouped case=%s entries_per_expert=%zu zero_rows=%d nonzero_rows=%d reversed_experts=%d gpu_vs_float=%.6g gpu_vs_ggml_cpu=%.6g canaries=ok\n",
                    c.name,c.token_order.size(),v.zero_rows,v.nonzero_rows,int(c.reverse_experts),ef,ec);
    }
}
#endif
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        require(argc>=2,"usage: q36_expert_parity MODEL.gguf [LAYER ...]");ggml_cpu_init();strata::GgufFile gguf(argv[1]);
        require(gguf.get("general.architecture")&&gguf.get("general.architecture")->s=="qwen35moe","this gate requires a qwen35moe GGUF");
        std::vector<int> layers;for(int i=2;i<argc;++i)layers.push_back(std::stoi(argv[i]));if(layers.empty())layers={0,34,39,40};
#ifndef STRATA_Q36_EXPERT_CPU_ONLY
        auto& q=dpct::get_in_order_queue();require(q.get_device().is_gpu(),"expert parity refuses a CPU SYCL device");
        std::printf("GPU: %s\n",q.get_device().get_info<sycl::info::device::name>().c_str());
#endif
        for(int layer:layers) {
            auto x=inputs(layer);std::vector<Expert> es;for(int id:{7,211})es.push_back(load(gguf,layer,id,x));
            cpu_case_gate(es);
#ifndef STRATA_Q36_EXPERT_CPU_ONLY
            gpu_gate(q,es,x);
#endif
        }
#ifdef STRATA_Q36_EXPERT_CPU_ONLY
        std::puts("Qwen3.6 expert CPU references passed; no GPU was used");
#else
        std::puts("Qwen3.6 native expert CPU/GPU parity passed");
#endif
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"q36 expert parity: %s\n",e.what());return 1;}
}
