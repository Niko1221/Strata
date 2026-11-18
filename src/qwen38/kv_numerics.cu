#include "kv_reference.hpp"
#include "resources.hpp"
#include "session_state.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include <cstdio>
#include <set>
#include <limits>
namespace Q=strata::qwen38;
namespace R=kv_reference;
struct Stream{cudaStream_t s;Stream(){Q::ck(cudaStreamCreate(&s));}~Stream(){cudaStreamDestroy(s);}};
template<class T>static void put(const Q::Buffer& b,const std::vector<T>& x){R::demand(b.bytes>=x.size()*sizeof(T),"upload too large");Q::ck(cudaMemcpy(b.data,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T,class B>static std::vector<T> get(const B& b){std::vector<T> v(b.bytes/sizeof(T));Q::ck(cudaMemcpy(v.data(),b.data,b.bytes,cudaMemcpyDeviceToHost));return v;}
static double compare(const std::vector<float>& got,const std::vector<double>& ref,double absolute,double relative){
    double worst=0;for(size_t i=0;i<ref.size();++i){double error=std::abs(got[i]-ref[i]);worst=std::max(worst,error);if(!std::isfinite(got[i])||error>absolute+relative*std::abs(ref[i])){std::fprintf(stderr,"index=%zu got=%g ref=%g error=%g\n",i,got[i],ref[i],error);throw std::runtime_error("KV numerical tolerance");}}return worst;
}
template<class F>static void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}R::demand(caught,"invalid operation accepted");}
static void reserve(){Q::host_resources();size_t free,total;Q::ck(cudaMemGetInfo(&free,&total));R::demand(free>Q::GPU_RESERVE+512*1024*1024,"KV fixture GPU reserve");}
static void spin_identity(const Q::KvConfig& c,cudaStream_t s){
    Q::Buffer x(1024*4),y(1024*4),z(1024*4),matrix(256);std::vector<float> host(1024);for(int i=0;i<1024;++i)host[i]=float(std::sin(i*.13));put(x,host);Q::ck(cudaMemcpy(matrix.data,c.matrix.data(),256,cudaMemcpyHostToDevice));
    double asymmetry=0;for(int i=0;i<8;++i)for(int j=0;j<8;++j)asymmetry=std::max(asymmetry,std::abs(double(c.matrix[i*8+j]-c.matrix[j*8+i])));R::demand(asymmetry>.1,"test rotation must be nonsymmetric");
    Q::kv_spin_transform(x.as(),y.as(),matrix.as(),4,false,s);Q::kv_spin_transform(y.as(),z.as(),matrix.as(),4,true,s);Q::ck(cudaStreamSynchronize(s));
    double err=compare(get<float>(z),std::vector<double>(host.begin(),host.end()),5e-7,5e-7);std::printf("PASS pure F32 Spin8 inverse %s max=%g\n",c.view_name(),err);
}
static void byte_compare(const Q::KvSpan& history,const std::vector<uint8_t>& expected,int pos,int ctx,size_t stride){
    for(int head=0;head<4;++head){std::vector<uint8_t> actual(stride);Q::ck(cudaMemcpy(actual.data(),history.as<uint8_t>()+(size_t(head)*ctx+pos)*stride,stride,cudaMemcpyDeviceToHost));
        if(std::memcmp(actual.data(),expected.data()+head*stride,stride)){for(size_t i=0;i<stride;++i)if(actual[i]!=expected[head*stride+i]){std::fprintf(stderr,"byte head=%d pos=%d i=%zu got=%u want=%u\n",head,pos,i,actual[i],expected[head*stride+i]);break;}throw std::runtime_error("packed byte mismatch");}}
}
static void sequence(const Q::KvConfig& c,cudaStream_t s){
    reserve();constexpr int ctx=257;Q::KvCache cache(c,ctx,2,s);Q::Buffer q(6144*4),k(1024*4),v(1024*4),gate(6144*4),output(6144*4),scores(24*ctx*4);
    std::vector<std::vector<double>> keys,values;std::set<int> lengths={1,31,32,33,127,128,129,255,256,257};double worst=0;
    for(int pos=0;pos<ctx;++pos){
        std::vector<float> hq(6144),hk(1024),hv(1024),hg(6144);
        for(int i=0;i<6144;++i){hq[i]=float(std::sin(i*.037+pos*.29)*.8);hg[i]=float(std::cos(i*.13-pos*.17)*3);}
        for(int i=0;i<1024;++i){hk[i]=float(std::cos(i*.093+pos*.17)*.7);hv[i]=float(std::sin(i*.071-pos*.31)*1.3);}
        // Include zero blocks and signed extreme maxima to exercise scale/packing.
        if(!pos){std::fill(hv.begin(),hv.begin()+32,0.f);hk[32]=127;hk[33]=.5f;hk[34]=-.5f;hk[35]=-1.5f;hk[36]=1.5f;
            hv[32]=-8;hv[33]=8;hv[34]=.5f;hv[35]=-.5f;hv[36]=7.5f;hv[37]=-7.5f;}
        put(q,hq);put(k,hk);put(v,hv);put(gate,hg);
        cache.step(0,pos,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());
        if(c.mode==Q::KvMode::spin8_k8v4){hq=R::spin_f32(hq,c);hk=R::spin_f32(hk,c);hv=R::spin_f32(hv,c);}
        auto kp=R::pack(hk,c,true),vp=R::pack(hv,c,false);byte_compare(cache.layer(0).key,kp,pos,ctx,c.key_row_bytes());byte_compare(cache.layer(0).value,vp,pos,ctx,c.value_row_bytes());
        keys.push_back(R::unpack(kp,c,true));values.push_back(R::unpack(vp,c,false));
        if(lengths.count(pos+1)){
            // Fixed before execution: packed reference vs GPU FP32 reductions.
            worst=std::max(worst,compare(get<float>(output),R::attention(hq,hg,keys,values,c),3e-5,4e-5));
        }
    }
    rejects([&]{cache.step(0,ctx,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());});
    rejects([&]{cache.step(0,0,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());});
    R::demand(cache.layer(0).length==ctx&&cache.layer(1).length==0,"layer position isolation");
    std::vector<Q::RecurrentState> states(2);Q::Buffer error(4);int position=ctx;
    for(auto& st:states){st.conv.allocate(10240*3*4);st.gdn.allocate(48*128*128*4);Q::ck(cudaMemset(st.conv.data,0x3f,st.conv.bytes));Q::ck(cudaMemset(st.gdn.data,0x3f,st.gdn.bytes));}
    Q::ck(cudaMemset(error.data,1,4));Q::reset_session(states,cache,error,position,s);R::demand(position==0&&get<int>(error)[0]==0,"whole-session reset metadata");
    for(auto& st:states){for(auto b:get<uint8_t>(st.conv))R::demand(!b,"conv whole reset");for(auto b:get<uint8_t>(st.gdn))R::demand(!b,"GDN whole reset");}
    for(int i=0;i<2;++i){R::demand(cache.layer(i).length==0,"reset length");for(auto b:get<uint8_t>(cache.layer(i).key))R::demand(!b,"key reset");for(auto b:get<uint8_t>(cache.layer(i).value))R::demand(!b,"value reset");}
    auto saved=get<float>(v);auto bad=saved;bad[7]=NAN;put(v,bad);rejects([&]{cache.step(0,0,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());});put(v,saved);
    rejects([&]{cache.step(0,0,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());});cache.reset();cache.step(0,0,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());
    cache.step(1,0,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());R::demand(cache.layer(1).length==1,"error reset/reuse");
    bad=saved;bad[0]=1e20f;put(v,bad);rejects([&]{cache.step(0,1,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());});
    Q::reset_session(states,cache,error,position,s);put(v,saved);cache.step(0,0,q.as(),k.as(),v.as(),gate.as(),output.as(),scores.as());
    std::printf("PASS KV mode=%s view=%s positions=257 boundaries=10 exact_packed_bytes=all max_attention=%g atol=3e-5 rtol=4e-5 reset/error/reuse=pass\n",c.name(),c.view_name(),worst);
}
static void spin8_batch_and_split(const Q::KvConfig& c,cudaStream_t s){
    reserve();constexpr int ctx=1200;Q::KvCache cache(c,ctx,2,s);
    std::vector<std::vector<double>> keys,values;double worst=0;int pos0=0;
    // Covers BT=8 boundaries (7,8,9), BK=16 boundaries (15,16,17) from non-zero
    // prefix (after 1+2+3+4=10, n_tok=7 spans causal KV lengths 11..17), and multi-CTA chunks.
    for(int n_tok:{1,2,3,4,7,8,9,15,16,17,63,64,65,127}){
        Q::Buffer qb(size_t(n_tok)*6144*4),kb(size_t(n_tok)*1024*4),vb(size_t(n_tok)*1024*4),gb(size_t(n_tok)*6144*4),ob(size_t(n_tok)*6144*4),scores(24*ctx*4);
        std::vector<float> hq(size_t(n_tok)*6144),hk(size_t(n_tok)*1024),hv(size_t(n_tok)*1024),hg(size_t(n_tok)*6144);
        for(int t=0;t<n_tok;++t){
            const int pos=pos0+t;
            for(int i=0;i<6144;++i){hq[size_t(t)*6144+i]=float(std::sin(i*.037+pos*.29)*.8);hg[size_t(t)*6144+i]=float(std::cos(i*.13-pos*.17)*3);}
            for(int i=0;i<1024;++i){hk[size_t(t)*1024+i]=float(std::cos(i*.093+pos*.17)*.7);hv[size_t(t)*1024+i]=float(std::sin(i*.071-pos*.31)*1.3);}
        }
        put(qb,hq);put(kb,hk);put(vb,hv);put(gb,hg);
        cache.enqueue_batch(0,pos0,n_tok,qb.as(),kb.as(),vb.as(),gb.as(),ob.as(),scores.as());
        cache.check();
        auto got_all=get<float>(ob);
        for(int t=0;t<n_tok;++t){
            std::vector<float> tk(hk.begin()+size_t(t)*1024,hk.begin()+size_t(t+1)*1024);
            std::vector<float> tv(hv.begin()+size_t(t)*1024,hv.begin()+size_t(t+1)*1024);
            auto sk=R::spin_f32(tk,c),sv=R::spin_f32(tv,c);
            auto kp=R::pack(sk,c,true),vp=R::pack(sv,c,false);
            byte_compare(cache.layer(0).key,kp,pos0+t,ctx,c.key_row_bytes());
            byte_compare(cache.layer(0).value,vp,pos0+t,ctx,c.value_row_bytes());
            keys.push_back(R::unpack(kp,c,true));values.push_back(R::unpack(vp,c,false));
            if(n_tok<=17 || t<8 || t>=n_tok-4){
                std::vector<float> tq(hq.begin()+size_t(t)*6144,hq.begin()+size_t(t+1)*6144);
                std::vector<float> tg(hg.begin()+size_t(t)*6144,hg.begin()+size_t(t+1)*6144);
                std::vector<float> got(got_all.begin()+size_t(t)*6144,got_all.begin()+size_t(t+1)*6144);
                worst=std::max(worst,compare(got,R::attention(R::spin_f32(tq,c),tg,keys,values,c),3e-5,4e-5));
            }
        }
        pos0+=n_tok;
    }
    // Exercise Split-KV decode (pos0 == 401 >= 128, ctx == 1200 -> n_splits == 4)
    {
        Q::Buffer q(6144*4),k(1024*4),v(1024*4),gate(6144*4),out(6144*4),scores(24*ctx*4);
        std::vector<float> hq(6144),hk(1024),hv(1024),hg(6144);
        for(int i=0;i<6144;++i){hq[i]=float(std::sin(i*.037+pos0*.29)*.8);hg[i]=float(std::cos(i*.13-pos0*.17)*3);}
        for(int i=0;i<1024;++i){hk[i]=float(std::cos(i*.093+pos0*.17)*.7);hv[i]=float(std::sin(i*.071-pos0*.31)*1.3);}
        put(q,hq);put(k,hk);put(v,hv);put(gate,hg);
        cache.step(0,pos0,q.as(),k.as(),v.as(),gate.as(),out.as(),scores.as());
        auto kp=R::pack(R::spin_f32(hk,c),c,true),vp=R::pack(R::spin_f32(hv,c),c,false);
        byte_compare(cache.layer(0).key,kp,pos0,ctx,c.key_row_bytes());
        byte_compare(cache.layer(0).value,vp,pos0,ctx,c.value_row_bytes());
        keys.push_back(R::unpack(kp,c,true));values.push_back(R::unpack(vp,c,false));
        worst=std::max(worst,compare(get<float>(out),R::attention(R::spin_f32(hq,c),hg,keys,values,c),3e-5,4e-5));
    }
    // Explicitly exercise non-zero prefix (len=9) + query counts {7, 8, 9} crossing KV lengths {15, 16, 17}.
    for(int q_cnt:{7,8,9}){
        cache.reset();
        std::vector<std::vector<double>> pkeys,pvals;
        constexpr int pref=9;
        for(int pass=0;pass<2;++pass){
            const int p0=(pass==0)?0:pref;
            const int nt=(pass==0)?pref:q_cnt;
            Q::Buffer qb(size_t(nt)*6144*4),kb(size_t(nt)*1024*4),vb(size_t(nt)*1024*4),gb(size_t(nt)*6144*4),ob(size_t(nt)*6144*4),scores(24*ctx*4);
            std::vector<float> hq(size_t(nt)*6144),hk(size_t(nt)*1024),hv(size_t(nt)*1024),hg(size_t(nt)*6144);
            for(int t=0;t<nt;++t){
                const int pos=p0+t;
                for(int i=0;i<6144;++i){hq[size_t(t)*6144+i]=float(std::sin(i*.041+pos*.31)*.85);hg[size_t(t)*6144+i]=float(std::cos(i*.11-pos*.19)*2.7);}
                for(int i=0;i<1024;++i){hk[size_t(t)*1024+i]=float(std::cos(i*.089+pos*.23)*.75);hv[size_t(t)*1024+i]=float(std::sin(i*.067-pos*.29)*1.2);}
            }
            put(qb,hq);put(kb,hk);put(vb,hv);put(gb,hg);
            cache.enqueue_batch(0,p0,nt,qb.as(),kb.as(),vb.as(),gb.as(),ob.as(),scores.as());
            cache.check();
            auto got_all=get<float>(ob);
            for(int t=0;t<nt;++t){
                std::vector<float> tk(hk.begin()+size_t(t)*1024,hk.begin()+size_t(t+1)*1024);
                std::vector<float> tv(hv.begin()+size_t(t)*1024,hv.begin()+size_t(t+1)*1024);
                auto kp=R::pack(R::spin_f32(tk,c),c,true),vp=R::pack(R::spin_f32(tv,c),c,false);
                pkeys.push_back(R::unpack(kp,c,true));pvals.push_back(R::unpack(vp,c,false));
                if(pass==1){
                    std::vector<float> tq(hq.begin()+size_t(t)*6144,hq.begin()+size_t(t+1)*6144);
                    std::vector<float> tg(hg.begin()+size_t(t)*6144,hg.begin()+size_t(t+1)*6144);
                    std::vector<float> got(got_all.begin()+size_t(t)*6144,got_all.begin()+size_t(t+1)*6144);
                    worst=std::max(worst,compare(got,R::attention(R::spin_f32(tq,c),tg,pkeys,pvals,c),3e-5,4e-5));
                }
            }
        }
    }
    // Verify NaN rejection in batch path (q, k, v, gate) and reset reuse.
    for(int bad_kind=0;bad_kind<4;++bad_kind){
        cache.reset();
        Q::Buffer qb(3*6144*4),kb(3*1024*4),vb(3*1024*4),gb(3*6144*4),ob(3*6144*4),scores(24*ctx*4);
        std::vector<float> hq(3*6144,0.1f),hk(3*1024,0.1f),hv(3*1024,0.1f),hg(3*6144,0.1f);
        if(bad_kind==0)hq[6144+7]=NAN; else if(bad_kind==1)hk[1024+7]=NAN; else if(bad_kind==2)hv[1024+7]=NAN; else hg[6144+7]=NAN;
        put(qb,hq);put(kb,hk);put(vb,hv);put(gb,hg);
        cache.enqueue_batch(0,0,3,qb.as(),kb.as(),vb.as(),gb.as(),ob.as(),scores.as());
        rejects([&]{cache.check();});
    }
    cache.reset();
    Q::Buffer qb(3*6144*4),kb(3*1024*4),vb(3*1024*4),gb(3*6144*4),ob(3*6144*4),scores(24*ctx*4);
    std::vector<float> hq(3*6144,0.1f),hk(3*1024,0.1f),hv(3*1024,0.1f),hg(3*6144,0.1f);
    put(qb,hq);put(kb,hk);put(vb,hv);put(gb,hg);
    cache.enqueue_batch(0,0,3,qb.as(),kb.as(),vb.as(),gb.as(),ob.as(),scores.as());
    cache.check();
    std::printf("PASS Spin8 batch(1..127,BT=7/8/9,BK=15/16/17)+split_decode view=%s max_attention=%g reset/nan=pass\n",c.view_name(),worst);
}
static void allocations(const Q::KvConfig& c,cudaStream_t s){
    reserve();size_t previous_scratch=0;for(int capacity:{513,4096}){
        size_t before,total,after;Q::ck(cudaMemGetInfo(&before,&total));Q::KvCache cache(c,capacity,16,s);Q::ck(cudaMemGetInfo(&after,&total));
        size_t ideal=size_t(16)*4*capacity*(c.key_row_bytes()+c.value_row_bytes());R::demand(cache.persistent_bytes()==ideal,"history allocation mismatch");
        if(capacity==4096)R::demand(cache.scratch_bytes()==previous_scratch,"scratch grows with history");previous_scratch=cache.scratch_bytes();
        // Actual allocation accounting: all history buffers + fixed scratch + coefficients.
        R::demand(before>=after&&before-after>=ideal,"CUDA allocation accounting");
        std::printf("ALLOC mode=%s ctx=%d layers=16 payload=%zu scratch=%zu coefficients=%zu cuda_used_delta=%zu\n",c.name(),capacity,ideal,cache.scratch_bytes(),cache.coefficient_bytes(),before-after);
        R::demand(before-after<=ideal+cache.scratch_bytes()+cache.coefficient_bytes()+Q::kv_allocator_slack(c,capacity,16),"unaccounted history shadow");
    }
}
static void normalization(cudaStream_t s){
    constexpr int ctx=3;auto c=Q::KvConfig::admit("f16");Q::KvCache cache(c,ctx,1,s);
    Q::Buffer qg(12288*4),k(1024*4),v(1024*4),weight(256*4),q(6144*4),gate(6144*4),nk(1024*4),oldq(6144*4),oldgate(6144*4),kc(4*ctx*256*2),vc(4*ctx*256*2),out(6144*4),oldout(6144*4),score(24*ctx*4);
    std::vector<float> hg(12288),hk(1024),hv(1024),hw(256,1.f);for(int i=0;i<12288;++i)hg[i]=float(std::sin(i*.017));for(int i=0;i<1024;++i){hk[i]=float(std::cos(i*.047));hv[i]=float(std::sin(i*.031));}
    put(qg,hg);put(v,hv);put(weight,hw);
    for(int pos=0;pos<ctx;++pos){put(k,hk);Q::attention_prepare(qg.as(),k.as(),v.as(),weight.as(),weight.as(),oldq.as(),oldgate.as(),kc.as<half>(),vc.as<half>(),pos,ctx,1e-6f,1e7f,s);
        Q::attention_apply(oldq.as(),oldgate.as(),kc.as<half>(),vc.as<half>(),oldout.as(),score.as(),pos,ctx,s);
        Q::attention_normalize(qg.as(),k.as(),weight.as(),weight.as(),q.as(),gate.as(),k.as(),pos,1e-6f,1e7f,s);cache.step(0,pos,q.as(),k.as(),v.as(),gate.as(),out.as(),score.as());
        auto old=get<float>(oldout);compare(get<float>(out),std::vector<double>(old.begin(),old.end()),0,0);
    }std::puts("PASS F16 normalize/in-place-K/RoPE/cache exact legacy output");
}
static void fused_gdn_finite_check(cudaStream_t s){
    reserve();
    constexpr int HK=2,HV=4,D=128,C=(HK*2+HV)*D,T=3;
    Q::Buffer state(size_t(HV)*D*D*4),qkv(size_t(T)*C*4),gate(size_t(T)*HV*4),beta(size_t(T)*HV*4),
              z(size_t(T)*HV*D*4),gamma(D*4),y(size_t(T)*HV*D*4),err(4);
    auto fill_finite=[&]{
        state.zero(s);
        std::vector<float> hqkv(size_t(T)*C,0.05f),hg(size_t(T)*HV,-0.1f),hb(size_t(T)*HV,0.5f),
                           hz(size_t(T)*HV*D,0.2f),hgam(D,1.0f);
        put(qkv,hqkv);put(gate,hg);put(beta,hb);put(z,hz);put(gamma,hgam);
        err.zero(s);
    };
    fill_finite();
    strata::kernels::fused_gdn_step_norm_silu(state.as(),qkv.as(),qkv.as()+HK*D,qkv.as()+2*HK*D,
        gate.as(),beta.as(),z.as(),gamma.as(),1e-6f,y.as(),HK,HV,s,err.as<int>());
    Q::ck(cudaStreamSynchronize(s));R::demand(get<int>(err)[0]==0,"GDN step false positive");
    fill_finite();
    {float nan_v=NAN;Q::ck(cudaMemcpy(state.as()+19,&nan_v,4,cudaMemcpyHostToDevice));}
    strata::kernels::fused_gdn_step_norm_silu(state.as(),qkv.as(),qkv.as()+HK*D,qkv.as()+2*HK*D,
        gate.as(),beta.as(),z.as(),gamma.as(),1e-6f,y.as(),HK,HV,s,err.as<int>());
    Q::ck(cudaStreamSynchronize(s));R::demand(get<int>(err)[0]==1,"GDN step missed state NaN");
    fill_finite();
    {float nan_v=NAN;Q::ck(cudaMemcpy(z.as()+7,&nan_v,4,cudaMemcpyHostToDevice));}
    strata::kernels::fused_gdn_step_norm_silu(state.as(),qkv.as(),qkv.as()+HK*D,qkv.as()+2*HK*D,
        gate.as(),beta.as(),z.as(),gamma.as(),1e-6f,y.as(),HK,HV,s,err.as<int>());
    Q::ck(cudaStreamSynchronize(s));R::demand(get<int>(err)[0]==1,"GDN step missed output NaN");
    fill_finite();
    strata::kernels::fused_gdn_step_norm_silu_batch(state.as(),qkv.as(),C,gate.as(),beta.as(),z.as(),
        gamma.as(),1e-6f,y.as(),HK,HV,T,s,err.as<int>());
    Q::ck(cudaStreamSynchronize(s));R::demand(get<int>(err)[0]==0,"GDN batch false positive");
    fill_finite();
    {float nan_v=NAN;Q::ck(cudaMemcpy(state.as()+37,&nan_v,4,cudaMemcpyHostToDevice));}
    strata::kernels::fused_gdn_step_norm_silu_batch(state.as(),qkv.as(),C,gate.as(),beta.as(),z.as(),
        gamma.as(),1e-6f,y.as(),HK,HV,T,s,err.as<int>());
    Q::ck(cudaStreamSynchronize(s));R::demand(get<int>(err)[0]==1,"GDN batch missed state NaN");
    fill_finite();
    {float nan_v=NAN;Q::ck(cudaMemcpy(z.as()+HV*D+11,&nan_v,4,cudaMemcpyHostToDevice));}
    strata::kernels::fused_gdn_step_norm_silu_batch(state.as(),qkv.as(),C,gate.as(),beta.as(),z.as(),
        gamma.as(),1e-6f,y.as(),HK,HV,T,s,err.as<int>());
    Q::ck(cudaStreamSynchronize(s));R::demand(get<int>(err)[0]==1,"GDN batch missed output NaN");
    std::puts("PASS fused GDN step+batch state/output finite fence detection");
}
int main(){try{Q::host_resources();Q::ck(cudaSetDevice(0));Stream stream;
    normalization(stream.s);fused_gdn_finite_check(stream.s);
    for(const char* name:{"f16","k8v4","spin8_k8v4","f16_turbo2","f16_turbo3","f16_turbo4"}){
        for(const char* view:{"vector","plus","minus"}){bool spin=std::string(name)=="spin8_k8v4";if(!spin&&std::string(view)!="vector")continue;
            auto c=Q::KvConfig::admit(name,spin?view:"");if(spin){spin_identity(c,stream.s);spin8_batch_and_split(c,stream.s);}sequence(c,stream.s);if(std::string(view)=="vector")allocations(c,stream.s);
        }
    }std::puts("PASS owned packed KV production CUDA suite");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
