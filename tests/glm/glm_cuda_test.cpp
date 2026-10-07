#include "strata/glm/cuda.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace strata::glm;
struct Matrix {
    int O, I; std::vector<uint8_t> q; std::vector<float> s;
    Matrix(int o, int i, std::mt19937& rng) : O(o), I(i), q((size_t)o*i/2), s((size_t)o*i/64) {
        for (auto& b:q) b=(uint8_t)rng();
        for (auto& f:s) f=0.001f*(1+rng()%20);
    }
    Q4 view() const { return {O,I,q.data(),s.data()}; }
};
static int bad=0;
void check(const std::vector<float>& got, const std::vector<float>& ref, const char* name, float tol=3e-5f) {
    float scale=1e-6f, diff=0;
    for (float f:ref) scale=std::max(scale,std::fabs(f));
    for (size_t i=0;i<got.size();++i) {
        if (!std::isfinite(got[i])) { ++bad; std::printf("FAIL %s nonfinite\n",name); return; }
        diff=std::max(diff,std::fabs(got[i]-ref[i]));
    }
    std::printf("%s %s max normalized error %.3g\n",diff/scale<tol?"PASS":"FAIL",name,diff/scale);
    bad+=diff/scale>=tol;
}
int main() {
    std::mt19937 rng(123);
    auto fill=[&](std::vector<float>& a) { for(auto& x:a) x=(int(rng()%2001)-1000)/1000.0f; };
    CudaBackend gpu; std::string err;
    if (!gpu.init(0,128,64,55296,700ull<<20,1,err)) { std::fprintf(stderr,"CUDA unavailable: %s\n",err.c_str()); return 77; }
    Matrix W(37,512,rng);
    for(int rows:{1,3,19}) {
        std::vector<float> x(rows*512),y(rows*37),ref(y.size()); fill(x);
        q4_rows_ref(W.view(),x.data(),rows,0,37,ref.data(),37);
        gpu.gemm(W.view(),x.data(),rows,y.data()); check(y,ref,"int4-g64 GEMV/GEMM");
    }
    Matrix gate(64,128,rng),up(64,128,rng),down(128,64,rng);
    ExpertView ev{gate.view(),up.view(),down.view()};
    for(int rows:{1,7}) {
        std::vector<float>x(rows*128),g(rows*64),u(g.size()),y(x.size()),ref(x.size()); fill(x);
        q4_rows_ref(ev.gate,x.data(),rows,0,64,g.data(),64); q4_rows_ref(ev.up,x.data(),rows,0,64,u.data(),64);
        for(size_t i=0;i<g.size();++i)g[i]=silu(g[i])*u[i];
        q4_rows_ref(ev.down,g.data(),rows,0,128,ref.data(),128);
        gpu.expert(3,4,&ev,x.data(),rows,y.data()); check(y,ref,"streamed expert");
        gpu.promote(3,4,ev); gpu.expert(3,4,nullptr,x.data(),rows,y.data()); check(y,ref,"resident expert");
    }
    // Includes a partially filled softmax tile, a prefix, multiple heads and causal query positions.
    const int H=2,N=64,R=64,V=64,L=128,S=5,P=137,T=P+S;
    Matrix K(H*(N+V),L,rng);
    std::vector<float> q(S*H*(N+R)),kv(T*(L+R)),y(S*H*V),ref(y.size()); fill(q); fill(kv);
    for(int s=0;s<S;++s)for(int h=0;h<H;++h) {
        std::vector<float> qa(L,0),sc(P+s+1),ctx(L,0);
        const float* query=q.data()+(s*H+h)*(N+R);
        q4_rows_t(K.view(),h*(N+V),N,query,qa.data());
        for(int t=0;t<=P+s;++t) sc[t]=(dot_f32(qa.data(),kv.data()+t*(L+R),L)+dot_f32(query+N,kv.data()+t*(L+R)+L,R))/std::sqrt(float(N+R));
        softmax_inplace(sc.data(),(int)sc.size());
        for(int t=0;t<=P+s;++t) axpy_f32(ctx.data(),sc[t],kv.data()+t*(L+R),L);
        for(int v=0;v<V;++v)for(int i=0;i<L;++i)ref[(s*H+h)*V+v]+=q4_weight(K.view(),h*(N+V)+N+v,i)*ctx[i];
    }
    gpu.attention(K.view(),q.data(),kv.data(),S,P,H,N,R,V,y.data()); check(y,ref,"absorbed causal MLA, online softmax");
    return bad?1:0;
}
