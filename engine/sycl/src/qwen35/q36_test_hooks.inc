// Include at the very end of q36_engine.dp.cpp, after its final function.
// Enabled only for a dedicated test build, never the normal engine build.
#ifdef STRATA_Q36_TEST_HOOKS
#include "q36_test_hooks.hpp"
namespace strata::qwen35::test {
void conv_sequence(sycl::queue* q, float* hist, const float* input,
                   const float* weights, float* out, int C, int qk_heads,
                   float eps, int T, bool write) {
    const int tiles = (T + kConvTile - 1) / kConvTile;
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> part(sycl::range<1>(4),h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,tiles,C),sycl::range<3>(1,1,128)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                k_conv_seq(hist,input,weights,out,C,qk_heads,eps,T,write,
                    part.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
void gdn_sequence(sycl::queue* q, float* state, const float* input, int C,
                  const float* alpha, const float* beta, const float* z,
                  const float* gamma, float eps, float* out,
                  int HK, int HV, int T, bool write) {
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> sk(sycl::range<1>(128),h),sq(sycl::range<1>(128),h),sum(sycl::range<1>(16),h);
        sycl::local_accessor<float[4][128],0> kv(h),o(h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,4,HV*128),sycl::range<3>(1,4,128)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_seq(state,input,C,alpha,beta,z,gamma,eps,out,HK,HV,T,write,
                    sk.get_multi_ptr<sycl::access::decorated::no>().get(),
                    sq.get_multi_ptr<sycl::access::decorated::no>().get(),kv,o,
                    sum.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
void gdn_sequence_columns(sycl::queue* q, float* state, const float* input,
                          int C, const float* alpha, const float* beta,
                          const float* z, const float* gamma, float eps,
                          float* out, int HK, int HV, int T) {
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float[2][128],0> sk(h),sq(h);
        sycl::local_accessor<float[16][32],0> kv(h),o(h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,4*16,HV*32),sycl::range<3>(1,16,32)),
            [=](sycl::nd_item<3>) {
                k_gdn_seq_cols(state,input,C,alpha,beta,out,HK,HV,T,sk,sq,kv,o);
            });
    });
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> part(sycl::range<1>(4),h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,T,HV*128),sycl::range<3>(1,1,128)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                k_gdn_norm(out,z,gamma,eps,out,HV,
                    part.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
void attention_prepare(sycl::queue* q, const float* qg, const float* kx,
                       const float* vx, const float* qw, const float* kw,
                       float* query, float* gate, float* kc, float* vc,
                       int T, int H, int HKV, const int32_t* pos,
                       int ctx, float base, float eps,
                       const int32_t* delta, const int32_t* mtab) {
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> sum(sycl::range<1>(256),h),buf(sycl::range<1>(256),h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,T,(H+HKV)*256),sycl::range<3>(1,1,256)),
            [=](sycl::nd_item<3>) {
                k_attn_prep(qg,kx,vx,qw,kw,query,gate,kc,vc,H,HKV,256,64,pos,ctx,base,eps,delta,mtab,RowSel{},
                    sum.get_multi_ptr<sycl::access::decorated::no>().get(),
                    buf.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
void attention_split(sycl::queue* q, const float* query, const float* keys,
                     const float* values, const float* gate, const int32_t* pos,
                     int T, int H, int HKV, int ctx, float* part,
                     float* pm, float* pl, float* out) {
    const int chunks=(ctx+kChunkCells-1)/kChunkCells;
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> qs(sycl::range<1>(256),h),sc(sycl::range<1>(256),h),sum(sycl::range<1>(256),h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,T*H,chunks*256),sycl::range<3>(1,1,256)),
            [=](sycl::nd_item<3>) {
                k_attn_part(query,keys,values,pos,0,H,HKV,ctx,1.0f/16.0f,chunks,part,pm,pl,RowSel{},
                    qs.get_multi_ptr<sycl::access::decorated::no>().get(),
                    sc.get_multi_ptr<sycl::access::decorated::no>().get(),
                    sum.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
    q->parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,T*H*256),sycl::range<3>(1,1,256)),
        [=](sycl::nd_item<3>) {k_attn_merge(part,pm,pl,gate,pos,0,H,chunks,out,RowSel{});});
}
}
#endif

#ifdef STRATA_Q36_TEST_HOOKS
namespace strata::qwen35::test {
void attention_values_gemm(sycl::queue* q, const sycl::half* values,
        const sycl::half* probabilities, float* out, int HD, int nt, int P,
        int ldp, int H, int G) {
    dpct::blas::descriptor handle;
    handle.set_queue(q);
    q36_attention_values_gemm(&handle, values, probabilities, out, HD, nt, P, ldp, H, G);
}
}
#endif
