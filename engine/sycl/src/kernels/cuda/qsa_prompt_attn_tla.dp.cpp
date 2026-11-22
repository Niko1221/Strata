// FP32 sparse prompt attention with TLA block K loads and cached query tiles.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_prompt_attn_tla.hpp"
#include <cute/tensor.hpp>
#include "strata/kernels/mrope.hpp"
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#ifdef printf
#undef printf
#endif

namespace strata::kernels::tla_prompt_fp32 {
constexpr int HD=256,G=12,CH=64,Threads=64;
class Core;
class Merge;
inline float h2f(uint16_t b){return float(sycl::bit_cast<sycl::half>(b));}
// For every signed byte x, x is reconstructed exactly as
// bit_cast<float>(0x4b000000 | (uint8_t(x)^0x80)) - (2^23+128).
// The subtraction is exact over [-128,127]; scaling remains original FP32.
inline auto exact_signed_bytes(uint64_t bits){
 using UInt8=uint32_t __attribute__((ext_vector_type(8)));
 using Float8=float __attribute__((ext_vector_type(8)));
 const auto words=sycl::bit_cast<sycl::uint2>(bits);
 const UInt8 data{words[0],words[0],words[0],words[0],words[1],words[1],words[1],words[1]};
 const UInt8 shifts{0,8,16,24,0,8,16,24};
 const UInt8 biased=((data>>shifts)&UInt8(255))^UInt8(0x4b000080U);
 return sycl::bit_cast<Float8>(biased)-Float8(8388736.f);
}
// One nontranspose TLA copy supplies the original paired eight-byte inputs.
// The pinned copy operation declares at least64 elements of operand storage;
// keep four uint64 registers per lane behind the two-element logical view.
inline void key16_tla(const QsaAttnPools&p,long long row,int lane,sycl::sub_group,float*out0,float*out16){
 using namespace cute;
 const auto*base=reinterpret_cast<const uint64_t*>(p.k_q+row*HD);
 auto K=make_tensor(make_gmem_ptr(base),make_shape(_4{},_8{}),make_stride(_8{},_1{}));
 auto cp=make_block_2d_copy(XE_LOAD_2D<64,4,8>{},K);
 auto ct=cp.get_slice(lane);auto coords=make_identity_tensor(K.shape());
 auto storage=make_tensor<uint64_t>(make_shape(_4{}));
 auto layout=make_fragment_like<uint64_t>(ct.partition_S(coords)).layout();
 auto codes=make_tensor(make_rmem_ptr(storage.data()),layout);
 copy(cp,ct.partition_S(coords),codes);
 const auto lo=exact_signed_bytes(codes(0));
 const auto hi=exact_signed_bytes(codes(1));
 const float sc0=h2f(p.k_scale[row*4+lane/8]),sc16=h2f(p.k_scale[row*4+lane/8+2]);
#pragma unroll
 for(int j=0;j<8;++j){out0[j]=lo[j]*sc0;out16[j]=hi[j]*sc16;}
}
// Retain the scalar dot rounding order.
// Explicit vector FMAs avoid loss of contraction in the vector expression.
inline sycl::float4 dot8_head4(const float q[4][8],const float*k){
 // The scalar dot rounds product1, then fuses product0 into it.
 sycl::float4 v=k[1]*sycl::float4(q[0][1],q[1][1],q[2][1],q[3][1]);
 v=sycl::fma(sycl::float4(k[0]),sycl::float4(q[0][0],q[1][0],q[2][0],q[3][0]),v);
#pragma unroll
 for(int j=2;j<8;++j)v=sycl::fma(sycl::float4(k[j]),sycl::float4(q[0][j],q[1][j],q[2][j],q[3][j]),v);
 return v;
}
void launch(sycl::queue&queue,const float*q,QsaAttnPools p,const int32_t*ids,const int32_t*steps,
            int cap,QsaShapes sh,float*scratch,float*out,int nq){
 using namespace cute;
 const int chunks=(cap+CH-1)/CH,nkv=sh.n_head_kv,heads=sh.n_head,page=sh.page_size;
 const long long stride=qsa_decode_attn_scratch_floats(cap,sh);
 float*pm=scratch+size_t(chunks)*heads*HD,*pl=pm+size_t(chunks)*heads;
 queue.parallel_for<Core>(
  sycl::nd_range<3>(sycl::range<3>(nq,nkv,size_t(chunks)*Threads),sycl::range<3>(1,1,Threads)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(16)]]{
   const int qi=it.get_group(0),kv=it.get_group(1),chunk=it.get_group(2),t=it.get_local_id(2),lane=t&15,sgid=t>>4;
   const auto sg=it.get_sub_group();
   auto&sp=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CH]>(it.get_group());
   auto&srow=*sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CH]>(it.get_group());
   const int width=steps[size_t(qi)*kStepCount+kStepWidth],c0=chunk*CH,n_here=sycl::min(CH,width-c0),slot=kv*chunks+chunk;
   float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   const float*query=q+(size_t(qi)*heads+kv*G)*HD;
   if(n_here<=0){if(t<G){ms[slot*G+t]=-FLT_MAX;ls[slot*G+t]=0.f;}return;}
   if(t<CH){long long r=-1;if(t<n_here){const int cell=ids[size_t(qi)*cap+c0+t];const int pg=p.page_table[cell/page];if(pg>=0)r=(static_cast<long long>(pg)*nkv+kv)*page+cell%page;}srow[t]=r;}
   const float down=1.f/16.f;
   it.barrier(sycl::access::fence_space::local_space);
    // Existing cute universal 128-bit vector copy, preserving each lane's
    // original eight FP32 dimensions. Keep TLA block key copy unchanged.
#pragma clang loop unroll(disable)
    for(int hg=0;hg<G;hg+=4){
     float q0[4][8],q16[4][8];
#pragma unroll
     for(int h=0;h<4;++h){
      auto src0=make_tensor(make_gmem_ptr(query+(hg+h)*HD+lane*8),make_shape(_8{}),make_stride(_1{}));
      auto src16=make_tensor(make_gmem_ptr(query+(hg+h)*HD+(lane+16)*8),make_shape(_8{}),make_stride(_1{}));
      auto dst0=make_tensor(make_rmem_ptr(&q0[h][0]),make_shape(_8{}),make_stride(_1{}));
      auto dst16=make_tensor(make_rmem_ptr(&q16[h][0]),make_shape(_8{}),make_stride(_1{}));
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src0,dst0);
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src16,dst16);
     }
     for(int c=sgid;c<CH;c+=Threads/16){
      if(c>=n_here||srow[c]<0){if(lane<4)sp[hg+lane][c]=-FLT_MAX;continue;}
      float k0[8],k16[8];key16_tla(p,srow[c],lane,sg,k0,k16);
      sycl::float4 v=dot8_head4(q0,k0)+dot8_head4(q16,k16);
      // Native reduction retains the lane-zero offset8/4/2/1 tree.
      v=sycl::reduce_over_group(sg,v,sycl::plus<sycl::float4>{});
      if(lane==0){
#pragma unroll
       for(int h=0;h<4;++h)sp[hg+h][c]=v[h]*down;
      }
     }
    }
   it.barrier(sycl::access::fence_space::local_space);
   // Fold original warp32 offset16 first, then reproduce offsets8/4/2/1.
   for(int h=sgid;h<G;h+=Threads/16){
    float m=sycl::fmax(sycl::fmax(sp[h][lane],sp[h][lane+32]),sycl::fmax(sp[h][lane+16],sp[h][lane+48]));
#pragma unroll
    for(int off=8;off>0;off>>=1)m=sycl::fmax(m,sycl::permute_group_by_xor(sg,m,off));
    const float ea=(lane<n_here&&srow[lane]>=0)?sycl::native::exp(sp[h][lane]-m):0.f;
    const float eb=(lane+32<n_here&&srow[lane+32]>=0)?sycl::native::exp(sp[h][lane+32]-m):0.f;
    const float ec=(lane+16<n_here&&srow[lane+16]>=0)?sycl::native::exp(sp[h][lane+16]-m):0.f;
    const float ed=(lane+48<n_here&&srow[lane+48]>=0)?sycl::native::exp(sp[h][lane+48]-m):0.f;
    sp[h][lane]=ea;sp[h][lane+32]=eb;sp[h][lane+16]=ec;sp[h][lane+48]=ed;
    float l=(ea+eb)+(ec+ed);
#pragma unroll
    for(int off=8;off>0;off>>=1)l+=sycl::permute_group_by_xor(sg,l,off);
    if(lane==0){ms[slot*G+h]=m;ls[slot*G+h]=l;}
   }
   it.barrier(sycl::access::fence_space::local_space);
   const int d=4*t;sycl::float4 acc[G];
#pragma unroll
   for(int h=0;h<G;++h)acc[h]=sycl::float4(0.f);
   for(int c=0;c<n_here;c+=8){long long rows[8];bool full=c+8<=n_here;
#pragma unroll
    for(int j=0;j<8;++j){rows[j]=c+j<n_here?srow[c+j]:-1;full=full&&rows[j]>=0;}
    if(full){sycl::float4 values[8];
#pragma unroll
     for(int j=0;j<8;++j){const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      values[j]=codes.convert<float,sycl::rounding_mode::automatic>()*h2f(p.v_scale[rows[j]*4+d/64]);}
#pragma unroll
     for(int h=0;h<G;++h){const auto p0=*reinterpret_cast<const sycl::float4*>(&sp[h][c]);const auto p1=*reinterpret_cast<const sycl::float4*>(&sp[h][c+4]);
      acc[h]=sycl::fma(sycl::float4(p0[0]),values[0],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[1]),values[1],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p0[2]),values[2],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[3]),values[3],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[0]),values[4],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[1]),values[5],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[2]),values[6],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[3]),values[7],acc[h]);
     }
    }else{
#pragma unroll
     for(int j=0;j<8;++j){if(rows[j]<0)continue;
      const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      const auto value=codes.convert<float,sycl::rounding_mode::automatic>()*h2f(p.v_scale[rows[j]*4+d/64]);
#pragma unroll
      for(int h=0;h<G;++h)acc[h]=sycl::fma(sycl::float4(sp[h][c+j]),value,acc[h]);
     }
    }
   }
#pragma unroll
   for(int h=0;h<G;++h){
#pragma unroll
    for(int j=0;j<4;++j)pa[(size_t(slot)*G+h)*HD+d+j]=acc[h][j];}
  });
 queue.parallel_for<Merge>(sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(heads)*HD),sycl::range<3>(1,1,HD)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{
   const int qi=it.get_group(1),head=it.get_group(2),d=it.get_local_id(2),kv=head/G,h=head%G;
   const float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   float maximum=-FLT_MAX;
#pragma unroll
   for(int c=0;c<chunks;++c)maximum=sycl::fmax(maximum,ms[(kv*chunks+c)*G+h]);
   float sum=0.f,value=0.f;
   for(int c=0;c<chunks;++c){const int slot=kv*chunks+c;const float m=ms[slot*G+h];if(m==-FLT_MAX)continue;
    const float w=sycl::native::exp(m-maximum);sum=sycl::fma(ls[slot*G+h],w,sum);value=sycl::fma(pa[(size_t(slot)*G+h)*HD+d],w,value);}
   out[(size_t(qi)*heads+head)*HD+d]=sum>0.f?value/sum:0.f;
  });
}
}
#include "qsa_prompt_attn_tla_fp32.hpp"

namespace strata::kernels {
// Battlemage through Level Zero: the devices the ESIMD prompt kernel serves.
static bool battlemage_device(sycl::queue* queue) {
    return queue->get_backend() == sycl::backend::ext_oneapi_level_zero &&
           strata::intel_gpu_gen(queue->get_device()) == strata::IntelGpuGen::Battlemage;
}

bool qsa_tla_prepare_q(const float* q_full, const float* w, float* q, int64_t nq, int64_t pos, float eps,
                       const RopeScaling& scaling, void* stream) {
    auto* queue = strata::q_of(stream);
    if (!q_full || !w || !q || nq <= 0 || nq > INT32_MAX / 24 || pos < 0 || pos > INT32_MAX - nq ||
        !battlemage_device(queue) || rope_scaling_invalid(scaling))
        return false;
    tla_prompt_fp32::prepare_queries(*queue, q_full, w, q, nq, pos, eps, scaling);
    return true;
}

bool qsa_tla_sparse_prefill(const float* q, const QsaAttnPools& p, const int32_t* ids, const int32_t* steps,
                            int64_t cap, const QsaShapes& sh, float* scratch, float* out, int64_t nq,
                            int64_t scratch_queries, void* stream, QsaTlaKeyWorkspace workspace,
                            const QsaTlaGate* gate) {
    if (!q || !ids || !steps || !scratch || !out || !p.k_q || !p.v_q || !p.k_scale || !p.v_scale || !p.page_table ||
        p.k_q4 || p.v_q4 || nq <= 0 || scratch_queries <= 0 || cap <= 0 || cap > 32768 || sh.head_dim != 256 ||
        sh.n_head_kv != 2 || sh.n_head != 24 || sh.page_size <= 0)
        return false;
    if ((reinterpret_cast<uintptr_t>(p.v_q) & 3) || (reinterpret_cast<uintptr_t>(q) & 63) ||
        (reinterpret_cast<uintptr_t>(p.k_q) & 63))
        return false;
    auto* queue = strata::q_of(stream);
    if (!battlemage_device(queue))
        return false;
    const bool tiled = cap == 2051 && sh.page_size == 4;
    if (gate && (!tiled || !gate->query_and_gate || !gate->output || gate->leading_dimension < sh.n_head * sh.head_dim))
        return false;
    const float* q_full = gate ? gate->query_and_gate : nullptr;
    uint16_t* gated = gate ? gate->output : nullptr;
    int64_t ld = gate ? gate->leading_dimension : 0;
    const int64_t capacity = tiled ? std::min<int64_t>(96, scratch_queries) * qsa_decode_attn_scratch_floats(cap, sh) : 0;
    // Even the widest prefix tile must fit one query before submitting any preparation.
    if (tiled && capacity < 32 * sh.n_head * (sh.head_dim + 2))
        return false;
    const uint64_t* packed = nullptr;
    float* decoded_v = nullptr;
    if (tiled && nq >= 256 && workspace.data && (reinterpret_cast<uintptr_t>(workspace.data) & 15) == 0 &&
        workspace.identity_cells > 0 && workspace.identity_cells <= INT32_MAX) {
        const size_t pages = (size_t(workspace.identity_cells) + 3) / 4;
        if (pages <= workspace.bytes / (8 * 256)) {
            auto* dst = static_cast<uint64_t*>(workspace.data);
            tla_prompt_fp32::pack_sparse_keys(*queue, p.k_q, dst, pages, workspace.identity_cells);
            packed = dst;
            // When the caller supplies consecutive query positions, cap the grid and
            // scratch for the early causal tiles by their last possible selected cell.
            if (workspace.first_query_position >= 0 && workspace.identity_cells <= 4096 &&
                workspace.identity_cells - nq == workspace.first_query_position &&
                pages <= workspace.bytes / (8 * 256 * (1 + sizeof(float)))) {
                decoded_v = reinterpret_cast<float*>(dst + pages * 256);
                tla_prompt_fp32::expand_prefix_values(*queue, p, decoded_v, pages, workspace.identity_cells);
            }
        }
    }
    const int64_t width = std::min<int64_t>(tiled ? 16 : 32, scratch_queries);
    const int64_t query_stride = sh.n_head * sh.head_dim;
    for (int64_t first = 0; first < nq;) {
        const size_t offset = size_t(first) * query_stride;
        int count = int(std::min(width, nq - first));
        const auto* gate_input = q_full ? q_full + 2 * offset : nullptr;
        auto* gate_output = gated ? gated + first * ld : nullptr;
        if (decoded_v) {
            const int64_t position = workspace.first_query_position + first;
            // Small causal tiles need fewer partials per query and can share one submission.
            const int64_t tile = position < 1536 ? 96 : 16;
            count = int(std::min(tile, nq - first));
            auto active_chunks = [position](int n) { return int(std::min<int64_t>(32, (position + n + 63) / 64)); };
            const int per_query = active_chunks(count) * 24 * 258;
            count = int(std::min<int64_t>(count, capacity / per_query));
            tla_prompt_fp32::launch_sparse<true, true>(
                *queue, q + offset, p, ids + first * cap, steps + first * kStepCount, int(cap), sh, scratch,
                out + offset, count, gate_input, gate_output, ld, packed, active_chunks(count), decoded_v);
        } else if (tiled) {
            auto launch = packed ? tla_prompt_fp32::launch_sparse<true> : tla_prompt_fp32::launch_sparse<false>;
            launch(*queue, q + offset, p, ids + first * cap, steps + first * kStepCount, int(cap), sh, scratch,
                   out + offset, count, gate_input, gate_output, ld, packed, 32, nullptr);
        } else
            tla_prompt_fp32::launch(*queue, q + offset, p, ids + first * cap, steps + first * kStepCount, int(cap), sh,
                                    scratch, out + offset, count);
        first += count;
    }
    static thread_local bool announced = false;
    if (!announced) {
        std::fprintf(stderr, "strata: using SYCL TLA FP32 prompt attention\n");
        announced = true;
    }
    return true;
}

// A JIT build compiles each kernel when a process first launches it (one device module per kernel). The tiled path
// picks its kernels by how much of the prefix fits the key workspace, so a new engine met a compile in the middle of
// its first prompt to pass each threshold: on a B70 with 512-token chunks, +0.9 s at a 4K prompt and +1.4 s at 32K.
void qsa_tla_prebuild(void* stream) {
    auto* queue = strata::q_of(stream);
    if (!battlemage_device(queue))
        return;
    using namespace tla_prompt_fp32;
    const auto start = std::chrono::steady_clock::now();
    try {
        (void)sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            queue->get_context(), {queue->get_device()},
            {sycl::get_kernel_id<PrepareQueries>(), sycl::get_kernel_id<PackSparseKeys>(),
             sycl::get_kernel_id<ExpandPrefixValues>(), sycl::get_kernel_id<PrefixCore>(),
             sycl::get_kernel_id<PrefixMerge>(), sycl::get_kernel_id<PrefixSubgroupMerge>(),
             sycl::get_kernel_id<SparseCore<true>>(), sycl::get_kernel_id<SparseMerge<true>>(),
             sycl::get_kernel_id<SparseCore<false>>(), sycl::get_kernel_id<SparseMerge<false>>()});
    } catch (const sycl::exception& e) {
        // The first launch reports the same failure; the prompt path keeps its fallbacks.
        std::fprintf(stderr, "strata: SYCL TLA prompt attention kernels were not built ahead: %s\n", e.what());
        return;
    }
    std::fprintf(stderr, "strata: SYCL TLA prompt attention kernels built in %.0f ms\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

bool qsa_prompt_attn_tla_prefill(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& shapes, float* scratch, float* output, int64_t queries,
                                 int64_t scratch_queries, void* stream) {
    return qsa_tla_sparse_prefill(q, pools, ids, steps, cap, shapes, scratch, output, queries, scratch_queries, stream,
                                  {}, nullptr);
}

bool qsa_prompt_attn_tla_batch(const float* q, const QsaAttnPools& p, const int32_t* ids, const int32_t* steps,
                               int64_t cap, const QsaShapes& sh, float* scratch, float* out, int64_t nq, void* stream) {
    return nq <= 32 && qsa_prompt_attn_tla_prefill(q, p, ids, steps, cap, sh, scratch, out, nq, nq, stream);
}
} // namespace strata::kernels
