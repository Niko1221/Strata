#include "native_cache_fixture.hpp"
#include "strata/core/session.hpp"
#include <cuda_runtime_api.h>
#include <cstdio>
using namespace strata::core;
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
int main() {
  try {
    ModelGeometry g;g.n_layers=8;
    NativeSessionConfig config;config.rows=3;config.selection_width=5;
    auto req=native_fixture::request();NativeCacheBinding binding=native_fixture::binding();
    binding.normal_layers={{3,24,2,256},{7,24,2,256}};std::string error;
    for(auto layer:binding.normal_layers){binding.selected_layer=layer.layer_index;auto p=NativeCachePlan::bind(req,binding,nullptr,0,error);CHECK(p);config.plans.push_back(p);}
    auto refused=[&](const NativeSessionConfig &c,int64_t capacity) {
      try { return session_bytes(g,capacity,10,0,8,&c)==0; } catch(const std::invalid_argument&) { return true; }
    };
    CHECK(refused(config,9));
    auto wrong=config;std::swap(wrong.plans[0],wrong.plans[1]);CHECK(refused(wrong,8));
    wrong=config;wrong.rows=4;CHECK(refused(wrong,8));
    cudaStream_t stream{};CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    SessionState s;void *arena=nullptr;const auto bytes=session_bytes(g,8,10,0,8,&config);
    CHECK(bytes>0);CHECK(cudaMalloc(&arena,bytes)==cudaSuccess);
    const auto used=session_init(g,8,10,arena,s,0,8,&config);CHECK(used>0 && used<=bytes);
    CHECK(native_session_attach(s,g,config,stream,error));session_zero(s,g,nullptr,stream);
    for(int i=0;i<2;++i){auto &q=s.qsa_states[i];CHECK(q.native && q.native_arena && !q.k_pool && !q.v_pool && !q.k_q && !q.v_q && !q.k_q4 && !q.v_q4);CHECK(q.idx_tail && q.page_table);CHECK(q.native->stage_status(error));}
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess);CHECK(native_session_observe(s,true,error));
    bool copied=false;try{SessionState clone=s;}catch(const std::logic_error&){copied=true;}CHECK(copied);
    SessionState legacy;SessionState legacy_copy=legacy;
    // A pageable H2D cudaMemcpy can return before the device write completes.
    // Order the injected fault before status staging on the bound nonblocking stream.
    int failure=1;CHECK(cudaMemcpyAsync(s.qsa_states[1].native->owner().query_status_device(),&failure,sizeof(failure),cudaMemcpyHostToDevice,stream)==cudaSuccess);
    CHECK(s.qsa_states[1].native->stage_status(error));CHECK(cudaStreamSynchronize(stream)==cudaSuccess);
    CHECK(!native_session_observe(s,true,error));CHECK(!native_session_ready(s,error));
    CHECK(!s.qsa_states[0].native->owner().host_usable());
    session_zero(s,g,nullptr,stream);CHECK(native_session_ready(s,error));
    for(auto &layer:s.native.state->layers)CHECK(layer->stage_status(error));
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess);CHECK(native_session_observe(s,true,error));
    {NativeSessionCall cancelled(s);}CHECK(!native_session_ready(s,error));
    session_zero(s,g,nullptr,stream);CHECK(native_session_ready(s,error));
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess);session_release(s);delete[] s.qsa_states;
    CHECK(cudaFree(arena)==cudaSuccess);CHECK(cudaStreamDestroy(stream)==cudaSuccess);
    std::puts("PASS native session accounting, absent legacy pools, logical ownership, whole-state poisoning, cancellation/reset");return 0;
  }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
