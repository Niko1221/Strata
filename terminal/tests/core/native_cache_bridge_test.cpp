#include "native_cache_fixture.hpp"
#include "strata/core/native_cache_bridge.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>
using namespace strata::core;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define CUDA(x) CHECK((x) == cudaSuccess)
struct Device {
  void *p = nullptr;
  explicit Device(size_t n) { CUDA(cudaMalloc(&p, n)); }
  ~Device() { cudaFree(p); }
  template<class T> T *as() { return static_cast<T *>(p); }
  template<class T> void put(const std::vector<T> &v) {
    CUDA(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice));
  }
};
int main() {
  try {
    // These tests use actual device commands and graph replay, not host mocks.
    CHECK(native_bridge_bytes(0, 3) == 0);
    CHECK(native_bridge_bytes(SIZE_MAX, 3) == 0);
    CHECK(native_bridge_bytes(3, 5) >= 3 * 6 * sizeof(int64_t));
    auto r = native_fixture::request();
    r.representation = NativeCacheRepresentation::DiagnosticF32;
    auto b = native_fixture::binding();
    std::string error;
    auto plan = NativeCachePlan::bind(r, b, nullptr, 0, error);
    CHECK(plan);
    cudaStream_t stream{};
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
      Device arena(plan->layout().total_bytes), scratch(native_bridge_bytes(3,5));
      int rc = -1;
      auto owner = NativeCacheOwner::attach(plan, arena.p, plan->layout().total_bytes,
                                             0, stream, rc);
      CHECK(owner && !rc);
      auto bridge = NativeCacheBridge::attach(std::move(owner), scratch.p,
                        native_bridge_bytes(3,5), 3, 5, error);
      CHECK(bridge);
      Device k(3*2*128*4), v(3*2*128*4), q(3*4*128*4), out(3*4*128*4);
      Device steps(3*4*4), ids(3*17*4), target(12);
      auto put_ids = [&](std::vector<int32_t> compact) {
        std::vector<int32_t> padded(3*17,999999);
        for(size_t t=0;t<3;++t) for(size_t j=0;j<5 && t*5+j<compact.size();++j) padded[t*17+j]=compact[t*5+j];
        ids.put(padded);
      };
      k.put(std::vector<float>(3*2*128, 0));
      q.put(std::vector<float>(3*4*128, 1));
      std::vector<float> values(3*2*128);
      for(size_t t=0;t<3;++t)
        for(size_t i=0;i<2*128;++i) values[t*256+i]=float(t+1);
      v.put(values);
      // Poison padding with out-of-range values: widening must never read it.
      put_ids(std::vector<int32_t>{0,99999,99999,99999,99999,
                                   0,1,99999,99999,99999,
                                   0,1,2,99999,99999});
      steps.put(std::vector<int32_t>{0,1,0,1, 1,2,0,2, 2,3,0,3});
      target.put(std::vector<int32_t>{2,1,0});
      cudaGraph_t graph{};
      cudaGraphExec_t exec{};
      CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      CHECK(bridge->query(q.as<float>(),3,error));
      CHECK(bridge->append(k.as<float>(),v.as<float>(),steps.as<int32_t>(),3,error));
      CHECK(bridge->attend(q.as<float>(),out.as<float>(),ids.as<int32_t>(),
                           steps.as<int32_t>(),3,17,1.f,error));
      CHECK(bridge->publish_window(target.as<int32_t>(),error));
      CUDA(cudaStreamEndCapture(stream,&graph));
      CUDA(cudaGraphInstantiateWithFlags(&exec,graph,0));
      CUDA(cudaGraphLaunch(exec,stream));
      CUDA(cudaStreamSynchronize(stream));
      CHECK(bridge->observe(true,error));
      CHECK(bridge->observed().committed==2 && bridge->observed().tentative==2);
      std::vector<float> got(3*4*128);
      CUDA(cudaMemcpy(got.data(),out.p,got.size()*4,cudaMemcpyDeviceToHost));
      for(size_t t=0;t<3;++t) for(size_t i=0;i<4*128;++i)
        CHECK(std::abs(got[t*512+i]-float(t+2)/2)<1e-4);
      std::vector<int64_t> cells(15);
      CUDA(cudaMemcpy(cells.data(),bridge->cells_device(),15*8,cudaMemcpyDeviceToHost));
      CHECK(cells==std::vector<int64_t>({0,-1,-1,-1,-1,0,1,-1,-1,-1,0,1,2,-1,-1}));
      // Replay with new positions and new target; rejected candidate row 2 is overwritten.
      steps.put(std::vector<int32_t>{2,3,0,3,3,4,1,2,4,5,1,2});
      put_ids(std::vector<int32_t>{0,1,2,99999,99999,0,3,99999,99999,99999,
                                   1,4,99999,99999,99999});
      target.put(std::vector<int32_t>{2,1,2});
      CUDA(cudaGraphLaunch(exec,stream));
      CUDA(cudaStreamSynchronize(stream));
      CHECK(bridge->observe(true,error));
      CHECK(bridge->observed().committed==4 && bridge->observed().tentative==4);
      // A T1 window uses its actual accessible count, not the plan maximum.
      steps.put(std::vector<int32_t>{4,5,1,1}); put_ids(std::vector<int32_t>{4});
      CHECK(bridge->append(k.as<float>(),v.as<float>(),steps.as<int32_t>(),1,error));
      CHECK(bridge->attend(q.as<float>(),out.as<float>(),ids.as<int32_t>(),
                           steps.as<int32_t>(),1,17,1.f,error));
      target.put(std::vector<int32_t>{1,0,4});
      CHECK(bridge->publish_window(target.as<int32_t>(),error));
      CUDA(cudaStreamSynchronize(stream)); CHECK(bridge->observe(true,error));
      CHECK(bridge->observed().committed==5);
      // Cancellation is fail-stop. Only explicit quiescent reset permits replay.
      bridge->invalidate();
      CHECK(!bridge->query(q.as<float>(),1,error));
      CHECK(bridge->reset(error));
      steps.put(std::vector<int32_t>{0,1,0,1, 1,2,0,2, 2,3,0,3});
      put_ids(std::vector<int32_t>{0,99999,99999,99999,99999,0,1,99999,99999,99999,
                                   0,1,2,99999,99999});
      target.put(std::vector<int32_t>{2,1,0});
      CUDA(cudaGraphLaunch(exec,stream)); CUDA(cudaStreamSynchronize(stream));
      CHECK(bridge->observe(true,error));
      std::vector<float> replay(3*4*128);
      CUDA(cudaMemcpy(replay.data(),out.p,replay.size()*4,cudaMemcpyDeviceToHost));
      CHECK(replay==got);
      // Wrong prefix invalidates, even though the previous output is finite.
      CUDA(cudaGraphLaunch(exec,stream)); CUDA(cudaStreamSynchronize(stream));
      CHECK(!bridge->observe(true,error));
      CHECK(!bridge->owner().host_usable());
      CHECK(bridge->reset(error));
      int failure=1;
      // Keep the injected device fault ordered before status staging on the bound stream.
      CUDA(cudaMemcpyAsync(bridge->owner().query_status_device(),&failure,sizeof(failure),cudaMemcpyHostToDevice,stream));
      CHECK(bridge->stage_status(error)); CUDA(cudaStreamSynchronize(stream));
      CHECK(!bridge->observe(true,error));
      CUDA(cudaGraphExecDestroy(exec)); CUDA(cudaGraphDestroy(graph));
    }
    CUDA(cudaStreamDestroy(stream));
    std::puts("native bridge: admission, GQA, partial rows, padded sparse cells, replay, T1/T3, reject, cancel/reset/replay, sticky status passed");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
