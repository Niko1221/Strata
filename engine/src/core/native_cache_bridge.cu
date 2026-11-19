#include "strata/core/native_cache_bridge.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <limits>

namespace strata::core {
namespace {
using namespace strata::kernels;
__device__ void fail(tqc_cache_cuda_state *s, unsigned code) {
  atomicOr(&s->status, code);
  s->valid = 0;
}
// One ordered preparation node validates the actual window prefix and count
// before any tensor read. Positions remain device data across graph replay.
__global__ void prepare_append(tqc_cache_cuda_state *s,
                               tqc_cache_cuda_command *cmd,
                               const int32_t *steps, int64_t *positions,
                               size_t n) {
  if (threadIdx.x || blockIdx.x) return;
  cmd->append_rows = 0;
  if (!s->valid || s->status) return;
  const int32_t first = steps[kStepPos];
  if (first < 0 || s->tentative != s->committed ||
      uint64_t(first) != s->tentative) {
    fail(s,TQC_CC_INVALID_STATE); return;
  }
  for (size_t i=0;i<n;++i) {
    const int32_t p=steps[i*kStepCount+kStepPos];
    if (int64_t(p)!=int64_t(first)+int64_t(i) ||
        int64_t(steps[i*kStepCount+kStepNKv])!=int64_t(p)+1) {
      fail(s,TQC_CC_INVALID_STATE); return;
    }
    positions[i]=p;
  }
  cmd->append_rows=n;
}
__global__ void prepare_attention(tqc_cache_cuda_state *s,
                                  tqc_cache_cuda_command *cmd,
                                  const int32_t *steps, const int32_t *ids,
                                  int64_t *positions, int64_t *cells,
                                  size_t n, size_t width, size_t input_stride) {
  if (threadIdx.x || blockIdx.x) return;
  cmd->query_rows=0;
  if (!s->valid || s->status) return;
  for(size_t t=0;t<n;++t) {
    const int32_t valid=steps[t*kStepCount+kStepWidth];
    const int32_t pos=steps[t*kStepCount+kStepPos];
    if(valid<1 || size_t(valid)>width || pos<0 || uint64_t(pos)>=s->tentative) {
      fail(s,TQC_CC_INDEX); return;
    }
    positions[t]=pos;
    for(size_t c=0;c<width;++c)
      cells[t*width+c]=c<size_t(valid) ? int64_t(ids[t*input_stride+c]) : -1;
  }
  cmd->query_rows=n;
}
__global__ void target_rows(tqc_cache_cuda_state *s,
                            tqc_cache_cuda_command *cmd,
                            const int32_t *target, bool window = false) {
  if(threadIdx.x || blockIdx.x) return;
  const int64_t n=window ? int64_t(target[2])+int64_t(target[0]) : int64_t(*target);
  if(n<0 || uint64_t(n)<s->committed || uint64_t(n)>s->tentative) {
    fail(s,TQC_CC_INVALID_STATE); return;
  }
  cmd->target_rows=uint64_t(n);
}
bool span_ok(const void *p,size_t n) {
  return p && n && n<=UINTPTR_MAX-reinterpret_cast<uintptr_t>(p);
}
} // namespace
size_t native_bridge_bytes(size_t rows,size_t width) {
  if(!rows || !width || width>SIZE_MAX/8-1 ||
     rows>(SIZE_MAX-255)/(8*(width+1))) return 0;
  return (rows*8*(width+1)+255)&~size_t(255);
}
std::unique_ptr<NativeCacheBridge> NativeCacheBridge::attach(
    std::unique_ptr<NativeCacheOwner> owner,void *scratch,size_t bytes,
    size_t rows,size_t width,std::string &error) {
  const size_t need=native_bridge_bytes(rows,width);
  if(!owner || !need || rows>owner->plan().max_queries() ||
     rows>owner->plan().max_append_rows() || width>owner->plan().capacity() || !span_ok(scratch,bytes) ||
     bytes<need || reinterpret_cast<uintptr_t>(scratch)%256) {
    error="native bridge: invalid bounded scratch"; return {};
  }
  const auto &ctx=owner->context();
  auto a=reinterpret_cast<uintptr_t>(scratch), b=reinterpret_cast<uintptr_t>(ctx.arena);
  if(a<b+ctx.arena_bytes && b<a+bytes) {
    error="native bridge: arena/scratch overlap"; return {};
  }
  std::unique_ptr<NativeCacheBridge> result(new NativeCacheBridge);
  result->owner_=std::move(owner);
  result->rows_=rows; result->width_=width;
  result->positions_=static_cast<int64_t *>(scratch);
  result->cells_=result->positions_+rows;
  if(cudaHostAlloc(reinterpret_cast<void **>(&result->status_),sizeof(Status),
                   cudaHostAllocPortable)!=cudaSuccess) {
    error="native bridge: status staging allocation failed"; return {};
  }
  *result->status_=Status{};
  return result;
}
NativeCacheBridge::~NativeCacheBridge() {
  // Caller must quiesce and destroy graphs first, as for the borrowed arena.
  if(status_) cudaFreeHost(status_);
}
bool NativeCacheBridge::check(int code,std::string &error) {
  if(!code) return true;
  invalidate(); error="native bridge: operation failed ("+std::to_string(code)+")";
  return false;
}
bool NativeCacheBridge::query(float *q,size_t n,std::string &error) {
  if(!n || n>rows_) { error="native bridge: query rows out of bounds"; return false; }
  const auto &g=owner_->plan().geometry();
  const size_t stride=size_t(g.q_heads)*g.head_width*4, bytes=n*stride;
  return check(owner_->query({q,bytes,size_t(g.head_width)*4,stride,
                              q,bytes,size_t(g.head_width)*4,stride,n}),error);
}
bool NativeCacheBridge::append(const float *k,const float *v,const int32_t *steps,
                                size_t n,std::string &error) {
  if(!n || n>rows_ || !steps || !k || !v || !owner_->host_usable()) {
    error="native bridge: invalid append request"; return false;
  }
  const auto &g=owner_->plan().geometry();
  auto cs=static_cast<cudaStream_t>(owner_->context().stream);
  prepare_append<<<1,1,0,cs>>>(owner_->cache_status_device(),owner_->command_device(),steps,positions_,n);
  if(!check(int(cudaGetLastError()),error)) return false;
  const size_t row=size_t(g.head_width)*4, bytes=n*g.kv_heads*row;
  tqc_cache_cuda_append r{1,sizeof(r),{const_cast<float *>(k),bytes,row},
    {const_cast<float *>(v),bytes,row},{positions_,n*8,8}};
  return check(owner_->append(r,n),error);
}
bool NativeCacheBridge::attend(const float *q,float *out,const int32_t *ids,
    const int32_t *steps,size_t n,size_t stride,float scale,std::string &error) {
  if(!n || n>rows_ || stride<width_ || stride>SIZE_MAX/n || !ids || !steps ||
     !q || !out || !owner_->host_usable()) {
    error="native bridge: invalid attention request"; return false;
  }
  const auto &g=owner_->plan().geometry();
  auto cs=static_cast<cudaStream_t>(owner_->context().stream);
  prepare_attention<<<1,1,0,cs>>>(owner_->cache_status_device(),owner_->command_device(),
      steps,ids,positions_,cells_,n,width_,stride);
  if(!check(int(cudaGetLastError()),error)) return false;
  const size_t row=size_t(g.head_width)*4,bytes=n*g.q_heads*row;
  tqc_cache_cuda_attention r{1,sizeof(r),1,0,scale,
    {const_cast<float *>(q),bytes,row},{out,bytes,row},{positions_,n*8,8},{},{},
    {cells_,n*width_*8,width_*8},width_};
  return check(owner_->attend(r,n),error) && stage_status(error);
}
bool NativeCacheBridge::publish(const int32_t *boundary,std::string &error) {
  if(!boundary || !owner_->host_usable()) { error="native bridge: invalid publication"; return false; }
  auto cs=static_cast<cudaStream_t>(owner_->context().stream);
  target_rows<<<1,1,0,cs>>>(owner_->cache_status_device(),owner_->command_device(),boundary);
  return check(int(cudaGetLastError()),error) &&
    check(owner_->control(TQC_CC_COMMIT),error) &&
    check(owner_->control(TQC_CC_TRUNCATE),error) && stage_status(error);
}
bool NativeCacheBridge::publish_window(const int32_t *words,std::string &error) {
  if(!words || !owner_->host_usable()) { error="native bridge: invalid window publication"; return false; }
  auto cs=static_cast<cudaStream_t>(owner_->context().stream);
  target_rows<<<1,1,0,cs>>>(owner_->cache_status_device(),owner_->command_device(),words,true);
  return check(int(cudaGetLastError()),error) &&
    check(owner_->control(TQC_CC_COMMIT),error) &&
    check(owner_->control(TQC_CC_TRUNCATE),error) && stage_status(error);
}
bool NativeCacheBridge::publish_all(std::string &error) {
  return check(owner_->control(TQC_CC_COMMIT),error) && stage_status(error);
}
bool NativeCacheBridge::stage_status(std::string &error) {
  auto cs=static_cast<cudaStream_t>(owner_->context().stream);
  return check(int(cudaMemcpyAsync(&status_->cache,owner_->cache_status_device(),
      sizeof(status_->cache),cudaMemcpyDeviceToHost,cs)),error) &&
    check(int(cudaMemcpyAsync(&status_->query,owner_->query_status_device(),
      sizeof(status_->query),cudaMemcpyDeviceToHost,cs)),error);
}
bool NativeCacheBridge::observe(bool runtime_ok,std::string &error) {
  if(owner_->observe_completion(runtime_ok,status_->cache,status_->query)) return true;
  error="native cache/query state invalid; whole session requires reset and accepted-token replay";
  return false;
}
void NativeCacheBridge::invalidate() {
  // Host poisoning also works when a runtime error prevents a GPU flag write.
  owner_->observe_completion(false,{},-1);
}
bool NativeCacheBridge::reset(std::string &error) {
  if(!check(owner_->initialize_quiescent(),error)) return false;
  *status_=Status{};
  return true;
}
const tqc_cache_cuda_state &NativeCacheBridge::observed() const { return status_->cache; }
} // namespace strata::core
