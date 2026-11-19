#include "strata/core/native_cache_runtime.hpp"
#include "strata/core/session.hpp"
#include <cuda_runtime_api.h>
namespace strata::core {
void native_session_validate(const NativeSessionConfig &c,const ModelGeometry &g,int64_t capacity) {
  if(capacity<1 || !c.rows || !c.selection_width || c.selection_width>uint64_t(capacity) ||
     c.plans.size()!=size_t(g.n_qsa_layers()) || c.plans.empty())
    throw std::invalid_argument("native session: invalid whole-model limits");
  size_t ordinal=0;
  for(int64_t l=0;l<g.n_layers;++l) if(is_qsa_layer(g,l)) {
    const auto &p=c.plans[ordinal++];
    if(!p || p->capacity()!=uint64_t(capacity) || c.rows>p->max_queries() || c.rows>p->max_append_rows())
      throw std::invalid_argument("native session: bound capacity/row mismatch");
    const auto &shape=p->geometry();
    if(shape.layer_index!=uint32_t(l) || shape.q_heads!=g.n_head ||
       shape.kv_heads!=g.n_head_kv || shape.head_width!=g.head_dim)
      throw std::invalid_argument("native session: actual logical layer/geometry mismatch");
    if(p->layout().total_bytes!=c.plans.front()->layout().total_bytes)
      throw std::invalid_argument("native session: heterogeneous owner arenas unsupported");
  }
}
size_t native_layer_bytes(const NativeSessionConfig &c,size_t i) {
  if(i>=c.plans.size() || !c.plans[i]) throw std::invalid_argument("native layer plan missing");
  const size_t scratch=native_bridge_bytes(c.rows,c.selection_width);
  const size_t owner=c.plans[i]->layout().total_bytes;
  if(!scratch || owner>SIZE_MAX-255 || ((owner+255)&~size_t(255))>SIZE_MAX-scratch)
    throw std::invalid_argument("native layer capacity overflow");
  return ((owner+255)&~size_t(255))+scratch;
}
bool native_session_attach(SessionState &s,const ModelGeometry &g,const NativeSessionConfig &c,
                            void *stream,std::string &error) {
  if(s.native.state || c.plans.size()!=size_t(g.n_qsa_layers()) ||
     s.layer_lo || s.layer_hi!=g.n_layers || !stream) {
    error="native session: only a new whole resident session is supported"; return false;
  }
  try { native_session_validate(c,g,s.max_cells); }
  catch(const std::invalid_argument &e) { error=e.what(); return false; }
  auto state=std::make_shared<NativeSession>();
  state->stream=stream;
  int device=-1;
  if(cudaGetDevice(&device)!=cudaSuccess) { error="native session: device unavailable"; return false; }
  size_t ordinal=0;
  for(int64_t l=0;l<g.n_layers;++l) if(is_qsa_layer(g,l)) {
    if(c.plans[ordinal]->geometry().layer_index!=uint32_t(l)) {
      error="native session: logical layer identity mismatch"; return false;
    }
    auto &q=s.qsa_states[ordinal];
    const size_t owner_bytes=c.plans[ordinal]->layout().total_bytes;
    const size_t offset=(owner_bytes+255)&~size_t(255);
    int code=-1;
    auto owner=NativeCacheOwner::attach(c.plans[ordinal],q.native_arena,owner_bytes,
                                        device,stream,code);
    if(!owner) { error="native session: owner initialization failed "+std::to_string(code); return false; }
    auto bridge=NativeCacheBridge::attach(std::move(owner),
      static_cast<char *>(q.native_arena)+offset,native_layer_bytes(c,ordinal)-offset,
      c.rows,c.selection_width,error);
    if(!bridge) return false;
    state->layers.push_back(std::move(bridge));
    ++ordinal;
  }
  // Publish borrowed pointers only after every layer has attached successfully.
  s.native.state=std::move(state);
  for(size_t i=0;i<c.plans.size();++i) s.qsa_states[i].native=s.native.state->layers[i].get();
  return true;
}
void native_session_invalidate(SessionState &s) {
  if(!s.native.state) return;
  s.native.state->valid=false;
  for(auto &l:s.native.state->layers) l->invalidate();
}
bool native_session_ready(SessionState &s,std::string &error) {
  if(!s.native.state || s.native.state->valid) return true;
  error="native session invalid (K/V, GDN, PLE, indexer and graph reuse); reset and replay accepted tokens";
  return false;
}
bool native_session_observe(SessionState &s,bool runtime_ok,std::string &error) {
  if(!s.native.state) return runtime_ok;
  bool good=runtime_ok && s.native.state->valid;
  for(auto &l:s.native.state->layers) if(!l->observe(runtime_ok,error)) good=false;
  if(!good) { native_session_invalidate(s); return native_session_ready(s,error); }
  return true;
}
bool native_session_reset(SessionState &s,std::string &error) {
  if(!s.native.state) return true;
  s.native.state->valid=false;
  for(auto &l:s.native.state->layers) if(!l->reset(error)) { native_session_invalidate(s); return false; }
  s.native.state->valid=true;
  return true;
}
bool native_session_publish(SessionState &s,const int32_t *boundary,std::string &error) {
  if(!native_session_ready(s,error)) return false;
  if(!s.native.state) return true;
  for(auto &l:s.native.state->layers) if(!l->publish_window(boundary,error)) { native_session_invalidate(s); return false; }
  return true;
}
} // namespace strata::core
