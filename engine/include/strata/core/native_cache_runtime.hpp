#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#if defined(STRATA_WITH_NATIVE_CACHE)
#include "strata/core/native_cache_bridge.hpp"
#include <vector>
#endif
namespace strata::core {
struct SessionState;
struct ModelGeometry;
struct NativeSessionConfig;
struct NativeSession;
// Legacy empty-state value copies remain legal. An enabled session cannot
// acquire shared mutable cache ownership through memberwise copying.
struct NativeSessionHandle {
  std::shared_ptr<NativeSession> state;
  NativeSessionHandle() = default;
  NativeSessionHandle(const NativeSessionHandle &other) {
    if(other.state) throw std::logic_error("native session clone requires explicit distinct storage");
  }
  NativeSessionHandle &operator=(const NativeSessionHandle &other) {
    if(this!=&other) {
      if(state || other.state) throw std::logic_error("native session assignment requires explicit release");
    }
    return *this;
  }
  NativeSessionHandle(NativeSessionHandle &&) = default;
  NativeSessionHandle &operator=(NativeSessionHandle &&) = default;
};
#if defined(STRATA_WITH_NATIVE_CACHE)
struct NativeSessionConfig {
  std::vector<std::shared_ptr<const NativeCachePlan>> plans;
  size_t rows = 0, selection_width = 0;
};
struct NativeSession {
  std::vector<std::unique_ptr<NativeCacheBridge>> layers;
  void *stream = nullptr;
  bool valid = true;
};
void native_session_validate(const NativeSessionConfig &, const ModelGeometry &, int64_t capacity);
size_t native_layer_bytes(const NativeSessionConfig &, size_t ordinal);
bool native_session_attach(SessionState &, const ModelGeometry &, const NativeSessionConfig &,
                           void *stream, std::string &error);
bool native_session_ready(SessionState &, std::string &error);
bool native_session_observe(SessionState &, bool runtime_ok, std::string &error);
void native_session_invalidate(SessionState &);
bool native_session_reset(SessionState &, std::string &error);
bool native_session_publish(SessionState &, const int32_t *boundary, std::string &error);
#endif
// An unsuccessful call poisons the entire enabled session, including cancellation.
class NativeSessionCall {
public:
  explicit NativeSessionCall(SessionState &s): s_(s) {}
  ~NativeSessionCall();
  bool done = false;
private:
  SessionState &s_;
};
} // namespace strata::core
