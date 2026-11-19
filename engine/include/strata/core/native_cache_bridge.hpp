#pragma once
#include "strata/core/native_cache_owner.hpp"
#include <memory>
#include <string>

namespace strata::core {
// Only current-call cell IDs and positions. No uncompressed cache/history.
size_t native_bridge_bytes(size_t rows, size_t selection_width);
class NativeCacheBridge final {
public:
  static std::unique_ptr<NativeCacheBridge>
  attach(std::unique_ptr<NativeCacheOwner>, void *scratch, size_t bytes,
         size_t rows, size_t selection_width, std::string &error);
  ~NativeCacheBridge();
  NativeCacheBridge(const NativeCacheBridge &) = delete;
  NativeCacheBridge &operator=(const NativeCacheBridge &) = delete;
  bool query(float *normalized, size_t tokens, std::string &error);
  bool append(const float *k, const float *v, const int32_t *steps,
              size_t tokens, std::string &error);
  bool attend(const float *q, float *out, const int32_t *ids,
              const int32_t *steps, size_t tokens, size_t id_row_stride, float scale,
              std::string &error);
  // Device words carry current absolute accepted boundary, including T1.
  bool publish(const int32_t *boundary, std::string &error);
  bool publish_window(const int32_t *commit_words, std::string &error);
  bool publish_all(std::string &error);
  bool stage_status(std::string &error);
  bool observe(bool runtime_ok, std::string &error);
  bool reset(std::string &error);
  void invalidate();
  NativeCacheOwner &owner() { return *owner_; }
  const tqc_cache_cuda_state &observed() const;
  size_t rows() const { return rows_; }
  size_t width() const { return width_; }
  const int64_t *cells_device() const { return cells_; }
private:
  NativeCacheBridge() = default;
  struct Status { tqc_cache_cuda_state cache{}; int query = -1; };
  bool check(int code, std::string &error);
  std::unique_ptr<NativeCacheOwner> owner_;
  int64_t *positions_ = nullptr, *cells_ = nullptr;
  Status *status_ = nullptr;
  size_t rows_ = 0, width_ = 0;
};
} // namespace strata::core
